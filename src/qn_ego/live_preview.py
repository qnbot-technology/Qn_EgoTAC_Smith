"""Bounded latest-JPEG HTTP preview; independent of the collector GUI."""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Thread
from urllib.parse import urlsplit


PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AprilTag · HumanV2 实时映射</title>
<style>body{margin:0;background:#111820;color:#e8eef4;font:16px sans-serif}
header{padding:14px 20px;display:flex;justify-content:space-between;gap:16px}
img{display:block;width:100%;max-height:calc(100vh - 110px);object-fit:contain}
footer{padding:12px 20px;color:#b6c8d6}.ok{color:#68dfa0}</style>
<header><b>AprilTag · HumanV2 实时骨骼</b><span id="status">等待相机…</span></header>
<img src="/stream" alt="实时手套骨骼叠加画面">
<footer>绿色：左手　橙色：右手　| 无视觉纠正；缺测不保留旧骨骼</footer>
<script>setInterval(async()=>{try{let s=await(await fetch('/state',{cache:'no-store'})).json();
let v=s.mapping_valid||{},t=s.tag_valid||{};
let stale=s.frame_age_s===null||s.frame_age_s>1;
document.querySelector('img').style.opacity=stale?'.25':'1';
document.getElementById('status').textContent=stale?'等待新画面…':(s.preview_fps||0).toFixed(1)+' fps　'+
['left','right'].map((k,i)=>(i?'右手':'左手')+'：'+(v[k]?'有效':(t[k]?'骨骼无效':'Tag缺测'))).join('　');
}catch(e){document.getElementById('status').textContent='预览连接已关闭';}},500);</script></html>'''


class Preview:
    def __init__(self, host, port):
        self.condition = Condition()
        self.closed = False
        self.sequence = 0
        self.latest = b''
        self.state = {}
        self.published_at = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                route = urlsplit(self.path).path
                if route in ('/', '/state', '/snapshot.jpg'):
                    with owner.condition:
                        state=dict(owner.state,sequence=owner.sequence,
                                   frame_age_s=(time.monotonic()-owner.published_at
                                                if owner.published_at is not None else None))
                        payload = (PAGE.encode() if route == '/' else owner.latest if route == '/snapshot.jpg'
                                   else json.dumps(state,allow_nan=False).encode())
                    if route == '/snapshot.jpg' and not payload:
                        self.send_error(503,'No current frame');return
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8' if route == '/' else
                                     'image/jpeg' if route == '/snapshot.jpg' else 'application/json')
                    self.send_header('Content-Length', str(len(payload)))
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    self.wfile.write(payload)
                elif route == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    previous = 0
                    try:
                        while True:
                            with owner.condition:
                                owner.condition.wait_for(lambda: owner.closed or owner.sequence > previous,timeout=1)
                                if owner.closed:return
                                if owner.sequence == previous:continue
                                previous, payload = owner.sequence, owner.latest
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+
                                             str(len(payload)).encode()+b'\r\n\r\n'+payload+b'\r\n')
                            self.wfile.flush()
                    except (BrokenPipeError,ConnectionResetError,TimeoutError):
                        pass
                else:
                    self.send_error(404)

            def setup(self):
                super().setup()
                self.connection.settimeout(3)

        self.server = ThreadingHTTPServer((host,port),Handler)
        self.server.daemon_threads = True
        self.thread = Thread(target=self.server.serve_forever,kwargs={'poll_interval':.1},daemon=True)
        self.thread.start()

    def publish(self, jpeg, state):
        with self.condition:
            self.latest, self.state = jpeg, state
            self.published_at = time.monotonic()
            self.sequence += 1
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
