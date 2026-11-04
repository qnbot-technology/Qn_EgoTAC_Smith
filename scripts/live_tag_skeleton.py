#!/usr/bin/env python3
"""Actual camera + serial gloves; independent C++ Tag, two native hand workers."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from qn_ego.pipeline import env_for
from qn_ego.tag_skeleton import NativeStream, map_points
from qn_ego.live_stream import FrameReader, SolutionHistory, SolveRate, cad_root_bridges
from qn_ego.live_preview import Preview
from qn_ego.live_protocol.tactile_glove import load_config, TactileFramer
from qn_smith.ball23_registration import registered_transforms
from qn_smith.emf_spatial import decode_native
from qn_smith.artifacts import write_json, sha256, now
from render_human_v2 import project, draw, EDGES, verify_video

SIDES = ('left','right')


def quantiles(values):
    a = np.asarray(values,float)
    return dict(mean=float(a.mean()),p50=float(np.median(a)),p95=float(np.percentile(a,95)),
                max=float(a.max())) if len(a) else None


def put_latest(q, item):
    """Single producer, single consumer; a concurrent drain is harmless."""
    try:
        q.put_nowait(item)
        return False
    except queue.Full:
        try:q.get_nowait()
        except queue.Empty:pass
        q.put_nowait(item)
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=30.)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--decimate',type=float,default=2.)
    p.add_argument('--preprocess',choices=['cpu','opencl'],default='cpu')
    p.add_argument('--solver-hz',type=int,choices=[0,30,60,120],default=0,
                   help='Explicit native solve sampling; all serial input remains audited')
    p.add_argument('--preview-port',type=int,default=0,help='0 disables HTTP; otherwise latest JPEG live overlay')
    p.add_argument('--preview-host',default='127.0.0.1')
    p.add_argument('--calibration',type=Path,default=ROOT/'configs/live')
    a = p.parse_args()
    if not np.isfinite(a.seconds) or not 1 <= a.seconds <= 300 or a.threads not in range(1,9) or not 1 <= a.decimate <= 4:
        p.error('Bounded 1..300s test, 1..8 threads, decimate 1..4 required')
    if not 0 <= a.preview_port <= 65535:p.error('Invalid preview port')
    if not (ROOT/"runtime/bin/capture_orbbec_live").is_file():
        raise RuntimeError("Live capture requires a separately authorized native camera runtime; this repository supports recorded H5/MCAP processing without it")
    import serial
    out = a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    cal = a.calibration.resolve()
    for name in ['registration.json','left.yaml','right.yaml','tactile_emf.json']:
        if not (cal/name).is_file():
            raise FileNotFoundError(cal/name)
    ports = sorted(Path('/dev/serial/by-id').glob('usb-Artery_AT32_Virtual_Com_Port_*-if00'))
    if len(ports) != 2:
        raise RuntimeError('Exactly two independent AT32 by-id devices required')
    layout_path = ROOT/'configs/ball23/layout-23mm-roi-candidate.json'
    mount_path = ROOT/'configs/ball23/mount.json'
    layout, mount = json.loads(layout_path.read_text()),json.loads(mount_path.read_text())
    if layout['mount_sha256'] != sha256(mount_path):raise ValueError('Mount SHA mismatch')
    # Native fits CAD-bottom binding axes; compensate the documented render
    # basis once, rather than silently relabelling the raw camera poses.
    render_bridges = registered_transforms(json.loads((cal/'registration.json').read_text()),layout,mount)
    bridges = cad_root_bridges(layout,render_bridges)
    protocol = load_config(cal/'tactile_emf.json')
    if protocol['protocol'] != 'independent_glove_v1_emf97':raise ValueError('Verified live V1 required')
    cv2.setNumThreads(1)
    stop = threading.Event(); errors = queue.Queue(); counts = Counter(); lock = threading.Lock()
    histories = {s:SolutionHistory() for s in SIDES}
    pending = {s:queue.Queue(maxsize=64) for s in SIDES}
    workers = {}; threads = []; bindings = {}
    timings = {s:[] for s in SIDES}; rows = []; points = {s:[] for s in SIDES}
    render_queue = queue.Queue(maxsize=1); render_rows = []
    (out/'frames').mkdir()
    actual = dict(at=now(),argv=sys.argv,live_camera=True,visual_correction=False,training_ready=False,
        unit='metre',coordinate_frame='camera_color_optical',input_rectified=True,
        time_axis='host_receive_monotonic_ns',sync_calibrated=False,
        calibration_scope='Inherited operator/mount candidate; not a new physical calibration',
        solver_hz=a.solver_hz,solver_sampling=('all distinct receive times' if not a.solver_hz else
            'first received packet per monotonic bucket; original timestamps'),
        preview=dict(enabled=bool(a.preview_port),host=a.preview_host,port=a.preview_port,
                     scope='current rectified camera plus mapped skeleton; no display latency measurement'),
        native_root_frame='CAD_bottom',
        T_render_root_cad_bottom=layout['T_render_root_cad_bottom'],
        T_cad_bottom_emf={s:B.tolist() for s,B in bridges.items()},transform_applied_once=True,
        files=[dict(path=str(f.relative_to(ROOT)),sha256=sha256(f)) for f in
            [Path(__file__),ROOT/'src/qn_ego/live_stream.py',ROOT/'src/qn_ego/live_preview.py',ROOT/'src/qn_ego/tag_skeleton.py',
             ROOT/'runtime/bin/capture_orbbec_live',layout_path,mount_path,*[cal/n for n in
               ['registration.json','left.yaml','right.yaml','tactile_emf.json']]]])
    write_json(out/'actual-call.json',actual)

    def failure(fn,*args):
        try:fn(*args)
        except BaseException as e:
            errors.put(f'{fn.__name__}: {type(e).__name__}: {e}');stop.set()

    def receive(port):
        framer = TactileFramer(protocol)
        # Exclusive read-only subscription; no command/calibration written to device.
        with serial.Serial(str(port),protocol['baudrate'],timeout=.01,exclusive=True) as dev, \
                (out/(port.name+'-raw.jsonl')).open('x',buffering=1) as audit:
            while not stop.is_set():
                data = dev.read(min(65536,max(1,dev.in_waiting)));ns = time.monotonic_ns()
                if not data:continue
                audit.write(json.dumps(dict(host_receive_monotonic_ns=ns,payload_hex=data.hex()))+'\n')
                for event in framer.feed(data):
                    s = event['side']
                    with lock:
                        if s in bindings and bindings[s] != port.name:raise ValueError('Conflicting side identity')
                        bindings[s] = port.name;counts[s+'_received'] += 1
                    event['receive_ns'] = ns
                    if put_latest(pending[s],event):
                        with lock:counts[s+'_queue_dropped'] += 1
            with lock:
                counts.update({port.name+':'+k:v for k,v in framer.counts.items()})

    def solve(s):
        previous_ns = -1
        rate = SolveRate(a.solver_hz) if a.solver_hz else None
        while not stop.is_set() or not pending[s].empty():
            try:e = pending[s].get(timeout=.02)
            except queue.Empty:continue
            ns = e['receive_ns']
            if ns <= previous_ns:
                with lock:counts[s+'_same_read_time_skipped'] += 1
                continue
            previous_ns = ns
            if rate and not rate.take(ns):
                with lock:counts[s+'_rate_skipped'] += 1
                continue
            poses, valid = decode_native(np.asarray(e['finger_pose_i16'])[None],
                                         np.asarray(e['finger_valid'])[None],protocol)
            integrity = bool(e.get('checksum_valid') and e.get('frame_structure_valid'))
            reply, ms = workers[s].solve(poses[0],ns/1e9,bool(valid.all() and integrity),timeout_s=10.)
            histories[s].append(ns,reply);timings[s].append(ms)
            with lock:counts[s+'_solved'] += 1

    def render():
        while not stop.is_set() or not render_queue.empty():
            try:image,meta,mapped = render_queue.get(timeout=.02)
            except queue.Empty:continue
            K = np.asarray(camera['K'],float)
            for s,color in [('left',(40,220,100)),('right',(20,170,250))]:
                draw(image,project(mapped[s][None],K,np.zeros(8))[0],EDGES,color)
            image = cv2.resize(image,(1280,720),interpolation=cv2.INTER_AREA)
            status = ' | '.join(s+(' OK' if meta['mapping_valid'][s] else ' LOST') for s in SIDES)
            cv2.putText(image,'TAG + HumanV2 | '+status,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
            name=f'{len(render_rows):06d}.jpg'
            ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
            if not ok:raise IOError('JPEG encode failed')
            jpeg=encoded.tobytes();(out/'frames'/name).write_bytes(jpeg)
            render_rows.append(dict(file=name,host_receive_monotonic_ns=meta['camera_receive_ns']))
            if preview:
                window=render_rows[-60:]
                dt=(window[-1]['host_receive_monotonic_ns']-window[0]['host_receive_monotonic_ns'])/1e9
                preview.publish(jpeg,dict(meta,preview_fps=(len(window)-1)/dt if dt else 0,
                                          visual_correction=False,training_ready=False))

    camera_proc = None; reader = None; camera_log = None; preview = None
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    try:
        if a.preview_port:
            preview=Preview(a.preview_host,a.preview_port)
            print(f'Live overlay preview http://{a.preview_host}:{a.preview_port}',flush=True)
        for s in SIDES:workers[s] = NativeStream(ROOT/'third_party/humanv2',cal/(s+'.yaml'),out/s,s)
        for fn,arg in [(solve,s) for s in SIDES]+[(receive,port) for port in ports]:
            t=threading.Thread(target=failure,args=(fn,arg),daemon=True);t.start();threads.append(t)
        command = [str(ROOT/'runtime/bin/capture_orbbec_live'),str(out/'camera'),str(a.seconds),'30',
            '--realtime','--headless','--detector','opencv','--threads',str(a.threads),
            '--decimate',str(a.decimate),'--preprocess',a.preprocess,'--dual-hand','--layout',str(layout_path)]
        actual['camera_command'] = command;write_json(out/'actual-call.json',actual)
        camera_log = (out/'camera.log').open('x')
        camera_proc = subprocess.Popen(command,stdout=camera_log,stderr=subprocess.STDOUT)
        deadline = time.monotonic()+a.seconds+30
        while not (out/'camera/calibration.json').exists() or not (out/'camera/latest-frame.bin').exists():
            if camera_proc.poll() is not None or stop.is_set() or time.monotonic()>deadline:raise RuntimeError('Camera startup failed; see camera.log')
            time.sleep(.02)
        while True:
            try:
                camera = json.loads((out/'camera/calibration.json').read_text())
                break
            except json.JSONDecodeError:
                if camera_proc.poll() is not None or time.monotonic()>deadline:raise
                time.sleep(.01)
        if camera.get('distortion_model_sdk') != 4:raise ValueError('Unreviewed live distortion model')
        reader = FrameReader(out/'camera/latest-frame.bin',camera['width'],camera['height'])
        t=threading.Thread(target=failure,args=(render,),daemon=True);t.start();threads.append(t)
        while not stop.is_set():
            item = reader.take()
            if item is None:
                if camera_proc.poll() is not None:break
                if time.monotonic()>deadline:raise TimeoutError('Live test exceeded bound')
                time.sleep(.001);continue
            image,meta = item;tic = time.monotonic_ns();ns = meta['host_receive_monotonic_ns']
            mapped={};ages={};matched={};valid={};tagvalid={}
            for s in SIDES:
                reply,stamp,age,ok = histories[s].causal(ns)
                root = dict(meta['hands'][s],valid_current_observation=meta['hands'][s]['valid'])
                mapped[s] = map_points(root,bridges[s],reply,ok)
                points[s].append(mapped[s]);ages[s]=age;matched[s]=stamp
                valid[s] = bool(np.isfinite(mapped[s]).all());tagvalid[s] = bool(root['valid_current_observation'])
            finished=time.monotonic_ns()
            row=dict(camera_receive_ns=ns,source_index=meta['source_index'],
                camera_result_ns=meta['host_result_monotonic_ns'],mapping_result_ns=finished,
                native_receive_ns=matched,emf_age_ms=ages,tag_valid=tagvalid,mapping_valid=valid,
                receive_to_mapping_ms=(finished-ns)/1e6,map_ms=(finished-tic)/1e6,
                mailbox_sequence=meta['mailbox_sequence'])
            rows.append(row)
            if put_latest(render_queue,(image,row,mapped)):counts['render_dropped']+=1
            if len(rows)%100==0:print('live mapped',len(rows),'valid',valid,'age_ms',ages,flush=True)
        if stop.is_set() and camera_proc.poll() is None:camera_proc.terminate()
        if camera_proc.wait(timeout=10) and not stop.is_set():raise RuntimeError('C++ camera worker failed')
    except BaseException as e:
        errors.put(f'main: {type(e).__name__}: {e}')
    finally:
        stop.set()
        for t in threads:t.join(timeout=15)
        for t in threads:
            if t.is_alive():errors.put('Worker did not stop: '+t.name)
        if preview:preview.close()
        if camera_proc and camera_proc.poll() is None:
            camera_proc.terminate();camera_proc.wait(timeout=10)
        if reader:reader.close()
        for s,w in workers.items():
            try:w.close()
            except Exception as e:errors.put(f'{s} close: {type(e).__name__}: {e}')
        if camera_log:camera_log.close()
        (out/'frame-timings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        arrays=dict(camera_receive_ns=[r['camera_receive_ns'] for r in rows],source_index=[r['source_index'] for r in rows])
        for s in SIDES:
            arrays[s+'_camera_points_m']=np.asarray(points[s])
            arrays[s+'_mapping_valid']=np.asarray([r['mapping_valid'][s] for r in rows],bool)
        np.savez_compressed(out/'mapped-skeleton.npz',**arrays)
    failures=[]
    while not errors.empty():failures.append(errors.get())
    (out/'render-timeline.json').write_text(json.dumps(render_rows,indent=2))
    duration=(rows[-1]['camera_receive_ns']-rows[0]['camera_receive_ns'])/1e9 if len(rows)>1 else 0
    fps=(len(rows)-1)/duration if duration else 0
    valid_counts={s:sum(r['mapping_valid'][s] for r in rows) for s in SIDES}
    both=sum(all(r['mapping_valid'].values()) for r in rows)
    summary=dict(at=now(),live_camera=True,frames=len(rows),source_span_s=duration,mapping_output_fps=fps,
        target24_throughput=fps>=24,target30_throughput=fps>=29.5,training_ready=False,
        dual_hand_all_frames_valid=bool(rows) and both==len(rows),
        target24_dual_hand_pass=fps>=24 and both==len(rows) and not failures,
        solver_hz=a.solver_hz,
        visual_correction=False,counts=dict(counts),device_bindings=bindings,errors=failures,
        rendered_frames=len(render_rows),receive_to_mapping_ms=quantiles([r['receive_to_mapping_ms'] for r in rows]),
        preview=actual['preview'],
        native_ms={s:quantiles(timings[s]) for s in SIDES},
        tag_valid={s:sum(r['tag_valid'][s] for r in rows) for s in SIDES},
        mapping_valid=valid_counts,
        emf_age_ms={s:quantiles([r['emf_age_ms'][s] for r in rows if r['emf_age_ms'][s] is not None]) for s in SIDES},
        valid_mapping_fps={s:fps*valid_counts[s]/len(rows) if rows else 0 for s in SIDES},
        valid_fps_definition='output fps multiplied by valid frame fraction; not solver packet rate',
        latency_scope='SDK callback to mapped result; excludes exposure/USB before callback and display',
        no_future_glove_used=all(r['native_receive_ns'][s]<=r['camera_receive_ns'] for r in rows for s in SIDES),
        input_logs=[dict(file=f.name,sha256=sha256(f)) for f in out.glob('*-raw.jsonl')])
    write_json(out/'summary.json',summary)
    if failures:raise RuntimeError('; '.join(failures))
    if not rows:raise RuntimeError('No live mapping frames')
    if render_rows:
        lines=['ffconcat version 1.0']
        for i,r in enumerate(render_rows):
            dt=(render_rows[i+1]['host_receive_monotonic_ns']-r['host_receive_monotonic_ns'])/1e9 if i+1<len(render_rows) else 1/30
            lines += ["file 'frames/"+r['file']+"'",'duration '+str(dt)]
        # image demuxer timebase must resolve 30Hz and dropped-frame intervals.
        lines = [v for line in lines for v in ([line,'option framerate 1000'] if line.startswith('file ') else [line])]
        (out/'preview.ffconcat').write_text('\n'.join(lines)+'\n')
        cmd=['ffmpeg','-y','-v','error','-safe','0','-i',str(out/'preview.ffconcat'),'-fps_mode','vfr',
             '-c:v','libx264','-preset','fast','-crf','21','-threads','2','-pix_fmt','yuv420p',str(out/'live-tag-humanv2.mp4')]
        with (out/'encode.log').open('w') as log:subprocess.run(cmd,env=env_for(0,'cpu',2),stdout=log,stderr=log,check=True)
        write_json(out/'video-verification.json',verify_video(out/'live-tag-humanv2.mp4',len(render_rows),1280,720))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
