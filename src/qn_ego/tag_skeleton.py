"""Causal recorded-stream building blocks; no visual inference or pose holding."""
from collections import Counter
import json
import os
from pathlib import Path
import select
import subprocess
import time

import numpy as np

from qn_smith.apriltag_pose import detect_frame


def causal_index(times, query_ns, max_age_ms=30.):
    times = np.asarray(times, np.int64)
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError('Strictly increasing nonempty receive times required')
    i = int(np.searchsorted(times, query_ns, side='right')) - 1
    age = (int(query_ns)-int(times[i]))/1e6 if i >= 0 else float('inf')
    return i, age, i >= 0 and 0 <= age <= max_age_ms


def crop_calibration(camera, box):
    x, y, w, h = box
    K = np.asarray(camera['K'], float).copy()
    K[0, 2] -= x; K[1, 2] -= y
    return dict(camera, K=K.tolist(), width=w, height=h)


class NativeStream:
    """One persistent vendor process per hand, newline request/reply protocol."""
    def __init__(self, package, calibration, output, side):
        package, output = Path(package).resolve(), Path(output)
        output.mkdir(parents=True, exist_ok=False)
        lines = Path(calibration).read_text().splitlines()
        lines = [f'__resource_root__: {package / "native/qnbot_hand/resources"}'
                 if s.startswith('__resource_root__:') else s for s in lines]
        cfg = output/'runtime.yaml'; cfg.write_text('\n'.join(lines)+'\n')
        self.command = [str(package/'runtime/bin/human_hand_v2_runtime'), '--config', str(cfg),
                        '--side', side, '--resource-root', str(package/'native/qnbot_hand/resources'),
                        '--options', str(package/'native/qnbot_hand/configs/human_hand_v2_options.yaml')]
        env = dict(os.environ)
        env['LD_LIBRARY_PATH'] = str(package/'runtime/lib')+':'+env.get('LD_LIBRARY_PATH', '')
        self.err = (output/'stderr.log').open('w')
        self.log = (output/'native-output.jsonl').open('w')
        self.started = time.perf_counter()
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.err, env=env, bufsize=0)
        self.buffer = b''; self.last_time = -float('inf'); self.first_reply_ms = None

    def solve(self, poses, timestamp_s, valid=True, timeout_s=45.):
        if timestamp_s <= self.last_time:
            raise ValueError('Native timestamps must increase')
        self.last_time = timestamp_s
        ok = bool(valid and np.isfinite(poses).all())
        request = dict(timestamp_sec=float(timestamp_s), valid=ok,
                       poses=np.asarray(poses).tolist() if ok else [])
        t = time.perf_counter()
        self.process.stdin.write((json.dumps(request, allow_nan=False)+'\n').encode())
        deadline = t + timeout_s
        while b'\n' not in self.buffer:
            remain = deadline-time.perf_counter()
            if remain <= 0 or not select.select([self.process.stdout], [], [], max(0., remain))[0]:
                raise TimeoutError('HumanV2 persistent worker timed out')
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError('HumanV2 exited before a reply')
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b'\n', 1)
        row = json.loads(line)
        ms = (time.perf_counter()-t)*1000
        if self.first_reply_ms is None:
            self.first_reply_ms = (time.perf_counter()-self.started)*1000
        self.log.write(line.decode()+'\n')
        return row, ms

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate(); self.process.wait(timeout=5)
        self.log.close(); self.err.close()
        if self.process.returncode:
            raise RuntimeError(f'HumanV2 worker exit {self.process.returncode}')


class RoiTagDetector:
    """Unscaled native-pixel ROI search, periodic/full reacquisition.

    Historical corners select search pixels only. Every returned pose is solved
    exclusively from current image observations, never from a held pose.
    """
    def __init__(self, camera, sizes, bodies, margin=100, refresh=30):
        self.camera, self.sizes, self.bodies = camera, sizes, bodies
        self.margin, self.refresh = margin, refresh
        self.previous = []; self.count = 0

    def boxes(self):
        boxes = []
        for body in self.bodies.values():
            points = [d['corners_px'] for d in self.previous if str(d['id']) in body['T_root_tag']]
            # A hand that has never produced a valid observation must not
            # disable ROI for the other hand. It remains missing/NaN; only
            # bodies with a current tracking history contribute a search box.
            if not points:
                continue
            p = np.asarray(points).reshape(-1, 2)
            lo = np.floor(p.min(0)-self.margin).astype(int)
            hi = np.ceil(p.max(0)+self.margin).astype(int)
            lo = np.maximum(lo, 0)
            hi = np.minimum(hi, [self.camera['width'], self.camera['height']])
            boxes.append((int(lo[0]), int(lo[1]), int(hi[0]-lo[0]), int(hi[1]-lo[1])))
        if len(boxes) < 2:
            return boxes
        a, b = boxes
        if max(a[0], b[0]) < min(a[0]+a[2], b[0]+b[2]) and max(a[1], b[1]) < min(a[1]+a[3], b[1]+b[3]):
            x, y = min(a[0], b[0]), min(a[1], b[1])
            boxes = [(x, y, max(a[0]+a[2], b[0]+b[2])-x, max(a[1]+a[3], b[1]+b[3])-y)]
        return boxes

    def detect(self, image):
        boxes = self.boxes() if self.count % self.refresh else []
        self.count += 1
        if boxes:
            rows = []
            for x, y, w, h in boxes:
                row = detect_frame(image[y:y+h, x:x+w], crop_calibration(self.camera, (x,y,w,h)), self.sizes)
                for d in row['detections']:
                    d['corners_px'] = (np.asarray(d['corners_px'])+[x,y]).tolist()
                rows.append(row)
            row = dict(detections=sum([r['detections'] for r in rows], []),
                       states=sum([r['states'] for r in rows], []),
                       coordinate_frame='camera_color_optical', input_rectified=False)
            counts = Counter(d['id'] for d in row['detections'])
            row['duplicate_ids'] = [i for i,n in counts.items() if n > 1]
            present = {s['id'] for s in row['states'] if s['valid_current_observation']}
            # An empty/missed hand triggers immediate current-frame full search.
            tracked_ids = {int(d['id']) for d in self.previous}
            if any(tracked_ids.intersection(map(int, b['T_root_tag']))
                   and not present.intersection(map(int, b['T_root_tag']))
                   for b in self.bodies.values()):
                boxes = []
        if not boxes:
            row = detect_frame(image, self.camera, self.sizes)
        fresh = {s['id'] for s in row['states'] if s['valid_current_observation']}
        self.previous = [d for d in row['detections'] if d['id'] in fresh]
        return row, dict(search='roi' if boxes else 'full', boxes=boxes)


def map_points(root, bridge, native, time_valid):
    points = np.full((25, 3), np.nan)
    if root['valid_current_observation'] and time_valid and native and native.get('tracking_valid') and native.get('output_available'):
        local = np.asarray(native['positions_glove_m'], float)
        if local.shape == (25, 3) and np.isfinite(local).all():
            T = np.asarray(root['T_camera_root']) @ bridge
            points = local @ T[:3, :3].T + T[:3, 3]
    return points
