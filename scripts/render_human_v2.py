#!/usr/bin/env python3
"""Render native Human25 on full RGB timelines using explicit frame contracts.

The JSON job supplies hdf5, spatial, native, bridge, output, and optionally
mano (a track NPZ) and mano_summary. All paths must be absolute. The bridge
file supplies T_emf_solver and evidence; identity must also be explicit.
Native source_index means the original EMF packet row, never an RGB row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np

NAMES = ['wrist', 'thumb_metacarpal', 'thumb_proximal', 'thumb_distal', 'thumb_tip']
for finger in ('index', 'middle', 'ring', 'little'):
    NAMES += [f'{finger}_{part}' for part in ('metacarpal', 'proximal', 'intermediate', 'distal', 'tip')]
EDGES = [(0, s) for s in (1, 5, 10, 15, 20)]
for s in (1, 5, 10, 15, 20):
    EDGES += [(s+i, s+i+1) for i in range(3 if s == 1 else 4)]
MANO_EDGES = [(0, s) for s in (1, 5, 9, 13, 17)]
for s in (1, 5, 9, 13, 17):
    MANO_EDGES += [(s+i, s+i+1) for i in range(3)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def packet_mapping(requested, available):
    available = np.asarray(available, np.int64)
    if len(np.unique(available)) != len(available):
        raise ValueError('duplicate native source_index')
    lookup = {int(row): i for i, row in enumerate(available)}
    return np.array([lookup.get(int(row), -1) for row in requested], np.int64)


def stabilize(points, times, valid, cutoff=5., beta=.10, dcutoff=1., max_gap_s=.1):
    """One-Euro in solver metres; reset across invalid/nonpositive/long dt.

    Reproject onto median native bone lengths. Only a position candidate is
    returned: native rotations, q and wrist-local output are not mislabelled.
    """
    p = np.asarray(points, float)
    t = np.asarray(times, float)
    valid = np.asarray(valid, bool) & np.isfinite(p).all((1, 2))
    if np.any(np.diff(t) < 0):
        raise ValueError('native time reversed')
    target = np.array([np.median(np.linalg.norm(p[valid, b]-p[valid, a], axis=1))
                       if valid.any() else np.nan for a, b in EDGES])
    result = np.full_like(p, np.nan)
    previous = derivative = None
    for i in range(len(p)):
        if not valid[i]:
            previous = None
            continue
        dt = float(t[i]-t[i-1]) if i else 0.
        if previous is None or dt <= 0. or dt > max_gap_s:
            previous = p[i].copy()
            derivative = np.zeros_like(previous)
        else:
            ad = 1./(1.+1./(2*np.pi*dcutoff*dt))
            derivative = ad*(p[i]-previous)/dt+(1-ad)*derivative
            a = 1./(1.+1./(2*np.pi*(cutoff+beta*np.abs(derivative))*dt))
            previous = a*p[i]+(1-a)*previous
        result[i] = previous
        for (parent, child), length in zip(EDGES, target):
            delta = result[i, child]-result[i, parent]
            norm = np.linalg.norm(delta)
            if norm < 1e-10:
                delta = p[i, child]-p[i, parent]
                norm = np.linalg.norm(delta)
            if norm > 1e-10:
                result[i, child] = result[i, parent]+delta*(length/norm)
    return result


def project(points, K, D):
    uv = np.full(points.shape[:-1]+(2,), np.nan)
    for i, p in enumerate(points):
        keep = np.isfinite(p).all(1) & (p[:, 2] > 1e-6)
        if keep.any():
            uv[i, keep] = cv2.projectPoints(p[keep], np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    return uv


def draw(image, uv, edges, color):
    # Clip lines before rasterizing so invalid/extreme projections cannot
    # overflow OpenCV coordinates. Offscreen points remain in saved labels.
    valid = np.isfinite(uv).all(1) & (np.abs(uv).max(1) < 1e6)
    xy = np.rint(np.nan_to_num(uv, nan=0., posinf=0., neginf=0.)).clip(-1000000, 1000000).astype(int)
    h, w = image.shape[:2]
    for a, b in edges:
        if valid[a] and valid[b]:
            ok, p, q = cv2.clipLine((0, 0, w, h), tuple(xy[a]), tuple(xy[b]))
            if ok:
                cv2.line(image, p, q, color, 3, cv2.LINE_AA)
    for i in np.flatnonzero(valid):
        x, y = xy[i]
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), 4, color, -1, cv2.LINE_AA)


def title(image, text, footer):
    cv2.rectangle(image, (0, 0), (image.shape[1], 66), (20, 20, 20), -1)
    cv2.putText(image, text, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, .9, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.rectangle(image, (0, image.shape[0]-44), (image.shape[1], image.shape[0]), (20, 20, 20), -1)
    cv2.putText(image, footer, (20, image.shape[0]-13), cv2.FONT_HERSHEY_SIMPLEX, .62, (245, 245, 245), 1, cv2.LINE_AA)


class Encoder:
    def __init__(self, path, size, fps, *, square_pixels=False):
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg: raise RuntimeError('ffmpeg is required on PATH')
        self.command = [ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'bgr24',
                        '-video_size', f'{size[0]}x{size[1]}', '-framerate', str(fps), '-i', '-',
                        '-an', '-c:v', 'libx264', '-threads', '4', '-preset', 'fast', '-crf', '19',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
        if square_pixels:self.command[-1:-1]=['-vf','setsar=1']
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE)

    def write(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.process.stdin.close()
        if self.process.wait():
            raise RuntimeError('ffmpeg encoder failed')


def verify_video(path, expected_frames, width, height):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (height, width):
            raise ValueError('decoded dimensions changed')
        n += 1
    cap.release()
    if n != expected_frames:
        raise ValueError(f'incomplete decode: {path}: {n}/{expected_frames}')
    return dict(frames=n, width=width, height=height, bytes=path.stat().st_size, sha256=sha(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--job', type=Path, required=True)
    args = ap.parse_args()
    job = json.loads(args.job.read_text())
    for key in ('hdf5', 'spatial', 'native', 'bridge', 'output'):
        if not Path(job[key]).is_absolute():
            raise ValueError(f'{key} must be absolute')
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    # A directory is a single reproducible run, not an overwrite target.
    if any(output.iterdir()):
        raise ValueError('output directory must be empty')
    write_json(output/'job.json', job)
    write_json(output/'actual-call.json', dict(argv=sys.argv, cwd=str(Path.cwd()), at=datetime.now(timezone.utc).isoformat()))
    spatial = Path(job['spatial'])
    spatial_report = json.loads((spatial/'report.json').read_text())
    if sha(job['hdf5']) != spatial_report['source_sha256']:
        raise ValueError('HDF5 does not match spatial stage')
    z = np.load(spatial/'trajectory.npz', allow_pickle=False)
    n = np.load(job['native'], allow_pickle=False)
    native_report_path = Path(job.get('native_report', str(Path(job['native']).with_name('native-report.json'))))
    native_report = json.loads(native_report_path.read_text())
    if native_report['skeleton_names'] != NAMES or native_report['side'] != str(z['sensor_side']):
        raise ValueError('Human25 topology or recorded side mismatch')
    # Verify the solver input independently against this HDF5. A familiar
    # directory name or a neighbouring report is insufficient provenance.
    input_pose_path = Path(job.get('input_poses', str(Path(job['native']).with_name('input-poses.npz'))))
    inp = np.load(input_pose_path, allow_pickle=False)
    pi = packet_mapping(n['source_index'], inp['source_index'])
    if (pi < 0).any():
        raise ValueError('native output lacks original five-point input')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
    from qn_smith.emf_spatial import decode_native, text_json
    with h5py.File(job['hdf5'], 'r') as f:
        side = str(z['sensor_side'])
        base = f'raw/tactile_emf/{side}/fingertips/native'
        protocol = text_json(f[f'calibration/tactile_emf/{side}/protocol_config_json'])
        expected_pose, expected_valid = decode_native(f[base+'/pose_i16'][:], f[base+'/point_valid'][:], protocol)
        ns = f[base+'/host_receive_monotonic_ns'][:].ravel()
        si = np.asarray(n['source_index'], np.int64)
        if (si < 0).any() or (si >= len(ns)).any():
            raise ValueError('native original packet row out of range')
        if not np.array_equal(inp['receive_ns'][pi], ns[si]) or not np.allclose(inp['poses'][pi], expected_pose[si], atol=1e-12, rtol=0., equal_nan=True):
            raise ValueError('native five points/timestamps do not match this HDF5')
        source_native_valid = expected_valid.all(1) & f[base+'/valid'][:].ravel().astype(bool)
        for key in ('checksum_verified', 'frame_structure_valid'):
            qpath = f'raw/tactile_emf/{side}/native_quality/{key}'
            if qpath in f:
                source_native_valid &= f[qpath][:].ravel().astype(bool)
        source_native_valid = source_native_valid[si]
    bridge = json.loads(Path(job['bridge']).read_text())
    B = np.asarray(bridge['T_emf_solver'], float)
    if not bridge.get('evidence') or B.shape != (4, 4) or not np.isfinite(B).all():
        raise ValueError('explicit finite solver-to-EMF bridge and evidence required')
    if not np.allclose(B[3], [0, 0, 0, 1]) or not np.allclose(B[:3,:3].T@B[:3,:3], np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(B[:3,:3]), 1.):
        raise ValueError('bridge must be a proper SE3 transform')
    camera = json.loads((spatial/'camera.json').read_text())
    if camera['camera_model'] != 'opencv_brown':
        raise ValueError('unsupported raw camera model')
    K, D = np.asarray(camera['K'], float), np.asarray(camera['distortion'], float)
    w, h = camera['width'], camera['height']
    rows = np.asarray(z['source_row'], np.int64)
    count = len(rows)
    times = np.asarray(z['sample_time_s'], float)
    if count < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('source time must increase')
    idx = packet_mapping(z['native_emf_index'], n['source_index'])
    mapped = idx >= 0
    safe_idx = np.maximum(idx, 0)
    p = np.asarray(n['positions_glove_m'], float)
    native_ok = np.asarray(n['tracking_valid'], bool) & np.asarray(n['output_available'], bool) & np.isfinite(p).all((1,2)) & source_native_valid
    filtered = stabilize(p, n['timestamp_sec'], native_ok)
    valid = mapped & z['pose_valid'] & native_ok[safe_idx] & np.isfinite(z['T_camera_emf']).all((1,2))
    valid &= np.abs(z['nearest_receive_delta_ms']) <= float(job.get('max_receive_delta_ms', 30.))
    receive_coverage = ((z['camera_receive_monotonic_ns'] >= ns[0]) &
                        (z['camera_receive_monotonic_ns'] <= ns[-1]))
    valid &= receive_coverage
    # A host receive timestamp is a nearest-sample candidate, not a proven
    # hardware synchronization. Retain this distinction in every output.
    T = z['T_camera_emf'] @ B
    def camera_points(src):
        points = src[safe_idx]
        result = np.einsum('fij,fkj->fki', T[:,:3,:3], points)+T[:,:3,3][:,None]
        result[~valid] = np.nan
        return result
    raw_cam, filtered_cam = camera_points(p), camera_points(filtered)
    raw_uv, filtered_uv = project(raw_cam, K, D), project(filtered_cam, K, D)
    mano_uv = np.full((count, 21, 2), np.nan)
    mano_valid = np.zeros(count, bool)
    input_files = [Path(job['hdf5']), spatial/'trajectory.npz', spatial/'report.json', spatial/'camera.json',
                   Path(job['native']), native_report_path, input_pose_path, Path(job['bridge']), args.job.resolve(), Path(__file__).resolve()]
    if job.get('mano'):
        mano = np.load(job['mano'], allow_pickle=False)
        summary = json.loads(Path(job['mano_summary']).read_text())
        if str(mano['side']) != 'left' or summary['input_sha256'] != spatial_report['source_sha256']:
            raise ValueError('MANO side/source mismatch')
        if not summary['joint_order'].startswith('21 OpenPose') or not summary['camera_space'].startswith('rectified color optical'):
            raise ValueError('unsupported MANO camera/topology')
        if not np.allclose(mano['K'], K, atol=1e-7):
            raise ValueError('MANO rectified K differs: an explicit camera rotation/rectification bridge is needed')
        mi = packet_mapping(rows, mano['source_rows'])
        mano_valid = mi >= 0
        mp = np.asarray(mano['joints_camera_m'], float)[np.maximum(mi, 0)]
        mano_valid &= np.isfinite(mp).all((1,2)) & (mp[..., 2] > 1e-6).all(1)
        mp[~mano_valid] = np.nan
        # The inference used undistort with the same K and no rectification
        # rotation. Forward Brown distortion maps its optical rays to raw RGB.
        mano_uv = project(mp, K, D)
        input_files += [Path(job['mano']), Path(job['mano_summary'])]
    np.savez_compressed(output/'labels.npz', source_row=rows, sample_time_s=times,
        camera_device_relative_s=z['camera_device_relative_s'], native_packet_row=z['native_emf_index'],
        aligned_native_output_index=idx, valid=valid, spatial_pose_valid=z['pose_valid'],
        native_tracking_valid=native_ok[safe_idx]&mapped, nearest_receive_delta_ms=z['nearest_receive_delta_ms'],
        native_receive_coverage=receive_coverage,
        positions_camera_raw_m=raw_cam, positions_camera_filtered_m=filtered_cam,
        uv_raw=raw_uv, uv_filtered=filtered_uv, T_camera_solver=T,
        mano_left_uv=mano_uv, mano_left_valid=mano_valid, training_ready=np.array(False))
    np.savez_compressed(output/'filtered-native-positions.npz', positions_solver_m=filtered,
                        source_index=n['source_index'], timestamp_sec=n['timestamp_sec'], valid=native_ok)
    fps = count/float(times[-1]-times[0]+np.median(np.diff(times)))
    videos = ['human-v2-overlay.mp4', 'human-v2-four-panel.mp4']
    encoders = [Encoder(output/name, (w,h), fps) for name in videos]
    contacts = []
    qa_rows = set(np.linspace(0,count-1,min(9,count)).round().astype(int))
    try:
        with h5py.File(job['hdf5'], 'r') as f:
            payload = f['raw/cameras/head_rgb/payload']
            if not np.array_equal(rows, np.arange(len(payload))):
                raise ValueError('render must cover every original RGB row')
            for i, row in enumerate(rows):
                image = cv2.imdecode(np.asarray(payload[row], np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != (h,w):
                    raise ValueError(f'RGB decode/dimension failure at {row}')
                footer = f'row {row}/{count-1} | {times[i]-times[0]:.3f}s | '+('receive-nearest / calibration unverified' if valid[i] else 'NO VALID SPATIAL/SOLVER SUPPORT - not filled')
                panels = [image.copy() for _ in range(4)]
                draw(panels[1], raw_uv[i], EDGES, (255,190,0))
                draw(panels[2], filtered_uv[i], EDGES, (255,190,0))
                draw(panels[3], filtered_uv[i], EDGES, (255,190,0))
                if job.get('mano'):
                    draw(panels[3], mano_uv[i], MANO_EDGES, (80,230,80))
                else:
                    draw(panels[3], raw_uv[i], EDGES, (0,140,255))
                labels = ['A Original RGB (fixed full FOV)', 'B Recorded RIGHT / native Human25',
                          'C Recorded RIGHT / filtered position candidate',
                          'D RIGHT filtered + LEFT visual MANO' if job.get('mano') else 'D Raw orange / filtered cyan']
                for panel, label in zip(panels,labels):
                    title(panel,label,footer)
                overlay = panels[3] if job.get('mano') else panels[2]
                encoders[0].write(overlay)
                small = [cv2.resize(x,(w//2,h//2),interpolation=cv2.INTER_AREA) for x in panels]
                grid = np.vstack([np.hstack(small[:2]),np.hstack(small[2:])])
                encoders[1].write(grid)
                if i in qa_rows:
                    cv2.imwrite(str(output/f'qa-row-{row:04d}.jpg'),grid)
                    contacts.append(cv2.resize(grid,(640,360)))
                if (i+1)%100 == 0:
                    print(f'rendered {i+1}/{count}',flush=True)
    finally:
        for encoder in encoders:
            encoder.close()
    if contacts:
        while len(contacts)%3:
            contacts.append(np.zeros_like(contacts[0]))
        cv2.imwrite(str(output/'contact-sheet.jpg'), np.vstack([np.hstack(contacts[i:i+3]) for i in range(0,len(contacts),3)]))
    verification = {name: verify_video(output/name,count,w,h) for name in videos}
    report = dict(schema='qn_smith.human_v2_cooperative_render.v1', frames=count, fps=fps,
        duration_s=count/fps, valid_frames=int(valid.sum()), missing_frames=int((~valid).sum()),
        mano_left_frames=int(mano_valid.sum()), recorded_sensor_side=str(z['sensor_side']),
        topology=dict(human_nodes=25,human_edges=EDGES,mano_nodes=21,mano_edges=MANO_EDGES),
        filter=dict(type='OneEuro + median native bone lengths',cutoff_hz=5.,beta=.10,dcutoff_hz=1.,
                    reset_on_invalid_or_gap_s=.1,frame='solver/EMF local',camera_pose_filtered=False),
        missing_policy='No interpolation/hold/extrapolation; missing frames show original RGB and status',
        camera='Original full FOV; no follow/crop/mirror; forward Brown distortion for rectified MANO',
        coordinate_contract='p_C=T_camera_emf @ T_emf_solver @ p_solver; metre column vectors',
        bridge=bridge, physical_synchronization_verified=False,
        anatomical_hand_binding_verified=False, visual_alignment_pass=False, training_ready=False,
        native_source_hashes={str(p.resolve()):sha(p) for p in input_files},
        video_verification=verification,encoder_commands=[e.command for e in encoders])
    write_json(output/'report.json',report)
    manifest={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=sha(p))
              for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'manifest.json',manifest)
    write_json(output/'complete.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
                                          payload_files=len(manifest),training_ready=False))
    print(json.dumps(dict(output=str(output),frames=count,valid=int(valid.sum())),ensure_ascii=False))


if __name__ == '__main__':
    main()
