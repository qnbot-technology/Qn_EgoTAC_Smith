#!/usr/bin/env python3
"""Recorded causal Tag + persistent HumanV2 benchmark, explicitly no vision."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from qn_ego.pipeline import load_config, Journal, env_for
from qn_ego.tag_skeleton import NativeStream, RoiTagDetector, causal_index, map_points
from qn_smith.apriltag_pose import body_poses, detect_frame
from qn_smith.ball23_registration import registered_transforms
from qn_smith.emf_spatial import decode_native, text_json
from qn_smith.artifacts import sha256, write_json, now
from process_ball23_humanv2_tactile import body_configuration
from render_human_v2 import project, draw, EDGES, Encoder, verify_video

SIDES = ('left', 'right')


def stats(values):
    a = np.asarray(values, float)
    return dict(count=len(a), mean_ms=float(a.mean()), p50_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a,95)), max_ms=float(a.max())) if len(a) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT/'configs/pipeline.json')
    p.add_argument('--frames', type=int, default=0, help='0 means full recording')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--modes', nargs='+', choices=['full','roi'], default=['full','roi'])
    p.add_argument('--roi-margin', type=int, default=100)
    p.add_argument('--refresh', type=int, default=30)
    p.add_argument('--right-tag-start', type=int, default=13,
                   help='Physical Ball23 ID for right local layout ID 0 (legacy HDF5 default)')
    p.add_argument('--parallel-workers',action='store_true',help='Overlap independent left/right native solves and Tag search')
    a = p.parse_args()
    if a.frames < 0 or a.refresh < 1 or a.roi_margin < 1:
        p.error('Invalid frame or ROI arguments')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    source = a.input.resolve(); cfg = load_config(a.config)
    write_json(out/'actual-call.json', dict(argv=sys.argv, at=now(), source=str(source),
        visual_correction=False, mode='recorded_causal_replay_not_live_camera', training_ready=False))
    cv2.setNumThreads(a.threads)
    journal = Journal(out, False, env_for(0,'cpu',a.threads))
    journal.call('01_prepare', [sys.executable, ROOT/'scripts/prepare_ball23_recording.py',
        '--source', source, '--run', out, '--registration-template', cfg['registration']],
        [out/'provenance/input.json', out/'provenance/registration.json'])
    source_sha = json.loads((out/'provenance/input.json').read_text())['source_sha256']
    layout = json.loads(cfg['layout'].read_text()); mount = json.loads(cfg['mount'].read_text())
    if layout['mount_sha256'] != sha256(cfg['mount']):
        raise ValueError('Mount binding mismatch')
    bridge = registered_transforms(json.loads((out/'provenance/registration.json').read_text()), layout, mount)
    bodies = body_configuration(layout, right_tag_start=a.right_tag_start); sizes = {str(i):.023 for i in range(27)}
    write_json(out/'provenance/code-and-config.json', dict(files=[dict(path=str(f.relative_to(ROOT)),sha256=sha256(f))
        for f in [Path(__file__),ROOT/'src/qn_ego/tag_skeleton.py',ROOT/'src/qn_smith/apriltag_pose.py',
                  cfg['layout'],cfg['mount'],cfg['humanv2']/'runtime/bin/human_hand_v2_runtime']],
        T_root_emf={s:B.tolist() for s,B in bridge.items()}, transform_applied_once=True,
        point_frame='emf', output_frame='camera_color_optical', unit='metre',
        time_axis='host_receive_monotonic_ns', sync_calibrated=False))
    summaries = {}
    with h5py.File(source,'r') as f:
        c = f['calibration/cameras/head_rgb/color']
        if c.attrs.get('distortion_model') != 'orbbec_model_4':
            raise ValueError('Unreviewed distortion model')
        camera = dict(camera_model='opencv_brown',K=c['K'][:].tolist(),distortion=c['distortion'][:].tolist(),
                      width=int(c.attrs['width']),height=int(c.attrs['height']))
        ns = f['raw/cameras/head_rgb/receive_monotonic_ns'][:].ravel()
        times = f['time/relative_s'][:].ravel()
        n = min(len(ns), a.frames) if a.frames else len(ns)
        frame_ids = f['raw/cameras/head_rgb/frame_index'][:].ravel()
        if n < 2 or np.any(np.diff(ns) < 0):
            raise ValueError('At least two nondecreasing RGB timestamps required')
        duplicate = np.r_[False,np.diff(ns)==0]
        if np.any(duplicate & np.r_[False,np.diff(frame_ids)!=0]):
            raise ValueError('Same receive time with different camera frame identity')
        native = {}
        for s in SIDES:
            base = f'raw/tactile_emf/{s}/fingertips/native'
            proto = text_json(f[f'calibration/tactile_emf/{s}/protocol_config_json'])
            poses, pv = decode_native(f[base+'/pose_i16'][:],f[base+'/point_valid'][:],proto)
            valid = pv.all(1) & f[base+'/valid'][:].ravel().astype(bool)
            for key in ('checksum_verified','frame_structure_valid'):
                path = f'raw/tactile_emf/{s}/native_quality/{key}'
                if path not in f or f[path].size != len(valid):
                    raise ValueError('Native quality metadata required')
                valid &= f[path][:].ravel().astype(bool)
            t = f[base+'/host_receive_monotonic_ns'][:].ravel()
            causal_index(t,ns[0])  # validates monotonic native axis
            native[s] = dict(poses=poses,valid=valid,ns=t)
        for mode in a.modes:
            target = out/mode; target.mkdir()
            workers = {}; last = {s:None for s in SIDES}; cursor = {s:-1 for s in SIDES}
            packet_ms = {s:[] for s in SIDES}
            points = {s:np.full((n,25,3),np.nan) for s in SIDES}
            transforms = {s:np.full((n,4,4),np.nan) for s in SIDES}
            tag_valid = {s:np.zeros(n,bool) for s in SIDES}
            mapping_valid = {s:np.zeros(n,bool) for s in SIDES}
            indices = {s:np.full(n,-1,np.int64) for s in SIDES}
            ages = {s:np.full(n,np.nan) for s in SIDES}
            rows = []; detector = RoiTagDetector(camera,sizes,bodies,a.roi_margin,a.refresh)
            pool = ThreadPoolExecutor(max_workers=3) if a.parallel_workers else None
            def advance(s,i):
                tic = time.perf_counter()
                data = native[s]; ix, age, ok = causal_index(data['ns'],ns[i])
                indices[s][i] = ix; ages[s][i] = age if np.isfinite(age) else np.nan
                while cursor[s] < ix:
                    j = cursor[s]+1
                    reply, ms = workers[s].solve(data['poses'][j],1.+(int(data['ns'][j])-int(data['ns'][0]))/1e9,data['valid'][j])
                    last[s] = reply; cursor[s] = j; packet_ms[s].append(ms)
                return (time.perf_counter()-tic)*1000
            def tag(image):
                tic = time.perf_counter()
                if mode=='full':
                    row = detect_frame(image,camera,sizes); search = dict(search='full',boxes=[])
                else:row, search = detector.detect(image)
                roots = body_poses(row,bodies,sizes,camera)
                return row,search,roots,(time.perf_counter()-tic)*1000
            try:
                for s in SIDES:
                    workers[s] = NativeStream(cfg['humanv2'],out/'provenance'/f'embedded-seven-gesture-{s}.yaml',target/s,s)
                for i in range(n):
                    start = time.perf_counter()
                    image = cv2.imdecode(np.asarray(f['raw/cameras/head_rgb/payload'][i],np.uint8),1)
                    decoded = time.perf_counter()
                    if pool:
                        jobs = [pool.submit(advance,s,i) for s in SIDES]
                        tag_job = pool.submit(tag,image)
                        native_times = [j.result() for j in jobs]
                        row,search,roots,tag_ms = tag_job.result()
                    else:
                        native_times = [advance(s,i) for s in SIDES]
                        row,search,roots,tag_ms = tag(image)
                    mapped = time.perf_counter()
                    if duplicate[i]:
                        for root in roots.values():
                            root.update(valid_current_observation=False,T_camera_root=None,
                                        invalid_reason='duplicate_camera_observation')
                    for s in SIDES:
                        tag_valid[s][i] = roots[s]['valid_current_observation']
                        if tag_valid[s][i]:transforms[s][i] = roots[s]['T_camera_root']
                        _, _, ok = causal_index(native[s]['ns'],ns[i])
                        points[s][i] = map_points(roots[s],bridge[s],last[s],ok)
                        mapping_valid[s][i] = np.isfinite(points[s][i]).all()
                    end = time.perf_counter()
                    timing = dict(source_row=i,camera_receive_ns=int(ns[i]),duplicate_camera_observation=bool(duplicate[i]),
                        decode_ms=(decoded-start)*1000,native_ms=max(native_times) if pool else sum(native_times),
                        tag_pnp_ms=tag_ms,map_ms=(end-mapped)*1000,
                        total_ms=(end-start)*1000,**search,
                        tag_valid={s:bool(tag_valid[s][i]) for s in SIDES},
                        mapping_valid={s:bool(mapping_valid[s][i]) for s in SIDES})
                    rows.append(timing)
                    if i%100==0:print(mode,i,n,'total_ms',round(timing['total_ms'],2),flush=True)
            finally:
                if pool:pool.shutdown(wait=True)
                for w in workers.values():w.close()
            (target/'frame-timings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            arrays = dict(source_row=np.arange(n),camera_receive_ns=ns[:n],sample_time_s=times[:n],
                          camera_frame_index=frame_ids[:n],duplicate_camera_observation=duplicate[:n])
            for s in SIDES:
                arrays.update({s+'_camera_points_m':points[s],s+'_T_camera_root':transforms[s],
                    s+'_tag_valid':tag_valid[s],s+'_mapping_valid':mapping_valid[s],
                    s+'_emf_index':indices[s],s+'_emf_age_ms':ages[s]})
            np.savez_compressed(target/'mapped-skeleton.npz',**arrays)
            warm = rows[1:]
            total = np.asarray([r['total_ms'] for r in warm])
            summary = dict(frames=n,duplicate_camera_observations=int(duplicate[:n].sum()),first_frame_ms=rows[0]['total_ms'],warm_frames=len(warm),
                warm_stages={k:stats([r[k] for r in warm]) for k in ['decode_ms','native_ms','tag_pnp_ms','map_ms','total_ms']},
                warm_compute_fps=1000/float(total.mean()),deadline_30fps_miss_fraction=float(np.mean(total>1000/30)),
                tag_valid={s:int(tag_valid[s].sum()) for s in SIDES},
                mapping_valid={s:int(mapping_valid[s].sum()) for s in SIDES},
                search_counts={kind:sum(r['search']==kind for r in rows) for kind in ['full','roi']},
                native={s:dict(packets=cursor[s]+1,warm_request_roundtrip=stats(packet_ms[s][1:]),
                    first_reply_since_launch_ms=workers[s].first_reply_ms,
                    command=workers[s].command) for s in SIDES},
                future_native_samples_used=0,max_emf_age_ms=30.,
                sequential_workers=not a.parallel_workers,video_encoding_in_timing=False,live_capture_tested=False,training_ready=False)
            write_json(target/'summary.json',summary); summaries[mode]=summary
            # Render after timing, not hidden inside claimed compute throughput.
            uv = {s:project(points[s],np.asarray(camera['K']),np.asarray(camera['distortion'])) for s in SIDES}
            video = target/'tag-only-preview.mp4'
            encoder = Encoder(video,(camera['width'],camera['height']),1e9/float(np.median(np.diff(ns[:n]))),square_pixels=True)
            try:
                for i in range(n):
                    image = cv2.imdecode(np.asarray(f['raw/cameras/head_rgb/payload'][i],np.uint8),1)
                    for s,color in zip(SIDES,[(70,210,255),(255,160,80)]):draw(image,uv[s][i],EDGES,color)
                    encoder.write(image)
            finally:encoder.close()
            write_json(target/'video-verification.json',verify_video(video,n,camera['width'],camera['height']))
            print(json.dumps(dict(mode=mode,summary=summary)),flush=True)
    comparison = {}
    if 'full' in summaries and 'roi' in summaries:
        with np.load(out/'full/mapped-skeleton.npz') as b, np.load(out/'roi/mapped-skeleton.npz') as r:
            for s in SIDES:
                bv,rv = b[s+'_tag_valid'],r[s+'_tag_valid']; both = bv&rv
                delta = np.linalg.norm(b[s+'_T_camera_root'][both,:3,3]-r[s+'_T_camera_root'][both,:3,3],axis=1)*1000
                rot = b[s+'_T_camera_root'][both,:3,:3].transpose(0,2,1)@r[s+'_T_camera_root'][both,:3,:3]
                degrees = np.degrees(np.arccos(np.clip((np.trace(rot,axis1=1,axis2=2)-1)/2,-1,1)))
                comparison[s] = dict(both_valid=int(both.sum()),lost_vs_full=int((bv&~rv).sum()),
                    gained_vs_full=int((~bv&rv).sum()),translation_delta_mm_p50_p95_max=np.percentile(delta,[50,95,100]).tolist() if len(delta) else [],
                    rotation_delta_deg_p50_p95_max=np.percentile(degrees,[50,95,100]).tolist() if len(degrees) else [])
    unchanged = source_sha == sha256(source)
    write_json(out/'summary.json',dict(schema='qn_ego.tag_skeleton_benchmark.v1',at=now(),source_sha256=source_sha,
        raw_unchanged=unchanged,frames=n,total_source_frames=len(ns),visual_correction=False,
        benchmark='unpaced_recorded_causal_replay',includes=['JPEG decode','all arrived EMF packets','current Tag detection','PnP','EMF skeleton mapping'],
        excludes=['camera exposure/transport','live collector integration','UI rendering/encoding','hardware synchronization calibration'],
        training_ready=False,modes=summaries,roi_comparison=comparison))
    if not unchanged:raise ValueError('Source changed')


if __name__=='__main__':main()
