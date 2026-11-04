#!/usr/bin/env python3
"""Source-bound dual ball23 -> native HumanV2 -> RGB + tactile delivery.

Explicitly overrides obsolete ball251 metadata without changing recordings.
No pose filling, image mirroring, inferred world frame or force calibration.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
from qn_smith.apriltag_pose import detect_frame, body_poses, camera_contract
from qn_smith.emf_spatial import decode_native, nearest_indices, text_json
from qn_ego.tag_skeleton import RoiTagDetector
from qn_smith.tactile_protocol import FINGERS, PROTOCOL_SHA256, finger_channels, finger_strength, display_grid, palm_channels, palm_strength
from qn_smith.ball23_registration import registered_transforms
from qn_smith.tactile_hand_panel import render_panel as render_hand_panel
from render_human_v2 import NAMES, EDGES, stabilize, project, draw, Encoder, verify_video, sha, write_json

SIDES = ('left', 'right')
COLORS = ((70, 210, 255), (255, 160, 80))


def electrical_axis_transform(config):
    """Map runtime EMF coordinates into the CAD-predicted EMF frame.

    This is a rotation about the unchanged EMF origin, not a solver bridge,
    camera correction, left/right reflection, or additional fixture offset.
    """
    if config['asset_id'] != 'apriltag_emf_ball23' or config['applies_to_sides'] != list(SIDES):
        raise ValueError('explicit same-assembly, dual-hand ball23 registration required')
    A = np.asarray(config['T_cad_emf_emf_runtime'], dtype=float)
    if A.shape != (4, 4) or not np.isfinite(A).all():
        raise ValueError('finite homogeneous axis transform required')
    if (not np.allclose(A[3], [0, 0, 0, 1], atol=1e-10)
            or not np.allclose(A[:3, 3], 0, atol=1e-10)
            or not np.allclose(A[:3, :3].T @ A[:3, :3], np.eye(3), atol=1e-10)
            or not np.isclose(np.linalg.det(A[:3, :3]), 1, atol=1e-10)):
        raise ValueError('proper rotation with zero added translation required')
    return A


def body_configuration(layout, *, right_tag_start=13):
    if layout['asset_id'] not in ('apriltag_emf_ball23', 'apriltag_emf_ball26') or layout['tag_size_m'] not in (.023, .026):
        raise ValueError('explicit 23mm or 26mm Ball layout required')
    if set(map(int, layout['T_bottom_tag'])) != set(range(10)):
        raise ValueError('only observed bound IDs 0..9 may enter root estimation')
    if int(right_tag_start) < 0:
        raise ValueError('right tag start must be non-negative')
    bodies = {}
    for side, offset in zip(SIDES, (0, int(right_tag_start))):
        bodies[side] = dict(root_frame='fixture_bottom_emf_axes_candidate',
            extrinsic_evidence=f'ball23 layout + right physical tag start {int(right_tag_start)}, same printed order/roll',
            T_root_tag={str(int(k)+offset): v for k, v in layout['T_bottom_tag'].items()},
            max_reprojection_rms_px=5., max_tag_reprojection_rms_px=6.,
            consensus_outlier_rejection=True, consensus_tag_rms_px=10., min_consensus_fraction=.6)
    return bodies


def nearest_contract(ns, query, max_delta_ms):
    index = nearest_indices(ns, query)
    delta = (ns[index]-query)/1e6
    covered = (query >= ns[0]) & (query <= ns[-1])
    return index, delta, covered & (abs(delta) <= max_delta_ms)


def trajectory(source, out, layout_path, mount_path, *, roi=False, roi_margin=180, roi_refresh=30,
               right_tag_start=13, frame_workers=1):
    out.mkdir(parents=True, exist_ok=True)
    source_sha = sha(source)
    layout = json.loads(layout_path.read_text())
    mount = json.loads(mount_path.read_text())
    if layout['mount_sha256'] != sha(mount_path):
        raise ValueError('layout / mount provenance mismatch')
    B = np.asarray(layout['T_render_root_emf_cad_candidate'])
    R = np.asarray(layout['T_render_root_cad_bottom'])
    np.testing.assert_allclose(B, R @ mount['transforms']['T_bottom_emf']['matrix'], atol=1e-12)
    bodies = body_configuration(layout, right_tag_start=right_tag_start)
    # IDs 0..26 are valid Ball23 labels. The layout currently binds ten local
    # positions per hand; labels beyond the bound positions remain diagnostics
    # until their CAD transforms are measured.
    sizes = {str(i): float(layout['tag_size_m']) for i in range(27)}
    arrays = {}
    with h5py.File(source, 'r') as f:
        c = f['calibration/cameras/head_rgb/color']
        if c.attrs.get('distortion_model') != 'orbbec_model_4':
            raise ValueError('unreviewed camera distortion model')
        camera = dict(camera_model='opencv_brown', K=c['K'][:].tolist(), distortion=c['distortion'][:].tolist(),
            width=int(c.attrs['width']), height=int(c.attrs['height']), source='embedded Orbbec color model4, raw JPEG')
        camera_contract(camera)
        write_json(out/'camera.json', camera)
        for name, path in [('embedded-ball251.json', 'calibration/rigid_bodies/apriltag_emf_ball251/config_json')]:
            if path in f: write_json(out/name, text_json(f[path]))
        count = len(f['time/relative_s'])
        arrays.update(source_row=np.arange(count), sample_time_s=f['time/relative_s'][:].ravel(),
            sample_monotonic_ns=f['time/sample_monotonic_ns'][:].ravel(),
            camera_receive_ns=f['raw/cameras/head_rgb/receive_monotonic_ns'][:].ravel(),
            camera_device_us=f['raw/cameras/head_rgb/color_device_timestamp_us'][:].ravel(),
            camera_frame_index=f['raw/cameras/head_rgb/frame_index'][:].ravel())
        if np.any(np.diff(arrays['sample_time_s']) <= 0): raise ValueError('non-increasing source timeline')
        query = arrays['camera_receive_ns']
        for side in SIDES:
            root = out/f'native-{side}'
            root.mkdir(exist_ok=True)
            base = f'raw/tactile_emf/{side}/fingertips/native'
            protocol = text_json(f[f'calibration/tactile_emf/{side}/protocol_config_json'])
            poses, pv = decode_native(f[base+'/pose_i16'][:], f[base+'/point_valid'][:], protocol)
            valid = pv.all(1) & f[base+'/valid'][:].ravel().astype(bool)
            for key in ('checksum_verified', 'frame_structure_valid'):
                qp = f'raw/tactile_emf/{side}/native_quality/{key}'
                if qp in f and f[qp].size == len(valid): valid &= f[qp][:].ravel().astype(bool)
            if not valid.all():
                # Keep the source immutable and fail before Tag/PnP rather than
                # silently copying a neighboring hand pose.  The compact marker
                # is parsed by the unified entry point so platform callers get a
                # short actionable error while the complete traceback remains in
                # calls/02_apriltag.log for audit.
                bad = np.flatnonzero(~valid)
                details = []
                quality_names = ('checksum_verified', 'frame_structure_valid')
                for i in bad[:8]:
                    missing = [FINGERS[j] for j, ok in enumerate(np.asarray(pv[i]).ravel()[:5]) if not bool(ok)]
                    reasons = []
                    if missing:
                        reasons.append('missing_points=' + ','.join(missing))
                    for key in quality_names:
                        qp = f'raw/tactile_emf/{side}/native_quality/{key}'
                        if qp in f and not bool(f[qp][i]):
                            reasons.append(key + '=0')
                    details.append(f'row={int(i)} ' + (' '.join(reasons) if reasons else 'valid=0'))
                extra = '' if len(bad) <= 8 else f' (+{len(bad)-8} more)'
                raise ValueError('EMF_DATA_LOSS: side=' + side + ' invalid_packets=' + str(len(bad)) +
                                 ' ' + '; '.join(details) + extra)
            ns = f[base+'/host_receive_monotonic_ns'][:].ravel()
            np.savez_compressed(root/'prepared-input.npz', poses=poses, receive_ns=ns, source_index=np.arange(len(ns)))
            write_json(root/'protocol.json', protocol)
            ix, delta, ok = nearest_contract(ns, query, 30.)
            arrays.update({side+'_emf_index': ix, side+'_emf_delta_ms': delta, side+'_emf_time_valid': ok})
            tactile_base = f'raw/tactile_emf/{side}/tactile/native'
            tn = f[tactile_base+'/host_receive_monotonic_ns'][:].ravel()
            ti, td, tok = nearest_contract(tn, query, 50.)
            values = f[tactile_base+'/values_u8'][:].reshape(-1, 460)[ti]
            tv = f[tactile_base+'/valid'][:].ravel().astype(bool)[ti] & tok
            peak, mean = finger_strength(values, tv, side)
            arrays.update({side+'_tactile_index': ti, side+'_tactile_delta_ms': td, side+'_tactile_valid': tv,
                           side+'_tactile_values': values, side+'_finger_peak_count': peak, side+'_finger_mean_count': mean})
            arrays[side+'_pose_valid'] = np.zeros(count, bool)
            arrays[side+'_T_camera_root'] = np.full((count,4,4), np.nan)
            arrays[side+'_rms_px'] = np.full(count, np.nan)
        detector = RoiTagDetector(camera, sizes, bodies, roi_margin, roi_refresh) if roi else None
        search_counts = {'full': 0, 'roi': 0}
        # ROI search carries a causal last-seen state and must remain serial.
        # Full-frame detection is independent per RGB row, so decode and PnP
        # can be overlapped safely.  Results are consumed in source-row order
        # to keep arrays, JSONL, and all timestamps deterministic.
        frame_workers = max(1, int(frame_workers))
        if roi:
            frame_workers = 1
        if frame_workers > 1:
            cv2.setNumThreads(1)

        def detect_one(item):
            i, payload = item
            image = cv2.imdecode(np.asarray(payload, np.uint8), cv2.IMREAD_COLOR)
            row = detect_frame(image, camera, sizes)
            roots = body_poses(row, bodies, sizes, camera)
            return i, row, roots

        def bounded_parallel_results(payloads):
            """Yield frame results in source order with a bounded work window.

            ``Executor.map`` eagerly submits the whole HDF5 dataset on Python
            versions before ``buffersize`` was added.  That made long
            recordings retain every compressed JPEG in memory while the
            AprilTag workers caught up.  Keep only two windows of work in
            flight; HDF5 remains read-only and output ordering stays exact.
            """
            window = max(1, frame_workers * 2)
            with ThreadPoolExecutor(max_workers=frame_workers,
                                    thread_name_prefix='apriltag') as pool:
                source = enumerate(payloads)
                pending = {}
                for _ in range(window):
                    try:
                        i, payload = next(source)
                    except StopIteration:
                        break
                    pending[i] = pool.submit(detect_one, (i, payload))
                next_index = 0
                while pending:
                    result = pending.pop(next_index).result()
                    yield result
                    next_index += 1
                    try:
                        i, payload = next(source)
                    except StopIteration:
                        continue
                    pending[i] = pool.submit(detect_one, (i, payload))

        with (out/'detections-and-poses.jsonl').open('w') as journal:
            payloads = f['raw/cameras/head_rgb/payload']
            if detector is None and frame_workers > 1:
                iterator = bounded_parallel_results(payloads)
            else:
                iterator = None
            if iterator is not None:
                frame_results = iterator
            else:
                def serial_results():
                    for i, payload in enumerate(payloads):
                        image = cv2.imdecode(np.asarray(payload, np.uint8), cv2.IMREAD_COLOR)
                        if detector is None:
                            row = detect_frame(image, camera, sizes)
                            roots = body_poses(row, bodies, sizes, camera)
                        else:
                            row, info = detector.detect(image)
                            roots = body_poses(row, bodies, sizes, camera)
                        yield i, row, roots
                frame_results = serial_results()
            for i, row, roots in frame_results:
                search = 'full' if detector is None else 'roi'
                search_counts[search] += 1
                for side in SIDES:
                    p = roots[side]
                    if p['reprojection_rms_px'] is not None: arrays[side+'_rms_px'][i] = p['reprojection_rms_px']
                    if p['valid_current_observation']:
                        arrays[side+'_pose_valid'][i] = True
                        arrays[side+'_T_camera_root'][i] = p['T_camera_root']
                journal.write(json.dumps(dict(source_row=i, sample_time_s=float(arrays['sample_time_s'][i]),
                    camera_receive_ns=int(query[i]), search=search, **row, bodies=roots), separators=(',', ':'))+'\n')
                if i % 100 == 0: print(f'{source.stem} trajectory {i}/{count}', flush=True)
    arrays['T_render_root_emf'] = B
    for side in SIDES:
        arrays[side+'_T_camera_emf'] = arrays[side+'_T_camera_root'] @ B
    np.savez_compressed(out/'trajectory-and-tactile.npz', **arrays)
    with (out/'frame-map.csv').open('w') as fo:
        keys = ['source_row', 'sample_time_s', 'camera_receive_ns', 'camera_frame_index']
        keys += [side+suffix for side in SIDES for suffix in ('_pose_valid', '_rms_px', '_emf_index', '_emf_delta_ms',
                                                            '_tactile_index', '_tactile_delta_ms', '_tactile_valid')]
        writer = csv.writer(fo); writer.writerow(keys)
        writer.writerows(zip(*(arrays[k] for k in keys)))
    write_json(out/'configuration.json', dict(layout=layout, mount=mount, bodies=bodies, tag_sizes_m=sizes,
        tag_binding=dict(left_start=0, right_start=int(right_tag_start),
                         status=('temporary_mcap_correction' if int(right_tag_start) == 15 else 'legacy_or_explicit'),
                         bound_local_ids=list(range(10)),
                         bound_right_ids=list(range(int(right_tag_start), int(right_tag_start)+10)),
                         declared_right_ids=list(range(int(right_tag_start), int(right_tag_start)+12)),
                         note='Only local CAD IDs 0..9 are bound; additional observed labels need CAD transforms'),
        tag_search=dict(mode='roi' if roi else 'full', margin_px=roi_margin if roi else None,
                         refresh_frames=roi_refresh if roi else None,
                         frame_workers=1 if roi else frame_workers, counts=search_counts),
        layout_source=str(layout_path), layout_sha256=sha(layout_path), mount_source=str(mount_path),
        mount_sha256=sha(mount_path), old_embedded_ball251_overridden=True,
        solver_to_emf=dict(T=np.eye(4).tolist(), evidence='human_hand_v2.cpp skeleton_glove = wrist_pose * skeleton_wrist'),
        tactile=dict(protocol_sha256=PROTOCOL_SHA256, pages=[6,7],
            channels={side: finger_channels(side) for side in SIDES}, aggregation='max and mean of 24 cells/finger',
            display='fixed sqrt(count/255); raw full grid preserved; not Newtons'),
        no_interpolation=True, frame='camera_color_optical; CAD electrical-axis candidate',
        electrical_axes_verified=False, hardware_sync_verified=False))
    def finite_median(values):
        values = np.asarray(values, float)
        return float(np.nanmedian(values)) if np.isfinite(values).any() else None
    report = dict(source=str(source), source_sha256=source_sha, frames=count,
        tag_binding=dict(left_start=0, right_start=int(right_tag_start),
                         status=('temporary_mcap_correction' if int(right_tag_start) == 15 else 'legacy_or_explicit'),
                         bound_local_ids=list(range(10)),
                         bound_right_ids=list(range(int(right_tag_start), int(right_tag_start)+10)),
                         declared_right_ids=list(range(int(right_tag_start), int(right_tag_start)+12))),
        tag_search=dict(mode='roi' if roi else 'full', margin_px=roi_margin if roi else None,
                         refresh_frames=roi_refresh if roi else None,
                         frame_workers=1 if roi else frame_workers, counts=search_counts),
        repeated_camera_frames=int(np.sum(np.diff(arrays['camera_frame_index']) == 0)),
        sides={s: dict(pose_valid=int(arrays[s+'_pose_valid'].sum()),
                      emf_time_valid=int(arrays[s+'_emf_time_valid'].sum()),
                      tactile_valid=int(arrays[s+'_tactile_valid'].sum()),
                      rms_median_px=finite_median(arrays[s+'_rms_px'])) for s in SIDES},
        training_ready=False, reason='CAD electrical registration, bundled personal HumanV2 calibration and visual joint accuracy unverified')
    write_json(out/'trajectory-report.json', report)
    print(json.dumps(report), flush=True)


def native(out, package):
    for side in SIDES:
        root = out/f'native-{side}'
        cmd = [sys.executable, str(PROJECT/'scripts/run_human_hand_v2.py'), '--input-npz', str(root/'prepared-input.npz'),
               '--output', str(root), '--package', str(package), '--side', side]
        print(f'{out.name} HumanV2 {side} starting', flush=True)
        with (root/'driver-stdout.log').open('w') as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        print(f'{out.name} HumanV2 {side} complete', flush=True)


def put(image, text, x, y, scale=.65, color=(235,235,240)):
    cv2.putText(image, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def tactile_panel(canvas, x0, z, i, axis_label):
    panel = canvas[:, x0:]
    panel[:] = (24,27,35)
    put(panel, 'TACTILE | finger contact', 20, 37, .88)
    put(panel, 'Weak                       Strong', 20, 70)
    put(panel, 'Raw counts; fixed display scale, no force units', 20, 102, .55)
    for si, side in enumerate(SIDES):
        x = 20+si*300; color = COLORS[si]
        valid = bool(z[side+'_tactile_valid'][i])
        put(panel, side.upper(), x, 147, .8, color)
        put(panel, 'RECEIVED' if valid else 'MISSING', x, 175, .55, (160,220,160) if valid else (100,100,250))
        peak = z[side+'_finger_peak_count'][i]
        for fi, name in enumerate(FINGERS):
            y = 215+fi*59
            put(panel, name.title(), x, y, .65)
            cv2.rectangle(panel,(x,y+8),(x+270,y+27),(60,60,66),-1)
            if valid:
                width = int(270*np.sqrt(peak[fi]/255.))
                if width: cv2.rectangle(panel,(x,y+8),(x+width,y+27),color,-1)
                put(panel, str(int(peak[fi])), x+223, y, .6, color)
        put(panel,'Full 23 x 20 grid',x,540,.6)
        if valid:
            grid = np.rint(255*np.sqrt(display_grid(z[side+'_tactile_values'][i]).astype(float)/255)).astype(np.uint8)
            heat = cv2.applyColorMap(grid, cv2.COLORMAP_INFERNO)
            heat = cv2.resize(heat, (260,299), interpolation=cv2.INTER_NEAREST)
            panel[558:857,x:x+260] = heat
            # Divider under top six rows; do not attribute palm/white cells to fingers.
            cv2.line(panel,(x,636),(x+259,636),(240,240,240),1)
        else: put(panel,'NO VALID PACKET',x,650,.6)
        put(panel,f'dt {z[side+"_tactile_delta_ms"][i]:+.1f} ms',x,887,.6)
        pose_ok = z[side+'_pose_valid'][i]
        put(panel,'TAG ROOT OK' if pose_ok else 'TAG ROOT MISSING',x,931,.6,color)
        if pose_ok: put(panel,f'RMS {z[side+"_rms_px"][i]:.2f} px',x,961,.6)
        put(panel,f'EMF dt {z[side+"_emf_delta_ms"][i]:+.1f} ms',x,992,.6)
    put(panel,axis_label,20,1035,.55)
    put(panel,'HumanV2 local smoothing; no camera following',20,1060,.55)


def render(source, out, electrical_bridge=None, render_subdir=None, mechanical_registration=None,
           tactile_display='bars', tactile_signal='raw', tactile_residual_subdir=None, prepare_only=False):
    report = json.loads((out/'trajectory-report.json').read_text())
    if report['source_sha256'] != sha(source): raise ValueError('source changed after trajectory stage')
    dest = out/render_subdir if render_subdir else out
    dest.mkdir(parents=True, exist_ok=True)
    A = np.eye(4)
    bridge = None
    if electrical_bridge:
        if not render_subdir:
            raise ValueError('axis registration must use an isolated render subdirectory')
        bridge = json.loads(electrical_bridge.read_text())
        base_config = json.loads((out/'configuration.json').read_text())
        if bridge['layout_sha256'] != base_config['layout_sha256']:
            raise ValueError('electrical registration belongs to a different CAD layout')
        if report['source_sha256'] not in bridge['source_sha256']:
            raise ValueError('electrical registration has not been scoped to this recording')
        A = electrical_axis_transform(bridge)
    with np.load(out/'trajectory-and-tactile.npz') as data:
        z = dict(data)
    if tactile_display == 'hands':
        for side in SIDES:
            z[side+'_palm_peak_count'], z[side+'_palm_mean_count'] = palm_strength(
                z[side+'_tactile_values'], z[side+'_tactile_valid'], side)
    residual_metadata = None
    if tactile_signal == 'residual':
        if tactile_display != 'hands' or not render_subdir or not tactile_residual_subdir:
            raise ValueError('residual display requires hands, isolated output, and prepared residual folder')
        residual_root=out/tactile_residual_subdir
        residual_metadata=json.loads((residual_root/'baseline-model.json').read_text())
        if residual_metadata['source_sha256'] != report['source_sha256']:
            raise ValueError('residual source binding mismatch')
        if residual_metadata.get('no_load_confirmed') or not residual_metadata.get('baseline_estimated'):
            raise ValueError('this UI labels estimated-baseline candidates; confirmed calibration needs explicit new labeling')
        with np.load(residual_root/'frame-residual.npz') as residual:
            for key in ('source_row','sample_time_s','camera_receive_ns'):
                np.testing.assert_array_equal(residual[key],z[key])
            for side in SIDES:
                for suffix in ('_tactile_values','_tactile_index','_tactile_valid','_tactile_delta_ms'):
                    np.testing.assert_array_equal(residual[side+suffix],z[side+suffix])
            for key in residual.files:
                if key not in z: z[key]=residual[key]
    B_by_side = {s: z['T_render_root_emf'] @ A for s in SIDES}
    registration = None
    if mechanical_registration:
        if electrical_bridge or not render_subdir:
            raise ValueError('mechanical registration embeds the selected axes; use alone in an isolated folder')
        registration = json.loads(mechanical_registration.read_text())
        base_config = json.loads((out/'configuration.json').read_text())
        if (registration['layout_sha256'] != base_config['layout_sha256']
                or registration['mount_sha256'] != base_config['mount_sha256']
                or report['source_sha256'] not in registration['source_sha256']):
            raise ValueError('mechanical registration source/layout/mount binding mismatch')
        B_by_side = registered_transforms(registration, base_config['layout'], base_config['mount'])
    c = json.loads((out/'camera.json').read_text())
    K,D = camera_contract(c)
    count = report['frames']; labels = {}; cache = {}; applied_trajectory = {}; calibrations = {}
    for side in SIDES:
        nr = out/f'native-{side}'
        n = np.load(nr/'native-output.npz')
        inp = np.load(nr/'input-poses.npz')
        prep = np.load(nr/'prepared-input.npz')
        meta = json.loads((nr/'native-report.json').read_text())
        calibrations[side] = {k: meta.get(k) for k in ('config', 'config_sha256',
            'calibration_source', 'calibration_source_sha256', 'bundled_calibration_used')}
        if meta['skeleton_names'] != NAMES or meta['side'] != side: raise ValueError('topology or side mismatch')
        if not np.array_equal(n['source_index'], prep['source_index']): raise ValueError('native row identity mismatch')
        np.testing.assert_array_equal(inp['poses'], prep['poses'])
        np.testing.assert_array_equal(inp['receive_ns'], prep['receive_ns'])
        p = n['positions_glove_m']
        valid_native = n['tracking_valid'] & n['output_available'] & np.isfinite(p).all((1,2))
        pf = stabilize(p, n['timestamp_sec'], valid_native)
        filtered_dest = dest/f'filtered-positions-{side}.npz' if render_subdir else nr/'filtered-positions.npz'
        np.savez_compressed(filtered_dest, positions_glove_m=pf, source_index=n['source_index'],
                            receive_ns=inp['receive_ns'], valid=valid_native)
        ix = z[side+'_emf_index']
        valid = z[side+'_pose_valid'] & z[side+'_emf_time_valid'] & valid_native[ix]
        B = B_by_side[side]
        T = z[side+'_T_camera_root'] @ B
        applied_trajectory.update({side+'_T_camera_emf': T, side+'_T_camera_root': z[side+'_T_camera_root'],
                                   side+'_T_root_emf_applied': B,
                                   side+'_pose_valid': z[side+'_pose_valid']})
        def transform(v):
            outp = np.einsum('fij,fkj->fki',T[:,:3,:3],v[ix])+T[:,:3,3][:,None]
            outp[~valid] = np.nan
            return outp
        raw, smooth = transform(p), transform(pf)
        uv = project(smooth,K,D)
        labels.update({side+'_valid': valid, side+'_raw_camera_m': raw, side+'_filtered_camera_m': smooth,
                       side+'_filtered_uv_px': uv})
        cache[side] = uv
    np.savez_compressed(dest/'bone-projections.npz', **labels, source_row=z['source_row'], sample_time_s=z['sample_time_s'])
    if not registration:
        applied_trajectory['T_cad_emf_emf_runtime'] = A
    np.savez_compressed(dest/'applied-camera-trajectory.npz', **applied_trajectory,
                        electrical_bridge_applied_separately=np.array(not bool(registration)),
                        source_row=z['source_row'], camera_receive_ns=z['camera_receive_ns'], sample_time_s=z['sample_time_s'])
    write_json(dest/'render-configuration.json', dict(
        baseline_directory=str(out), source_sha256=report['source_sha256'],
        electrical_bridge=bridge, electrical_bridge_path=str(electrical_bridge) if electrical_bridge else None,
        electrical_bridge_sha256=sha(electrical_bridge) if electrical_bridge else None,
        mechanical_registration=registration,
        mechanical_registration_path=str(mechanical_registration) if mechanical_registration else None,
        mechanical_registration_sha256=sha(mechanical_registration) if mechanical_registration else None,
        formula='p_camera = T_camera_root @ T_root_emf_applied @ p_emf',
        T_cad_emf_emf_runtime=A.tolist() if not registration else None,
        T_root_emf_applied={s: b.tolist() for s,b in B_by_side.items()},
        electrical_bridge_applied_separately=not bool(registration),
        solver_to_emf='identity; unchanged', camera_frame='camera_color_optical',
        source_rgb_unmirrored=True, input_trajectory_unchanged=True, hardware_sync_verified=False,
        tactile_display=tactile_display,
        native_calibrations=calibrations,
        tactile_signal=tactile_signal, tactile_baseline=residual_metadata,
        tactile_residual_directory=str(out/tactile_residual_subdir) if tactile_residual_subdir else None,
        tactile_display_semantics=('fixed shaded palms; five fingers and palm sqrt(peak/255); '
            + ('channel baseline and deadband removed before aggregation; estimated display only' if tactile_signal=='residual' else 'raw counts; missing is not zero')) if tactile_display=='hands' else 'per-finger bars and raw grid',
        electrical_axes_verified=False, training_ready=False))
    if tactile_display == 'hands':
        display_data = {k:z[k] for k in ('source_row','sample_time_s','camera_receive_ns')}
        for side in SIDES:
            for suffix in ('_tactile_valid','_tactile_index','_tactile_delta_ms','_finger_peak_count','_finger_mean_count','_palm_peak_count','_palm_mean_count'):
                display_data[side+suffix] = z[side+suffix]
            if tactile_signal=='residual':
                for suffix in ('_tactile_values','_signed_residual_count','_positive_residual_count',
                               '_baseline_count','_noise_proxy_count','_deadband_count',
                               '_finger_peak_residual_count','_finger_mean_residual_count',
                               '_palm_peak_residual_count','_palm_mean_residual_count'):
                    display_data[side+suffix]=z[side+suffix]
            suffix='_residual_count' if tactile_signal=='residual' else '_count'
            display_data[side+'_display_level'] = np.sqrt(z[side+'_finger_peak'+suffix]/255.)
            display_data[side+'_palm_display_level'] = np.sqrt(z[side+'_palm_peak'+suffix]/255.)
        np.savez_compressed(dest/'tactile-display.npz', **display_data)
        write_json(dest/'tactile-display-mapping.json', dict(
            source_sha256=report['source_sha256'], protocol_sha256=PROTOCOL_SHA256,
            channels={side:finger_channels(side) for side in SIDES}, fingers=list(FINGERS),
            palm_channels={side:palm_channels(side) for side in SIDES},
            aggregation='max of 24 channels per finger; max of 240 assigned palm channels; means retained',
            signal=tactile_signal,baseline_model=residual_metadata,
            display='sqrt(positive residual peak/255)' if tactile_signal=='residual' else 'sqrt(peak_count/255)',
            panel_module_sha256=sha(PROJECT/'src/qn_smith/tactile_hand_panel.py'),
            protocol_module_sha256=sha(PROJECT/'src/qn_smith/tactile_protocol.py'),
            palm_region_from_vendor_protocol=True, anatomical_palm_subregions_inferred=False,
            force_calibrated=False, pose_estimated_from_tactile=False,
            silhouette_view='fixed open palms; anatomical left thumb on viewer right; right on viewer left',
            missing_packet='grey silhouette plus no-data label; never held or filled'))
    if prepare_only:
        write_json(dest/'preparation-verification.json',dict(
            source_sha256=report['source_sha256'],frames=count,video_rendered=False,
            purpose='Prepare identical bone, calibrated root and tactile arrays for a separate renderer',
            training_ready=False))
        print(f'{source.stem}: arrays prepared; no video encoded',flush=True)
        return
    w,h = c['width'], c['height']
    if h != 1080: raise ValueError('panel layout is for 1080 high raw RGB; explicit new layout needed')
    fps = (count-1)/(z['sample_time_s'][-1]-z['sample_time_s'][0])
    video = dest/'humanv2_tactile_full.mp4'
    encoder = Encoder(video, (w+640,h), fps)
    qa_rows = set(np.linspace(0,count-1,9).astype(int).tolist())
    thumbs = []
    with h5py.File(source,'r') as f:
        for i,payload in enumerate(f['raw/cameras/head_rgb/payload']):
            rgb = cv2.imdecode(np.asarray(payload,np.uint8), cv2.IMREAD_COLOR)
            for si,side in enumerate(SIDES): draw(rgb,cache[side][i],EDGES,COLORS[si])
            # A narrow overlay title; no crop or rescale of the underlying calibrated image.
            cv2.rectangle(rgb,(0,0),(w,40),(24,27,35),-1)
            put(rgb,f'HumanV2 | {source.stem} | row {i+1}/{count} | {z["sample_time_s"][i]:.2f}s',15,28,.7)
            canvas = np.zeros((h,w+640,3),np.uint8); canvas[:,:w] = rgb
            axis_label = (registration or bridge or {}).get('display_label', 'Camera frame | CAD-axis candidate | no gap fill')
            if tactile_display == 'hands':
                canvas[:,w:] = render_hand_panel(z,i,signal=tactile_signal)
            else:
                tactile_panel(canvas,w,z,i,axis_label)
            encoder.write(canvas)
            if i in qa_rows:
                cv2.imwrite(str(dest/f'qa-row-{i:04d}.jpg'),canvas)
                thumbs.append(cv2.resize(canvas,(768,324)))
            if i % 150 == 0: print(f'{source.stem} render {i}/{count}',flush=True)
    encoder.close()
    cv2.imwrite(str(dest/'contact-sheet.jpg'),np.vstack([np.hstack(thumbs[j:j+3]) for j in range(0,9,3)]))
    verified = verify_video(video,count,w+640,h)
    verified.update(fps=fps, duration_s=count/fps, source_duration_s=float(z['sample_time_s'][-1]-z['sample_time_s'][0]),
                    cfr_sample_time_error_max_ms=float(np.max(abs(np.arange(count)/fps-(z['sample_time_s']-z['sample_time_s'][0])))*1000),
                    sides={s:dict(projected_frames=int(labels[s+'_valid'].sum()),
                        missing_frames=int((~labels[s+'_valid']).sum())) for s in SIDES},
                    training_ready=False, visual_review='pending',
                    tactile_display=tactile_display,
                    tactile_signal=tactile_signal,
                    native_calibrations=calibrations,
                    calibration='source recorded per hand; HumanV2 visual joint accuracy unverified',
                    smoothing=dict(cutoff_hz=5.,beta=.1,derivative_hz=1.,median_bone_lengths=True,
                                   raw_native_retained=True,camera_pose_filtered=False),
                    hardware_sync_verified=False, electrical_axes_verified=False,
                    visual_direction_user_selected=bool((registration or bridge or {}).get('visual_direction_user_selected')),
                    mechanical_registration_sha256=sha(mechanical_registration) if mechanical_registration else None,
                    electrical_bridge_sha256=sha(electrical_bridge) if electrical_bridge else None)
    write_json(dest/'video-verification.json',verified)
    print(json.dumps(verified),flush=True)


def process(job):
    source,out,layout,mount,package,stage,bridge,subdir,registration,tactile_display,tactile_signal,residual_subdir,prepare_only,episode,roi,roi_margin,roi_refresh,right_tag_start,frame_workers = job
    cv2.setNumThreads(2)
    source,out,layout,mount,package = [Path(p).resolve() for p in (source,out,layout,mount,package)]
    # MCAP uses an ephemeral view filename; keep the stable recording/session
    # name in every derived path and provenance instead of leaking the temp
    # basename into the dataset layout.
    out = out/(episode if episode else source.stem)
    out.mkdir(parents=True,exist_ok=True)
    call_out = out/subdir if subdir else out
    call_out.mkdir(parents=True,exist_ok=True)
    write_json(call_out/f'actual-call-{stage}.json',dict(argv=sys.argv,source=str(source),stage=stage,
        pid=os.getpid(),started_unix_s=time.time(),script_sha256=sha(__file__)))
    if stage in ('all','trajectory'): trajectory(source,out,layout,mount,roi=roi,roi_margin=roi_margin,roi_refresh=roi_refresh,right_tag_start=right_tag_start,frame_workers=frame_workers)
    if stage in ('all','native'): native(out,package)
    if stage in ('all','render'): render(source,out,Path(bridge).resolve() if bridge else None,subdir,
                                      Path(registration).resolve() if registration else None,tactile_display,
                                      tactile_signal,residual_subdir,prepare_only)
    return str(call_out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,nargs='+',required=True)
    ap.add_argument('--episode', help='Stable source/session name for MCAP; defaults to source stem')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--layout',type=Path,required=True)
    ap.add_argument('--mount',type=Path,required=True)
    ap.add_argument('--package',type=Path,required=True)
    ap.add_argument('--stage',choices=('all','trajectory','native','render'),default='all')
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--frame-workers',type=int,default=1,
                    help='Independent full-frame AprilTag workers for one recording; ROI mode forces 1')
    ap.add_argument('--roi', action='store_true', help='Use causal ROI search with periodic full-frame reacquisition')
    ap.add_argument('--roi-margin', type=int, default=180)
    ap.add_argument('--roi-refresh', type=int, default=30)
    ap.add_argument('--right-tag-start', type=int, default=13,
                    help='Physical Ball23 ID corresponding to the right-hand local layout ID 0 (legacy HDF5=13; corrected MCAP=15)')
    ap.add_argument('--electrical-bridge',type=Path,help='Source-scoped CAD-to-electrical rotation JSON; never alters baseline')
    ap.add_argument('--mechanical-registration',type=Path,help='Source-bound registered CAD bottom/coil chain; embeds selected electrical axes')
    ap.add_argument('--render-subdir',help='Isolated output folder inside each episode')
    ap.add_argument('--tactile-display',choices=('bars','hands'),default='bars',help='Fixed palm diagrams or legacy bars; data mapping is unchanged')
    ap.add_argument('--tactile-signal',choices=('raw','residual'),default='raw',help='Explicit raw counts or fixed-baseline positive residual; raw default reproduces history')
    ap.add_argument('--tactile-residual-subdir',help='Prepared per-episode residual folder; source/time/raw mapping are verified')
    ap.add_argument('--prepare-only',action='store_true',help='With --stage render, prepare arrays without encoding a redundant baseline video')
    a=ap.parse_args()
    if a.prepare_only and (a.stage!='render' or not a.render_subdir):
        ap.error('--prepare-only requires --stage render and an isolated --render-subdir')
    if a.electrical_bridge and not a.render_subdir:
        ap.error('--electrical-bridge requires --render-subdir')
    if a.mechanical_registration and (a.electrical_bridge or not a.render_subdir):
        ap.error('--mechanical-registration requires --render-subdir and cannot also apply --electrical-bridge')
    if a.render_subdir and (Path(a.render_subdir).name != a.render_subdir or a.render_subdir in ('.','..')):
        ap.error('--render-subdir must be one relative folder name')
    if a.tactile_display == 'hands' and not a.render_subdir:
        ap.error('--tactile-display hands requires an isolated --render-subdir')
    if a.tactile_signal == 'residual' and (a.tactile_display!='hands' or not a.render_subdir or not a.tactile_residual_subdir):
        ap.error('residual signal requires hands, isolated render folder, and --tactile-residual-subdir')
    if a.tactile_residual_subdir and (Path(a.tactile_residual_subdir).name!=a.tactile_residual_subdir or a.tactile_residual_subdir in ('.','..')):
        ap.error('--tactile-residual-subdir must be one relative folder name')
    if a.roi and (a.roi_margin < 1 or a.roi_refresh < 1):
        ap.error('--roi-margin and --roi-refresh must be positive')
    if a.right_tag_start < 0:
        ap.error('--right-tag-start must be non-negative')
    if a.frame_workers < 1:
        ap.error('--frame-workers must be positive')
    jobs=[(s,a.output,a.layout,a.mount,a.package,a.stage,a.electrical_bridge,a.render_subdir,a.mechanical_registration,a.tactile_display,a.tactile_signal,a.tactile_residual_subdir,a.prepare_only,a.episode,a.roi,a.roi_margin,a.roi_refresh,a.right_tag_start,a.frame_workers) for s in a.source]
    if a.workers == 1:
        for j in jobs: print(process(j),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for result in pool.map(process,jobs): print(result,flush=True)


if __name__ == '__main__': main()
