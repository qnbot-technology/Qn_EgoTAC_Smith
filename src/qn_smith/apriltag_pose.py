"""Source-bound AprilTag optical poses; no implicit world, mount or gap filling.

Conventions follow HDF5_Collector's pinned qnbot_nav offline bridge. Detection
uses portable OpenCV PnP, not the upstream temporal filter. Its JSONL can also
be imported, retaining current-observation gating and original timestamps.
"""
from collections import Counter
import json
from pathlib import Path
import threading

import cv2
import numpy as np

from .artifacts import manifest, now, sha256, write_json
from .input_identity import file_record, resolve_path
from .geometry import se3


# Detector construction is surprisingly expensive on the CPU.  Keep one
# immutable detector per worker thread.  The detector has no frame state; a
# thread-local instance avoids relying on undocumented cross-thread safety
# while still removing one dictionary/parameter allocation per RGB frame.
_DETECTOR_LOCAL = threading.local()


def _apriltag_detector():
    detector = getattr(_DETECTOR_LOCAL, 'detector', None)
    if detector is None:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        params.aprilTagQuadDecimate = 1.
        params.aprilTagQuadSigma = 0.
        params.aprilTagMinWhiteBlackDiff = 5
        params.aprilTagMinClusterPixels = 5
        params.aprilTagCriticalRad = .05
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        _DETECTOR_LOCAL.detector = detector
    return detector


def settings(config, episode):
    cfg = dict(config.get('apriltag_pose', {}))
    overrides = cfg.pop('episodes', {})
    if overrides and episode not in overrides:
        raise ValueError('apriltag_pose: missing episode configuration')
    cfg.update(overrides.get(episode, {}))
    return cfg


def camera_contract(c):
    if c.get('camera_model', 'opencv_brown') != 'opencv_brown':
        raise ValueError('Only explicit OpenCV Brown pinhole is supported; KB/fisheye needs a separate adapter')
    K, D = np.asarray(c['K'], float), np.asarray(c.get('distortion', [0.] * 5), float)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1])+abs(K[1, 0]) > 1e-9
            or D.shape not in ((4,), (5,), (8,)) or not np.isfinite(D).all()
            or int(c['width']) <= 0 or int(c['height']) <= 0):
        raise ValueError('Invalid calibrated image geometry or Brown distortion')
    return K, D


def tag_points(size):
    if not np.isfinite(size) or size <= 0:
        raise ValueError('Positive measured tag black-square edge in metres required')
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def solve_pose(points, pixels, K, D, *, square=False, initial_poses=()):
    """Enumerate planar solutions, keep positive-depth minimum reprojection.

    Ambiguity is reported, never treated as metric accuracy certification.
    Iterative fallback covers exactly frontal IPPE degeneracy in some OpenCVs.
    """
    points, pixels = np.asarray(points, float), np.asarray(pixels, float)
    candidates = []
    for pose in initial_poses:
        pose=se3(pose)
        try:
            ok,r,t=cv2.solvePnP(points,pixels,K,D,cv2.Rodrigues(pose[:3,:3])[0],
                pose[:3,3].copy().reshape(3,1),True,flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:candidates.append((r,t))
        except cv2.error:
            pass
    flag = cv2.SOLVEPNP_IPPE_SQUARE if square else cv2.SOLVEPNP_SQPNP
    try:
        ok, rs, ts, *_ = cv2.solvePnPGeneric(points, pixels, K, D, flags=flag)
        if ok:
            candidates.extend(zip(rs, ts))
    except cv2.error:
        pass
    try:
        ok, r, t = cv2.solvePnP(points, pixels, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            candidates.append((r, t))
    except cv2.error:
        pass
    solutions = []
    for r, t in candidates:
        T = np.eye(4); T[:3, :3] = cv2.Rodrigues(r)[0]; T[:3, 3] = np.asarray(t).ravel()
        z = (points @ T[:3, :3].T + T[:3, 3])[:, 2]
        if not np.isfinite(T).all() or (z <= 0).any():
            continue
        projected = cv2.projectPoints(points, r, t, K, D)[0].reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1))))
        if not np.isfinite(error):
            continue
        if not any(np.allclose(T, old[1], atol=1e-7) for old in solutions):
            solutions.append((error, T))
    solutions.sort(key=lambda v: v[0])
    if not solutions:
        return None, None, None
    return solutions[0][1], solutions[0][0], (solutions[1][0]-solutions[0][0] if len(solutions)>1 else None)


def detect_frame(image, calibration, sizes, *, max_reprojection_rms_px=3., min_edge_px=12.):
    K, D = camera_contract(calibration)
    if image is None or image.dtype != np.uint8 or image.shape != (calibration['height'], calibration['width'], 3):
        raise ValueError('Frame does not match calibrated BGR8 grid; resizing is forbidden')
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError('AprilTag requires OpenCV aruco; use an isolated environment with that module')
    corners, ids, _ = _apriltag_detector().detectMarkers(image)
    ids = [] if ids is None else ids.ravel().tolist()
    counts = Counter(ids); detections, states = [], []
    for tag, c in zip(ids, corners):
        c = c.reshape(4, 2); detections.append({'id': tag, 'corners_px': c.tolist()})
        if str(tag) not in sizes:
            continue
        T, error, gap = solve_pose(tag_points(float(sizes[str(tag)])), c, K, D, square=True)
        edge = float(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1).min())
        reasons = []
        if counts[tag] != 1: reasons.append('duplicate_id')
        if edge < min_edge_px: reasons.append('tag_too_small_in_pixels')
        if T is None: reasons.append('pnp_failed_or_behind_camera')
        elif error > max_reprojection_rms_px: reasons.append('reprojection_high')
        states.append(dict(id=tag, T_camera_tag=T.tolist() if T is not None else None,
            reprojection_rms_px=error, alternative_solution_error_gap_px=gap,
            shortest_edge_px=edge, valid_current_observation=not reasons,
            updated=True, age_s=0., invalid_reasons=reasons))
    return dict(detections=detections, states=states, duplicate_ids=[k for k,v in counts.items() if v>1],
                coordinate_frame='camera_color_optical', input_rectified=False)


def body_poses(row, bodies, sizes, calibration):
    """Joint corner PnP in a declared body/EMF frame, not tag-pose averaging."""
    K, D = camera_contract(calibration)
    if row.get('input_rectified'):
        D = np.zeros(5)  # Upstream rectifies into the same K, without resizing.
    counts = Counter(d['id'] for d in row['detections'])
    fresh = {s['id'] for s in row['states'] if s.get('valid_current_observation')
             and s.get('updated') and abs(float(s.get('age_s', 1))) <= 1e-8 and counts[s['id']]==1}
    result = {}
    for name, body in bodies.items():
        if not body.get('extrinsic_evidence') or body.get('root_frame') not in (
                'emf_glove', 'asset', 'fixture_bottom_emf_axes_candidate'):
            raise ValueError('Explicit body root frame and mount evidence required')
        points, pixels, visible, seeds = [], [], [], []
        for detection in row['detections']:
            key = str(detection['id'])
            if detection['id'] not in fresh or key not in body['T_root_tag']:
                continue
            T = se3(body['T_root_tag'][key]); p = tag_points(float(sizes[key]))
            points.extend(p @ T[:3, :3].T + T[:3, 3])
            pixels.extend(detection['corners_px']); visible.append(int(key))
            state=next(s for s in row['states'] if s['id']==detection['id'])
            if state.get('T_camera_tag') is not None:
                seeds.append(se3(state['T_camera_tag'])@np.linalg.inv(T))
        used=list(visible);excluded=[]
        points,pixels=np.asarray(points),np.asarray(pixels)
        if body.get('consensus_outlier_rejection') and len(visible)>=3 and seeds:
            choices=[]
            for seed in seeds:
                projected=cv2.projectPoints(points,cv2.Rodrigues(seed[:3,:3])[0],seed[:3,3],K,D)[0].reshape(-1,2)
                error=np.sqrt(np.mean(np.sum((projected-pixels)**2,axis=1).reshape(-1,4),axis=1))
                keep=error<=body.get('consensus_tag_rms_px',10.)
                if keep.sum()>=2 and keep.mean()>=body.get('min_consensus_fraction',.6):
                    choices.append((int(keep.sum()),-float(np.median(error[keep])),keep))
            if choices:
                _,_,keep=max(choices,key=lambda x:x[:2])
                used=[v for v,k in zip(visible,keep) if k]
                excluded=[v for v,k in zip(visible,keep) if not k]
                mask=np.repeat(keep,4)
            else:mask=np.ones(len(points),bool)
        else:mask=np.ones(len(points),bool)
        T, rms, gap = solve_pose(points[mask], pixels[mask], K, D, square=False,initial_poses=seeds) if visible else (None,None,None)
        valid = T is not None and rms <= body.get('max_reprojection_rms_px', 3.)
        residuals = {}
        if T is not None:
            predicted = cv2.projectPoints(np.asarray(points), cv2.Rodrigues(T[:3,:3])[0], T[:3,3], K, D)[0].reshape(-1,2)
            errors = np.linalg.norm(predicted-np.asarray(pixels),axis=1).reshape(-1,4)
            residuals={str(tag):float(np.sqrt(np.mean(e**2))) for tag,e in zip(visible,errors)}
            valid = valid and max(residuals[str(tag)] for tag in used) <= body.get('max_tag_reprojection_rms_px', 5.)
        result[name] = dict(root_frame=body['root_frame'], visible_ids=visible,
            used_ids=used,excluded_ids=excluded,
            T_camera_root=T.tolist() if valid else None, reprojection_rms_px=rms,
            alternative_solution_error_gap_px=gap,
            per_tag_reprojection_rms_px=residuals,
            suspicious_ids=[int(k) for k,v in residuals.items() if v>body.get('max_tag_reprojection_rms_px',5.)],
            valid_current_observation=valid,
            invalid_reason=None if valid else 'no_fresh_bound_tags_or_joint_pnp_rejected')
    return result


def capture_rows(capture):
    rows = [json.loads(s) for s in (capture/'frames.jsonl').read_text().splitlines()]
    if not rows: raise ValueError('Empty capture frame map')
    first = rows[0]['device_us']; previous = -1.
    for i, row in enumerate(rows):
        stamp = (row['device_us']-first)/1e6
        if row['index'] != i or not np.isfinite(stamp) or stamp <= previous:
            raise ValueError('Capture index/time must be dense and strictly increasing')
        previous = stamp
        path = (capture/row['file']).resolve()
        if not path.is_relative_to(capture.resolve()): raise ValueError('Image outside capture')
        yield dict(frame_index=i, source_row=i, time_s=stamp, source=row, image=path)


def asset_observations(report, rows, packet, frame_map, frames, cameras, bindings):
    """Optional camera/body -> world/canonical-asset bridge with exact provenance.

    Hand/EMF roots are deliberately excluded from the rigid-object asset route.
    Camera poses must use exactly the same source time base and frame map.
    """
    digest = packet['input']['source']['sha256']
    if (report['input_source_sha256']!=digest or report['episode_id']!=packet['episode_id']
            or cameras.get('schema')!='qn_smith.apriltag_camera_poses.v1'
            or cameras.get('input_source_sha256')!=digest
            or cameras.get('frame_map_sha256')!=frame_map['sha256']
            or cameras.get('episode_id')!=packet['episode_id']):
        raise ValueError('Tag/camera/source episode or frame-map mismatch')
    if len(cameras['frames'])!=len(frames):raise ValueError('Camera poses need the full source timeline')
    byrow={(r['frame_index'],r['source_row']):r for r in rows}
    if len(byrow)!=len(rows):raise ValueError('Duplicate AprilTag source rows')
    result=dict(schema='qn_smith.object_pose_observations.v1',episode_id=packet['episode_id'],
        input_source_sha256=digest,frame_map_sha256=frame_map['sha256'],world=cameras['world'],
        instances={},certified_metric_labels=False,rl_ready=False)
    for name,binding in bindings.items():
        body=report['body_bindings'][binding['body_id']]
        if body['root_frame']!='asset':raise ValueError('EMF hand root is not a rigid object asset')
        if not binding.get('canonical_registration_evidence') or not binding.get('scale_evidence'):
            raise ValueError('Canonical registration and metric scale evidence required')
        bridge=se3(binding['T_root_asset']);out=[]
        for f,c in zip(frames,cameras['frames']):
            if any(c[k]!=f[k] for k in ('frame_index','source_row','time_s')):
                raise ValueError('Camera pose frame/time mismatch')
            r=byrow.get((f['frame_index'],f['source_row']));b=r['bodies'][binding['body_id']] if r else None
            if r and abs(r['time_s']-f['time_s'])>1e-9:raise ValueError('Tag source time mismatch')
            valid=b and b['valid_current_observation'] and c.get('T_W_C') is not None
            pose=se3(c['T_W_C'])@se3(b['T_camera_root'])@bridge if valid else None
            out.append(dict(frame_index=f['frame_index'],source_row=f['source_row'],time_s=f['time_s'],
                T_W_A=pose.tolist() if valid else None,provenance='estimated',
                invalid_reason=None if valid else 'tag_unobserved_or_camera_world_missing',
                confidence=float(1/(1+b['reprojection_rms_px'])) if valid else None,
                evidence=dict(tag_body=b,body_mount=body['extrinsic_evidence'],
                    camera_pose=c,canonical_bridge=binding['canonical_registration_evidence'],
                    confidence_semantics='uncalibrated inverse reprojection diagnostic score') if valid else None))
        result['instances'][name]={k:binding[k] for k in ('asset_id','canonical_asset_sha256',
            'asset_metric_scale','scale_evidence','canonical_registration_evidence')}
        result['instances'][name]['frames']=out
    return result


def resource_identity(config, project):
    result = {}
    base = config.get('apriltag_pose', {})
    for ep in list(base.get('episodes', {})) or ['default']:
        c = settings(config, ep)
        if c.get('mode') == 'detect_episode':
            path=resolve_path(c['calibration'],project)
            result[ep]={str(path):sha256(path)}
            continue
        capture = resolve_path(c['capture_dir'], project)
        paths = [capture/'frames.jsonl', resolve_path(c.get('calibration', str(capture/'calibration.json')), project)]
        if c.get('poses_jsonl'): paths.append(resolve_path(c['poses_jsonl'], project))
        if c.get('mode')=='import_capture':
            poses=resolve_path(c['poses_jsonl'],project)
            paths.append(resolve_path(c.get('import_summary',str(poses.parent/'summary.json')),project))
        for body in c.get('bodies',{}).values():
            if body.get('extrinsic_evidence'):
                evidence=resolve_path(body['extrinsic_evidence'],project)
                if evidence.is_file():paths.append(evidence)
        paths.extend(r['image'] for r in capture_rows(capture))
        result[ep] = {str(p): sha256(p) for p in paths}
    return result


def apriltag_pose(context, output):
    cfg = settings(context['config'], context['episode_root'].name)
    project, output = context['project'], Path(output)
    mode = cfg.get('mode', 'detect_capture')
    quality = {}
    if mode == 'detect_episode':
        import pyarrow.parquet as pq
        observed = context['episode_root']/'observed_rgb'
        frame_file = observed/'model_frames.parquet'
        rows = pq.read_table(frame_file).to_pylist()
        quality = {r['source_row']:r for r in pq.read_table(
            context['episode_root']/'quality_gate/row_decisions.parquet').to_pylist()}
        source_rows = [dict(frame_index=r['model_frame_index'],source_row=r['source_row'],
            time_s=r['timestamp_s'],source=r,image=observed/r['image_path']) for r in rows]
        calibration_path = resolve_path(cfg['calibration'],project)
    else:
        capture = resolve_path(cfg['capture_dir'], project)
        frame_file = capture/'frames.jsonl'
        if context['input'].resolve() != frame_file.resolve() or sha256(frame_file) != context['input_sha256']:
            raise ValueError('Capture input must be its exact source frames.jsonl; use a new run per capture')
        source_rows = list(capture_rows(capture))
        calibration_path = resolve_path(cfg.get('calibration', str(capture/'calibration.json')), project)
    calibration = json.loads(calibration_path.read_text()); camera_contract(calibration)
    sizes = {str(k): float(v) for k,v in cfg['tag_sizes_m'].items()}
    if not sizes or any(not k.isdigit() or not 0<=int(k)<=586 for k in sizes):
        raise ValueError('Explicit tag36h11 IDs/sizes required')
    for size in sizes.values(): tag_points(size)
    bodies = cfg.get('bodies', {})
    owners = [k for b in bodies.values() for k in b['T_root_tag']]
    if len(owners)!=len(set(owners)) or not set(owners)<=set(sizes):
        raise ValueError('Each bound ID needs a size and belongs to exactly one body')
    for body in bodies.values():
        if not body.get('extrinsic_evidence') or not body['T_root_tag']:
            raise ValueError('Nonempty tag layout with mount provenance required')
        for transform in body['T_root_tag'].values():se3(transform)
        if body.get('max_reprojection_rms_px',3.)<=0 or body.get('max_tag_reprojection_rms_px',5.)<=0:
            raise ValueError('Positive body reprojection limits required')
    imported = None
    import_summary = None
    if mode == 'import_capture':
        pose_path=resolve_path(cfg['poses_jsonl'],project)
        summary_path=resolve_path(cfg.get('import_summary',str(pose_path.parent/'summary.json')),project)
        import_summary=json.loads(summary_path.read_text())
        if (Path(import_summary['source']).resolve()!=capture.resolve()
                or import_summary['calibration']!=calibration
                or any(abs(v-float(import_summary['marker_size_m']))>1e-12 for v in sizes.values())
                or import_summary['frames']!=len(source_rows)
                or import_summary['coordinate_frame']!='camera_color_optical'):
            raise ValueError('Collector import source/calibration/tag size/frame count mismatch')
        imported = [json.loads(s) for s in pose_path.read_text().splitlines()]
    elif mode not in ('detect_capture','detect_episode'):
        raise ValueError('mode must be detect_capture, detect_episode or import_capture')
    if imported is not None and len(imported) != len(source_rows): raise ValueError('Full capture timeline required')
    rows = []; images = []
    previous = -float('inf')
    for index, source in enumerate(source_rows):
        if source['frame_index']!=index or not np.isfinite(source['time_s']) or source['time_s']<=previous:
            raise ValueError('Strictly increasing source exposure timeline required')
        previous=source['time_s']
        if imported is None:
            row = detect_frame(cv2.imread(str(source['image'])), calibration, sizes,
                **{k:cfg[k] for k in ('max_reprojection_rms_px','min_edge_px') if k in cfg})
        else:
            row = dict(imported[index])
            if (row['frame_index'] != index or abs(row['timestamp_s']-source['time_s'])>1e-8
                    or row['source'] != source['source'] or row['coordinate_frame'] != 'camera_color_optical'):
                raise ValueError('Imported pose source frame/time/camera mismatch')
            counts = Counter(d['id'] for d in row['detections'])
            for state in row['states']:
                rms=state.get('reprojection_rms_px')
                state['valid_current_observation'] = bool(state.get('valid_current_observation')
                    and state.get('updated') and abs(float(state.get('age_s',1)))<=1e-8
                    and counts[state['id']]==1 and str(state['id']) in sizes
                    and isinstance(rms,(float,int)) and np.isfinite(rms)
                    and 0 <= rms <= cfg.get('max_reprojection_rms_px',3.))
                if state.get('T_camera_tag') is not None: se3(state['T_camera_tag'])
            if bool(row.get('input_rectified'))!=bool(import_summary['rectified_for_tracking']):
                raise ValueError('Imported rectification contract mismatch')
        row.update(frame_index=index, source_row=source['source_row'], time_s=source['time_s'],
                   timestamp_s=source['time_s'], source=source['source'])
        if mode=='detect_episode':
            q=quality[source['source_row']]
            if abs(q['timestamp_s']-source['time_s'])>1e-9:raise ValueError('Loader quality/source time mismatch')
            row['source_quality']={k:q[k] for k in ('default_loader_row_eligible','decision','reason_codes')}
            # Optical tag diagnostics can still exist when unrelated VIVE/depth
            # fails. Preserve that distinction; downstream loader decides usage.
        row['bodies'] = body_poses(row, bodies, sizes, calibration)
        rows.append(row); images.append(dict(frame_index=index, **file_record(source['image'])))
    with (output/'poses.jsonl').open('w') as stream:
        for row in rows: stream.write(json.dumps(row,allow_nan=False)+ '\n')
    valid = sum(any(s.get('valid_current_observation') for s in r['states']) for r in rows)
    result = dict(schema='qn_smith.apriltag_pose.v1', created_at=now(),
        episode_id=context['episode_root'].name, input_source_sha256=context['input_sha256'],
        mode=mode, dictionary='DICT_APRILTAG_36h11', tag_sizes_m=sizes,
        coordinate_contract='T_camera_tag maps tag into color optical (right,down,forward); metres',
        calibration=calibration, calibration_evidence=file_record(calibration_path),
        source_frame_map=file_record(frame_file), frame_count=len(rows), current_pose_frames=valid,
        body_current_frames={n:sum(r['bodies'][n]['valid_current_observation'] for r in rows) for n in bodies},
        metric_calibration_status='estimated' if calibration.get('estimated') else 'sdk_or_declared_not_accuracy_certified',
        backend='opencv_pnp_no_temporal_filter' if imported is None else 'collector_filtered_pose_import',
        opencv_version=cv2.__version__, body_bindings=bodies,
        missing_pose_fill='none', world_pose_available=False, training_ready=False,
        caveats=['planar PnP ambiguity remains','body extrinsics need ID/printed-roll registration',
                 'camera-relative motion is not world trajectory','not physical contact ground truth'])
    write_json(output/'report.json', result); write_json(output/'source-images.json', images)
    write_json(output/'provenance.json',dict(input=file_record(frame_file),
        imported_pose=file_record(resolve_path(cfg['poses_jsonl'], project)) if imported is not None else None,
        import_summary=import_summary,
        mount_evidence=bodies, actual_backend=result['backend']))
    write_json(output/'manifest.json',manifest(output,exclude=('manifest.json',)))
    return {k:result[k] for k in ('frame_count','current_pose_frames','body_current_frames','world_pose_available','training_ready')}
