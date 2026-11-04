#!/usr/bin/env python3
"""Use fixed EgoSmith detector/HaWoR/MANO nodes without a second SLAM frame."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

from qn_smith.artifacts import manifest, now, sha256, verify_manifest, write_json
from qn_smith.repositories import repository


def resolve_paths(args):
    # EgoSmith uses a repository cwd below; freeze all caller-relative paths
    # before that change, including optional crops/cache/accelerator files.
    for name in ('input', 'output', 'checkpoint', 'detection_cache', 'sam_crops', 'openvino_model'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    return args


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--backbone-backend', choices=('torch','openvino'), default='torch')
    p.add_argument('--openvino-model', type=Path)
    p.add_argument('--openvino-device', choices=('CPU','GPU'), default='CPU')
    p.add_argument('--chunk-batch-size', type=int, default=2)
    p.add_argument('--detection-cache', type=Path)
    p.add_argument('--sam-crops', type=Path)
    p.add_argument('--assisted-crops-only', action='store_true',
                   help='Use only explicitly supplied crop tracks; skip official detection')
    p.add_argument('--image-extension', choices=('png', 'jpg'), default='png',
                   help='Explicit prepared image encoding; default preserves historical inputs')
    args = p.parse_args()
    args = resolve_paths(args)
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]; upstream = repository('egosmith', root)
    os.chdir(upstream); sys.path[:0] = [str(upstream/'src'), str(upstream)]
    # Import the public standalone loader (TORCH_COMPILE=0 in model config), then
    # call EgoSmith's original inference routine. Do not patch dependency sources.
    from hawor_loader import load_hawor
    from lib.pipeline.io.frame_source import ImageFolderFrameSource
    from lib.pipeline.hands.hawor_inference import batched_hawor_inference
    from lib.pipeline.hands.mano_runtime import run_mano, run_mano_left, get_mano_faces
    from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis

    info = json.loads((args.input/'input.json').read_text()); n = info['rows']; K = np.asarray(info['K'])
    records = json.loads((args.input/'manifest.json').read_text())
    if verify_manifest(args.input, records): raise ValueError('Input manifest mismatch')
    source = ImageFolderFrameSource([args.input/'images'/f'{i:06d}.{args.image_extension}' for i in range(n)], use_turbojpeg=False)
    if source.get_size() != (info['height'], info['width']): raise ValueError('Input image grid mismatch')
    device=os.environ.get('QN_EGO_DEVICE','cuda')
    torch.set_num_threads(int(os.environ.get('QN_EGO_THREADS','4'))); cv2.setNumThreads(2); start = time.time()
    write_json(args.output/'invocation.json', dict(at=now(), argv=sys.argv, input_info_sha256=sha256(args.input/'input.json'),
        input_manifest_sha256=sha256(args.input/'manifest.json'), script_sha256=sha256(__file__),
        checkpoint_sha256=sha256(args.checkpoint), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        upstream_files={str(p.relative_to(upstream)): sha256(p) for p in [
            upstream/'src/lib/stage_runners/hawor_video.py', upstream/'src/lib/pipeline/hands/hawor_inference.py',
            upstream/'src/lib/pipeline/hands/detect_track_batched.py', upstream/'src/lib/pipeline/hands/mano_runtime.py']},
        purpose='Independent camera-space MANO evidence; not full EgoSmith SLAM/infiller/export'))
    if args.assisted_crops_only and not args.sam_crops:
        raise ValueError('--assisted-crops-only requires --sam-crops')
    if args.assisted_crops_only:
        boxes, tracks = [], {}
    elif args.detection_cache:
        boxes = np.load(args.detection_cache/'model-boxes.npy', allow_pickle=True)
        tracks = np.load(args.detection_cache/'model-tracks.npy', allow_pickle=True)
    else:
        from lib.pipeline.hands.detect_track_batched import detect_track
        boxes, tracks = detect_track(source, thresh=.35, edge_margin_ratio=.1, min_edge_conf=.4,
            detect_batch_size=24, num_io_workers=2, device=device, half_precision=device=='cuda')
    if isinstance(tracks, np.ndarray) and tracks.shape == (): tracks = tracks.item()
    if not isinstance(tracks, dict): raise ValueError('Unknown official detector track container')
    np.save(args.output/'model-boxes.npy', boxes)
    np.save(args.output/'model-tracks.npy', tracks)
    write_json(args.output/'detector-summary.json', dict(tracks={str(k):len(v) for k,v in tracks.items()},
        elapsed_s=time.time()-start, note='Detector anatomical handedness is a model estimate, not certified identity'))
    if args.sam_crops:
        with np.load(args.sam_crops, allow_pickle=False) as c:
            if str(c['input_sha256']) != info['input_sha256']: raise ValueError('SAM crop source mismatch')
            # Separate assisted experiment: SAM masks + frozen native side prior
            # provide crops. These are not claimed as official YOLO detections.
            for si, side in enumerate(('left','right')):
                track = []
                for row in np.flatnonzero(c[side+'_valid']):
                    track.append(dict(frame=int(row), det=True, det_box=c[side+'_boxes'][row][None],
                        det_handedness=np.array([si]), source='sam_native_associated_crop'))
                tracks[1000+si] = track
        write_json(args.output/'sam-crop-source.json',dict(path=str(args.sam_crops),sha256=sha256(args.sam_crops),
            semantics=('Explicit assisted crop tracks; inspect crop provenance for anatomical identity; '
                       'not official YOLO detections or certified hand identity')))
    model, cfg = load_hawor(str(args.checkpoint)); model = model.to(device).eval()
    accelerator = None
    if args.backbone_backend == 'openvino':
        if device != 'cpu' or args.openvino_model is None:
            raise ValueError('OpenVINO requires --device cpu and an explicitly validated model')
        from qn_ego.openvino_backbone import OpenVINOBackbone
        import gc
        del model.backbone; gc.collect()
        model.backbone = OpenVINOBackbone(args.openvino_model, args.checkpoint,
            args.openvino_device, int(os.environ.get('QN_EGO_THREADS','4')))
        accelerator = dict(model=str(args.openvino_model.resolve()), device=args.openvino_device,
            validation_sha256=sha256(args.openvino_model.with_name('validation.json')))
    candidates = []
    for tid, track in tracks.items():
        detected = [v for v in track if v['det']]
        if len(detected) < 5: continue
        frames = np.array([int(v['frame']) for v in detected])
        bboxes = np.concatenate([v['det_box'] for v in detected])
        handedness = np.concatenate([v['det_handedness'] for v in detected]).reshape(-1)
        right = bool(np.mean(handedness) >= .5)
        good = np.isfinite(bboxes).all(1) & (bboxes[:, 2]>bboxes[:, 0]) & (bboxes[:, 3]>bboxes[:, 1])
        frames, bboxes, handedness = frames[good], bboxes[good], handedness[good]
        if len(frames) < 5: continue
        # No interpolation across missing detections. Split original row gaps so
        # the temporal model does not infer adjacency across absent source rows.
        cuts = np.r_[0, np.flatnonzero(np.diff(frames) != 1)+1, len(frames)]
        result = batched_hawor_inference(model, source, frames, bboxes,
            img_focal=float(K[0, 0]), img_center=K[:2, 2].tolist(), device=device,
            do_flip=not right, chunk_batch_size=args.chunk_batch_size, num_workers=2,
            output_device='cpu', chunk_boundaries=cuts.tolist())
        rot = result['pred_rotmat'].clone(); aa = rotation_matrix_to_angle_axis(rot)
        if not right: aa[..., 1:] *= -1
        rot = angle_axis_to_rotation_matrix(aa)
        trans = result['pred_trans'][:, 0]
        all_vertices, all_joints = [], []
        for first in range(0, len(frames), 64):
            last = first+64
            kwargs = dict(trans=trans[None, first:last], root_orient=aa[None, first:last, 0],
                hand_pose=aa[None, first:last, 1:].reshape(1, -1, 45), betas=result['pred_shape'][None, first:last])
            output = (run_mano if right else run_mano_left)(**kwargs,use_cuda=device=='cuda')
            all_vertices.append(output['vertices'][0].cpu().numpy())
            all_joints.append(output['joints'][0].cpu().numpy())
        path = args.output/f'track-{int(tid):03d}.npz'
        np.savez_compressed(path, source_rows=frames, detector_boxes=bboxes, detector_handedness=handedness,
            vertices_camera_m=np.concatenate(all_vertices), joints_camera_m=np.concatenate(all_joints),
            rotations=rot.numpy(), betas=result['pred_shape'].numpy(), translation=trans.numpy(),
            K=K, side=np.array('right' if right else 'left'))
        candidates.append(dict(track_id=int(tid), file=path.name, side='right' if right else 'left', rows=frames.tolist(),
            crop_source=('explicit_assisted_crop' if args.assisted_crops_only else 'sam_native_associated_crop')
                        if int(tid)>=1000 else 'official_detector'))
        print(info['episode'], 'TRACK', tid, 'right' if right else 'left', len(frames), flush=True)
    faces = get_mano_faces(); np.savez_compressed(args.output/'faces.npz', right=faces, left=faces[:, [0, 2, 1]])
    write_json(args.output/'summary.json', dict(schema='qn_smith.visual_mano.v1', at=now(), episode=info['episode'],
        input_sha256=info['input_sha256'], rows=n, K=K, candidates=candidates, elapsed_s=time.time()-start,
        camera_space='rectified color optical, +Z forward, metric MANO model estimate',
        joint_order='21 OpenPose joints: wrist, thumb,index,middle,ring,little; four nodes each',
        detection_gap_handling='unknown, no interpolation or infiller',
        device=device,backbone_backend=args.backbone_backend,accelerator=accelerator,
        physical_calibration_verified=False, visual_alignment_pass=False, robot_action_training=False))
    m = manifest(args.output); write_json(args.output/'manifest.json', m)
    if verify_manifest(args.output, m): raise ValueError('Output manifest mismatch')
    write_json(args.output/'verification.json', dict(at=now(), files=len(m), manifest_readback=True))


if __name__ == '__main__': main()
