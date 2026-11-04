"""AprilTag pause receipt: invalid pose samples are NOT dropped RGB frames."""
from pathlib import Path

import numpy as np


def count_tag_samples(arrays, start=0, stop=None):
    """Count fresh, accepted Tag PnP poses separately from EMF time validity.

    Counts use HDF5 RGB-aligned sample rows (not unique camera frames). Repeated
    camera indices are reported separately; no sensor acquisition loss is inferred.
    """
    left = np.asarray(arrays['left_pose_valid'], dtype=bool)
    right = np.asarray(arrays['right_pose_valid'], dtype=bool)
    if left.ndim != 1 or right.shape != left.shape:
        raise ValueError('Tag validity arrays must have the same 1D shape')
    stop = len(left) if stop is None else stop
    if not 0 <= start < stop <= len(left):
        raise ValueError('Invalid Tag summary range')
    l, r = left[start:stop], right[start:stop]
    camera = np.asarray(arrays['camera_frame_index'])[start:stop]
    if camera.shape != l.shape:
        raise ValueError('Camera indices do not match Tag timeline')
    sides = {}
    for side, valid in [('left', l), ('right', r)]:
        emf = np.asarray(arrays[side+'_emf_time_valid'], dtype=bool)
        if emf.shape != left.shape:
            raise ValueError('EMF validity does not match Tag timeline')
        sides[side] = dict(tag_valid=int(valid.sum()), tag_missing=int((~valid).sum()),
                           emf_time_valid=int(emf[start:stop].sum()))
    return dict(frames=stop-start, first_source_row_0based=start,
                last_source_row_0based=stop-1, sides=sides,
                any_hand_tag_missing=int((~l | ~r).sum()),
                both_hands_tag_missing=int((~l & ~r).sum()),
                repeated_camera_samples=int((np.diff(camera) == 0).sum()))


def artifact_paths(output, episode):
    """Absolute, predictable paths; visual paths are planned until completion."""
    run = Path(output).resolve()
    base = run/'baseline'/episode
    return {name: str(path) for name, path in {
        'tag_checkpoint': run/'apriltag-checkpoint.json',
        'tag_trajectory': base/'trajectory-and-tactile.npz',
        'tag_report': base/'trajectory-report.json',
        'tag_detections': base/'detections-and-poses.jsonl',
        'frame_map': base/'frame-map.csv',
        'camera': base/'camera.json',
        'left_local_skeleton': base/'native-left/native-output.npz',
        'right_local_skeleton': base/'native-right/native-output.npz',
        'tactile_residual': base/'tactile-baseline-v1/frame-residual.npz',
        'sam_crops': run/'visual/sam'/episode/'sam-hand-crops.npz',
        'mano_summary': run/'visual/sam-mano'/episode/'summary.json',
        'fused_skeleton': run/'visual/fused-stable-v3'/episode/'fused-humanv2.npz',
        'fusion_report': run/'visual/fused-stable-v3'/episode/'fusion-report.json',
        'tag_video': run/'final-display/delivery/01_tag_only_final_16x9.mp4',
        'assisted_video': run/'final-display/delivery/02_visual_assisted_final_16x9.mp4',
        'result': run/'result.json', 'verification': run/'verification.json',
        'calls': run/'calls', 'request': run/'request.json',
    }.items()}


def tag_checkpoint(output, episode, source_identity, start, stop):
    from .pipeline import path_record
    paths = artifact_paths(output, episode)
    if path_record(source_identity['path']) != source_identity:
        raise RuntimeError('Source changed during Tag detection')
    with np.load(paths['tag_trajectory'], allow_pickle=False) as arrays:
        full = count_tag_samples(arrays)
        selected = count_tag_samples(arrays, start, stop)
    return dict(schema='qn_ego.apriltag_checkpoint.v1', completed=False,
                stage='apriltag_completed', full_visual_pipeline=False,
                training_ready=False, input=source_identity,
                count_unit='RGB-aligned HDF5 sample row; not acquisition frame loss',
                tag_missing_definition='No fresh accepted Tag PnP pose; independent of EMF validity',
                full_timeline=full, selected_video_range=selected,
                camera_frame='original RGB optical (+X right,+Y down,+Z forward), metres; NOT fixed world',
                trajectory=path_record(paths['tag_trajectory']), artifacts=paths,
                note='Checkpoint snapshot only; visual artifacts are planned, not yet generated')
