#!/usr/bin/env python3
"""Render adapter for validated Tag/HumanV2 arrays; NO visual localization."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from qn_smith.artifacts import write_json, now, sha256


def prepare(run, episode):
    base=run/'baseline'/episode
    prior=base/'tag3-tag16-handmap-residual-v4'
    out=run/'mapped/tag-only'/episode
    out.mkdir(parents=True,exist_ok=False)
    inputs=[base/'trajectory-and-tactile.npz',prior/'bone-projections.npz',prior/'applied-camera-trajectory.npz']
    data={};sides={}
    with np.load(inputs[0]) as z, np.load(inputs[1]) as b, np.load(inputs[2]) as applied:
        for key in ('source_row','sample_time_s','camera_receive_ns'):data[key]=z[key]
        for side in ('left','right'):
            valid=b[side+'_valid'].astype(bool)
            points=b[side+'_filtered_camera_m'].copy()
            roots=applied[side+'_T_camera_emf'].copy()
            points[~valid]=np.nan;roots[~valid]=np.nan
            data.update({side+'_valid':valid,side+'_camera_m':points,
                         side+'_T_camera_emf':roots,side+'_display_T_camera_emf':roots.copy(),
                         side+'_tag_observed_valid':z[side+'_pose_valid'],
                         side+'_source':np.where(valid,'TAG','MISSING_TAG_OR_EMF_OR_NATIVE')})
            sides[side]=dict(valid=int(valid.sum()),missing=int((~valid).sum()),added_by_visual=0)
    # Compatibility name for the common renderer, explicitly a Tag-only adapter.
    np.savez_compressed(out/'fused-humanv2.npz',**data)
    report=dict(at=now(),processing_mode='tag-only',visual_assist_enabled=False,
                no_visual_candidates=True,native_local_fingers_unchanged=True,
                root_smoothing_added=False,sides=sides,training_ready=False,
                note='Common render schema only; no visual fusion was performed',
                hashes={str(p):sha256(p) for p in inputs})
    write_json(out/'fusion-report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--episode',required=True)
    a=p.parse_args();prepare(a.run.resolve(),a.episode)
