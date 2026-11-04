#!/usr/bin/env python3
"""Audit final metric hand bounds and display-only deadband onset."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from qn_smith.artifacts import write_json,now
from qn_smith.tactile_display import PROFILES
from qn_smith.tactile_channel_panel import channel_colours
from qn_smith.tactile_hand_panel import NEUTRAL
from qn_smith.hand_motion_final import VIEWPORT


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True)
    a=ap.parse_args();run=a.run.resolve();z=np.load(run/'display-source-arrays.npz')
    m=json.loads((run/'recipe.json').read_text());v=m['spatial_display'];hands={}
    for s in ('left','right'):
        p=z['assisted_'+s+'_points'];valid=z['assisted_'+s+'_valid'].astype(bool)
        finite=np.isfinite(p).all(-1);q=p@np.asarray(v['basis']).T*v['pixels_per_m']+v['offset_px']
        visible=q[valid[:,None]&finite]
        assert (visible>=np.asarray(VIEWPORT[:2])-1e-7).all()
        assert (visible<=np.asarray(VIEWPORT[2:])+1e-7).all()
        values=z['tactile_'+s+'_positive_residual_count'];dead=z['tactile_'+s+'_deadband_count']
        colors=channel_colours(values,PROFILES['D'],dead)
        within=np.isfinite(values)&(values<=dead)
        assert np.all(colors[within]==np.asarray(NEUTRAL))
        strong=np.isfinite(values)&(values>=dead+12)
        np.testing.assert_array_equal(colors[strong],channel_colours(values,PROFILES['D'])[strong])
        hands[s]=dict(valid_pose_frames=int(valid.sum()),finite_pose_nodes=int(finite.sum()),
            total_pose_nodes=int(finite.size),projected_xy_min=visible.min(0) if len(visible) else None,
            projected_xy_max=visible.max(0) if len(visible) else None,
            all_observed_pose_nodes_inside_viewport=True,channel_deadband_min=float(dead.min()),
            channel_deadband_max=float(dead.max()),neutral_cells_at_or_below_deadband=int(within.sum()),
            strong_cells_identical_to_original_D_colour=int(strong.sum()))
    edges=m['skeleton_edges'];assert len(edges)==24 and [4,5] not in edges
    write_json(run/'delivery/qa/display-contract-audit.json',dict(at=now(),passed=True,hands=hands,
        viewport_px=VIEWPORT,source_counts_changed=False,temporal_filter_added=False,
        deadband_or_baseline_refit=False,hand_local_scale_changed=False,
        anatomical_channel_placement_confirmed=False,training_ready=False))
    print('Metric pose bounds and channel onset verified for both hands',flush=True)


if __name__=='__main__':main()
