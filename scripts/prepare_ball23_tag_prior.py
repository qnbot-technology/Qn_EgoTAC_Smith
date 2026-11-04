#!/usr/bin/env python3
"""Create or audit the rectified Tag association prior; never modify source RGB."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from qn_smith.artifacts import now, sha256, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--episode', nargs='+', required=True)
    ap.add_argument('--create', action='store_true', help='Create missing priors; refuse overwrites')
    ap.add_argument('--baseline-root', type=Path, default=PROJECT/'reports/ball23-humanv2-tactile-20261022')
    args = ap.parse_args()
    rows = []
    for ep in args.episode:
        inp = args.run / 'input' / ep
        info = json.loads((inp / 'input.json').read_text())
        source = args.baseline_root / ep / 'tag3-tag16-handmap-residual-v4/bone-projections.npz'
        old = np.load(source)
        c = info['original_camera']
        data = {}
        for side in ('left', 'right'):
            data[side + '_uv'] = cv2.undistortPoints(old[side + '_filtered_uv_px'].reshape(-1, 1, 2),
                np.array(c['K']), np.array(c['distortion']), P=np.array(info['K'])).reshape(-1, 25, 2)
            data[side + '_valid'] = old[side + '_valid']
        path = inp / 'tag-prior.npz'
        if args.create:
            if path.exists():
                raise ValueError('Refuse prior overwrite: ' + str(path))
            np.savez_compressed(path, **data)
        actual = np.load(path)
        if set(actual.files) != set(data) or not all(
                np.array_equal(actual[k], v, equal_nan=True) for k, v in data.items()):
            raise ValueError('Rectified prior differs: ' + ep)
        rows.append(dict(episode=ep, prior_sha256=sha256(path), source_sha256=sha256(source),
                         all_arrays_exact_including_nan=True, created=args.create))
    write_json(args.run / 'tag-prior-reproduction-audit.json', dict(at=now(), passed=True,
        records=rows, script_sha256=sha256(__file__), scope='Exact UV undistortion and validity reproduction; not physical calibration'))
    print('Exact prior reproduction passed', len(rows), flush=True)


if __name__ == '__main__':
    main()
