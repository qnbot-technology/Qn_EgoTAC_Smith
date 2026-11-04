#!/usr/bin/env python3
"""Prepare calibrated, rectified model inputs; original HDF5/RGB stays intact."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
from qn_smith.artifacts import now, sha256, write_json, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--episode', nargs='+', required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--baseline-root', type=Path, default=PROJECT/'reports/ball23-humanv2-tactile-20261022')
    args = ap.parse_args()
    baseline = args.baseline_root
    cv2.setNumThreads(2)
    for episode in args.episode:
        tic = time.monotonic()
        old = baseline/episode
        report = json.loads((old/'trajectory-report.json').read_text())
        source = Path(report['source'])
        if sha256(source) != report['source_sha256']:
            raise ValueError('source SHA mismatch')
        out = args.run/'input'/episode
        out.mkdir(parents=True, exist_ok=False)
        (out/'images').mkdir()
        c = json.loads((old/'camera.json').read_text())
        K = np.asarray(c['K'], float); D = np.asarray(c['distortion'], float)
        width, height = 960, 540
        scale = width/c['width']
        Ks = K.copy(); Ks[:2] *= scale
        # resize() maps pixel centres, not their top-left corners.
        Ks[:2, 2] += (scale-1)/2
        mx,my = cv2.initUndistortRectifyMap(K,D,None,Ks,(width,height),cv2.CV_32FC1)
        with h5py.File(source, 'r') as f:
            n = len(f['time/relative_s'])
            for row,payload in enumerate(f['raw/cameras/head_rgb/payload']):
                image = cv2.imdecode(np.asarray(payload, np.uint8), 1)
                rect = cv2.remap(image,mx,my,cv2.INTER_LINEAR)
                if not cv2.imwrite(str(out/'images'/f'{row:06d}.jpg'),rect,[cv2.IMWRITE_JPEG_QUALITY,95]):
                    raise IOError('JPEG write failed')
            np.savez_compressed(out/'frame-map.npz', source_row=np.arange(n),
                sample_time_s=f['time/relative_s'][:].ravel(),
                camera_receive_ns=f['raw/cameras/head_rgb/receive_monotonic_ns'][:].ravel(),
                camera_frame_index=f['raw/cameras/head_rgb/frame_index'][:].ravel())
        write_json(out/'input.json',dict(schema='qn_smith.ball23_visual_root_input.v1',
            episode=episode, input_sha256=report['source_sha256'],source_hdf5=str(source),
            rows=n, width=width,height=height,K=Ks, distortion=np.zeros_like(D),
            original_camera=c, model_image_space='rectified_color_optical',
            original_rgb_modified=False, image_extension='jpg', image_encoding_quality=95,
            scale=scale, pixel_center_resize=True, prepared_at=now(),
            script_sha256=sha256(__file__), elapsed_s=time.monotonic()-tic))
        write_json(out/'manifest.json', manifest(out))
        print(episode,n,round(time.monotonic()-tic,3),flush=True)


if __name__ == '__main__': main()
