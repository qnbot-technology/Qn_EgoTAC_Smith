#!/usr/bin/env python3
"""Bind a new, complete recording to an isolated NAS Ball23/HumanV2 run."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from qn_smith.artifacts import now, sha256, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--registration-template', type=Path, required=True)
    a = ap.parse_args()
    source = a.source.resolve()
    before = source.stat()
    digest = sha256(source)
    out = a.run.resolve() / 'provenance'
    # MCAP native runs write adapter provenance before this stage.  Reuse the
    # existing provenance directory while keeping every generated file
    # source-bound; HDF5 runs retain the same isolated RUN layout.
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, 'r') as f:
        if f.attrs.get('complete') != 1:
            raise ValueError('Recording is not marked complete')
        times = f['time/relative_s'][:].ravel()
        if not len(times) or not np.all(np.diff(times) > 0):
            raise ValueError('Invalid source timeline')
        reference = f['calibration/tactile_emf/seven_gesture_reference/reference_json'][()]
        if isinstance(reference, bytes):
            reference = reference.decode()
        reference = json.loads(reference)
        calibration = {}
        for side in ('left', 'right'):
            text = reference['native_fit']['sides'][side]['result_yaml']
            target = out / f'embedded-seven-gesture-{side}.yaml'
            target.write_text(text)
            calibration[side] = dict(path=str(target), sha256=sha256(target),
                hdf5_path='calibration/tactile_emf/seven_gesture_reference/reference_json',
                json_path=f'native_fit.sides.{side}.result_yaml', bundled_calibration_used=False)
        thumbs = []
        for i in np.linspace(0, len(times)-1, 9).astype(int):
            rgb = cv2.imdecode(np.asarray(f['raw/cameras/head_rgb/payload'][i], np.uint8), 1)
            thumb = cv2.resize(rgb, (640, 360))
            cv2.putText(thumb, f'row {i} / {times[i]:.2f}s', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 1)
            thumbs.append(thumb)
        cv2.imwrite(str(out/'source-contact-sheet.jpg'),
                    np.vstack([np.hstack(thumbs[i:i+3]) for i in range(0, 9, 3)]))
        info = dict(at=now(), source=str(source), source_sha256=digest,
                    source_size=before.st_size, frames=len(times),
                    duration_between_first_last_rgb_s=float(times[-1]-times[0]),
                    embedded_personal_calibrations=calibration, training_ready=False)
        if f.attrs.get('deployment_diagnostic_derivative', False):
            info['diagnostic_derivative'] = dict(
                original_source_sha256=str(f.attrs['source_sha256']),
                first_row_0based=int(f.attrs['source_first_row_0based']),
                last_row_0based=int(f.attrs['source_last_row_0based']),
                original_full_recording_acceptance=False)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed while preparing')
    template = json.loads(a.registration_template.read_text())
    template.update(calibration_id=source.stem+'-ball23-existing-mount-candidate',
                    source_sha256=[digest], sources=[dict(source=str(source),
                    source_sha256=digest, frames=len(times))],
                    reused_from=dict(path=str(a.registration_template.resolve()),
                    sha256=sha256(a.registration_template),
                    evidence='Same Ball23 workflow requested; mounting is inherited, not newly measured'))
    write_json(out/'registration.json', template)
    write_json(out/'input.json', info)
    shutil.copy2(__file__, out/Path(__file__).name)
    print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
