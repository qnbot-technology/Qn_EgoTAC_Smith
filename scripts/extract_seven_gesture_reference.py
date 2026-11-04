#!/usr/bin/env python3
"""Extract the embedded HumanV2 seven-gesture reference from an HDF5 recording.

The input HDF5 is opened read-only. The resulting JSON is suitable for
``run_pipeline.py --mcap-reference``; it is a calibration artifact, not a
conversion of the recording. Use only when the HDF5 and MCAP share the same
operator, glove installation, protocol and coordinate convention.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import h5py


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-h5', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    src = a.input_h5.resolve(); dst = a.output.resolve()
    with h5py.File(src, 'r') as f:
        raw = f['calibration/tactile_emf/seven_gesture_reference/reference_json'][()]
    ref = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    if 'native_fit' not in ref or set(ref['native_fit'].get('sides', {})) != {'left', 'right'}:
        raise ValueError('HDF5 does not contain both left/right native_fit seven-gesture references')
    out = dict(ref)
    out['_extracted_from'] = {'path': str(src), 'sha256': sha256(src), 'read_only': True}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(dst), 'source_sha256': out['_extracted_from']['sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
