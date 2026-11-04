#!/usr/bin/env python3
"""Verify/copy explicitly authorized assets; stdlib only, no automatic downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Unsafe asset path: ' + str(relative))
    path = root / relative
    if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root) and p != root):
        raise ValueError('Linked asset path: ' + str(relative))
    if not path.resolve().is_relative_to(root):
        raise ValueError('Asset escapes project')
    return path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate(path, record):
    if not path.is_file() or path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
        raise ValueError('Missing or mismatched asset: ' + record['path'])


def install(bundle, destination, records, check=False):
    # Validate the complete request before changing any destination.
    plan = []
    for record in records:
        source = safe_path(bundle, record['path'])
        target = safe_path(destination, record['path'])
        validate(source, record)
        if target.exists():
            validate(target, record)
        else:
            plan.append((source, target, record))
    if not check:
        for source, target, record in plan:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.asset-', delete=False) as f:
                temporary = Path(f.name)
            try:
                shutil.copyfile(source, temporary)
                shutil.copymode(source, temporary)
                validate(temporary, record)
                # Hard-link atomically refuses an unexpected existing destination.
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    return dict(passed=True, checked=len(records), copied=0 if check else len(plan), check_only=check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-root', required=True, type=Path)
    parser.add_argument('--manifest', type=Path, default=ROOT/'weights/manifest.json')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print(json.dumps(install(args.bundle_root, ROOT, json.loads(args.manifest.read_text()), args.check)))


if __name__ == '__main__':
    main()
