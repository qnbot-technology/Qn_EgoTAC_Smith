#!/usr/bin/env python3
"""Index the portable payload, including models and the sample, but not environments/results."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ('AGENTS.md', 'README.md', 'run_pipeline.py', 'pyproject.toml',
         'requirements.txt', 'requirements-visual.txt', '.gitignore', 'LICENSE',
         'THIRD_PARTY_NOTICES.md', 'CONTRIBUTING.md', 'SECURITY.md')
DIRECTORIES = ('src', 'scripts', 'tests', 'configs', 'third_party', 'weights',
               'assets', 'runtime', 'doc', 'environment', 'provenance', 'data', 'examples', 'licenses')
EXCLUDED = {'__pycache__', '.pytest_cache', '.git', '.cache'}


def main():
    files = {ROOT / name for name in FILES}
    for name in DIRECTORIES:
        for path in (ROOT / name).rglob('*'):
            if EXCLUDED.intersection(path.relative_to(ROOT).parts) or path.suffix == '.pyc':
                continue
            if path.is_symlink():
                raise ValueError('Linked payload: ' + str(path))
            if path.is_file():
                files.add(path)
    records = []
    for path in sorted(files):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing/linked payload: ' + str(path))
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        records.append(dict(path=path.relative_to(ROOT).as_posix(),
                            bytes=path.stat().st_size, sha256=digest.hexdigest()))
    target = ROOT / 'bundle-manifest.json'
    target.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(files=len(records), bytes=sum(r['bytes'] for r in records),
                          manifest_sha256=hashlib.sha256(target.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
