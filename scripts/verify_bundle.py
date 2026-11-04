#!/usr/bin/env python3
"""Verify a copied delivery before installing Python dependencies (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,default=ROOT/'bundle-manifest.json')
a=p.parse_args()
records=json.loads(a.manifest.read_text())
errors=[]
for rec in records:
    relative=Path(rec['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe manifest path')
    path=ROOT/relative
    if not path.is_file() or any(p.is_symlink() for p in [path, *path.parents] if p != ROOT and p.is_relative_to(ROOT)) or not path.resolve().is_relative_to(ROOT):
        errors.append(dict(path=rec['path'],reason='missing or linked'));continue
    if path.stat().st_size!=rec['bytes']:
        errors.append(dict(path=rec['path'],reason='size'));continue
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=rec['sha256']:errors.append(dict(path=rec['path'],reason='SHA256'))
print(json.dumps(dict(passed=not errors,files=len(records),bytes=sum(r['bytes'] for r in records),errors=errors),ensure_ascii=False,indent=2))
raise SystemExit(1 if errors else 0)
