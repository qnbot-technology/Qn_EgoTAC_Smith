"""Content identities, atomic metadata, and verified portable run artifacts."""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(json_value(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def identity(value):
    payload = json.dumps(json_value(value), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def manifest(directory, exclude=()):
    root = Path(directory)
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in exclude or relative.endswith(".partial"):
            continue
        records.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha256(path)))
    return records


def verify_manifest(directory, records):
    root = Path(directory)
    errors = []
    for record in records:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            errors.append(record["path"] + ": outside archive")
        elif not path.is_file() or path.stat().st_size != record["bytes"]:
            errors.append(record["path"] + ": missing/size mismatch")
        elif sha256(path) != record["sha256"]:
            errors.append(record["path"] + ": hash mismatch")
    return errors
