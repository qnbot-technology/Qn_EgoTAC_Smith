import os
from pathlib import Path
from .artifacts import sha256

def resolve_path(value, project):
    expanded = os.path.expandvars(str(value))
    if '$' in expanded:
        raise ValueError(f'Unresolved environment variable in path: {value}')
    p = Path(expanded).expanduser()
    return p if p.is_absolute() else project / p

def file_record(path):
    path = Path(path)
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Input changed while reading: {path}')
    return {'uri': str(path.resolve()), 'bytes': after.st_size, 'sha256': digest}
