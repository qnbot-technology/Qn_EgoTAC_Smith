"""Record actual project-module origins in child processes, without site patches."""
import atexit
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
output = os.environ.get('QN_EGO_IMPORT_AUDIT')

@atexit.register
def record_origins():
    if not output:
        return
    names = ('qn_ego', 'qn_smith', 'sam3', 'hawor', 'lib', 'infiller')
    records = {}
    errors = []
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] not in names:
            continue
        source = getattr(module, '__file__', None)
        if source:
            path = Path(source).resolve()
            records[name] = str(path)
            if not path.is_relative_to(root):
                errors.append(dict(module=name, source=str(path)))
    Path(output).write_text(json.dumps(dict(passed=not errors, project_root=str(root),
        modules=records, errors=errors), indent=2)+'\n')
