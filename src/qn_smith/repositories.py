"""One dependency registry for all Qn entrypoints; no import-time sys.path edits."""
import json
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[2]


def catalog(project=None):
    root = Path(project) if project is not None else project_root()
    values = json.loads((root/"third_party/catalog.json").read_text())
    if values.get("schema") != "qn_smith.dependencies.v1":
        raise ValueError("Unsupported dependency catalog")
    for item in values["repositories"].values():
        for key in ("path", "source_path", "legacy_path"):
            if key in item and (Path(item[key]).is_absolute() or ".." in Path(item[key]).parts):
                raise ValueError("Dependency path must stay inside project")
    return values["repositories"]


def repository(name, project=None):
    root = Path(project) if project is not None else project_root()
    entries = catalog(root)
    if name not in entries:
        raise ValueError(f"Unknown dependency {name}")
    path = root/entries[name]["path"]
    if not path.is_dir():
        raise FileNotFoundError(f"Dependency {name} unavailable at {path}; initialize/restore project layout")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dependency source outside project; restore a code copy")
    return path.resolve()
