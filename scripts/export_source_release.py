#!/usr/bin/env python3
"""Export a curated source tree. Models/private deployment files stay in place."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TOP = ['AGENTS.md', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'CONTRIBUTING.md',
       'SECURITY.md', '.gitignore', '.gitleaks.toml', 'run_pipeline.py', 'pyproject.toml',
       'requirements.txt', 'requirements-visual.txt']
DIRECTORIES = ['src', 'scripts', 'tests', 'configs', 'doc', 'licenses', 'environment', 'examples', 'third_party', 'runtime/python_startup', 'assets']
EXCLUDED_PARTS = {'.git', '__pycache__', '.pytest_cache', '.cache', 'build', 'dist', '.github'}
BINARY_SUFFIXES = {'.pyc', '.pt', '.pth', '.ckpt', '.pkl', '.safetensors', '.h5', '.hdf5', '.mcap',
                   '.mp4', '.tiff', '.ipynb', '.png', '.jpg', '.jpeg', '.gif', '.ttc', '.so', '.a'}


def included(relative):
    s = relative.as_posix()
    if s.startswith(('third_party/live_capture/', 'third_party/egosmith/', 'configs/live/', 'configs/mcap/')) or s == 'scripts/build_live_capture.py':
        return False
    # Deny by default: never export proprietary HumanV2 source or internal patches.
    if s.startswith('third_party/humanv2/'):
        base = 'third_party/humanv2/'
        tail = s[len(base):]
        return (tail in {'README.md', 'LICENSE.txt', 'common.py', 'requirements.txt',
                         'runtime-manifest.json',
                         'native/qnbot_hand/configs/human_hand_v2_options.yaml',
                         'runtime/bin/human_hand_v2_runtime'} or
                (tail.startswith(('runtime/lib/', 'runtime/third_party_licenses/',
                                  'native/qnbot_hand/resources/')) and
                 not EXCLUDED_PARTS.intersection(relative.parts) and
                 relative.suffix.lower() not in {'.cpp', '.cc', '.c', '.h', '.hpp', '.patch', '.a'}))
    if s in {'assets/fonts/NotoSansCJK-Regular.ttc', 'assets/readme/qnbot-logo.png', 'assets/readme/raw-rgb.gif', 'assets/readme/humanv2-result.gif', 'assets/readme/tactile-detail.gif',
             'assets/readme/tactile-action.gif', 'assets/readme/hand-trajectory.gif'}:
        return True
    if s == 'licenses/MANO-terms.html':
        return False
    if EXCLUDED_PARTS.intersection(relative.parts) or '.bak' in relative.name or relative.suffix in BINARY_SUFFIXES:
        return False
    if s.startswith(('configs/live/', 'configs/mcap/', 'third_party/humanv2/calibration_results/',
                     'third_party/humanv2/runtime/bin/', 'third_party/humanv2/runtime/lib/',
                     'third_party/live_capture/vendor/orbbec/lib/', 'third_party/live_capture/vendor/orbbec/extensions/',
                     'third_party/efficientsam3/images/', 'third_party/efficientsam3/docs/',
                     'configs/ball26/sources/')):
        return False
    if '.so.' in relative.name or relative.suffix in ('.local-before', '.minipc'):
        return False
    if s.startswith('third_party/sam3/'):
        return s.startswith('third_party/sam3/sam3/') or len(relative.parts) == 3
    if s.startswith('third_party/efficientsam3/sam3/'):
        # Only the inference package and tokenizer, plus top-level notices/configs.
        return (s.startswith('third_party/efficientsam3/sam3/sam3/') or
                s.startswith('third_party/efficientsam3/sam3/assets/bpe_') or len(relative.parts) == 4)
    return True


def export(destination):
    destination = Path(destination).absolute()
    if destination == ROOT or ROOT.is_relative_to(destination):
        raise ValueError('Destination must not contain the working project')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Export requires an empty destination; do not overwrite a checkout')
    destination.mkdir(parents=True, exist_ok=True)
    # A source-only local checkout must not silently produce an incomplete native package.
    native = ROOT / 'third_party/humanv2'
    manifest = native / 'runtime-manifest.json'
    if not manifest.is_file():
        raise ValueError('HumanV2 binary runtime manifest is required; use the reviewed release package')
    for record in json.loads(manifest.read_text())['files']:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe HumanV2 manifest path')
        payload = native / relative
        if payload.is_symlink() or not payload.is_file() or hashlib.sha256(payload.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('HumanV2 binary package is incomplete or modified: ' + str(relative))
    paths = [ROOT / name for name in TOP if (ROOT / name).is_file()]
    for directory in DIRECTORIES:
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and included(p.relative_to(ROOT)))
    paths += [ROOT/'weights/manifest.json', ROOT/'weights/README.md', ROOT/'runtime/manifest.json']
    for source in sorted(set(paths)):
        if source.is_symlink():
            raise ValueError('Linked source: '+str(source))
        target = destination/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    records = [dict(path=p.relative_to(destination).as_posix(), bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest())
               for p in sorted(destination.rglob('*')) if p.is_file()]
    (destination/'source-manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(files=len(records), bytes=sum(r['bytes'] for r in records))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    export(parser.parse_args().destination)
