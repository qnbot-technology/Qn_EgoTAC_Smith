"""Portable paths and the private native runtime environment."""
import hashlib
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parent
RESOURCES = ROOT / 'native/qnbot_hand/resources'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def native_binary():
    override = os.environ.get('HUMAN_HAND_V2_BINARY')
    if override:
        binary = Path(override).expanduser().resolve()
    else:
        if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
            raise RuntimeError('Native calibration/replay requires Linux x86_64; use Ubuntu 22.04+ or Windows WSL.')
        binary = ROOT / 'runtime/bin/human_hand_v2_runtime'
    if not binary.is_file():
        raise FileNotFoundError(f'Native runtime missing: {binary}')
    binary.chmod(binary.stat().st_mode | 0o111)
    return binary


def native_env(cache):
    env = dict(os.environ)
    env['QNBOT_HOME'] = str(Path(cache).resolve())
    env['LD_LIBRARY_PATH'] = str(ROOT / 'runtime/lib') + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env


def active_calibration():
    path = ROOT / 'active_calibration.json'
    if path.exists():
        return (ROOT / read_json(path)['directory']).resolve()
    return ROOT / 'calibration_results/bundled'
