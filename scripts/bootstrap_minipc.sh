#!/usr/bin/env bash
# Isolated Intel/CPU runtime. No CUDA packages and no collector environment edits.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON="${PYTHON:-python3}"
TASK_ENV="${VENV:-$TASK_ROOT/.venv}"
TASK_CONSTRAINTS="$TASK_ROOT/environment/minipc-cpu.constraints.txt"
"$TASK_PYTHON" -I -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
"$TASK_PYTHON" -m venv --copies "$TASK_ENV"
"$TASK_ENV/bin/python" -I -m pip install --no-cache-dir -c "$TASK_CONSTRAINTS" -r "$TASK_ROOT/requirements.txt" setuptools==80.9.0
if [[ -n "${TORCH_WHEEL:-}" ]]; then
  # An optional locally transferred official CPU wheel avoids slow CDN redirects.
  "$TASK_ENV/bin/python" -I -m pip install --no-cache-dir -c "$TASK_CONSTRAINTS" "$TORCH_WHEEL"
fi
"$TASK_ENV/bin/python" -I -m pip install --no-cache-dir -c "$TASK_CONSTRAINTS" torch==2.8.0+cpu torchvision==0.23.0+cpu --index-url https://download.pytorch.org/whl/cpu
"$TASK_ENV/bin/python" -I -m pip install --no-cache-dir -c "$TASK_CONSTRAINTS" -r "$TASK_ROOT/requirements-visual.txt" openvino==2026.4.1
"$TASK_ENV/bin/python" -I -m pip install --no-cache-dir --no-build-isolation --no-deps "chumpy @ git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
"$TASK_ENV/bin/python" -I -m pip check
"$TASK_ENV/bin/python" "$TASK_ROOT/run_pipeline.py" doctor --deep
