#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON="${PYTHON:-python3}"
TASK_ENV="${VENV:-$TASK_ROOT/.venv}"
"$TASK_PYTHON" -I -c 'import sys; assert sys.version_info[:2] == (3, 12), "This bundle requires Python 3.12"'
TASK_CONSTRAINTS=()
if [[ -f "$TASK_ROOT/environment/visual-local.constraints.txt" ]]; then
  TASK_CONSTRAINTS=(-c "$TASK_ROOT/environment/visual-local.constraints.txt")
fi
"$TASK_PYTHON" -m venv --copies "$TASK_ENV"
"$TASK_ENV/bin/python" -I -m pip install --no-cache-dir "${TASK_CONSTRAINTS[@]}" -r "$TASK_ROOT/requirements.txt" 'contourpy<1.4'
if [[ "${1:-}" == "--visual" ]]; then
  "$TASK_ENV/bin/python" -I -m pip install --no-cache-dir --no-deps "${TASK_CONSTRAINTS[@]}" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  "$TASK_ENV/bin/python" -I "$TASK_ROOT/scripts/install_torch_dependencies.py"
  "$TASK_ENV/bin/python" -I -m pip install --no-cache-dir "${TASK_CONSTRAINTS[@]}" -r "$TASK_ROOT/requirements-visual.txt"
  "$TASK_ENV/bin/python" -I -m pip install --no-cache-dir --no-build-isolation --no-deps "${TASK_CONSTRAINTS[@]}" "chumpy @ git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
fi
"$TASK_ENV/bin/python" -I -m pip check
"$TASK_ENV/bin/python" "$TASK_ROOT/run_pipeline.py" doctor --deep
