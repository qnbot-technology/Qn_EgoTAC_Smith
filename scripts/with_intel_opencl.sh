#!/usr/bin/env bash
# Subprocess-only Intel driver selection; never source the collector project.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_RUNTIME="$TASK_ROOT/runtime/intel-opencl/25.18.33578.6"
TASK_DRIVER="$TASK_RUNTIME/root/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so"
if [[ ! -f "$TASK_DRIVER" ]]; then
  echo "Missing project-local Intel runtime: $TASK_DRIVER" >&2
  exit 1
fi
mkdir -p "$TASK_ROOT/.runtime/intel-opencl/icd"
printf '%s\n' "$TASK_DRIVER" > "$TASK_ROOT/.runtime/intel-opencl/icd/intel.icd"
export OCL_ICD_VENDORS="$TASK_ROOT/.runtime/intel-opencl/icd"
export LD_LIBRARY_PATH="$TASK_RUNTIME/root/usr/local/lib:$TASK_RUNTIME/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$@"
