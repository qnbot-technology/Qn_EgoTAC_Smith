#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$TASK_UNIT_DIR"
cat >"$TASK_UNIT_DIR/qn-ego-processing.service" <<EOF
[Unit]
Description=Qn EGO offline HDF5 processing queue
After=default.target

[Service]
Type=simple
WorkingDirectory=$TASK_ROOT
ExecStart=/bin/bash $TASK_ROOT/scripts/with_intel_opencl.sh $TASK_ROOT/.venv/bin/python $TASK_ROOT/run_pipeline.py worker
Restart=on-failure
RestartSec=10
Nice=10
CPUWeight=20
IOWeight=20
CPUAffinity=4 5 6 7
MemoryHigh=11G
MemoryMax=13G
TasksMax=128
KillMode=control-group
TimeoutStopSec=30
Environment=PYTHONNOUSERSITE=1

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now qn-ego-processing.service
systemctl --user status qn-ego-processing.service --no-pager
