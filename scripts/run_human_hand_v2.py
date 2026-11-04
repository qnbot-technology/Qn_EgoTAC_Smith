#!/usr/bin/env python3
"""Run the fixed-wrist Human Hand V2 solver on prepared all-valid EMF poses.

This is intentionally an isolated experiment driver. It never modifies the
source HDF5, the OpenXR baseline, or the downloaded colleague package.
Promoted unchanged from the April1 experiment for reproducible Git delivery.
See docs/HumanV2_完整视频核查与四数据集协作_20261022.md for the
all-valid input restriction and required upstream HDF5/source-index checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-npz", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--side", choices=["left", "right"], default="right")
    ap.add_argument("--config", type=Path, help="Explicit source-bound calibration YAML; copied into output, package stays unchanged")
    args = ap.parse_args()
    # Missing inputs and time gaps must not carry solver state across segments.
    from native_segments import segment_ranges, run_segments
    with np.load(args.input_npz) as prepared:
        if 'valid' in prepared:
            mask = np.asarray(prepared['valid'], bool)
            ranges = segment_ranges(mask, prepared['receive_ns'])
            if not mask.all() or len(ranges) != 1:
                run_segments(args, Path(__file__).resolve(), dict(prepared))
                return
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    package = args.package.resolve()
    input_npz = args.input_npz.resolve()

    # Copy the fixed-wrist candidate and overlay into the experiment so the
    # run is reproducible without rewriting the colleague's package.
    src_cfg = args.config.resolve() if args.config else package / f"calibration_results/bundled/{args.side}-runtime.yaml"
    cfg = out / f"{args.side}-runtime.yaml"
    text = src_cfg.read_text()
    text = text.replace("__resource_root__: .", f"__resource_root__: {package / 'native/qnbot_hand/resources'}")
    # The bundled file currently contains an absolute resource root in some
    # snapshots; normalize it to this package copy in either case.
    lines = text.splitlines()
    lines = [f"__resource_root__: {package / 'native/qnbot_hand/resources'}" if l.startswith("__resource_root__:") else l for l in lines]
    cfg.write_text("\n".join(lines) + "\n")
    options_src = package / "native/qnbot_hand/configs/human_hand_v2_options.yaml"
    options = out / "options.yaml"
    shutil.copy2(options_src, options)

    z = np.load(input_npz, allow_pickle=True)
    poses = np.asarray(z["poses"], dtype=np.float64)
    receive_ns = np.asarray(z["receive_ns"], dtype=np.int64)
    timestamp = (receive_ns - receive_ns[0]) / 1e9
    ndjson = out / "input.ndjson"
    with ndjson.open("w") as f:
        for t, p in zip(timestamp, poses):
            f.write(json.dumps({"timestamp_sec": float(t), "poses": p.tolist(), "valid": True}, separators=(",", ":")) + "\n")

    binary = package / "runtime/bin/human_hand_v2_runtime"
    raw_out = out / "native-output.jsonl"
    stderr = out / "native-stderr.log"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(package / "runtime/lib") + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    cmd = [str(binary), "--config", str(cfg), "--side", args.side, "--resource-root", str(package / "native/qnbot_hand/resources"), "--options", str(options), "--input", str(ndjson)]
    with raw_out.open("w") as fo, stderr.open("w") as fe:
        p = subprocess.run(cmd, stdout=fo, stderr=fe, env=env, check=False)
    (out / "run-command.json").write_text(json.dumps({"command": cmd, "returncode": p.returncode, "input_sha256": sha256(input_npz), "config_sha256": sha256(cfg), "options_sha256": sha256(options), "package": str(package)}, indent=2) + "\n")
    if p.returncode:
        raise SystemExit(f"human_hand_v2_runtime failed with {p.returncode}; see {stderr}")

    rows = []
    with raw_out.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != len(poses):
        raise RuntimeError(f"expected {len(poses)} output rows, got {len(rows)}")
    names = rows[0].get("skeleton_names", [])
    joint_names = rows[0].get("joint_names", [])
    def stack(key, shape_tail, fill=np.nan):
        a = np.full((len(rows),) + shape_tail, fill, dtype=np.float64)
        for i, r in enumerate(rows):
            v = r.get(key)
            if v is not None:
                vv = np.asarray(v, dtype=np.float64)
                if vv.size == int(np.prod(shape_tail)):
                    vv = vv.reshape(shape_tail)
                a[i] = vv
        return a
    valid = np.asarray([bool(r.get("tracking_valid", False) and r.get("output_available", False)) for r in rows], dtype=bool)
    np.savez_compressed(out / "native-output.npz",
        timestamp_sec=timestamp, source_index=z["source_index"], tracking_valid=valid,
        output_available=np.asarray([bool(r.get("output_available", False)) for r in rows]),
        sensor_rmse_m=np.asarray([r.get("sensor_rmse_m", np.nan) for r in rows], dtype=np.float64),
        sensor_orientation_error_rad=stack("sensor_orientation_error_rad", (5,)),
        sensor_position_error_m=stack("sensor_position_error_m", (5,)),
        wrist_position_glove_m=stack("wrist_position_glove_m", (3,)),
        wrist_rotation_glove=stack("wrist_rotation_glove", (3,3)),
        q_active_rad=stack("q_active_rad", (len(joint_names),)),
        positions_glove_m=stack("positions_glove_m", (len(names),3)),
        positions_wrist_m=stack("positions_wrist_m", (len(names),3)),
        orientations_glove_xyzw=stack("orientations_glove_xyzw", (len(names),4)),
        orientations_wrist_xyzw=stack("orientations_wrist_xyzw", (len(names),4)),
    )
    # Keep the exact five-point input beside the solver output so downstream
    # rendering/diagnostics do not need to reach into the OpenXR experiment.
    # This is a copy of the input arrays only; the source HDF5 remains read-only.
    np.savez_compressed(out / "input-poses.npz",
        poses=poses,
        source_index=np.asarray(z["source_index"]),
        receive_ns=receive_ns,
        timestamp_sec=timestamp)
    report = {
        "schema": "qn_smith.human_hand_v2_fixed_wrist.v2",
        "side": args.side,
        "input_npz": str(input_npz), "input_sha256": sha256(input_npz),
        "frames": len(rows), "tracking_valid_frames": int(valid.sum()),
        "tracking_valid_fraction": float(valid.mean()), "skeleton_names": names,
        "joint_names": joint_names, "runtime_returncode": p.returncode,
        "config": str(cfg), "config_sha256": sha256(cfg), "options": str(options),
        "calibration_source": str(src_cfg), "calibration_source_sha256": sha256(src_cfg),
        "bundled_calibration_used": args.config is None,
        "options_sha256": sha256(options), "solver_package": str(package),
    }
    (out / "native-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
