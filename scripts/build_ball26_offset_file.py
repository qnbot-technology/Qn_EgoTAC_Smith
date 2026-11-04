#!/usr/bin/env python3
"""Build the colleague-facing Ball26 CAD offset bundle.

The bundle deliberately contains geometry and frame conventions only.  It does
not encode the temporary MCAP right-hand tag numbering.  Acquisition adapters
must use the canonical HDF5 binding (left physical anchor 3, right physical
anchor 16) until a dataset-specific binding is explicitly reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mount", type=Path, default=ROOT / "configs/ball26/mount-26mm-new.json")
    ap.add_argument("--layout", type=Path, default=ROOT / "configs/ball26/layout-26mm-mcap-temporary.json")
    ap.add_argument("--geometry", type=Path, default=ROOT / "provenance/ball26-step-geometry.json")
    ap.add_argument("--output", type=Path, default=ROOT / "configs/ball26/26mm_new_offset_20261029.json")
    args = ap.parse_args()

    mount = json.loads(args.mount.read_text(encoding="utf-8"))
    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    if mount["asset_id"] != "apriltag_emf_ball26" or layout["asset_id"] != mount["asset_id"]:
        raise SystemExit("mount/layout asset mismatch")
    if float(layout["tag_size_m"]) != 0.023:
        raise SystemExit("the observed printed tag size must remain 23 mm")
    if mount.get("source", {}).get("sha256") != geometry.get("step_sha256"):
        raise SystemExit("mount and STEP geometry provenance mismatch")

    faces = {}
    for face in mount.get("faces", []):
        key = face["face_key"]
        faces[key] = {
            "step_face_26mm": face.get("step_face_26mm"),
            "T_bottom_support": face["T_bottom_support"],
            "normal_fixture_cad": face.get("normal_fixture_cad"),
            "center_delta_vs_previous_mm": face.get("center_delta_vs_previous_mm"),
            "geometry_match_error_mm": face.get("geometry_match_error_mm"),
        }

    out = {
        "schema": "qn_smith.ball26_new_offset.v1",
        "status": "colleague_delivery_candidate",
        "created_utc": "2026-10-29",
        "units": "m",
        "matrix_convention": "T_parent_child maps child into parent; column vectors",
        "purpose": "26mm fixture CAD face-center to bottom-center to EMF coil-center offset",
        "application_scope": {
            "mcap": "attach this file as calibration/26mm_new_offset.json; do not mutate source records",
            "hdf5": "unchanged; keep the established HDF5 MX calibration and tag order",
            "tag_to_bottom_to_coil_applied_once": True,
        },
        "asset": {
            "asset_id": mount["asset_id"],
            "printed_tag_size_m": 0.023,
            "printed_tag_size_note": "The CAD support is 26 mm, while the observed black-border printed Tag is 23 mm; do not use 0.026 for PnP.",
            "step": {
                "filename": mount["source"]["filename"],
                "sha256": mount["source"]["sha256"],
                "freecad_version": geometry.get("freecad_version"),
            },
        },
        "coordinate_convention": mount["emf_pose_convention"],
        "transforms": {
            "T_bottom_emf": mount["transforms"]["T_bottom_emf"],
            "T_bottom_emf_mechanical": mount["transforms"]["T_bottom_emf_mechanical"],
            "T_render_root_cad_bottom": layout["T_render_root_cad_bottom"],
            "T_render_root_emf_cad_candidate": layout["T_render_root_emf_cad_candidate"],
            "formula": "T_camera_emf = T_camera_root @ T_render_root_emf_cad_candidate @ p_emf",
        },
        "face_center_offsets": faces,
        "tag_face_relations": {
            "source": "configs/ball26/layout-26mm-mcap-temporary.json",
            "local_observed_ids": sorted(int(k) for k in layout["bindings"]),
            "T_bottom_tag": {k: v["T_bottom_tag"] for k, v in layout["bindings"].items()},
            "unobserved_local_faces": [10, 11, 12],
        },
        "canonical_h5_binding": {
            "left_physical_anchor_tag_id": 3,
            "right_physical_anchor_tag_id": 16,
            "local_layout_tag_ids": list(range(10)),
            "note": "Use the HDF5-established physical order. No MCAP temporary right-start override is part of this offset file.",
        },
        "provenance": {
            "mount_file": str(args.mount.relative_to(ROOT)),
            "mount_sha256": sha256(args.mount),
            "layout_file": str(args.layout.relative_to(ROOT)),
            "layout_sha256": sha256(args.layout),
            "geometry_file": str(args.geometry.relative_to(ROOT)),
            "geometry_sha256": sha256(args.geometry),
        },
        "quality": {
            "electrical_axes_verified": False,
            "training_ready": False,
            "review_required": "verify colleague's physical tag binding and camera reprojection before promotion",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), "schema": out["schema"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
