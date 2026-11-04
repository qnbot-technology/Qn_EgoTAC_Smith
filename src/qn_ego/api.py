"""Stable Python API for the portable Qn EGO processing entry points.

The functions in this module are the integration surface for a collection
platform or a frontend.  They return the JSON receipt written by the
corresponding run directory and do not alter the source HDF5.  The existing
CLI remains available through :mod:`qn_ego.pipeline` and ``run_pipeline.py``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
from typing import Any, Sequence

from . import pipeline

ROOT = pipeline.ROOT
PathLike = str | Path


def _path(value: PathLike) -> Path:
    return Path(value).expanduser().resolve()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Expected pipeline receipt was not created: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Pipeline receipt must be a JSON object: {path}")
    return value


def inspect_run(output_dir: PathLike) -> dict[str, Any]:
    """Read a completed run without rerunning any stage.

    ``verification`` is returned separately from ``result`` so a frontend
    cannot mistake a completed process for a verified artifact bundle.
    """

    output = _path(output_dir)
    return {
        "output_dir": str(output),
        "result": _read_json(output / "result.json"),
        "verification": _read_json(output / "verification.json"),
    }


def doctor(*, config: PathLike = ROOT / "configs/pipeline.json", deep: bool = False) -> dict[str, Any]:
    """Check bundled code, dependencies and (optionally) model files."""

    return pipeline.doctor(pipeline.load_config(_path(config)), full=True, deep=deep)


def run_recording(
    input_path: PathLike,
    output_dir: PathLike,
    *,
    config: PathLike = ROOT / "configs/pipeline.json",
    visual: str = "full",
    gpu: int = 0,
    device: str = "cuda",
    threads: int = 6,
    start_s: float | None = None,
    end_s: float | None = None,
    sam_python: PathLike | None = None,
    mano_python: PathLike | None = None,
    sam_backend: str = "torch",
    sam_model: str = "sam3",
    sam_openvino_dir: PathLike | None = None,
    hawor_backend: str = "torch",
    openvino_model: PathLike | None = None,
    openvino_device: str = "CPU",
    resume: bool = False,
    stop_after_apriltag: bool = False,
    visual_assist: bool = True,
    render_video: bool = False,
    mcap_reference: PathLike | None = None,
    mcap_tag_search: str = "full",
    mcap_right_tag_start: int = 13,
    tag_frame_workers: int = 0,
    mcap_cache: PathLike | None = None,
    mcap_offset_profile: str = "ball23-h5",
    mcap_depth_mode: str = "raw",
) -> dict[str, Any]:
    """Run a closed HDF5 or Qn DAQ v1 MCAP recording through the offline pipeline.

    The returned object is the content of ``output_dir/result.json``.  The
    ``mcap_reference`` should point to the recording's own seven-gesture JSON;
    omitting it retains the bundled H5 reference only as an explicitly
    ``bundled_reference_unverified`` diagnostic. The function blocks until all requested stages and the final verification have
    completed. ``stop_after_apriltag=True`` instead returns an explicit incomplete
    checkpoint; continue with the identical options, ``resume=True`` and
    ``stop_after_apriltag=False``. No visual runtime runs before this pause.
    ``visual_assist=False`` completes Tag/HumanV2/tactile and intermediate
    mapping artifacts without optical flow, SAM or MANO. Set ``render_video=True``
    to additionally encode the final Tag-only video; this is disabled by default.
    ``visual='full'`` is the default and cannot silently degrade;
    use ``visual='flow'`` only for an explicitly diagnostic run.
    ``mcap_depth_mode="metadata"`` creates the lightweight MCAP canonical view
    for Tag/HumanV2-only processing; ``"raw"`` preserves all depth payloads for
    point-cloud/depth consumers. The mode is part of the immutable request and
    cache identity. ``tag_frame_workers=0`` selects the bounded automatic default (MCAP
    ``min(6, threads, CPU_count-2)``, HDF5 up to six bounded workers); ROI
    mode remains serial for causal tracking. Set an explicit positive value
    for a platform-specific cap, such as 8 for an offline MCAP run.
    """

    if mcap_tag_search not in ("full", "roi"):
        raise ValueError("mcap_tag_search must be 'full' or 'roi'")
    if mcap_right_tag_start < 0:
        raise ValueError("mcap_right_tag_start must be non-negative")
    if tag_frame_workers < 0:
        raise ValueError("tag_frame_workers must be non-negative (0=automatic)")
    if mcap_offset_profile not in ("ball23-h5", "ball26-new"):
        raise ValueError("mcap_offset_profile must be 'ball23-h5' or 'ball26-new'")
    if mcap_depth_mode not in ("raw", "metadata"):
        raise ValueError("mcap_depth_mode must be 'raw' or 'metadata'")
    args = SimpleNamespace(
        input=_path(input_path), output=_path(output_dir), config=_path(config),
        visual=visual, sam_python=_path(sam_python) if sam_python else None,
        mano_python=_path(mano_python) if mano_python else None, gpu=gpu,
        start_s=start_s, end_s=end_s, device=device, threads=threads,
        hawor_backend=hawor_backend, sam_backend=sam_backend, sam_model=sam_model,
        sam_openvino_dir=_path(sam_openvino_dir) if sam_openvino_dir else None,
        openvino_model=_path(openvino_model) if openvino_model else None,
        openvino_device=openvino_device, resume=resume,
        stop_after_apriltag=stop_after_apriltag,
        visual_assist=visual_assist,
        render_video=bool(render_video),
        mcap_reference=_path(mcap_reference) if mcap_reference else None,
        mcap_tag_search=mcap_tag_search,
        mcap_right_tag_start=int(mcap_right_tag_start),
        tag_frame_workers=int(tag_frame_workers),
        mcap_cache=_path(mcap_cache) if mcap_cache else None,
        mcap_offset_profile=mcap_offset_profile,
        mcap_depth_mode=mcap_depth_mode,
    )
    result = pipeline.run_pipeline(args)
    if not isinstance(result, dict):
        result = _read_json(args.output / "result.json")
    return result


def run_tag_skeleton(
    input_path: PathLike,
    output_dir: PathLike,
    *,
    config: PathLike = ROOT / "configs/pipeline.json",
    frames: int = 0,
    threads: int = 4,
    modes: Sequence[str] = ("full", "roi"),
    roi_margin: int = 100,
    refresh: int = 30,
    parallel_workers: bool = False,
) -> dict[str, Any]:
    """Replay a closed HDF5 recording with causal Tag + HumanV2 mapping only."""

    output = _path(output_dir)
    source = _path(input_path)
    argv = [
        sys.executable, str(ROOT / "scripts/benchmark_tag_skeleton.py"),
        "--input", str(source), "--output", str(output), "--config", str(_path(config)),
        "--frames", str(frames), "--threads", str(threads), "--modes", *map(str, modes),
        "--roi-margin", str(roi_margin), "--refresh", str(refresh),
    ]
    if parallel_workers:
        argv.append("--parallel-workers")
    subprocess.run(argv, cwd=ROOT, env=pipeline.env_for(0, "cpu", threads), check=True)
    return _read_json(output / "summary.json")


def run_live_tag_skeleton(
    output_dir: PathLike,
    *,
    seconds: float = 30.0,
    threads: int = 4,
    decimate: float = 2.0,
    preprocess: str = "cpu",
    solver_hz: int = 0,
    preview_port: int = 0,
    preview_host: str = "127.0.0.1",
    calibration: PathLike = ROOT / "configs/live",
) -> dict[str, Any]:
    """Run the bounded current-camera Tag + HumanV2 overlay test.

    This call owns the camera and both glove serial devices for ``seconds``;
    it must not run beside the legacy collector.  The returned object is the
    content of ``output_dir/summary.json`` after the camera is released.
    """

    output = _path(output_dir)
    argv = [
        sys.executable, str(ROOT / "scripts/live_tag_skeleton.py"),
        "--output", str(output), "--seconds", str(seconds), "--threads", str(threads),
        "--decimate", str(decimate), "--preprocess", preprocess,
        "--solver-hz", str(solver_hz), "--preview-port", str(preview_port),
        "--preview-host", preview_host, "--calibration", str(_path(calibration)),
    ]
    subprocess.run(argv, cwd=ROOT, env=pipeline.env_for(0, "cpu", threads), check=True)
    return _read_json(output / "summary.json")


# Names that read naturally in a frontend adapter.
run_offline = run_recording
run_replay = run_tag_skeleton
run_live = run_live_tag_skeleton

__all__ = [
    "doctor", "inspect_run", "run_recording", "run_offline", "run_tag_skeleton",
    "run_replay", "run_live_tag_skeleton", "run_live",
]
