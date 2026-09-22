"""Identity department service: ArcFace scoring + person masks for the shot loop.

The heavy work runs in ``identity_runner.py`` under the SEPARATE identity ComfyUI's Python
(insightface / ultralytics are installed there, never in the worker or production venv).
One subprocess per request, so nothing stays resident between calls; callers score all of a
take's frames in ONE request.

GPU safety (lead ruling 2026-09-22): the per-GPU worker unit is already pinned with
CUDA_VISIBLE_DEVICES=<idx>; the subprocess inherits that pin explicitly, and a worker that
does not advertise ``identity_v1`` refuses, so the production card's worker can never run
this. LICENCES: insightface models non-commercial; ultralytics AGPL-3.0 -- private films only.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

IDENTITY_ASSET_GROUP = "identity_v1"
LICENCE = "non-commercial (insightface models); AGPL-3.0 (ultralytics)"
_RUNNER = Path(__file__).resolve().parent / "identity_runner.py"


def identity_root() -> Path:
    return Path(os.getenv("IDENTITY_COMFY_ROOT", "/workspace/ComfyUI_identity"))


def identity_python() -> Path:
    return identity_root() / ".venv" / "bin" / "python"


def models_dir() -> Path:
    return identity_root() / "models"


class IdentityScoreRequest(BaseModel):
    truth_url: str
    image_urls: list[str] = Field(default_factory=list)
    video_url: str | None = None
    threshold: float = 0.50
    min_face_px: int = 40
    sample_fps: float = 2.0
    max_frames: int = Field(default=48, le=240)
    person_id: str | None = None
    truth_asset_id: str | None = None


class SegmentPeopleRequest(BaseModel):
    video_url: str
    roi: list[int] | None = None
    n: int = 2
    dilate_px: int = 25


def identity_readiness(declared_capabilities: list[str]) -> dict[str, Any]:
    """Advertise identity_v1 only when it is DECLARED and actually installed.

    A homogeneous worker with no WORKER_CAPABILITIES advertises every registry group;
    without this gate it would claim identity_v1 on a box with no identity ComfyUI.
    """
    reasons = []
    if IDENTITY_ASSET_GROUP not in declared_capabilities and "identity" not in declared_capabilities:
        reasons.append("not_declared")
    if not identity_python().exists():
        reasons.append("identity_venv_missing")
    nodes = identity_root() / "custom_nodes"
    for node in ("ComfyUI_InfiniteYou", "ComfyUI-PuLID-Flux"):
        if not (nodes / node).is_dir():
            reasons.append(f"node_missing:{node}")
    if not (models_dir() / "insightface" / "models" / "buffalo_l" / "w600k_r50.onnx").exists():
        reasons.append("scorer_weights_missing")
    return {"ready": not reasons, "reasons": reasons, "licence": LICENCE}


def runner_env() -> dict[str, str]:
    """Environment for the runner: the worker's own GPU pin, re-stated, never widened."""
    pin = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not pin:
        raise RuntimeError("identity runner refused: worker is not pinned (CUDA_VISIBLE_DEVICES unset)")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV"))}
    env["CUDA_VISIBLE_DEVICES"] = pin
    return env


def run_runner(request: dict[str, Any], *, timeout_sec: int = 900) -> dict[str, Any]:
    proc = subprocess.run(
        [str(identity_python()), str(_RUNNER)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        env=runner_env(),
        timeout=timeout_sec,
    )
    try:
        reply = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        reply = {}
    if proc.returncode not in (0, 2) or not reply:
        raise RuntimeError(f"identity runner failed: {(proc.stderr or proc.stdout).strip()[-800:]}")
    return reply


def score(req: IdentityScoreRequest, fetch) -> dict[str, Any]:
    """``fetch(url, dest_path, kind)`` stages one allowlisted input (image|video)."""
    if not req.image_urls and not req.video_url:
        raise ValueError("identity score needs image_urls or video_url")
    with tempfile.TemporaryDirectory(prefix="identity_score_") as tmp:
        truth = Path(tmp) / "truth.png"
        fetch(req.truth_url, truth, "image")
        runner_req: dict[str, Any] = {
            "mode": "score",
            "models_root": str(models_dir() / "insightface"),
            "anchor": str(truth),
            "threshold": req.threshold,
            "min_face_px": req.min_face_px,
            "sample_fps": req.sample_fps,
            "max_frames": req.max_frames,
        }
        if req.video_url:
            video = Path(tmp) / "take.mp4"
            fetch(req.video_url, video, "video")
            runner_req["video"] = str(video)
        else:
            paths = []
            for i, url in enumerate(req.image_urls):
                p = Path(tmp) / f"take_{i}.png"
                fetch(url, p, "image")
                paths.append(str(p))
            runner_req["images"] = paths
        reply = run_runner(runner_req)
    reply["anchor"] = {"person_id": req.person_id, "asset_id": req.truth_asset_id}
    return reply


def segment_people(req: SegmentPeopleRequest, fetch, output_dir: Path, job_id: str) -> dict[str, Any]:
    out_dir = output_dir / "identity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{job_id}_mask.mp4"
    with tempfile.TemporaryDirectory(prefix="identity_seg_") as tmp:
        video = Path(tmp) / "in.mp4"
        fetch(req.video_url, video, "video")
        reply = run_runner({
            "mode": "segment",
            "video": str(video),
            "out": str(out),
            "yolo": str(models_dir() / "ultralytics" / "segm" / "yolov8x-seg.pt"),
            "roi": req.roi,
            "n": req.n,
            "dilate_px": req.dilate_px,
        }, timeout_sec=1800)
    reply["out"] = f"identity/{out.name}"  # served under /files/output/
    return reply
