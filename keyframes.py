"""Worker-side keyframe extraction.

The GPU worker already has the rendered video on local disk and ffmpeg is
installed (see deploy_gpu.py). Extracting the ~3 observation keyframes HERE —
instead of making the backend re-download the full clip and ffmpeg-decode it on
a tiny shared-CPU box — removes the backend's per-clip video download + decode,
which was OOM/health-check-restarting the orchestrator and does not scale.

Frames are returned inline as base64 so they ride along in the existing job
response, regardless of which of the N workers rendered the clip.
"""
from __future__ import annotations

import base64
import logging
import os
import subprocess
import tempfile
from pathlib import Path

LOGGER = logging.getLogger("filmforge.worker.keyframes")

_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".gif"}
# Same sampling ratios the backend used (≈15%, 50%, 85% of duration).
_RATIOS = (0.15, 0.5, 0.85)


def is_video_output(path: str) -> bool:
    return Path(path).suffix.lower() in _VIDEO_EXTS


def _duration_sec(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return float((out.stdout or "").strip())
    except Exception:
        return 0.0


def extract_keyframes_b64(video_path: str, max_frames: int = 3) -> list[dict]:
    """Return up to max_frames {timestamp_sec, image_b64, mime} dicts.

    Never raises — on any failure returns the frames gathered so far (possibly
    empty); the backend treats an empty list as "fall back to old path".
    """
    if not os.path.exists(video_path):
        return []
    duration = _duration_sec(video_path)
    frames: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="kf_") as tmp:
        for idx, ratio in enumerate(_RATIOS[:max_frames]):
            ts = round(duration * ratio, 3) if duration > 0 else float(idx)
            out_path = os.path.join(tmp, f"kf{idx:02d}.png")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
                     "-frames:v", "1", "-q:v", "2", out_path],
                    capture_output=True, text=True, timeout=60,
                )
            except Exception as exc:
                LOGGER.warning("keyframe ffmpeg failed idx=%d %s: %s", idx, video_path, exc)
                continue
            if os.path.exists(out_path):
                with open(out_path, "rb") as fh:
                    frames.append({
                        "timestamp_sec": ts,
                        "image_b64": base64.b64encode(fh.read()).decode("ascii"),
                        "mime": "image/png",
                    })
    return frames
