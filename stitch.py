"""Worker-side rough-cut / trailer stitching.

The GPU box has ffmpeg installed and idle CPU/RAM while the GPU generates. Doing
the stitch HERE — fetch the rendered clips by URL, normalize, concat, mux the
score, and upload the finished video straight to storage — keeps the 2 GB API
backend out of the video path entirely: no clip downloads, no ffmpeg, no video
bytes in its RAM (the thing that hung the backend at "Stitching rough cut").

Credentials stay off this (ephemeral, replaceable) box: the backend mints a
Supabase **signed upload URL** for a deterministic output path and passes it in;
the worker PUTs the result there with no storage key of its own.

Self-contained, like keyframes.py: pure functions + subprocess ffmpeg. The
ffmpeg recipe is ported verbatim from the backend's `_stitch_trailer_with_ffmpeg`
(`app/trailer/worker.py`) + `_build_normalize_clip_command` /
`_sanitize_trim_window` (`app/render/primitives.py`) so output is byte-compatible.

NOTE: v1 handles video concat + score-only audio mux (the common rough-cut case).
Foley/diegetic mixing is deferred — the backend pre-resolves the diegetic plan to
an ffmpeg filter spec and will pass it through here in a follow-up.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import urllib.request
from typing import Any, Optional

import httpx

LOGGER = logging.getLogger("filmforge.worker.stitch")


def _download(url: str, dest: str) -> None:
    urllib.request.urlretrieve(url, dest)


def _probe_duration_sec(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return float((out.stdout or "").strip())
    except Exception:
        return 0.0


def _sanitize_trim(
    edit_in_sec: Optional[float],
    edit_out_sec: Optional[float],
    actual_duration_sec: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    """Ported from backend `_sanitize_trim_window`."""
    if edit_in_sec is None or edit_out_sec is None:
        return None, None
    start = max(0.0, float(edit_in_sec))
    end = float(edit_out_sec)
    if actual_duration_sec is not None and actual_duration_sec > 0:
        start = min(start, actual_duration_sec)
        end = min(end, actual_duration_sec)
    if end <= start:
        return None, None
    return round(start, 3), round(end, 3)


def _cadence_filter_chain(fps: int) -> str:
    """The Director's approved conform chain (backend `render/cadence.py`).

    WAN 2.2 I2V generates at 16 fps. A bare `-r 24` does not resample motion, it
    DUPLICATES one frame in three — measured at 32% frozen frames in the
    delivered cut, with a cadence less regular than the 16 fps source. The fix
    interpolates to 48 with motion compensation, averages adjacent frame pairs
    for a synthetic 180 degree shutter (2 frames of 48 = 1/48 s of smear inside
    a 1/24 s frame), then decimates to `fps`. The shutter blur is part of the
    ruling, not an embellishment.

    Kept byte-compatible with the backend by hand, like the rest of this file.
    G46 is still open: nothing upstream can author a cadence, so `fps` still
    arrives as the request default.
    """
    return (
        f"minterpolate=fps={fps * 2}:mi_mode=mci:me_mode=bidir:mc_mode=aobmc:vsbmc=1,"
        f"tmix=frames=2:weights='1 1',"
        f"fps={fps}"
    )


def _normalize_cmd(
    raw: str, norm: str, w: int, h: int, fps: int,
    trim_start: Optional[float] = None, trim_dur: Optional[float] = None,
) -> list[str]:
    """Ported from backend `_build_normalize_clip_command`: scale+pad to target,
    conform to the delivery cadence, strip audio, re-encode to a concat-friendly
    H.264.

    Per clip, before concat: interpolating across a concat seam would invent
    motion between two unrelated shots.
    """
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
        f"{_cadence_filter_chain(fps)}"
    )
    cmd = ["ffmpeg", "-y", "-i", raw]
    if trim_start is not None and trim_dur is not None:
        cmd += ["-ss", f"{trim_start:.3f}", "-t", f"{trim_dur:.3f}"]
    cmd += ["-vf", vf, "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-pix_fmt", "yuv420p", norm]
    return cmd


def stitch_clips(
    *,
    clips: list[dict[str, Any]],
    audio_url: Optional[str],
    width: int,
    height: int,
    fps: int,
    work_dir: str,
) -> tuple[str, dict[str, Any]]:
    """Fetch clips by URL, normalize each, concat, mux the score.

    `clips` is the ordered list of {url, edit_in_sec, edit_out_sec,
    target_duration_sec, source_duration_sec}. Returns (output_path, metadata).
    Raises on hard ffmpeg failure (the backend keeps its local fallback until
    this path is proven, per ADR-0002).
    """
    if not clips:
        raise RuntimeError("stitch: no clips provided")

    local_clips: list[str] = []
    selected = 0
    actual_total = 0.0

    for i, entry in enumerate(clips):
        url = str(entry.get("url") or "").strip()
        if not url:
            continue
        raw = os.path.join(work_dir, f"clip_{i:03d}_raw.mp4")
        norm = os.path.join(work_dir, f"clip_{i:03d}.mp4")
        _download(url, raw)

        dur = _probe_duration_sec(raw)
        trim_start, trim_end = _sanitize_trim(
            entry.get("edit_in_sec"), entry.get("edit_out_sec"), dur,
        )
        trim_dur = round(trim_end - trim_start, 3) if (trim_start is not None and trim_end is not None) else None

        try:
            subprocess.run(_normalize_cmd(raw, norm, width, height, fps, trim_start, trim_dur),
                           check=True, capture_output=True)
        except Exception as exc:
            # Trim can fail on odd keyframe boundaries — fall back to the full clip.
            if trim_dur is not None:
                LOGGER.warning("stitch: trim failed clip=%d, using full clip: %s", i, exc)
                subprocess.run(_normalize_cmd(raw, norm, width, height, fps),
                               check=True, capture_output=True)
                trim_dur = None
            else:
                raise
        local_clips.append(norm)
        selected += 1
        actual_total += float(trim_dur or dur or entry.get("source_duration_sec") or entry.get("target_duration_sec") or 0.0)

    if not local_clips:
        raise RuntimeError("stitch: no usable clips after fetch/normalize")

    manifest = os.path.join(work_dir, "concat.txt")
    with open(manifest, "w", encoding="utf-8") as fh:
        for clip in local_clips:
            safe = clip.replace("'", "'\\''")
            fh.write(f"file '{safe}'\n")

    no_audio = os.path.join(work_dir, "video_no_audio.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", manifest,
         "-an", "-c:v", "copy", no_audio],
        check=True, capture_output=True,
    )

    meta = {
        "selected_shots_count": selected,
        "actual_assembled_duration_sec": round(actual_total, 3),
    }

    if not audio_url:
        meta.update({"audio_attached": False, "audio_mode": "none"})
        return no_audio, meta

    music = os.path.join(work_dir, "music.mp3")
    _download(audio_url, music)
    final = os.path.join(work_dir, "stitched_final.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", no_audio, "-i", music,
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
         "-c:a", "aac", "-shortest", final],
        check=True, capture_output=True,
    )
    meta.update({"audio_attached": True, "audio_mode": "score"})
    return final, meta


def upload_via_signed_put(signed_put_url: str, file_path: str, content_type: str = "video/mp4") -> None:
    """PUT the finished file to a Supabase signed upload URL.

    No storage credentials needed on this box — the backend minted the signed URL
    for a known path. Raises on non-2xx so the backend can fall back / retry.
    """
    with open(file_path, "rb") as fh:
        data = fh.read()
    resp = httpx.put(signed_put_url, content=data, headers={"content-type": content_type}, timeout=600.0)
    resp.raise_for_status()
