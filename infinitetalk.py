"""Runtime guards for the proven InfiniteTalk single-speaker graph.

This module deliberately owns only the worker-side prerequisites shared by a
future compatibility render door.  It does not choose a renderer or expose a
product endpoint.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import subprocess
from typing import Any
import wave

import requests

from gpu_worker.asset_registry import get_asset_group
from gpu_worker.config import get_settings


INFINITETALK_ASSET_GROUP = "infinitetalk_v1"
# This is intentionally a very small, compatibility-proof-only input contract:
# one spoken line no longer than roughly 15 seconds.  Keeping the limits here
# (rather than trusting ffmpeg) prevents a broker-approved mp3 from consuming
# unbounded disk or CPU on a worker.
MAX_APPROVED_MPEG_BYTES = 2 * 1024 * 1024
MAX_APPROVED_DURATION_SECONDS = 15.0
MAX_NORMALIZED_WAV_BYTES = 512 * 1024
# ``job_`` plus unpadded base64 expands 188 UTF-8 bytes to exactly 255 ASCII
# bytes (the portable filesystem component limit); byte 189 would make 256.
MAX_JOB_SCOPE_COMPONENT_BYTES = 255
MAX_JOB_ID_UTF8_BYTES = 188
_HASH_CHUNK_BYTES = 1024 * 1024
REQUIRED_NODE_CLASSES = frozenset({
    "WanVideoLoraSelect",
    "MultiTalkModelLoader",
    "WanVideoModelLoader",
    "WanVideoVAELoader",
    "WanVideoTextEncodeCached",
    "LoadImage",
    "CLIPVisionLoader",
    "WanVideoClipVisionEncode",
    "DownloadAndLoadWav2VecModel",
    "LoadAudio",
    "MultiTalkWav2VecEmbeds",
    "WanVideoImageToVideoMultiTalk",
    "WanVideoSampler",
    "WanVideoDecode",
    "VHS_VideoCombine",
})
_WAV2VEC_IMPORT_CHECK = "import librosa, soundfile, transformers"


@dataclass(frozen=True)
class InfiniteTalkReadiness:
    """Exact worker facts needed before advertising ``infinitetalk_v1``."""

    ready: bool
    missing_files: tuple[str, ...] = ()
    missing_node_classes: tuple[str, ...] = ()
    wav2vec_dependency_error: str | None = None
    comfy_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "missing_files": list(self.missing_files),
            "missing_node_classes": list(self.missing_node_classes),
            "wav2vec_dependency_error": self.wav2vec_dependency_error,
            "comfy_error": self.comfy_error,
        }


def _comfy_python() -> Path:
    return Path(os.getenv("COMFY_DIR", "/workspace/ComfyUI")) / ".venv" / "bin" / "python"


def _required_files() -> tuple[str, ...]:
    return tuple(
        asset["path"]
        for asset in get_asset_group(INFINITETALK_ASSET_GROUP)
        if not Path(asset["path"]).expanduser().is_file()
        or Path(asset["path"]).expanduser().stat().st_size <= 0
    )


def check_infinitetalk_readiness() -> InfiniteTalkReadiness:
    """Check the A1 graph's actual Comfy classes, files, and wav2vec imports.

    A declared capability is intentionally insufficient: custom nodes load only
    after a Comfy restart, and their audio path runs in Comfy's venv rather than
    this worker process's venv.
    """

    missing_files = _required_files()
    python = _comfy_python()
    wav2vec_error: str | None = None
    if not python.is_file():
        wav2vec_error = f"ComfyUI Python missing: {python}"
    else:
        try:
            result = subprocess.run(
                [str(python), "-c", _WAV2VEC_IMPORT_CHECK],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if result.returncode:
                wav2vec_error = (result.stderr or result.stdout or "wav2vec dependency import failed").strip()[:500]
        except (OSError, subprocess.SubprocessError) as exc:
            wav2vec_error = str(exc)

    missing_nodes: tuple[str, ...] = ()
    comfy_error: str | None = None
    try:
        base_url = get_settings().comfy_base_url.rstrip("/")
        response = requests.get(f"{base_url}/object_info", timeout=10)
        response.raise_for_status()
        object_info = response.json()
        if not isinstance(object_info, dict):
            raise RuntimeError("ComfyUI /object_info did not return an object")
        missing_nodes = tuple(sorted(REQUIRED_NODE_CLASSES - set(object_info)))
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        comfy_error = str(exc)
        missing_nodes = tuple(sorted(REQUIRED_NODE_CLASSES))

    return InfiniteTalkReadiness(
        ready=not (missing_files or missing_nodes or wav2vec_error or comfy_error),
        missing_files=missing_files,
        missing_node_classes=missing_nodes,
        wav2vec_dependency_error=wav2vec_error,
        comfy_error=comfy_error,
    )


def _hash_approved_mpeg(source: Path) -> str:
    """Return a streaming digest, rejecting sources outside the proof budget."""

    digest = hashlib.sha256()
    total_bytes = 0
    with source.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            total_bytes += len(chunk)
            if total_bytes > MAX_APPROVED_MPEG_BYTES:
                raise ValueError(
                    "InfiniteTalk audio exceeds the "
                    f"{MAX_APPROVED_MPEG_BYTES}-byte compatibility-proof limit"
                )
            digest.update(chunk)
    if total_bytes <= 0:
        raise ValueError(f"InfiniteTalk audio source is missing or empty: {source}")
    return digest.hexdigest()


def _job_scope(job_id: str) -> str:
    """Losslessly encode a bounded job id for a single job-owned directory."""

    if not isinstance(job_id, str) or not job_id:
        raise ValueError("InfiniteTalk audio staging needs a non-empty job_id")
    encoded = job_id.encode("utf-8")
    if len(encoded) > MAX_JOB_ID_UTF8_BYTES:
        raise ValueError(
            "InfiniteTalk audio staging job_id exceeds the "
            f"{MAX_JOB_ID_UTF8_BYTES}-byte limit"
        )
    # URL-safe base64 is a lossless mapping.  In particular, job/one and
    # job:one cannot collapse to one mutable staging directory.  Keep the
    # explicit encoded check as a guard if either the prefix or encoding changes.
    scope = "job_" + base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")
    if len(scope.encode("ascii")) > MAX_JOB_SCOPE_COMPONENT_BYTES:
        raise ValueError(
            "InfiniteTalk audio staging job_id exceeds the portable "
            f"{MAX_JOB_SCOPE_COMPONENT_BYTES}-byte path-component limit"
        )
    return scope


def _probe_approved_duration(source: Path) -> float:
    """Require a finite decoded duration within the one-line proof envelope."""

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nokey=1:noprint_wrappers=1", str(source),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not inspect InfiniteTalk audio duration: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout or "ffprobe failed").strip()[:500]
        raise ValueError(f"InfiniteTalk audio duration could not be read: {detail}")
    try:
        duration = float(result.stdout.strip())
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("InfiniteTalk audio duration is invalid") from exc
    if not math.isfinite(duration) or duration <= 0 or duration > MAX_APPROVED_DURATION_SECONDS:
        raise ValueError(
            "InfiniteTalk audio must be a single line between 0 and "
            f"{MAX_APPROVED_DURATION_SECONDS:g} seconds"
        )
    return duration


def _is_valid_normalized_wav(path: Path) -> bool:
    """Validate the exact Comfy input contract before a file is reused."""

    try:
        size = path.stat().st_size
        if size <= 44 or size > MAX_NORMALIZED_WAV_BYTES:
            return False
        with path.open("rb") as handle:
            if handle.read(4) != b"RIFF":
                return False
            handle.read(4)  # RIFF chunk size
            if handle.read(4) != b"WAVE":
                return False
        with wave.open(str(path), "rb") as wav:
            if (
                wav.getnchannels() != 1
                or wav.getframerate() != 16_000
                or wav.getsampwidth() != 2
                or wav.getcomptype() != "NONE"
                or wav.getnframes() <= 0
                or wav.getnframes() / wav.getframerate() > MAX_APPROVED_DURATION_SECONDS
            ):
                return False
    except (OSError, wave.Error):
        return False
    return True


def _discard_invalid_wav(path: Path) -> None:
    if path.exists() and not _is_valid_normalized_wav(path):
        path.unlink()


def normalize_approved_mpeg_to_wav(source: Path, *, job_id: str) -> Path:
    """Convert an approved MPEG take to deterministic 16 kHz mono PCM WAV.

    The source digest, rather than a caller filename, supplies the staging name;
    a losslessly encoded, bounded job id keeps distinct jobs from sharing a
    mutable input path.
    """

    if source.suffix.lower() != ".mp3":
        raise ValueError("InfiniteTalk accepts approved audio/mpeg (.mp3) input only")
    if not source.is_file():
        raise ValueError(f"InfiniteTalk audio source is missing or empty: {source}")

    job_scope = _job_scope(job_id)
    digest = _hash_approved_mpeg(source)
    _probe_approved_duration(source)
    input_root = Path(get_settings().comfy_input_dir).expanduser()
    destination = input_root / "infinitetalk" / job_scope / f"{digest}.wav"
    if _is_valid_normalized_wav(destination):
        return destination
    _discard_invalid_wav(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.part.wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
                "-map_metadata", "-1", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", "-bitexact", str(temporary),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        if not _is_valid_normalized_wav(temporary):
            raise RuntimeError("ffmpeg produced no bounded 16 kHz mono PCM WAV")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
