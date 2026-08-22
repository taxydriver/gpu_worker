"""Runtime guards for the proven InfiniteTalk single-speaker graph.

This module deliberately owns only the worker-side prerequisites shared by a
future compatibility render door.  It does not choose a renderer or expose a
product endpoint.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
from typing import Any
import wave
import zlib

import requests

from gpu_worker.asset_registry import get_asset_group
from gpu_worker.comfy_client import apply_comfy_input_files
from gpu_worker.config import get_settings
from gpu_worker.schemas import ComfyInputFile, InfiniteTalkTwoPersonRouting
from gpu_worker.utils import ensure_no_symlink_path


INFINITETALK_ASSET_GROUP = "infinitetalk_v1"
INFINITETALK_TWO_PERSON_ASSET_GROUP = "infinitetalk_two_person_v1"
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
A2_REQUIRED_NODE_CLASSES = REQUIRED_NODE_CLASSES | {"ImageBatch", "ImageToMask"}
_WAV2VEC_IMPORT_CHECK = "import librosa, soundfile, transformers"
_MULTITALK_MODEL = "Wan2_1-InfiniteTalk-Multi_fp16.safetensors"
_MULTITALK_MODEL_PATH = Path(
    "/workspace/ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp16.safetensors"
)
_MULTITALK_MODEL_BYTES = 5_124_439_112
_MULTITALK_MODEL_SHA256 = "4c2486cdfb6ff9a9f27408e98e11e20619136933b20411e0c365b1e84075d195"
_WAN_WRAPPER_COMMIT = "088128b224242e110d3906c6750e9a3a348a659b"
_A2_FPS = 25.0
_A2_DECLARED_DURATION_TOLERANCE_SEC = 0.050
_A2_NODE_IDS = {
    "model": "2",
    "still": "6",
    "speaker_audio": "10",
    "embeds": "11",
    "image_to_video": "12",
    "video_combine": "15",
    "listener_audio": "16",
    "slot_1_mask": "90",
    "slot_2_mask": "91",
    "slot_mask_batch": "92",
    "mask_to_tensor": "93",
    "background_mask": "94",
    "all_mask_batch": "95",
}


@dataclass(frozen=True)
class InfiniteTalkReadiness:
    """Exact worker facts needed before advertising ``infinitetalk_v1``."""

    ready: bool
    missing_files: tuple[str, ...] = ()
    missing_node_classes: tuple[str, ...] = ()
    wav2vec_dependency_error: str | None = None
    multitalk_contract_error: str | None = None
    multi_checkpoint_error: str | None = None
    comfy_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "missing_files": list(self.missing_files),
            "missing_node_classes": list(self.missing_node_classes),
            "wav2vec_dependency_error": self.wav2vec_dependency_error,
            "multitalk_contract_error": self.multitalk_contract_error,
            "multi_checkpoint_error": self.multi_checkpoint_error,
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


def _multitalk_object_contract_error(object_info: dict[str, Any]) -> str | None:
    """Verify the wrapper advertises the A2 inputs, not merely its class name."""

    node = object_info.get("MultiTalkWav2VecEmbeds")
    node_inputs = node.get("input") if isinstance(node, dict) else None
    required = node_inputs.get("required") if isinstance(node_inputs, dict) else None
    optional = node_inputs.get("optional") if isinstance(node_inputs, dict) else None
    if not isinstance(required, dict) or not isinstance(optional, dict):
        return "MultiTalkWav2VecEmbeds input schema is unavailable"
    missing = sorted({"audio_2", "ref_target_masks"} - set(optional))
    if missing:
        return "MultiTalkWav2VecEmbeds is missing optional inputs: " + ", ".join(missing)
    modes = required.get("multi_audio_type")
    choices = modes[0] if isinstance(modes, (list, tuple)) and modes else None
    if not isinstance(choices, (list, tuple)) or "para" not in choices:
        return "MultiTalkWav2VecEmbeds does not advertise para audio routing"
    return None


def _safetensors_header_is_valid(path: Path, *, exact_size: int) -> bool:
    """Cheap structural check before the expensive exact checkpoint digest."""

    try:
        with path.open("rb") as handle:
            header_size_raw = handle.read(8)
            if len(header_size_raw) != 8:
                return False
            header_size = struct.unpack("<Q", header_size_raw)[0]
            if not 2 <= header_size <= 64 * 1024 * 1024:
                return False
            header = json.loads(handle.read(header_size))
        if not isinstance(header, dict):
            return False
        tensors = [value for key, value in header.items() if key != "__metadata__"]
        if not tensors:
            return False
        data_bytes = exact_size - 8 - header_size
        for tensor in tensors:
            offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= data_bytes
            ):
                return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _multi_checkpoint_error() -> str | None:
    """Fast gate for the exact Multi checkpoint verified during asset ensure."""

    # Keep the pure graph-contract helpers importable by the backend's sibling
    # contract test; portalocker is a worker runtime dependency only.
    from gpu_worker.asset_manager import asset_checksum_is_verified

    try:
        stat = _MULTITALK_MODEL_PATH.stat()
    except OSError:
        return f"MultiTalk checkpoint is missing: {_MULTITALK_MODEL_PATH}"
    if stat.st_size != _MULTITALK_MODEL_BYTES:
        return "MultiTalk checkpoint size does not match the approved release"
    if not _safetensors_header_is_valid(
        _MULTITALK_MODEL_PATH,
        exact_size=_MULTITALK_MODEL_BYTES,
    ):
        return "MultiTalk checkpoint has an invalid safetensors header"
    if not asset_checksum_is_verified(_MULTITALK_MODEL_PATH, _MULTITALK_MODEL_SHA256):
        return "MultiTalk checkpoint has not passed exact digest verification"
    return None


def _wrapper_patch_error() -> str | None:
    """Prove the pinned wrapper and its two-speaker mask handoff are present."""

    wrapper = Path(os.getenv("COMFY_DIR", "/workspace/ComfyUI")) / "custom_nodes" / "ComfyUI-WanVideoWrapper"
    sampler = wrapper / "nodes_sampler.py"
    try:
        result = subprocess.run(
            ["git", "-C", str(wrapper), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit = result.stdout.strip() if result.returncode == 0 else ""
        source = sampler.read_text(encoding="utf-8")
    except (OSError, subprocess.SubprocessError):
        return "Pinned WanVideoWrapper runtime is unavailable"
    if commit != _WAN_WRAPPER_COMMIT:
        return "WanVideoWrapper commit does not match the approved A2 release"
    patched = '"ref_target_masks": (multitalk_embeds or {}).get("ref_target_masks", ref_target_masks)'
    old = '"ref_target_masks": ref_target_masks if multitalk_audio_embeds is not None else None,'
    if source.count(patched) != 1 or old in source:
        return "WanVideoWrapper two-speaker mask patch is not positively verified"
    return None


def check_infinitetalk_readiness(*, require_two_person: bool = False) -> InfiniteTalkReadiness:
    """Check the A1/A2 graph's actual Comfy classes, files, and wav2vec imports.

    A declared capability is intentionally insufficient: custom nodes load only
    after a Comfy restart, and their audio path runs in Comfy's venv rather than
    this worker process's venv.
    """

    missing_files = _required_files()
    multi_checkpoint_error = _multi_checkpoint_error() if require_two_person else None
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
    multitalk_contract_error: str | None = None
    comfy_error: str | None = None
    try:
        base_url = get_settings().comfy_base_url.rstrip("/")
        response = requests.get(f"{base_url}/object_info", timeout=10)
        response.raise_for_status()
        object_info = response.json()
        if not isinstance(object_info, dict):
            raise RuntimeError("ComfyUI /object_info did not return an object")
        required_nodes = A2_REQUIRED_NODE_CLASSES if require_two_person else REQUIRED_NODE_CLASSES
        missing_nodes = tuple(sorted(required_nodes - set(object_info)))
        if require_two_person and not missing_nodes:
            multitalk_contract_error = _multitalk_object_contract_error(object_info)
            multitalk_contract_error = multitalk_contract_error or _wrapper_patch_error()
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        comfy_error = str(exc)
        required_nodes = A2_REQUIRED_NODE_CLASSES if require_two_person else REQUIRED_NODE_CLASSES
        missing_nodes = tuple(sorted(required_nodes))

    return InfiniteTalkReadiness(
        ready=not (
            missing_files
            or missing_nodes
            or wav2vec_error
            or multitalk_contract_error
            or multi_checkpoint_error
            or comfy_error
        ),
        missing_files=missing_files,
        missing_node_classes=missing_nodes,
        wav2vec_dependency_error=wav2vec_error,
        multitalk_contract_error=multitalk_contract_error,
        multi_checkpoint_error=multi_checkpoint_error,
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


def normalized_audio_path_for_source(source_content_sha256: str, *, job_id: str) -> Path:
    """Return the only worker-owned WAV path allowed for a source/job pair."""

    digest = str(source_content_sha256 or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("InfiniteTalk source audio digest is invalid")
    input_root = Path(get_settings().comfy_input_dir).expanduser().resolve(strict=False)
    return input_root / "infinitetalk" / _job_scope(job_id) / f"{digest}.wav"


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

    _job_scope(job_id)  # Reject an unusable ownership scope before probing media.
    digest = _hash_approved_mpeg(source)
    destination = normalized_audio_path_for_source(digest, job_id=job_id)
    input_root = Path(get_settings().comfy_input_dir).expanduser().resolve(strict=False)
    ensure_no_symlink_path(
        input_root,
        destination,
        require_leaf=False,
        action="InfiniteTalk normalized audio",
    )
    _probe_approved_duration(source)
    if _is_valid_normalized_wav(destination):
        return destination
    _discard_invalid_wav(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ensure_no_symlink_path(
        input_root,
        destination,
        require_leaf=False,
        action="InfiniteTalk normalized audio",
    )
    temporary = destination.with_name(f".{destination.stem}.part.wav")
    ensure_no_symlink_path(
        input_root,
        temporary,
        require_leaf=False,
        action="InfiniteTalk normalized audio temporary",
    )
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
        ensure_no_symlink_path(
            input_root,
            temporary,
            require_leaf=True,
            require_regular_file=True,
            action="InfiniteTalk normalized audio temporary",
        )
        if not _is_valid_normalized_wav(temporary):
            raise RuntimeError("ffmpeg produced no bounded 16 kHz mono PCM WAV")
        temporary.replace(destination)
        ensure_no_symlink_path(
            input_root,
            destination,
            require_leaf=True,
            require_regular_file=True,
            action="InfiniteTalk normalized audio",
        )
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _graph_node_inputs(
    payload: dict[str, Any],
    node_id: str,
    class_type: str,
) -> dict[str, Any]:
    node = payload.get(node_id)
    if not isinstance(node, dict) or node.get("class_type") != class_type:
        raise ValueError(f"A2 graph requires node {node_id} as {class_type}")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"A2 graph node {node_id} has no inputs")
    return inputs


def _is_two_person_graph(payload: dict[str, Any]) -> bool:
    model = payload.get(_A2_NODE_IDS["model"])
    model_inputs = model.get("inputs") if isinstance(model, dict) else None
    image_to_video = payload.get(_A2_NODE_IDS["image_to_video"])
    image_inputs = image_to_video.get("inputs") if isinstance(image_to_video, dict) else None
    embeds = payload.get(_A2_NODE_IDS["embeds"])
    embed_inputs = embeds.get("inputs") if isinstance(embeds, dict) else None
    return bool(
        isinstance(model_inputs, dict)
        and model_inputs.get("model") == _MULTITALK_MODEL
        or isinstance(image_inputs, dict)
        and image_inputs.get("mode") == "multitalk"
        or isinstance(embed_inputs, dict)
        and embed_inputs.get("audio_2") is not None
    )


def _validate_a2_graph_shape(
    payload: dict[str, Any],
    routing: InfiniteTalkTwoPersonRouting,
) -> tuple[int, int]:
    """Require the exact explicit-mask graph and its semantic slot wiring."""

    model = _graph_node_inputs(payload, _A2_NODE_IDS["model"], "MultiTalkModelLoader")
    still = _graph_node_inputs(payload, _A2_NODE_IDS["still"], "LoadImage")
    speaker_audio = _graph_node_inputs(payload, _A2_NODE_IDS["speaker_audio"], "LoadAudio")
    listener_audio = _graph_node_inputs(payload, _A2_NODE_IDS["listener_audio"], "LoadAudio")
    embeds = _graph_node_inputs(payload, _A2_NODE_IDS["embeds"], "MultiTalkWav2VecEmbeds")
    image_to_video = _graph_node_inputs(
        payload,
        _A2_NODE_IDS["image_to_video"],
        "WanVideoImageToVideoMultiTalk",
    )
    video_combine = _graph_node_inputs(
        payload,
        _A2_NODE_IDS["video_combine"],
        "VHS_VideoCombine",
    )
    _graph_node_inputs(payload, _A2_NODE_IDS["slot_1_mask"], "LoadImage")
    _graph_node_inputs(payload, _A2_NODE_IDS["slot_2_mask"], "LoadImage")
    slot_batch = _graph_node_inputs(payload, _A2_NODE_IDS["slot_mask_batch"], "ImageBatch")
    mask_to_tensor = _graph_node_inputs(payload, _A2_NODE_IDS["mask_to_tensor"], "ImageToMask")
    _graph_node_inputs(payload, _A2_NODE_IDS["background_mask"], "LoadImage")
    all_batch = _graph_node_inputs(payload, _A2_NODE_IDS["all_mask_batch"], "ImageBatch")

    if model.get("model") != _MULTITALK_MODEL:
        raise ValueError("A2 graph must load the approved InfiniteTalk Multi checkpoint")
    if not isinstance(still.get("image"), str) or not still["image"]:
        raise ValueError("A2 graph source still is missing")
    if not isinstance(speaker_audio.get("audio"), str) or not speaker_audio["audio"]:
        raise ValueError("A2 graph approved speaker audio is missing")
    if not isinstance(listener_audio.get("audio"), str) or not listener_audio["audio"]:
        raise ValueError("A2 graph listener audio placeholder is missing")

    expected_audio = {
        f"audio_{routing.speaker_slot}": [_A2_NODE_IDS["speaker_audio"], 0],
        f"audio_{routing.listener_slot}": [_A2_NODE_IDS["listener_audio"], 0],
    }
    if any(embeds.get(name) != connection for name, connection in expected_audio.items()):
        raise ValueError("A2 graph audio slots do not match frozen speaker/listener authority")
    if any(embeds.get(name) is not None for name in ("audio_3", "audio_4")):
        raise ValueError("A2 graph must contain exactly two audio tracks")
    if embeds.get("multi_audio_type") != routing.multi_audio_type or routing.multi_audio_type != "para":
        raise ValueError("A2 graph requires parallel audio routing")
    if embeds.get("normalize_loudness") is not False:
        raise ValueError("A2 graph loudness normalization must be disabled")
    if embeds.get("add_noise_floor") not in (None, False):
        raise ValueError("A2 graph may not add a stochastic audio noise floor")
    if embeds.get("smooth_transients") not in (None, False):
        raise ValueError("A2 graph may not rewrite authored audio transients")
    if embeds.get("ref_target_masks") != [_A2_NODE_IDS["mask_to_tensor"], 0]:
        raise ValueError("A2 graph requires explicit authored target masks")
    if float(embeds.get("fps", 0.0)) != _A2_FPS:
        raise ValueError("A2 graph requires exact 25 fps audio conditioning")
    if image_to_video.get("mode") != "multitalk":
        raise ValueError("A2 graph requires multitalk sampling mode")
    if image_to_video.get("start_image") != [_A2_NODE_IDS["still"], 0]:
        raise ValueError("A2 graph start image is not the attested still")
    try:
        width = int(image_to_video["width"])
        height = int(image_to_video["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("A2 graph output dimensions are invalid") from exc
    if (width, height) != (832, 480):
        raise ValueError("A2 graph requires the proven 832x480 MultiTalk canvas")
    if float(video_combine.get("frame_rate", 0.0)) != _A2_FPS:
        raise ValueError("A2 output requires exact 25 fps")
    if video_combine.get("audio") != [_A2_NODE_IDS["speaker_audio"], 0]:
        raise ValueError("A2 output audio must remain the approved speaker take")

    if slot_batch != {
        "image1": [_A2_NODE_IDS["slot_1_mask"], 0],
        "image2": [_A2_NODE_IDS["slot_2_mask"], 0],
    }:
        raise ValueError("A2 graph mask batch must begin in slot 1, slot 2 order")
    if all_batch != {
        "image1": [_A2_NODE_IDS["slot_mask_batch"], 0],
        "image2": [_A2_NODE_IDS["background_mask"], 0],
    }:
        raise ValueError("A2 graph mask batch must append the neutral background")
    if mask_to_tensor != {
        "image": [_A2_NODE_IDS["all_mask_batch"], 0],
        "channel": "red",
    }:
        raise ValueError("A2 graph must convert the exact mask batch red channel")
    return width, height


def _png_dimensions(path: Path) -> tuple[int, int]:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError as exc:
        raise ValueError("A2 attested still could not be read") from exc
    if (
        len(header) != 24
        or not header.startswith(b"\x89PNG\r\n\x1a\n")
        or header[12:16] != b"IHDR"
    ):
        raise ValueError("A2 attested still must be a PNG with a valid IHDR")
    width, height = struct.unpack(">II", header[16:24])
    if not 1 <= width <= 8192 or not 1 <= height <= 8192:
        raise ValueError("A2 attested still dimensions are invalid")
    return width, height


def _normalized_region_to_pixels(
    region: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = region
    box = (
        max(0, math.floor(x0 * width)),
        max(0, math.floor(y0 * height)),
        min(width, math.ceil(x1 * width)),
        min(height, math.ceil(y1 * height)),
    )
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("A2 routing region became empty on the target canvas")
    return box


def _boxes_overlap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    return (
        left[0] < right[2]
        and right[0] < left[2]
        and left[1] < right[3]
        and right[1] < left[3]
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _mask_png_bytes(
    *,
    width: int,
    height: int,
    foreground: tuple[int, int, int, int] | None = None,
    background_of: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None = None,
) -> bytes:
    """Encode one deterministic 8-bit mask in the exact target pixel space."""

    rows = bytearray()
    foreground_pixels = 0
    for y in range(height):
        rows.append(0)  # PNG filter: None
        for x in range(width):
            if foreground is not None:
                value = 255 if foreground[0] <= x < foreground[2] and foreground[1] <= y < foreground[3] else 0
            elif background_of is not None:
                in_person = any(
                    box[0] <= x < box[2] and box[1] <= y < box[3]
                    for box in background_of
                )
                value = 0 if in_person else 255
            else:
                raise ValueError("A2 mask lacks an owned region")
            foreground_pixels += value == 255
            rows.append(value)
    if not 0 < foreground_pixels < width * height:
        raise ValueError("A2 masks and neutral background must each be nonempty")
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9))
        + _png_chunk(b"IEND", b"")
    )


def _pcm_wav_frame_count(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as wav:
            if (
                wav.getnchannels() != 1
                or wav.getframerate() != 16_000
                or wav.getsampwidth() != 2
                or wav.getcomptype() != "NONE"
                or wav.getnframes() <= 0
            ):
                raise ValueError("A2 speaker audio is not exact 16 kHz mono s16 PCM")
            return wav.getnframes()
    except (OSError, wave.Error) as exc:
        raise ValueError("A2 speaker audio is not a readable PCM WAV") from exc


def _silent_pcm_wav_bytes(frame_count: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\0\0" * frame_count)
    return output.getvalue()


def _observer_only(spec: ComfyInputFile) -> ComfyInputFile:
    return spec.model_copy(
        update={
            "source_data": None,
            "source_path": None,
            "source_url": None,
            "subfolder": "",
        }
    )


def prepare_two_person_routing_inputs(
    comfy_payload: dict[str, Any],
    source_specs: list[ComfyInputFile],
    effective_specs: list[ComfyInputFile],
    *,
    routing: InfiniteTalkTwoPersonRouting | None,
    job_id: str,
) -> tuple[dict[str, Any], list[ComfyInputFile]]:
    """Materialize the fail-closed A2 slot masks and exact silent track."""

    _job_scope(job_id)
    if routing is None:
        if _is_two_person_graph(comfy_payload):
            raise ValueError("Two-person MultiTalk graph requires frozen routing authority")
        return comfy_payload, effective_specs
    if not _is_two_person_graph(comfy_payload):
        raise ValueError("Frozen two-person routing requires the approved MultiTalk graph")
    checkpoint_error = _multi_checkpoint_error()
    if checkpoint_error:
        raise RuntimeError(checkpoint_error)
    wrapper_error = _wrapper_patch_error()
    if wrapper_error:
        raise RuntimeError(wrapper_error)

    expected_source_keys = {
        (_A2_NODE_IDS["still"], "image"),
        (_A2_NODE_IDS["speaker_audio"], "audio"),
    }
    observed_source_keys = [(spec.node_id, spec.input_name) for spec in source_specs]
    if len(observed_source_keys) != 2 or set(observed_source_keys) != expected_source_keys:
        raise ValueError("A2 accepts only one attested still and one approved MPEG speaker take")
    source_by_key = {(spec.node_id, spec.input_name): spec for spec in source_specs}
    still_spec = source_by_key[(_A2_NODE_IDS["still"], "image")]
    speaker_spec = source_by_key[(_A2_NODE_IDS["speaker_audio"], "audio")]
    if still_spec.content_type != "image/png" or still_spec.expected_sha256 != routing.source_still_sha256:
        raise ValueError("A2 routing authority does not match the attested source still")
    if speaker_spec.content_type != "audio/mpeg" or not speaker_spec.expected_sha256:
        raise ValueError("A2 speaker authority must be one digest-attested audio/mpeg take")

    width, height = _validate_a2_graph_shape(comfy_payload, routing)
    input_root = Path(get_settings().comfy_input_dir).expanduser().resolve(strict=False)
    still_name = comfy_payload[_A2_NODE_IDS["still"]]["inputs"]["image"]
    still_path = ensure_no_symlink_path(
        input_root,
        input_root / still_name,
        require_leaf=True,
        require_regular_file=True,
        action="A2 attested still",
    )
    observed_dimensions = _png_dimensions(still_path)
    expected_dimensions = (
        routing.source_dimensions.width,
        routing.source_dimensions.height,
    )
    if observed_dimensions != expected_dimensions:
        raise ValueError("A2 routing authority dimensions do not match the attested still")

    effective_by_key = {(spec.node_id, spec.input_name): spec for spec in effective_specs}
    speaker_wav_spec = effective_by_key.get((_A2_NODE_IDS["speaker_audio"], "audio"))
    if speaker_wav_spec is None or speaker_wav_spec.content_type != "audio/wav":
        raise ValueError("A2 approved speaker take was not normalized and attested")
    speaker_wav_name = comfy_payload[_A2_NODE_IDS["speaker_audio"]]["inputs"]["audio"]
    speaker_wav_path = ensure_no_symlink_path(
        input_root,
        input_root / speaker_wav_name,
        require_leaf=True,
        require_regular_file=True,
        action="A2 normalized speaker audio",
    )
    frame_count = _pcm_wav_frame_count(speaker_wav_path)
    duration_sec = frame_count / 16_000
    # Backend probe authority is rounded to milliseconds and MPEG decoding may
    # expose bounded padding.  The listener is still derived from the exact
    # normalized PCM frame count below; this tolerance is only for the frozen
    # metadata cross-check.
    if abs(duration_sec - routing.expected_duration_sec) > _A2_DECLARED_DURATION_TOLERANCE_SEC:
        raise ValueError("A2 normalized speaker duration does not match frozen authority")
    # The backend authored the graph from its frozen, millisecond-rounded
    # duration. Validate that same authority here; using decoded padding for
    # this guard can cross a 4n+1 bucket and reject an otherwise valid take.
    raw_frames = max(1, math.ceil(routing.expected_duration_sec * _A2_FPS))
    expected_video_frames = raw_frames + ((1 - raw_frames) % 4)
    embeds = comfy_payload[_A2_NODE_IDS["embeds"]]["inputs"]
    if embeds.get("num_frames") != expected_video_frames:
        raise ValueError("A2 graph frame guard does not match the exact speaker duration")

    slot_1_box = _normalized_region_to_pixels(
        routing.slot_regions[0],
        width=width,
        height=height,
    )
    slot_2_box = _normalized_region_to_pixels(
        routing.slot_regions[1],
        width=width,
        height=height,
    )
    if _boxes_overlap(slot_1_box, slot_2_box):
        raise ValueError("A2 routing regions overlap after target-canvas transform")
    slot_1_mask = _mask_png_bytes(width=width, height=height, foreground=slot_1_box)
    slot_2_mask = _mask_png_bytes(width=width, height=height, foreground=slot_2_box)
    background_mask = _mask_png_bytes(
        width=width,
        height=height,
        background_of=(slot_1_box, slot_2_box),
    )
    silence_wav = _silent_pcm_wav_bytes(frame_count)

    generated: list[ComfyInputFile] = []
    for node_id, filename, data, content_type, input_name in (
        (_A2_NODE_IDS["listener_audio"], "listener_silence.wav", silence_wav, "audio/wav", "audio"),
        (_A2_NODE_IDS["slot_1_mask"], "slot_1_mask.png", slot_1_mask, "image/png", "image"),
        (_A2_NODE_IDS["slot_2_mask"], "slot_2_mask.png", slot_2_mask, "image/png", "image"),
        (_A2_NODE_IDS["background_mask"], "background_mask.png", background_mask, "image/png", "image"),
    ):
        generated.append(
            ComfyInputFile(
                node_id=node_id,
                input_name=input_name,
                filename=filename,
                source_data=base64.b64encode(data).decode("ascii"),
                expected_sha256=hashlib.sha256(data).hexdigest(),
                content_type=content_type,
            )
        )
    prepared = apply_comfy_input_files(comfy_payload, generated)
    return prepared, [*effective_specs, *(_observer_only(spec) for spec in generated)]


def build_two_person_routing_receipt(
    routing: InfiniteTalkTwoPersonRouting | None,
    staged_receipts: list[dict[str, str]],
) -> dict[str, Any] | None:
    """Bind the frozen authority to the three effective mask observations."""

    if routing is None:
        return None
    by_key = {
        (str(receipt.get("node_id")), str(receipt.get("input_name"))): str(
            receipt.get("content_sha256") or ""
        )
        for receipt in staged_receipts
    }
    mask_sha256 = {
        "slot_1": by_key.get((_A2_NODE_IDS["slot_1_mask"], "image"), ""),
        "slot_2": by_key.get((_A2_NODE_IDS["slot_2_mask"], "image"), ""),
        "background": by_key.get((_A2_NODE_IDS["background_mask"], "image"), ""),
    }
    if any(
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in mask_sha256.values()
    ):
        raise RuntimeError("A2 effective mask receipts are incomplete")
    return {
        "schema_version": "infinitetalk_two_person_routing_receipt_v1",
        "spatial_authority_sha256": routing.spatial_authority_sha256,
        "source_still_sha256": routing.source_still_sha256,
        "speaker_slot": routing.speaker_slot,
        "listener_slot": routing.listener_slot,
        "mode": routing.mode,
        "multi_audio_type": routing.multi_audio_type,
        "mask_sha256": mask_sha256,
    }
