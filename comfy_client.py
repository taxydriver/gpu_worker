"""HTTP client helpers for local ComfyUI."""

from __future__ import annotations

from copy import deepcopy
import logging
import os
from pathlib import Path
import shutil
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, urlparse

import requests

from gpu_worker.config import get_settings
from gpu_worker.schemas import ComfyInputFile, OutputFile
from gpu_worker.utils import is_non_empty_file, safe_unlink


LOGGER = logging.getLogger(__name__)
PROMPT_LOGGER = logging.getLogger("gpu_worker.prompts")

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _comfy_url(path: str) -> str:
    """Build a ComfyUI URL from a relative API path."""

    return f"{get_settings().comfy_base_url.rstrip('/')}{path}"


def comfy_queue_depth() -> dict[str, int] | None:
    """Read ComfyUI's live queue depth — ground truth for whether the GPU is
    generating right now.

    ``queue_running`` is what ComfyUI is executing this instant (the same engine
    that prints "got prompt" / "Prompt executed in N seconds"); ``queue_pending``
    is what's waiting. Returns None when ComfyUI is unreachable so callers can
    distinguish "idle" from "unknown".
    """

    try:
        response = requests.get(_comfy_url("/queue"), timeout=3)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return {
        "running": len(data.get("queue_running") or []),
        "pending": len(data.get("queue_pending") or []),
    }


def served_file_roots() -> dict[str, Path]:
    """Return the configured roots the worker is allowed to expose."""

    return get_settings().served_file_roots()


def comfy_input_dir() -> Path:
    """Return the local ComfyUI input directory."""

    return Path(get_settings().comfy_input_dir).expanduser().resolve(strict=False)


def resolve_served_file(root_name: str, relative_path: str) -> Path:
    """Resolve a download request to a safe file path under an allowed root."""

    root = served_file_roots().get(root_name)
    if root is None:
        raise ValueError(f"Unknown file root: {root_name}")

    candidate_rel = Path(relative_path)
    if candidate_rel.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {relative_path}")

    candidate = (root / candidate_rel).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes served root {root_name}: {relative_path}") from exc
    return candidate


def _log_payload_prompts(comfy_payload: dict[str, Any]) -> None:
    """Extract and log all text prompts from a ComfyUI payload graph."""

    TEXT_FIELDS = ("text", "prompt", "positive", "negative", "negative_prompt")
    for node_id, node in comfy_payload.items():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            continue
        for field in TEXT_FIELDS:
            value = inputs.get(field)
            if isinstance(value, str) and value.strip():
                PROMPT_LOGGER.info("node=%s field=%s | %s", node_id, field, value)


def free_comfy_memory(*, unload_models: bool = True) -> None:
    """Ask ComfyUI to release cached VRAM between jobs.

    Calls POST /free which flushes the model cache and torch CUDA allocator.
    Safe to call after every job — prevents VRAM fragmentation OOM over time.
    """
    try:
        response = requests.post(
            _comfy_url("/free"),
            json={"unload_models": unload_models, "free_memory": True},
            timeout=10,
        )
        if response.ok:
            LOGGER.info("ComfyUI /free: VRAM released (unload_models=%s)", unload_models)
        else:
            LOGGER.warning("ComfyUI /free returned HTTP %s", response.status_code)
    except requests.RequestException as exc:
        LOGGER.warning("ComfyUI /free failed (non-fatal): %s", exc)


def submit_prompt(comfy_payload: dict[str, Any]) -> str:
    """Submit a prompt graph to ComfyUI and return the prompt id."""

    _log_payload_prompts(comfy_payload)
    response = requests.post(
        _comfy_url("/prompt"),
        json={"prompt": comfy_payload, "client_id": "filmforge-gpu-worker"},
        timeout=30,
    )
    if not response.ok:
        LOGGER.error(
            "ComfyUI /prompt HTTP %s: %s", response.status_code, response.text[:2000]
        )
    response.raise_for_status()
    payload = response.json()
    prompt_id = payload.get("prompt_id") or payload.get("promptId")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI /prompt response missing prompt_id: {payload}")
    return str(prompt_id)


class ComfyExecutionError(RuntimeError):
    """Raised when ComfyUI reports an execution error (e.g. OOM) in prompt history."""

    def __init__(self, message: str, *, is_oom: bool = False):
        super().__init__(message)
        self.is_oom = is_oom


class ComfyRestartDetectedError(RuntimeError):
    """Raised when ComfyUI was restarted and the prompt was lost from history."""

    def __init__(self, message: str):
        super().__init__(message)


def _history_entry_complete(entry: dict[str, Any]) -> bool:
    """Return True when a history entry looks complete (success or error)."""

    outputs = entry.get("outputs")
    if isinstance(outputs, dict) and outputs:
        return True

    status = entry.get("status")
    if isinstance(status, dict):
        if status.get("completed") is True:
            return True
        if status.get("status_str") == "error":
            return True

    return False


def _check_history_for_errors(history: dict[str, Any]) -> None:
    """Raise ComfyExecutionError if the history entry contains an execution error."""

    status = history.get("status")
    if not isinstance(status, dict):
        return

    if status.get("status_str") != "error":
        return

    # Walk messages to find the execution_error entry
    for item in status.get("messages", []):
        if not (isinstance(item, (list, tuple)) and len(item) >= 2):
            continue
        msg_type, msg_data = item[0], item[1]
        if msg_type == "execution_error" and isinstance(msg_data, dict):
            exc_type = msg_data.get("exception_type", "")
            exc_msg = msg_data.get("exception_message", str(msg_data))
            is_oom = "OutOfMemory" in exc_type or "OutOfMemory" in exc_msg
            raise ComfyExecutionError(
                f"ComfyUI execution error — {exc_type}: {exc_msg}",
                is_oom=is_oom,
            )

    raise ComfyExecutionError("ComfyUI reported execution error with no detail")


def _check_comfy_restarted(
    prompt_id: str,
    payload: dict[str, Any],
    elapsed_sec: float,
) -> None:
    """Detect if ComfyUI was restarted by checking if the prompt_id is missing from history
    and the queue is empty (no running or pending prompts).

    A grace period of 10 seconds is given to allow the prompt to appear in the queue
    after submission (e.g. after a ComfyUI restart).
    """

    # If the prompt_id is not in history at all, check if ComfyUI has an empty queue
    # which means the prompt was lost due to a restart
    if prompt_id in payload:
        return  # prompt is still in history, no restart detected

    # Give a grace period for the prompt to appear in the queue after submission
    if elapsed_sec < 10.0:
        return

    # Check if the queue is empty (no running or pending prompts).
    # An empty history response ({}) means the prompt hasn't finished yet,
    # but we need to check the queue to distinguish between:
    #   - A restart (empty history + empty queue)
    #   - A prompt still being processed (empty history + non-empty queue)
    try:
        queue_response = requests.get(_comfy_url("/queue"), timeout=10)
        if queue_response.ok:
            queue_data = queue_response.json()
            queue_running = queue_data.get("queue_running", [])
            queue_pending = queue_data.get("queue_pending", [])
            if not queue_running and not queue_pending:
                raise ComfyRestartDetectedError(
                    f"ComfyUI queue is empty and prompt_id={prompt_id} not found in history "
                    f"— ComfyUI was likely restarted and the prompt was lost"
                )
            # Queue has items — prompt is still being processed, not a restart
            return
    except ComfyRestartDetectedError:
        raise
    except requests.RequestException:
        pass  # if we can't check the queue, fall through to normal polling

    # If we couldn't check the queue, fall back to the empty history check
    if not payload:
        raise ComfyRestartDetectedError(
            f"ComfyUI history is empty — ComfyUI was restarted and prompt_id={prompt_id} was lost"
        )


def poll_for_completion(
    prompt_id: str,
    timeout_sec: int,
    poll_interval_sec: float,
    *,
    progress_callback: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Poll ComfyUI history until a prompt appears complete or errors out.

    Raises ComfyRestartDetectedError if ComfyUI was restarted and the prompt
    was lost from history (empty history + empty queue).
    """

    deadline = time.monotonic() + timeout_sec
    last_payload: Any = None
    started_at = time.monotonic()
    last_log_at = started_at

    while time.monotonic() < deadline:
        now = time.monotonic()
        if progress_callback is not None:
            try:
                progress_callback(max(now - started_at, 0.0))
            except Exception:
                LOGGER.debug("Progress callback failed for prompt_id=%s", prompt_id, exc_info=True)
        try:
            response = requests.get(
                _comfy_url(f"/history/{prompt_id}"),
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            last_payload = str(exc)
            time.sleep(poll_interval_sec)
            continue

        last_payload = payload
        if not isinstance(payload, dict):
            time.sleep(poll_interval_sec)
            continue

        # Detect if ComfyUI was restarted (prompt lost from history)
        elapsed = now - started_at
        try:
            _check_comfy_restarted(prompt_id, payload, elapsed)
        except ComfyRestartDetectedError:
            LOGGER.error("ComfyUI restart detected for prompt_id=%s", prompt_id)
            raise

        history = payload.get(prompt_id)
        if history is None and len(payload) == 1:
            history = next(iter(payload.values()))

        if isinstance(history, dict) and _history_entry_complete(history):
            _check_history_for_errors(history)  # raises ComfyExecutionError on OOM / error
            return history

        if now - last_log_at >= 30:
            LOGGER.info("Waiting for ComfyUI prompt_id=%s (%.0fs elapsed)", prompt_id, now - started_at)
            last_log_at = now

        time.sleep(poll_interval_sec)

    raise TimeoutError(
        f"Timed out waiting for ComfyUI prompt_id={prompt_id}. Last payload: {last_payload}"
    )


def _collect_strings(value: Any, output_paths: list[str]) -> None:
    """Walk nested output data and collect likely output file paths."""

    if isinstance(value, dict):
        full_path = value.get("fullpath") or value.get("full_path") or value.get("path")
        filename = value.get("filename")
        subfolder = value.get("subfolder")

        if isinstance(full_path, str) and full_path:
            output_paths.append(full_path)
        elif isinstance(filename, str) and filename:
            if isinstance(subfolder, str) and subfolder:
                output_paths.append(str(Path(subfolder) / filename))
            else:
                output_paths.append(filename)

        for nested in value.values():
            _collect_strings(nested, output_paths)
        return

    if isinstance(value, list):
        for item in value:
            _collect_strings(item, output_paths)


def collect_output_paths(history: dict[str, Any]) -> list[str]:
    """Extract a flat list of output file paths from ComfyUI history."""

    raw_paths: list[str] = []
    _collect_strings(history.get("outputs", history), raw_paths)

    seen: set[str] = set()
    ordered_paths: list[str] = []
    for path in raw_paths:
        if path in seen:
            continue
        seen.add(path)
        ordered_paths.append(path)

    LOGGER.info("Collected %s output path(s)", len(ordered_paths))
    return ordered_paths


def _resolve_local_input_source(source_path: str) -> Path:
    """Resolve a worker-local staging source and ensure it is under an allowed root."""

    candidate = Path(str(source_path or "")).expanduser().resolve(strict=False)
    allowed_roots = list(served_file_roots().values()) + [comfy_input_dir()]
    for root in allowed_roots:
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    raise ValueError(f"Input source path is outside allowed roots: {candidate}")


def _download_input_source(source_url: str, destination: Path) -> None:
    """Download an external file into the ComfyUI input directory."""

    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported input source URL: {source_url}")

    temp_path = Path(f"{destination}.part")
    safe_unlink(temp_path)
    try:
        with requests.get(source_url, stream=True, timeout=(10, 300)) as response:
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=get_settings().download_chunk_size):
                    if not chunk:
                        continue
                    handle.write(chunk)
        if not is_non_empty_file(temp_path):
            raise RuntimeError(f"Downloaded input file is empty: {destination.name}")
        os.replace(temp_path, destination)
    except Exception:
        safe_unlink(temp_path)
        raise


def _is_supported_image_file(path: Path) -> bool:
    """Return True when the file has a known image container signature."""

    if not is_non_empty_file(path):
        return False
    try:
        data = path.read_bytes()
    except OSError:
        return False

    header = data[:16]
    return (
        (header.startswith(b"\x89PNG\r\n\x1a\n") and b"IEND" in data[-32:])
        or (header.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9"))
        or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
        or header.startswith((b"GIF87a", b"GIF89a"))
        or header.startswith(b"BM")
    )


def _requires_image_validation(file_spec: ComfyInputFile) -> bool:
    """Identify staged inputs that ComfyUI will load as images."""

    if str(file_spec.input_name or "").strip().lower() == "image":
        return True
    return Path(str(file_spec.filename or "")).suffix.lower() in _IMAGE_EXTENSIONS


def _is_valid_staged_input(path: Path, file_spec: ComfyInputFile) -> bool:
    if not _requires_image_validation(file_spec):
        return is_non_empty_file(path)
    return _is_supported_image_file(path)


def _ensure_valid_staged_input(path: Path, file_spec: ComfyInputFile, action: str) -> None:
    if _is_valid_staged_input(path, file_spec):
        return
    if _requires_image_validation(file_spec):
        raise RuntimeError(f"{action} input file is not a valid image: {path.name}")
    raise RuntimeError(f"{action} input file is empty: {path.name}")


def _resolve_input_destination(file_spec: ComfyInputFile, staged_filename: str) -> Path:
    """Resolve a safe destination under ComfyUI input, honoring an optional subfolder."""

    base_dir = comfy_input_dir()
    subfolder = Path(str(file_spec.subfolder or "").strip())
    if subfolder.is_absolute():
        raise ValueError(f"Comfy input subfolder must be relative: {file_spec.subfolder}")

    destination_dir = (base_dir / subfolder).resolve(strict=False)
    try:
        destination_dir.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError(f"Comfy input subfolder escapes input dir: {file_spec.subfolder}") from exc
    return destination_dir / staged_filename


def stage_comfy_input_file(file_spec: ComfyInputFile) -> str:
    """Stage one file into ComfyUI input and return the LoadImage path."""

    staged_filename = Path(file_spec.filename).name
    if not staged_filename:
        raise ValueError(f"Invalid comfy input filename: {file_spec.filename!r}")

    subfolder = str(file_spec.subfolder or "").strip().strip("/")
    staged_input_path = str(Path(subfolder) / staged_filename) if subfolder else staged_filename
    destination = _resolve_input_destination(file_spec, staged_filename)
    destination_dir = destination.parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    if _is_valid_staged_input(destination, file_spec):
        return staged_input_path
    if destination.exists():
        LOGGER.warning("Replacing invalid staged Comfy input: %s", destination)
        safe_unlink(destination)

    if file_spec.source_data:
        import base64 as _base64
        destination.write_bytes(_base64.b64decode(file_spec.source_data))
        _ensure_valid_staged_input(destination, file_spec, "Decoded source_data")
        return staged_input_path

    if file_spec.source_path:
        source = _resolve_local_input_source(file_spec.source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Comfy input source not found: {source}")
        if source.resolve(strict=False) != destination.resolve(strict=False):
            shutil.copyfile(source, destination)
        _ensure_valid_staged_input(destination, file_spec, "Copied")
        return staged_input_path

    if file_spec.source_url:
        _download_input_source(file_spec.source_url, destination)
        _ensure_valid_staged_input(destination, file_spec, "Downloaded")
        return staged_input_path

    raise ValueError(f"Comfy input file has no usable source: {file_spec}")


def apply_comfy_input_files(
    comfy_payload: dict[str, Any],
    input_files: list[ComfyInputFile],
) -> dict[str, Any]:
    """Stage input files and patch the payload to load them from ComfyUI input."""

    if not input_files:
        return comfy_payload

    patched_payload = deepcopy(comfy_payload)
    for file_spec in input_files:
        staged_filename = stage_comfy_input_file(file_spec)
        node = patched_payload.get(file_spec.node_id)
        if not isinstance(node, dict):
            raise ValueError(f"Comfy payload missing node {file_spec.node_id!r}")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError(f"Comfy payload node {file_spec.node_id!r} has no inputs dict")

        inputs[file_spec.input_name] = staged_filename
        if file_spec.type:
            inputs["type"] = file_spec.type
        if "subfolder" in inputs or file_spec.subfolder:
            inputs["subfolder"] = file_spec.subfolder

    return patched_payload


def _resolve_output_path(raw_path: str) -> Path:
    """Map a Comfy history path to a local absolute path when possible."""

    candidate = Path(str(raw_path or "")).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)

    roots = served_file_roots()
    if candidate.parts and candidate.parts[0] in roots and len(candidate.parts) > 1:
        explicit_root_candidate = (
            roots[candidate.parts[0]] / Path(*candidate.parts[1:])
        ).resolve(strict=False)
        if explicit_root_candidate.exists():
            return explicit_root_candidate

    for root in roots.values():
        rooted_candidate = (root / candidate).resolve(strict=False)
        if rooted_candidate.exists():
            return rooted_candidate

    if candidate.parts and candidate.parts[0] in roots and len(candidate.parts) > 1:
        return (roots[candidate.parts[0]] / Path(*candidate.parts[1:])).resolve(strict=False)

    default_root = roots.get("output")
    if default_root is None:
        return candidate.resolve(strict=False)
    return (default_root / candidate).resolve(strict=False)


def build_output_files(paths: list[str]) -> list[OutputFile]:
    """Build downloadable metadata for worker-produced files."""

    output_files: list[OutputFile] = []
    for raw_path in paths:
        resolved_path = _resolve_output_path(raw_path)
        served_root_name: str | None = None
        relative_path: Path | None = None

        for root_name, root in served_file_roots().items():
            try:
                relative_path = resolved_path.relative_to(root)
                served_root_name = root_name
                break
            except ValueError:
                continue

        if served_root_name is None or relative_path is None:
            LOGGER.warning(
                "Skipping undownloadable output outside served roots: raw=%s resolved=%s",
                raw_path,
                resolved_path,
            )
            continue

        output_files.append(
            OutputFile(
                path=str(resolved_path),
                filename=resolved_path.name,
                download_url=f"/files/{served_root_name}/{quote(relative_path.as_posix(), safe='/')}",
            )
        )

    return output_files
