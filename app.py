"""FastAPI entrypoint for the FilmForge GPU worker."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse
import requests

from gpu_worker.asset_canonical import canonical_asset_group, canonicalize_groups
from gpu_worker.asset_manager import (
    active_download_status,
    ensure_asset_group,
    ensure_asset_group_provisioned,
    is_asset_group_warm,
)
from gpu_worker.asset_registry import (
    ASSET_REGISTRY,
    asset_group_supported_by_capabilities,
    asset_groups_for_capabilities,
)
from gpu_worker.comfy_client import (
    ComfyExecutionError,
    ComfyRestartDetectedError,
    apply_comfy_input_files,
    build_output_files,
    collect_output_paths,
    comfy_queue_depth,
    free_comfy_memory,
    observe_staged_input_receipts,
    poll_for_completion,
    resolve_served_file,
    submit_prompt,
)
from gpu_worker.comfy_process import (
    is_comfy_healthy,
    restart_comfy,
    run_gpu_diagnostics,
    run_gpu_smoke_test,
)
from gpu_worker.config import get_settings
from gpu_worker.flux_ipadapter import (
    FLUX_IPADAPTER_ASSET_GROUP,
    check_flux_ipadapter_readiness,
)
from gpu_worker.infinitetalk import (
    INFINITETALK_ASSET_GROUP,
    INFINITETALK_TWO_PERSON_ASSET_GROUP,
    INFINITETALK_TWO_PERSON_V2_ASSET_GROUP,
    build_two_person_routing_receipt,
    check_infinitetalk_readiness,
    normalize_approved_mpeg_to_wav,
    normalized_audio_path_for_source,
    postprocess_two_person_v2_outputs,
    prepare_two_person_routing_inputs,
)
from gpu_worker.keyframes import extract_keyframes_b64, is_video_output
from gpu_worker.stitch import stitch_clips, upload_via_signed_put
from gpu_worker.utils import ensure_no_symlink_path, sha256_file
from gpu_worker.worker_auth import (
    WorkerAPIAuthError,
    verify_worker_api_authorization,
    worker_api_auth_is_ready,
)
from gpu_worker.worker_ingress import WorkerIngressMiddleware
from gpu_worker.schemas import (
    ActiveJobSummary,
    ClipKeyframe,
    ClipKeyframes,
    ComfyInputFile,
    EnsureAssetGroupResult,
    EnsureAssetsRequest,
    EnsureAssetsResponse,
    AssetGroupStats,
    HealthResponse,
    JobProgressResponse,
    JobStatusResponse,
    JobSubmitResponse,
    OutputFile,
    OutputUploadTarget,
    RunDebug,
    RunRequest,
    RunResponse,
    RunTimings,
    StatsResponse,
    StitchRequest,
    StitchResponse,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)

# When LOG_PROMPTS_ONLY=true, suppress all normal logs and only show the
# dedicated gpu_worker.prompts logger.
if get_settings().log_prompts_only:
    logging.getLogger().setLevel(logging.WARNING)
    _prompt_handler = logging.StreamHandler()
    _prompt_handler.setFormatter(logging.Formatter("%(asctime)s [PROMPT] %(message)s"))
    logging.getLogger("gpu_worker.prompts").addHandler(_prompt_handler)
    logging.getLogger("gpu_worker.prompts").setLevel(logging.INFO)
    logging.getLogger("gpu_worker.prompts").propagate = False

app = FastAPI(title="FilmForge GPU Worker", version="0.1.0")
app.add_middleware(
    WorkerIngressMiddleware,
    settings_provider=lambda: get_settings(),
)

# Self-heal model wiring BEFORE any asset check: guarantee that the hardcoded
# /workspace/ComfyUI/models paths resolve to the populated data volume, so we
# reuse already-downloaded models instead of re-downloading the warm set. Runs
# synchronously so it wins the race against the preload thread below.
try:
    from gpu_worker.model_store import ensure_model_store

    ensure_model_store()
except Exception:  # never block worker startup on self-heal
    logging.getLogger(__name__).exception("model-store self-heal failed; continuing")

# Run GPU diagnostics and smoke test in a background thread so they don't
# block uvicorn startup. Results appear in the worker log (journalctl / /tmp/gpu_worker.log).
threading.Thread(
    target=lambda: (run_gpu_diagnostics(), run_gpu_smoke_test()),
    name="gpu-startup-check",
    daemon=True,
).start()

_STILL_ASSET_GROUPS = {"flux_stills_v1"}
_VIDEO_ASSET_GROUPS = {
    "wan_i2v_v1",
    INFINITETALK_ASSET_GROUP,
    INFINITETALK_TWO_PERSON_ASSET_GROUP,
    INFINITETALK_TWO_PERSON_V2_ASSET_GROUP,
}
_FINALIZATION_BUFFER_SEC = 10.0
_BASE_STILL_SEC = 60.0
_BASE_VIDEO_SEC = 30.0
_ETA_WINDOW = 12

# ── VRAM pre-flight guard ─────────────────────────────────────────────────────
# Minimum free VRAM (MiB) required before this worker accepts a job.
# Based on observed inference activation peaks on a single shared GPU device.
# When multiple ComfyUI instances share one physical GPU, a job that would OOM
# mid-inference is rejected here instantly so the broker routes it elsewhere.
_VRAM_FLOOR_MIB: dict[str, int] = {
    "wan_i2v_v1": 55_000,    # WAN 14B: ~55GB activation headroom observed
    "flux_stills_v1": 12_000, # FLUX: ~12GB activation headroom
}


def _own_gpu_index() -> int:
    """Physical GPU index this worker owns. nvidia-smi ignores CUDA_VISIBLE_DEVICES
    and lists ALL GPUs, so on a multi-GPU box we must index by it ourselves — else
    every worker reads physical GPU 0's VRAM and they all reject/accept in lockstep.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return 0
    first = cvd.split(",")[0].strip()
    try:
        return int(first)
    except ValueError:
        return 0


def _query_vram_mib(field: str) -> int | None:
    """Return memory.<field> in MiB for THIS worker's GPU via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu=memory.{field}", "--format=csv,noheader,nounits"],
            timeout=3,
            stderr=subprocess.DEVNULL,
        )
        lines = out.decode().strip().splitlines()
        idx = _own_gpu_index()
        if 0 <= idx < len(lines):
            return int(lines[idx])
        return int(lines[0])
    except Exception:
        return None


def _free_vram_mib() -> int | None:
    """Return free VRAM in MiB for this worker's GPU. None if unavailable."""
    return _query_vram_mib("free")


def _total_vram_mib() -> int | None:
    """Return total VRAM in MiB for this worker's GPU. None if unavailable."""
    return _query_vram_mib("total")


def _effective_vram_floor(canonical_group: str) -> int | None:
    """VRAM floor for a cold load, capped to the card's capacity.

    The configured floors size a COLD load on a big (96GB) card. On smaller cards
    (e.g. 40GB A100) WAN/FLUX still run via CPU offload, but a floor above the
    card's total VRAM is unsatisfiable and would reject every cold job. Cap the
    floor at 90% of total VRAM so it scales to the GPU instead of hard-failing.
    """
    floor = _VRAM_FLOOR_MIB.get(canonical_group)
    if floor is None:
        return None
    total = _total_vram_mib()
    if total is not None:
        floor = min(floor, int(total * 0.9))
    return floor


# ── Active-job tracking (used by watchdog to avoid restarting mid-run) ────────
_ACTIVE_JOBS_LOCK = threading.Lock()
_ACTIVE_JOBS: int = 0  # count of jobs currently executing
_MAX_EXECUTION_SLOTS = max(1, get_settings().resolved_max_concurrent_jobs())
_EXECUTION_SEMAPHORE = threading.BoundedSemaphore(value=_MAX_EXECUTION_SLOTS)

# Last model family run, so we only release VRAM when it actually changes (see
# _free_vram_on_group_switch). ComfyUI owns the actual GPU execution queue; this
# value is a best-effort resident-model hint for the API's concurrent callers.
_LAST_ASSET_GROUP: str | None = None

# Monotonic count of jobs that have finished executing (success or failure).
# The watchdog reads it to drive the proactive VRAM recycle (see
# _maybe_recycle_comfy); it tracks its own baseline, so this is never reset.
_JOBS_COMPLETED_LOCK = threading.Lock()
_JOBS_COMPLETED: int = 0

# ── Warmed asset groups tracking ─────────────────────────────────────────────────
_WARMED_GROUPS_LOCK = threading.Lock()
_WARMED_GROUPS: set[str] = set()  # asset groups that have been successfully downloaded

_STATS_LOCK = threading.Lock()
_STATS: dict[str, deque] = {}
_STATS_WINDOW = 20
_ETA_HISTORY: dict[str, deque] = {}


def _declared_capabilities() -> list[str]:
    """Return capabilities exactly as declared by WORKER_CAPABILITIES."""

    return get_settings().resolved_capabilities()


def _preload_asset_groups() -> list[str]:
    """Asset groups this worker should preload at startup.

    When WORKER_CAPABILITIES is unset, default to every known group — the
    homogeneous worker model assumes every worker can serve every group.
    Explicit WORKER_CAPABILITIES still narrows coverage when intentionally set.
    """

    declared = _declared_capabilities()
    if not declared:
        return sorted(ASSET_REGISTRY)
    return asset_groups_for_capabilities(declared)


def _asset_group_allowed(asset_group: str) -> bool:
    """Return whether this worker is allowed to run a requested asset group."""

    return asset_group_supported_by_capabilities(asset_group, _declared_capabilities())


def _asset_group_mismatch_error(asset_group: str) -> ValueError:
    declared = _declared_capabilities()
    supported_groups = _preload_asset_groups()
    declared_text = ", ".join(declared) if declared else "(all asset groups)"
    supported_text = ", ".join(supported_groups) if supported_groups else "(none)"
    return ValueError(
        f"Worker does not support asset_group={asset_group!r}. "
        f"WORKER_CAPABILITIES={declared_text}; supported asset groups={supported_text}"
    )


def _advertised_capabilities() -> tuple[list[str], dict | None, dict | None]:
    """Return only capabilities whose runtime contract is actually present.

    Returns ``(capabilities, infinitetalk_readiness, flux_ipadapter_readiness)``;
    a readiness dict is ``None`` when its group was not declared at all.
    """

    raw_capabilities = get_settings().resolved_capabilities() or sorted(ASSET_REGISTRY)
    capabilities = canonicalize_groups(raw_capabilities)
    infinitetalk_readiness = None
    if INFINITETALK_ASSET_GROUP in capabilities:
        readiness = check_infinitetalk_readiness(require_two_person=False)
        infinitetalk_readiness = readiness.as_dict()
        if not readiness.ready:
            capabilities.remove(INFINITETALK_ASSET_GROUP)
    if INFINITETALK_TWO_PERSON_ASSET_GROUP in capabilities:
        readiness = check_infinitetalk_readiness(require_two_person=True)
        infinitetalk_readiness = readiness.as_dict()
        if not readiness.ready:
            capabilities.remove(INFINITETALK_TWO_PERSON_ASSET_GROUP)
    if INFINITETALK_TWO_PERSON_V2_ASSET_GROUP in capabilities:
        readiness = check_infinitetalk_readiness(
            require_two_person=True,
            require_roomtone_v2=True,
        )
        infinitetalk_readiness = readiness.as_dict()
        if not readiness.ready:
            capabilities.remove(INFINITETALK_TWO_PERSON_V2_ASSET_GROUP)
    flux_ipadapter_readiness = None
    if FLUX_IPADAPTER_ASSET_GROUP in capabilities:
        # The custom node loads only at a Comfy start and the weights arrive
        # asynchronously; until both are true the group is a promise the first
        # render breaks (missing_node_type, 2026-08-22). Withhold it, and say why
        # in /health so the operator reaches for the provisioner, not the key.
        readiness = check_flux_ipadapter_readiness()
        flux_ipadapter_readiness = readiness.as_dict()
        if not readiness.ready:
            capabilities.remove(FLUX_IPADAPTER_ASSET_GROUP)
    return capabilities, infinitetalk_readiness, flux_ipadapter_readiness


def _ensure_runtime_provisioned(asset_group: str) -> bool:
    """Keep this runtime hook narrow: legacy audio provisioners retain their path."""

    if canonical_asset_group(asset_group) not in {
        INFINITETALK_ASSET_GROUP,
        INFINITETALK_TWO_PERSON_ASSET_GROUP,
        INFINITETALK_TWO_PERSON_V2_ASSET_GROUP,
    }:
        return False
    # A1 and A2 share one pinned wrapper install; track the provisioner once.
    return ensure_asset_group_provisioned(INFINITETALK_ASSET_GROUP)


def _normalize_infinitetalk_audio_inputs(
    comfy_payload: dict,
    comfy_input_files: list[ComfyInputFile],
    *,
    job_id: str,
) -> tuple[dict, list[ComfyInputFile]]:
    """Replace MPEG inputs and return specs for the effective PNG+WAV graph."""

    input_root = Path(get_settings().comfy_input_dir).expanduser().resolve(strict=False)
    observer_specs: list[ComfyInputFile] = []
    for file_spec in comfy_input_files:
        if file_spec.input_name != "audio":
            observer_specs.append(file_spec)
            continue
        if (file_spec.content_type or "").lower() != "audio/mpeg":
            raise ValueError("InfiniteTalk audio input must be approved audio/mpeg")
        source_digest = str(file_spec.expected_sha256 or "").strip().lower()
        if not source_digest:
            raise ValueError("InfiniteTalk audio input requires an approved source digest")
        node = comfy_payload.get(str(file_spec.node_id))
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            raise ValueError(f"InfiniteTalk audio node is missing: {file_spec.node_id}")
        staged_name = node["inputs"].get(file_spec.input_name)
        if not isinstance(staged_name, str) or not staged_name:
            raise ValueError(f"InfiniteTalk audio input is missing: {file_spec.node_id}")
        source = ensure_no_symlink_path(
            input_root,
            input_root / staged_name,
            require_leaf=True,
            require_regular_file=True,
            action="InfiniteTalk staged audio",
        )
        normalized_path = normalize_approved_mpeg_to_wav(source, job_id=job_id)
        expected_normalized = normalized_audio_path_for_source(
            source_digest,
            job_id=job_id,
        )
        normalized = ensure_no_symlink_path(
            input_root,
            normalized_path,
            require_leaf=True,
            require_regular_file=True,
            action="InfiniteTalk normalized audio",
        )
        if normalized != expected_normalized:
            raise RuntimeError("InfiniteTalk normalized audio lost source/job ownership")
        try:
            normalized_name = str(normalized.relative_to(input_root))
        except ValueError as exc:
            raise RuntimeError("InfiniteTalk normalized audio escapes Comfy input") from exc
        node["inputs"][file_spec.input_name] = normalized_name
        observer_specs.append(
            ComfyInputFile(
                node_id=file_spec.node_id,
                input_name=file_spec.input_name,
                filename=normalized.name,
                expected_sha256=sha256_file(normalized),
                content_type="audio/wav",
                type=file_spec.type,
            )
        )
    return comfy_payload, observer_specs


def _prepare_comfy_inputs(request: RunRequest) -> tuple[dict, list[ComfyInputFile]]:
    """Stage source bytes, attest them, then describe the effective graph inputs."""

    prepared_payload = apply_comfy_input_files(
        request.comfy_payload,
        request.comfy_input_files,
    )
    # Transformations must never hide a bad source receipt. In particular, the
    # approved MP3 digest is checked before its graph entry becomes a WAV.
    observe_staged_input_receipts(prepared_payload, request.comfy_input_files)
    observer_specs = request.comfy_input_files
    canonical_group = canonical_asset_group(request.asset_group)
    if canonical_group in {
        INFINITETALK_ASSET_GROUP,
        INFINITETALK_TWO_PERSON_ASSET_GROUP,
        INFINITETALK_TWO_PERSON_V2_ASSET_GROUP,
    }:
        routing_schema = (
            request.infinitetalk_routing.schema_version
            if request.infinitetalk_routing is not None
            else None
        )
        expected_routing_schema = {
            INFINITETALK_TWO_PERSON_ASSET_GROUP: "infinitetalk_two_person_routing_v1",
            INFINITETALK_TWO_PERSON_V2_ASSET_GROUP: "infinitetalk_two_person_routing_v2",
        }.get(canonical_group)
        if canonical_group == INFINITETALK_ASSET_GROUP and routing_schema is not None:
            required_group = (
                INFINITETALK_TWO_PERSON_V2_ASSET_GROUP
                if routing_schema == "infinitetalk_two_person_routing_v2"
                else INFINITETALK_TWO_PERSON_ASSET_GROUP
            )
            raise ValueError(f"Two-person routing requires {required_group}")
        if expected_routing_schema is not None and routing_schema != expected_routing_schema:
            raise ValueError(
                f"{canonical_group} requires frozen routing authority {expected_routing_schema}"
            )
        prepared_payload, observer_specs = _normalize_infinitetalk_audio_inputs(
            prepared_payload,
            request.comfy_input_files,
            job_id=request.job_id,
        )
        if routing_schema == "infinitetalk_two_person_routing_v2":
            # Bind the normalized intermediate before node 10 is replaced by
            # its full-window, slot-specific roomtone conditioning derivative.
            observe_staged_input_receipts(prepared_payload, observer_specs)
        prepared_payload, observer_specs = prepare_two_person_routing_inputs(
            prepared_payload,
            request.comfy_input_files,
            observer_specs,
            routing=request.infinitetalk_routing,
            job_id=request.job_id,
        )
    return prepared_payload, observer_specs


def _note_job_completed() -> None:
    """Bump the monotonic completed-jobs counter (once per finished job)."""
    global _JOBS_COMPLETED
    with _JOBS_COMPLETED_LOCK:
        _JOBS_COMPLETED += 1


def _jobs_completed() -> int:
    with _JOBS_COMPLETED_LOCK:
        return _JOBS_COMPLETED


def _acquire_all_slots(timeout_sec: float) -> bool:
    """Grab every execution slot so no job can run during a restart.

    Returns True with all slots held — the caller MUST call _release_all_slots.
    Returns False (having released anything it took) if a job is holding a slot
    within the timeout. Closes the race between an idle snapshot and the restart.
    """
    acquired = 0
    deadline = time.monotonic() + timeout_sec
    for _ in range(_MAX_EXECUTION_SLOTS):
        remaining = max(0.0, deadline - time.monotonic())
        if _EXECUTION_SEMAPHORE.acquire(timeout=remaining):
            acquired += 1
        else:
            break
    if acquired == _MAX_EXECUTION_SLOTS:
        return True
    for _ in range(acquired):
        _EXECUTION_SEMAPHORE.release()
    return False


def _release_all_slots() -> None:
    for _ in range(_MAX_EXECUTION_SLOTS):
        _EXECUTION_SEMAPHORE.release()


def _periodic_recycle_due(completed: int, baseline: int, after_jobs: int) -> bool:
    """True when at least ``after_jobs`` jobs finished since the last recycle."""
    return after_jobs > 0 and (completed - baseline) >= after_jobs


def _maybe_recycle_comfy(settings, recycle_baseline: int) -> bool:
    """Proactively restart ComfyUI to clear accumulated VRAM — only while idle.

    Restarting ComfyUI is what actually reclaims the GPU's VRAM: the /free
    endpoint only unloads models and flushes the CUDA cache, which does NOT
    recover leaked or fragmented allocations that build up across many jobs.

    Fires on either trigger (both opt-in via env; each disabled when 0):
      • WORKER_RECYCLE_AFTER_JOBS   — jobs completed since the last recycle.
      • WORKER_RECYCLE_MIN_FREE_MIB — free-VRAM floor. Low free VRAM alone is
        normal under --highvram (the warm model is pinned), so we first call
        /free and re-read; we only recycle if the memory does NOT come back —
        i.e. it is genuinely leaked/fragmented, not just a resident model.

    Returns True iff ComfyUI was restarted (the caller then re-baselines).
    """
    after_jobs = getattr(settings, "worker_recycle_after_jobs", 0)
    floor_mib = getattr(settings, "worker_recycle_min_free_mib", 0)
    if after_jobs <= 0 and floor_mib <= 0:
        return False  # feature disabled

    # Cheap idle snapshot — never touch a busy worker.
    with _ACTIVE_JOBS_LOCK:
        if _ACTIVE_JOBS > 0:
            return False

    periodic_due = _periodic_recycle_due(_jobs_completed(), recycle_baseline, after_jobs)

    raw_low = False
    if floor_mib > 0:
        free_mib = _free_vram_mib()
        raw_low = free_mib is not None and free_mib < floor_mib

    if not periodic_due and not raw_low:
        return False

    # A trigger is plausible — quiesce so a job can't start mid-restart.
    if not _acquire_all_slots(timeout_sec=5.0):
        LOGGER.info("[recycle] recycle wanted but a job started — deferring")
        return False
    try:
        if periodic_due:
            reason = f"periodic: {_jobs_completed() - recycle_baseline} jobs since last recycle"
        else:
            # Confirm the pressure is real: release the resident model and re-read.
            # If /free brings VRAM back above the floor it was just the warm model
            # — no restart needed (and we've done a useful flush anyway).
            free_comfy_memory(unload_models=True)
            after_free = _free_vram_mib()
            if after_free is None or after_free >= floor_mib:
                LOGGER.info(
                    "[recycle] /free recovered VRAM (free=%sMiB >= %dMiB floor) — no restart",
                    after_free, floor_mib,
                )
                return False
            reason = f"free VRAM {after_free}MiB < {floor_mib}MiB even after /free (leaked)"
        LOGGER.warning("[recycle] restarting ComfyUI proactively — %s", reason)
        restart_comfy()
        LOGGER.info("[recycle] ComfyUI restarted")
        return True
    finally:
        _release_all_slots()


def _watchdog_loop() -> None:
    """Background thread: keep ComfyUI healthy AND clear its VRAM creep.

    Two idle-gated actions, never taken while a job is running:
      1. Restart ComfyUI when it goes unhealthy.
      2. Proactively recycle ComfyUI to clear accumulated / fragmented VRAM,
         driven by WORKER_RECYCLE_AFTER_JOBS / WORKER_RECYCLE_MIN_FREE_MIB.
    """

    settings = get_settings()
    if not settings.comfy_start_cmd:
        LOGGER.info("[watchdog] COMFY_START_CMD not set — watchdog disabled")
        return

    recycle_baseline = _jobs_completed()
    LOGGER.info("[watchdog] started, checking every 30s")
    while True:
        time.sleep(30)
        try:
            if not is_comfy_healthy():
                with _ACTIVE_JOBS_LOCK:
                    active = _ACTIVE_JOBS
                if active > 0:
                    LOGGER.warning("[watchdog] ComfyUI unhealthy but %d job(s) active — waiting", active)
                    continue
                LOGGER.warning("[watchdog] ComfyUI unhealthy and idle — restarting")
                restart_comfy()
                recycle_baseline = _jobs_completed()
                LOGGER.info("[watchdog] ComfyUI restarted successfully")
                continue

            # ComfyUI is healthy — proactively recycle to clear VRAM creep.
            if _maybe_recycle_comfy(settings, recycle_baseline):
                recycle_baseline = _jobs_completed()
        except Exception as exc:
            LOGGER.error("[watchdog] error: %s", exc)


threading.Thread(
    target=_watchdog_loop,
    name="comfy-watchdog",
    daemon=True,
).start()

def _require_worker_api_token(
    authorization: str | None = Header(default=None),
) -> None:
    """FastAPI dependency to require WORKER_API_TOKEN on protected endpoints."""
    settings = get_settings()
    try:
        verify_worker_api_authorization(
            expected_token=settings.worker_api_token,
            auth_mode=settings.worker_api_auth_mode,
            authorization=authorization,
        )
    except WorkerAPIAuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from None


def _broker_headers() -> dict[str, str]:
    settings = get_settings()
    headers = {"Content-Type": "application/json"}
    token = (settings.resolved_registration_token() or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-Render-Broker-Token"] = token
    return headers


def _worker_api_auth_ready() -> bool:
    settings = get_settings()
    return worker_api_auth_is_ready(
        expected_token=settings.worker_api_token,
        auth_mode=settings.worker_api_auth_mode,
    )


def _performance_snapshot() -> dict[str, dict[str, float | int]]:
    """Return rolling worker timing stats as heartbeat-safe metadata."""

    with _STATS_LOCK:
        snapshot = {asset_group: list(entries) for asset_group, entries in _STATS.items()}

    performance: dict[str, dict[str, float | int]] = {}
    for asset_group, entries in sorted(snapshot.items()):
        if not entries:
            continue
        sample_count = len(entries)
        avg_comfy_run_sec = sum(entry[0] for entry in entries) / sample_count
        avg_total_sec = sum(entry[1] for entry in entries) / sample_count
        performance[asset_group] = {
            "sample_count": sample_count,
            "avg_comfy_run_sec": round(avg_comfy_run_sec, 2),
            "avg_total_sec": round(avg_total_sec, 2),
        }
    return performance


def _broker_worker_payload() -> dict[str, object]:
    settings = get_settings()
    with _ACTIVE_JOBS_LOCK:
        active_jobs = _ACTIVE_JOBS

    free_vram_mb = _free_vram_mib()
    if free_vram_mb is None and settings.worker_vram_gb is not None:
        free_vram_mb = int(settings.worker_vram_gb * 1024)

    capabilities, infinitetalk_readiness, flux_ipadapter_readiness = _advertised_capabilities()
    public_url = settings.resolved_worker_public_url()

    with _WARMED_GROUPS_LOCK:
        warmed = canonicalize_groups(sorted(_WARMED_GROUPS))

    max_concurrent_jobs = settings.resolved_max_concurrent_jobs()
    comfy_queue = comfy_queue_depth()
    metadata: dict = {
        "comfy_base_url": settings.comfy_base_url,
        "comfy_reachable": is_comfy_healthy(),
        "comfy_queue_running": comfy_queue["running"] if comfy_queue else None,
        "comfy_queue_pending": comfy_queue["pending"] if comfy_queue else None,
        # Cloud provider + instance id let the deploy UI cross-check this
        "provider": settings.worker_provider,
        "instance_id": settings.worker_instance_id,
        "performance": _performance_snapshot(),
        "performance_window": _STATS_WINDOW,
        "active_jobs": active_jobs,
        "max_concurrent_jobs": max_concurrent_jobs,
        "warmed_asset_groups": warmed,
        "infinitetalk_readiness": infinitetalk_readiness,
        "flux_ipadapter_readiness": flux_ipadapter_readiness,
        # Non-secret readiness fact: the backend withholds paid dispatch unless
        # the worker proves it can fetch this deployment's public storage host.
        "input_url_allowed_hosts": sorted(
            settings.resolved_input_url_allowed_hosts()
        ),
    }
    if settings.worker_vision_base_url:
        # Vision boxes: the vLLM tunnel URL the backend's LLM gateway should call
        # (distinct from base_url, which is this worker app's own :9000 tunnel).
        metadata["vision_base_url"] = settings.worker_vision_base_url.rstrip("/")
    return {
        "worker_id": settings.resolved_worker_id(),
        "worker_name": settings.resolved_worker_name(),
        "base_url": public_url,
        "public_url": public_url,
        "gpu_name": settings.worker_gpu_name,
        "provider": settings.worker_provider,
        "status": "online",
        "supported_asset_groups": capabilities,
        "capabilities": capabilities,
        "warmed_asset_groups": warmed,
        "free_vram_mb": free_vram_mb,
        "vram_gb": settings.worker_vram_gb,
        "used_vram_mb": None,
        "active_jobs": active_jobs,
        "max_concurrency": max_concurrent_jobs,
        "max_concurrent_jobs": max_concurrent_jobs,
        "metadata": metadata,
    }


def _register_with_broker() -> None:
    settings = get_settings()
    if not _worker_api_auth_ready():
        LOGGER.error("[broker] worker API unavailable; registration suppressed")
        return
    backend_url = settings.resolved_backend_url()
    if not backend_url:
        LOGGER.info("[broker] backend URL not configured — worker registration disabled")
        return

    url = f"{backend_url.rstrip('/')}/api/render-broker/workers/register"
    payload = _broker_worker_payload()
    try:
        response = requests.post(url, json=payload, headers=_broker_headers(), timeout=15)
        response.raise_for_status()
        LOGGER.info("[broker] registered worker_id=%s target=%s", settings.resolved_worker_id(), url)
    except Exception as exc:
        LOGGER.warning("[broker] worker registration failed: %s", exc)


def _send_heartbeat_now() -> None:
    """Fire a single heartbeat immediately. Best-effort — never raises."""
    settings = get_settings()
    if not _worker_api_auth_ready():
        return
    backend_url = settings.resolved_backend_url()
    if not backend_url:
        return
    worker_id = settings.resolved_worker_id()
    if not worker_id:
        return
    url = f"{backend_url.rstrip('/')}/api/render-broker/workers/{worker_id}/heartbeat"
    payload = _broker_worker_payload()
    payload.pop("worker_id", None)
    try:
        response = requests.post(url, json=payload, headers=_broker_headers(), timeout=5)
        response.raise_for_status()
    except Exception as exc:
        LOGGER.debug("[broker] immediate heartbeat failed: %s", exc)


def _broker_heartbeat_loop() -> None:
    settings = get_settings()
    if not _worker_api_auth_ready():
        LOGGER.error("[broker] worker API unavailable; heartbeat disabled")
        return
    backend_url = settings.resolved_backend_url()
    if not backend_url:
        return

    worker_id = settings.resolved_worker_id()
    interval_sec = max(10, settings.resolved_heartbeat_seconds())
    url = f"{backend_url.rstrip('/')}/api/render-broker/workers/{worker_id}/heartbeat"
    LOGGER.info("[broker] heartbeat loop started worker_id=%s interval=%ss", worker_id, interval_sec)

    while True:
        time.sleep(interval_sec)
        payload = _broker_worker_payload()
        payload.pop("worker_id", None)
        try:
            response = requests.post(url, json=payload, headers=_broker_headers(), timeout=15)
            response.raise_for_status()
        except Exception as exc:
            LOGGER.warning("[broker] heartbeat failed for worker_id=%s: %s", worker_id, exc)


def _recover_warm_state_from_disk() -> None:
    """Repopulate _WARMED_GROUPS at startup based on which asset groups are
    fully present on local disk. Without this, every worker restart blanks
    the in-memory warm set and the broker thinks a fully-provisioned box is
    cold until preflight finishes downloading anything missing."""
    recovered: list[str] = []
    for group in sorted(ASSET_REGISTRY):
        if is_asset_group_warm(group):
            with _WARMED_GROUPS_LOCK:
                _WARMED_GROUPS.add(group)
            recovered.append(group)
    if recovered:
        LOGGER.info("[warm-recovery] disk-warm groups: %s", recovered)
    else:
        LOGGER.info("[warm-recovery] no asset groups fully present on disk")


_recover_warm_state_from_disk()
_register_with_broker()
threading.Thread(
    target=_broker_heartbeat_loop,
    name="broker-heartbeat",
    daemon=True,
).start()


def _preflight_download_all() -> None:
    """Background thread: download only model groups supported by this worker."""

    groups = _preload_asset_groups()
    if not groups:
        LOGGER.info(
            "[preflight] no asset groups mapped from WORKER_CAPABILITIES=%s; skipping preload",
            _declared_capabilities(),
        )
        return
    LOGGER.info("[preflight] downloading %d asset group(s): %s", len(groups), groups)

    runtime_changed = False
    failed_groups: list[str] = []
    for group in groups:
        try:
            result = ensure_asset_group(group)
            if result.downloaded_assets:
                runtime_changed = True
                LOGGER.info(
                    "[preflight] %s — downloaded %s in %.1fs",
                    group, result.downloaded_assets, result.download_sec,
                )
            else:
                LOGGER.info("[preflight] %s — already cached (%.2fs check)", group, result.asset_check_sec)
            if _ensure_runtime_provisioned(group):
                runtime_changed = True
                LOGGER.info("[preflight] %s — provisioner completed", group)
            # Mark this asset group as warmed
            with _WARMED_GROUPS_LOCK:
                _WARMED_GROUPS.add(group)
        except Exception as exc:
            failed_groups.append(group)
            LOGGER.error("[preflight] failed to ensure group=%s: %s", group, exc)

    if failed_groups:
        LOGGER.error("[preflight] incomplete; failed group(s): %s", failed_groups)
        return

    if runtime_changed:
        with _ACTIVE_JOBS_LOCK:
            active = _ACTIVE_JOBS
        if active > 0:
            LOGGER.info(
                "[preflight] runtime changed but %d job(s) active — deferring ComfyUI restart",
                active,
            )
        else:
            LOGGER.info("[preflight] runtime changed — restarting ComfyUI once")
            try:
                restart_comfy()
                LOGGER.info("[preflight] ComfyUI restarted, worker ready")
            except Exception as exc:
                LOGGER.error("[preflight] ComfyUI restart failed: %s", exc)
    else:
        LOGGER.info("[preflight] all models already present, worker ready")


threading.Thread(
    target=_preflight_download_all,
    name="model-preflight",
    daemon=True,
).start()


@dataclass
class _JobRecord:
    """Mutable state for one async worker job."""

    job_id: str
    request: RunRequest | None
    client_job_id: str
    asset_group: str
    created_monotonic: float = field(default_factory=time.monotonic)
    finished_monotonic: float | None = None
    status: str = "queued"
    result: RunResponse | None = None
    error: str | None = None
    progress: JobProgressResponse | None = None


_JOB_LOCK = threading.Lock()
_JOBS: dict[str, _JobRecord] = {}
_JOB_REGISTRY_MAX_RECORDS = max(
    1,
    get_settings().worker_job_registry_max_records,
)
_JOB_REGISTRY_TTL_SEC = max(
    1.0,
    float(get_settings().worker_job_registry_ttl_sec),
)
_TERMINAL_JOB_STATES = {"completed", "failed"}


def _prune_job_registry_locked(*, reserve: int = 0) -> bool:
    """Evict expired/old terminal jobs and report whether reserve entries fit."""

    now = time.monotonic()
    expired = [
        job_id
        for job_id, record in _JOBS.items()
        if record.status in _TERMINAL_JOB_STATES
        and record.finished_monotonic is not None
        and now - record.finished_monotonic >= _JOB_REGISTRY_TTL_SEC
    ]
    for job_id in expired:
        _JOBS.pop(job_id, None)

    while len(_JOBS) + reserve > _JOB_REGISTRY_MAX_RECORDS:
        terminal = [
            record
            for record in _JOBS.values()
            if record.status in _TERMINAL_JOB_STATES
        ]
        if not terminal:
            break
        oldest = min(
            terminal,
            key=lambda record: (
                record.finished_monotonic
                if record.finished_monotonic is not None
                else record.created_monotonic
            ),
        )
        _JOBS.pop(oldest.job_id, None)

    return len(_JOBS) + reserve <= _JOB_REGISTRY_MAX_RECORDS


def _detect_resolution_bucket(comfy_payload: dict) -> Literal["low", "medium", "high"]:
    max_dimension = 0
    for node in comfy_payload.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key in ("width", "height", "image_width", "image_height"):
            value = inputs.get(key)
            if isinstance(value, (int, float)) and value > 0:
                max_dimension = max(max_dimension, int(value))

    if max_dimension > 1536:
        return "high"
    if max_dimension > 1024:
        return "medium"
    return "low"


def _resolution_multiplier(bucket: Literal["low", "medium", "high"]) -> float:
    if bucket == "medium":
        return 1.5
    if bucket == "high":
        return 2.0
    return 1.0


def _stage_for_asset_group(asset_group: str) -> Literal["generating_stills", "generating_video"]:
    if asset_group in _STILL_ASSET_GROUPS:
        return "generating_stills"
    return "generating_video"


def _eta_history_key(asset_group: str, resolution_bucket: str) -> str:
    return f"{asset_group}:{resolution_bucket}"


def _measured_prompt_baseline_sec(asset_group: str, resolution_bucket: str) -> float | None:
    key = _eta_history_key(asset_group, resolution_bucket)
    with _STATS_LOCK:
        samples = list(_ETA_HISTORY.get(key, ()))
    if not samples:
        return None
    return sum(samples) / len(samples)


def _default_prompt_baseline_sec(
    asset_group: str,
    resolution_bucket: Literal["low", "medium", "high"],
) -> float:
    base = _BASE_STILL_SEC if asset_group in _STILL_ASSET_GROUPS else _BASE_VIDEO_SEC
    return base * _resolution_multiplier(resolution_bucket)


def _initial_prompt_eta_sec(
    asset_group: str,
    resolution_bucket: Literal["low", "medium", "high"],
) -> float:
    measured = _measured_prompt_baseline_sec(asset_group, resolution_bucket)
    return measured if measured is not None else _default_prompt_baseline_sec(asset_group, resolution_bucket)


def _extended_remaining_sec(expected_sec: float) -> float:
    return max(expected_sec * 0.25, 15.0)


def _build_initial_progress(job_id: str, request: RunRequest) -> JobProgressResponse:
    resolution_bucket = _detect_resolution_bucket(request.comfy_payload)
    stage = _stage_for_asset_group(request.asset_group)
    initial_eta = _initial_prompt_eta_sec(request.asset_group, resolution_bucket) + _FINALIZATION_BUFFER_SEC
    return JobProgressResponse(
        job_id=job_id,
        stage="queued",
        message="Queued",
        eta_sec=initial_eta,
        num_stills_total=1 if request.asset_group in _STILL_ASSET_GROUPS else 0,
        num_stills_done=0,
        still_avg_sec=_measured_prompt_baseline_sec(request.asset_group, resolution_bucket)
        if request.asset_group in _STILL_ASSET_GROUPS
        else None,
        video_expected_sec=(
            _initial_prompt_eta_sec(request.asset_group, resolution_bucket)
            if stage == "generating_video"
            else None
        ),
        resolution_bucket=resolution_bucket,
    )


def _set_progress_stage(
    progress: JobProgressResponse,
    stage: Literal["queued", "starting", "generating_stills", "generating_video", "finalizing", "done", "failed"],
    *,
    message: str,
    eta_sec: float | None = None,
) -> None:
    now = time.time()
    if progress.started_at is None:
        progress.started_at = now
    progress.stage = stage
    progress.message = message
    progress.stage_started_at = now
    progress.stage_elapsed_sec = 0.0
    progress.eta_sec = eta_sec


def _update_progress_elapsed(
    progress: JobProgressResponse,
    *,
    total_started_monotonic: float,
    stage_started_monotonic: float,
    eta_sec: float | None = None,
    message: str | None = None,
) -> None:
    monotonic_now = time.monotonic()
    progress.elapsed_sec = max(monotonic_now - total_started_monotonic, 0.0)
    progress.stage_elapsed_sec = max(monotonic_now - stage_started_monotonic, 0.0)
    if eta_sec is not None:
        progress.eta_sec = max(eta_sec, 0.0)
    if message is not None:
        progress.message = message


def _store_eta_sample(asset_group: str, resolution_bucket: str, comfy_run_sec: float) -> None:
    key = _eta_history_key(asset_group, resolution_bucket)
    with _STATS_LOCK:
        if key not in _ETA_HISTORY:
            _ETA_HISTORY[key] = deque(maxlen=_ETA_WINDOW)
        _ETA_HISTORY[key].append(comfy_run_sec)


def _error_response(
    *,
    request: RunRequest,
    timings: RunTimings,
    downloaded_assets: list[str],
    restart_performed: bool,
    comfy_prompt_id: str | None,
    history_found: bool,
    error: Exception,
) -> RunResponse:
    """Build a structured error response."""

    timings.total_sec = max(timings.total_sec, 0.0)
    return RunResponse(
        ok=False,
        job_id=request.job_id,
        asset_group=request.asset_group,
        downloaded_assets=downloaded_assets,
        restart_performed=restart_performed,
        comfy_prompt_id=comfy_prompt_id,
        outputs=[],
        output_files=[],
        timings=timings,
        debug=RunDebug(
            history_found=history_found,
            comfy_base_url=get_settings().comfy_base_url,
        ),
        error=str(error),
    )


def _prompt_stage_message(progress: JobProgressResponse) -> str:
    if progress.stage == "generating_stills":
        total = max(progress.num_stills_total, 1)
        current = min(progress.num_stills_done + 1, total)
        return f"Generating still {current}/{total}"
    if progress.stage == "generating_video":
        bucket = progress.resolution_bucket or "low"
        return f"Generating video ({bucket} resolution)"
    return progress.message or "Executing prompt"


@app.get("/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    """Return rolling average generation times per asset group."""

    with _STATS_LOCK:
        snapshot = {ag: list(dq) for ag, dq in _STATS.items()}

    groups: list[AssetGroupStats] = []
    for ag, entries in sorted(snapshot.items()):
        if not entries:
            continue
        n = len(entries)
        groups.append(AssetGroupStats(
            asset_group=ag,
            sample_count=n,
            avg_comfy_run_sec=round(sum(e[0] for e in entries) / n, 2),
            avg_total_sec=round(sum(e[1] for e in entries) / n, 2),
        ))
    return StatsResponse(groups=groups)


@app.post("/assets/ensure", response_model=EnsureAssetsResponse)
def ensure_assets(
    request: EnsureAssetsRequest,
    _: None = Depends(_require_worker_api_token),
) -> EnsureAssetsResponse:
    """Preload one or more asset groups and restart ComfyUI once if needed."""

    requested_groups = [str(group).strip() for group in request.asset_groups if str(group).strip()]
    if not requested_groups:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="asset_groups must not be empty")

    results: list[EnsureAssetGroupResult] = []
    runtime_changed = False
    for asset_group in requested_groups:
        if not _asset_group_allowed(asset_group):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(_asset_group_mismatch_error(asset_group)),
            )
        ensure_result = ensure_asset_group(asset_group)
        if ensure_result.downloaded_assets:
            runtime_changed = True
        if _ensure_runtime_provisioned(asset_group):
            runtime_changed = True
        with _WARMED_GROUPS_LOCK:
            _WARMED_GROUPS.add(asset_group)
        results.append(
            EnsureAssetGroupResult(
                asset_group=asset_group,
                downloaded_assets=ensure_result.downloaded_assets,
                asset_check_sec=ensure_result.asset_check_sec,
                download_sec=ensure_result.download_sec,
            )
        )

    restart_performed = False
    if runtime_changed:
        restart_comfy()
        restart_performed = True

    return EnsureAssetsResponse(
        ok=True,
        results=results,
        restart_performed=restart_performed,
    )


def _active_jobs_detail() -> list[ActiveJobSummary]:
    """Compact summaries of jobs currently executing, for the infra dashboard.

    Built from the async-job registry (the production path); the legacy sync
    /run path doesn't register here, so detail is best-effort and the dashboard
    falls back to the bare active_jobs count when this is empty.
    """
    with _JOB_LOCK:
        running = [
            r for r in _JOBS.values()
            if r.status == "running" and r.progress is not None
        ]
    out: list[ActiveJobSummary] = []
    for r in running:
        p = r.progress
        out.append(ActiveJobSummary(
            job_id=r.client_job_id,
            asset_group=r.asset_group,
            stage=p.stage,
            message=p.message,
            elapsed_sec=round(p.elapsed_sec, 1),
            eta_sec=(round(p.eta_sec, 1) if p.eta_sec is not None else None),
            resolution_bucket=p.resolution_bucket,
        ))
    return out


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return worker and ComfyUI health state."""

    settings = get_settings()
    with _ACTIVE_JOBS_LOCK:
        active_jobs = _ACTIVE_JOBS

    capabilities, infinitetalk_readiness, flux_ipadapter_readiness = _advertised_capabilities()
    auth_ready = _worker_api_auth_ready()

    return HealthResponse(
        ok=auth_ready,
        worker_ok=auth_ready,
        comfy_reachable=is_comfy_healthy(),
        comfy_base_url=settings.comfy_base_url,
        known_asset_groups=sorted(ASSET_REGISTRY),
        worker_name=settings.resolved_worker_name(),
        provider=settings.worker_provider,
        public_url=settings.resolved_worker_public_url(),
        code_release_id=os.getenv("WORKER_CODE_RELEASE_ID") or None,
        gpu_name=settings.worker_gpu_name,
        vram_gb=settings.worker_vram_gb,
        capabilities=capabilities,
        infinitetalk_readiness=infinitetalk_readiness,
        flux_ipadapter_readiness=flux_ipadapter_readiness,
        active_jobs=active_jobs,
        max_concurrent_jobs=settings.resolved_max_concurrent_jobs(),
        download_status=active_download_status(),
        active_jobs_detail=_active_jobs_detail(),
    )


def _free_vram_on_group_switch(asset_group: str) -> None:
    """Release the resident ComfyUI model only when the model *family* changes.

    ComfyUI runs with --highvram on big cards, which pins models on the GPU.
    We used to flush VRAM after every job, which evicted a model we were about
    to reuse — so consecutive same-family shots (all stills, then all clips)
    needlessly reloaded the 30 GB+ model each time. Now we keep it warm within a
    family and only unload when switching (e.g. Flux2 stills → LTX clips), which
    also makes room so the incoming model doesn't OOM on top of the old one.
    """
    global _LAST_ASSET_GROUP
    last = _LAST_ASSET_GROUP
    _LAST_ASSET_GROUP = asset_group
    if last is not None and canonical_asset_group(last) != canonical_asset_group(asset_group):
        LOGGER.info("asset_group switch %s -> %s: releasing resident model", last, asset_group)
        free_comfy_memory(unload_models=True)


def _maybe_upload_primary_output(
    output_files: list[OutputFile],
    output_upload: OutputUploadTarget | None,
    *,
    job_id: str,
) -> list[OutputFile]:
    """ADR-0002 media offload: PUT the primary output straight to storage.

    When ``output_upload`` is set, PUT ``output_files[0]`` to the backend-minted
    signed URL and stamp its ``storage_url`` so the backend records the URL
    instead of downloading + re-uploading. On any failure (or no target / no
    outputs) the list is returned unchanged — ``storage_url`` stays None and the
    backend falls back to its ``/files`` download. Never raises."""
    if output_upload is None or not output_files:
        return output_files
    primary = output_files[0]
    try:
        upload_via_signed_put(output_upload.signed_put_url, primary.path, output_upload.content_type)
    except Exception:  # noqa: BLE001 — degrade to backend download
        LOGGER.warning(
            "[output_upload] job=%s failed — backend will download via /files",
            job_id,
        )
        return output_files
    LOGGER.info("[output_upload] job=%s uploaded primary output", job_id)
    updated = list(output_files)
    updated[0] = primary.model_copy(update={"storage_url": output_upload.public_url})
    return updated


def _run_tts_dialogue(request: RunRequest, total_started: float) -> RunResponse:
    """Dialogue TTS via the resident parler_server (filmforge-parler.service).

    Unlike SA3 there is no per-job model load and no VRAM eviction — the Parler
    model (~4GB) stays resident in its own process on :9101. Payload:
    {text, description?}. Returns a wav under the served output root.
    Self-contained slot/heartbeat accounting, mirroring _run_sa3_audio.
    """
    timings = RunTimings()
    _EXECUTION_SEMAPHORE.acquire()
    with _ACTIVE_JOBS_LOCK:
        global _ACTIVE_JOBS
        _ACTIVE_JOBS += 1
    threading.Thread(target=_send_heartbeat_now, daemon=True).start()
    try:
        payload = request.comfy_payload or {}
        text = str(payload.get("text") or payload.get("prompt") or "").strip()
        if not text:
            raise ValueError("tts_dialogue: empty text")

        out_dir = get_settings().served_file_roots()["output"] / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_job = re.sub(r"[^A-Za-z0-9_.-]", "_", request.job_id)[:80] or "tts"
        out_path = out_dir / f"tts_{safe_job}.wav"

        started = time.monotonic()
        resp = requests.post(
            f"http://127.0.0.1:{os.environ.get('PARLER_PORT', '9101')}/tts",
            json={"text": text, "description": payload.get("description")},
            timeout=min(request.timeout_sec, 600),
        )
        if resp.status_code != 200:
            raise RuntimeError(f"parler_server returned {resp.status_code}: {resp.text[:300]}")
        out_path.write_bytes(resp.content)
        timings.comfy_run_sec = time.monotonic() - started

        output_files = build_output_files([str(out_path)])
        timings.total_sec = time.monotonic() - total_started
        return RunResponse(
            ok=True, job_id=request.job_id, asset_group=request.asset_group,
            downloaded_assets=[], restart_performed=False, comfy_prompt_id=None,
            outputs=[str(out_path)], output_files=output_files, keyframes=[],
            timings=timings,
            debug=RunDebug(history_found=True, comfy_base_url=get_settings().comfy_base_url),
            error=None,
        )
    except Exception as exc:
        LOGGER.exception("tts_dialogue job failed job_id=%s", request.job_id)
        timings.total_sec = time.monotonic() - total_started
        return _error_response(
            request=request, timings=timings, downloaded_assets=[],
            restart_performed=False, comfy_prompt_id=None, history_found=False, error=exc,
        )


@app.post("/tts")
def tts_direct(payload: dict, _: None = Depends(_require_worker_api_token)):
    """Low-latency dialogue TTS for Maya's interactive voice (the rail's /speak).

    Proxies straight to the resident parler_server (:9101), bypassing the broker
    /run job path + slot/heartbeat accounting so Maya's voice never queues behind
    renders. Body: {text, description?}. Returns audio/wav. This is the endpoint
    the rail reaches after discovering this worker via the broker's
    /api/render-broker/voice-worker.
    """
    from fastapi.responses import Response

    text = str((payload or {}).get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    port = os.environ.get("PARLER_PORT", "9101")
    try:
        resp = requests.post(
            f"http://127.0.0.1:{port}/tts",
            json={"text": text, "description": (payload or {}).get("description")},
            timeout=120,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"parler_server unreachable: {exc}")
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"parler_server returned {resp.status_code}: {resp.text[:200]}",
        )
    return Response(content=resp.content, media_type="audio/wav")


def _gpu_free_mib() -> int:
    """Free VRAM on GPU 0 in MiB; 0 on any failure (treated as 'tight')."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def _run_sa3_audio(request: RunRequest, total_started: float) -> RunResponse:
    """Stable Audio 3 generation — provisioner-backed, NOT ComfyUI.

    SA3 is a standalone pip lib in its own venv (see provision_sa3.sh), so instead of
    a ComfyUI graph the payload carries {prompt, seconds} and we invoke sa3_infer.py
    in the sa3_spike venv as a subprocess. Returns an mp3 written under the served
    output root (so the backend fetches it via the normal /files download path).

    Self-contained: owns the execution slot + active-job accounting + heartbeat, so
    the ComfyUI path in _execute_run is untouched.
    """
    timings = RunTimings()
    _EXECUTION_SEMAPHORE.acquire()
    with _ACTIVE_JOBS_LOCK:
        global _ACTIVE_JOBS
        _ACTIVE_JOBS += 1
    threading.Thread(target=_send_heartbeat_now, daemon=True).start()
    try:
        payload = request.comfy_payload or {}
        prompt = str(payload.get("prompt") or payload.get("audio_prompt") or "").strip()
        seconds = float(payload.get("seconds") or payload.get("duration") or 30.0)
        seconds = max(5.0, min(120.0, seconds))  # SA3 tops out at 120s
        if not prompt:
            raise ValueError("stable_audio3: empty prompt")

        out_dir = get_settings().served_file_roots()["output"] / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_job = re.sub(r"[^A-Za-z0-9_.-]", "_", request.job_id)[:80] or "sa3"
        out_path = out_dir / f"sa3_{safe_job}.mp3"

        # Warm path first: the resident sa3_server (filmforge-sa3.service) skips
        # the ~50s per-job model load. On any failure (not running, OOM, card
        # full) fall through to the proven subprocess path below.
        try:
            started = time.monotonic()
            resp = requests.post(
                f"http://127.0.0.1:{os.environ.get('SA3_PORT', '9102')}/music",
                json={"prompt": prompt, "seconds": seconds},
                timeout=min(request.timeout_sec, 900),
            )
            if resp.status_code == 200 and resp.content:
                out_path.write_bytes(resp.content)
                timings.comfy_run_sec = time.monotonic() - started
                output_files = build_output_files([str(out_path)])
                timings.total_sec = time.monotonic() - total_started
                return RunResponse(
                    ok=True, job_id=request.job_id, asset_group=request.asset_group,
                    downloaded_assets=[], restart_performed=False, comfy_prompt_id=None,
                    outputs=[str(out_path)], output_files=output_files, keyframes=[],
                    timings=timings,
                    debug=RunDebug(history_found=True, comfy_base_url=get_settings().comfy_base_url),
                    error=None,
                )
            LOGGER.warning("sa3: warm server returned %s — falling back to subprocess", resp.status_code)
        except requests.RequestException as exc:
            LOGGER.info("sa3: warm server unavailable (%s) — subprocess path", type(exc).__name__)

        # Subprocess fallback loads SA3 fresh (~12GB). Only evict resident comfy
        # models when the card is actually tight — an unconditional eviction
        # forces the next render to reload ~60GB (observed 2026-07-17: audio job
        # stalled the director's render session).
        free_mib = _gpu_free_mib()
        if free_mib < 14000:
            try:
                LOGGER.info("sa3: only %dMiB free — evicting comfy models", free_mib)
                free_comfy_memory()
            except Exception as exc:  # best-effort — SA3 loads its own weights regardless
                LOGGER.warning("sa3: free_comfy_memory failed (%s) — continuing", exc)

        sa3_python = Path(os.environ.get("SA3_VENV", "/workspace/sa3_spike")) / "bin" / "python"
        infer_script = Path(__file__).resolve().parent / "sa3_infer.py"
        if not sa3_python.exists():
            raise RuntimeError(f"stable_audio3 venv not provisioned at {sa3_python} (run provision_sa3.sh)")

        LOGGER.info("sa3: generating %.1fs job_id=%s", seconds, request.job_id)
        started = time.monotonic()
        proc = subprocess.run(
            [str(sa3_python), str(infer_script),
             "--prompt", prompt, "--seconds", str(seconds), "--out", str(out_path)],
            capture_output=True, text=True, timeout=request.timeout_sec,
        )
        timings.comfy_run_sec = time.monotonic() - started
        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(
                f"stable_audio3 inference failed (rc={proc.returncode}): "
                f"{(proc.stderr or '')[-800:]}"
            )

        output_files = build_output_files([str(out_path)])
        timings.total_sec = time.monotonic() - total_started
        return RunResponse(
            ok=True, job_id=request.job_id, asset_group=request.asset_group,
            downloaded_assets=[], restart_performed=False, comfy_prompt_id=None,
            outputs=[str(out_path)], output_files=output_files, keyframes=[],
            timings=timings,
            debug=RunDebug(history_found=True, comfy_base_url=get_settings().comfy_base_url),
            error=None,
        )
    except Exception as exc:
        LOGGER.exception("stable_audio3 job failed job_id=%s", request.job_id)
        timings.total_sec = time.monotonic() - total_started
        return _error_response(
            request=request, timings=timings, downloaded_assets=[],
            restart_performed=False, comfy_prompt_id=None, history_found=False, error=exc,
        )
    finally:
        with _ACTIVE_JOBS_LOCK:
            _ACTIVE_JOBS = max(0, _ACTIVE_JOBS - 1)
        _note_job_completed()
        _EXECUTION_SEMAPHORE.release()
        threading.Thread(target=_send_heartbeat_now, daemon=True).start()


def _execute_run(request: RunRequest, progress: JobProgressResponse | None = None) -> RunResponse:
    """Ensure assets, run a ComfyUI prompt, and report structured results."""

    total_started = time.monotonic()
    timings = RunTimings()
    downloaded_assets: list[str] = []
    restart_performed = False
    prompt_id: str | None = None
    history_found = False
    staged_input_receipts: list[dict[str, str]] = []
    prompt_stage = _stage_for_asset_group(request.asset_group)
    resolution_bucket = _detect_resolution_bucket(request.comfy_payload)
    stage_started = total_started

    if not _asset_group_allowed(request.asset_group):
        timings.total_sec = time.monotonic() - total_started
        return _error_response(
            request=request,
            timings=timings,
            downloaded_assets=downloaded_assets,
            restart_performed=restart_performed,
            comfy_prompt_id=prompt_id,
            history_found=history_found,
            error=_asset_group_mismatch_error(request.asset_group),
        )

    # Provisioner-backed standalone audio (Stable Audio 3) runs its own venv, not
    # ComfyUI — delegate before any comfy-specific VRAM/asset/prompt handling.
    if canonical_asset_group(request.asset_group) == "stable_audio3_v1":
        return _run_sa3_audio(request, total_started)
    if canonical_asset_group(request.asset_group) == "tts_dialogue_v1":
        return _run_tts_dialogue(request, total_started)

    # Pre-flight VRAM check — fast-reject before acquiring execution slot so
    # the broker can immediately route to a worker that has headroom.
    # SKIP it whenever a model is ALREADY resident — the worker manages its own
    # VRAM inside the execution slot: same family → reuse the warm model (needs
    # only activation room); different family → _free_vram_on_group_switch unloads
    # the old model FIRST, freeing the card before the new load. The floor sizes a
    # COLD load and runs BEFORE that unload, so on a family switch the still-resident
    # old model (~37GB) leaves < floor free and falsely rejects a switch that would
    # actually have room. Only gate a GENUINE cold start (no resident model) where
    # free < floor means something else is eating VRAM.
    canonical_group = canonical_asset_group(request.asset_group)
    vram_floor = _effective_vram_floor(canonical_group)
    model_resident = _LAST_ASSET_GROUP is not None
    if vram_floor is not None and not model_resident:
        free_mib = _free_vram_mib()
        if free_mib is not None and free_mib < vram_floor:
            timings.total_sec = time.monotonic() - total_started
            LOGGER.warning(
                "vram_pressure job_id=%s asset_group=%s free=%dMiB required=%dMiB — rejecting",
                request.job_id, request.asset_group, free_mib, vram_floor,
            )
            return _error_response(
                request=request,
                timings=timings,
                downloaded_assets=[],
                restart_performed=False,
                comfy_prompt_id=None,
                history_found=False,
                error=RuntimeError(f"vram_pressure: {free_mib}MiB free < {vram_floor}MiB required for {canonical_group}"),
            )

    if progress is not None:
        _set_progress_stage(progress, "starting", message="Waiting for worker capacity", eta_sec=progress.eta_sec)
        _update_progress_elapsed(
            progress,
            total_started_monotonic=total_started,
            stage_started_monotonic=total_started,
        )

    _EXECUTION_SEMAPHORE.acquire()
    with _ACTIVE_JOBS_LOCK:
        global _ACTIVE_JOBS
        _ACTIVE_JOBS += 1
    threading.Thread(target=_send_heartbeat_now, daemon=True).start()

    try:
        LOGGER.info("Starting job_id=%s asset_group=%s", request.job_id, request.asset_group)
        _free_vram_on_group_switch(request.asset_group)
        if progress is not None:
            _set_progress_stage(progress, "starting", message="Preparing job", eta_sec=progress.eta_sec)
            _update_progress_elapsed(
                progress,
                total_started_monotonic=total_started,
                stage_started_monotonic=total_started,
            )

        ensure_result = ensure_asset_group(request.asset_group)
        downloaded_assets = ensure_result.downloaded_assets
        timings.asset_check_sec = ensure_result.asset_check_sec
        timings.download_sec = ensure_result.download_sec
        with _WARMED_GROUPS_LOCK:
            _WARMED_GROUPS.add(request.asset_group)

        provisioned = _ensure_runtime_provisioned(request.asset_group)
        if downloaded_assets or provisioned:
            timings.restart_sec = restart_comfy()
            restart_performed = True

        prepared_payload, observer_specs = _prepare_comfy_inputs(request)
        expected_prompt_sec = _initial_prompt_eta_sec(request.asset_group, resolution_bucket)
        stage_started = time.monotonic()
        if progress is not None:
            _set_progress_stage(
                progress,
                prompt_stage,
                message=_prompt_stage_message(progress),
                eta_sec=expected_prompt_sec + _FINALIZATION_BUFFER_SEC,
            )
            progress.video_expected_sec = expected_prompt_sec if prompt_stage == "generating_video" else None

        # The paid v2 proof is exactly one Comfy submission.  Legacy groups
        # retain the established single recovery retry, but v2 must return a
        # terminal failure so neither worker nor broker can silently spend on
        # a second render under the same authority.
        max_attempts = (
            1
            if canonical_group == INFINITETALK_TWO_PERSON_V2_ASSET_GROUP
            else 2
        )
        for attempt in range(max_attempts):
            try:
                comfy_started = time.monotonic()
                staged_input_receipts = observe_staged_input_receipts(
                    prepared_payload,
                    observer_specs,
                )
                prompt_id = submit_prompt(prepared_payload)
                history = poll_for_completion(
                    prompt_id=prompt_id,
                    timeout_sec=request.timeout_sec,
                    poll_interval_sec=request.poll_interval_sec,
                    progress_callback=(
                        None if progress is None else
                        lambda elapsed_sec: _update_progress_elapsed(
                            progress,
                            total_started_monotonic=total_started,
                            stage_started_monotonic=stage_started,
                            eta_sec=(
                                max(expected_prompt_sec - elapsed_sec, 0.0) + _FINALIZATION_BUFFER_SEC
                                if elapsed_sec <= expected_prompt_sec
                                else _extended_remaining_sec(expected_prompt_sec) + _FINALIZATION_BUFFER_SEC
                            ),
                            message=_prompt_stage_message(progress),
                        )
                    ),
                )
                timings.comfy_run_sec = time.monotonic() - comfy_started
                history_found = True
                break  # success — exit retry loop

            except Exception as exc:
                is_oom = isinstance(exc, ComfyExecutionError) and exc.is_oom
                is_restart_detected = isinstance(exc, ComfyRestartDetectedError)
                needs_restart = is_oom or is_restart_detected or not is_comfy_healthy()

                if (
                    attempt + 1 < max_attempts
                    and needs_restart
                    and get_settings().comfy_start_cmd
                ):
                    reason = "OOM" if is_oom else ("ComfyUI restart detected" if is_restart_detected else type(exc).__name__)
                    LOGGER.warning(
                        "job_id=%s attempt %d failed (%s) — restarting ComfyUI and retrying",
                        request.job_id, attempt + 1,
                        reason,
                    )
                    try:
                        if progress is not None:
                            _set_progress_stage(
                                progress,
                                "starting",
                                message="Restarting ComfyUI after failed attempt",
                                eta_sec=_extended_remaining_sec(expected_prompt_sec) + _FINALIZATION_BUFFER_SEC,
                            )
                            _update_progress_elapsed(
                                progress,
                                total_started_monotonic=total_started,
                                stage_started_monotonic=time.monotonic(),
                            )
                        restart_comfy()
                        restart_performed = True
                        LOGGER.info("ComfyUI restarted — retrying job_id=%s", request.job_id)
                        stage_started = time.monotonic()
                        if progress is not None:
                            _set_progress_stage(
                                progress,
                                prompt_stage,
                                message=_prompt_stage_message(progress),
                                eta_sec=expected_prompt_sec + _FINALIZATION_BUFFER_SEC,
                            )
                    except Exception as restart_exc:
                        LOGGER.error("ComfyUI restart failed: %s — giving up", restart_exc)
                        timings.total_sec = time.monotonic() - total_started
                        return _error_response(
                            request=request, timings=timings,
                            downloaded_assets=downloaded_assets,
                            restart_performed=restart_performed,
                            comfy_prompt_id=prompt_id, history_found=history_found,
                            error=exc,
                        )
                    continue  # go to attempt 1

                # Non-recoverable or second attempt failed
                LOGGER.exception("Worker run failed for job_id=%s (attempt %d)", request.job_id, attempt + 1)
                timings.total_sec = time.monotonic() - total_started
                return _error_response(
                    request=request, timings=timings,
                    downloaded_assets=downloaded_assets,
                    restart_performed=restart_performed,
                    comfy_prompt_id=prompt_id, history_found=history_found,
                    error=exc,
                )

        if progress is not None:
            if prompt_stage == "generating_stills":
                progress.num_stills_done = progress.num_stills_total
                progress.still_avg_sec = timings.comfy_run_sec
            stage_started = time.monotonic()
            _set_progress_stage(
                progress,
                "finalizing",
                message="Collecting outputs",
                eta_sec=_FINALIZATION_BUFFER_SEC,
            )
            _update_progress_elapsed(
                progress,
                total_started_monotonic=total_started,
                stage_started_monotonic=stage_started,
                eta_sec=_FINALIZATION_BUFFER_SEC,
            )

        outputs = collect_output_paths(history)
        output_files = build_output_files(outputs)
        outputs, postprocess_evidence = postprocess_two_person_v2_outputs(
            [output.path for output in output_files],
            routing=request.infinitetalk_routing,
            source_specs=request.comfy_input_files,
            observer_specs=observer_specs,
            job_id=request.job_id,
        )
        output_files = build_output_files(outputs)

        # Extract observation keyframes here (the worker has the video on disk +
        # ffmpeg) so the backend never re-downloads + ffmpeg-decodes the clip.
        keyframes: list[ClipKeyframes] = []
        for of in output_files:
            if not is_video_output(of.path):
                continue
            try:
                frames = extract_keyframes_b64(of.path)
            except Exception as exc:  # never fail a render over keyframes
                LOGGER.warning("keyframe extraction failed for %s: %s", of.filename, exc)
                frames = []
            if frames:
                keyframes.append(ClipKeyframes(
                    output_filename=of.filename,
                    frames=[ClipKeyframe(**f) for f in frames],
                ))

        # Media offload (ADR-0002): when the backend minted a signed upload URL,
        # PUT the primary output straight to storage so the 2 GB API box never
        # downloads + re-uploads it. On any failure the output keeps storage_url
        # unset and the backend falls back to its /files download + re-upload.
        output_files = _maybe_upload_primary_output(
            output_files, request.output_upload, job_id=request.job_id
        )

        timings.total_sec = time.monotonic() - total_started

        # NB: we deliberately do NOT flush VRAM here. With --highvram the model
        # stays pinned, so a per-job flush only forces the next same-family shot
        # to reload it. VRAM is instead released on model-family switch, at job
        # start (see _free_vram_on_group_switch).

        _store_eta_sample(request.asset_group, resolution_bucket, timings.comfy_run_sec)
        routing_receipt = build_two_person_routing_receipt(
            request.infinitetalk_routing,
            staged_input_receipts,
            source_specs=request.comfy_input_files,
            job_id=request.job_id,
            postprocess_evidence=postprocess_evidence,
        )

        return RunResponse(
            ok=True,
            job_id=request.job_id,
            asset_group=request.asset_group,
            downloaded_assets=downloaded_assets,
            restart_performed=restart_performed,
            comfy_prompt_id=prompt_id,
            outputs=outputs,
            output_files=output_files,
            staged_input_receipts=staged_input_receipts,
            infinitetalk_routing_receipt=routing_receipt,
            keyframes=keyframes,
            timings=timings,
            debug=RunDebug(
                history_found=history_found,
                comfy_base_url=get_settings().comfy_base_url,
            ),
            error=None,
        )

    except Exception as exc:
        LOGGER.exception("Unexpected error for job_id=%s", request.job_id)
        timings.total_sec = time.monotonic() - total_started
        return _error_response(
            request=request, timings=timings,
            downloaded_assets=downloaded_assets,
            restart_performed=restart_performed,
            comfy_prompt_id=prompt_id, history_found=history_found,
            error=exc,
        )
    finally:
        with _ACTIVE_JOBS_LOCK:
            _ACTIVE_JOBS = max(0, _ACTIVE_JOBS - 1)
        _note_job_completed()
        _EXECUTION_SEMAPHORE.release()
        threading.Thread(target=_send_heartbeat_now, daemon=True).start()


def _run_job_async(worker_job_id: str) -> None:
    """Execute a queued async job and store its result."""

    with _JOB_LOCK:
        job = _JOBS.get(worker_job_id)
        if job is None or job.request is None:
            return
        job.status = "running"
        request = job.request
        # The executing thread now owns the request. Do not retain a second
        # potentially large inline-body graph in the poll registry.
        job.request = None
        progress = job.progress

    result = _execute_run(request, progress=progress)

    with _JOB_LOCK:
        job = _JOBS.get(worker_job_id)
        if job is None:
            return
        job.result = result
        job.error = result.error
        job.status = "completed" if result.ok else "failed"
        job.finished_monotonic = time.monotonic()
        if job.progress is not None:
            stage = "done" if result.ok else "failed"
            message = "Completed" if result.ok else (result.error or "Failed")
            _set_progress_stage(job.progress, stage, message=message, eta_sec=0.0)
            job.progress.elapsed_sec = max(result.timings.total_sec, job.progress.elapsed_sec)
            job.progress.stage_elapsed_sec = 0.0

    if result.ok and result.timings:
        ag = request.asset_group
        entry = (result.timings.comfy_run_sec, result.timings.total_sec)
        with _STATS_LOCK:
            if ag not in _STATS:
                _STATS[ag] = deque(maxlen=_STATS_WINDOW)
            _STATS[ag].append(entry)


@app.post("/run", response_model=RunResponse)
def run_job(request: RunRequest, _: None = Depends(_require_worker_api_token)) -> RunResponse:
    """Run a job synchronously and return the final worker response."""

    return _execute_run(request)


@app.post("/stitch", response_model=StitchResponse)
def stitch_job(request: StitchRequest, _: None = Depends(_require_worker_api_token)) -> StitchResponse:
    """Stitch a rough cut on the box and upload it straight to storage.

    Keeps the API backend out of the video path entirely — no clip downloads,
    no ffmpeg, no bytes through it. Always cleans up its temp dir.
    """
    work_dir = tempfile.mkdtemp(prefix="ff_stitch_")
    try:
        out_path, meta = stitch_clips(
            clips=[c.model_dump() for c in request.clips],
            audio_url=request.audio_url,
            width=request.width,
            height=request.height,
            fps=request.fps,
            work_dir=work_dir,
        )
        upload_via_signed_put(request.signed_put_url, out_path, request.content_type)
        return StitchResponse(ok=True, job_id=request.job_id, public_url=request.public_url, metadata=meta)
    except Exception as exc:  # noqa: BLE001 — return ok=False so the backend can fall back
        LOGGER.error(
            "[stitch] failed job=%s exception_class=%s",
            request.job_id,
            type(exc).__name__,
        )
        return StitchResponse(ok=False, job_id=request.job_id, error="stitch_failed")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/jobs", response_model=JobSubmitResponse)
def submit_job(request: RunRequest, _: None = Depends(_require_worker_api_token)) -> JobSubmitResponse:
    """Accept a worker run for async execution."""

    worker_job_id = uuid.uuid4().hex
    record = _JobRecord(
        job_id=worker_job_id,
        request=request,
        client_job_id=request.job_id,
        asset_group=request.asset_group,
        progress=_build_initial_progress(worker_job_id, request),
    )
    with _JOB_LOCK:
        in_flight = sum(
            record.status not in _TERMINAL_JOB_STATES
            for record in _JOBS.values()
        )
        if in_flight >= _MAX_EXECUTION_SLOTS:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Async worker execution capacity is full",
            )
        if not _prune_job_registry_locked(reserve=1):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Async job registry is at capacity",
            )
        _JOBS[worker_job_id] = record

    thread = threading.Thread(
        target=_run_job_async,
        args=(worker_job_id,),
        name=f"gpu-worker-{worker_job_id[:8]}",
        daemon=True,
    )
    thread.start()
    return JobSubmitResponse(job_id=worker_job_id, status=record.status)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(
    job_id: str,
    _: None = Depends(_require_worker_api_token),
) -> JobStatusResponse:
    """Return async job state and final result when available."""

    with _JOB_LOCK:
        _prune_job_registry_locked()
        record = _JOBS.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
        return JobStatusResponse(
            job_id=record.job_id,
            status=record.status,
            result=record.result,
            error=record.error,
            progress=record.progress,
        )


@app.get("/jobs/{job_id}/progress", response_model=JobProgressResponse)
def get_job_progress(
    job_id: str,
    _: None = Depends(_require_worker_api_token),
) -> JobProgressResponse:
    """Return structured execution progress and ETA for one async worker job."""

    with _JOB_LOCK:
        _prune_job_registry_locked()
        record = _JOBS.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
        if record.progress is None:
            raise HTTPException(status_code=404, detail=f"No progress tracked for job_id: {job_id}")
        return record.progress


@app.get("/files/{root_name}/{relative_path:path}")
def download_file(
    root_name: str,
    relative_path: str,
    _: None = Depends(_require_worker_api_token),
) -> FileResponse:
    """Serve a generated file from an explicitly allowed Comfy directory."""

    try:
        path = resolve_served_file(root_name, relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {relative_path}")

    return FileResponse(path=path, filename=path.name)
