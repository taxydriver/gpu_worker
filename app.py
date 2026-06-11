"""FastAPI entrypoint for the FilmForge GPU worker."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse
import requests

from gpu_worker.asset_canonical import canonical_asset_group, canonicalize_groups
from gpu_worker.asset_manager import active_download_status, ensure_asset_group, is_asset_group_warm
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
from gpu_worker.schemas import (
    EnsureAssetGroupResult,
    EnsureAssetsRequest,
    EnsureAssetsResponse,
    AssetGroupStats,
    HealthResponse,
    JobProgressResponse,
    JobStatusResponse,
    JobSubmitResponse,
    RunDebug,
    RunRequest,
    RunResponse,
    RunTimings,
    StatsResponse,
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

# Run GPU diagnostics and smoke test in a background thread so they don't
# block uvicorn startup. Results appear in the worker log (journalctl / /tmp/gpu_worker.log).
threading.Thread(
    target=lambda: (run_gpu_diagnostics(), run_gpu_smoke_test()),
    name="gpu-startup-check",
    daemon=True,
).start()

_STILL_ASSET_GROUPS = {"flux_stills_v1"}
_VIDEO_ASSET_GROUPS = {"wan_i2v_v1"}
_FINALIZATION_BUFFER_SEC = 10.0
_BASE_STILL_SEC = 60.0
_BASE_VIDEO_SEC = 30.0
_ETA_WINDOW = 12

# ── Active-job tracking (used by watchdog to avoid restarting mid-run) ────────
_ACTIVE_JOBS_LOCK = threading.Lock()
_ACTIVE_JOBS: int = 0  # count of jobs currently executing
_MAX_EXECUTION_SLOTS = max(1, get_settings().resolved_max_concurrent_jobs())
_EXECUTION_SEMAPHORE = threading.BoundedSemaphore(value=_MAX_EXECUTION_SLOTS)

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


def _watchdog_loop() -> None:
    """Background thread: restart ComfyUI when it goes unhealthy and no job is running."""

    settings = get_settings()
    if not settings.comfy_start_cmd:
        LOGGER.info("[watchdog] COMFY_START_CMD not set — watchdog disabled")
        return

    LOGGER.info("[watchdog] started, checking every 30s")
    while True:
        time.sleep(30)
        try:
            if is_comfy_healthy():
                continue

            with _ACTIVE_JOBS_LOCK:
                active = _ACTIVE_JOBS

            if active > 0:
                LOGGER.warning("[watchdog] ComfyUI unhealthy but %d job(s) active — waiting", active)
                continue

            LOGGER.warning("[watchdog] ComfyUI unhealthy and idle — restarting")
            restart_comfy()
            LOGGER.info("[watchdog] ComfyUI restarted successfully")
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
    expected_token = (settings.worker_api_token or "").strip()
    if not expected_token:
        return

    bearer_token: str | None = None
    if authorization:
        parts = authorization.strip().split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            bearer_token = parts[1].strip()

    provided = bearer_token or ""
    if provided != expected_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid worker API token")


def _broker_headers() -> dict[str, str]:
    settings = get_settings()
    headers = {"Content-Type": "application/json"}
    token = (settings.resolved_registration_token() or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-Render-Broker-Token"] = token
    return headers


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

    free_vram_mb = None
    if settings.worker_vram_gb is not None:
        free_vram_mb = int(settings.worker_vram_gb * 1024)

    raw_capabilities = settings.resolved_capabilities() or sorted(ASSET_REGISTRY)
    capabilities = canonicalize_groups(raw_capabilities)
    public_url = settings.resolved_worker_public_url()

    with _WARMED_GROUPS_LOCK:
        warmed = canonicalize_groups(sorted(_WARMED_GROUPS))

    max_concurrent_jobs = settings.resolved_max_concurrent_jobs()
    comfy_queue = comfy_queue_depth()
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
        "metadata": {
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
        },
    }


def _register_with_broker() -> None:
    settings = get_settings()
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


def _broker_heartbeat_loop() -> None:
    settings = get_settings()
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

    downloaded_any = False
    failed_groups: list[str] = []
    for group in groups:
        try:
            result = ensure_asset_group(group)
            if result.downloaded_assets:
                downloaded_any = True
                LOGGER.info(
                    "[preflight] %s — downloaded %s in %.1fs",
                    group, result.downloaded_assets, result.download_sec,
                )
            else:
                LOGGER.info("[preflight] %s — already cached (%.2fs check)", group, result.asset_check_sec)
            # Mark this asset group as warmed
            with _WARMED_GROUPS_LOCK:
                _WARMED_GROUPS.add(group)
        except Exception as exc:
            failed_groups.append(group)
            LOGGER.error("[preflight] failed to ensure group=%s: %s", group, exc)

    if failed_groups:
        LOGGER.error("[preflight] incomplete; failed group(s): %s", failed_groups)
        return

    if downloaded_any:
        with _ACTIVE_JOBS_LOCK:
            active = _ACTIVE_JOBS
        if active > 0:
            LOGGER.info(
                "[preflight] new models downloaded but %d job(s) active — deferring ComfyUI restart",
                active,
            )
        else:
            LOGGER.info("[preflight] new models downloaded — restarting ComfyUI once")
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
    request: RunRequest
    status: str = "queued"
    result: RunResponse | None = None
    error: str | None = None
    progress: JobProgressResponse | None = None


_JOB_LOCK = threading.Lock()
_JOBS: dict[str, _JobRecord] = {}


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
    downloaded_any = False
    for asset_group in requested_groups:
        if not _asset_group_allowed(asset_group):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(_asset_group_mismatch_error(asset_group)),
            )
        ensure_result = ensure_asset_group(asset_group)
        if ensure_result.downloaded_assets:
            downloaded_any = True
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
    if downloaded_any:
        restart_comfy()
        restart_performed = True

    return EnsureAssetsResponse(
        ok=True,
        results=results,
        restart_performed=restart_performed,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return worker and ComfyUI health state."""

    settings = get_settings()
    with _ACTIVE_JOBS_LOCK:
        active_jobs = _ACTIVE_JOBS

    capabilities = settings.resolved_capabilities() or sorted(ASSET_REGISTRY)

    return HealthResponse(
        ok=True,
        worker_ok=True,
        comfy_reachable=is_comfy_healthy(),
        comfy_base_url=settings.comfy_base_url,
        known_asset_groups=sorted(ASSET_REGISTRY),
        worker_name=settings.resolved_worker_name(),
        provider=settings.worker_provider,
        public_url=settings.resolved_worker_public_url(),
        gpu_name=settings.worker_gpu_name,
        vram_gb=settings.worker_vram_gb,
        capabilities=capabilities,
        active_jobs=active_jobs,
        max_concurrent_jobs=settings.resolved_max_concurrent_jobs(),
        download_status=active_download_status(),
    )


def _execute_run(request: RunRequest, progress: JobProgressResponse | None = None) -> RunResponse:
    """Ensure assets, run a ComfyUI prompt, and report structured results."""

    total_started = time.monotonic()
    timings = RunTimings()
    downloaded_assets: list[str] = []
    restart_performed = False
    prompt_id: str | None = None
    history_found = False
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

    try:
        LOGGER.info("Starting job_id=%s asset_group=%s", request.job_id, request.asset_group)
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

        if downloaded_assets:
            timings.restart_sec = restart_comfy()
            restart_performed = True

        prepared_payload = apply_comfy_input_files(request.comfy_payload, request.comfy_input_files)
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

        # Attempt the ComfyUI run; on OOM or crash, restart and retry once.
        for attempt in range(2):
            try:
                comfy_started = time.monotonic()
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

                if attempt == 0 and needs_restart and get_settings().comfy_start_cmd:
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
        timings.total_sec = time.monotonic() - total_started

        # Release VRAM after every job so it doesn't accumulate across shots
        free_comfy_memory(unload_models=False)  # keep models loaded, just flush cache

        _store_eta_sample(request.asset_group, resolution_bucket, timings.comfy_run_sec)

        return RunResponse(
            ok=True,
            job_id=request.job_id,
            asset_group=request.asset_group,
            downloaded_assets=downloaded_assets,
            restart_performed=restart_performed,
            comfy_prompt_id=prompt_id,
            outputs=outputs,
            output_files=output_files,
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
        _EXECUTION_SEMAPHORE.release()


def _run_job_async(worker_job_id: str) -> None:
    """Execute a queued async job and store its result."""

    with _JOB_LOCK:
        job = _JOBS.get(worker_job_id)
        if job is None:
            return
        job.status = "running"
        request = job.request
        progress = job.progress

    result = _execute_run(request, progress=progress)

    with _JOB_LOCK:
        job = _JOBS.get(worker_job_id)
        if job is None:
            return
        job.result = result
        job.error = result.error
        job.status = "completed" if result.ok else "failed"
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


@app.post("/jobs", response_model=JobSubmitResponse)
def submit_job(request: RunRequest, _: None = Depends(_require_worker_api_token)) -> JobSubmitResponse:
    """Accept a worker run for async execution."""

    worker_job_id = uuid.uuid4().hex
    record = _JobRecord(
        job_id=worker_job_id,
        request=request,
        progress=_build_initial_progress(worker_job_id, request),
    )
    with _JOB_LOCK:
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
def get_job(job_id: str) -> JobStatusResponse:
    """Return async job state and final result when available."""

    with _JOB_LOCK:
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
def get_job_progress(job_id: str) -> JobProgressResponse:
    """Return structured execution progress and ETA for one async worker job."""

    with _JOB_LOCK:
        record = _JOBS.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
        if record.progress is None:
            raise HTTPException(status_code=404, detail=f"No progress tracked for job_id: {job_id}")
        return record.progress


@app.get("/files/{root_name}/{relative_path:path}")
def download_file(root_name: str, relative_path: str) -> FileResponse:
    """Serve a generated file from an explicitly allowed Comfy directory."""

    try:
        path = resolve_served_file(root_name, relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {relative_path}")

    return FileResponse(path=path, filename=path.name)
