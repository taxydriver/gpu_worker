"""Pydantic schemas for the GPU worker API."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field


class ActiveJobSummary(BaseModel):
    """Compact view of one in-flight job, surfaced on /health so the infra
    dashboard can show *what* a box is generating (and how far along), not just
    a count. Fields mirror the live JobProgressResponse for running jobs."""

    job_id: str
    asset_group: str
    stage: str
    message: str = ""
    elapsed_sec: float = 0.0
    eta_sec: float | None = None
    resolution_bucket: str | None = None


class HealthResponse(BaseModel):
    """Health response payload."""

    ok: bool = True
    worker_ok: bool = True
    comfy_reachable: bool
    comfy_base_url: str
    known_asset_groups: list[str]
    worker_name: str | None = None
    provider: str = "dedicated_worker"
    public_url: str | None = None
    gpu_name: str | None = None
    vram_gb: float | None = None
    capabilities: list[str] = Field(default_factory=list)
    active_jobs: int = 0
    max_concurrent_jobs: int = 1
    download_status: dict[str, Any] | None = None
    active_jobs_detail: list[ActiveJobSummary] = Field(default_factory=list)


class AssetGroupStats(BaseModel):
    """Rolling timing stats for one asset group."""

    asset_group: str
    sample_count: int
    avg_comfy_run_sec: float
    avg_total_sec: float


class StatsResponse(BaseModel):
    """Aggregated timing stats across all asset groups."""

    groups: list[AssetGroupStats]


class EnsureAssetsRequest(BaseModel):
    """Request payload for preloading one or more asset groups."""

    asset_groups: list[str] = Field(default_factory=list)


class EnsureAssetGroupResult(BaseModel):
    """Warmup result for one asset group."""

    asset_group: str
    downloaded_assets: list[str] = Field(default_factory=list)
    asset_check_sec: float = 0.0
    download_sec: float = 0.0


class EnsureAssetsResponse(BaseModel):
    """Response payload for asset warmup."""

    ok: bool = True
    results: list[EnsureAssetGroupResult] = Field(default_factory=list)
    restart_performed: bool = False


class OutputUploadTarget(BaseModel):
    """Where the worker should PUT its produced output (ADR-0002 media-offload).

    The backend mints a Supabase signed upload URL for a deterministic path and
    passes it in; the worker PUTs the file there with **no storage credentials of
    its own** and echoes ``public_url`` back on the OutputFile so the backend
    records the URL without downloading + re-uploading. Same creds-off-the-box
    pattern as ``/stitch``. Absent → the worker behaves exactly as before (the
    backend downloads via ``/files``)."""

    signed_put_url: str
    public_url: str
    content_type: str = "application/octet-stream"


class RunRequest(BaseModel):
    """Request payload for a worker run."""

    job_id: str
    asset_group: str
    comfy_payload: dict[str, Any] = Field(description="Raw ComfyUI workflow payload.")
    comfy_input_files: list["ComfyInputFile"] = Field(default_factory=list)
    timeout_sec: int = 3600
    poll_interval_sec: float = 2.0
    # When set, the worker uploads its primary output straight to storage.
    output_upload: "OutputUploadTarget | None" = None


class RunTimings(BaseModel):
    """Timing breakdown for a worker run."""

    asset_check_sec: float = 0.0
    download_sec: float = 0.0
    restart_sec: float = 0.0
    comfy_run_sec: float = 0.0
    total_sec: float = 0.0


class RunDebug(BaseModel):
    """Debug details for a worker run."""

    history_found: bool
    comfy_base_url: str


class OutputFile(BaseModel):
    """Downloadable metadata for one worker-produced file."""

    path: str
    filename: str
    download_url: str
    # Set when the worker uploaded this output straight to storage (option A):
    # the public URL the backend records instead of downloading via download_url.
    storage_url: str | None = None


class ComfyInputFile(BaseModel):
    """File the worker should stage into ComfyUI input before prompt submission."""

    node_id: str
    filename: str
    input_name: str = "image"
    source_path: str | None = None
    source_url: str | None = None
    source_data: str | None = None  # base64-encoded file bytes (used when URL is not remotely reachable)
    type: str = "input"
    subfolder: str = ""


class ClipKeyframe(BaseModel):
    """One sampled frame from a video output, carried inline so the backend
    never has to download + ffmpeg-decode the clip itself."""

    timestamp_sec: float
    image_b64: str
    mime: str = "image/png"


class ClipKeyframes(BaseModel):
    """Keyframes the worker extracted for one video output."""

    output_filename: str
    frames: list[ClipKeyframe] = Field(default_factory=list)


class RunResponse(BaseModel):
    """Structured response for a worker run."""

    ok: bool
    job_id: str
    asset_group: str
    downloaded_assets: list[str]
    restart_performed: bool
    comfy_prompt_id: str | None
    outputs: list[str]
    output_files: list[OutputFile] = Field(default_factory=list)
    # Worker-extracted keyframes for any video outputs (inline base64). The
    # backend uses these for observation instead of re-downloading + ffmpeg.
    keyframes: list[ClipKeyframes] = Field(default_factory=list)
    timings: RunTimings
    debug: RunDebug
    error: str | None = None


class JobProgressResponse(BaseModel):
    """Execution progress snapshot for an active or completed worker job."""

    job_id: str
    stage: Literal[
        "queued",
        "starting",
        "generating_stills",
        "generating_video",
        "finalizing",
        "done",
        "failed",
    ]
    message: str = ""
    started_at: float | None = None
    stage_started_at: float | None = None
    elapsed_sec: float = 0.0
    stage_elapsed_sec: float = 0.0
    eta_sec: float | None = None
    num_stills_total: int = 0
    num_stills_done: int = 0
    still_avg_sec: float | None = None
    video_expected_sec: float | None = None
    resolution_bucket: Literal["low", "medium", "high"] | None = None


class JobSubmitResponse(BaseModel):
    """Accepted async job submission."""

    job_id: str
    status: Literal["queued", "running", "completed", "failed"]


class JobStatusResponse(BaseModel):
    """Async job polling response."""

    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    result: RunResponse | None = None
    error: str | None = None
    progress: JobProgressResponse | None = None


class StitchClip(BaseModel):
    """One clip in the ordered stitch sequence (fetched by URL on the worker)."""

    url: str
    edit_in_sec: float | None = None
    edit_out_sec: float | None = None
    target_duration_sec: float | None = None
    source_duration_sec: float | None = None


class StitchRequest(BaseModel):
    """Stitch a rough cut on the worker and upload it straight to storage.

    The backend fetches nothing and stores no bytes: it mints ``signed_put_url``
    (a Supabase signed upload URL for a deterministic path) and the worker PUTs
    the finished video there. v1 = video concat + score-only audio mux; foley is
    deferred (backend will pass a pre-resolved filter spec later).
    """

    job_id: str
    clips: list[StitchClip]
    audio_url: str | None = None
    width: int = 768
    height: int = 432
    fps: int = 24
    signed_put_url: str
    public_url: str | None = None
    content_type: str = "video/mp4"


class StitchResponse(BaseModel):
    """Result of a worker stitch. ``ok=False`` lets the backend fall back."""

    ok: bool
    job_id: str
    public_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
