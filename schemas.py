"""Pydantic schemas for the GPU worker API."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field


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


class RunRequest(BaseModel):
    """Request payload for a worker run."""

    job_id: str
    asset_group: str
    comfy_payload: dict[str, Any] = Field(description="Raw ComfyUI workflow payload.")
    comfy_input_files: list["ComfyInputFile"] = Field(default_factory=list)
    timeout_sec: int = 3600
    poll_interval_sec: float = 2.0


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
