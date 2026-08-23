"""Pydantic schemas for the GPU worker API."""

from __future__ import annotations

import math
from typing import Annotated, Any
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    code_release_id: str | None = None
    gpu_name: str | None = None
    vram_gb: float | None = None
    capabilities: list[str] = Field(default_factory=list)
    infinitetalk_readiness: dict[str, Any] | None = None
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


class InfiniteTalkSourceDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)


class InfiniteTalkTwoPersonRouting(BaseModel):
    """Frozen A2 authority for one speaker and one silent listener.

    Regions are semantic authorities, not caller-supplied mask files.  The
    worker rasterizes them against the attested source still and owns every
    graph-bound mask and silence byte.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "infinitetalk_two_person_routing_v1",
        "infinitetalk_two_person_routing_v2",
    ]
    mode: Literal["two_person_parallel"]
    multi_audio_type: Literal["para"]
    speaker_slot: Literal[1, 2]
    listener_slot: Literal[1, 2]
    slot_regions: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    speaker_region: tuple[float, float, float, float]
    listener_region: tuple[float, float, float, float]
    coordinate_space: Literal["normalized_0_1"]
    source_still_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dimensions: InfiniteTalkSourceDimensions
    spatial_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_duration_sec: float = Field(gt=0, le=15.0)
    listener_audio_kind: Literal[
        "silence_pcm",
        "deterministic_pink_roomtone_pcm_s16le_16000_mono_v1",
    ]
    conditioning_lead_in_frames: int | None = Field(default=None, ge=0)
    effective_video_frames: int | None = Field(default=None, ge=81)
    effective_duration_sec: float | None = Field(default=None, gt=0, le=30.0)
    delivered_video_frames: int | None = Field(default=None, gt=0)
    delivered_duration_sec: float | None = Field(default=None, gt=0, le=30.0)

    @field_validator("speaker_region", "listener_region")
    @classmethod
    def _validate_region(cls, value: tuple[float, float, float, float]):
        if any(not math.isfinite(coordinate) for coordinate in value):
            raise ValueError("routing region coordinates must be finite")
        x0, y0, x1, y1 = value
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError("routing regions must be positive normalized xyxy boxes")
        return value

    @model_validator(mode="after")
    def _validate_two_person_authority(self):
        if {self.speaker_slot, self.listener_slot} != {1, 2}:
            raise ValueError("speaker_slot and listener_slot must be exactly {1, 2}")
        sx0, sy0, sx1, sy1 = self.speaker_region
        lx0, ly0, lx1, ly1 = self.listener_region
        overlaps = sx0 < lx1 and lx0 < sx1 and sy0 < ly1 and ly0 < sy1
        if overlaps:
            raise ValueError("speaker and listener routing regions must not overlap")
        if self.slot_regions[self.speaker_slot - 1] != self.speaker_region:
            raise ValueError("speaker_region must equal its ordered slot_regions entry")
        if self.slot_regions[self.listener_slot - 1] != self.listener_region:
            raise ValueError("listener_region must equal its ordered slot_regions entry")
        if self.schema_version == "infinitetalk_two_person_routing_v1":
            if self.listener_audio_kind != "silence_pcm":
                raise ValueError("v1 routing requires silence_pcm listener conditioning")
            if any(
                value is not None
                for value in (
                    self.conditioning_lead_in_frames,
                    self.effective_video_frames,
                    self.effective_duration_sec,
                    self.delivered_video_frames,
                    self.delivered_duration_sec,
                )
            ):
                raise ValueError("v1 routing may not carry v2 effective-duration authority")
        else:
            if self.listener_audio_kind != "deterministic_pink_roomtone_pcm_s16le_16000_mono_v1":
                raise ValueError("v2 routing requires deterministic pink roomtone conditioning")
            if self.conditioning_lead_in_frames != 3:
                raise ValueError("v2 routing requires an exact three-frame conditioning lead-in")
            if any(
                value is None
                for value in (
                    self.effective_video_frames,
                    self.effective_duration_sec,
                    self.delivered_video_frames,
                    self.delivered_duration_sec,
                )
            ):
                raise ValueError("v2 routing requires exact effective and delivered duration authority")
            if (self.effective_video_frames - 1) % 4:
                raise ValueError("v2 effective_video_frames must be 4n+1")
            if not math.isclose(
                self.effective_duration_sec,
                self.effective_video_frames / 25.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("v2 effective duration must equal effective_video_frames / 25")
            if self.delivered_video_frames != self.effective_video_frames - 3:
                raise ValueError("v2 delivered frames must exclude the conditioning lead-in")
            if not math.isclose(
                self.delivered_duration_sec,
                self.delivered_video_frames / 25.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("v2 delivered duration must equal delivered_video_frames / 25")
        return self


class InfiniteTalkMaskHashes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_1: str = Field(pattern=r"^[0-9a-f]{64}$")
    slot_2: str = Field(pattern=r"^[0-9a-f]{64}$")
    background: str = Field(pattern=r"^[0-9a-f]{64}$")


class InfiniteTalkSpeakerConditioning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roomtone_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_count: int = Field(gt=0)
    sample_rate: Literal[16000]
    channels: Literal[1]
    overlay_start_sample: int = Field(ge=0)
    overlay_frame_count: int = Field(gt=0)
    kind: Literal["approved_take_over_deterministic_pink_roomtone_pcm_s16le_16000_mono_v1"]


class InfiniteTalkListenerConditioning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_count: int = Field(gt=0)
    sample_rate: Literal[16000]
    channels: Literal[1]
    kind: Literal["deterministic_pink_roomtone_pcm_s16le_16000_mono_v1"]
    usage: Literal["conditioning_only_not_audible"]
    rms: float = Field(ge=0.00020, le=0.00035)
    peak: float = Field(ge=0.00080, le=0.00130)

    @model_validator(mode="after")
    def _validate_levels(self):
        if not math.isfinite(self.rms) or not math.isfinite(self.peak):
            raise ValueError("listener conditioning levels must be finite")
        if self.rms > self.peak:
            raise ValueError("listener conditioning RMS may not exceed peak")
        return self


class InfiniteTalkPostprocessPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_policy: Literal["trim_exact_conditioning_lead_in_frames_v1"]
    audio_policy: Literal["approved_take_pcm_then_silence_tail_remux_v1"]


class InfiniteTalkFinalAudio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_wav_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_count: int = Field(gt=0)
    sample_rate: Literal[16000]
    channels: Literal[1]
    kind: Literal["approved_take_canonical_wav_s16le_16000_mono_zero_tail_v1"]
    decoded_tail_rms: float = Field(ge=0, le=0.00010)

    @field_validator("decoded_tail_rms")
    @classmethod
    def _validate_decoded_tail_rms(cls, value: float):
        if not math.isfinite(value):
            raise ValueError("decoded audible-tail RMS must be finite")
        return value


class InfiniteTalkRoutingReceiptV1(BaseModel):
    """Locator-free proof of the exact A2-v1 routing bytes submitted to Comfy."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["infinitetalk_two_person_routing_receipt_v1"]
    spatial_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_still_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    speaker_slot: Literal[1, 2]
    listener_slot: Literal[1, 2]
    mode: Literal["two_person_parallel"]
    multi_audio_type: Literal["para"]
    mask_sha256: InfiniteTalkMaskHashes


class InfiniteTalkRoutingReceiptV2(InfiniteTalkRoutingReceiptV1):
    """Proof of full-window v2 conditioning and delivered-output provenance."""

    schema_version: Literal["infinitetalk_two_person_routing_receipt_v2"]
    conditioning_lead_in_frames: Literal[3]
    effective_video_frames: int = Field(ge=81)
    effective_duration_sec: float = Field(gt=0, le=30.0)
    delivered_video_frames: int = Field(gt=0)
    delivered_duration_sec: float = Field(gt=0, le=30.0)
    speaker_conditioning: InfiniteTalkSpeakerConditioning
    listener_conditioning: InfiniteTalkListenerConditioning
    postprocess: InfiniteTalkPostprocessPolicy
    final_audio: InfiniteTalkFinalAudio


InfiniteTalkRoutingReceipt = Annotated[
    InfiniteTalkRoutingReceiptV1 | InfiniteTalkRoutingReceiptV2,
    Field(discriminator="schema_version"),
]


class RunRequest(BaseModel):
    """Request payload for a worker run."""

    job_id: str
    asset_group: str
    comfy_payload: dict[str, Any] = Field(description="Raw ComfyUI workflow payload.")
    comfy_input_files: list["ComfyInputFile"] = Field(default_factory=list)
    infinitetalk_routing: InfiniteTalkTwoPersonRouting | None = None
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
    # Base64 bytes. Required as the sole source whenever expected_sha256 is set.
    source_data: str | None = None
    expected_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Exact bytes that must be staged before prompt submission.",
    )
    content_type: str | None = None
    type: str = "input"
    subfolder: str = ""


class StagedInputReceipt(BaseModel):
    """Locator-free observation of bytes actually staged for one graph input."""

    node_id: str
    input_name: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


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
    staged_input_receipts: list[StagedInputReceipt] = Field(default_factory=list)
    infinitetalk_routing_receipt: InfiniteTalkRoutingReceipt | None = None
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
