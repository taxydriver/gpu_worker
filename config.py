"""Configuration for the GPU worker."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import socket

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    comfy_base_url: str = Field(default="http://127.0.0.1:8188", alias="COMFY_BASE_URL")
    comfy_health_timeout_sec: int = Field(default=60, alias="COMFY_HEALTH_TIMEOUT_SEC")
    model_download_timeout_sec: int = Field(default=1800, alias="MODEL_DOWNLOAD_TIMEOUT_SEC")
    worker_host: str = Field(default="0.0.0.0", alias="WORKER_HOST")
    worker_port: int = Field(default=9000, alias="WORKER_PORT")
    comfy_start_cmd: str | None = Field(default=None, alias="COMFY_START_CMD")
    comfy_stop_cmd: str | None = Field(default=None, alias="COMFY_STOP_CMD")
    download_chunk_size: int = Field(default=8 * 1024 * 1024, alias="DOWNLOAD_CHUNK_SIZE")
    comfy_output_dir: str = Field(default="/workspace/ComfyUI/output", alias="COMFY_OUTPUT_DIR")
    comfy_temp_dir: str | None = Field(default=None, alias="COMFY_TEMP_DIR")
    comfy_input_dir: str = Field(default="/workspace/ComfyUI/input", alias="COMFY_INPUT_DIR")
    log_prompts_only: bool = Field(default=False, alias="LOG_PROMPTS_ONLY")
    render_broker_base_url: str | None = Field(default=None, alias="RENDER_BROKER_BASE_URL")
    render_broker_worker_token: str | None = Field(default=None, alias="RENDER_BROKER_WORKER_TOKEN")
    render_broker_worker_id: str = Field(
        default_factory=lambda: f"{socket.gethostname()}-gpu-worker",
        alias="RENDER_BROKER_WORKER_ID",
    )
    render_broker_worker_name: str | None = Field(default=None, alias="RENDER_BROKER_WORKER_NAME")
    render_broker_worker_public_url: str | None = Field(default=None, alias="RENDER_BROKER_WORKER_PUBLIC_URL")
    render_broker_heartbeat_sec: int = Field(default=30, alias="RENDER_BROKER_HEARTBEAT_SEC")
    render_broker_max_concurrency: int = Field(default=1, alias="RENDER_BROKER_MAX_CONCURRENCY")

    # Phase 1 worker-pool env vars (new naming convention)
    filmforge_backend_url: str | None = Field(default=None, alias="FILMFORGE_BACKEND_URL")
    worker_name: str | None = Field(default=None, alias="WORKER_NAME")
    worker_public_url: str | None = Field(default=None, alias="WORKER_PUBLIC_URL")
    worker_provider: str = Field(default="dedicated_worker", alias="WORKER_PROVIDER")
    worker_gpu_name: str | None = Field(default=None, alias="WORKER_GPU_NAME")
    worker_vram_gb: float | None = Field(default=None, alias="WORKER_VRAM_GB")
    worker_capabilities: str | None = Field(default=None, alias="WORKER_CAPABILITIES")
    worker_max_concurrent_jobs: int = Field(default=1, alias="WORKER_MAX_CONCURRENT_JOBS")
    worker_registration_token: str | None = Field(default=None, alias="WORKER_REGISTRATION_TOKEN")
    worker_api_token: str | None = Field(default=None, alias="WORKER_API_TOKEN")
    worker_heartbeat_seconds: int = Field(default=60, alias="WORKER_HEARTBEAT_SECONDS")

    def served_file_roots(self) -> dict[str, Path]:
        """Return the named roots the worker is allowed to serve files from."""

        roots = {
            "output": Path(self.comfy_output_dir).expanduser().resolve(strict=False),
        }
        if self.comfy_temp_dir:
            roots["temp"] = Path(self.comfy_temp_dir).expanduser().resolve(strict=False)
        return roots

    def broker_enabled(self) -> bool:
        return bool((self.render_broker_base_url or "").strip())

    def resolved_backend_url(self) -> str | None:
        """Resolve backend URL: Phase 1 (FILMFORGE_BACKEND_URL) or legacy (RENDER_BROKER_BASE_URL)."""
        return self.filmforge_backend_url or self.render_broker_base_url

    def resolved_worker_id(self) -> str:
        """Resolve worker ID: use RENDER_BROKER_WORKER_ID as unique identifier."""
        return self.render_broker_worker_id

    def resolved_worker_name(self) -> str:
        """Resolve worker name: Phase 1 (WORKER_NAME) or legacy (RENDER_BROKER_WORKER_NAME) or worker_id."""
        return self.worker_name or self.render_broker_worker_name or self.render_broker_worker_id

    def resolved_worker_public_url(self) -> str | None:
        """Resolve public URL: Phase 1 (WORKER_PUBLIC_URL) or legacy (RENDER_BROKER_WORKER_PUBLIC_URL)."""
        return self.worker_public_url or self.render_broker_worker_public_url

    def resolved_capabilities(self) -> list[str]:
        """Parse WORKER_CAPABILITIES comma-separated string into list."""
        if self.worker_capabilities:
            return [c.strip() for c in self.worker_capabilities.split(",") if c.strip()]
        return []

    def resolved_registration_token(self) -> str | None:
        """Resolve registration token: Phase 1 (WORKER_REGISTRATION_TOKEN) or legacy (RENDER_BROKER_WORKER_TOKEN)."""
        return self.worker_registration_token or self.render_broker_worker_token

    def resolved_heartbeat_seconds(self) -> int:
        """Resolve heartbeat interval: Phase 1 (WORKER_HEARTBEAT_SECONDS) or legacy (RENDER_BROKER_HEARTBEAT_SEC)."""
        if self.worker_heartbeat_seconds and self.worker_heartbeat_seconds > 0:
            return self.worker_heartbeat_seconds
        if self.render_broker_heartbeat_sec and self.render_broker_heartbeat_sec > 0:
            return self.render_broker_heartbeat_sec
        return 60

    def resolved_max_concurrent_jobs(self) -> int:
        """Resolve max concurrent: Phase 1 (WORKER_MAX_CONCURRENT_JOBS) or legacy (RENDER_BROKER_MAX_CONCURRENCY)."""
        if self.worker_max_concurrent_jobs and self.worker_max_concurrent_jobs > 0:
            return self.worker_max_concurrent_jobs
        if self.render_broker_max_concurrency and self.render_broker_max_concurrency > 0:
            return self.render_broker_max_concurrency
        return 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached worker settings."""

    return Settings()
