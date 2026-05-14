"""Minimal ComfyUI process lifecycle helpers."""

from __future__ import annotations

import logging
import time

import requests

from gpu_worker.config import get_settings
from gpu_worker.utils import run_shell_command


LOGGER = logging.getLogger(__name__)


def is_comfy_healthy() -> bool:
    """Return True when the local ComfyUI API responds successfully."""

    base_url = get_settings().comfy_base_url.rstrip("/")
    for path in ("/system_stats", "/queue"):
        try:
            response = requests.get(
                f"{base_url}{path}",
                timeout=5,
            )
            if response.status_code == 200:
                return True
        except requests.RequestException:
            continue
    return False


def wait_for_comfy_ready(timeout_sec: int) -> bool:
    """Poll ComfyUI until it is reachable or the timeout expires."""

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if is_comfy_healthy():
            return True
        time.sleep(2.0)
    return False


def restart_comfy() -> float:
    """Restart ComfyUI using simple shell commands.

    This is intentionally MVP-grade process control. It can later be replaced
    with systemd, supervisor, docker-aware controls, or another orchestrator.
    """

    settings = get_settings()
    if not settings.comfy_start_cmd:
        raise RuntimeError("COMFY_START_CMD is required to restart ComfyUI")

    started_at = time.monotonic()
    if settings.comfy_stop_cmd:
        LOGGER.info("Stopping ComfyUI")
        run_shell_command(settings.comfy_stop_cmd)
        time.sleep(1.0)
    else:
        LOGGER.info("COMFY_STOP_CMD not set. Skipping explicit stop.")

    LOGGER.info("Starting ComfyUI")
    run_shell_command(settings.comfy_start_cmd)

    if not wait_for_comfy_ready(settings.comfy_health_timeout_sec):
        raise RuntimeError(
            "ComfyUI did not become healthy after restart within "
            f"{settings.comfy_health_timeout_sec} seconds"
        )

    return time.monotonic() - started_at
