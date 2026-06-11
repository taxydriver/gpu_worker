"""Minimal ComfyUI process lifecycle helpers."""

from __future__ import annotations

import logging
import subprocess
import time

import requests

from gpu_worker.config import get_settings
from gpu_worker.utils import run_shell_command


LOGGER = logging.getLogger(__name__)

# Candidate Python paths that have torch installed (ComfyUI venvs).
# Tried in order; first one that responds is used for diagnostics/smoke tests.
_COMFY_PYTHON_CANDIDATES = [
    "/workspace/ComfyUI/.venv/bin/python",
    "/workspace/runpod-slim/ComfyUI/.venv-cu128/bin/python",
    "/venv/main/bin/python",
]

_GPU_DIAG_SCRIPT = """
import sys
try:
    import torch
    print("torch=" + torch.__version__)
    print("cuda=" + str(torch.version.cuda))
    print("cudnn=" + str(torch.backends.cudnn.version()))
    if torch.cuda.is_available():
        print("device=" + torch.cuda.get_device_name(0))
        print("capability=" + str(torch.cuda.get_device_capability(0)))
        print("cudnn_sdp=" + str(torch.backends.cuda.cudnn_sdp_enabled()))
        print("flash_sdp=" + str(torch.backends.cuda.flash_sdp_enabled()))
        print("math_sdp=" + str(torch.backends.cuda.math_sdp_enabled()))
    else:
        print("cuda_available=False")
    try:
        import xformers
        print("xformers=" + xformers.__version__)
    except ImportError:
        print("xformers=not_installed")
except ImportError:
    print("torch=not_installed")
"""

_GPU_SMOKE_SCRIPT = """
import sys
try:
    import torch
    import torch.nn.functional as F
    t = torch.zeros(4, device="cuda")
    x = torch.randn(1, 1, 4, 4, device="cuda")
    w = torch.randn(1, 1, 2, 2, device="cuda")
    F.conv2d(x, w)
    q = torch.randn(1, 1, 4, 8, device="cuda")
    k = torch.randn(1, 1, 4, 8, device="cuda")
    v = torch.randn(1, 1, 4, 8, device="cuda")
    F.scaled_dot_product_attention(q, k, v)
    print("SMOKE_TEST_OK")
    sys.exit(0)
except Exception as e:
    print("SMOKE_TEST_FAIL: " + str(e))
    sys.exit(1)
"""


def _find_comfy_python() -> str | None:
    """Return the first ComfyUI Python interpreter found on disk."""
    import os
    for candidate in _COMFY_PYTHON_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


def run_gpu_diagnostics() -> None:
    """Run torch/CUDA diagnostics via ComfyUI's Python and log the results."""
    python = _find_comfy_python()
    if python is None:
        LOGGER.info("[gpu-diag] no ComfyUI Python found — skipping diagnostics")
        return
    try:
        result = subprocess.run(
            [python, "-c", _GPU_DIAG_SCRIPT],
            capture_output=True, text=True, timeout=20,
        )
        for line in result.stdout.strip().splitlines():
            LOGGER.info("[gpu-diag] %s", line)
        if result.returncode != 0 and result.stderr:
            LOGGER.warning("[gpu-diag] stderr: %s", result.stderr[:500])
    except subprocess.TimeoutExpired:
        LOGGER.warning("[gpu-diag] timed out")
    except Exception as exc:
        LOGGER.warning("[gpu-diag] failed: %s", exc)


def run_gpu_smoke_test() -> bool:
    """Run a minimal CUDA/SDPA smoke test. Returns True on pass, False on fail."""
    python = _find_comfy_python()
    if python is None:
        LOGGER.info("[gpu-smoke] no ComfyUI Python found — skipping")
        return True
    try:
        result = subprocess.run(
            [python, "-c", _GPU_SMOKE_SCRIPT],
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout.strip()
        if result.returncode == 0 and "SMOKE_TEST_OK" in output:
            LOGGER.info("[gpu-smoke] PASS")
            return True
        LOGGER.error("[gpu-smoke] FAIL — %s %s", output, result.stderr[:200])
        return False
    except subprocess.TimeoutExpired:
        LOGGER.warning("[gpu-smoke] timed out")
        return False
    except Exception as exc:
        LOGGER.warning("[gpu-smoke] error: %s", exc)
        return False


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
