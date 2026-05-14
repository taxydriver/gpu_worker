"""
Backend registration for GPU worker.

This module handles registering the GPU worker with the Filmforge backend
when the worker starts, and sending periodic heartbeats to keep it marked as active.

Environment variables:
    FILMFORGE_BACKEND_URL: Backend URL (e.g., http://localhost:8000 or https://filmforge-backend.fly.dev)
    WORKER_NAME: Unique worker name (default: hostname)
    WORKER_PUBLIC_URL: Public URL of this GPU worker (e.g., http://localhost:9000 or https://xyz.runpod.io)
    REGISTER_WITH_BACKEND: Enable registration (default: true if FILMFORGE_BACKEND_URL is set)
    HEARTBEAT_INTERVAL_SEC: Heartbeat interval in seconds (default: 300)
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from typing import Optional

import requests

LOGGER = logging.getLogger(__name__)


def get_backend_url() -> Optional[str]:
    """Get the backend URL from environment."""
    url = (os.getenv("FILMFORGE_BACKEND_URL") or "").strip()
    return url.rstrip("/") if url else None


def get_worker_name() -> str:
    """Get the worker name (defaults to hostname)."""
    return (os.getenv("WORKER_NAME") or "").strip() or socket.gethostname()


def get_worker_public_url() -> Optional[str]:
    """Get the public URL for this worker.

    Tries multiple sources:
    1. WORKER_PUBLIC_URL env var
    2. RENDER_BROKER_WORKER_PUBLIC_URL env var (for render broker compatibility)
    3. http://localhost:9000 (fallback for local dev)
    """
    url = (os.getenv("WORKER_PUBLIC_URL") or "").strip()
    if url:
        return url.rstrip("/")

    # Fallback to render broker setting
    url = (os.getenv("RENDER_BROKER_WORKER_PUBLIC_URL") or "").strip()
    if url:
        return url.rstrip("/")

    # Local fallback
    worker_port = os.getenv("WORKER_PORT", "9000")
    return f"http://localhost:{worker_port}"


def register_with_backend(
    backend_url: str,
    worker_name: str,
    worker_url: str,
    is_active: bool = True,
    timeout_sec: int = 10,
) -> bool:
    """Register this GPU worker with the backend.

    Args:
        backend_url: Backend base URL
        worker_name: Unique worker identifier
        worker_url: Public URL of this GPU worker
        is_active: Whether to mark as active
        timeout_sec: HTTP request timeout

    Returns:
        True if successful, False otherwise
    """
    register_url = f"{backend_url}/api/gpu-workers/register"
    payload = {
        "name": worker_name,
        "base_url": worker_url,
        "is_active": is_active,
    }

    try:
        response = requests.post(register_url, json=payload, timeout=timeout_sec)
        if response.status_code == 200:
            data = response.json()
            LOGGER.info(
                "[backend] Registered GPU worker: name=%s url=%s id=%s",
                worker_name,
                worker_url,
                data.get("id"),
            )
            return True
        else:
            LOGGER.warning(
                "[backend] Registration failed: HTTP %s response=%s",
                response.status_code,
                response.text[:200] if response.text else "(empty)",
            )
            return False
    except requests.RequestException as exc:
        LOGGER.warning("[backend] Registration error: %s", exc)
        return False


def heartbeat_with_backend(
    backend_url: str,
    worker_name: str,
    timeout_sec: int = 10,
) -> bool:
    """Send a heartbeat to keep the worker marked as active.

    Args:
        backend_url: Backend base URL
        worker_name: Worker identifier
        timeout_sec: HTTP request timeout

    Returns:
        True if successful, False otherwise
    """
    heartbeat_url = f"{backend_url}/api/gpu-workers/{worker_name}/heartbeat"

    try:
        response = requests.post(heartbeat_url, timeout=timeout_sec)
        if response.status_code == 200:
            LOGGER.debug("[backend] Heartbeat sent: %s", worker_name)
            return True
        else:
            LOGGER.warning(
                "[backend] Heartbeat failed: HTTP %s",
                response.status_code,
            )
            return False
    except requests.RequestException as exc:
        LOGGER.debug("[backend] Heartbeat error: %s", exc)
        return False


def _registration_loop(
    backend_url: str,
    worker_name: str,
    worker_url: str,
    heartbeat_interval_sec: int = 300,
) -> None:
    """Background thread: register on startup, then send periodic heartbeats.

    Args:
        backend_url: Backend URL
        worker_name: Worker name
        worker_url: Worker public URL
        heartbeat_interval_sec: Heartbeat interval in seconds
    """
    LOGGER.info(
        "[backend] Registration loop starting: name=%s url=%s backend=%s",
        worker_name,
        worker_url,
        backend_url,
    )

    # Initial registration with retries
    max_retries = 3
    for attempt in range(max_retries):
        if register_with_backend(backend_url, worker_name, worker_url):
            break
        if attempt < max_retries - 1:
            wait_sec = 2 ** attempt
            LOGGER.info("[backend] Retrying registration in %ds", wait_sec)
            time.sleep(wait_sec)
    else:
        LOGGER.error("[backend] Failed to register after %d attempts", max_retries)

    # Periodic heartbeats
    while True:
        time.sleep(heartbeat_interval_sec)
        heartbeat_with_backend(backend_url, worker_name)


def start_registration_thread() -> Optional[threading.Thread]:
    """Start the backend registration thread if configured.

    Returns:
        Thread object if registration is enabled, None otherwise
    """
    backend_url = get_backend_url()
    if not backend_url:
        LOGGER.debug("[backend] Backend URL not configured — registration disabled")
        return None

    # Check if explicitly disabled
    if os.getenv("REGISTER_WITH_BACKEND", "").lower() in {"0", "false", "no", "off"}:
        LOGGER.info("[backend] Registration explicitly disabled")
        return None

    worker_name = get_worker_name()
    worker_url = get_worker_public_url()

    if not worker_url:
        LOGGER.error("[backend] Cannot determine worker public URL for registration")
        return None

    heartbeat_interval = int(os.getenv("HEARTBEAT_INTERVAL_SEC", "300"))

    thread = threading.Thread(
        target=_registration_loop,
        args=(backend_url, worker_name, worker_url, heartbeat_interval),
        name="backend-registration",
        daemon=True,
    )
    thread.start()
    return thread
