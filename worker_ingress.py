"""Earliest ASGI ingress gate for the public GPU-worker HTTP surface."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from gpu_worker.worker_auth import WorkerAPIAuthError, verify_worker_api_authorization


_RUN_BODY_LIMIT_BYTES = 32 * 1024 * 1024
_STITCH_BODY_LIMIT_BYTES = 2 * 1024 * 1024
_TTS_BODY_LIMIT_BYTES = 64 * 1024
_ASSET_BODY_LIMIT_BYTES = 64 * 1024
_DEFAULT_BODY_LIMIT_BYTES = 64 * 1024
_BODY_READ_TIMEOUT_SEC = 60.0


def is_protected_worker_path(path: str) -> bool:
    """Return whether a path belongs to a generation, job, or file surface."""

    normalized = str(path or "")
    return (
        normalized in {"/assets/ensure", "/run", "/stitch", "/tts", "/jobs"}
        or normalized.startswith("/jobs/")
        or normalized.startswith("/files/")
    )


def request_body_limit_bytes(path: str) -> int:
    """Return the pre-parser byte budget for one request path."""

    if path in {"/run", "/jobs"}:
        return _RUN_BODY_LIMIT_BYTES
    if path == "/stitch":
        return _STITCH_BODY_LIMIT_BYTES
    if path == "/tts":
        return _TTS_BODY_LIMIT_BYTES
    if path == "/assets/ensure":
        return _ASSET_BODY_LIMIT_BYTES
    return _DEFAULT_BODY_LIMIT_BYTES


def _header_values(scope: dict[str, Any], name: bytes) -> list[bytes]:
    return [
        bytes(value)
        for key, value in scope.get("headers") or []
        if bytes(key).lower() == name
    ]


async def _send_json_error(send, status_code: int, detail: str) -> None:
    body = json.dumps({"detail": detail}, separators=(",", ":")).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


def _declared_content_length(scope: dict[str, Any]) -> tuple[int | None, str | None]:
    lengths = _header_values(scope, b"content-length")
    transfer_encodings = _header_values(scope, b"transfer-encoding")
    if lengths and transfer_encodings:
        return None, "Conflicting request body framing"
    if len(lengths) > 1:
        return None, "Ambiguous request content length"
    if not lengths:
        return None, None
    try:
        rendered = lengths[0].decode("ascii", errors="strict")
        if not rendered or not rendered.isdigit():
            raise ValueError
        length = int(rendered)
    except (UnicodeError, ValueError):
        return None, "Invalid request content length"
    return length, None


async def _read_bounded_body(receive, limit: int) -> tuple[bytes | None, str | None]:
    """Read at most limit bytes, enforcing one absolute receive deadline."""

    chunks: list[bytes] = []
    total = 0
    deadline = time.monotonic() + _BODY_READ_TIMEOUT_SEC
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, "Request body receive timeout"
        try:
            message = await asyncio.wait_for(receive(), timeout=remaining)
        except TimeoutError:
            return None, "Request body receive timeout"
        message_type = message.get("type")
        if message_type == "http.disconnect":
            return None, "Request body disconnected"
        if message_type != "http.request":
            continue
        chunk = bytes(message.get("body") or b"")
        total += len(chunk)
        if total > limit:
            return None, "Request body exceeds byte limit"
        if chunk:
            chunks.append(chunk)
        if not message.get("more_body", False):
            return b"".join(chunks), None


class WorkerIngressMiddleware:
    """Authenticate and size-bound requests before FastAPI parses their bodies."""

    def __init__(
        self,
        app,
        *,
        settings_provider: Callable[[], Any],
    ) -> None:
        self.app = app
        self.settings_provider = settings_provider

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path") or "")
        if is_protected_worker_path(path):
            authorization_values = _header_values(scope, b"authorization")
            authorization = None
            if len(authorization_values) == 1:
                authorization = authorization_values[0].decode(
                    "latin-1", errors="strict"
                )
            settings = self.settings_provider()
            try:
                verify_worker_api_authorization(
                    expected_token=settings.worker_api_token,
                    auth_mode=settings.worker_api_auth_mode,
                    authorization=authorization,
                )
            except WorkerAPIAuthError as exc:
                await _send_json_error(send, exc.status_code, exc.detail)
                return

        limit = request_body_limit_bytes(path)
        declared_length, framing_error = _declared_content_length(scope)
        if framing_error:
            await _send_json_error(send, 400, framing_error)
            return
        if declared_length is not None and declared_length > limit:
            await _send_json_error(send, 413, "Request body exceeds byte limit")
            return

        body, body_error = await _read_bounded_body(receive, limit)
        if body_error:
            status_code = 413 if body_error.endswith("byte limit") else 408
            if body_error == "Request body disconnected":
                status_code = 400
            await _send_json_error(send, status_code, body_error)
            return
        assert body is not None
        if declared_length is not None and len(body) != declared_length:
            await _send_json_error(send, 400, "Request content length mismatch")
            return

        delivered = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }

        await self.app(scope, replay_receive, send)
