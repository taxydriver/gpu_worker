"""Small, dependency-free worker API authentication contract."""

from __future__ import annotations

import secrets


class WorkerAPIAuthError(RuntimeError):
    """Authentication failure with an HTTP-safe status and detail."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def worker_api_auth_is_ready(*, expected_token: str | None, auth_mode: str) -> bool:
    """Return whether protected routes have an intentional auth configuration."""

    expected = str(expected_token or "").strip()
    mode = str(auth_mode or "required").strip().lower()
    return bool(expected) or mode in {"development", "test"}


def verify_worker_api_authorization(
    *,
    expected_token: str | None,
    auth_mode: str,
    authorization: str | None,
) -> None:
    """Require a bearer token unless an explicit development/test mode is set."""

    expected = str(expected_token or "").strip()
    mode = str(auth_mode or "required").strip().lower()
    if not expected:
        if worker_api_auth_is_ready(
            expected_token=expected,
            auth_mode=mode,
        ):
            return
        raise WorkerAPIAuthError(
            503,
            "Worker API authentication is not configured",
        )

    provided = ""
    if authorization:
        scheme, separator, token = authorization.strip().partition(" ")
        if separator and scheme.lower() == "bearer":
            provided = token.strip()
    if not secrets.compare_digest(provided, expected):
        raise WorkerAPIAuthError(401, "Invalid worker API token")
