from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gpu_worker import app as app_mod
from gpu_worker import worker_ingress
from gpu_worker.config import Settings


def _settings(token: str | None, mode: str = "required") -> SimpleNamespace:
    return SimpleNamespace(worker_api_token=token, worker_api_auth_mode=mode)


def _invoke(
    middleware,
    *,
    path: str,
    headers: list[tuple[bytes, bytes]],
    messages: list[dict],
) -> tuple[list[dict], int]:
    sent: list[dict] = []
    receive_calls = 0

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers,
        "client": ("203.0.113.20", 40000),
        "server": ("worker.example", 443),
    }
    asyncio.run(middleware(scope, receive, send))
    return sent, receive_calls


def _status(events: list[dict]) -> int:
    return next(event["status"] for event in events if event["type"] == "http.response.start")


@pytest.mark.parametrize("authorization", [None, b"Bearer wrong-token"])
def test_unauthorized_huge_valid_json_is_rejected_without_reading_body(
    authorization,
) -> None:
    inner_called = False

    async def inner(_scope, _receive, _send):
        nonlocal inner_called
        inner_called = True

    middleware = worker_ingress.WorkerIngressMiddleware(
        inner,
        settings_provider=lambda: _settings("fixture-worker-token"),
    )
    body = b'{"source_data":"' + (b"A" * (2 * 1024 * 1024)) + b'"}'
    headers = [(b"content-length", str(len(body)).encode("ascii"))]
    if authorization is not None:
        headers.append((b"authorization", authorization))

    events, receive_calls = _invoke(
        middleware,
        path="/jobs",
        headers=headers,
        messages=[{"type": "http.request", "body": body, "more_body": False}],
    )

    assert _status(events) == 401
    assert receive_calls == 0
    assert inner_called is False


def test_missing_required_auth_configuration_reads_no_body() -> None:
    async def inner(_scope, _receive, _send):
        raise AssertionError("inner app must not run")

    middleware = worker_ingress.WorkerIngressMiddleware(
        inner,
        settings_provider=lambda: _settings(None),
    )
    events, receive_calls = _invoke(
        middleware,
        path="/run",
        headers=[(b"content-length", b"999999999")],
        messages=[{"type": "http.request", "body": b"ignored", "more_body": False}],
    )

    assert _status(events) == 503
    assert receive_calls == 0


@pytest.mark.parametrize(
    "path",
    [
        "/assets/ensure",
        "/tts",
        "/run",
        "/stitch",
        "/jobs",
        "/jobs/job-1",
        "/jobs/job-1/progress",
        "/files/output/cut.mp4",
    ],
)
def test_every_protected_route_space_is_gated_before_body(path: str) -> None:
    async def inner(_scope, _receive, _send):
        raise AssertionError("inner app must not run")

    middleware = worker_ingress.WorkerIngressMiddleware(
        inner,
        settings_provider=lambda: _settings("fixture-worker-token"),
    )
    events, receive_calls = _invoke(
        middleware,
        path=path,
        headers=[],
        messages=[{"type": "http.request", "body": b"{}", "more_body": False}],
    )

    assert _status(events) == 401
    assert receive_calls == 0


def test_authenticated_declared_over_cap_is_413_without_body_read() -> None:
    async def inner(_scope, _receive, _send):
        raise AssertionError("inner app must not run")

    middleware = worker_ingress.WorkerIngressMiddleware(
        inner,
        settings_provider=lambda: _settings("fixture-worker-token"),
    )
    limit = worker_ingress.request_body_limit_bytes("/tts")
    events, receive_calls = _invoke(
        middleware,
        path="/tts",
        headers=[
            (b"authorization", b"Bearer fixture-worker-token"),
            (b"content-length", str(limit + 1).encode("ascii")),
        ],
        messages=[{"type": "http.request", "body": b"ignored", "more_body": False}],
    )

    assert _status(events) == 413
    assert receive_calls == 0


def test_authenticated_chunked_body_is_capped_before_inner_parser(monkeypatch) -> None:
    inner_called = False

    async def inner(_scope, _receive, _send):
        nonlocal inner_called
        inner_called = True

    monkeypatch.setattr(worker_ingress, "request_body_limit_bytes", lambda _path: 8)
    middleware = worker_ingress.WorkerIngressMiddleware(
        inner,
        settings_provider=lambda: _settings("fixture-worker-token"),
    )
    events, receive_calls = _invoke(
        middleware,
        path="/jobs",
        headers=[
            (b"authorization", b"Bearer fixture-worker-token"),
            (b"transfer-encoding", b"chunked"),
        ],
        messages=[
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"56789", "more_body": False},
        ],
    )

    assert _status(events) == 413
    assert receive_calls == 2
    assert inner_called is False


def test_authenticated_bounded_body_is_replayed_once_to_inner_app() -> None:
    observed = []

    async def inner(_scope, receive, send):
        observed.append(await receive())
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    middleware = worker_ingress.WorkerIngressMiddleware(
        inner,
        settings_provider=lambda: _settings("fixture-worker-token"),
    )
    body = b'{"text":"hello"}'
    events, receive_calls = _invoke(
        middleware,
        path="/tts",
        headers=[
            (b"authorization", b"Bearer fixture-worker-token"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        messages=[{"type": "http.request", "body": body, "more_body": False}],
    )

    assert _status(events) == 204
    assert receive_calls == 1
    assert observed == [
        {"type": "http.request", "body": body, "more_body": False}
    ]


def test_actual_fastapi_stack_authenticates_before_malformed_json(monkeypatch) -> None:
    monkeypatch.setattr(
        app_mod,
        "get_settings",
        lambda: _settings("fixture-worker-token"),
    )
    response = pytest.importorskip("fastapi.testclient").TestClient(app_mod.app).post(
        "/jobs",
        content=b"{",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid worker API token"}


def test_gpu_worker_api_token_alias_is_supported(monkeypatch) -> None:
    monkeypatch.delenv("WORKER_API_TOKEN", raising=False)
    monkeypatch.setenv("GPU_WORKER_API_TOKEN", "fixture-alias-token")
    settings = Settings(_env_file=None)
    assert settings.worker_api_token == "fixture-alias-token"


def test_health_is_not_ready_when_required_auth_is_unconfigured(monkeypatch) -> None:
    base = app_mod.get_settings()
    settings = base.model_copy(
        update={
            "worker_api_token": None,
            "worker_api_auth_mode": "required",
            "worker_name": "fixture-worker",
        }
    )
    monkeypatch.setattr(app_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(app_mod, "is_comfy_healthy", lambda: True)
    monkeypatch.setattr(app_mod, "_advertised_capabilities", lambda: ([], None, None))
    monkeypatch.setattr(app_mod, "active_download_status", lambda: None)

    response = pytest.importorskip("fastapi.testclient").TestClient(app_mod.app).get(
        "/health"
    )

    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert response.json()["worker_ok"] is False
    assert "token" not in response.text.lower()


def test_job_registry_evicts_terminal_records_and_bounds_active_records(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app_mod, "_JOB_REGISTRY_MAX_RECORDS", 2)
    monkeypatch.setattr(app_mod, "_JOB_REGISTRY_TTL_SEC", 3600.0)
    with app_mod._JOB_LOCK:
        app_mod._JOBS.clear()
        app_mod._JOBS["old"] = app_mod._JobRecord(
            job_id="old",
            request=None,
            client_job_id="client-old",
            asset_group="flux_stills_v1",
            status="completed",
            finished_monotonic=1.0,
        )
        app_mod._JOBS["running"] = app_mod._JobRecord(
            job_id="running",
            request=None,
            client_job_id="client-running",
            asset_group="flux_stills_v1",
            status="running",
        )
        assert app_mod._prune_job_registry_locked(reserve=1) is True
        assert set(app_mod._JOBS) == {"running"}
        app_mod._JOBS["running-2"] = app_mod._JobRecord(
            job_id="running-2",
            request=None,
            client_job_id="client-running-2",
            asset_group="flux_stills_v1",
            status="running",
        )
        assert app_mod._prune_job_registry_locked(reserve=1) is False
        app_mod._JOBS.clear()


def test_async_job_registry_releases_request_before_execution(monkeypatch) -> None:
    request = SimpleNamespace(asset_group="flux_stills_v1")

    def execute(observed_request, *, progress):
        assert observed_request is request
        assert progress is None
        with app_mod._JOB_LOCK:
            assert app_mod._JOBS["worker-job"].request is None
        return SimpleNamespace(ok=True, error=None, timings=None)

    monkeypatch.setattr(app_mod, "_execute_run", execute)
    with app_mod._JOB_LOCK:
        app_mod._JOBS.clear()
        app_mod._JOBS["worker-job"] = app_mod._JobRecord(
            job_id="worker-job",
            request=request,
            client_job_id="client-job",
            asset_group="flux_stills_v1",
        )

    app_mod._run_job_async("worker-job")

    with app_mod._JOB_LOCK:
        record = app_mod._JOBS.pop("worker-job")
    assert record.request is None
    assert record.status == "completed"
