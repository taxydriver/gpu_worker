from __future__ import annotations

from types import SimpleNamespace

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
app_mod = pytest.importorskip("gpu_worker.app")


def _settings(token: str | None, mode: str = "required"):
    return SimpleNamespace(worker_api_token=token, worker_api_auth_mode=mode)


def test_job_status_rejects_missing_worker_auth_configuration(monkeypatch) -> None:
    monkeypatch.setattr(app_mod, "get_settings", lambda: _settings(None))
    response = fastapi_testclient.TestClient(app_mod.app).get("/jobs/not-present")
    assert response.status_code == 503
    assert response.json() == {"detail": "Worker API authentication is not configured"}


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-token"},
        {"Authorization": "Basic fixture-worker-token"},
    ],
)
def test_job_status_rejects_missing_or_wrong_worker_token(monkeypatch, headers) -> None:
    monkeypatch.setattr(
        app_mod,
        "get_settings",
        lambda: _settings("fixture-worker-token"),
    )
    response = fastapi_testclient.TestClient(app_mod.app).get(
        "/jobs/not-present",
        headers=headers,
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid worker API token"}


def test_job_status_accepts_right_worker_token(monkeypatch) -> None:
    monkeypatch.setattr(
        app_mod,
        "get_settings",
        lambda: _settings("fixture-worker-token"),
    )
    response = fastapi_testclient.TestClient(app_mod.app).get(
        "/jobs/not-present",
        headers={"Authorization": "Bearer fixture-worker-token"},
    )
    # The request crossed the auth boundary and reached the job lookup.
    assert response.status_code == 404
    assert response.json() == {"detail": "Unknown job_id: not-present"}
