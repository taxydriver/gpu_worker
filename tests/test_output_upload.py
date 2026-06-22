"""Worker-side media-offload (ADR-0002): the worker PUTs its primary output to a
backend-minted signed upload URL and stamps ``storage_url`` so the backend records
the URL instead of downloading + re-uploading. On any failure ``storage_url`` stays
unset and the backend falls back to its ``/files`` download.

Schema tests run anywhere; the ``_maybe_upload_primary_output`` tests importorskip
``gpu_worker.app`` (needs fastapi) so they run in the worker environment / CI.
"""
from __future__ import annotations

import pytest

from gpu_worker.schemas import OutputFile, OutputUploadTarget, RunRequest


def test_run_request_accepts_output_upload():
    req = RunRequest(
        job_id="j", asset_group="flux_stills_v1", comfy_payload={},
        output_upload={"signed_put_url": "https://put/x", "public_url": "https://pub/x.png",
                       "content_type": "image/png"},
    )
    assert isinstance(req.output_upload, OutputUploadTarget)
    assert req.output_upload.public_url == "https://pub/x.png"


def test_run_request_output_upload_defaults_none():
    assert RunRequest(job_id="j", asset_group="a", comfy_payload={}).output_upload is None


def test_output_file_storage_url_defaults_none():
    of = OutputFile(path="/w/x.png", filename="x.png", download_url="/files/x")
    assert of.storage_url is None


def _target():
    return OutputUploadTarget(
        signed_put_url="https://put/x", public_url="https://pub/x.png", content_type="image/png"
    )


def test_maybe_upload_stamps_storage_url_on_success(monkeypatch):
    app_mod = pytest.importorskip("gpu_worker.app")
    calls = {}
    monkeypatch.setattr(
        app_mod, "upload_via_signed_put",
        lambda url, path, ct: calls.update(url=url, path=path, ct=ct),
    )
    files = [OutputFile(path="/w/x.png", filename="x.png", download_url="/files/x")]
    out = app_mod._maybe_upload_primary_output(files, _target(), job_id="j")
    assert out[0].storage_url == "https://pub/x.png"
    assert calls == {"url": "https://put/x", "path": "/w/x.png", "ct": "image/png"}


def test_maybe_upload_noop_when_no_target(monkeypatch):
    app_mod = pytest.importorskip("gpu_worker.app")
    monkeypatch.setattr(app_mod, "upload_via_signed_put",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not upload")))
    files = [OutputFile(path="/w/x.png", filename="x.png", download_url="/files/x")]
    out = app_mod._maybe_upload_primary_output(files, None, job_id="j")
    assert out[0].storage_url is None


def test_maybe_upload_falls_back_on_failure(monkeypatch):
    app_mod = pytest.importorskip("gpu_worker.app")

    def _boom(url, path, ct):
        raise RuntimeError("PUT failed")

    monkeypatch.setattr(app_mod, "upload_via_signed_put", _boom)
    files = [OutputFile(path="/w/x.png", filename="x.png", download_url="/files/x")]
    out = app_mod._maybe_upload_primary_output(files, _target(), job_id="j")
    assert out[0].storage_url is None  # fallback: backend downloads via /files
