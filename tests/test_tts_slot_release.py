"""G573 — _run_tts_dialogue took an execution slot and an _ACTIVE_JOBS count and never gave
either back, so a VM stopped taking work after N TTS jobs. Both paths must release."""
from __future__ import annotations

import threading
import time
import types

import pytest

app_mod = pytest.importorskip("gpu_worker.app")
schemas = pytest.importorskip("gpu_worker.schemas")


@pytest.fixture
def one_slot(monkeypatch):
    sem = threading.BoundedSemaphore(1)
    monkeypatch.setattr(app_mod, "_EXECUTION_SEMAPHORE", sem)
    monkeypatch.setattr(app_mod, "_ACTIVE_JOBS", 0)
    monkeypatch.setattr(app_mod, "_send_heartbeat_now", lambda: None)
    monkeypatch.setattr(app_mod, "_note_job_completed", lambda: None)
    return sem


def _request(text: str):
    return schemas.RunRequest(job_id="tts-1", asset_group="tts_dialogue_v1",
                              comfy_payload={"text": text}, timeout_sec=30)


def test_a_failed_tts_job_gives_its_slot_back(one_slot):
    for _ in range(3):  # empty text fails inside the try
        response = app_mod._run_tts_dialogue(_request(""), time.monotonic())
        assert response.ok is False
    assert one_slot.acquire(blocking=False) and app_mod._ACTIVE_JOBS == 0


def test_a_successful_tts_job_gives_its_slot_back(one_slot, monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod.requests, "post",
                        lambda *a, **k: types.SimpleNamespace(status_code=200, content=b"RIFF", text=""))
    monkeypatch.setattr(app_mod, "get_settings", lambda: types.SimpleNamespace(
        served_file_roots=lambda: {"output": tmp_path}, comfy_base_url="http://x"))
    monkeypatch.setattr(app_mod, "build_output_files", lambda paths: [])
    for _ in range(2):
        assert app_mod._run_tts_dialogue(_request("Amma."), time.monotonic()).ok is True
    assert one_slot.acquire(blocking=False) and app_mod._ACTIVE_JOBS == 0
