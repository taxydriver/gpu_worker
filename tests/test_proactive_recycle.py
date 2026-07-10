"""Proactive ComfyUI recycle to clear accumulated / fragmented VRAM.

The worker's watchdog restarts ComfyUI while idle when either trigger fires:
  • WORKER_RECYCLE_AFTER_JOBS   — N jobs completed since the last recycle.
  • WORKER_RECYCLE_MIN_FREE_MIB — free-VRAM floor, but only after a /free fails
    to recover it (so a warm model pinned by --highvram never false-triggers).

Covers the decision logic (_periodic_recycle_due), the quiesce guard
(_acquire_all_slots), and the orchestration (_maybe_recycle_comfy).
"""
from __future__ import annotations

import threading
import types

import pytest

app_mod = pytest.importorskip("gpu_worker.app")


def _settings(*, after_jobs: int = 0, min_free_mib: int = 0):
    return types.SimpleNamespace(
        worker_recycle_after_jobs=after_jobs,
        worker_recycle_min_free_mib=min_free_mib,
    )


# ── _periodic_recycle_due ─────────────────────────────────────────────────────

def test_periodic_disabled_when_after_jobs_zero():
    assert app_mod._periodic_recycle_due(completed=100, baseline=0, after_jobs=0) is False


def test_periodic_due_at_threshold():
    assert app_mod._periodic_recycle_due(completed=40, baseline=0, after_jobs=40) is True


def test_periodic_not_due_below_threshold():
    assert app_mod._periodic_recycle_due(completed=39, baseline=0, after_jobs=40) is False


def test_periodic_counts_from_baseline():
    assert app_mod._periodic_recycle_due(completed=90, baseline=50, after_jobs=40) is True
    assert app_mod._periodic_recycle_due(completed=89, baseline=50, after_jobs=40) is False


# ── _acquire_all_slots ────────────────────────────────────────────────────────

def test_acquire_all_slots_grabs_when_idle(monkeypatch):
    sem = threading.BoundedSemaphore(2)
    monkeypatch.setattr(app_mod, "_MAX_EXECUTION_SLOTS", 2)
    monkeypatch.setattr(app_mod, "_EXECUTION_SEMAPHORE", sem)

    assert app_mod._acquire_all_slots(0.1) is True
    # All permits held — a further non-blocking acquire must fail.
    assert sem.acquire(blocking=False) is False
    app_mod._release_all_slots()
    # Balanced release restored both permits.
    assert sem.acquire(blocking=False) is True
    assert sem.acquire(blocking=False) is True


def test_acquire_all_slots_defers_when_busy(monkeypatch):
    sem = threading.BoundedSemaphore(1)
    sem.acquire()  # a job holds the only slot
    monkeypatch.setattr(app_mod, "_MAX_EXECUTION_SLOTS", 1)
    monkeypatch.setattr(app_mod, "_EXECUTION_SEMAPHORE", sem)

    assert app_mod._acquire_all_slots(0.2) is False
    # It must not have leaked a release: the slot is still held by the "job".
    assert sem.acquire(blocking=False) is False


# ── _maybe_recycle_comfy ──────────────────────────────────────────────────────

@pytest.fixture
def spy(monkeypatch):
    """Record restart/free calls; neutralise the real semaphore quiesce."""
    calls: list = []
    monkeypatch.setattr(app_mod, "restart_comfy", lambda: calls.append("restart"))
    monkeypatch.setattr(app_mod, "free_comfy_memory", lambda **kw: calls.append(("free", kw)))
    monkeypatch.setattr(app_mod, "_acquire_all_slots", lambda timeout_sec: True)
    monkeypatch.setattr(app_mod, "_release_all_slots", lambda: None)
    monkeypatch.setattr(app_mod, "_ACTIVE_JOBS", 0)
    monkeypatch.setattr(app_mod, "_JOBS_COMPLETED", 0)
    return calls


def test_disabled_does_nothing(spy):
    assert app_mod._maybe_recycle_comfy(_settings(), recycle_baseline=0) is False
    assert spy == []


def test_skips_when_busy(spy, monkeypatch):
    monkeypatch.setattr(app_mod, "_ACTIVE_JOBS", 1)
    monkeypatch.setattr(app_mod, "_JOBS_COMPLETED", 999)
    assert app_mod._maybe_recycle_comfy(_settings(after_jobs=10), recycle_baseline=0) is False
    assert spy == []


def test_periodic_trigger_restarts(spy, monkeypatch):
    monkeypatch.setattr(app_mod, "_JOBS_COMPLETED", 40)
    assert app_mod._maybe_recycle_comfy(_settings(after_jobs=40), recycle_baseline=0) is True
    assert calls_has(spy, "restart")


def test_periodic_not_yet_due(spy, monkeypatch):
    monkeypatch.setattr(app_mod, "_JOBS_COMPLETED", 39)
    assert app_mod._maybe_recycle_comfy(_settings(after_jobs=40), recycle_baseline=0) is False
    assert spy == []


def test_vram_floor_restarts_when_free_does_not_recover(spy, monkeypatch):
    # Free VRAM below floor, and STILL below after /free → genuinely leaked.
    vals = iter([1000, 800])
    monkeypatch.setattr(app_mod, "_free_vram_mib", lambda: next(vals))
    assert app_mod._maybe_recycle_comfy(_settings(min_free_mib=4000), recycle_baseline=0) is True
    # /free was tried first, then a restart.
    assert any(c == "restart" for c in spy)
    assert any(isinstance(c, tuple) and c[0] == "free" for c in spy)


def test_vram_floor_no_restart_when_free_recovers(spy, monkeypatch):
    # Low free VRAM was just the pinned warm model — /free brings it back.
    vals = iter([1000, 9000])
    monkeypatch.setattr(app_mod, "_free_vram_mib", lambda: next(vals))
    assert app_mod._maybe_recycle_comfy(_settings(min_free_mib=4000), recycle_baseline=0) is False
    assert any(isinstance(c, tuple) and c[0] == "free" for c in spy)  # flushed
    assert "restart" not in spy  # but did not recycle


def test_vram_floor_skips_when_plenty_free(spy, monkeypatch):
    monkeypatch.setattr(app_mod, "_free_vram_mib", lambda: 50000)
    assert app_mod._maybe_recycle_comfy(_settings(min_free_mib=4000), recycle_baseline=0) is False
    assert spy == []  # never even called /free


def calls_has(calls, name) -> bool:
    return any(c == name for c in calls)
