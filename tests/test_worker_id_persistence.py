"""Tests for persistent WORKER_ID resolution.

Verifies:
  - explicit RENDER_BROKER_WORKER_ID env var is used as-is
  - missing env var → generates UUID and persists to WORKER_ID_FILE
  - subsequent reads from the same WORKER_ID_FILE return the same UUID
  - non-UUID env var is rejected at construction
  - corrupted WORKER_ID_FILE contents trigger regeneration
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest


def _fresh_settings(env: dict[str, str]):
    """Build a fresh Settings instance bypassing the module-level lru_cache."""
    for k in ("RENDER_BROKER_WORKER_ID", "WORKER_ID_FILE"):
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = v
    from gpu_worker.config import Settings
    return Settings()


def test_env_var_overrides(tmp_path):
    explicit = str(uuid.uuid4())
    s = _fresh_settings({
        "RENDER_BROKER_WORKER_ID": explicit,
        "WORKER_ID_FILE": str(tmp_path / "id"),
    })
    assert s.resolved_worker_id() == explicit
    # File should NOT be created when env var supplies the ID.
    assert not (tmp_path / "id").exists()


def test_generates_and_persists_when_missing(tmp_path):
    path = tmp_path / "id"
    s = _fresh_settings({"WORKER_ID_FILE": str(path)})
    first = s.resolved_worker_id()
    # Persisted
    assert path.exists()
    assert path.read_text().strip() == first
    # New Settings reading the same file returns the same UUID
    s2 = _fresh_settings({"WORKER_ID_FILE": str(path)})
    assert s2.resolved_worker_id() == first


def test_rejects_non_uuid_env_var(tmp_path):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        _fresh_settings({
            "RENDER_BROKER_WORKER_ID": "eb83e44d2353-gpu-worker",
            "WORKER_ID_FILE": str(tmp_path / "id"),
        })


def test_corrupted_file_regenerates(tmp_path):
    path = tmp_path / "id"
    path.write_text("not-a-uuid")
    s = _fresh_settings({"WORKER_ID_FILE": str(path)})
    fresh = s.resolved_worker_id()
    uuid.UUID(fresh)  # raises if not a UUID
    # File was overwritten with the fresh UUID
    assert path.read_text().strip() == fresh
