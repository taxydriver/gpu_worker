"""Tests for is_asset_group_warm() — disk-based warm detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_worker import asset_manager
from gpu_worker.asset_manager import is_asset_group_warm
from gpu_worker.asset_registry import ASSET_REGISTRY


def _fake_group(name: str, paths: list[Path]) -> dict[str, list[dict[str, str]]]:
    """Build a single-group registry that points at the given local paths."""
    return {
        name: [
            {"name": f"asset_{i}", "path": str(p), "url": "https://example/none"}
            for i, p in enumerate(paths)
        ]
    }


def test_warm_when_all_files_present(monkeypatch, tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    monkeypatch.setattr(asset_manager, "get_asset_group",
                        lambda name: _fake_group("g1", [a, b])["g1"])
    assert is_asset_group_warm("g1") is True


def test_cold_when_one_file_missing(monkeypatch, tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"x")
    # b is missing
    monkeypatch.setattr(asset_manager, "get_asset_group",
                        lambda name: _fake_group("g1", [a, b])["g1"])
    assert is_asset_group_warm("g1") is False


def test_cold_when_file_empty(monkeypatch, tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"x")
    b.write_bytes(b"")  # empty file — treat as missing
    monkeypatch.setattr(asset_manager, "get_asset_group",
                        lambda name: _fake_group("g1", [a, b])["g1"])
    assert is_asset_group_warm("g1") is False


def test_cold_for_unknown_group():
    assert is_asset_group_warm("definitely_not_a_group") is False


def test_real_registry_groups_resolve_without_error():
    # All keys in the real ASSET_REGISTRY should produce a bool answer
    # without raising. Disk presence depends on the test machine, so just
    # check the call succeeds.
    for group in ASSET_REGISTRY:
        result = is_asset_group_warm(group)
        assert isinstance(result, bool)
