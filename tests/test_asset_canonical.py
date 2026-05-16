"""Tests for canonical asset-group naming."""

from __future__ import annotations

from gpu_worker.asset_canonical import canonical_asset_group, canonicalize_groups
from gpu_worker.asset_registry import ASSET_REGISTRY


def test_registry_keys_are_canonical():
    for key in ASSET_REGISTRY:
        assert canonical_asset_group(key) == key


def test_legacy_aliases_resolve_to_canonical():
    # Legacy capability names → canonical asset-group names
    assert canonical_asset_group("wan_i2v") == "wan_i2v_v1"
    assert canonical_asset_group("flux2_stills") == "flux_stills_v1"
    assert canonical_asset_group("stable_audio") == "stable_audio_v1"
    assert canonical_asset_group("juggernaut_stills") == "juggernaut_stills_v1"


def test_idempotent():
    for alias in ("wan_i2v", "wan_i2v_v1", "flux2_stills", "flux_stills_v1"):
        once = canonical_asset_group(alias)
        twice = canonical_asset_group(once)
        assert once == twice


def test_unknown_returns_none():
    assert canonical_asset_group("not_a_real_group") is None
    assert canonical_asset_group("") is None
    assert canonical_asset_group(None) is None
    assert canonical_asset_group("   ") is None


def test_canonicalize_groups_dedups_and_preserves_order():
    result = canonicalize_groups(["wan_i2v", "wan_i2v_v1", "flux_stills_v1", "flux2_stills"])
    assert result == ["wan_i2v_v1", "flux_stills_v1"]


def test_canonicalize_groups_drops_unknown():
    result = canonicalize_groups(["wan_i2v", "garbage", "flux_stills_v1"])
    assert result == ["wan_i2v_v1", "flux_stills_v1"]


def test_canonicalize_groups_empty():
    assert canonicalize_groups([]) == []
    assert canonicalize_groups(None) == []
