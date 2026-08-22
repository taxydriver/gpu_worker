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
    assert canonical_asset_group("stable_audio3") == "stable_audio3_v1"


def test_flux_ipadapter_capability_token_is_not_dropped():
    # The backend's rent path emits WORKER_CAPABILITIES=...,flux_ipadapter and
    # the worker canonicalises that list before advertising. An alias missing
    # here is silently dropped, so the box advertised only flux_stills_v1 while
    # the group sat in known_asset_groups (observed 2026-08-22).
    assert canonical_asset_group("flux_ipadapter") == "flux_ipadapter_v1"
    assert canonical_asset_group("flux2_ipadapter") == "flux_ipadapter_v1"
    assert canonicalize_groups(["flux2_stills", "flux_ipadapter", "character_loras"]) == [
        "flux_stills_v1",
        "flux_ipadapter_v1",
        "character_loras_v1",
    ]


def test_flux_ipadapter_weights_are_named_for_the_xlabs_graph():
    # Filename contract with the backend graph (flux2_stills_xlabs_ipa.json,
    # node LoadFluxIPAdapter): ComfyUI lists these directories and the graph
    # names a file in each. The encoder must be OpenAI CLIP ViT-L/14 — XLabs'
    # projection model takes 768-dim embeds, so a bigG file cannot feed it.
    paths = {a["name"]: a["path"] for a in ASSET_REGISTRY["flux_ipadapter_v1"]}
    assert paths["flux_ip_adapter_v2"].endswith(
        "/models/xlabs/ipadapters/flux-ip-adapter-v2.safetensors"
    )
    assert paths["clip_vision_large"].endswith(
        "/models/clip_vision/clip-vit-large-patch14.safetensors"
    )
    urls = {a["name"]: a["url"] for a in ASSET_REGISTRY["flux_ipadapter_v1"]}
    assert urls["clip_vision_large"].startswith(
        "https://huggingface.co/openai/clip-vit-large-patch14/"
    )


def test_retired_stable_audio1_aliases_are_unknown():
    # The legacy Stable Audio 1 ComfyUI path is retired (SA3 is the audio engine).
    assert canonical_asset_group("stable_audio") is None
    assert canonical_asset_group("stable_audio1") is None
    assert canonical_asset_group("stable_audio_v1") is None



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
