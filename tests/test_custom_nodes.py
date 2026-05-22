from __future__ import annotations

from pathlib import Path

import requests

from gpu_worker import app as worker_app


IPADAPTER_REQUIREMENT = {
    "name": "ComfyUI_IPAdapter_plus",
    "repo": "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git",
    "dest": "ComfyUI_IPAdapter_plus",
}


def test_juggernaut_requires_ipadapter_custom_node():
    requirements = worker_app._CUSTOM_NODE_REQUIREMENTS["juggernaut_stills_v1"]
    assert requirements == [IPADAPTER_REQUIREMENT]


def test_flux_stills_requires_ipadapter_custom_node():
    """flux_stills_v1 also uses IPAdapterModelLoader for flux2_ipadapter workflow."""
    requirements = worker_app._CUSTOM_NODE_REQUIREMENTS["flux_stills_v1"]
    assert requirements == [IPADAPTER_REQUIREMENT]


def test_ensure_custom_nodes_installs_missing_ipadapter(monkeypatch, tmp_path: Path):
    comfy_root = tmp_path / "ComfyUI"
    calls = []

    monkeypatch.setenv("COMFY_ROOT", str(comfy_root))

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["git", "clone", "--depth=1"]:
            Path(cmd[-1]).mkdir(parents=True)

    monkeypatch.setattr(worker_app.subprocess, "run", fake_run)

    installed, changed = worker_app._ensure_custom_nodes_for_asset_group("juggernaut_stills_v1")

    assert installed == ["ComfyUI_IPAdapter_plus"]
    assert changed is True
    assert (comfy_root / "custom_nodes" / "ComfyUI_IPAdapter_plus").exists()
    assert calls[0][:3] == ["git", "clone", "--depth=1"]


def test_ensure_custom_nodes_installs_ipadapter_for_flux_stills(monkeypatch, tmp_path: Path):
    """Same install step should work for flux_stills_v1."""
    comfy_root = tmp_path / "ComfyUI"
    calls = []

    monkeypatch.setenv("COMFY_ROOT", str(comfy_root))

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["git", "clone", "--depth=1"]:
            Path(cmd[-1]).mkdir(parents=True)

    monkeypatch.setattr(worker_app.subprocess, "run", fake_run)

    installed, changed = worker_app._ensure_custom_nodes_for_asset_group("flux_stills_v1")

    assert installed == ["ComfyUI_IPAdapter_plus"]
    assert changed is True
    assert (comfy_root / "custom_nodes" / "ComfyUI_IPAdapter_plus").exists()


def test_ensure_custom_nodes_noops_when_ipadapter_exists(monkeypatch, tmp_path: Path):
    comfy_root = tmp_path / "ComfyUI"
    (comfy_root / "custom_nodes" / "ComfyUI_IPAdapter_plus").mkdir(parents=True)

    monkeypatch.setenv("COMFY_ROOT", str(comfy_root))

    installed, changed = worker_app._ensure_custom_nodes_for_asset_group("juggernaut_stills_v1")

    assert installed == []
    assert changed is False


def test_ensure_custom_nodes_refreshes_existing_ipadapter(monkeypatch, tmp_path: Path):
    comfy_root = tmp_path / "ComfyUI"
    custom_node = comfy_root / "custom_nodes" / "ComfyUI_IPAdapter_plus"
    custom_node.mkdir(parents=True)
    calls = []

    monkeypatch.setenv("COMFY_ROOT", str(comfy_root))

    def fake_run(cmd, **kwargs):
        calls.append(cmd)

    monkeypatch.setattr(worker_app.subprocess, "run", fake_run)

    installed, changed = worker_app._ensure_custom_nodes_for_asset_group(
        "juggernaut_stills_v1",
        refresh_existing=True,
    )

    assert installed == ["ComfyUI_IPAdapter_plus"]
    assert changed is True
    assert calls[0] == ["git", "-C", str(custom_node), "pull", "--ff-only"]


def test_missing_node_type_from_http_error():
    response = requests.Response()
    response.status_code = 400
    response._content = (
        b'{"error":{"type":"missing_node_type",'
        b'"extra_info":{"class_type":"IPAdapterModelLoader"}}}'
    )
    error = requests.exceptions.HTTPError(response=response)

    assert worker_app._missing_node_type_from_exception(error) == "IPAdapterModelLoader"
