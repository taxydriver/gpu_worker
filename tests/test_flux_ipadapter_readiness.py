"""flux_ipadapter_v1 is advertised only once the box can actually render it.

The 2026-08-22 render 400d with `missing_node_type: LoadFluxIPAdapter` after the
GPU had been paid for, because the asset map said "present" and nothing had
checked ComfyUI. The gate mirrors gpu_worker/infinitetalk.py: staged weights on
disk AND the node classes registered in /object_info, or the group is withheld
from /health `capabilities` and from broker registration — with the reason.
"""

from __future__ import annotations

from types import SimpleNamespace

from gpu_worker import app
from gpu_worker import flux_ipadapter as runtime


def test_readiness_requires_both_xlabs_nodes(monkeypatch):
    monkeypatch.setattr(runtime, "_required_files", lambda: ())
    monkeypatch.setattr(runtime, "get_settings", lambda: SimpleNamespace(comfy_base_url="http://comfy:8188"))
    monkeypatch.setattr(
        runtime.requests,
        "get",
        lambda *a, **k: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"LoadFluxIPAdapter": {}, "KSampler": {}},
        ),
    )

    result = runtime.check_flux_ipadapter_readiness()

    assert result.ready is False
    assert result.missing_node_classes == ("ApplyFluxIPAdapter",)
    assert result.comfy_error is None


def test_readiness_requires_the_staged_weights(monkeypatch):
    monkeypatch.setattr(runtime, "_required_files", lambda: ("/workspace/ComfyUI/models/xlabs/ipadapters/flux-ip-adapter-v2.safetensors",))
    monkeypatch.setattr(runtime, "get_settings", lambda: SimpleNamespace(comfy_base_url="http://comfy:8188"))
    monkeypatch.setattr(
        runtime.requests,
        "get",
        lambda *a, **k: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"LoadFluxIPAdapter": {}, "ApplyFluxIPAdapter": {}},
        ),
    )

    result = runtime.check_flux_ipadapter_readiness()

    assert result.ready is False
    assert result.missing_node_classes == ()
    assert result.missing_files == ("/workspace/ComfyUI/models/xlabs/ipadapters/flux-ip-adapter-v2.safetensors",)


def test_readiness_holds_when_weights_and_nodes_are_present(monkeypatch):
    monkeypatch.setattr(runtime, "_required_files", lambda: ())
    monkeypatch.setattr(runtime, "get_settings", lambda: SimpleNamespace(comfy_base_url="http://comfy:8188"))
    monkeypatch.setattr(
        runtime.requests,
        "get",
        lambda *a, **k: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"LoadFluxIPAdapter": {}, "ApplyFluxIPAdapter": {}},
        ),
    )

    assert runtime.check_flux_ipadapter_readiness().ready is True


def test_unreachable_comfy_reports_every_node_missing(monkeypatch):
    monkeypatch.setattr(runtime, "_required_files", lambda: ())
    monkeypatch.setattr(runtime, "get_settings", lambda: SimpleNamespace(comfy_base_url="http://comfy:8188"))

    def _down(*a, **k):
        raise runtime.requests.ConnectionError("refused")

    monkeypatch.setattr(runtime.requests, "get", _down)

    result = runtime.check_flux_ipadapter_readiness()

    assert result.ready is False
    assert result.missing_node_classes == ("ApplyFluxIPAdapter", "LoadFluxIPAdapter")
    assert "refused" in (result.comfy_error or "")


def test_unready_flux_ipadapter_is_withheld_from_advertisement(monkeypatch):
    monkeypatch.setattr(
        app,
        "get_settings",
        lambda: SimpleNamespace(resolved_capabilities=lambda: ["flux2_stills", "flux_ipadapter", "character_loras"]),
    )
    monkeypatch.setattr(
        app,
        "check_flux_ipadapter_readiness",
        lambda: runtime.FluxIPAdapterReadiness(ready=False, missing_node_classes=("LoadFluxIPAdapter",)),
    )

    capabilities, infinitetalk_readiness, readiness = app._advertised_capabilities()

    # The symptom the gate exists for: the group is declared, so it would have
    # been advertised — now it is withheld and /health carries the cause.
    assert capabilities == ["flux_stills_v1", "character_loras_v1"]
    assert infinitetalk_readiness is None
    assert readiness == {
        "ready": False,
        "missing_files": [],
        "missing_node_classes": ["LoadFluxIPAdapter"],
        "comfy_error": None,
    }


def test_ready_flux_ipadapter_is_advertised(monkeypatch):
    monkeypatch.setattr(
        app,
        "get_settings",
        lambda: SimpleNamespace(resolved_capabilities=lambda: ["flux2_stills", "flux_ipadapter"]),
    )
    monkeypatch.setattr(
        app,
        "check_flux_ipadapter_readiness",
        lambda: runtime.FluxIPAdapterReadiness(ready=True),
    )

    capabilities, _, readiness = app._advertised_capabilities()

    assert capabilities == ["flux_stills_v1", "flux_ipadapter_v1"]
    assert readiness["ready"] is True


def test_undeclared_flux_ipadapter_is_not_probed(monkeypatch):
    monkeypatch.setattr(
        app,
        "get_settings",
        lambda: SimpleNamespace(resolved_capabilities=lambda: ["flux2_stills"]),
    )

    def _must_not_run():
        raise AssertionError("readiness probed for an undeclared group")

    monkeypatch.setattr(app, "check_flux_ipadapter_readiness", _must_not_run)

    capabilities, _, readiness = app._advertised_capabilities()

    assert capabilities == ["flux_stills_v1"]
    assert readiness is None
