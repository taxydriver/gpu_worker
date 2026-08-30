"""Runtime guard for the XLabs Flux IPAdapter graph (``flux_ipadapter_v1``).

A declared capability is intentionally insufficient. ``LoadFluxIPAdapter`` and
``ApplyFluxIPAdapter`` come from a custom node that ComfyUI only discovers at
start, and the adapter + CLIP-vision weights land on the data volume
asynchronously. Advertising the group before both are true is exactly how the
2026-08-22 render 400d with ``missing_node_type`` after the GPU had been paid
for: the asset map said "present", nothing had checked.

Same shape as :mod:`gpu_worker.infinitetalk`, minus its audio-venv import probe
— the XLabs node's own imports (transformers, safetensors) run inside ComfyUI's
process, so a failed import shows up here as the class missing from
``/object_info``. This module owns the readiness facts only; it does not choose
a renderer or expose a product endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from gpu_worker.asset_registry import get_asset_group
from gpu_worker.config import get_settings


FLUX_IPADAPTER_ASSET_GROUP = "flux_ipadapter_v1"
# The two classes the backend graph (flux2_stills_xlabs_ipa.json) is built
# against. Both come from XLabs-AI/x-flux-comfyui (provision_flux_ipadapter.sh).
REQUIRED_NODE_CLASSES = frozenset({"LoadFluxIPAdapter", "ApplyFluxIPAdapter"})


@dataclass(frozen=True)
class FluxIPAdapterReadiness:
    """Exact worker facts needed before advertising ``flux_ipadapter_v1``."""

    ready: bool
    missing_files: tuple[str, ...] = ()
    missing_node_classes: tuple[str, ...] = ()
    comfy_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "missing_files": list(self.missing_files),
            "missing_node_classes": list(self.missing_node_classes),
            "comfy_error": self.comfy_error,
        }


def _required_files() -> tuple[str, ...]:
    return tuple(
        asset["path"]
        for asset in get_asset_group(FLUX_IPADAPTER_ASSET_GROUP)
        if not Path(asset["path"]).expanduser().is_file()
        or Path(asset["path"]).expanduser().stat().st_size <= 0
    )


def check_flux_ipadapter_readiness() -> FluxIPAdapterReadiness:
    """Check the staged weights and ComfyUI's actually-registered node classes."""

    missing_files = _required_files()

    missing_nodes: tuple[str, ...] = ()
    comfy_error: str | None = None
    try:
        base_url = get_settings().comfy_base_url.rstrip("/")
        response = requests.get(f"{base_url}/object_info", timeout=10)
        response.raise_for_status()
        object_info = response.json()
        if not isinstance(object_info, dict):
            raise RuntimeError("ComfyUI /object_info did not return an object")
        missing_nodes = tuple(sorted(REQUIRED_NODE_CLASSES - set(object_info)))
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        comfy_error = str(exc)
        missing_nodes = tuple(sorted(REQUIRED_NODE_CLASSES))

    return FluxIPAdapterReadiness(
        ready=not (missing_files or missing_nodes or comfy_error),
        missing_files=missing_files,
        missing_node_classes=missing_nodes,
        comfy_error=comfy_error,
    )
