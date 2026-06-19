"""Self-healing model store wiring.

The asset registry hardcodes model paths under /workspace/ComfyUI/models (the OS
disk). The actual models live on the persistent data volume at
/mnt/data/ComfyUI/models. If the deploy-time bind mount is missing or was lost,
the worker would see an empty /workspace/ComfyUI/models and re-download the full
warm set even though the volume is already populated.

This module runs at worker startup (before any asset check) and guarantees that
/workspace/ComfyUI/models resolves to the data volume, so already-downloaded
models are reused. It is idempotent and degrades to a warning — it never blocks
worker startup.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_MODELS = Path("/mnt/data/ComfyUI/models")
WORKSPACE_MODELS = Path("/workspace/ComfyUI/models")
COMFY_ROOT = Path("/workspace/ComfyUI")

_MODEL_SUBDIRS = (
    "checkpoints", "clip", "vae", "diffusion_models",
    "text_encoders", "loras", "unet", "controlnet", "upscale_models",
)


def _is_mountpoint(path: Path) -> bool:
    try:
        return subprocess.run(
            ["mountpoint", "-q", str(path)],
            timeout=5,
        ).returncode == 0
    except Exception:
        return False


def _dir_has_models(path: Path) -> bool:
    """True if path holds real model files (not just empty stub dirs)."""
    try:
        for sub in ("diffusion_models", "vae", "text_encoders"):
            d = path / sub
            if d.is_dir():
                for f in d.iterdir():
                    if f.is_file() and f.suffix == ".safetensors" and f.stat().st_size > 0:
                        return True
        return False
    except Exception:
        return False


def _write_extra_model_paths() -> None:
    """Point ComfyUI directly at the data volume, independent of the bind."""
    yaml = "filmforge_worker:\n    base_path: /mnt/data/ComfyUI/models\n"
    for sub in _MODEL_SUBDIRS:
        yaml += f"    {sub}: {sub}\n"
    target = COMFY_ROOT / "extra_model_paths.yaml"
    try:
        if not target.exists() or target.read_text() != yaml:
            target.write_text(yaml)
        logger.info("extra_model_paths.yaml points ComfyUI at %s", DATA_MODELS)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Could not write extra_model_paths.yaml: %s", exc)


def ensure_model_store() -> None:
    """Make sure /workspace/ComfyUI/models resolves to the data volume.

    Order of preference:
      1. If /workspace/ComfyUI/models is already a mountpoint or a symlink to the
         volume, do nothing.
      2. If the data volume holds models, symlink /workspace/ComfyUI/models →
         /mnt/data/ComfyUI/models (replacing any empty OS-disk dir).
      3. Always (re)write extra_model_paths.yaml as a belt-and-suspenders pointer.
    """
    try:
        DATA_MODELS.mkdir(parents=True, exist_ok=True)
        for sub in _MODEL_SUBDIRS:
            (DATA_MODELS / sub).mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("model-store: cannot prepare %s: %s — skipping self-heal", DATA_MODELS, exc)
        _write_extra_model_paths()
        return

    # Already wired via bind mount?
    if _is_mountpoint(WORKSPACE_MODELS):
        logger.info("model-store: %s is bind-mounted — OK", WORKSPACE_MODELS)
        _write_extra_model_paths()
        return

    # Already a symlink to the volume?
    if WORKSPACE_MODELS.is_symlink():
        try:
            if WORKSPACE_MODELS.resolve() == DATA_MODELS.resolve():
                logger.info("model-store: %s already symlinks to volume — OK", WORKSPACE_MODELS)
                _write_extra_model_paths()
                return
        except Exception:
            pass

    # Decide whether the workspace dir holds real models we must not clobber.
    workspace_has_models = _dir_has_models(WORKSPACE_MODELS)
    volume_has_models = _dir_has_models(DATA_MODELS)

    if workspace_has_models and not volume_has_models:
        # Models were downloaded to the OS disk; migrate them onto the volume so
        # they survive teardown, then symlink.
        logger.warning(
            "model-store: models found on OS disk but not on volume — migrating to %s",
            DATA_MODELS,
        )
        try:
            subprocess.run(
                ["cp", "-an", f"{WORKSPACE_MODELS}/.", f"{DATA_MODELS}/"],
                check=True, timeout=3600,
            )
            volume_has_models = True
        except Exception as exc:
            logger.warning("model-store: migration failed: %s — leaving OS-disk models in place", exc)
            _write_extra_model_paths()
            return

    # Replace the workspace path with a symlink to the volume.
    try:
        COMFY_ROOT.mkdir(parents=True, exist_ok=True)
        if WORKSPACE_MODELS.is_symlink() or WORKSPACE_MODELS.exists():
            if WORKSPACE_MODELS.is_dir() and not WORKSPACE_MODELS.is_symlink():
                # Empty/stale OS-disk dir — remove it so the symlink can take its place.
                subprocess.run(["rm", "-rf", str(WORKSPACE_MODELS)], check=True, timeout=120)
            else:
                WORKSPACE_MODELS.unlink()
        os.symlink(DATA_MODELS, WORKSPACE_MODELS)
        logger.info(
            "model-store: symlinked %s -> %s (volume models present: %s)",
            WORKSPACE_MODELS, DATA_MODELS, volume_has_models,
        )
    except Exception as exc:
        logger.warning("model-store: could not symlink workspace to volume: %s", exc)

    _write_extra_model_paths()
