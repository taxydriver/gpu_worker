"""FilmForge: disable cuDNN SDPA on ComfyUI startup.

Deployed as a ComfyUI custom node:
  /workspace/ComfyUI/custom_nodes/filmforge_cuda_patch/__init__.py

Runs before any model is loaded, setting torch backend flags that prevent
"cuDNN Frontend error: No valid execution plans built" on Blackwell (B300,
SM 10.x) and other GPUs where cuDNN has no compiled fusion plan for the
attention kernels it attempts to use.
"""

import logging as _logging

_log = _logging.getLogger("filmforge.cuda_patch")

try:
    import torch as _torch

    # ── Startup diagnostics ────────────────────────────────────────────────────
    _log.info(
        "[filmforge] torch=%s cuda=%s cudnn=%s",
        _torch.__version__,
        _torch.version.cuda,
        _torch.backends.cudnn.version(),
    )
    if _torch.cuda.is_available():
        _log.info(
            "[filmforge] device=%s capability=%s",
            _torch.cuda.get_device_name(0),
            _torch.cuda.get_device_capability(0),
        )

    try:
        import xformers as _xf
        _log.info("[filmforge] xformers=%s", _xf.__version__)
    except ImportError:
        _log.info("[filmforge] xformers=not_installed")

    # ── cuDNN SDPA disable ─────────────────────────────────────────────────────
    # Disabling the cuDNN SDP backend prevents "No valid execution plans built"
    # errors on architectures where cuDNN 9.x has no precompiled fusion plan
    # (Blackwell SM 10.x, early B200/B300 support).  The three remaining
    # backends (flash, mem_efficient, math) are always enabled as the fallback
    # chain; math_sdp in particular is a universal software fallback.
    _torch.backends.cuda.enable_cudnn_sdp(False)
    _torch.backends.cuda.enable_flash_sdp(True)
    _torch.backends.cuda.enable_mem_efficient_sdp(True)
    _torch.backends.cuda.enable_math_sdp(True)
    _torch.backends.cudnn.benchmark = False

    _log.info(
        "[filmforge] cuDNN SDPA disabled; flash=%s mem_efficient=%s math=%s",
        _torch.backends.cuda.flash_sdp_enabled(),
        _torch.backends.cuda.mem_efficient_sdp_enabled(),
        _torch.backends.cuda.math_sdp_enabled(),
    )

    # ── GPU smoke test ─────────────────────────────────────────────────────────
    if _torch.cuda.is_available():
        try:
            import torch.nn.functional as _F

            # Tiny tensor allocation
            _t = _torch.zeros(4, device="cuda")

            # Tiny conv2d
            _x = _torch.randn(1, 1, 4, 4, device="cuda")
            _w = _torch.randn(1, 1, 2, 2, device="cuda")
            _F.conv2d(_x, _w)

            # Tiny scaled_dot_product_attention — uses the patched backends
            _q = _torch.randn(1, 1, 4, 8, device="cuda")
            _k = _torch.randn(1, 1, 4, 8, device="cuda")
            _v = _torch.randn(1, 1, 4, 8, device="cuda")
            _F.scaled_dot_product_attention(_q, _k, _v)

            _log.info("[filmforge] GPU smoke test: PASS")
        except Exception as _smoke_exc:
            _log.error("[filmforge] GPU smoke test: FAIL — %s", _smoke_exc)

except Exception as _exc:
    _log.warning("[filmforge] cuDNN patch failed: %s", _exc)

# Required by ComfyUI to recognise this as a valid extension
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
