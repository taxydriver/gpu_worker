#!/usr/bin/env bash
# Provision Stable Audio 3 (music/score) on a GPU worker box.
#
# Companion to provision_tts.sh and asset_registry.py. Stable Audio 3 is a standalone
# pip package (NOT a single-file ComfyUI model), loading a gated HF-repo snapshot via
# from_pretrained — so it gets a provisioner, not an asset_registry file group.
#
# Model (validated 2026-07-15, replaces Stable Audio Open 1.0 — "leaps and bounds" better):
#   stabilityai/stable-audio-3-medium — full 2-min stereo, distilled 8-step, ~10.4GB weights.
#   Stability AI Community License (commercial-free under $1M revenue).
#   Runs on a 24GB card (or CPU, slowly). Wants ~15GB disk for the snapshot.
#
# Use the OFFICIAL `stable-audio-3` lib (StableAudioModel.from_pretrained("medium")),
# NOT `stable-audio-tools` (pins pandas==2.0.2 -> build-fails on Python 3.12).
#
# Idempotent. HF_TOKEN is REQUIRED (repo is gated: accept the license once at
# hf.co/stabilityai/stable-audio-3-medium with the token's account).
#
# Usage:  HF_TOKEN=hf_xxx bash gpu_worker/provision_sa3.sh
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
export HF_XET_HIGH_PERFORMANCE=1
# Keep the ~10GB model snapshot OFF the small root fs — Verda boxes bind-mount a big
# data volume at /mnt/data. Point the HF cache there when it exists (else default).
if [ -z "${HF_HOME:-}" ] && [ -d /mnt/data ]; then
  export HF_HOME=/mnt/data/hf_cache
  mkdir -p "$HF_HOME"
fi
log() { echo "[provision_sa3] $*" >&2; }

if [ -z "${HF_TOKEN:-}" ]; then
  log "ERROR: Stable Audio 3 repo is gated — set HF_TOKEN (accept terms at hf.co/stabilityai/stable-audio-3-medium first)."
  exit 1
fi

log "Stable Audio 3: venv + pip (torch from the CUDA index FIRST)"
python3 -m venv --system-site-packages "$WORKSPACE/sa3_spike"
"$WORKSPACE/sa3_spike/bin/pip" install -U pip setuptools wheel
# The stable-audio-3 repo is built for `uv`: its pyproject pins torch==2.7.1 /
# torchaudio==2.7.1 and routes them to the PyTorch cu126 index via
# [tool.uv.sources] — which pip does NOT read. Left to pip, torch resolves against
# PyPI alongside transformers>=5.8.0 / numpy>=2.2.6 / huggingface-hub>=1.7.1 and
# the resolver backtracks forever (the observed install stall, 2026-07-23). So
# install the pinned torch pair from the CUDA wheel index up front — this both
# gets the GPU build and satisfies the heaviest constraint before the git install,
# so pip no longer backtracks across every torch version.
"$WORKSPACE/sa3_spike/bin/pip" install torch==2.7.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
# There is no `stable-audio-3` package on PyPI — install the lib straight from git
# (no -q, so setup progress shows in the deploy logs). torch is already satisfied.
"$WORKSPACE/sa3_spike/bin/pip" install "git+https://github.com/Stability-AI/stable-audio-3.git"

# Blackwell (sm_100/sm_120) override. The pinned torch 2.7.1+cu126 above carries
# kernels only up to sm_90, so on an RTX PRO 6000 / B200 class card SA3 loads and
# then dies at inference with "CUDA error: no kernel image is available for
# execution on the device" (observed live on 4× RTX PRO 6000, 2026-07-26 — Parler
# was fine because its venv had torch 2.13+cu130). Upgrade the pair AFTER the
# package install so pip can't drag it back to the pyproject pin; the pin
# violation is deliberate and pip's warning about it is expected. The nvidia-*
# cu12 wheels must go first — a plain --upgrade leaves them in place and torch 13
# then fails to dlopen libcudart.so.13.
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2>/dev/null | head -1 | tr -d ' ')"
GPU_CC_MAJOR="${GPU_CC%%.*}"
if [ -n "$GPU_CC_MAJOR" ] && [ "$GPU_CC_MAJOR" -ge 10 ] 2>/dev/null; then
  log "GPU compute capability $GPU_CC — swapping SA3's torch for a Blackwell-capable cu130 build"
  "$WORKSPACE/sa3_spike/bin/pip" list 2>/dev/null | awk '/^nvidia-/ {print $1}' \
    | xargs -r "$WORKSPACE/sa3_spike/bin/pip" uninstall -y >/dev/null 2>&1 || true
  "$WORKSPACE/sa3_spike/bin/pip" uninstall -y torch torchaudio >/dev/null 2>&1 || true
  "$WORKSPACE/sa3_spike/bin/pip" install torch==2.13.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130
  "$WORKSPACE/sa3_spike/bin/python" -c \
    "import torch; assert 'sm_120' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list(); print('[provision_sa3] torch', torch.__version__, 'has Blackwell kernels')"
fi

log "Downloading stable-audio-3-medium snapshot (~10.4GB, gated)"
HF_TOKEN="$HF_TOKEN" "$WORKSPACE/sa3_spike/bin/python" - <<'PY'
import os
from huggingface_hub import login, snapshot_download
login(token=os.environ["HF_TOKEN"])
print("[provision_sa3] snapshot ->", snapshot_download("stabilityai/stable-audio-3-medium",
                                                       token=os.environ["HF_TOKEN"]))
PY
log "Stable Audio 3 ready: $WORKSPACE/sa3_spike"
