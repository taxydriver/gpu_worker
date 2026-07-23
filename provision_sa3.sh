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

log "Downloading stable-audio-3-medium snapshot (~10.4GB, gated)"
HF_TOKEN="$HF_TOKEN" "$WORKSPACE/sa3_spike/bin/python" - <<'PY'
import os
from huggingface_hub import login, snapshot_download
login(token=os.environ["HF_TOKEN"])
print("[provision_sa3] snapshot ->", snapshot_download("stabilityai/stable-audio-3-medium",
                                                       token=os.environ["HF_TOKEN"]))
PY
log "Stable Audio 3 ready: $WORKSPACE/sa3_spike"
