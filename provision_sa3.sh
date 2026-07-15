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
log() { echo "[provision_sa3] $*" >&2; }

if [ -z "${HF_TOKEN:-}" ]; then
  log "ERROR: Stable Audio 3 repo is gated — set HF_TOKEN (accept terms at hf.co/stabilityai/stable-audio-3-medium first)."
  exit 1
fi

log "Stable Audio 3: venv + pip (setuptools first to avoid build-wheel errors)"
python3 -m venv --system-site-packages "$WORKSPACE/sa3_spike"
"$WORKSPACE/sa3_spike/bin/pip" -q install -U pip setuptools wheel
"$WORKSPACE/sa3_spike/bin/pip" -q install stable-audio-3 \
  || "$WORKSPACE/sa3_spike/bin/pip" -q install "git+https://github.com/Stability-AI/stable-audio-3.git"

log "Downloading stable-audio-3-medium snapshot (~10.4GB, gated)"
HF_TOKEN="$HF_TOKEN" "$WORKSPACE/sa3_spike/bin/python" - <<'PY'
import os
from huggingface_hub import login, snapshot_download
login(token=os.environ["HF_TOKEN"])
print("[provision_sa3] snapshot ->", snapshot_download("stabilityai/stable-audio-3-medium",
                                                       token=os.environ["HF_TOKEN"]))
PY
log "Stable Audio 3 ready: $WORKSPACE/sa3_spike"
