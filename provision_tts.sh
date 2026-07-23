#!/usr/bin/env bash
# Provision open-source dialogue TTS on a GPU worker box.
#
# Companion to asset_registry.py: that registry downloads single-file ComfyUI
# models; TTS models are whole HF repos loaded by pip packages in ISOLATED venvs
# (their transformers pins conflict), so they get their own provisioner.
#
# Models (validated 2026-07-15, see backend/docs/discoveries/oss-dialogue-tts-validation-2026-07-15.md):
#   - Chatterbox Multilingual V3 (Resemble AI, MIT)  -> EN + Hindi, clones a reference clip
#   - Indic Parler-TTS (AI4Bharat, Apache-2.0)        -> Telugu + Tamil + Hindi, prompt-described
#
# Idempotent: safe to re-run. Existing venvs/cached snapshots are reused.
#
# Usage:
#   HF_TOKEN=hf_xxx bash gpu_worker/provision_tts.sh            # both models
#   bash gpu_worker/provision_tts.sh chatterbox                 # Chatterbox only (no token needed)
#   HF_TOKEN=hf_xxx bash gpu_worker/provision_tts.sh parler     # Parler only (token REQUIRED — gated repo)
#
# HF_TOKEN is required for Parler only (its repo is gated; still Apache/commercial,
# the gate is a one-click terms accept). Chatterbox is tokenless.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
WHICH="${1:-all}"
export HF_XET_HIGH_PERFORMANCE=1
# Keep model snapshots OFF the small root fs — use the big /mnt/data volume when present.
if [ -z "${HF_HOME:-}" ] && [ -d /mnt/data ]; then
  export HF_HOME=/mnt/data/hf_cache
  mkdir -p "$HF_HOME"
fi

log() { echo "[provision_tts] $*" >&2; }

provision_chatterbox() {
  log "Chatterbox: venv + pip + snapshot (tokenless)"
  python3 -m venv --system-site-packages "$WORKSPACE/tts_spike"
  "$WORKSPACE/tts_spike/bin/pip" -q install --upgrade pip
  "$WORKSPACE/tts_spike/bin/pip" -q install chatterbox-tts
  "$WORKSPACE/tts_spike/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("ResembleAI/chatterbox")
print("[provision_tts] chatterbox snapshot ->", p)
PY
  log "Chatterbox ready: $WORKSPACE/tts_spike"
}

provision_parler() {
  if [ -z "${HF_TOKEN:-}" ]; then
    log "ERROR: Parler repo is gated — set HF_TOKEN (accept terms at hf.co/ai4bharat/indic-parler-tts first)."
    return 1
  fi
  log "Parler: venv + pip (torch FIRST) + snapshot (gated, using HF_TOKEN)"
  python3 -m venv --system-site-packages "$WORKSPACE/parler_spike"
  "$WORKSPACE/parler_spike/bin/pip" -q install --upgrade pip
  # torch first: on a bare box, letting the parler-tts git install resolve torch
  # sends pip's resolver into silent hours-long backtracking (observed 2026-07-23/24,
  # same failure as provision_sa3.sh — see discovery sa3-barebox-install-stall).
  # Satisfying the heaviest dep up front makes the git install resolve instantly.
  "$WORKSPACE/parler_spike/bin/pip" install torch torchaudio
  "$WORKSPACE/parler_spike/bin/pip" install "git+https://github.com/huggingface/parler-tts.git" soundfile
  HF_TOKEN="$HF_TOKEN" "$WORKSPACE/parler_spike/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
# Indic Parler weights (gated) + its description text-encoder (flan-t5-large, ungated).
print("[provision_tts] parler snapshot ->", snapshot_download("ai4bharat/indic-parler-tts", token=tok))
print("[provision_tts] flan-t5-large  ->", snapshot_download("google/flan-t5-large"))
PY
  log "Parler ready: $WORKSPACE/parler_spike"
}

case "$WHICH" in
  chatterbox) provision_chatterbox ;;
  parler)     provision_parler ;;
  all)        provision_chatterbox; provision_parler ;;
  *) log "Unknown target: $WHICH (use: chatterbox | parler | all)"; exit 2 ;;
esac
log "Done ($WHICH)."
