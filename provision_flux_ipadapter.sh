#!/usr/bin/env bash
# Provision the XLabs Flux IPAdapter custom node into ComfyUI.
#
# WHY THIS EXISTS
# ---------------
# Flux2 can condition on a reference image two ways, and they are not
# interchangeable:
#
#   ReferenceLatent  (flux2_ref)       — conditions on COMPOSITION and style.
#                                        Takes the wardrobe, the palette, the
#                                        kind of person, then draws a face from
#                                        the model's own priors.
#   IPAdapter        (flux2_ipadapter) — conditions on the FACE.
#
# Our own May evaluation (backend `scripts/eval_flux_ipadapter.py`, outputs in
# `outputs/ipadapter_eval/`) ran both paths against the same reference, prompt
# and seed: IPAdapter returned the same person every time, ReferenceLatent
# returned a different one. That eval ran on a box where these nodes had been
# installed BY HAND — they have never been part of worker provisioning, which
# is why every production render silently took the ReferenceLatent path and
# why a family's own photograph produced a plausible stranger.
#
# Weights are NOT fetched here — asset_registry's `flux_ipadapter_v1` group
# carries them so they land on the data volume and survive a spot reclaim.
# This script installs only what asset_manager cannot: the custom node set the
# ipadapter graph is built against, plus its Python deps.
#
# Idempotent — safe to re-run. ComfyUI must be restarted afterwards for a
# newly-cloned node to load; this script does NOT restart it (the caller owns
# that, and bouncing a serving box is never a side effect).
set -euo pipefail

COMFY=${COMFY_DIR:-/workspace/ComfyUI}
PY="$COMFY/.venv/bin/python"
PIP="$PY -m pip install --break-system-packages"

# Serialise with the other provisioners on the persistent Comfy volume: without
# this lock a fresh box can clone/update custom_nodes concurrently.
LOCK_FILE="$COMFY/.filmforge_flux_ipadapter.provision.lock"
exec 9>"$LOCK_FILE"
flock -w "${FLUX_IPADAPTER_PROVISION_LOCK_TIMEOUT_SEC:-1800}" 9 || {
  echo "[flux_ipadapter] FATAL: timed out waiting for provision lock" >&2
  exit 1
}

[ -x "$PY" ] || { echo "[flux_ipadapter] FATAL: no ComfyUI venv at $PY" >&2; exit 1; }

node () {  # node <dir-name> <git-url>
  local d="$COMFY/custom_nodes/$1"
  if [ -d "$d" ]; then
    echo "[flux_ipadapter] have node $1"
    return
  fi
  echo "[flux_ipadapter] cloning $1"
  git clone --depth 1 "$2" "$d"
  [ -f "$d/requirements.txt" ] && $PIP -r "$d/requirements.txt"
}

# Provides LoadFluxIPAdapter + ApplyFluxIPAdapter. The node's display name
# carries an upstream typo ("ipadatper") — that is upstream's, not ours, and
# the class_type we build against is LoadFluxIPAdapter.
node x-flux-comfyui https://github.com/XLabs-AI/x-flux-comfyui

# The node imports these directly; a partial install must fail provisioning,
# because readiness otherwise advertises a capability whose first render 400s
# with `missing_node_type` — which is exactly how this was found.
$PIP -q transformers safetensors

# --- verify the node actually registered -------------------------------------
# Cloning is not installing: a node whose deps failed still leaves the directory
# behind, and ComfyUI then rejects the graph at prompt time with a 400 that the
# broker reports as a generic worker failure. Fail HERE instead, where the
# message names the cause.
if [ ! -f "$COMFY/custom_nodes/x-flux-comfyui/nodes.py" ]; then
  echo "[flux_ipadapter] FATAL: x-flux-comfyui cloned but nodes.py missing" >&2
  exit 1
fi

echo "[flux_ipadapter] provisioned — restart ComfyUI for LoadFluxIPAdapter to load"
