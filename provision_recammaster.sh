#!/usr/bin/env bash
# Provision the ReCamMaster custom-node dependency into ComfyUI.
#
# Weights are NOT fetched here — asset_registry's `recammaster_v1` group carries
# them so they land on the data volume and survive a spot reclaim. This script
# installs only what asset_manager cannot: Kijai's ComfyUI-WanVideoWrapper (the
# node set the reshoot graph is built against) and VideoHelperSuite (video I/O).
#
# Mirrors gpu_worker/provision_infinitetalk.sh — same nodes, none of its audio
# deps and no two-speaker patch. Running both is safe: node() is a no-op when
# the clone already exists, and the two scripts use separate lock files.
#
# Idempotent — safe to re-run. ComfyUI must be restarted afterwards for a
# newly-cloned node to load; this script does NOT restart it (the caller owns
# that, and bouncing a serving box is never a side effect).
set -euo pipefail

COMFY=${COMFY_DIR:-/workspace/ComfyUI}
PY="$COMFY/.venv/bin/python"
PIP="$PY -m pip install --break-system-packages"

# Serialise concurrent callers (box rehydration vs the worker's asset ensure
# path) on the persistent Comfy volume, same reasoning as the infinitetalk
# provisioner.
LOCK_FILE="$COMFY/.filmforge_recammaster.provision.lock"
exec 9>"$LOCK_FILE"
flock -w "${RECAMMASTER_PROVISION_LOCK_TIMEOUT_SEC:-1800}" 9 || {
  echo "[recammaster] FATAL: timed out waiting for provision lock" >&2
  exit 1
}

[ -x "$PY" ] || { echo "[recammaster] FATAL: no ComfyUI venv at $PY" >&2; exit 1; }

node () {  # node <dir-name> <git-url>
  local d="$COMFY/custom_nodes/$1"
  if [ -d "$d" ]; then
    echo "[recammaster] have node $1"
    return
  fi
  echo "[recammaster] cloning $1"
  git clone --depth 1 "$2" "$d"
  [ -f "$d/requirements.txt" ] && $PIP -r "$d/requirements.txt"
}

node ComfyUI-WanVideoWrapper https://github.com/kijai/ComfyUI-WanVideoWrapper
node ComfyUI-VideoHelperSuite https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

echo "[recammaster] provision complete"
