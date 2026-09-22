#!/usr/bin/env bash
# Provision the identity kit into ComfyUI: InfiniteYou-FLUX, PuLID-FLUX, insightface
# (ArcFace scorer + the modules' face analysis) and ultralytics (YOLOv8-seg person masks).
#
# WHY THIS EXISTS
# ---------------
# On 2026-09-21/22 this kit was the first route to hold a real family face on our own
# A100 -- installed BY HAND on a side ComfyUI, and it died with the box. Hosted doors cannot
# replace it: Seedance refused the same family photos as "likenesses of real people".
# Record: backend/docs/discoveries/endayya-grandfather-identity-benchmark-own-a100-2026-09-21.md
#
# Weights are NOT fetched here -- asset_registry's `identity_v1` (+ `wan_vace_v1`) groups
# carry them so they land on the data volume. This script installs what asset_manager
# cannot: the pinned custom nodes, the PuLID signature patch, and the Python deps.
#
# FRESH-BOX SAFE: everything comes from GitHub at pinned commits and from PyPI; nothing
# is read from a previous box. Idempotent -- safe to re-run. Does NOT restart ComfyUI.
#
# ISOLATION (Director ruling 2026-09-22): the identity department runs its OWN ComfyUI
# root with its OWN venv (copied from production, then extended). Production's
# /workspace/ComfyUI, its venv and its custom_nodes are only ever read here.
#
# LICENCES: insightface model weights (buffalo_l, antelopev2) are NON-COMMERCIAL;
# ultralytics YOLOv8 is AGPL-3.0. Fine for private family films; NOT for the product.
set -euo pipefail

# COMFY is the production ComfyUI (read-only source here); ID_ROOT is the identity
# department's SEPARATE ComfyUI (Director ruling 2026-09-22: full isolation). GPU0's
# ComfyUI never imports the identity nodes and its venv is never touched.
COMFY=${COMFY_DIR:-/workspace/ComfyUI}
ID_ROOT=${IDENTITY_COMFY_ROOT:-/workspace/ComfyUI_identity}
STAGE=${IDENTITY_STAGE:-full}   # root = clone code + copy venv only; full = root + nodes + deps
PY="$ID_ROOT/.venv/bin/python"
PIP="$PY -m pip install --break-system-packages"

INFU_REPO=https://github.com/bytedance/ComfyUI_InfiniteYou
INFU_SHA=${IDENTITY_INFU_SHA:-1c979397c5c80f5ac83a2473a2f7d4503104110f}
PULID_REPO=https://github.com/balazik/ComfyUI-PuLID-Flux
PULID_SHA=${IDENTITY_PULID_SHA:-a80912fc3435c358607bf4b43a58dbcbebdb09ff}

LOCK_FILE="${COMFY}/.filmforge_identity.provision.lock"
exec 9>"$LOCK_FILE"
flock -w "${IDENTITY_PROVISION_LOCK_TIMEOUT_SEC:-1800}" 9 || {
  echo "[identity] FATAL: timed out waiting for provision lock" >&2
  exit 1
}

[ -x "$COMFY/.venv/bin/python" ] || { echo "[identity] FATAL: no production ComfyUI venv at $COMFY/.venv" >&2; exit 1; }

# --- the separate root ----------------------------------------------------------
# The copied venv is large, so the root lives on the data volume when there is one
# (the old box's root disk had ~20 GB free), reached through the fixed path ID_ROOT.
prepare_root () {
  if [ ! -e "$ID_ROOT" ] && mountpoint -q /mnt/data 2>/dev/null; then
    mkdir -p /mnt/data/ComfyUI_identity
    ln -s /mnt/data/ComfyUI_identity "$ID_ROOT"
  fi
  mkdir -p "$ID_ROOT"
  # Same ComfyUI code as production, so graphs behave identically. The identity
  # root's own venv, nodes, models link and I/O dirs are excluded, hence protected.
  # Every exclude is ANCHORED to the root ("/..."): an unanchored 'user*' also matched
  # app/user_manager.py and the identity ComfyUI crash-looped on import (2026-09-22).
  # --BEGIN-CODE-SYNC--
  rsync -a --delete \
    --exclude /.venv --exclude /models --exclude /custom_nodes --exclude /input \
    --exclude /output --exclude /temp --exclude '/user*' --exclude '/.filmforge_*' \
    "$COMFY/" "$ID_ROOT/"
  # --END-CODE-SYNC--
  # Byte-for-byte check against production's tracked files: fail here, not in a
  # ComfyUI import at boot.
  if git -C "$COMFY" rev-parse --git-dir >/dev/null 2>&1; then
    if ! (cd "$COMFY" && git ls-files -z \
          | grep -zvE '^(custom_nodes|models|input|output|temp|user[^/]*)/' \
          | xargs -0 -I{} cmp -s "$COMFY/{}" "$ID_ROOT/{}"); then
      echo "[identity] FATAL: identity root is not a byte-for-byte copy of production's tracked files" >&2
      exit 1
    fi
  fi
  mkdir -p "$ID_ROOT/custom_nodes" "$ID_ROOT/input" "$ID_ROOT/output" "$ID_ROOT/temp"
  # FilmForge's own CUDA patch node is part of the runtime, not the identity kit.
  if [ -d "$COMFY/custom_nodes/filmforge_cuda_patch" ]; then
    rsync -a "$COMFY/custom_nodes/filmforge_cuda_patch/" "$ID_ROOT/custom_nodes/filmforge_cuda_patch/"
  fi
  # Weights live once on disk. Only nodes load weights, and GPU0 has no identity nodes.
  if [ -e "$ID_ROOT/models" ] && [ ! -L "$ID_ROOT/models" ]; then
    echo "[identity] FATAL: $ID_ROOT/models is a real directory; expected a link to $COMFY/models" >&2
    exit 1
  fi
  ln -sfn "$COMFY/models" "$ID_ROOT/models"
  if [ ! -x "$PY" ]; then
    local need_kb have_kb
    need_kb=$(du -sk "$COMFY/.venv" | cut -f1)
    have_kb=$(df -Pk "$(readlink -f "$ID_ROOT")" | awk 'NR==2 {print $4}')
    if [ "$have_kb" -lt $((need_kb + 5 * 1024 * 1024)) ]; then
      echo "[identity] FATAL: not enough disk for the identity venv copy (need $((need_kb/1024)) MB + 5 GB)" >&2
      exit 1
    fi
    echo "[identity] copying production venv into the identity root (production venv untouched)"
    cp -a "$COMFY/.venv" "$ID_ROOT/.venv"
  fi
  "$PY" -c "import torch" || { echo "[identity] FATAL: identity venv copy cannot import torch" >&2; exit 1; }
}

prepare_root
if [ "$STAGE" = "root" ]; then
  echo "[identity] root ready at $ID_ROOT (nodes install in the full stage)"
  exit 0
fi

# --- freeze the torch stack before touching anything ---------------------------
# The identity venv starts as a copy of production's, so this keeps the identity
# ComfyUI on the exact torch/numpy the production graphs were validated on.
CONSTRAINTS="$ID_ROOT/.filmforge_identity.constraints.txt"
"$PY" -m pip freeze 2>/dev/null \
  | grep -iE '^(torch|torchvision|torchaudio|numpy|protobuf|transformers|safetensors)==' \
  > "$CONSTRAINTS" || true
if ! grep -qi '^torch==' "$CONSTRAINTS"; then
  echo "[identity] FATAL: could not freeze torch from $PY -- refusing to install unguarded" >&2
  exit 1
fi
echo "[identity] pinned generation stack:"; sed 's/^/[identity]   /' "$CONSTRAINTS"
PIPC="$PIP -c $CONSTRAINTS"

node () {  # node <dir-name> <git-url> <commit>
  local d="$ID_ROOT/custom_nodes/$1"
  if [ ! -d "$d/.git" ]; then
    echo "[identity] cloning $1"
    rm -rf "$d"
    git clone --filter=blob:none "$2" "$d"
  fi
  git -C "$d" fetch --quiet origin "$3" 2>/dev/null || git -C "$d" fetch --quiet origin
  git -C "$d" checkout --quiet --force "$3"
  echo "[identity] $1 @ $(git -C "$d" rev-parse --short HEAD)"
  if [ -f "$d/requirements.txt" ]; then
    # requirements pins such as ml_dtypes==0.3.2 break under numpy 2 -- drop exact pins
    # of numpy-coupled packages and let the constraints file decide.
    # Also drop plain onnxruntime: PuLID lists it beside onnxruntime-gpu, and the two
    # conflict in one venv (we install -gpu only).
    grep -viE '^(torch|torchvision|numpy|ml_dtypes|protobuf|transformers|safetensors)\b' \
      "$d/requirements.txt" | grep -viE '^onnxruntime([<>=!~ ]|$)' \
      > "$d/.filmforge_requirements.txt" || true
    $PIPC -r "$d/.filmforge_requirements.txt" || {
      echo "[identity] FATAL: $1 requirements cannot install without moving the pinned torch stack" >&2
      exit 1
    }
  fi
}

node ComfyUI_InfiniteYou "$INFU_REPO" "$INFU_SHA"
node ComfyUI-PuLID-Flux "$PULID_REPO" "$PULID_SHA"

# insightface: ArcFace scorer + face analysis for both modules. onnxruntime-gpu for the
# detectors. ml_dtypes>=0.5 because 0.3.x fails to import under numpy 2. ultralytics for
# YOLOv8x-seg lead masks. facexlib for PuLID's face parsing.
$PIPC "ml_dtypes>=0.5" insightface onnxruntime-gpu facexlib ultralytics ftfy timm || {
  echo "[identity] FATAL: identity deps cannot install without moving the pinned torch stack" >&2
  exit 1
}

# --- PuLID forward_orig patch --------------------------------------------------
# Edits ONLY the custom node's own file (custom_nodes/ComfyUI-PuLID-Flux/pulidflux.py),
# never ComfyUI core. The node swaps its own forward_orig onto the FLUX model only when
# its Apply node runs; current ComfyUI calls forward_orig with extra keywords
# (timestep_zero_index), and the node's replacement has no **kwargs, so it raises.
# Adding **kwargs to that one signature is the whole patch. Idempotent.
"$PY" - "$ID_ROOT/custom_nodes/ComfyUI-PuLID-Flux/pulidflux.py" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
m = re.search(r"def forward_orig\((.*?)\)\s*(->[^:]*)?:", s, re.S)
if not m:
    sys.exit("[identity] FATAL: forward_orig not found in pulidflux.py -- upstream changed, re-derive the patch")
if "**kwargs" in m.group(1):
    print("[identity] PuLID forward_orig already accepts **kwargs")
else:
    args = m.group(1).rstrip().rstrip(",")
    s = s[:m.start(1)] + args + ", **kwargs" + s[m.end(1):]
    open(p, "w").write(s)
    print("[identity] patched PuLID forward_orig(**kwargs)")
PYEOF

# --- verify -------------------------------------------------------------------
# Cloning is not installing: fail HERE with the cause rather than at prompt time with
# a missing_node_type 400.
"$PY" -c "import insightface, onnxruntime, ultralytics, facexlib, ml_dtypes" || {
  echo "[identity] FATAL: identity Python deps do not import" >&2
  exit 1
}
if ! "$PY" -m pip freeze 2>/dev/null | grep -iE '^(torch|torchvision|torchaudio|numpy|protobuf|transformers|safetensors)==' \
     | diff -q - "$CONSTRAINTS" >/dev/null; then
  echo "[identity] FATAL: the torch stack moved during identity provisioning" >&2
  exit 1
fi

echo "[identity] provisioned -- restart ComfyUI for InfiniteYou / PuLID nodes to load"
