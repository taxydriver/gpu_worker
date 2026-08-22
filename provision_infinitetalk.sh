#!/usr/bin/env bash
# Provision the InfiniteTalk / MultiTalk custom node into ComfyUI.
#
# Weights are NOT fetched here — asset_registry's `infinitetalk_v1` group carries
# them so they land on the data volume and survive a spot reclaim. This script
# installs only what asset_manager cannot: Kijai's ComfyUI-WanVideoWrapper (the
# node set the talking graphs are built against) and its Python deps.
#
# Derived from the spike bootstrap that produced the A1/A2 passes:
# Filmforge/backend/spikes/audio_dialogue/a3_box_bootstrap.sh
#
# Idempotent — safe to re-run. ComfyUI must be restarted afterwards for a
# newly-cloned node to load; this script does NOT restart it (the caller owns
# that, and bouncing a serving box is never a side effect).
set -euo pipefail

COMFY=${COMFY_DIR:-/workspace/ComfyUI}
PY="$COMFY/.venv/bin/python"
PIP="$PY -m pip install --break-system-packages"
WAN_WRAPPER_COMMIT=088128b224242e110d3906c6750e9a3a348a659b

# This script is invoked both during box rehydration and from the worker's
# asset ensure path.  Serialise those callers on the persistent Comfy volume:
# without this lock a fresh box can clone/update custom_nodes concurrently.
LOCK_FILE="$COMFY/.filmforge_infinitetalk.provision.lock"
exec 9>"$LOCK_FILE"
flock -w "${INFINITETALK_PROVISION_LOCK_TIMEOUT_SEC:-1800}" 9 || {
  echo "[infinitetalk] FATAL: timed out waiting for provision lock" >&2
  exit 1
}

[ -x "$PY" ] || { echo "[infinitetalk] FATAL: no ComfyUI venv at $PY" >&2; exit 1; }

node () {  # node <dir-name> <git-url> [approved-commit]
  local d="$COMFY/custom_nodes/$1"
  local approved_commit="${3:-}"
  if [ ! -d "$d" ]; then
    echo "[infinitetalk] cloning $1"
    git clone --depth 1 "$2" "$d"
  else
    echo "[infinitetalk] have node $1"
  fi
  if [ -n "$approved_commit" ]; then
    local observed_commit
    observed_commit=$(git -C "$d" rev-parse HEAD)
    if [ "$observed_commit" != "$approved_commit" ]; then
      if ! git -C "$d" diff --quiet || ! git -C "$d" diff --cached --quiet; then
        echo "[infinitetalk] FATAL: $1 is modified at unapproved commit $observed_commit" >&2
        exit 1
      fi
      git -C "$d" fetch --depth 1 origin "$approved_commit"
      git -C "$d" checkout --detach "$approved_commit"
    fi
    [ "$(git -C "$d" rev-parse HEAD)" = "$approved_commit" ] || {
      echo "[infinitetalk] FATAL: $1 did not reach approved commit $approved_commit" >&2
      exit 1
    }
  fi
  [ -f "$d/requirements.txt" ] && $PIP -r "$d/requirements.txt"
}

node ComfyUI-WanVideoWrapper https://github.com/kijai/ComfyUI-WanVideoWrapper "$WAN_WRAPPER_COMMIT"
node ComfyUI-VideoHelperSuite https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

# These back the wav2vec audio embedding path in the wrapper's talking nodes.
# A partial install must fail provisioning; readiness otherwise withholds the
# capability, but a successful provision must mean its dependency set exists.
$PIP -q soundfile librosa transformers

# --- two-speaker closure-staleness patch --------------------------------------
# Any human_num==2 render (MultiTalk two-shot) dies at the first sampling step
# with `AttributeError: 'NoneType' object has no attribute 'max'` at
# multitalk/multitalk.py:343.
#
# Cause: multitalk_loop.py autogenerates `ref_target_masks` for the
# no-masks-supplied 2-speaker case and stores them INTO the `multitalk_embeds`
# dict, but predict_with_cfg closes over process()'s local `ref_target_masks`
# bound before that autogen ran — so base_params still receives None,
# x_ref_attn_map is never computed, and audio cross-attn dereferences it.
#
# Reading the mutated dict at call time restores upstream's own intent and
# changes no graph semantics. Diagnosed 2026-07-28
# (docs/discoveries/a3-assembled-conversation-2026-07-28.md), where it was fixed
# ON THE BOX ONLY and correctly predicted to "evaporate with the volume" — it
# did, and A2 failed again on 2026-08-01. It lives here now so it cannot.
#
# Still the right long-term move (per that discovery): supply ref_target_masks
# explicitly from our own authored blocking rather than trust the auto path.
SAMPLER="$COMFY/custom_nodes/ComfyUI-WanVideoWrapper/nodes_sampler.py"
if [ -f "$SAMPLER" ]; then
  if grep -q 'ref_target_masks": (multitalk_embeds or {})' "$SAMPLER"; then
    echo "[infinitetalk] have two-speaker mask patch"
  elif grep -q '"ref_target_masks": ref_target_masks if multitalk_audio_embeds' "$SAMPLER"; then
    cp -n "$SAMPLER" "$SAMPLER.pre_infinitetalk_patch"
    python3 - "$SAMPLER" <<'PATCH'
import sys
path = sys.argv[1]
src = open(path).read()
old = '"ref_target_masks": ref_target_masks if multitalk_audio_embeds is not None else None,'
new = ('"ref_target_masks": (multitalk_embeds or {}).get("ref_target_masks", ref_target_masks) '
       'if multitalk_audio_embeds is not None else None,')
assert src.count(old) == 1, f"expected exactly 1 site, found {src.count(old)}"
open(path, "w").write(src.replace(old, new))
print("[infinitetalk] applied two-speaker mask patch")
PATCH
  else
    echo "[infinitetalk] FATAL: mask-patch site not found in approved wrapper" >&2
    exit 1
  fi
else
  echo "[infinitetalk] FATAL: approved wrapper sampler is missing" >&2
  exit 1
fi

PATCHED='"ref_target_masks": (multitalk_embeds or {}).get("ref_target_masks", ref_target_masks)'
[ "$(grep -F -c "$PATCHED" "$SAMPLER")" -eq 1 ] || {
  echo "[infinitetalk] FATAL: two-speaker mask patch was not positively verified" >&2
  exit 1
}

echo "[infinitetalk] PROVISION OK — restart ComfyUI if a node was just cloned or patched"
