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

[ -x "$PY" ] || { echo "[infinitetalk] FATAL: no ComfyUI venv at $PY" >&2; exit 1; }

node () {  # node <dir-name> <git-url>
  local d="$COMFY/custom_nodes/$1"
  if [ -d "$d" ]; then
    echo "[infinitetalk] have node $1"
    return
  fi
  echo "[infinitetalk] cloning $1"
  git clone --depth 1 "$2" "$d"
  [ -f "$d/requirements.txt" ] && $PIP -r "$d/requirements.txt"
}

node ComfyUI-WanVideoWrapper https://github.com/kijai/ComfyUI-WanVideoWrapper
node ComfyUI-VideoHelperSuite https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

# soundfile/librosa back the wav2vec audio embedding path in the wrapper's
# talking nodes; they are not in either node's requirements.txt.
$PIP -q soundfile librosa || echo "[infinitetalk] WARN: audio deps failed" >&2

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
    echo "[infinitetalk] WARN: mask-patch site not found — wrapper changed upstream," >&2
    echo "[infinitetalk]       two-speaker renders may fail; re-diagnose before trusting A2." >&2
  fi
fi

echo "[infinitetalk] PROVISION OK — restart ComfyUI if a node was just cloned or patched"
