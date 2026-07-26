#!/usr/bin/env bash
# Reproducibly build the SkyPilot venv the broker shells out to, WITH durable vastai-sdk fixes.
#
# Why patches are needed at all: SkyPilot 0.12.3's vendored vast provisioner was written against
# one specific shape of the vastai-sdk package, and that shape has moved TWICE independently of
# SkyPilot's own releases (SkyPilot pins only `vastai-sdk>=0.1.12`, no upper bound, so `pip
# install` always pulls whatever's newest):
#   - vastai-sdk ~0.2.x flattened `VastAI().client.api_key` to `VastAI().api_key` (no `.client`),
#     breaking SkyPilot's original `vast().client.api_key` call site.
#   - vastai-sdk 1.x is now a deprecated shim over the `vastai` package (which merged the CLI +
#     SDK); `VastAI` moved from top-level `vastai.VastAI` to `vastai.sdk.VastAI`, AND the api key
#     moved BACK under `.client.api_key` (undoing the 0.2.x flattening). Found + root-caused
#     2026-07-11 (docs/discoveries/skypilot-vastai-sdk-1.3-compat-2026-07-11.md) — a 0.2.5 install
#     imported fine but every `create_instance` call silently returned '' instead of raising,
#     because the OLD SDK shape sent request bodies the (now-current) Vast.ai backend API rejects.
#
# Given the SDK's shape has already changed direction twice, we patch defensively — resolve
# `VastAI` from either location, and read `api_key` from whichever of `.client.api_key` /
# `.api_key` actually exists at runtime — instead of hardcoding one specific version's layout.
#
# Usage:  ./setup_venv.sh /abs/path/to/skypilot-venv
set -euo pipefail

VENV="${1:?usage: setup_venv.sh <venv-dir>}"
SKYPILOT_VERSION="0.12.3.post1"   # the version validated in the spike

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install "skypilot[vast,verda,runpod]==${SKYPILOT_VERSION}"
"$VENV/bin/pip" install --upgrade vastai vastai-sdk

ADAPTOR="$("$VENV/bin/python" -c "import sky, os; print(os.path.join(os.path.dirname(sky.__file__), 'adaptors/vast.py'))")"
UTILS="$("$VENV/bin/python" -c "import sky, os; print(os.path.join(os.path.dirname(sky.__file__), 'provision/vast/utils.py'))")"

# Patch 1: resolve VastAI from either vastai.VastAI (old) or vastai.sdk.VastAI (1.x). Idempotent.
if grep -q "_vast_sdk = _vast.VastAI()" "$ADAPTOR"; then
  python3 - "$ADAPTOR" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()
old = "                import vastai_sdk as _vast  # pylint: disable=import-outside-toplevel\n                _vast_sdk = _vast.VastAI()"
new = (
    "                import vastai_sdk as _vast  # pylint: disable=import-outside-toplevel\n"
    "                _vast_cls = getattr(_vast, 'VastAI', None)\n"
    "                if _vast_cls is None:\n"
    "                    from vastai.sdk import VastAI as _vast_cls  # pylint: disable=import-outside-toplevel\n"
    "                _vast_sdk = _vast_cls()"
)
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
PYEOF
  echo "patched VastAI class resolution (top-level or .sdk) in $ADAPTOR"
else
  echo "VastAI resolution patch already applied (or call site changed) in $ADAPTOR"
fi

# Patch 2: read api_key from whichever of .client.api_key / .api_key exists at runtime. Idempotent.
if ! grep -q "_vast_v_for_key" "$UTILS"; then
  python3 - "$UTILS" <<'PYEOF'
import re, sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()
robust = '{(lambda _vast_v_for_key=vast.vast(): (getattr(_vast_v_for_key, "client", None) or _vast_v_for_key).api_key)()}'
pattern = re.compile(r'\{vast\(\)\.(?:client\.)?api_key\}')
new_content, n = pattern.subn(robust, content)
if n == 0:
    print("api_key call site not found (already patched or changed) — leaving as-is")
else:
    with open(path, "w") as f:
        f.write(new_content)
    print(f"patched {n} api_key call site(s)")
PYEOF
  echo "vastai api_key patch step done for $UTILS"
else
  echo "vastai api_key patch already applied in $UTILS"
fi

# The API server caches the old module; stop it so the next command reloads the patched code.
"$VENV/bin/sky" api stop >/dev/null 2>&1 || true

echo "SkyPilot venv ready at $VENV"
echo "Verify with: $VENV/bin/sky check"
