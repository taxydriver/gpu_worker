#!/bin/bash
# One-command audio-department setup for a FilmForge Verda worker box.
# Run AFTER the normal worker deploy + provision_tts.sh/provision_sa3.sh.
# Idempotent. Codifies the hand-steps from 2026-07-17/18 (see
# backend/docs/discoveries/audio-stage-skypilot-recipe-2026-07-17.md).
#
#   bash setup_audio_services.sh
#
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-/mnt/data}"
PARLER_VENV="$WORKSPACE/parler_spike"
UNIT_WORKER="/etc/systemd/system/filmforge-worker-gpu0.service"

log() { echo "[setup_audio] $*"; }

# 1) Runtime HF cache on the big volume (root-fs-full lesson §6d).
if [ -d "$WORKSPACE" ] && [ ! -L /root/.cache/huggingface ]; then
  mkdir -p /root/.cache "$WORKSPACE/hf_cache"
  rm -rf /root/.cache/huggingface
  ln -s "$WORKSPACE/hf_cache" /root/.cache/huggingface
  log "HF cache symlinked -> $WORKSPACE/hf_cache"
fi

# 2) Venv symlinks so default paths (/workspace/*) resolve wherever provisioned.
mkdir -p /workspace
for d in tts_spike parler_spike sa3_spike; do
  [ -d "$WORKSPACE/$d" ] && ln -sfn "$WORKSPACE/$d" "/workspace/$d"
done

# 3) Resident Parler voice server (Maya's voice) as systemd unit.
[ -x "$PARLER_VENV/bin/python" ] || { log "ERROR: parler venv missing — run provision_tts.sh first"; exit 1; }
"$PARLER_VENV/bin/python" -c "import soundfile" 2>/dev/null || "$PARLER_VENV/bin/pip" -q install soundfile
cat > /etc/systemd/system/filmforge-parler.service <<EOF
[Unit]
Description=FilmForge resident Indic Parler TTS server (Maya voice)
After=network.target

[Service]
Environment=HF_HOME=$WORKSPACE/hf_cache
Environment=PARLER_PORT=9101
ExecStart=$PARLER_VENV/bin/python $REPO_DIR/parler_server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable -q filmforge-parler
systemctl restart filmforge-parler
log "filmforge-parler started (model load ~60s; check curl localhost:9101/health)"

# 3b) Resident SA3 music server (warm cues; lazy-load + idle-unload).
SA3_VENV="$WORKSPACE/sa3_spike"
if [ -x "$SA3_VENV/bin/python" ]; then
  cat > /etc/systemd/system/filmforge-sa3.service <<EOF
[Unit]
Description=FilmForge resident Stable Audio 3 server (warm music cues)
After=network.target

[Service]
Environment=HF_HOME=$WORKSPACE/hf_cache
Environment=SA3_PORT=9102
WorkingDirectory=$REPO_DIR
ExecStart=$SA3_VENV/bin/python $REPO_DIR/sa3_server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable -q filmforge-sa3
  systemctl restart filmforge-sa3
  log "filmforge-sa3 started (model loads lazily on first cue)"
else
  log "sa3 venv missing — skipping music server (run provision_sa3.sh)"
fi

# 4) Advertise audio capabilities on the worker (quoted Environment line!).
if [ -f "$UNIT_WORKER" ] && ! grep -q "stable_audio3" "$UNIT_WORKER"; then
  python3 - <<PY
import re
path = "$UNIT_WORKER"
text = open(path).read()
def add(m):
    caps = m.group(1)
    for extra in ("tts_dialogue", "stable_audio3"):
        if extra not in caps:
            caps += "," + extra
    return 'Environment="WORKER_CAPABILITIES=%s"' % caps
text = re.sub(r'Environment="?WORKER_CAPABILITIES=([^\n"]*)"?', add, text)
open(path, "w").write(text)
PY
  systemctl daemon-reload
  systemctl restart filmforge-worker-gpu0
  log "worker capabilities updated + restarted"
else
  log "worker capabilities already include audio (or unit missing) — untouched"
fi

log "Done. Verify: curl -s localhost:9000/health (caps) + curl -s localhost:9101/health (voice)"
