#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKEND_ENV="$SCRIPT_DIR/../filmforge_backend/app/.env"

# ── Help ──────────────────────────────────────────────────────────────────────
if [[ $# -ge 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  cat >&2 <<EOF

Usage: $0 <mode> [options]

MODES
  --vast [options]
      Fully automated Vast deploy.
      Searches Vast offers, rents an instance, waits for SSH, deploys the
      gpu_worker, and optionally warms asset groups before returning the URL.

      Options (passed to deploy_gpu.py):
        --vast-gpu NAME
        --vast-max-price PRICE
        --vast-min-vram-gb GB
        --vast-disk-gb GB
        --warm-asset-group NAME
        --skip-warmup

  --runpod [options]
      Fully automated RunPod deploy.
      Finds the existing filmforge_comfy pod (or creates one), waits for SSH,
      deploys the gpu_worker, and symlinks ComfyUI model paths.

      Options (passed to deploy_gpu.py):
        --gpu "GPU_TYPE_ID"   Override GPU type (default: NVIDIA L40S)
                              Run: python3 -c "import runpod; [print(g['id']) for g in runpod.get_gpus()]"
        --worker-port PORT    Worker port (default: 9000)
        --update-backend-env  Legacy local-dev mode: write GPU_WORKER_BASE_URL to .env
        --skip-backend-restart  Don't restart local backend

  --gpus
      List all available RunPod GPU types with their IDs and display names.
      Uses RUNPOD_API_KEY from filmforge_backend/app/.env.

  --logs
      Stream gpu_worker logs in real time.
      For RunPod: resolves the current SSH host automatically via the RunPod API.
      For Vast/manual SSH: reconnects to the last host used with the ssh deploy mode.

  ssh root@<host> -p <port> -i ~/.ssh/id_ed25519 [-- options]
      Deploy to an existing SSH host (Vast.ai, RunPod manual SSH, etc.).
      Options after -- are forwarded to deploy_gpu.py.

EXAMPLES
  ./setup_gpu.sh --vast --vast-gpu "RTX 4090" --vast-max-price 0.7 --warm-asset-group flux_stills_v1
  ./setup_gpu.sh --runpod
  ./setup_gpu.sh --runpod --gpu "NVIDIA GeForce RTX 4090"
  ./setup_gpu.sh --logs
  ./setup_gpu.sh --gpus
  ./setup_gpu.sh ssh root@1.2.3.4 -p 22022 -i ~/.ssh/id_ed25519

EOF
  exit 0
fi

# ── GPUs list mode ────────────────────────────────────────────────────────────
if [[ $# -ge 1 && "$1" == "--gpus" ]]; then
  exec "$PYTHON_BIN" - "$BACKEND_ENV" <<'EOF'
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
api_key = None
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.startswith("RUNPOD_API_KEY="):
            api_key = line.split("=", 1)[1].strip()
            break

if not api_key:
    print(f"ERROR: RUNPOD_API_KEY not found in {env_path}", file=sys.stderr)
    sys.exit(1)

try:
    import runpod
except ImportError:
    print("ERROR: runpod package not installed. Run: pip install runpod", file=sys.stderr)
    sys.exit(1)

runpod.api_key = api_key
for g in sorted(runpod.get_gpus(), key=lambda g: g.get("id", "")):
    print(f"{g.get('id', '?'):<40}  {g.get('displayName', '')}")
EOF
fi

# ── RunPod fully-automated mode ───────────────────────────────────────────────
if [[ $# -ge 1 && "$1" == "--vast" ]]; then
  shift
  exec "$PYTHON_BIN" "$SCRIPT_DIR/deploy_gpu.py" --vast "$@"
fi

# ── RunPod fully-automated mode ───────────────────────────────────────────────
if [[ $# -ge 1 && "$1" == "--runpod" ]]; then
  shift
  exec "$PYTHON_BIN" "$SCRIPT_DIR/deploy_gpu.py" --runpod "$@"
fi

# ── Logs mode ─────────────────────────────────────────────────────────────────
if [[ $# -ge 1 && "$1" == "--logs" ]]; then
  LAST_SSH_FILE="$SCRIPT_DIR/.last_ssh_dest"

  # Read VAST_SSH_DEST from backend .env if present
  VAST_SSH_DEST=""
  if [[ -f "$BACKEND_ENV" ]]; then
    VAST_SSH_DEST="$(grep -E '^VAST_SSH_DEST=' "$BACKEND_ENV" | cut -d= -f2- | xargs 2>/dev/null || true)"
  fi

  if SSH_DEST="$("$PYTHON_BIN" "$SCRIPT_DIR/runpod_ssh.py" "$BACKEND_ENV" 2>/dev/null)"; then
    # RunPod — identity not embedded in SSH_DEST, add default key
    echo "→ ssh (runpod) $SSH_DEST" >&2
    exec ssh -o StrictHostKeyChecking=accept-new -i "${HOME}/.ssh/id_ed25519" $SSH_DEST "tail -f /tmp/gpu_worker.log"
  elif [[ -f "$LAST_SSH_FILE" ]]; then
    # Latest deploy — always preferred over stale .env value
    SSH_DEST="$(cat "$LAST_SSH_FILE")"
    echo "→ ssh (last deploy) $SSH_DEST" >&2
    # shellcheck disable=SC2086
    exec ssh -o StrictHostKeyChecking=accept-new -i "${HOME}/.ssh/vast_deploy" $SSH_DEST "tail -f /tmp/gpu_worker.log"
  elif [[ -n "$VAST_SSH_DEST" ]]; then
    # Vast — fallback from backend .env VAST_SSH_DEST
    echo "→ ssh (vast env) $VAST_SSH_DEST" >&2
    # shellcheck disable=SC2086
    exec ssh -o StrictHostKeyChecking=accept-new $VAST_SSH_DEST "tail -f /tmp/gpu_worker.log"
  else
    echo "ERROR: No SSH destination found." >&2
    echo "  • For RunPod: set RUNPOD_API_KEY in filmforge_backend/app/.env" >&2
    echo "  • For Vast: add VAST_SSH_DEST=root@<host> -p <port> -i ~/.ssh/vast_deploy to filmforge_backend/app/.env" >&2
    exit 1
  fi
fi

# ── SSH command mode ──────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
  echo "Usage: $0 --vast | --runpod | --gpus | --logs | ssh root@<host> -p <port> ..." >&2
  echo "Run '$0 --help' for full usage." >&2
  exit 1
fi

ssh_parts=()
deploy_args=()
parsing_ssh=true

for arg in "$@"; do
  if [[ "$arg" == "--" && "$parsing_ssh" == true ]]; then
    parsing_ssh=false
    continue
  fi

  if [[ "$parsing_ssh" == true ]]; then
    ssh_parts+=("$arg")
  else
    deploy_args+=("$arg")
  fi
done

printf -v ssh_command '%q ' "${ssh_parts[@]}"
ssh_command="${ssh_command% }"

if (( ${#deploy_args[@]} > 0 )); then
  exec "$PYTHON_BIN" "$SCRIPT_DIR/deploy_gpu.py" "$ssh_command" "${deploy_args[@]}"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/deploy_gpu.py" "$ssh_command"
