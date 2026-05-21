#!/usr/bin/env python3
"""Deploy the Filmforge GPU worker to SSH, RunPod, or a freshly rented Vast box."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REMOTE_ROOT = "/workspace/filmforge_gpu_worker"
DEFAULT_BACKEND_ENV = SCRIPT_DIR.parent / "filmforge_backend" / "app" / ".env"
DEFAULT_BACKEND_ROOT = SCRIPT_DIR.parent / "filmforge_backend"
WORKER_URL_PATTERN = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")
DEFAULT_SSH_IDENTITY = Path.home() / ".ssh" / "vast_deploy"
DEFAULT_SSH_IDENTITY_RUNPOD = Path.home() / ".ssh" / "runpod_deploy"
DEFAULT_VAST_IMAGE = "vastai/comfy:v0.15.1-cuda-12.9-py312"
DEFAULT_VAST_GPU = "RTX 4090"
DEFAULT_VAST_MAX_PRICE = 0.75
DEFAULT_VAST_MIN_VRAM_GB = 24
DEFAULT_VAST_LIMIT = 25
DEFAULT_VAST_BOOT_TIMEOUT = 900
DEFAULT_VERDA_CLI = Path.home() / ".verda" / "bin" / "verda"
DEFAULT_VERDA_LOCATION = "FIN-01"
DEFAULT_VERDA_INSTANCE_TYPE = "2A100.44V"
DEFAULT_VERDA_OS_IMAGE = "ubuntu-24.04-cuda-12.8-open-docker"
DEFAULT_VERDA_OS_VOLUME_ID = "34ec939d-a8c1-4ee2-9637-533e324dfe39"
DEFAULT_VERDA_DATA_VOLUME_ID = "4ea18b04-564f-4218-ab79-e90d1ccc839b"
DEFAULT_VERDA_SSH_KEY_ID = "11ee08a4-858a-4ee7-98c8-250aad99eb37"
DEFAULT_VERDA_HOSTNAME = "filmforge-verda-worker"
DEFAULT_VERDA_FRESH_OS_VOLUME_SIZE = 100
DEFAULT_VERDA_FRESH_STORAGE_SIZE = 250
DEFAULT_VERDA_CONTRACT = "pay_as_go"
DEFAULT_WORKER_REPO_URL = "https://github.com/taxydriver/gpu_worker.git"
DEFAULT_COMFY_REPO_URL = "https://github.com/comfyanonymous/ComfyUI.git"
DEFAULT_PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
DEFAULT_VERDA_FRESH_WARM_GROUPS = ["flux_stills_v1", "wan_i2v_v1", "stable_audio_v1"]

# Ordered list of candidate identity files to try when none is specified
_CANDIDATE_IDENTITIES = [
    DEFAULT_SSH_IDENTITY,
    DEFAULT_SSH_IDENTITY_RUNPOD,
    Path.home() / ".ssh" / "runpod",  # common alternative name
    Path.home() / ".ssh" / "id_ed25519",  # default key used when provisioning RunPod
    Path.home() / ".ssh" / "id_rsa",
]


def log(message: str) -> None:
    print(message, file=sys.stderr)


def run(
    cmd: list[str],
    *,
    input_text: str | None = None,
    cwd: Path | None = None,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    log(f"+ {shlex.join(cmd)}")
    return subprocess.run(
        cmd,
        input=input_text,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=capture_output,
        check=check,
    )


def parse_ssh_command(ssh_command: str) -> tuple[list[str], list[str], str]:
    tokens = shlex.split(ssh_command)
    if not tokens or tokens[0] != "ssh":
        raise ValueError("SSH command must start with `ssh`.")

    ssh_opts: list[str] = []
    scp_opts: list[str] = []
    destination: str | None = None

    shared_opt_args = {"-F", "-i", "-J", "-l", "-o", "-S"}
    ignore_opt_args = {"-L", "-R", "-D", "-W", "-b", "-c", "-E", "-e", "-I", "-m", "-Q"}
    passthrough_flags = {
        "-4",
        "-6",
        "-A",
        "-a",
        "-C",
        "-K",
        "-k",
        "-q",
        "-T",
        "-v",
        "-vv",
        "-vvv",
        "-X",
        "-x",
        "-Y",
        "-y",
    }

    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token in shared_opt_args:
            if i + 1 >= len(tokens):
                raise ValueError(f"Missing value for SSH option {token}.")
            value = os.path.expanduser(tokens[i + 1])
            ssh_opts.extend([token, value])
            scp_opts.extend([token, value])
            i += 2
            continue

        if token == "-p":
            if i + 1 >= len(tokens):
                raise ValueError("Missing value for SSH option -p.")
            value = tokens[i + 1]
            ssh_opts.extend(["-p", value])
            scp_opts.extend(["-P", value])
            i += 2
            continue

        if token.startswith("-p") and token != "-p":
            value = token[2:]
            if not value:
                raise ValueError("Missing value for SSH option -p.")
            ssh_opts.extend(["-p", value])
            scp_opts.extend(["-P", value])
            i += 1
            continue

        if token in ignore_opt_args:
            if i + 1 >= len(tokens):
                raise ValueError(f"Missing value for SSH option {token}.")
            i += 2
            continue

        if token in {"-N", "-f", "-n"}:
            i += 1
            continue

        if token in passthrough_flags:
            ssh_opts.append(token)
            scp_opts.append(token)
            i += 1
            continue

        if token.startswith("-o") and token != "-o":
            ssh_opts.append(token)
            scp_opts.append(token)
            i += 1
            continue

        if token.startswith("-i") and token != "-i":
            value = os.path.expanduser(token[2:])
            ssh_opts.extend(["-i", value])
            scp_opts.extend(["-i", value])
            i += 1
            continue

        if token.startswith("-F") and token != "-F":
            value = os.path.expanduser(token[2:])
            ssh_opts.extend(["-F", value])
            scp_opts.extend(["-F", value])
            i += 1
            continue

        if token.startswith("-J") and token != "-J":
            value = token[2:]
            ssh_opts.extend(["-J", value])
            scp_opts.extend(["-J", value])
            i += 1
            continue

        if token.startswith("-"):
            raise ValueError(f"Unsupported SSH option in command: {token}")

        destination = token
        i += 1

    if not destination:
        raise ValueError("Could not find `user@host` in SSH command.")

    return ["ssh", *ssh_opts, destination], ["scp", *scp_opts], destination


def has_ssh_option(cmd: list[str], option_name: str) -> bool:
    i = 0
    while i < len(cmd):
        token = cmd[i]
        if token == "-o" and i + 1 < len(cmd):
            opt = cmd[i + 1]
            if opt == option_name or opt.startswith(f"{option_name}="):
                return True
            i += 2
            continue
        if token.startswith("-o") and token != "-o":
            opt = token[2:]
            if opt == option_name or opt.startswith(f"{option_name}="):
                return True
        i += 1
    return False


def has_identity_config(cmd: list[str]) -> bool:
    if "-i" in cmd:
        return True
    return has_ssh_option(cmd, "IdentityFile")


def add_default_host_key_policy(cmd: list[str]) -> list[str]:
    if has_ssh_option(cmd, "StrictHostKeyChecking"):
        return cmd
    return [cmd[0], "-o", "StrictHostKeyChecking=accept-new", *cmd[1:]]


def add_default_identity(cmd: list[str], override: Path | None = None) -> list[str]:
    if has_identity_config(cmd):
        return cmd
    # Use an explicitly provided identity (via --ssh-identity) if given
    if override is not None:
        return [cmd[0], "-i", str(override), *cmd[1:]]
    # Otherwise pass every existing candidate so SSH can try each one
    identity_args: list[str] = []
    for candidate in _CANDIDATE_IDENTITIES:
        if candidate.exists():
            identity_args.extend(["-i", str(candidate)])
    if not identity_args:
        return cmd
    return [cmd[0], *identity_args, *cmd[1:]]


def stage_worker_tree(source_dir: Path) -> tempfile.TemporaryDirectory[str]:
    temp_dir = tempfile.TemporaryDirectory(prefix="gpu_worker_deploy_")
    staged_root = Path(temp_dir.name) / source_dir.name
    shutil.copytree(
        source_dir,
        staged_root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    return temp_dir


def build_env_exports(env_items: list[str]) -> str:
    exports: list[str] = []
    for item in env_items:
        key, sep, value = str(item).partition("=")
        key = key.strip()
        if not sep or not key:
            continue
        exports.append(f"export {key}={shlex.quote(value)}")
    return "\n".join(exports)


def remote_script(remote_root: str, worker_port: int) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT={shlex.quote(remote_root)}
WORKER_PORT={worker_port}
DEFAULT_COMFY_BASE_URL="${{COMFYUI_API_BASE:-http://127.0.0.1:18188}}"

detect_comfy_base() {{
  if test -n "${{COMFYUI_API_BASE:-}}" && curl -fsS "${{COMFYUI_API_BASE%/}}/system_stats" >/dev/null 2>&1; then
    echo "${{COMFYUI_API_BASE%/}}"
    return
  fi
  if curl -fsS http://127.0.0.1:18188/system_stats >/dev/null 2>&1; then
    echo "http://127.0.0.1:18188"
    return
  fi
  if curl -fsS http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
    echo "http://127.0.0.1:8188"
    return
  fi
  return 1
}}

# Wait for ComfyUI to respond before probing its port (it may still be starting).
# Vast's ComfyUI template exposes COMFYUI_API_BASE=http://localhost:18188; using
# 8188 as an early fallback makes the worker boot "healthy" against the wrong
# port and then repeatedly restart ComfyUI.
echo "Waiting for ComfyUI to become reachable..." >&2
for _ in $(seq 1 60); do
  if detect_comfy_base >/dev/null 2>&1; then
    break
  fi
  sleep 3
done

COMFY_BASE_URL="$(detect_comfy_base || true)"
if test -z "$COMFY_BASE_URL"; then
  COMFY_BASE_URL="$DEFAULT_COMFY_BASE_URL"
  echo "ComfyUI probe did not respond yet; using expected base $COMFY_BASE_URL" >&2
fi
COMFY_OUTPUT_DIR="/workspace/ComfyUI/output"
COMFY_TEMP_DIR="/workspace/ComfyUI/temp"
COMFY_INPUT_DIR="/workspace/ComfyUI/input"

# ── RunPod path fix ───────────────────────────────────────────────────────────
# On runpod/comfyui:latest, ComfyUI lives in /workspace/runpod-slim/ComfyUI.
# We symlink its output/temp/input dirs to /workspace/ComfyUI/* (what the
# worker expects) and symlink model files into ComfyUI's native model dirs.
RUNPOD_COMFY="/workspace/runpod-slim/ComfyUI"
if test -f "$RUNPOD_COMFY/main.py"; then
  echo "RunPod ComfyUI detected at $RUNPOD_COMFY — applying path fixes" >&2
  COMFY_BASE_URL="http://127.0.0.1:8188"

  # Output / temp / input: ComfyUI writes here; worker reads from /workspace/ComfyUI/*
  mkdir -p /workspace/ComfyUI/output /workspace/ComfyUI/temp /workspace/ComfyUI/input
  for dir in output temp input; do
    if test ! -L "$RUNPOD_COMFY/$dir"; then
      rm -rf "$RUNPOD_COMFY/$dir"
      ln -sf "/workspace/ComfyUI/$dir" "$RUNPOD_COMFY/$dir"
    fi
  done

  # Write extra_model_paths.yaml so ComfyUI scans /workspace/ComfyUI/models/ directly.
  # This means newly downloaded models are found after any ComfyUI restart — no per-file symlinks needed.
  cat > "$RUNPOD_COMFY/extra_model_paths.yaml" <<'YAML'
filmforge_worker:
    base_path: /workspace/ComfyUI/models
    checkpoints: checkpoints
    clip: clip
    vae: vae
    diffusion_models: diffusion_models
    text_encoders: text_encoders
    loras: loras
    unet: unet
    controlnet: controlnet
    upscale_models: upscale_models
YAML
  echo "Wrote extra_model_paths.yaml → ComfyUI will scan /workspace/ComfyUI/models/ directly" >&2

  # Restart ComfyUI so it rescans model dirs with the new extra_model_paths in place
  echo "Restarting ComfyUI to pick up new model paths..." >&2
  pkill -f "main.py.*--listen" 2>/dev/null || true
  sleep 2
  cd "$RUNPOD_COMFY"
  source .venv-cu128/bin/activate
  nohup python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header >> /tmp/comfyui.log 2>&1 &
  echo "ComfyUI restarted (PID $!), waiting for ready..." >&2
  for _ in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
      echo "ComfyUI ready" >&2
      break
    fi
    sleep 3
  done
fi

mkdir -p "$REMOTE_ROOT"
cd "$REMOTE_ROOT"
python3 -m venv .venv
.venv/bin/pip install -r gpu_worker/requirements.txt

# Install aria2c for fast parallel model downloads (16-connection vs single-stream)
if ! command -v aria2c >/dev/null 2>&1; then
  echo "Installing aria2c..." >&2
  apt-get install -y -q aria2 2>/dev/null || true
fi

COMFY_STOP_CMD=""
COMFY_START_CMD=""
if supervisorctl status comfyui >/dev/null 2>&1; then
  # Patch supervisor config to write ComfyUI logs to /tmp/comfyui.log (default is /dev/stdout)
  COMFY_CONF="$(find /etc/supervisor -name '*.conf' -exec grep -l 'program:comfyui' {{}} \\; 2>/dev/null | head -1 || true)"
  if test -n "$COMFY_CONF" && grep -q 'stdout_logfile=/dev/stdout' "$COMFY_CONF" 2>/dev/null; then
    echo "Redirecting ComfyUI supervisor logs to /tmp/comfyui.log..." >&2
    sed -i 's|stdout_logfile=/dev/stdout|stdout_logfile=/tmp/comfyui.log|' "$COMFY_CONF"
    sed -i 's|stdout_logfile_maxbytes=0|stdout_logfile_maxbytes=20MB|' "$COMFY_CONF"
    supervisorctl reread >/dev/null 2>&1 || true
    supervisorctl update comfyui >/dev/null 2>&1 || true
    supervisorctl restart comfyui >/dev/null 2>&1 || true
    echo "Waiting for ComfyUI to restart..." >&2
    for _ in $(seq 1 60); do
      if curl -sf "${{COMFY_BASE_URL%/}}/system_stats" > /dev/null 2>&1; then
        echo "ComfyUI ready" >&2
        break
      fi
      sleep 3
    done
  fi
  COMFY_STOP_CMD="supervisorctl stop comfyui"
  COMFY_START_CMD="supervisorctl start comfyui"
elif test -f "/workspace/runpod-slim/ComfyUI/main.py"; then
  COMFY_STOP_CMD="pkill -f 'main.py.*--listen' 2>/dev/null || true"
  COMFY_START_CMD="cd /workspace/runpod-slim/ComfyUI && source .venv-cu128/bin/activate && nohup python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header >> /tmp/comfyui.log 2>&1 & sleep 15"
fi

pkill -f "uvicorn gpu_worker.app:app" || true

# Build env vars, only including non-empty ones (to avoid Pydantic validation errors on empty floats)
declare -a ENV_VARS
ENV_VARS+=(COMFY_BASE_URL="$COMFY_BASE_URL")
ENV_VARS+=(COMFY_OUTPUT_DIR="$COMFY_OUTPUT_DIR")
ENV_VARS+=(COMFY_TEMP_DIR="$COMFY_TEMP_DIR")
ENV_VARS+=(COMFY_INPUT_DIR="$COMFY_INPUT_DIR")
ENV_VARS+=(COMFY_STOP_CMD="$COMFY_STOP_CMD")
ENV_VARS+=(COMFY_START_CMD="$COMFY_START_CMD")
ENV_VARS+=(WORKER_PROVIDER="${{WORKER_PROVIDER:-dedicated_worker}}")
ENV_VARS+=(WORKER_MAX_CONCURRENT_JOBS="${{WORKER_MAX_CONCURRENT_JOBS:-1}}")
ENV_VARS+=(WORKER_HEARTBEAT_SECONDS="${{WORKER_HEARTBEAT_SECONDS:-60}}")

# Only include these if they're not empty
test -n "${{RENDER_BROKER_BASE_URL:-}}" && ENV_VARS+=(RENDER_BROKER_BASE_URL="${{RENDER_BROKER_BASE_URL}}")
test -n "${{RENDER_BROKER_WORKER_TOKEN:-}}" && ENV_VARS+=(RENDER_BROKER_WORKER_TOKEN="${{RENDER_BROKER_WORKER_TOKEN}}")
test -n "${{RENDER_BROKER_WORKER_ID:-}}" && ENV_VARS+=(RENDER_BROKER_WORKER_ID="${{RENDER_BROKER_WORKER_ID}}")
test -n "${{RENDER_BROKER_WORKER_NAME:-}}" && ENV_VARS+=(RENDER_BROKER_WORKER_NAME="${{RENDER_BROKER_WORKER_NAME}}")
test -n "${{RENDER_BROKER_WORKER_PUBLIC_URL:-}}" && ENV_VARS+=(RENDER_BROKER_WORKER_PUBLIC_URL="${{RENDER_BROKER_WORKER_PUBLIC_URL}}")
test -n "${{FILMFORGE_BACKEND_URL:-}}" && ENV_VARS+=(FILMFORGE_BACKEND_URL="${{FILMFORGE_BACKEND_URL}}")
test -n "${{WORKER_NAME:-}}" && ENV_VARS+=(WORKER_NAME="${{WORKER_NAME}}")
test -n "${{WORKER_PUBLIC_URL:-}}" && ENV_VARS+=(WORKER_PUBLIC_URL="${{WORKER_PUBLIC_URL}}")
test -n "${{WORKER_GPU_NAME:-}}" && ENV_VARS+=(WORKER_GPU_NAME="${{WORKER_GPU_NAME}}")
test -n "${{WORKER_VRAM_GB:-}}" && ENV_VARS+=(WORKER_VRAM_GB="${{WORKER_VRAM_GB}}")
ENV_VARS+=(WORKER_CAPABILITIES="${{WORKER_CAPABILITIES:-flux2_stills,wan_i2v,stable_audio1}}")
test -n "${{WORKER_REGISTRATION_TOKEN:-}}" && ENV_VARS+=(WORKER_REGISTRATION_TOKEN="${{WORKER_REGISTRATION_TOKEN}}")
test -n "${{WORKER_API_TOKEN:-}}" && ENV_VARS+=(WORKER_API_TOKEN="${{WORKER_API_TOKEN}}")

nohup env "${{ENV_VARS[@]}}" \\
  .venv/bin/python -m uvicorn gpu_worker.app:app --host 0.0.0.0 --port "$WORKER_PORT" \\
  >/tmp/gpu_worker.log 2>&1 </dev/null &

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$WORKER_PORT/health" >/tmp/gpu_worker_health.json 2>/dev/null; then
    break
  fi
  sleep 1
done

if ! test -s /tmp/gpu_worker_health.json; then
  echo "Worker failed to become healthy" >&2
  tail -n 100 /tmp/gpu_worker.log >&2 || true
  exit 1
fi

WORKER_URL=""
PRESET_PUBLIC_URL="${{WORKER_PUBLIC_URL:-}}"
if test -n "$PRESET_PUBLIC_URL"; then
  WORKER_URL="$PRESET_PUBLIC_URL"
  echo "Using preset WORKER_PUBLIC_URL=$WORKER_URL (skipping cloudflared)" >&2
elif test -x /opt/instance-tools/bin/cloudflared; then
  for _tunnel_attempt in 1 2 3; do
    pkill -f "cloudflared tunnel --url http://127.0.0.1:$WORKER_PORT" || true
    sleep 2
    >/tmp/filmforge_gpu_worker_tunnel.log
    nohup /opt/instance-tools/bin/cloudflared tunnel --url "http://127.0.0.1:$WORKER_PORT" --no-autoupdate \\
      >/tmp/filmforge_gpu_worker_tunnel.log 2>&1 </dev/null &
    for _ in $(seq 1 30); do
      if grep -aEo 'https://[-a-z0-9]+\\.trycloudflare\\.com' /tmp/filmforge_gpu_worker_tunnel.log | tail -n 1 >/tmp/filmforge_gpu_worker_tunnel_url.txt 2>/dev/null; then
        WORKER_URL="$(tr -d '\\000' </tmp/filmforge_gpu_worker_tunnel_url.txt)"
        break
      fi
      sleep 1
    done
    if test -n "$WORKER_URL"; then
      break
    fi
    echo "Cloudflare tunnel attempt $_tunnel_attempt failed, retrying in 5s..." >&2
    tail -n 10 /tmp/filmforge_gpu_worker_tunnel.log >&2 || true
    sleep 5
  done
  if ! test -n "$WORKER_URL"; then
    echo "Cloudflare tunnel did not return a public URL after 3 attempts." >&2
    echo "--- /tmp/filmforge_gpu_worker_tunnel.log ---" >&2
    tail -n 80 /tmp/filmforge_gpu_worker_tunnel.log >&2 || true
  fi
else
  echo "cloudflared not found at /opt/instance-tools/bin/cloudflared; no public tunnel created." >&2
fi

# ── RunPod proxy fallback (when no cloudflared) ─────────────────────────────
if test -z "$WORKER_URL" && test -n "${{RUNPOD_POD_ID:-}}"; then
  WORKER_URL="https://${{RUNPOD_POD_ID}}-${{WORKER_PORT}}.proxy.runpod.net"
  echo "Using RunPod proxy URL: $WORKER_URL" >&2
fi

# ── Restart worker with public URL injected (so it can self-register) ─────────
# Skip restart when WORKER_PUBLIC_URL was preset — first start already had it.
if test -n "$WORKER_URL" && test -z "$PRESET_PUBLIC_URL"; then
  echo "Restarting worker with WORKER_PUBLIC_URL=$WORKER_URL ..." >&2
  pkill -f "uvicorn gpu_worker.app:app" || true
  sleep 2
  ENV_VARS+=(WORKER_PUBLIC_URL="$WORKER_URL")
  nohup env "${{ENV_VARS[@]}}" \\
    .venv/bin/python -m uvicorn gpu_worker.app:app --host 0.0.0.0 --port "$WORKER_PORT" \\
    >/tmp/gpu_worker.log 2>&1 </dev/null &
  for _ in $(seq 1 20); do
    if curl -fsS "http://127.0.0.1:$WORKER_PORT/health" >/tmp/gpu_worker_health.json 2>/dev/null; then
      break
    fi
    sleep 1
  done
  echo "Worker restarted with public URL" >&2
fi

printf 'REMOTE_ROOT=%s\\n' "$REMOTE_ROOT"
printf 'COMFY_BASE_URL=%s\\n' "$COMFY_BASE_URL"
printf 'WORKER_PORT=%s\\n' "$WORKER_PORT"
printf 'WORKER_URL=%s\\n' "$WORKER_URL"
printf 'WORKER_HEALTH=%s\\n' "$(cat /tmp/gpu_worker_health.json)"
"""


def update_env_file(env_path: Path, worker_url: str) -> None:
    content = env_path.read_text()
    line = f"GPU_WORKER_BASE_URL={worker_url}"
    if "GPU_WORKER_BASE_URL=" in content:
        content = re.sub(r"^GPU_WORKER_BASE_URL=.*$", line, content, flags=re.MULTILINE)
    else:
        suffix = "" if content.endswith("\n") else "\n"
        content = f"{content}{suffix}{line}\n"
    env_path.write_text(content)


def backend_is_running(backend_root: Path) -> bool:
    pattern = f"{backend_root}/.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    result = run(["pgrep", "-f", pattern], capture_output=True, check=False)
    return result.returncode == 0 and bool(result.stdout.strip())


def wait_for_url(url: str, timeout_sec: int = 20) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except URLError:
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}")


def restart_backend(backend_root: Path) -> None:
    uvicorn_bin = backend_root / ".venv" / "bin" / "uvicorn"
    if not uvicorn_bin.exists():
        raise RuntimeError(f"Backend uvicorn not found at {uvicorn_bin}")

    pattern = f"{uvicorn_bin} app.main:app --reload --host 0.0.0.0 --port 8000"
    run(["pkill", "-f", pattern], check=False)
    time.sleep(2)

    log_file = Path("/tmp/filmforge_backend.log")
    with log_file.open("ab") as handle:
        subprocess.Popen(
            [str(uvicorn_bin), "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
            cwd=backend_root,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    wait_for_url("http://127.0.0.1:8000/docs")


def extract_worker_url(remote_output: str) -> str:
    for line in remote_output.splitlines():
        if line.startswith("WORKER_URL="):
            return line.split("=", 1)[1].strip()
    match = WORKER_URL_PATTERN.search(remote_output)
    return match.group(0) if match else ""


def extract_worker_name(remote_output: str) -> str:
    for line in remote_output.splitlines():
        if line.startswith("WORKER_HEALTH="):
            try:
                health = json.loads(line.split("=", 1)[1].strip())
                return health.get("worker_name", "")
            except (json.JSONDecodeError, IndexError):
                pass
    return ""


def _vastai(*args: str, timeout: int = 30) -> list | dict | str:
    result = subprocess.run(
        ["vastai", *args, "--raw"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        return {"error": stderr or stdout or f"vastai exited with {result.returncode}"}
    if not stdout:
        return []
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return stdout


def _vast_matches_gpu_name(offer: dict, desired_name: str) -> bool:
    desired = str(desired_name or "").strip().lower()
    if not desired:
        return True
    actual = str(offer.get("gpu_name") or "").strip().lower()
    return desired in actual


def _vast_transfer_cost(offer: dict) -> float:
    return float(offer.get("inet_up_cost") or 0.0) + float(offer.get("inet_down_cost") or 0.0)


def _vast_preferred_offer_sort_key(offer: dict) -> tuple[float, float, float, float]:
    reliability = float(offer.get("reliability2") or 0.0)
    price = float(offer.get("dph_total") or offer.get("dph") or 9999.0)
    dlperf = float(offer.get("dlperf") or 0.0)
    return (_vast_transfer_cost(offer), price, -reliability, -dlperf)


def _select_vast_offer(args: argparse.Namespace) -> dict:
    query = " ".join(
        [
            f"gpu_ram>={int(args.vast_min_vram_gb)}",
            f"dph_total<={float(args.vast_max_price)}",
            f"disk_space>={int(args.vast_disk_gb)}",
            "rentable=true",
            "verified=true",
        ]
    )
    if args.vast_max_upload_cost is not None:
        query = f"{query} inet_up_cost<={float(args.vast_max_upload_cost)}"
    if args.vast_max_download_cost is not None:
        query = f"{query} inet_down_cost<={float(args.vast_max_download_cost)}"
    data = _vastai(
        "search",
        "offers",
        query,
        "--limit",
        str(max(int(args.vast_limit), 1)),
        "--order",
        "dph_total",
        timeout=60,
    )
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Vast search failed: {data['error']}")
    offers = data if isinstance(data, list) else []
    if not offers:
        raise RuntimeError("No Vast offers matched the configured filters.")

    exact = [offer for offer in offers if _vast_matches_gpu_name(offer, args.vast_gpu)]
    if exact:
        candidates = exact
    elif args.vast_allow_fallback_gpu:
        log(f"No exact Vast GPU match for {args.vast_gpu!r}; falling back to the broader offer pool.")
        candidates = offers
    else:
        raise RuntimeError(f"No Vast offers matched the requested GPU {args.vast_gpu!r}.")
    selected = sorted(candidates, key=_vast_preferred_offer_sort_key)[0]
    log(
        "Selected Vast offer "
        f"id={selected.get('id')} gpu={selected.get('gpu_name')} "
        f"price=${float(selected.get('dph_total') or selected.get('dph') or 0.0):.3f}/hr "
        f"up=${float(selected.get('inet_up_cost') or 0.0):.4f}/GB "
        f"down=${float(selected.get('inet_down_cost') or 0.0):.4f}/GB"
    )
    return selected


def _attach_vast_ssh_key(instance_id: str, identity: Path) -> None:
    pub_key_path = Path(f"{identity}.pub")
    if not pub_key_path.exists():
        raise RuntimeError(f"SSH public key not found at {pub_key_path}")
    public_key = pub_key_path.read_text().strip()
    data = _vastai("attach", "ssh", instance_id, public_key, timeout=30)
    if isinstance(data, dict) and "error" in data:
        message = str(data["error"])
        lowered = message.lower()
        if "already" in lowered or "exists" in lowered:
            return
        raise RuntimeError(f"Failed to attach SSH key: {message}")


def _vast_ssh_command(instance_id: str) -> str | None:
    data = _vastai("ssh-url", instance_id, timeout=30)
    if isinstance(data, dict) and "error" in data:
        return None
    # take first line only (--raw appends a trailing "null" line)
    raw = str(data or "").strip().splitlines()[0].strip()
    if raw.startswith("ssh "):
        return raw
    if raw.startswith("ssh://"):
        from urllib.parse import urlparse
        parsed = urlparse(raw)
        host = parsed.hostname or ""
        port = parsed.port or 22
        user = parsed.username or "root"
        return f"ssh -p {port} {user}@{host}"
    return None


def _vast_direct_worker_url(
    instance_id: str,
    container_port: int,
    *,
    timeout_sec: int = 180,
) -> str | None:
    """Poll Vast for the public host:port mapping of a container port.

    With `--env '-p N:N'`, Vast publishes container port N on the host's public IP
    at a Vast-assigned host port. Returns http://<public_ipaddr>:<HostPort>, or None
    if the mapping doesn't appear before the deadline.
    """
    deadline = time.time() + timeout_sec
    target_key = f"{container_port}/tcp"
    while time.time() < deadline:
        data = _vastai("show", "instance", instance_id, timeout=30)
        if isinstance(data, dict) and "error" not in data:
            public_ip = data.get("public_ipaddr")
            ports = data.get("ports") or {}
            if isinstance(public_ip, str) and public_ip and isinstance(ports, dict):
                bindings = ports.get(target_key)
                if isinstance(bindings, list) and bindings:
                    host_port = bindings[0].get("HostPort")
                    if host_port:
                        return f"http://{public_ip}:{host_port}"
        time.sleep(5)
    return None


def _wait_for_vast_ssh_command(
    instance_id: str,
    *,
    identity: Path,
    timeout_sec: int,
) -> str:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        _attach_vast_ssh_key(instance_id, identity)
        ssh_command = _vast_ssh_command(instance_id)
        if ssh_command:
            log(f"Resolved Vast SSH URL for instance {instance_id}; probing connectivity…")
            ssh_cmd, _, _ = parse_ssh_command(ssh_command)
            ssh_cmd = add_default_identity(ssh_cmd, override=identity)
            ssh_cmd = add_default_host_key_policy(ssh_cmd)
            result = subprocess.run(
                [*ssh_cmd, "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", "echo ok"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                log(f"SSH ready on Vast instance {instance_id}: {ssh_command}")
                return ssh_command
            log("  SSH port reachable but not accepting connections yet, retrying…")
        else:
            log(f"  Waiting for Vast instance {instance_id} to boot…")
        time.sleep(10)
    raise RuntimeError(f"Timed out waiting for Vast SSH access on instance {instance_id}")


def warm_remote_worker(
    worker_url: str,
    asset_groups: list[str],
    *,
    api_token: str | None = None,
    timeout_sec: int = 3600,
) -> dict:
    payload = json.dumps({"asset_groups": asset_groups}).encode()
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    request = Request(
        f"{worker_url.rstrip('/')}/assets/ensure",
        data=payload,
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=timeout_sec) as response:
        body = response.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Worker warmup returned invalid JSON: {exc}") from exc


def register_worker_with_backend(backend_url: str, worker_name: str, worker_public_url: str) -> bool:
    payload = json.dumps({"name": worker_name, "base_url": worker_public_url, "is_active": True}).encode()
    req = Request(
        f"{backend_url}/api/gpu-workers/register",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:
            log(f"Registered GPU worker '{worker_name}' with backend → {resp.status}")
            return True
    except Exception as exc:
        log(f"Warning: could not register worker with backend: {exc}")
        return False


# ── Verda automation ─────────────────────────────────────────────────────────

def _verda_cmd(args: argparse.Namespace, *parts: str) -> list[str]:
    return [str(args.verda_cli.expanduser()), "--agent", *parts]


def _verda_json(args: argparse.Namespace, *parts: str, timeout: int = 60) -> object:
    result = subprocess.run(
        _verda_cmd(args, *parts),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        raise RuntimeError(stderr or stdout or f"verda exited with {result.returncode}")
    if not stdout:
        return None
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"verda returned non-JSON output for {' '.join(parts)}: {stdout[:500]}") from exc


def _verda_check(args: argparse.Namespace, *parts: str, timeout: int = 60) -> str:
    result = subprocess.run(
        _verda_cmd(args, *parts),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(output or f"verda exited with {result.returncode}")
    return output


def _verda_find_volume(volumes: object, volume_id: str) -> dict:
    if not isinstance(volumes, list):
        raise RuntimeError(f"Unexpected Verda volume list response: {volumes!r}")
    for volume in volumes:
        if isinstance(volume, dict) and volume.get("id") == volume_id:
            return volume
    raise RuntimeError(f"Verda volume not found: {volume_id}")


def _verda_preflight(args: argparse.Namespace) -> None:
    cli = args.verda_cli.expanduser()
    if not cli.exists():
        raise RuntimeError(f"Verda CLI not found at {cli}")

    log("Checking Verda auth...")
    _verda_check(args, "auth", "show", timeout=30)

    log("Checking Verda volumes...")
    volumes = _verda_json(args, "volume", "list", timeout=60)
    os_volume = _verda_find_volume(volumes, args.verda_os_volume_id)
    data_volume = _verda_find_volume(volumes, args.verda_data_volume_id)
    for label, volume in (("OS", os_volume), ("data", data_volume)):
        status = str(volume.get("status") or "").lower()
        location = str(volume.get("location") or "")
        if status != "detached":
            raise RuntimeError(f"Verda {label} volume must be detached before deploy: id={volume.get('id')} status={status}")
        if location != args.verda_location:
            raise RuntimeError(
                f"Verda {label} volume is in {location}, but deploy location is {args.verda_location}"
            )

    log("Checking Verda GPU availability...")
    availability = _verda_json(args, "availability", "--location", args.verda_location, timeout=60)
    available_types: set[str] = set()
    if isinstance(availability, list):
        for row in availability:
            if isinstance(row, dict):
                available_types.update(str(item) for item in (row.get("instance_types") or []))
    if args.verda_instance_type not in available_types:
        types = ", ".join(sorted(available_types)) or "(none)"
        raise RuntimeError(
            f"Verda instance type {args.verda_instance_type!r} is not currently available in "
            f"{args.verda_location}. Available: {types}"
        )


def _verda_fresh_preflight(args: argparse.Namespace) -> None:
    cli = args.verda_cli.expanduser()
    if not cli.exists():
        raise RuntimeError(f"Verda CLI not found at {cli}")

    log("Checking Verda auth...")
    _verda_check(args, "auth", "show", timeout=30)

    log("Checking Verda GPU availability...")
    availability = _verda_json(args, "availability", "--location", args.verda_location, timeout=60)
    available_types: set[str] = set()
    if isinstance(availability, list):
        for row in availability:
            if isinstance(row, dict):
                available_types.update(str(item) for item in (row.get("instance_types") or []))
    if args.verda_instance_type not in available_types:
        types = ", ".join(sorted(available_types)) or "(none)"
        raise RuntimeError(
            f"Verda instance type {args.verda_instance_type!r} is not currently available in "
            f"{args.verda_location}. Available: {types}"
        )


def _verda_instance_by_hostname(args: argparse.Namespace, hostname: str) -> dict | None:
    instances = _verda_json(args, "vm", "list", timeout=60)
    if not isinstance(instances, list):
        return None
    candidates = [
        item for item in instances
        if isinstance(item, dict) and item.get("hostname") == hostname
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return candidates[0]


def _wait_for_verda_instance_ip(args: argparse.Namespace, hostname: str) -> tuple[str, str]:
    deadline = time.time() + args.verda_ssh_timeout
    while time.time() < deadline:
        instance = _verda_instance_by_hostname(args, hostname)
        if instance:
            ip = str(instance.get("ip") or "").strip()
            instance_id = str(instance.get("id") or "").strip()
            status = str(instance.get("status") or "").strip()
            if ip and instance_id and status.lower() == "running":
                return ip, instance_id
            log(f"  Waiting for Verda instance status/ip... status={status or '?'} ip={ip or '?'}")
        else:
            log("  Waiting for Verda instance to appear in vm list...")
        time.sleep(10)
    raise RuntimeError(f"Timed out waiting for Verda instance {hostname!r} to become running with an IP")


def _wait_for_verda_ssh(ip: str, identity: Path, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        result = subprocess.run(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=8",
                "-o", "BatchMode=yes",
                "-i", str(identity),
                f"root@{ip}",
                "echo ok",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        reason = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no response"
        log(f"  SSH not ready on {ip} ({reason}), retrying...")
        time.sleep(10)
    raise RuntimeError(f"Timed out waiting for SSH on Verda instance {ip}")


def verda_rehydrate_script(
    *,
    public_ip: str,
    worker_port: int,
    comfy_port: int,
    worker_count: int,
    remote_root: str,
) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

PUBLIC_IP={shlex.quote(public_ip)}
WORKER_PORT_BASE={worker_port}
COMFY_PORT_BASE={comfy_port}
WORKER_COUNT_REQUESTED={worker_count}
REMOTE_ROOT={shlex.quote(remote_root)}
COMFY_ROOT="/workspace/ComfyUI"
WORKER_ROOT="/opt/filmforge_gpu_worker"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; GPU drivers are not ready" >&2
  exit 1
fi

GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
if test "$WORKER_COUNT_REQUESTED" -gt 0 && test "$WORKER_COUNT_REQUESTED" -lt "$GPU_COUNT"; then
  GPU_COUNT="$WORKER_COUNT_REQUESTED"
fi
if test "$GPU_COUNT" -lt 1; then
  echo "No GPUs detected" >&2
  exit 1
fi

mkdir -p /mnt/data
if ! mountpoint -q /mnt/data; then
  mount /dev/vdb /mnt/data
fi

for unit in $(systemctl list-units --type=service --all --no-legend 'filmforge-worker-gpu*.service' 'comfyui-gpu*.service' 2>/dev/null | awk '{{print $1}}'); do
  systemctl stop "$unit" || true
done

mkdir -p /mnt/data/ComfyUI/models /mnt/data/ComfyUI/input /mnt/data/ComfyUI/output /mnt/data/ComfyUI/temp
mkdir -p "$COMFY_ROOT"

bind_comfy_dir() {{
  src="$1"
  dst="$2"
  mkdir -p "$src"
  if mountpoint -q "$dst"; then
    return
  fi
  if test -L "$dst"; then
    rm "$dst"
  elif test -e "$dst" && ! test -d "$dst"; then
    rm -rf "$dst"
  fi
  mkdir -p "$dst"
  mount --bind "$src" "$dst"
}}

for dir in models input output temp; do
  bind_comfy_dir "/mnt/data/ComfyUI/$dir" "$COMFY_ROOT/$dir"
done

if ! test -f "$COMFY_ROOT/main.py"; then
  echo "ComfyUI not found at $COMFY_ROOT/main.py; OS volume is not worker-ready" >&2
  exit 1
fi
if ! test -x "$COMFY_ROOT/.venv/bin/python"; then
  echo "ComfyUI venv not found at $COMFY_ROOT/.venv/bin/python" >&2
  exit 1
fi
WORKER_MODULE_DIR="$(dirname "$WORKER_ROOT")"
if test -f "$WORKER_ROOT/app.py"; then
  mkdir -p "$WORKER_MODULE_DIR"
  ln -sfn "$WORKER_ROOT" "$WORKER_MODULE_DIR/gpu_worker"
elif test -f "$WORKER_ROOT/gpu_worker/app.py"; then
  WORKER_MODULE_DIR="$WORKER_ROOT"
elif test -f "$REMOTE_ROOT/app.py"; then
  WORKER_ROOT="$REMOTE_ROOT"
  WORKER_MODULE_DIR="$(dirname "$WORKER_ROOT")"
  ln -sfn "$WORKER_ROOT" "$WORKER_MODULE_DIR/gpu_worker"
elif test -f "$REMOTE_ROOT/gpu_worker/app.py"; then
    WORKER_ROOT="$REMOTE_ROOT"
    WORKER_MODULE_DIR="$WORKER_ROOT"
else
  echo "gpu_worker code not found at /opt/filmforge_gpu_worker or $REMOTE_ROOT" >&2
  exit 1
fi
if ! test -x "$WORKER_ROOT/.venv/bin/python"; then
  echo "gpu_worker venv not found at $WORKER_ROOT/.venv/bin/python" >&2
  exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)"
VRAM_GB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{{printf "%.0f", $1 / 1024}}')"

for idx in $(seq 0 $((GPU_COUNT - 1))); do
  comfy_port=$((COMFY_PORT_BASE + idx))
  worker_port=$((WORKER_PORT_BASE + idx))
  comfy_user_dir="$COMFY_ROOT/user_gpu${{idx}}"
  mkdir -p "$comfy_user_dir" "$COMFY_ROOT/temp/gpu${{idx}}"

  cat > "/etc/systemd/system/comfyui-gpu${{idx}}.service" <<UNIT
[Unit]
Description=ComfyUI GPU ${{idx}}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$COMFY_ROOT
Environment=CUDA_VISIBLE_DEVICES=${{idx}}
ExecStart=$COMFY_ROOT/.venv/bin/python main.py --listen 127.0.0.1 --port ${{comfy_port}} --enable-cors-header --user-directory ${{comfy_user_dir}} --temp-directory $COMFY_ROOT/temp/gpu${{idx}}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

  cat > "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service" <<UNIT
[Unit]
Description=FilmForge GPU Worker API GPU ${{idx}}
After=network-online.target comfyui-gpu${{idx}}.service
Wants=network-online.target comfyui-gpu${{idx}}.service

[Service]
Type=simple
WorkingDirectory=$WORKER_MODULE_DIR
Environment=CUDA_VISIBLE_DEVICES=${{idx}}
Environment=COMFY_BASE_URL=http://127.0.0.1:${{comfy_port}}
Environment=COMFY_OUTPUT_DIR=$COMFY_ROOT/output
Environment=COMFY_TEMP_DIR=$COMFY_ROOT/temp
Environment=COMFY_INPUT_DIR=$COMFY_ROOT/input
Environment=WORKER_HOST=0.0.0.0
Environment=WORKER_PORT=${{worker_port}}
Environment=WORKER_NAME=filmforge-verda-${{PUBLIC_IP}}-gpu${{idx}}
Environment=WORKER_PROVIDER=verda
Environment="WORKER_GPU_NAME=${{GPU_NAME}}"
Environment=WORKER_VRAM_GB=${{VRAM_GB}}
Environment="WORKER_CAPABILITIES=${{WORKER_CAPABILITIES:-flux2_stills,wan_i2v,stable_audio}}"
Environment=WORKER_PUBLIC_URL=http://${{PUBLIC_IP}}:${{worker_port}}
Environment=WORKER_ID_FILE=/workspace/.filmforge_worker_gpu${{idx}}.id
Environment=MODEL_DOWNLOAD_TIMEOUT_SEC=7200
Environment=COMFY_HEALTH_TIMEOUT_SEC=180
Environment="COMFY_STOP_CMD=systemctl stop comfyui-gpu${{idx}}.service"
Environment="COMFY_START_CMD=systemctl start comfyui-gpu${{idx}}.service"
Environment=WORKER_MAX_CONCURRENT_JOBS=1
Environment=WORKER_HEARTBEAT_SECONDS=30
Environment=RENDER_BROKER_HEARTBEAT_SEC=30
UNIT

  if test -n "${{FILMFORGE_BACKEND_URL:-}}"; then
    echo "Environment=FILMFORGE_BACKEND_URL=${{FILMFORGE_BACKEND_URL}}" >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
  fi
  if test -n "${{WORKER_REGISTRATION_TOKEN:-}}"; then
    echo "Environment=WORKER_REGISTRATION_TOKEN=${{WORKER_REGISTRATION_TOKEN}}" >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
  fi
  if test -n "${{RENDER_BROKER_WORKER_TOKEN:-}}"; then
    echo "Environment=RENDER_BROKER_WORKER_TOKEN=${{RENDER_BROKER_WORKER_TOKEN}}" >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
  fi
  if test -n "${{WORKER_API_TOKEN:-}}"; then
    echo "Environment=WORKER_API_TOKEN=${{WORKER_API_TOKEN}}" >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
  fi

  cat >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service" <<UNIT
ExecStart=$WORKER_ROOT/.venv/bin/python -m uvicorn gpu_worker.app:app --host 0.0.0.0 --port ${{worker_port}}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
done

systemctl daemon-reload
for idx in $(seq 0 $((GPU_COUNT - 1))); do
  systemctl enable --now "comfyui-gpu${{idx}}.service"
done

for idx in $(seq 0 $((GPU_COUNT - 1))); do
  port=$((COMFY_PORT_BASE + idx))
  for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${{port}}/system_stats" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
done

for idx in $(seq 0 $((GPU_COUNT - 1))); do
  systemctl enable --now "filmforge-worker-gpu${{idx}}.service"
done

for idx in $(seq 0 $((GPU_COUNT - 1))); do
  port=$((WORKER_PORT_BASE + idx))
  for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${{port}}/health" >/tmp/filmforge_worker_gpu${{idx}}_health.json 2>/dev/null; then
      break
    fi
    sleep 2
  done
  if ! test -s "/tmp/filmforge_worker_gpu${{idx}}_health.json"; then
    echo "Worker gpu${{idx}} failed health check" >&2
    journalctl -u "filmforge-worker-gpu${{idx}}.service" -n 100 --no-pager >&2 || true
    exit 1
  fi
  echo "WORKER_URL=http://${{PUBLIC_IP}}:${{port}}"
  echo "WORKER_HEALTH_GPU${{idx}}=$(cat /tmp/filmforge_worker_gpu${{idx}}_health.json)"
done

echo "GPU_COUNT=${{GPU_COUNT}}"
df -h /mnt/data
"""


def verda_fresh_install_script(
    *,
    worker_repo_url: str,
    comfy_repo_url: str,
    pytorch_index_url: str,
    remote_root: str,
) -> str:
    """Install ComfyUI and gpu_worker onto a fresh Verda base image."""

    return f"""#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
REMOTE_ROOT={shlex.quote(remote_root)}
WORKER_ROOT="/opt/filmforge_gpu_worker"
COMFY_ROOT="/workspace/ComfyUI"
WORKER_REPO_URL={shlex.quote(worker_repo_url)}
COMFY_REPO_URL={shlex.quote(comfy_repo_url)}
PYTORCH_INDEX_URL={shlex.quote(pytorch_index_url)}

apt-get update
apt-get install -y --no-install-recommends \\
  ca-certificates curl git rsync python3 python3-dev python3-pip python3-venv \\
  build-essential e2fsprogs ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1

mkdir -p /opt /workspace "$REMOTE_ROOT"

if test -b /dev/vdb && ! blkid /dev/vdb >/dev/null 2>&1; then
  mkfs.ext4 -F /dev/vdb
fi

if test -d "$WORKER_ROOT/.git"; then
  git -C "$WORKER_ROOT" pull --ff-only || true
else
  rm -rf "$WORKER_ROOT"
  GIT_TERMINAL_PROMPT=0 git clone "$WORKER_REPO_URL" "$WORKER_ROOT"
fi
ln -sfn "$WORKER_ROOT" /opt/gpu_worker

if ! test -x "$WORKER_ROOT/.venv/bin/python"; then
  python3 -m venv "$WORKER_ROOT/.venv"
fi
"$WORKER_ROOT/.venv/bin/python" -m pip install --upgrade pip wheel setuptools
"$WORKER_ROOT/.venv/bin/python" -m pip install -r "$WORKER_ROOT/requirements.txt"

if test -d "$COMFY_ROOT/.git"; then
  git -C "$COMFY_ROOT" pull --ff-only || true
else
  rm -rf "$COMFY_ROOT"
  GIT_TERMINAL_PROMPT=0 git clone "$COMFY_REPO_URL" "$COMFY_ROOT"
fi

if ! test -x "$COMFY_ROOT/.venv/bin/python"; then
  python3 -m venv "$COMFY_ROOT/.venv"
fi
"$COMFY_ROOT/.venv/bin/python" -m pip install --upgrade pip wheel setuptools
"$COMFY_ROOT/.venv/bin/python" -m pip install torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL"
"$COMFY_ROOT/.venv/bin/python" -m pip install -r "$COMFY_ROOT/requirements.txt"

mkdir -p "$COMFY_ROOT/custom_nodes"
if ! test -d "$COMFY_ROOT/custom_nodes/ComfyUI-VideoHelperSuite"; then
  GIT_TERMINAL_PROMPT=0 git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "$COMFY_ROOT/custom_nodes/ComfyUI-VideoHelperSuite" || true
fi
if ! test -d "$COMFY_ROOT/custom_nodes/ComfyUI_IPAdapter_plus"; then
  GIT_TERMINAL_PROMPT=0 git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git "$COMFY_ROOT/custom_nodes/ComfyUI_IPAdapter_plus" || true
fi

for req in "$COMFY_ROOT"/custom_nodes/*/requirements.txt; do
  if test -f "$req"; then
    "$COMFY_ROOT/.venv/bin/python" -m pip install -r "$req" || true
  fi
done

echo "FRESH_INSTALL_DONE=1"
"""


def extract_worker_urls(remote_output: str) -> list[str]:
    urls: list[str] = []
    for line in remote_output.splitlines():
        if line.startswith("WORKER_URLS="):
            for url in line.split("=", 1)[1].split(","):
                url = url.strip()
                if url and url not in urls:
                    urls.append(url)
        if line.startswith("WORKER_URL="):
            url = line.split("=", 1)[1].strip()
            if url and url not in urls:
                urls.append(url)
    return urls


def _verda_env_vars(args: argparse.Namespace) -> list[str]:
    env_vars = list(getattr(args, "env_vars", []) or [])
    existing_keys = {item.split("=", 1)[0] for item in env_vars if "=" in item}
    for key in ("WORKER_REGISTRATION_TOKEN", "RENDER_BROKER_WORKER_TOKEN", "WORKER_API_TOKEN"):
        if key not in existing_keys:
            value = _read_env_value(args.backend_env, key)
            if value:
                env_vars.append(f"{key}={value}")
                existing_keys.add(key)
                log(f"Auto-injected {key} from backend .env")
    has_registration_token = bool(
        {"WORKER_REGISTRATION_TOKEN", "RENDER_BROKER_WORKER_TOKEN"} & existing_keys
    )
    if has_registration_token and "FILMFORGE_BACKEND_URL" not in existing_keys:
        value = _read_env_value(args.backend_env, "FILMFORGE_BACKEND_URL")
        if value:
            env_vars.append(f"FILMFORGE_BACKEND_URL={value}")
            log("Auto-injected FILMFORGE_BACKEND_URL from backend .env")
    return env_vars


def _verda_contract(args: argparse.Namespace) -> str:
    value = str(getattr(args, "verda_contract", DEFAULT_VERDA_CONTRACT) or DEFAULT_VERDA_CONTRACT).lower()
    aliases = {
        "on_demand": "pay_as_go",
        "ondemand": "pay_as_go",
        "pay_as_you_go": "pay_as_go",
        "payg": "pay_as_go",
    }
    value = aliases.get(value, value)
    if value not in {"pay_as_go", "spot"}:
        raise RuntimeError(f"Unsupported Verda contract {value!r}; expected pay_as_go or spot")
    return value


def _verda_billing_flags(args: argparse.Namespace, *, fresh: bool) -> list[str]:
    contract = _verda_contract(args)
    flags = ["--contract", contract, "--pricing", "FIXED_PRICE"]
    if contract == "spot":
        flags.append("--is-spot")
        if fresh:
            flags.extend([
                "--os-volume-on-spot-discontinue",
                "keep_detached",
                "--storage-on-spot-discontinue",
                "keep_detached",
            ])
    return flags


def verda_deploy(args: argparse.Namespace) -> int:
    identity = (args.ssh_identity or Path.home() / ".ssh" / "id_ed25519").expanduser()
    if not identity.exists():
        raise RuntimeError(f"SSH private key not found at {identity}")

    _verda_preflight(args)

    hostname = args.verda_hostname
    log(
        "Creating Verda VM "
        f"hostname={hostname} type={args.verda_instance_type} location={args.verda_location} "
        f"os_volume={args.verda_os_volume_id} data_volume={args.verda_data_volume_id} "
        f"contract={_verda_contract(args)}"
    )
    create_cmd = [
        "vm",
        "create",
        "--kind",
        "gpu",
        "--instance-type",
        args.verda_instance_type,
        "--location",
        args.verda_location,
        "--os",
        args.verda_os_volume_id,
        "--hostname",
        hostname,
        "--ssh-key",
        args.verda_ssh_key_id,
        "--existing-volume",
        args.verda_data_volume_id,
        "--wait",
        "--wait-timeout",
        args.verda_wait_timeout,
    ]
    create_cmd.extend(_verda_billing_flags(args, fresh=False))
    create_output = _verda_check(args, *create_cmd, timeout=max(args.verda_create_timeout, 60))
    if create_output:
        print(create_output)

    ip, instance_id = _wait_for_verda_instance_ip(args, hostname)
    log(f"Verda instance running: id={instance_id} ip={ip}")

    _wait_for_verda_ssh(ip, identity, args.verda_ssh_timeout)
    ssh_command = f"ssh -i {identity} root@{ip}"
    ssh_cmd, _, _ = parse_ssh_command(ssh_command)
    ssh_cmd = add_default_host_key_policy(ssh_cmd)

    env_vars = _verda_env_vars(args)

    script = verda_rehydrate_script(
        public_ip=ip,
        worker_port=args.worker_port,
        comfy_port=args.verda_comfy_port,
        worker_count=args.verda_worker_count,
        remote_root=args.remote_root,
    )
    env_exports = build_env_exports(env_vars)
    if env_exports:
        script = f"{env_exports}\n\n{script}"

    log("Running Verda post-boot worker fixup...")
    try:
        remote_result = run([*ssh_cmd, "bash", "-s"], input_text=script, capture_output=True)
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        raise
    if remote_result.stdout:
        print(remote_result.stdout, end="")
    if remote_result.stderr:
        print(remote_result.stderr, file=sys.stderr, end="")

    worker_urls = extract_worker_urls(remote_result.stdout)
    if worker_urls:
        log("Verda worker URLs:")
        for url in worker_urls:
            log(f"  {url}")
        print(f"WORKER_URLS={','.join(worker_urls)}")
    else:
        log("No Verda worker URLs found in post-boot output.")

    (SCRIPT_DIR / ".last_ssh_dest").write_text(f"-i {identity} root@{ip}\n")

    if worker_urls and args.warm_asset_groups and not args.skip_warmup:
        for url in worker_urls:
            _run_worker_warmup(args, url)

    return 0


def verda_fresh_deploy(args: argparse.Namespace) -> int:
    identity = (args.ssh_identity or Path.home() / ".ssh" / "id_ed25519").expanduser()
    if not identity.exists():
        raise RuntimeError(f"SSH private key not found at {identity}")

    _verda_fresh_preflight(args)

    hostname = args.verda_hostname
    os_volume_name = args.verda_fresh_os_volume_name or f"{hostname}-os"
    storage_name = args.verda_fresh_storage_name or f"{hostname}-models"
    log(
        "Creating fresh Verda VM "
        f"hostname={hostname} type={args.verda_instance_type} location={args.verda_location} "
        f"os={args.verda_fresh_os_image} os_size={args.verda_fresh_os_volume_size} "
        f"storage={storage_name}:{args.verda_fresh_storage_size}GB "
        f"contract={_verda_contract(args)}"
    )
    create_cmd = [
        "vm",
        "create",
        "--kind",
        "gpu",
        "--instance-type",
        args.verda_instance_type,
        "--location",
        args.verda_location,
        "--os",
        args.verda_fresh_os_image,
        "--os-volume-size",
        str(args.verda_fresh_os_volume_size),
        "--os-volume-name",
        os_volume_name,
        "--storage-size",
        str(args.verda_fresh_storage_size),
        "--storage-name",
        storage_name,
        "--storage-type",
        args.verda_fresh_storage_type,
        "--hostname",
        hostname,
        "--ssh-key",
        args.verda_ssh_key_id,
        "--wait",
        "--wait-timeout",
        args.verda_wait_timeout,
    ]
    create_cmd.extend(_verda_billing_flags(args, fresh=True))
    create_output = _verda_check(args, *create_cmd, timeout=max(args.verda_create_timeout, 60))
    if create_output:
        print(create_output)

    ip, instance_id = _wait_for_verda_instance_ip(args, hostname)
    log(f"Fresh Verda instance running: id={instance_id} ip={ip}")

    _wait_for_verda_ssh(ip, identity, args.verda_ssh_timeout)
    ssh_command = f"ssh -i {identity} root@{ip}"
    ssh_cmd, _, _ = parse_ssh_command(ssh_command)
    ssh_cmd = add_default_host_key_policy(ssh_cmd)

    install_script = verda_fresh_install_script(
        worker_repo_url=args.verda_worker_repo_url,
        comfy_repo_url=args.verda_comfy_repo_url,
        pytorch_index_url=args.verda_pytorch_index_url,
        remote_root=args.remote_root,
    )
    log("Installing worker stack on fresh Verda OS volume...")
    try:
        install_result = run([*ssh_cmd, "bash", "-s"], input_text=install_script, capture_output=True)
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        raise
    if install_result.stdout:
        print(install_result.stdout, end="")
    if install_result.stderr:
        print(install_result.stderr, file=sys.stderr, end="")

    env_vars = _verda_env_vars(args)
    script = verda_rehydrate_script(
        public_ip=ip,
        worker_port=args.worker_port,
        comfy_port=args.verda_comfy_port,
        worker_count=args.verda_worker_count,
        remote_root=args.remote_root,
    )
    env_exports = build_env_exports(env_vars)
    if env_exports:
        script = f"{env_exports}\n\n{script}"

    log("Starting workers on fresh Verda VM...")
    try:
        remote_result = run([*ssh_cmd, "bash", "-s"], input_text=script, capture_output=True)
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        raise
    if remote_result.stdout:
        print(remote_result.stdout, end="")
    if remote_result.stderr:
        print(remote_result.stderr, file=sys.stderr, end="")

    worker_urls = extract_worker_urls(remote_result.stdout)
    if worker_urls:
        log("Fresh Verda worker URLs:")
        for url in worker_urls:
            log(f"  {url}")
        print(f"WORKER_URLS={','.join(worker_urls)}")
    else:
        log("No Verda worker URLs found in post-boot output.")

    (SCRIPT_DIR / ".last_ssh_dest").write_text(f"-i {identity} root@{ip}\n")

    if worker_urls and not args.skip_warmup:
        if not args.warm_asset_groups:
            args.warm_asset_groups = list(DEFAULT_VERDA_FRESH_WARM_GROUPS)
        _run_worker_warmup(args, worker_urls[0])

    return 0


# ── RunPod automation ────────────────────────────────────────────────────────

RUNPOD_IMAGE = "runpod/comfyui:latest"
RUNPOD_GPU_TYPE = "NVIDIA L40S"
RUNPOD_BOOT_TIMEOUT = 300   # seconds to wait for pod SSH to become ready


def _pod_name_for_gpu(gpu_type: str) -> str:
    """Derive a stable pod name from the GPU type ID, e.g. 'filmforge_rtx5090'."""
    slug = re.sub(r"[^a-z0-9]+", "_", gpu_type.lower()).strip("_")
    # Shorten common prefixes for readability
    slug = re.sub(r"^nvidia_geforce_", "", slug)
    slug = re.sub(r"^nvidia_", "", slug)
    slug = slug.replace("rtx_", "rtx").replace("geforce_", "")
    return f"filmforge_{slug}"
RUNPOD_SSH_POLL_INTERVAL = 10


def _read_env_value(env_path: Path, key: str) -> str | None:
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return None


def _runpod_find_running_pod(rp: object, pod_name: str) -> dict | None:
    """Return the pod with the given name if it exists (any status), or None."""
    try:
        pods = rp.get_pods()  # type: ignore[attr-defined]
    except Exception as exc:
        log(f"RunPod API error listing pods: {exc}")
        return None
    for pod in pods:
        if pod.get("name") == pod_name:
            status = pod.get("desiredStatus", "?")
            if status != "RUNNING":
                log(f"Found pod '{pod_name}' (status={status}) — will resume it.")
            return pod
    return None


def _runpod_get_ssh_port(pod: dict) -> tuple[str, int] | None:
    """Return (ip, port) for the pod's public SSH TCP port, or None."""
    runtime = pod.get("runtime") or {}
    ports = runtime.get("ports") or []
    for p in ports:
        if p.get("privatePort") == 22 and p.get("isIpPublic"):
            return p["ip"], p["publicPort"]
    return None


def _runpod_proxy_url(pod_id: str, worker_port: int) -> str:
    return f"https://{pod_id}-{worker_port}.proxy.runpod.net"


def _wait_for_ssh(ip: str, port: int, identity: Path, timeout: int) -> bool:
    """Poll until SSH accepts connections. Returns True on success."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=8",
                "-o", "BatchMode=yes",
                "-p", str(port),
                "-i", str(identity),
                f"root@{ip}",
                "echo ok",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True
        log(f"  SSH not ready yet (port {port}), retrying…")
        time.sleep(RUNPOD_SSH_POLL_INTERVAL)
    return False


def runpod_deploy(args: argparse.Namespace) -> int:
    """Fully automated RunPod deploy: find/create pod → wait → deploy worker → update env."""
    try:
        import runpod as rp  # type: ignore[import]
    except ImportError:
        log("ERROR: 'runpod' Python package not installed. Run: pip install runpod")
        return 1

    # Read API key from backend .env
    api_key = _read_env_value(args.backend_env, "RUNPOD_API_KEY")
    if not api_key:
        log(f"ERROR: RUNPOD_API_KEY not found in {args.backend_env}")
        return 1
    rp.api_key = api_key

    # Read SSH public key
    identity: Path = args.ssh_identity or Path.home() / ".ssh" / "id_ed25519"
    pub_key_path = Path(str(identity) + ".pub")
    if not pub_key_path.exists():
        log(f"ERROR: SSH public key not found at {pub_key_path}")
        return 1
    pub_key = pub_key_path.read_text().strip()

    # Find existing running pod — by explicit ID, or by GPU-derived name
    explicit_pod_id = getattr(args, "pod_id", None)
    gpu_type = getattr(args, "gpu", None) or RUNPOD_GPU_TYPE
    pod_name = _pod_name_for_gpu(gpu_type)
    if explicit_pod_id:
        pods = rp.get_pods()
        pod = next((p for p in pods if p.get("id") == explicit_pod_id), None)
        if pod:
            log(f"Using specified pod: {pod['id']} ({pod.get('name')})")
        else:
            log(f"ERROR: Pod ID '{explicit_pod_id}' not found")
            return 1
    else:
        pod = _runpod_find_running_pod(rp, pod_name)
    if pod:
        log(f"Found existing pod: {pod['id']} ({pod.get('name')})")
    else:
        log(f"No pod '{pod_name}' found — creating new pod ({gpu_type}, {RUNPOD_IMAGE})…")
        try:
            pod = rp.create_pod(
                name=pod_name,
                image_name=RUNPOD_IMAGE,
                gpu_type_id=gpu_type,
                cloud_type=getattr(args, "runpod_cloud_type", "COMMUNITY"),
                gpu_count=1,
                volume_in_gb=getattr(args, "runpod_volume_gb", 150),
                container_disk_in_gb=getattr(args, "runpod_container_disk_gb", 50),
                min_vcpu_count=8,
                min_memory_in_gb=50,
                ports=f"22/tcp,8188/http,{args.worker_port}/http",
                volume_mount_path="/workspace",
                env={"PUBLIC_KEY": pub_key, "FILMFORGE_OWNED": "1"},
            )
        except Exception as exc:
            log(f"ERROR: Failed to create RunPod pod: {exc}")
            return 1
        log(f"Pod created: {pod['id']}")

    pod_id = pod["id"]

    # Resume pod if it's not already running
    if pod.get("desiredStatus") != "RUNNING":
        log(f"Resuming pod {pod_id}…")
        try:
            rp.resume_pod(pod_id, int(pod.get("gpuCount") or 1))  # type: ignore[attr-defined]
        except Exception as exc:
            log(f"WARNING: resume_pod failed ({exc}) — pod may self-start, continuing…")

    # Wait for SSH to become available
    log(f"Waiting for pod {pod_id} SSH to become ready (up to {RUNPOD_BOOT_TIMEOUT}s)…")
    ssh_info = None
    deadline = time.time() + RUNPOD_BOOT_TIMEOUT
    while time.time() < deadline:
        pods = rp.get_pods()
        current = next((p for p in pods if p["id"] == pod_id), None)
        if current:
            ssh_info = _runpod_get_ssh_port(current)
            if ssh_info:
                break
        log("  Waiting for SSH port mapping…")
        time.sleep(RUNPOD_SSH_POLL_INTERVAL)

    if not ssh_info:
        log("ERROR: Timed out waiting for SSH port.")
        return 1

    ip, port = ssh_info
    log(f"SSH port ready: root@{ip} -p {port}")

    if not _wait_for_ssh(ip, port, identity, timeout=120):
        log("ERROR: SSH port is mapped but connection is refused. Pod may still be booting.")
        return 1

    log("SSH connected.")

    # Build ssh_command string and delegate to the normal deploy path
    ssh_command = f"ssh root@{ip} -p {port} -i {identity}"
    args.ssh_command = ssh_command

    # Run normal deploy
    exit_code, worker_url = _do_deploy(args, pod_id=pod_id)
    if exit_code == 0 and worker_url and args.warm_asset_groups and not args.skip_warmup:
        _run_worker_warmup(args, worker_url)
    return exit_code


# ── Shared deploy logic ───────────────────────────────────────────────────────

def _run_worker_warmup(args: argparse.Namespace, worker_url: str) -> None:
    token = (
        _read_env_value(args.backend_env, "WORKER_API_TOKEN")
        or _read_env_value(args.backend_env, "RENDER_BROKER_WORKER_TOKEN")
        or os.getenv("WORKER_API_TOKEN")
        or os.getenv("RENDER_BROKER_WORKER_TOKEN")
    )
    asset_groups = [group for group in args.warm_asset_groups if str(group).strip()]
    if not asset_groups:
        return
    wait_for_url(f"{worker_url.rstrip('/')}/health", timeout_sec=60)
    log(f"Warming worker asset groups via {worker_url}: {', '.join(asset_groups)}")
    result = warm_remote_worker(worker_url, asset_groups, api_token=token)
    log(f"Warmup result: {json.dumps(result, indent=2, sort_keys=True)}")


def vast_deploy(args: argparse.Namespace) -> int:
    identity = (args.ssh_identity or DEFAULT_SSH_IDENTITY).expanduser()
    if not identity.exists():
        raise RuntimeError(f"SSH private key not found at {identity}")

    offer = _select_vast_offer(args)
    create_args = [
        "create",
        "instance",
        str(offer["id"]),
    ]
    if args.vast_template_hash:
        create_args.extend(["--template_hash", str(args.vast_template_hash)])
    if args.vast_image:
        create_args.extend(["--image", str(args.vast_image)])
    create_args.extend(
        [
            "--disk",
            str(args.vast_disk_gb),
            "--ssh",
            "--direct",
            "--env",
            f"-p {args.worker_port}:{args.worker_port}",
        ]
    )
    create = _vastai(*create_args, timeout=120)
    if isinstance(create, dict) and "error" in create:
        raise RuntimeError(f"Failed to create Vast instance: {create['error']}")

    instance_id = None
    if isinstance(create, dict):
        instance_id = create.get("new_contract") or create.get("instance_id") or create.get("id")
    if instance_id is None:
        raise RuntimeError(f"Unexpected Vast create response: {create}")
    instance_id = str(instance_id)
    log(f"Created Vast instance {instance_id}")

    args.ssh_identity = identity

    # Start polling for direct port mapping in parallel with SSH wait
    import threading
    direct_url_result: list[str | None] = [None]

    def _poll_direct_url() -> None:
        direct_url_result[0] = _vast_direct_worker_url(
            instance_id, args.worker_port, timeout_sec=args.vast_boot_timeout
        )

    poll_thread = threading.Thread(target=_poll_direct_url, daemon=True)
    poll_thread.start()

    args.ssh_command = _wait_for_vast_ssh_command(
        instance_id,
        identity=identity,
        timeout_sec=args.vast_boot_timeout,
    )

    # Give the poll thread a moment to finish if it hasn't already
    poll_thread.join(timeout=30)
    direct_url = direct_url_result[0]

    if direct_url:
        log(f"Vast direct worker URL: {direct_url}")
        env_vars = list(getattr(args, "env_vars", []) or [])
        if not any(v.startswith("WORKER_PUBLIC_URL=") for v in env_vars):
            env_vars.append(f"WORKER_PUBLIC_URL={direct_url}")
            args.env_vars = env_vars
    else:
        log(
            f"Could not resolve Vast direct port mapping for container port "
            f"{args.worker_port} within {args.vast_boot_timeout}s; "
            f"falling back to cloudflared."
        )

    exit_code, worker_url = _do_deploy(args)
    if exit_code == 0 and worker_url and args.warm_asset_groups and not args.skip_warmup:
        _run_worker_warmup(args, worker_url)
    return exit_code


def _do_deploy(args: argparse.Namespace, pod_id: str | None = None) -> tuple[int, str | None]:
    """Core deploy: scp worker + run remote bootstrap + update env."""
    ssh_cmd, scp_cmd, destination = parse_ssh_command(args.ssh_command)
    ssh_cmd = add_default_identity(ssh_cmd, override=args.ssh_identity)
    scp_cmd = add_default_identity(scp_cmd, override=args.ssh_identity)
    ssh_cmd = add_default_host_key_policy(ssh_cmd)
    scp_cmd = add_default_host_key_policy(scp_cmd)

    run([*ssh_cmd, "mkdir", "-p", args.remote_root])
    # Clone or update gpu_worker from GitHub
    clone_script = f"""set -euo pipefail
REMOTE_ROOT={shlex.quote(args.remote_root)}
# Ensure git is available on the remote
if ! command -v git >/dev/null 2>&1; then
  echo "Installing git..." >&2
  apt-get update -qq 2>/dev/null || true
  apt-get install -y -qq git 2>/dev/null || true
fi
cd "$REMOTE_ROOT"
if test -d gpu_worker/.git; then
  echo "gpu_worker already exists, updating..." >&2
  cd gpu_worker
  git pull origin main 2>/dev/null || echo "git pull failed" >&2
  cd ..
else
  rm -rf gpu_worker
  echo "Cloning gpu_worker from GitHub..." >&2
  GIT_TERMINAL_PROMPT=0 git clone https://github.com/taxydriver/gpu_worker.git gpu_worker
fi
"""
    run([*ssh_cmd, "bash", "-s"], input_text=clone_script)

    try:
        # Auto-inject critical env vars from backend .env if not already provided
        env_vars = list(getattr(args, "env_vars", []) or [])
        existing_keys = {v.split("=", 1)[0] for v in env_vars if "=" in v}
        for key in ("FILMFORGE_BACKEND_URL", "WORKER_REGISTRATION_TOKEN", "RENDER_BROKER_WORKER_ID", "WORKER_CAPABILITIES"):
            if key not in existing_keys:
                value = _read_env_value(args.backend_env, key)
                if value:
                    env_vars.append(f"{key}={value}")
                    log(f"Auto-injected {key} from backend .env")

        script = remote_script(args.remote_root, args.worker_port)
        env_exports = build_env_exports(env_vars)
        if env_exports:
            script = f"{env_exports}\n\n{script}"
        remote_result = run(
            [*ssh_cmd, "bash", "-s"],
            input_text=script,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        log(f"Remote bootstrap failed with exit code {exc.returncode}")
        return exc.returncode or 1, None

    if remote_result.stdout:
        print(remote_result.stdout, end="")
    if remote_result.stderr:
        print(remote_result.stderr, file=sys.stderr, end="")

    # For RunPod: use proxy URL since cloudflared won't be available
    worker_url = extract_worker_url(remote_result.stdout)
    if not worker_url and pod_id:
        worker_url = _runpod_proxy_url(pod_id, args.worker_port)
        log(f"Using RunPod proxy URL: {worker_url}")
        print(f"WORKER_URL={worker_url}")
    if not worker_url and not pod_id and args.backend_env.exists():
        # SSH-mode deploy: try to infer pod_id from existing GPU_WORKER_BASE_URL in .env
        # e.g. https://kafmkv5qfjjdqm-9000.proxy.runpod.net → pod_id=kafmkv5qfjjdqm
        existing = args.backend_env.read_text()
        m = re.search(r"GPU_WORKER_BASE_URL=https://([\w]+)-\d+\.proxy\.runpod\.net", existing)
        if m:
            pod_id = m.group(1)
            worker_url = _runpod_proxy_url(pod_id, args.worker_port)
            log(f"Inferred RunPod proxy URL from existing .env: {worker_url}")
            print(f"WORKER_URL={worker_url}")

    if worker_url and not args.skip_backend_env and args.backend_env.exists():
        update_env_file(args.backend_env, worker_url)
        log(f"Updated {args.backend_env} with GPU_WORKER_BASE_URL={worker_url}")
    elif worker_url and not args.skip_backend_env:
        log(f"Backend env not found at {args.backend_env}; skipped env update.")
    elif not worker_url:
        log("No public worker URL found; skipped backend env update.")

    if worker_url and not args.skip_backend_restart:
        should_restart = backend_is_running(args.backend_root) or args.start_backend
        if should_restart and args.backend_root.exists():
            restart_backend(args.backend_root)
            log("Local backend restarted on http://127.0.0.1:8000")
        else:
            log("Local backend is not running; skipped restart.")
    elif not worker_url and not args.skip_backend_restart:
        log("No public worker URL found; skipped backend restart.")

    if worker_url:
        log("Worker registration is handled by FILMFORGE_BACKEND_URL / render broker heartbeat.")

    # Save SSH dest so --logs can reconnect without re-specifying the host
    ssh_dest = args.ssh_command
    if ssh_dest.startswith("ssh "):
        ssh_dest = ssh_dest[4:]
    (SCRIPT_DIR / ".last_ssh_dest").write_text(ssh_dest + "\n")

    return 0, worker_url or None


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "ssh_command",
        nargs="?",
        default=None,
        help="Full SSH command, e.g.: ssh -i ~/.ssh/id_ed25519 -p 22981 root@61.206.39.5",
    )
    mode.add_argument(
        "--runpod",
        action="store_true",
        help=(
            "Fully automated RunPod deploy: find or create a pod, wait for SSH, "
            "deploy the worker, and print the public worker URL."
        ),
    )
    mode.add_argument(
        "--vast",
        action="store_true",
        help=(
            "Fully automated Vast deploy: rent a Vast instance, wait for SSH, "
            "deploy the worker, optionally warm asset groups, and print the public worker URL."
        ),
    )
    mode.add_argument(
        "--verda",
        action="store_true",
        help=(
            "Fully automated Verda rehydrate: create a VM from an existing OS volume, "
            "attach the model data volume, configure systemd workers, and print worker URLs."
        ),
    )
    mode.add_argument(
        "--verda-fresh",
        action="store_true",
        help=(
            "Fully automated fresh Verda deploy: create OS/storage volumes, install ComfyUI "
            "and gpu_worker, download models, and print worker URLs."
        ),
    )

    parser.add_argument(
        "--gpu",
        default=None,
        metavar="GPU_TYPE_ID",
        help=(
            f"RunPod GPU type ID to use when creating a new pod. "
            f"Default: {RUNPOD_GPU_TYPE}. "
            "Run: python3 -c \"import runpod; [print(g['id']) for g in runpod.get_gpus()]\" to list all."
        ),
    )
    parser.add_argument(
        "--pod-id",
        default=None,
        metavar="POD_ID",
        help="Target a specific existing RunPod pod by ID, bypassing name lookup.",
    )
    parser.add_argument(
        "--runpod-volume-gb",
        type=int,
        default=150,
        help="RunPod persistent volume size in GB (mounted at /workspace). Default: 150",
    )
    parser.add_argument(
        "--runpod-container-disk-gb",
        type=int,
        default=50,
        help="RunPod container (ephemeral) disk size in GB. Default: 50",
    )
    parser.add_argument(
        "--runpod-cloud-type",
        default="COMMUNITY",
        choices=["SECURE", "COMMUNITY"],
        help="RunPod cloud type. COMMUNITY is cheaper and has better availability. Default: COMMUNITY",
    )
    parser.add_argument(
        "--ssh-identity",
        type=Path,
        default=None,
        metavar="KEY_FILE",
        help=(
            "SSH private key (e.g. ~/.ssh/id_ed25519). "
            f"Auto-detected from {DEFAULT_SSH_IDENTITY} or {DEFAULT_SSH_IDENTITY_RUNPOD} if omitted."
        ),
    )
    parser.add_argument(
        "--vast-gpu",
        default=DEFAULT_VAST_GPU,
        help=f"Preferred Vast GPU name substring. Default: {DEFAULT_VAST_GPU}",
    )
    parser.add_argument(
        "--vast-max-price",
        type=float,
        default=DEFAULT_VAST_MAX_PRICE,
        help=f"Maximum Vast hourly price. Default: {DEFAULT_VAST_MAX_PRICE}",
    )
    parser.add_argument(
        "--vast-max-upload-cost",
        type=float,
        default=None,
        help="Optional Vast upload bandwidth cost cap in $/GB.",
    )
    parser.add_argument(
        "--vast-max-download-cost",
        type=float,
        default=None,
        help="Optional Vast download bandwidth cost cap in $/GB.",
    )
    parser.add_argument(
        "--vast-allow-fallback-gpu",
        action="store_true",
        help="Allow Vast selection to fall back to a different GPU model when no exact match is found.",
    )
    parser.add_argument(
        "--vast-min-vram-gb",
        type=int,
        default=DEFAULT_VAST_MIN_VRAM_GB,
        help=f"Minimum Vast VRAM in GB. Default: {DEFAULT_VAST_MIN_VRAM_GB}",
    )
    parser.add_argument(
        "--vast-disk-gb",
        type=int,
        default=200,
        help="Vast instance disk size in GB. Default: 200",
    )
    parser.add_argument(
        "--vast-limit",
        type=int,
        default=DEFAULT_VAST_LIMIT,
        help=f"How many Vast offers to inspect. Default: {DEFAULT_VAST_LIMIT}",
    )
    parser.add_argument(
        "--vast-image",
        default=DEFAULT_VAST_IMAGE,
        help=f"Vast image to rent. Default: {DEFAULT_VAST_IMAGE}",
    )
    parser.add_argument(
        "--vast-template-hash",
        default=None,
        help="Optional Vast template hash to use for instance creation.",
    )
    parser.add_argument(
        "--vast-boot-timeout",
        type=int,
        default=DEFAULT_VAST_BOOT_TIMEOUT,
        help=f"Seconds to wait for Vast SSH readiness. Default: {DEFAULT_VAST_BOOT_TIMEOUT}",
    )
    parser.add_argument(
        "--verda-cli",
        type=Path,
        default=DEFAULT_VERDA_CLI,
        help=f"Path to Verda CLI. Default: {DEFAULT_VERDA_CLI}",
    )
    parser.add_argument(
        "--verda-location",
        default=os.getenv("VERDA_LOCATION", DEFAULT_VERDA_LOCATION),
        help=f"Verda location code. Default: {DEFAULT_VERDA_LOCATION}",
    )
    parser.add_argument(
        "--verda-instance-type",
        default=os.getenv("VERDA_INSTANCE_TYPE", DEFAULT_VERDA_INSTANCE_TYPE),
        help=f"Verda instance type. Default: {DEFAULT_VERDA_INSTANCE_TYPE}",
    )
    parser.add_argument(
        "--verda-contract",
        choices=("pay_as_go", "spot"),
        default=os.getenv("VERDA_CONTRACT", DEFAULT_VERDA_CONTRACT),
        help="Verda billing contract. Use spot for disposable install/model-download tests. Default: pay_as_go",
    )
    parser.add_argument(
        "--verda-os-volume-id",
        default=os.getenv("VERDA_OS_VOLUME_ID", DEFAULT_VERDA_OS_VOLUME_ID),
        help="Detached Verda OS volume ID to boot from.",
    )
    parser.add_argument(
        "--verda-data-volume-id",
        default=os.getenv("VERDA_DATA_VOLUME_ID", DEFAULT_VERDA_DATA_VOLUME_ID),
        help="Detached Verda data/model volume ID to attach.",
    )
    parser.add_argument(
        "--verda-ssh-key-id",
        default=os.getenv("VERDA_SSH_KEY_ID", DEFAULT_VERDA_SSH_KEY_ID),
        help="Verda SSH key ID to inject into the instance.",
    )
    parser.add_argument(
        "--verda-hostname",
        default=os.getenv("VERDA_HOSTNAME", DEFAULT_VERDA_HOSTNAME),
        help=f"Hostname for the Verda VM. Default: {DEFAULT_VERDA_HOSTNAME}",
    )
    parser.add_argument(
        "--verda-comfy-port",
        type=int,
        default=int(os.getenv("VERDA_COMFY_PORT", "8188")),
        help="Base ComfyUI port for Verda multi-GPU workers. Default: 8188",
    )
    parser.add_argument(
        "--verda-worker-count",
        type=int,
        default=int(os.getenv("VERDA_WORKER_COUNT", "0")),
        help="Number of GPU worker services to start. 0 means auto-detect all GPUs.",
    )
    parser.add_argument(
        "--verda-fresh-os-image",
        default=os.getenv("VERDA_FRESH_OS_IMAGE", DEFAULT_VERDA_OS_IMAGE),
        help=f"Base Verda OS image for --verda-fresh. Default: {DEFAULT_VERDA_OS_IMAGE}",
    )
    parser.add_argument(
        "--verda-fresh-os-volume-size",
        type=int,
        default=int(os.getenv("VERDA_FRESH_OS_VOLUME_SIZE", str(DEFAULT_VERDA_FRESH_OS_VOLUME_SIZE))),
        help=f"Fresh OS volume size in GiB. Default: {DEFAULT_VERDA_FRESH_OS_VOLUME_SIZE}",
    )
    parser.add_argument(
        "--verda-fresh-os-volume-name",
        default=os.getenv("VERDA_FRESH_OS_VOLUME_NAME", ""),
        help="Fresh OS volume name. Default: <hostname>-os",
    )
    parser.add_argument(
        "--verda-fresh-storage-size",
        type=int,
        default=int(os.getenv("VERDA_FRESH_STORAGE_SIZE", str(DEFAULT_VERDA_FRESH_STORAGE_SIZE))),
        help=f"Fresh model/data storage size in GiB. Default: {DEFAULT_VERDA_FRESH_STORAGE_SIZE}",
    )
    parser.add_argument(
        "--verda-fresh-storage-name",
        default=os.getenv("VERDA_FRESH_STORAGE_NAME", ""),
        help="Fresh model/data storage name. Default: <hostname>-models",
    )
    parser.add_argument(
        "--verda-fresh-storage-type",
        default=os.getenv("VERDA_FRESH_STORAGE_TYPE", "NVMe"),
        help="Fresh model/data storage type. Default: NVMe",
    )
    parser.add_argument(
        "--verda-worker-repo-url",
        default=os.getenv("VERDA_WORKER_REPO_URL", DEFAULT_WORKER_REPO_URL),
        help=f"gpu_worker git URL for --verda-fresh. Default: {DEFAULT_WORKER_REPO_URL}",
    )
    parser.add_argument(
        "--verda-comfy-repo-url",
        default=os.getenv("VERDA_COMFY_REPO_URL", DEFAULT_COMFY_REPO_URL),
        help=f"ComfyUI git URL for --verda-fresh. Default: {DEFAULT_COMFY_REPO_URL}",
    )
    parser.add_argument(
        "--verda-pytorch-index-url",
        default=os.getenv("VERDA_PYTORCH_INDEX_URL", DEFAULT_PYTORCH_INDEX_URL),
        help=f"PyTorch wheel index for --verda-fresh. Default: {DEFAULT_PYTORCH_INDEX_URL}",
    )
    parser.add_argument(
        "--verda-wait-timeout",
        default=os.getenv("VERDA_WAIT_TIMEOUT", "8m"),
        help="Verda VM create wait timeout. Default: 8m",
    )
    parser.add_argument(
        "--verda-create-timeout",
        type=int,
        default=int(os.getenv("VERDA_CREATE_TIMEOUT_SEC", "900")),
        help="Local timeout in seconds for the Verda create command. Default: 900",
    )
    parser.add_argument(
        "--verda-ssh-timeout",
        type=int,
        default=int(os.getenv("VERDA_SSH_TIMEOUT_SEC", "300")),
        help="Seconds to wait for the Verda VM SSH service. Default: 300",
    )
    parser.add_argument(
        "--remote-root",
        default=DEFAULT_REMOTE_ROOT,
        help=f"Remote install root. Default: {DEFAULT_REMOTE_ROOT}",
    )
    parser.add_argument(
        "--worker-port",
        type=int,
        default=9000,
        help="Remote local port for the GPU worker. Default: 9000",
    )
    parser.add_argument(
        "--backend-env",
        type=Path,
        default=DEFAULT_BACKEND_ENV,
        help=f"Optional backend .env to update. Default: {DEFAULT_BACKEND_ENV}",
    )
    parser.add_argument(
        "--backend-root",
        type=Path,
        default=DEFAULT_BACKEND_ROOT,
        help=f"Backend repo root for restart detection. Default: {DEFAULT_BACKEND_ROOT}",
    )
    parser.add_argument(
        "--skip-backend-env",
        action="store_true",
        default=True,
        help="Deprecated; backend GPU_WORKER_BASE_URL updates are skipped by default.",
    )
    parser.add_argument(
        "--update-backend-env",
        dest="skip_backend_env",
        action="store_false",
        help="Legacy local-dev mode: write GPU_WORKER_BASE_URL to backend .env.",
    )
    parser.add_argument(
        "--start-backend",
        action="store_true",
        help="Start the local backend even if it is not currently running.",
    )
    parser.add_argument(
        "--skip-backend-restart",
        action="store_true",
        help="Do not restart the local backend after updating env.",
    )
    parser.add_argument(
        "--env",
        dest="env_vars",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable to export before remote bootstrap. Can be repeated.",
    )
    parser.add_argument(
        "--warm-asset-group",
        dest="warm_asset_groups",
        action="append",
        default=[],
        help="Warm an asset group after deploy. Can be provided multiple times.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Do not preload any asset groups after the worker is deployed.",
    )

    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_args()

    try:
        if args.runpod:
            return runpod_deploy(args)
        if args.vast:
            return vast_deploy(args)
        if args.verda:
            return verda_deploy(args)
        if args.verda_fresh:
            return verda_fresh_deploy(args)

        exit_code, worker_url = _do_deploy(args)
        if exit_code == 0 and worker_url and args.warm_asset_groups and not args.skip_warmup:
            _run_worker_warmup(args, worker_url)
        return exit_code
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
