"""
FilmForge GPU Deploy UI
Run: python gpu_worker/deploy_ui.py        (from project root)
  or cd gpu_worker && python deploy_ui.py  (port 7860)
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import urllib.request
import uuid
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

DEFAULT_BACKEND_ENV = SCRIPT_DIR.parent / "filmforge_backend" / "app" / ".env"
PREFERRED_COMFY_IMAGE = "vastai/comfy"
PREFERRED_COMFY_TAG = "v0.15.1-cuda-12.9-py312"

app = FastAPI(title="FilmForge GPU Deploy")

_jobs: dict[str, dict] = {}

REMOTE_LOG_SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    "worker": ("GPU Worker", ("/tmp/gpu_worker.log",)),
    "comfy": ("ComfyUI", ("/tmp/comfyui.log",)),
    "tunnel": ("Tunnel", ("/tmp/filmforge_gpu_worker_tunnel.log",)),
}


# ── Request models ─────────────────────────────────────────────────────────────

class DeployRequest(BaseModel):
    ssh_command: str
    env_vars: dict[str, str] = {}
    worker_port: int = 9000
    remote_root: str = "/workspace/filmforge_gpu_worker"


class CreateInstanceRequest(BaseModel):
    gpu_type: str = "RTX 4090"
    max_price: float = 0.50
    disk_gb: int = 50
    image: str = "vastai/comfy:v0.15.1-cuda-12.9-py312"
    template_hash: str | None = None
    max_upload_cost: float | None = None
    max_download_cost: float | None = None
    allow_fallback_gpu: bool = False


class AutoProvisionRequest(BaseModel):
    gpu_type: str = "RTX 4090"
    max_price: float = 0.75
    min_vram_gb: int = 24
    disk_gb: int = 200
    image: str = "vastai/comfy:v0.15.1-cuda-12.9-py312"
    template_hash: str | None = None
    worker_port: int = 9000
    remote_root: str = "/workspace/filmforge_gpu_worker"
    warm_asset_groups: list[str] = []
    env_vars: dict[str, str] = {}
    max_upload_cost: float | None = None
    max_download_cost: float | None = None
    allow_fallback_gpu: bool = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_vast_raw_output(stdout: str) -> list | dict:
    text = stdout.strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        payload, _ = decoder.raw_decode(text)
        return payload


def _vastai(*args: str) -> list | dict:
    try:
        r = subprocess.run(["vastai", *args, "--raw"], capture_output=True, text=True, timeout=30)
        return _parse_vast_raw_output(r.stdout)
    except Exception:
        return []


def _preferred_comfy_templates(templates: list[dict]) -> list[dict]:
    candidates = [
        item for item in templates
        if item.get("name") == "ComfyUI" and item.get("image") == PREFERRED_COMFY_IMAGE
    ]
    if not candidates:
        return []
    automatic = [item for item in candidates if item.get("tag") == "@vastai-automatic-tag"]
    pool = automatic or candidates
    best = sorted(
        pool,
        key=lambda item: (
            -(float(item.get("recent_create_date") or 0.0)),
            -(float(item.get("count_created") or 0.0)),
            -(float(item.get("created_at") or 0.0)),
        ),
    )[0]
    pinned = dict(best)
    pinned["tag"] = PREFERRED_COMFY_TAG
    return [pinned]


def _vast_templates(query: str | None = "comfy") -> list[dict]:
    data = _vastai("search", "templates")
    if isinstance(data, dict):
        items = data.get("templates")
        templates = items if isinstance(items, list) else []
    elif isinstance(data, list):
        templates = data
    else:
        templates = []

    normalized: list[dict] = []
    query_text = (query or "").strip().lower()
    for item in templates:
        if not isinstance(item, dict):
            continue
        hash_id = str(item.get("hash_id") or "").strip()
        name = str(item.get("name") or "").strip()
        if not hash_id or not name:
            continue
        desc = str(item.get("desc") or "").strip()
        image = str(item.get("image") or "").strip()
        repo = str(item.get("repo") or "").strip()
        tag = str(item.get("tag") or item.get("default_tag") or "").strip()
        if query_text:
            haystack = " ".join(part for part in (name, desc, image, repo, tag) if part).lower()
            if query_text not in haystack:
                continue
        normalized.append({
            "name": name, "hash_id": hash_id, "image": image, "tag": tag,
            "desc": desc, "repo": repo,
            "recent_create_date": item.get("recent_create_date"),
            "count_created": item.get("count_created"),
            "created_at": item.get("created_at"),
        })
    if query_text == "comfy":
        preferred = _preferred_comfy_templates(normalized)
        if preferred:
            return preferred
    normalized.sort(key=lambda item: item["name"].lower())
    return normalized


def _offer_transfer_cost(offer: dict) -> float:
    return float(offer.get("inet_up_cost") or 0.0) + float(offer.get("inet_down_cost") or 0.0)


def _offer_sort_key(offer: dict) -> tuple[float, float, float]:
    return (
        _offer_transfer_cost(offer),
        float(offer.get("dph_total") or offer.get("dph") or 9999.0),
        -float(offer.get("reliability2") or 0.0),
    )


def _put(job: dict, msg: str) -> None:
    job["logs"].append(msg)
    job["queue"].put(msg)


def _wait_for_ssh_ready(ssh_cmd: list[str], job: dict, timeout: int = 180) -> bool:
    """Poll until SSH accepts a test command. Returns True on success."""
    import time
    deadline = time.time() + timeout
    # Don't add BatchMode=yes — it blocks SSH agent auth which Vast.ai relies on.
    # Insert ConnectTimeout before the host (ssh_cmd already ends with the destination).
    probe = [ssh_cmd[0], "-o", "ConnectTimeout=10", *ssh_cmd[1:], "echo ok"]
    while time.time() < deadline:
        result = subprocess.run(probe, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        reason = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no response"
        _put(job, f"[deploy] SSH not ready yet ({reason}), retrying in 10s…")
        time.sleep(10)
    return False


def _extract_ssh_command(text: str) -> str:
    for line in text.splitlines():
        if "Resolved Vast SSH command for instance" in line:
            _, _, command = line.partition(":")
            command = command.strip()
            if command.startswith("ssh "):
                return command
    return ""


def _last_ssh_command() -> str:
    path = SCRIPT_DIR / ".last_ssh_dest"
    if not path.exists():
        return ""
    dest = path.read_text().strip()
    return f"ssh {dest}" if dest else ""


def _remote_tail_script(paths: tuple[str, ...], label: str) -> str:
    quoted_paths = " ".join(shlex.quote(path) for path in paths)
    primary = shlex.quote(paths[0])
    quoted_label = shlex.quote(label)
    return f"""set -euo pipefail
paths=({quoted_paths})
found=""
for path in "${{paths[@]}}"; do
  if test -e "$path"; then
    found="$path"
    break
  fi
done
if test -z "$found"; then
  found={primary}
  mkdir -p "$(dirname "$found")" 2>/dev/null || true
  touch "$found" 2>/dev/null || true
  echo "[remote-log] waiting for {quoted_label} log at $found"
fi
echo "[remote-log] tailing {quoted_label}: $found"
tail -n 200 -F "$found"
"""


def _remote_comfy_script() -> str:
    return r"""set -euo pipefail
candidates=(
  /tmp/comfyui.log
  /var/log/supervisor/comfyui.log
  /var/log/supervisor/comfyui-stdout---supervisor-*.log
  /workspace/ComfyUI/comfyui.log
  /workspace/runpod-slim/ComfyUI/comfyui.log
)

for pattern in "${candidates[@]}"; do
  for path in $pattern; do
    if test -e "$path"; then
      echo "[comfy] tailing file: $path"
      exec tail -n 200 -F "$path"
    fi
  done
done

if command -v supervisorctl >/dev/null 2>&1; then
  if supervisorctl status comfyui >/tmp/filmforge_comfy_supervisor_status.txt 2>/dev/null; then
    echo "[comfy] streaming supervisor logs for comfyui"
    exec bash -lc 'supervisorctl tail -f comfyui stdout 2>/dev/null || supervisorctl tail -f comfyui 2>/dev/null'
  fi
fi

if command -v docker >/dev/null 2>&1; then
  container_id="$(docker ps --format '{{.ID}} {{.Image}} {{.Names}}' \
    | grep -iE 'comfy|runpod|vastai/comfy' \
    | head -n 1 \
    | awk '{print $1}')"
  if test -n "$container_id"; then
    echo "[comfy] streaming docker logs for container $container_id"
    exec docker logs --tail 200 -f "$container_id" 2>&1
  fi
fi

echo "[comfy] No ComfyUI log source found."
echo "[comfy] Checked: /tmp/comfyui.log, supervisor comfyui logs, and docker container logs."
echo "[comfy] Running Comfy-related processes:"
ps -ef | grep -i comfy | grep -v grep || true
sleep 2
"""


def _remote_downloads_script() -> str:
    return r"""set -euo pipefail
worker_log=/tmp/gpu_worker.log
mkdir -p "$(dirname "$worker_log")" 2>/dev/null || true
touch "$worker_log" 2>/dev/null || true
echo "[downloads] recent worker download lines"
grep -iE "download|asset|aria2|model|ensure|failed|error" "$worker_log" 2>/dev/null | tail -n 120 || true
(tail -n 0 -F "$worker_log" 2>/dev/null \
  | grep --line-buffered -iE "download|asset|aria2|model|ensure|failed|error" \
  | sed 's/^/[downloads] log /') &
tail_pid=$!
trap 'kill "$tail_pid" 2>/dev/null || true' EXIT
while true; do
  echo "[downloads] $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  if pgrep -fa aria2c >/tmp/filmforge_aria2c_processes.txt 2>/dev/null; then
    sed 's/^/[downloads] active /' /tmp/filmforge_aria2c_processes.txt
  else
    echo "[downloads] no active aria2c process"
  fi
  find /workspace/ComfyUI/models -type f \( -name "*.part" -o -name "*.aria2" -o -name "*.safetensors" -o -name "*.ckpt" -o -name "*.pt" \) \
    -printf "%T@ %TY-%Tm-%TdT%TH:%TM:%TSZ %s %p\n" 2>/dev/null \
    | sort -nr \
    | head -n 20 \
    | awk '{ts=$2; size=$3; $1=$2=$3=""; sub(/^   /,""); printf("[downloads] file %s bytes=%s path=%s\n", ts, size, $0)}'
  sleep 5
done
"""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML)


@app.get("/api/backend-env")
async def backend_env():
    if not DEFAULT_BACKEND_ENV.exists():
        return {}
    expose = {
        "FILMFORGE_BACKEND_URL",
        "RENDER_BROKER_BASE_URL", "RENDER_BROKER_WORKER_TOKEN",
        "WORKER_REGISTRATION_TOKEN", "WORKER_API_TOKEN",
    }
    result: dict[str, str] = {}
    for line in DEFAULT_BACKEND_ENV.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            if k.strip() in expose:
                result[k.strip()] = v.strip()
    return result


@app.get("/api/instances")
async def list_instances():
    data = _vastai("show", "instances")
    return data if isinstance(data, list) else []


@app.get("/api/templates")
async def list_templates(q: str = "comfy"):
    return _vast_templates(q)


@app.post("/api/instances")
async def create_instance(req: CreateInstanceRequest):
    try:
        query_parts = [f"gpu_name={req.gpu_type}", f"dph<{req.max_price}", "rentable=True"]
        if req.max_upload_cost is not None:
            query_parts.append(f"inet_up_cost<={req.max_upload_cost}")
        if req.max_download_cost is not None:
            query_parts.append(f"inet_down_cost<={req.max_download_cost}")
        offers = _vastai("search", "offers", " ".join(query_parts))
        if not isinstance(offers, list) or not offers:
            raise HTTPException(400, f"No offers found for '{req.gpu_type}' under ${req.max_price}/hr")
        exact = [o for o in offers if req.gpu_type.strip().lower() in str(o.get("gpu_name") or "").lower()]
        if exact:
            best = sorted(exact, key=_offer_sort_key)[0]
        elif req.allow_fallback_gpu:
            best = sorted(offers, key=_offer_sort_key)[0]
        else:
            raise HTTPException(400, f"No exact offers found for '{req.gpu_type}'.")
        command = ["vastai", "create", "instance", str(best["id"])]
        if req.template_hash:
            command.extend(["--template_hash", req.template_hash])
        if req.image:
            command.extend(["--image", req.image])
        command.extend(["--disk", str(req.disk_gb), "--ssh", "--direct"])
        r = subprocess.run(command, capture_output=True, text=True, timeout=60)
        return {"ok": r.returncode == 0, "output": (r.stdout + r.stderr).strip(), "offer": best}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/instances/{instance_id}/start")
async def start_instance(instance_id: str):
    r = subprocess.run(["vastai", "start", "instance", instance_id],
                       capture_output=True, text=True, timeout=30)
    return {"ok": r.returncode == 0, "output": (r.stdout + r.stderr).strip()}


@app.delete("/api/instances/{instance_id}")
async def destroy_instance(instance_id: str):
    r = subprocess.run(["vastai", "destroy", "instance", instance_id],
                       capture_output=True, text=True, timeout=30)
    return {"ok": r.returncode == 0, "output": (r.stdout + r.stderr).strip()}


@app.get("/api/workers")
async def list_workers():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/api/render-broker/workers", timeout=3) as r:
            payload = json.loads(r.read())
            return payload.get("items", []) if isinstance(payload, dict) else payload
    except Exception:
        return []


@app.post("/api/deploy")
async def start_deploy(req: DeployRequest):
    job_id = str(uuid.uuid4())
    q: queue.Queue[str | None] = queue.Queue()
    _jobs[job_id] = {"status": "running", "logs": [], "queue": q, "worker_url": None}

    def _run() -> None:
        job = _jobs[job_id]
        try:
            from gpu_worker.deploy_gpu import (
                parse_ssh_command, add_default_identity, add_default_host_key_policy,
                stage_worker_tree, remote_script, extract_worker_url,
                SCRIPT_DIR as GPU_SCRIPT_DIR,
            )
            _put(job, "[deploy] Parsing SSH command...")
            try:
                ssh_cmd, scp_cmd, destination = parse_ssh_command(req.ssh_command)
            except ValueError as exc:
                _put(job, f"[deploy] ERROR: {exc}")
                job["status"] = "failed"
                return

            ssh_cmd = add_default_host_key_policy(add_default_identity(ssh_cmd))
            scp_cmd = add_default_host_key_policy(add_default_identity(scp_cmd))
            _put(job, f"[deploy] Waiting for SSH on {destination}…")
            if not _wait_for_ssh_ready(ssh_cmd, job, timeout=180):
                _put(job, "[deploy] ERROR: timed out waiting for SSH to become ready")
                job["status"] = "failed"
                return
            _put(job, "[deploy] SSH ready — copying worker files…")
            try:
                subprocess.run([*ssh_cmd, "mkdir", "-p", req.remote_root], check=True, capture_output=True)
                subprocess.run([*ssh_cmd, "rm", "-rf", f"{req.remote_root.rstrip('/')}/{GPU_SCRIPT_DIR.name}"],
                               check=True, capture_output=True)
                with stage_worker_tree(GPU_SCRIPT_DIR) as staged_dir:
                    staged = Path(staged_dir) / GPU_SCRIPT_DIR.name
                    subprocess.run([*scp_cmd, "-r", str(staged), f"{destination}:{req.remote_root}/"],
                                   check=True, capture_output=True)
            except subprocess.CalledProcessError as exc:
                stderr = getattr(exc, "stderr", "") or ""
                _put(job, f"[deploy] ERROR copying files: {stderr or exc}")
                job["status"] = "failed"
                return

            env_exports = "\n".join(
                f"export {k}={shlex.quote(str(v))}" for k, v in req.env_vars.items() if k and v
            )
            script = remote_script(req.remote_root, req.worker_port)
            if env_exports:
                script = env_exports + "\n\n" + script

            _put(job, "[deploy] Running bootstrap — this takes ~2 min...")
            proc = subprocess.Popen(
                [*ssh_cmd, "bash", "-s"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            assert proc.stdin and proc.stdout
            proc.stdin.write(script)
            proc.stdin.close()

            stdout_lines: list[str] = []
            for line in proc.stdout:
                line = line.rstrip()
                stdout_lines.append(line)
                _put(job, line)
            proc.wait()

            if proc.returncode != 0:
                _put(job, f"[deploy] Bootstrap failed (exit {proc.returncode})")
                job["status"] = "failed"
                return

            stdout_text = "\n".join(stdout_lines)
            worker_url = extract_worker_url(stdout_text)

            if worker_url:
                _put(job, f"[deploy] Worker URL: {worker_url}")
                _put(job, "[deploy] Worker registration is handled by FILMFORGE_BACKEND_URL / render broker heartbeat.")
                job["worker_url"] = worker_url
            else:
                _put(job, "[deploy] ERROR: no public worker URL found")
                job["status"] = "failed"
                return

            dest_str = req.ssh_command.removeprefix("ssh ").strip()
            (SCRIPT_DIR / ".last_ssh_dest").write_text(dest_str + "\n")
            job["status"] = "done"
            _put(job, "[deploy] ✓ Deploy complete")
        except Exception as exc:
            _put(job, f"[deploy] ERROR: {exc}")
            job["status"] = "failed"
        finally:
            job["queue"].put(None)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/provision-vast")
async def start_vast_provision(req: AutoProvisionRequest):
    job_id = str(uuid.uuid4())
    q: queue.Queue[str | None] = queue.Queue()
    _jobs[job_id] = {"status": "running", "logs": [], "queue": q, "worker_url": None, "ssh_command": None}

    def _run() -> None:
        job = _jobs[job_id]
        try:
            from gpu_worker.deploy_gpu import extract_worker_url
            command = [
                sys.executable, "-u", str(SCRIPT_DIR / "deploy_gpu.py"),
                "--vast",
                "--vast-gpu", req.gpu_type,
                "--vast-max-price", str(req.max_price),
                "--vast-min-vram-gb", str(req.min_vram_gb),
                "--vast-disk-gb", str(req.disk_gb),
                "--vast-image", req.image,
                "--worker-port", str(req.worker_port),
                "--remote-root", req.remote_root,
            ]
            if req.template_hash:
                command.extend(["--vast-template-hash", req.template_hash])
            if req.max_upload_cost is not None:
                command.extend(["--vast-max-upload-cost", str(req.max_upload_cost)])
            if req.max_download_cost is not None:
                command.extend(["--vast-max-download-cost", str(req.max_download_cost)])
            if req.allow_fallback_gpu:
                command.append("--vast-allow-fallback-gpu")
            command.append("--skip-warmup")
            for key, value in req.env_vars.items():
                if key and value:
                    command.extend(["--env", f"{key}={value}"])

            _put(job, "[vast] Renting instance, deploying worker, and warming models...")
            _put(job, "[vast] GPU Worker log will open automatically when deploy completes.")
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            assert proc.stdout is not None
            output_lines: list[str] = []
            for line in proc.stdout:
                line = line.rstrip()
                output_lines.append(line)
                _put(job, line)
            proc.wait()
            output_text = "\n".join(output_lines)
            job["worker_url"] = extract_worker_url(output_text) or None
            job["ssh_command"] = _extract_ssh_command(output_text) or None

            if proc.returncode == 0:
                job["status"] = "done"
                _put(job, "[vast] Provision complete")
            else:
                job["status"] = "failed"
                _put(job, f"[vast] Provision failed (exit {proc.returncode})")
        except Exception as exc:
            _put(job, f"[vast] ERROR: {exc}")
            job["status"] = "failed"
        finally:
            job["queue"].put(None)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/deploy/{job_id}/stream")
async def stream_logs(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]

    async def generate() -> AsyncGenerator[str, None]:
        q = job["queue"]
        if job["status"] != "running":
            for line in job["logs"]:
                yield f"data: {json.dumps(line)}\n\n"
            yield "data: __DONE__\n\n"
            return
        while True:
            while True:
                try:
                    item = q.get_nowait()
                    if item is None:
                        yield "data: __DONE__\n\n"
                        return
                    yield f"data: {json.dumps(item)}\n\n"
                except queue.Empty:
                    break
            await asyncio.sleep(0.2)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/deploy/{job_id}")
async def deploy_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404)
    j = _jobs[job_id]
    return {"status": j["status"], "worker_url": j.get("worker_url"), "ssh_command": j.get("ssh_command")}


@app.get("/api/remote-logs/{source}/stream")
async def stream_remote_log(source: str):
    if source != "downloads" and source not in REMOTE_LOG_SOURCES:
        raise HTTPException(404, "Unknown log source")
    label, paths = REMOTE_LOG_SOURCES.get(source, ("Downloads", ()))

    async def generate() -> AsyncGenerator[str, None]:
        ssh_command = _last_ssh_command()
        if not ssh_command:
            yield f"data: {json.dumps('[remote-log] No saved SSH destination yet. Run a deploy first.')}\n\n"
            yield "data: __DONE__\n\n"
            return
        try:
            from gpu_worker.deploy_gpu import add_default_host_key_policy, add_default_identity, parse_ssh_command
            ssh_cmd, _, _ = parse_ssh_command(ssh_command)
            ssh_cmd = add_default_host_key_policy(add_default_identity(ssh_cmd))
            ssh_cmd = [ssh_cmd[0], "-q", "-o", "LogLevel=ERROR", *ssh_cmd[1:]]
        except Exception as exc:
            yield f"data: {json.dumps(f'[remote-log] ERROR: {exc}')}\n\n"
            yield "data: __DONE__\n\n"
            return

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd, "bash", "-s",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            assert proc.stdin is not None and proc.stdout is not None
            script = (
                _remote_downloads_script() if source == "downloads"
                else _remote_comfy_script() if source == "comfy"
                else _remote_tail_script(paths, label)
            )
            proc.stdin.write(script.encode())
            await proc.stdin.drain()
            proc.stdin.close()
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                yield f"data: {json.dumps(line.decode(errors='replace').rstrip())}\n\n"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield f"data: {json.dumps(f'[remote-log] ERROR: {exc}')}\n\n"
        finally:
            if proc and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── HTML ───────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FilmForge GPU Deploy</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0f1117;color:#e2e8f0;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* Header */
header{background:#1a1d2e;border-bottom:1px solid #2d3148;padding:12px 20px;display:flex;align-items:center;gap:10px;flex-shrink:0}
header h1{font-size:15px;font-weight:600;color:#a78bfa}

/* Layout */
.layout{display:flex;flex:1;overflow:hidden}
.left{width:360px;flex-shrink:0;border-right:1px solid #2d3148;display:flex;flex-direction:column;overflow-y:auto}
.right{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}

/* Section */
.sec{padding:14px 16px;border-bottom:1px solid #2d3148}
.sec-hdr{display:flex;align-items:center;gap:8px;margin-bottom:12px}
.sec-title{font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.08em;flex:1}

/* Buttons */
.btn{padding:7px 14px;border-radius:5px;border:none;cursor:pointer;font-size:12px;font-weight:500;transition:opacity .15s}
.btn:hover{opacity:.82}.btn:disabled{opacity:.35;cursor:not-allowed}
.btn-xs{padding:4px 10px;font-size:11px;border-radius:4px}
.btn-primary{background:#7c3aed;color:#fff}
.btn-success{background:#059669;color:#fff}
.btn-danger{background:#dc2626;color:#fff}
.btn-ghost{background:#252a38;color:#94a3b8;border:1px solid #2d3148}
.btn-ghost:hover{color:#e2e8f0;border-color:#475569}
.btn-full{width:100%}

/* Instance cards */
.inst-card{background:#13161f;border:1px solid #252a38;border-radius:8px;padding:12px 14px;margin-bottom:8px;transition:border-color .15s}
.inst-card:last-child{margin-bottom:0}
.inst-card:hover{border-color:#3d4461}
.inst-top{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.inst-id{font-family:monospace;font-size:11px;color:#475569}
.inst-gpu{font-size:13px;font-weight:600;color:#e2e8f0;flex:1}
.inst-meta{font-size:11px;color:#475569;margin-bottom:10px;font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.inst-actions{display:flex;gap:6px}

/* Status badge */
.badge{display:inline-flex;align-items:center;gap:5px;padding:2px 8px;border-radius:9999px;font-size:10px;font-weight:600}
.dot{width:5px;height:5px;border-radius:50%;flex-shrink:0}
.b-running{background:#0d2b1e;color:#34d399;border:1px solid #1a4d35}.b-running .dot{background:#34d399}
.b-exited{background:#1c1f2a;color:#64748b;border:1px solid #2d3148}.b-exited .dot{background:#475569}
.b-loading{background:#0f2744;color:#60a5fa;border:1px solid #1a3a5f}.b-loading .dot{background:#60a5fa;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* Forms */
label{display:block;font-size:10px;font-weight:600;color:#475569;margin-bottom:4px;text-transform:uppercase;letter-spacing:.06em}
input,select{width:100%;padding:7px 10px;background:#0c0f16;border:1px solid #2d3148;border-radius:5px;color:#e2e8f0;font-size:12px;font-family:inherit;transition:border-color .15s}
input:focus,select:focus{outline:none;border-color:#7c3aed}
.fg{margin-bottom:10px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}

/* Collapsible */
.collapse-btn{display:flex;align-items:center;gap:7px;width:100%;padding:10px 16px;background:none;border:none;border-top:1px solid #2d3148;color:#475569;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;cursor:pointer;transition:color .15s;text-align:left}
.collapse-btn:hover{color:#94a3b8}
.carrow{transition:transform .2s;font-style:normal;font-size:9px;line-height:1}
.collapse-btn.open .carrow{transform:rotate(90deg)}
.collapse-body{display:none;padding:14px 16px;border-top:1px solid #1e2130}
.collapse-body.open{display:block}

/* Advanced toggle (inline link style) */
.adv-toggle{background:none;border:none;color:#475569;font-size:11px;cursor:pointer;padding:0;text-decoration:underline;text-underline-offset:2px}
.adv-toggle:hover{color:#94a3b8}
#adv-body{display:none;margin-top:12px;padding-top:12px;border-top:1px solid #1e2130}

/* Env rows */
.env-row{display:grid;grid-template-columns:1fr 1fr 26px;gap:6px;margin-bottom:6px;align-items:center}
.env-row input{margin-bottom:0}
.rm-btn{width:26px;height:30px;background:#1c1f2a;border:none;border-radius:4px;color:#64748b;cursor:pointer;font-size:14px;line-height:1;transition:color .15s}
.rm-btn:hover{color:#f87171}

/* Log area */
.log-area{flex:1;display:flex;flex-direction:column;overflow:hidden}
.log-hdr{display:flex;align-items:center;gap:6px;padding:10px 16px;border-bottom:1px solid #2d3148;flex-shrink:0;flex-wrap:wrap;gap:8px}
.log-tabs{display:flex;gap:3px}
.ltab{padding:4px 10px;border-radius:4px;font-size:11px;cursor:pointer;background:none;border:none;color:#475569;transition:all .15s}
.ltab:hover{color:#94a3b8}
.ltab.active{background:#252a38;color:#e2e8f0}
#sbar{font-size:11px;padding:3px 10px;border-radius:4px;display:none;margin-left:auto}
#sbar.running{background:#0f2744;color:#93c5fd;display:inline-block}
#sbar.done{background:#0d2b1e;color:#6ee7b7;display:inline-block}
#sbar.failed{background:#2d1010;color:#fca5a5;display:inline-block}
#log{flex:1;overflow-y:auto;padding:12px 16px;font-family:'Courier New',monospace;font-size:11.5px;line-height:1.7;background:#080b14;min-height:0}
.ll{color:#3d4461}
.ll.info{color:#60a5fa}.ll.ok{color:#475569}.ll.signal{color:#34d399}.ll.err{color:#f87171}.ll.warn{color:#fbbf24}

/* Workers */
.workers-pane{flex-shrink:0;border-top:1px solid #2d3148;max-height:200px;overflow-y:auto}
.workers-hdr{display:flex;align-items:center;padding:8px 16px;background:#0f1117;position:sticky;top:0;border-bottom:1px solid #1e2130}
.wrow{display:flex;align-items:center;gap:10px;padding:8px 16px;border-bottom:1px solid #13161f}
.wrow:last-child{border-bottom:none}
.wname{font-size:12px;color:#e2e8f0;font-weight:500}
.wurl{font-size:10px;color:#3d4461;font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:160px}

.muted{color:#3d4461}.text-xs{font-size:11px}
.flex{display:flex}.gap6{gap:6px}.gap8{gap:8px}.items-center{align-items:center}.flex1{flex:1}
.mt10{margin-top:10px}.mb0{margin-bottom:0}
</style>
</head>
<body>
<header>
  <span>🎬</span>
  <h1>FilmForge GPU Deploy</h1>
</header>

<div class="layout">

  <!-- ── LEFT PANEL ────────────────────────────────────────────────────── -->
  <div class="left">

    <!-- Instances -->
    <div class="sec">
      <div class="sec-hdr">
        <span class="sec-title">Vast.ai Instances</span>
        <button class="btn btn-ghost btn-xs" onclick="loadInstances()">↻ Refresh</button>
      </div>
      <div id="instances-out"><span class="muted text-xs">Loading…</span></div>
    </div>

    <!-- Create & Deploy -->
    <div class="sec">
      <div class="sec-hdr mb0">
        <span class="sec-title mb0">Create &amp; Deploy</span>
      </div>
      <div class="grid2" style="margin-bottom:10px">
        <div class="fg" style="margin-bottom:0">
          <label>GPU</label>
          <select id="c-gpu">
            <option>L40S</option><option>A100 SXM</option><option>A100 PCIe</option>
            <option>H100 PCIe</option><option>H100 SXM</option>
            <option>RTX 6000 Ada</option><option>RTX PRO 6000</option>
            <option>A6000</option><option>RTX 4090</option><option>B200</option>
          </select>
        </div>
        <div class="fg" style="margin-bottom:0">
          <label>Max $/hr</label>
          <input id="c-price" type="number" value="0.75" step="0.05">
        </div>
      </div>
      <button class="btn btn-success btn-full" id="provision-btn" onclick="quickDeploy()" style="margin-bottom:8px">
        ⚡ Create &amp; Deploy
      </button>
      <button class="adv-toggle" onclick="toggleAdv()">▶ Advanced options</button>
      <div id="adv-body">
        <div class="grid2">
          <div class="fg"><label>Min VRAM (GB)</label><input id="c-vram" type="number" value="24"></div>
          <div class="fg"><label>Disk (GB)</label><input id="c-disk" type="number" value="200"></div>
        </div>
        <div class="fg"><label>Template Hash</label><input id="c-tmpl" placeholder="Auto-detect ComfyUI template"></div>
        <div class="fg"><label>Docker Image</label><input id="c-image" value="vastai/comfy:v0.15.1-cuda-12.9-py312"></div>
        <div class="grid2">
          <div class="fg"><label>Max Upload $/GB</label><input id="c-up" type="number" step="0.001" placeholder="any"></div>
          <div class="fg"><label>Max Download $/GB</label><input id="c-down" type="number" step="0.001" placeholder="any"></div>
        </div>
        <div class="fg"><label>Worker Port</label><input id="c-port" type="number" value="9000"></div>
        <button class="btn btn-ghost btn-xs" onclick="autoDetectTemplate()" style="margin-bottom:6px">↻ Auto-detect template</button>
        <div id="tmpl-status" class="muted text-xs"></div>
      </div>
    </div>

    <!-- Env Vars (collapsible) -->
    <button class="collapse-btn" id="env-toggle" onclick="toggleEnv()">
      <i class="carrow">▶</i> Env Vars &amp; Settings
    </button>
    <div class="collapse-body" id="env-body">
      <div class="flex gap6 items-center" style="margin-bottom:10px">
        <button class="btn btn-ghost btn-xs" onclick="loadEnvFromBackend()">📂 Load .env</button>
        <button class="btn btn-ghost btn-xs" onclick="addEnvRow()">+ Add Row</button>
      </div>
      <div id="env-table"></div>
      <div class="fg mt10">
        <label>Remote Root</label>
        <input id="remote-root" value="/workspace/filmforge_gpu_worker">
      </div>
    </div>

  </div>

  <!-- ── RIGHT PANEL ───────────────────────────────────────────────────── -->
  <div class="right">

    <!-- Log -->
    <div class="log-area">
      <div class="log-hdr">
        <span class="sec-title" style="flex:none">Log</span>
        <div class="log-tabs">
          <button class="ltab active" data-src="deploy" onclick="selectLog('deploy')">Deploy</button>
          <button class="ltab" data-src="downloads" onclick="selectLog('downloads')">Downloads</button>
          <button class="ltab" data-src="worker" onclick="selectLog('worker')">GPU Worker</button>
          <button class="ltab" data-src="comfy" onclick="selectLog('comfy')">ComfyUI</button>
          <button class="ltab" data-src="tunnel" onclick="selectLog('tunnel')">Tunnel</button>
        </div>
        <span id="sbar"></span>
        <button class="btn btn-ghost btn-xs" style="margin-left:auto" onclick="clearLog()">Clear</button>
      </div>
      <div id="log"><span class="muted text-xs">Ready — create an instance or click Deploy on a running one.</span></div>
    </div>

    <!-- Workers -->
    <div class="workers-pane">
      <div class="workers-hdr">
        <span class="sec-title flex1" style="margin-bottom:0">Registered Workers</span>
        <button class="btn btn-ghost btn-xs" onclick="loadWorkers()">↻</button>
      </div>
      <div id="workers-out"><span class="muted text-xs" style="padding:10px 16px;display:block">Loading…</span></div>
    </div>

  </div>
</div>

<script>
// ── State ─────────────────────────────────────────────────────────────────────
let evtSrc = null;
let currentLogSrc = 'deploy';
const logBuffers = { deploy: [] };
const MAX_LOG_LINES = 2000;

const LOG_LABELS = {
  deploy: 'Deploy', downloads: 'Downloads',
  worker: 'GPU Worker', comfy: 'ComfyUI', tunnel: 'Tunnel',
};

// ── Instances ─────────────────────────────────────────────────────────────────
async function loadInstances() {
  const el = document.getElementById('instances-out');
  el.innerHTML = '<span class="muted text-xs">Loading…</span>';
  try {
    const list = await fetch('/api/instances').then(r => r.json());
    if (!list.length) {
      el.innerHTML = '<p class="muted text-xs">No instances — use Create &amp; Deploy below.</p>';
      return;
    }
    el.innerHTML = list.map(inst => renderInstCard(inst)).join('');
  } catch (e) {
    el.innerHTML = '<p class="muted text-xs" style="color:#f87171">Failed to load instances.</p>';
  }
}

function renderInstCard(inst) {
  const ssh = getSsh(inst);
  const status = (inst.actual_status || '?').toLowerCase();
  const isRunning = status === 'running';
  const isStopped = status === 'exited' || status === 'stopped';
  const isLoading = !isRunning && !isStopped;

  const badgeClass = isRunning ? 'b-running' : isStopped ? 'b-exited' : 'b-loading';
  const price = `$${(inst.dph_total || 0).toFixed(3)}/hr`;
  const meta = ssh ? `${price} · ${ssh}` : price;

  const actions = isRunning
    ? `<button class="btn btn-success btn-xs" onclick='deployToInst(${JSON.stringify(ssh)})'>▶ Deploy</button>
       <button class="btn btn-ghost btn-xs" onclick="destroyInst('${inst.id}')">Destroy</button>`
    : isStopped
    ? `<button class="btn btn-primary btn-xs" onclick="activateInst('${inst.id}')">Activate</button>
       <button class="btn btn-ghost btn-xs" onclick="destroyInst('${inst.id}')">Destroy</button>`
    : `<span class="muted text-xs">${status}…</span>
       <button class="btn btn-ghost btn-xs" onclick="destroyInst('${inst.id}')">Destroy</button>`;

  return `<div class="inst-card">
    <div class="inst-top">
      <span class="inst-id">#${inst.id}</span>
      <span class="inst-gpu">${esc(inst.gpu_name || '?')}</span>
      <span class="badge ${badgeClass}"><span class="dot"></span>${inst.actual_status || '?'}</span>
    </div>
    <div class="inst-meta" title="${esc(ssh)}">${esc(meta)}</div>
    <div class="inst-actions">${actions}</div>
  </div>`;
}

function getSsh(inst) {
  // Prefer direct IP over relay (relay can break after instance restarts)
  const ports = inst.ports || {};
  for (const [k, v] of Object.entries(ports)) {
    if (k.startsWith('22/') && v && v[0]) {
      const host = inst.public_ipaddr || v[0].HostIp;
      if (host && host !== '0.0.0.0' && host !== '::')
        return `ssh -p ${v[0].HostPort} root@${host}`;
    }
  }
  if (inst.ssh_host && inst.ssh_port) return `ssh -p ${inst.ssh_port} root@${inst.ssh_host}`;
  return inst.public_ipaddr ? `ssh root@${inst.public_ipaddr}` : '';
}

async function activateInst(id) {
  const r = await fetch(`/api/instances/${id}/start`, {method: 'POST'});
  const d = await r.json();
  if (!d.ok) alert(`Failed to start: ${d.output}`);
  setTimeout(loadInstances, 1500);
}

async function destroyInst(id) {
  if (!confirm(`Destroy instance ${id}?`)) return;
  const r = await fetch(`/api/instances/${id}`, {method: 'DELETE'});
  const d = await r.json();
  if (!d.ok) alert(`Failed: ${d.output}`);
  loadInstances();
}

// ── Deploy to a running instance ──────────────────────────────────────────────
async function deployToInst(ssh) {
  if (!ssh) { alert('No SSH command available for this instance.'); return; }
  selectLog('deploy');
  clearLog();
  setStatus('running', '⟳ Deploying…');
  disableActions(true);

  try {
    const res = await fetch('/api/deploy', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        ssh_command: ssh,
        env_vars: getEnv(),
        worker_port: +document.getElementById('c-port').value || 9000,
        remote_root: document.getElementById('remote-root').value || '/workspace/filmforge_gpu_worker',
      }),
    });
    const {job_id} = await res.json();
    streamJob(job_id, false);
  } catch (e) {
    appendLog(`[deploy] ERROR: ${e}`);
    setStatus('failed', '✗ Failed');
    disableActions(false);
  }
}

// ── Create & Deploy (auto-provision) ─────────────────────────────────────────
async function quickDeploy() {
  selectLog('deploy');
  clearLog();
  setStatus('running', '⟳ Provisioning + deploying…');
  disableActions(true);

  try {
    const res = await fetch('/api/provision-vast', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        gpu_type: document.getElementById('c-gpu').value,
        max_price: +document.getElementById('c-price').value,
        min_vram_gb: +(document.getElementById('c-vram').value) || 24,
        disk_gb: +(document.getElementById('c-disk').value) || 200,
        template_hash: document.getElementById('c-tmpl').value.trim() || null,
        image: document.getElementById('c-image').value || 'vastai/comfy:v0.15.1-cuda-12.9-py312',
        max_upload_cost: numOrNull('c-up'),
        max_download_cost: numOrNull('c-down'),
        allow_fallback_gpu: false,
        worker_port: +(document.getElementById('c-port').value) || 9000,
        remote_root: document.getElementById('remote-root').value || '/workspace/filmforge_gpu_worker',
        warm_asset_groups: [],
        env_vars: getEnv(),
      }),
    });
    const {job_id} = await res.json();
    streamJob(job_id, true);
  } catch (e) {
    appendLog(`[vast] ERROR: ${e}`);
    setStatus('failed', '✗ Failed');
    disableActions(false);
  }
}

// ── Job streaming ─────────────────────────────────────────────────────────────
function streamJob(jobId, isProvision) {
  if (evtSrc) { evtSrc.close(); evtSrc = null; }
  evtSrc = new EventSource(`/api/deploy/${jobId}/stream`);
  evtSrc.onmessage = e => {
    if (e.data === '__DONE__') {
      evtSrc.close(); evtSrc = null;
      onJobDone(jobId, isProvision);
      return;
    }
    appendLog(JSON.parse(e.data), 'deploy');
  };
  evtSrc.onerror = () => { evtSrc.close(); evtSrc = null; onJobDone(jobId, isProvision); };
}

async function onJobDone(jobId, isProvision) {
  disableActions(false);
  const d = await fetch(`/api/deploy/${jobId}`).then(r => r.json());
  if (d.status === 'done') {
    setStatus('done', '✓ Done' + (d.worker_url ? ' — ' + d.worker_url : ''));
    if (isProvision) selectLog('worker');
    loadInstances();
    loadWorkers();
  } else {
    setStatus('failed', '✗ Failed — check log');
  }
}

function disableActions(on) {
  document.getElementById('provision-btn').disabled = on;
}

// ── Workers ───────────────────────────────────────────────────────────────────
async function loadWorkers() {
  const el = document.getElementById('workers-out');
  const list = await fetch('/api/workers').then(r => r.json()).catch(() => []);
  if (!list.length) {
    el.innerHTML = '<span class="muted text-xs" style="padding:10px 16px;display:block">No workers registered. (Is the backend running?)</span>';
    return;
  }
  el.innerHTML = list.map(w => {
    const alive = !!w.is_live || w.status === 'online';
    const age = timeAgo(w.last_heartbeat_at || w.last_seen_at);
    const caps = (w.capabilities || w.supported_asset_groups || []).join(', ');
    return `<div class="wrow">
      <div class="flex1" style="min-width:0">
        <div class="wname">${esc(w.worker_name || w.id || '?')}</div>
        <div class="wurl" title="${esc(w.base_url || '')}">${esc(w.base_url || '—')}</div>
        <div class="wurl" title="${esc(caps)}">${esc(caps || 'no capabilities')}</div>
      </div>
      <span class="badge ${alive ? 'b-running' : 'b-exited'}" style="flex-shrink:0">
        <span class="dot"></span>${alive ? 'Online' : 'Offline'}
      </span>
      <span class="muted text-xs" style="flex-shrink:0;min-width:44px;text-align:right">${age}</span>
    </div>`;
  }).join('');
}

function timeAgo(iso) {
  if (!iso) return '?';
  const s = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

// ── Log ───────────────────────────────────────────────────────────────────────
let remoteEvt = null;

function selectLog(src) {
  currentLogSrc = src;
  document.querySelectorAll('.ltab').forEach(b => b.classList.toggle('active', b.dataset.src === src));
  if (remoteEvt) { remoteEvt.close(); remoteEvt = null; }
  renderLog(src);
  if (src === 'deploy') return;
  clearLog(`Connecting to ${LOG_LABELS[src] || src}…`, src);
  remoteEvt = new EventSource(`/api/remote-logs/${src}/stream`);
  remoteEvt.onmessage = e => {
    if (e.data === '__DONE__') { remoteEvt.close(); remoteEvt = null; return; }
    appendLog(JSON.parse(e.data), src);
  };
  remoteEvt.onerror = () => { appendLog('[remote-log] disconnected', src); remoteEvt.close(); remoteEvt = null; };
}

function appendLog(line, src = currentLogSrc) {
  if (!line) return;
  if (!logBuffers[src]) logBuffers[src] = [];
  const isNoise = /"\s*(get|head)\s+\//i.test(line) || /http\/1\.1"\s+200/i.test(line);
  if (isNoise) return;
  logBuffers[src].push(line);
  if (logBuffers[src].length > MAX_LOG_LINES) {
    logBuffers[src].splice(0, logBuffers[src].length - MAX_LOG_LINES);
  }
  if (src !== currentLogSrc) return;
  renderLog(src);
}

function renderLog(src = currentLogSrc) {
  const el = document.getElementById('log');
  const lines = logBuffers[src] || [];
  if (!lines.length) {
    el.innerHTML = `<span class="muted text-xs">Ready.</span>`;
    return;
  }
  el.innerHTML = '';
  for (const line of lines) {
    const lo = line.toLowerCase();
    const d = document.createElement('div');
    const cls = lo.includes('error') || lo.includes('failed') ? 'err'
      : lo.includes('✓') || lo.includes(' done') ? 'ok'
      : lo.startsWith('[deploy]') || lo.startsWith('[vast]') ? 'info'
      : lo.includes('download') || lo.includes('asset') || lo.includes('comfy') ? 'signal'
      : lo.includes('warn') ? 'warn' : '';
    d.className = 'll' + (cls ? ' ' + cls : '');
    d.textContent = line;
    el.appendChild(d);
  }
  el.scrollTop = el.scrollHeight;
}

function clearLog(msg = 'Ready.', src = currentLogSrc) {
  logBuffers[src] = [];
  if (src === currentLogSrc) {
    document.getElementById('log').innerHTML = `<span class="muted text-xs">${esc(msg)}</span>`;
  }
}

function setStatus(type, msg) {
  const el = document.getElementById('sbar');
  el.className = type;
  el.textContent = msg;
}

// ── Env vars ──────────────────────────────────────────────────────────────────
const ENV_DEFAULTS = [
  ['FILMFORGE_BACKEND_URL', ''],
  ['WORKER_PROVIDER', 'dedicated_worker'],
  ['WORKER_MAX_CONCURRENT_JOBS', '1'],
  ['WORKER_HEARTBEAT_SECONDS', '60'],
  ['WORKER_CAPABILITIES', ''],
  ['WORKER_REGISTRATION_TOKEN', ''],
  ['WORKER_NAME', ''],
  ['WORKER_GPU_NAME', ''],
  ['RENDER_BROKER_BASE_URL', ''],
  ['RENDER_BROKER_WORKER_TOKEN', ''],
];

function initEnv() { ENV_DEFAULTS.forEach(([k, v]) => addEnvRow(k, v)); }

function addEnvRow(k = '', v = '') {
  const t = document.getElementById('env-table');
  const d = document.createElement('div');
  d.className = 'env-row';
  d.innerHTML = `<input class="ek" placeholder="KEY" value="${esc(k)}">
    <input class="ev" placeholder="value" value="${esc(v)}">
    <button class="rm-btn" onclick="this.parentElement.remove()">×</button>`;
  t.appendChild(d);
}

function getEnv() {
  const o = {};
  document.querySelectorAll('.env-row').forEach(r => {
    const k = r.querySelector('.ek').value.trim();
    const v = r.querySelector('.ev').value.trim();
    if (k && v) o[k] = v;
  });
  return o;
}

async function loadEnvFromBackend() {
  const data = await fetch('/api/backend-env').then(r => r.json()).catch(() => ({}));
  document.querySelectorAll('.env-row').forEach(row => {
    const k = row.querySelector('.ek').value.trim();
    if (data[k] !== undefined) row.querySelector('.ev').value = data[k];
  });
  Object.entries(data).forEach(([k, v]) => {
    const exists = [...document.querySelectorAll('.ek')].some(i => i.value.trim() === k);
    if (!exists && v) addEnvRow(k, v);
  });
}

// ── Template auto-detect ──────────────────────────────────────────────────────
async function autoDetectTemplate() {
  const statusEl = document.getElementById('tmpl-status');
  statusEl.textContent = 'Searching…';
  try {
    const templates = await fetch('/api/templates?q=comfy').then(r => r.json());
    if (templates.length) {
      const t = templates[0];
      document.getElementById('c-tmpl').value = t.hash_id || '';
      if (t.image) {
        const img = t.image + (t.tag ? ':' + t.tag : '');
        document.getElementById('c-image').value = img;
      }
      statusEl.textContent = `✓ Found: ${t.name}`;
      statusEl.style.color = '#34d399';
    } else {
      statusEl.textContent = 'No ComfyUI template found.';
    }
  } catch {
    statusEl.textContent = 'Failed to fetch templates.';
  }
}

// ── Collapsible sections ──────────────────────────────────────────────────────
function toggleEnv() {
  const btn = document.getElementById('env-toggle');
  const body = document.getElementById('env-body');
  btn.classList.toggle('open');
  body.classList.toggle('open');
}

function toggleAdv() {
  const body = document.getElementById('adv-body');
  const btn = event.target;
  const open = body.style.display === 'block';
  body.style.display = open ? 'none' : 'block';
  btn.textContent = (open ? '▶' : '▼') + ' Advanced options';
}

// ── Util ──────────────────────────────────────────────────────────────────────
function numOrNull(id) {
  const raw = document.getElementById(id).value.trim();
  if (!raw) return null;
  const v = Number(raw);
  return Number.isFinite(v) ? v : null;
}

function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Init ──────────────────────────────────────────────────────────────────────
initEnv();
loadInstances();
loadWorkers();
setInterval(loadInstances, 15000);
setInterval(loadWorkers, 15000);
</script>
</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    print(f"FilmForge GPU Deploy UI → http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
