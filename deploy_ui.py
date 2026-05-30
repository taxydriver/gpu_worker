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
import re
import shlex
import subprocess
import sys
import threading
import time
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
    worker_count: int = 0
    comfy_port: int = 18188
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
    worker_count: int = 0
    comfy_port: int = 18188
    remote_root: str = "/workspace/filmforge_gpu_worker"
    warm_asset_groups: list[str] = []
    env_vars: dict[str, str] = {}
    max_upload_cost: float | None = None
    max_download_cost: float | None = None
    allow_fallback_gpu: bool = False


class RunPodProvisionRequest(BaseModel):
    gpu_type: str = "NVIDIA L40S"
    cloud_type: str = "COMMUNITY"
    volume_gb: int = 150
    container_disk_gb: int = 50
    worker_port: int = 9000
    remote_root: str = "/workspace/filmforge_gpu_worker"
    env_vars: dict[str, str] = {}
    pod_id: str | None = None


class VerdaHostRequest(BaseModel):
    name: str
    ssh_command: str


class VerdaTeardownRequest(BaseModel):
    name: str = ""
    ssh_command: str = ""
    instance_id: str = ""
    delete_volumes: bool = False


class VerdaProvisionRequest(BaseModel):
    fresh: bool = False
    location: str = "FIN-01"
    instance_type: str = "2A100.44V"
    contract: str = "pay_as_go"
    os_volume_id: str = "34ec939d-a8c1-4ee2-9637-533e324dfe39"
    data_volume_id: str = "4ea18b04-564f-4218-ab79-e90d1ccc839b"
    ssh_key_id: str = "11ee08a4-858a-4ee7-98c8-250aad99eb37"
    hostname: str = "filmforge-verda-worker"
    worker_count: int = 0
    worker_port: int = 9000
    comfy_port: int = 8188
    remote_root: str = "/workspace/filmforge_gpu_worker"
    fresh_os_volume_size: int = 100
    fresh_storage_size: int = 250
    fresh_os_volume_name: str = ""
    fresh_storage_name: str = ""
    skip_warmup: bool = True
    warm_asset_groups: list[str] = []
    env_vars: dict[str, str] = {}


# RunPod proxy URL pattern (no cloudflared on RunPod pods)
_RUNPOD_PROXY_PATTERN = re.compile(r"https://[\w]+-\d+\.proxy\.runpod\.net")


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


def _find_runpod_python() -> str:
    """Return a Python executable that has the runpod package.

    Tries sys.executable first (deploy UI's own Python), then common
    locations.  Falls back to sys.executable even if runpod isn't there
    so the subprocess at least produces a clear error.
    """
    candidates = [
        sys.executable,
        "/opt/homebrew/bin/python3.13",
        "python3.13",
        "python3.12",
        "python3",
    ]
    for py in candidates:
        result = subprocess.run(
            [py, "-c", "import runpod"],
            capture_output=True,
        )
        if result.returncode == 0:
            return py
    return sys.executable


def _read_backend_env_key(key: str) -> str | None:
    if not DEFAULT_BACKEND_ENV.exists():
        return None
    for line in DEFAULT_BACKEND_ENV.read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return None


def _runpod_client():
    try:
        import runpod as rp  # type: ignore[import]
    except ImportError:
        return None
    api_key = _read_backend_env_key("RUNPOD_API_KEY")
    if not api_key:
        return None
    rp.api_key = api_key
    return rp


def _is_filmforge_pod(pod: dict) -> bool:
    env = pod.get("env") or []
    if isinstance(env, dict):
        return env.get("FILMFORGE_OWNED") == "1"
    return any(str(e).startswith("FILMFORGE_OWNED=") for e in env)


def _normalize_runpod_pod(pod: dict) -> dict:
    runtime = pod.get("runtime") or {}
    ports = runtime.get("ports") or []
    ssh = next((p for p in ports if p.get("privatePort") == 22 and p.get("isIpPublic")), None)
    machine = pod.get("machine") or {}
    return {
        "id": pod["id"],
        "name": pod.get("name", ""),
        "status": pod.get("desiredStatus", "UNKNOWN"),
        "gpu_name": machine.get("gpuDisplayName") or "GPU",
        "cost_per_hr": pod.get("costPerHr"),
        "ssh_ip": ssh["ip"] if ssh else None,
        "ssh_port": ssh["publicPort"] if ssh else None,
    }


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


VERDA_HOSTS_PATH = SCRIPT_DIR / ".verda_hosts.json"
VERDA_CLI = Path.home() / ".verda" / "bin" / "verda"
VERDA_AVAILABILITY_LOCATIONS = ("FIN-01", "FIN-02", "FIN-03")
_VERDA_INSTANCE_TYPES_CACHE: dict[str, object] = {"loaded_at": 0.0, "prices": {}}
_VERDA_INSTANCE_TYPES_TTL_SEC = 600


def _normalize_ssh_command(cmd: str) -> str:
    cmd = (cmd or "").strip()
    if not cmd:
        return ""
    return cmd if cmd.startswith("ssh ") else f"ssh {cmd}"


def _load_verda_hosts() -> list[dict]:
    if VERDA_HOSTS_PATH.exists():
        try:
            data = json.loads(VERDA_HOSTS_PATH.read_text() or "[]")
            if isinstance(data, list):
                return [h for h in data if isinstance(h, dict) and h.get("name") and h.get("ssh_command")]
        except json.JSONDecodeError:
            return []
        return []
    seed = _read_backend_env_key("VERDA_SSH_DEST")
    if seed:
        hosts = [{"name": "default", "ssh_command": _normalize_ssh_command(seed)}]
        _save_verda_hosts(hosts)
        return hosts
    return []


def _save_verda_hosts(hosts: list[dict]) -> None:
    VERDA_HOSTS_PATH.write_text(json.dumps(hosts, indent=2) + "\n")


def _verda_ssh_command_for_ip(ip: str) -> str:
    identity = Path.home() / ".ssh" / "id_ed25519"
    if identity.exists():
        return f"ssh -i {identity} root@{ip}"
    return f"ssh root@{ip}"


def _verda_credentials() -> dict[str, str]:
    path = Path.home() / ".verda" / "credentials"
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("[") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def _verda_api_request(method: str, path: str, payload: dict | None = None, token: str | None = None) -> object:
    creds = _verda_credentials()
    base_url = (creds.get("verda_base_url") or "https://api.verda.com/v1").rstrip("/")
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{base_url}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except Exception as exc:
        raise RuntimeError(f"Verda API {method} {path} failed: {exc}") from exc


def _verda_api_token() -> str:
    creds = _verda_credentials()
    client_id = creds.get("verda_client_id")
    client_secret = creds.get("verda_client_secret")
    if not client_id or not client_secret:
        raise RuntimeError("Verda credentials missing client id/secret")
    payload = _verda_api_request("POST", "/oauth2/token", {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    })
    if not isinstance(payload, dict) or not payload.get("access_token"):
        raise RuntimeError("Verda token response did not include access_token")
    return str(payload["access_token"])


def _verda_volume_api_action(volume_id: str, action: str) -> None:
    token = _verda_api_token()
    _verda_api_request("PUT", "/volumes", {"action": action, "id": volume_id}, token)


def _verda_wait_for_detached(volume_ids: list[str], timeout_sec: int = 300) -> list[dict]:
    deadline = time.time() + timeout_sec
    wanted = set(volume_ids)
    last: list[dict] = []
    while time.time() < deadline:
        volumes = _verda_list_volumes()
        last = [v for v in volumes if isinstance(v, dict) and str(v.get("id")) in wanted]
        if len(last) == len(wanted) and all(str(v.get("status") or "").lower() == "detached" for v in last):
            return last
        time.sleep(5)
    status = ", ".join(f"{v.get('id')}:{v.get('status')}" for v in last) or "not found"
    raise RuntimeError(f"Timed out waiting for Verda volumes to detach: {status}")


def _verda_wait_for_vm_deleted(instance_id: str, timeout_sec: int = 300) -> None:
    deadline = time.time() + timeout_sec
    last_status = "unknown"
    while time.time() < deadline:
        vms = _verda_list_vms()
        match = next((vm for vm in vms if isinstance(vm, dict) and vm.get("id") == instance_id), None)
        if not match:
            return
        last_status = str(match.get("status") or "unknown")
        time.sleep(5)
    raise RuntimeError(f"Timed out waiting for Verda instance {instance_id} to be deleted; last status={last_status}")


def _verda_delete_instance(instance_id: str, volume_ids: list[str], *, delete_volumes: bool) -> object:
    token = _verda_api_token()
    payload: dict[str, object] = {
        "action": "delete",
        "id": instance_id,
        # Empty list is the documented safe form: delete instance, preserve volumes.
        "volume_ids": volume_ids if delete_volumes else [],
    }
    if delete_volumes:
        payload["delete_permanently"] = False
    return _verda_api_request("PUT", "/instances", payload, token)


def _verda_hosts_with_live_vms() -> list[dict]:
    hosts = _load_verda_hosts()
    by_key: dict[str, dict] = {}

    def add_host(host: dict) -> None:
        name = str(host.get("name") or "").strip()
        ssh = _normalize_ssh_command(str(host.get("ssh_command") or ""))
        if not name or not ssh:
            return
        key = _host_from_ssh_command(ssh) or name
        merged = dict(host)
        merged["name"] = name
        merged["ssh_command"] = ssh
        by_key[key] = {**by_key.get(key, {}), **merged}

    for host in hosts:
        add_host(host)

    if not VERDA_CLI.exists():
        return list(by_key.values())

    try:
        vms = _verda_list_vms()
    except RuntimeError:
        return list(by_key.values())

    used_names = {str(host.get("name")) for host in by_key.values()}
    for vm in vms:
        if not isinstance(vm, dict):
            continue
        ip = str(vm.get("ip") or "").strip()
        if not ip:
            continue
        instance_id = str(vm.get("id") or "").strip()
        base_name = str(vm.get("hostname") or vm.get("description") or "").strip()
        name = base_name or (f"verda-{instance_id[:8]}" if instance_id else f"verda-{ip}")
        if name in used_names and ip not in by_key:
            suffix = instance_id[:8] if instance_id else ip.replace(".", "-")
            name = f"{name}-{suffix}"
        used_names.add(name)
        add_host({
            "name": name,
            "ssh_command": _verda_ssh_command_for_ip(ip),
            "source": "live",
            "instance_id": instance_id,
            "status": vm.get("status"),
            "location": vm.get("location"),
            "instance_type": vm.get("instance_type"),
            "price_per_hour": vm.get("price_per_hour"),
            "is_spot": vm.get("is_spot"),
        })

    return list(by_key.values())


def _verda_instance_type_prices() -> dict[str, dict]:
    now = time.time()
    cached_prices = _VERDA_INSTANCE_TYPES_CACHE.get("prices")
    loaded_at = float(_VERDA_INSTANCE_TYPES_CACHE.get("loaded_at") or 0.0)
    if isinstance(cached_prices, dict) and cached_prices and now - loaded_at < _VERDA_INSTANCE_TYPES_TTL_SEC:
        return cached_prices

    proc = subprocess.run(
        [str(VERDA_CLI), "--agent", "instance-types"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        return cached_prices if isinstance(cached_prices, dict) else {}
    try:
        rows = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return cached_prices if isinstance(cached_prices, dict) else {}

    prices: dict[str, dict] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            instance_type = str(row.get("instance_type") or "")
            if not instance_type:
                continue
            prices[instance_type] = {
                "price_per_hour": row.get("price_per_hour"),
                "spot_price": row.get("spot_price"),
                "currency": row.get("currency") or "usd",
                "gpu_memory_gb": (row.get("gpu_memory") or {}).get("size_in_gigabytes"),
                "ram_gb": (row.get("memory") or {}).get("size_in_gigabytes"),
                "name": row.get("name") or row.get("model") or "",
            }
    _VERDA_INSTANCE_TYPES_CACHE["loaded_at"] = now
    _VERDA_INSTANCE_TYPES_CACHE["prices"] = prices
    return prices


def _verda_price_label(price_per_hour: object, spot_price: object = None) -> str:
    try:
        hourly = float(price_per_hour)
    except (TypeError, ValueError):
        return "price unavailable"
    label = f"${hourly:.3f}/hr"
    try:
        spot = float(spot_price)
    except (TypeError, ValueError):
        return label
    return f"{label}, spot ${spot:.3f}/hr"


def _verda_list_volumes() -> list[dict]:
    proc = subprocess.run(
        [str(VERDA_CLI), "--agent", "volume", "list"],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "Verda volume list failed").strip())
    try:
        payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Verda returned non-JSON volume list: {proc.stdout[:300]}") from exc
    return payload if isinstance(payload, list) else []


def _verda_list_vms() -> list[dict]:
    proc = subprocess.run(
        [str(VERDA_CLI), "--agent", "vm", "list"],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "Verda VM list failed").strip())
    try:
        payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Verda returned non-JSON VM list: {proc.stdout[:300]}") from exc
    return payload if isinstance(payload, list) else []


def _host_from_ssh_command(ssh_command: str) -> str:
    try:
        parts = shlex.split(_normalize_ssh_command(ssh_command))
    except ValueError:
        return ""
    for part in reversed(parts):
        if "@" in part and not part.startswith("-"):
            return part.rsplit("@", 1)[1]
    return ""


def _host_from_ssh_destination(destination: str) -> str:
    """Extract the network host from a parsed SSH destination string."""

    target = str(destination or "").strip()
    if "@" in target:
        target = target.rsplit("@", 1)[1]
    if ":" in target and not target.startswith("["):
        target = target.split(":", 1)[0]
    return target.strip("[]")


def _verda_find_vm_for_teardown(req: VerdaTeardownRequest) -> dict:
    vms = _verda_list_vms()
    if req.instance_id:
        match = next((vm for vm in vms if isinstance(vm, dict) and vm.get("id") == req.instance_id), None)
        if match:
            return match
        raise RuntimeError(f"Verda instance not found: {req.instance_id}")
    host = _host_from_ssh_command(req.ssh_command)
    if host:
        match = next((vm for vm in vms if isinstance(vm, dict) and vm.get("ip") == host), None)
        if match:
            return match
    if req.name:
        match = next((vm for vm in vms if isinstance(vm, dict) and vm.get("hostname") == req.name), None)
        if match:
            return match
    raise RuntimeError("Could not match selected Verda host to a running VM")


def _verda_validate_existing_volumes(req: VerdaProvisionRequest) -> None:
    volumes = _verda_list_volumes()
    os_volume = next((v for v in volumes if isinstance(v, dict) and v.get("id") == req.os_volume_id), None)
    data_volume = next((v for v in volumes if isinstance(v, dict) and v.get("id") == req.data_volume_id), None)
    if not os_volume:
        raise RuntimeError(f"OS volume not found: {req.os_volume_id}")
    if not data_volume:
        raise RuntimeError(f"Data/model volume not found: {req.data_volume_id}")
    problems: list[str] = []
    if not os_volume.get("is_os_volume"):
        problems.append("OS volume field is not an OS volume")
    if data_volume.get("is_os_volume"):
        problems.append("data/model volume field points to an OS volume")
    for label, volume in (("OS", os_volume), ("data/model", data_volume)):
        status = str(volume.get("status") or "").lower()
        location = str(volume.get("location") or "")
        if status != "detached":
            problems.append(f"{label} volume {volume.get('id')} is {status}, not detached")
        if location != req.location:
            problems.append(f"{label} volume {volume.get('id')} is in {location}, not {req.location}")
    if problems:
        raise RuntimeError("; ".join(problems))


def _verda_gpu_count(instance_type: str) -> int:
    match = re.match(r"^(\d+)", instance_type.upper())
    return int(match.group(1)) if match else 1


def _verda_gpu_rank(instance_type: str, preference: str = "single") -> tuple[int, int, str]:
    """Rank FilmForge-friendly Verda GPU types for UI defaults."""

    name = instance_type.upper()
    if name.startswith("CPU"):
        return (999, 999, name)

    gpu_count = _verda_gpu_count(instance_type)

    if "H200" in name:
        family = 0
    elif "B300" in name:
        family = 1
    elif "B200" in name:
        family = 2
    elif "H100" in name:
        family = 3
    elif "A100" in name:
        family = 4
    elif "RTXPRO6000" in name or "RTXPRO" in name:
        family = 5
    elif "L40S" in name:
        family = 6
    elif "A6000" in name:
        family = 7
    else:
        family = 50

    if preference == "sprint":
        count_penalty = 0 if gpu_count >= 2 else 100
        return (count_penalty + family, -gpu_count, name)
    if preference == "four_plus":
        count_penalty = 0 if gpu_count >= 4 else 100
        return (count_penalty + family, -gpu_count, name)
    if preference == "any":
        return (family, -gpu_count, name)

    count_penalty = 0 if gpu_count == 1 else 100
    return (count_penalty + family, gpu_count, name)


def _verda_instance_label(location: str, instance_type: str, price: dict | None = None) -> str:
    gpu_count = _verda_gpu_count(instance_type)
    if "H200" in instance_type.upper() and gpu_count == 1:
        hint = "recommended"
    elif gpu_count >= 4:
        hint = f"{gpu_count} GPU sprint"
    elif gpu_count > 1:
        hint = f"{gpu_count} GPU"
    else:
        hint = "available"
    parts = [location, instance_type, hint]
    if price:
        parts.append(_verda_price_label(price.get("price_per_hour"), price.get("spot_price")))
    return " · ".join(parts)


_SYSTEMD_UNIT_PATTERNS: dict[str, str] = {
    "GPU Worker": r"^filmforge-worker-gpu[0-9]+\.service$",
}


def _remote_tail_script(paths: tuple[str, ...], label: str) -> str:
    quoted_paths = " ".join(shlex.quote(path) for path in paths)
    primary = shlex.quote(paths[0])
    quoted_label = shlex.quote(label)
    systemd_pattern = _SYSTEMD_UNIT_PATTERNS.get(label)
    systemd_branch = ""
    if systemd_pattern:
        quoted_pattern = shlex.quote(systemd_pattern)
        systemd_branch = f"""if command -v systemctl >/dev/null 2>&1; then
  units=$(systemctl list-units --type=service --no-legend 2>/dev/null \\
          | awk '{{print $1}}' | grep -E {quoted_pattern} | sort -u || true)
  if test -n "$units"; then
    echo "[remote-log] tailing {quoted_label} systemd units: $(echo $units | tr '\\n' ' ')"
    pids=""
    for u in $units; do
      tag=$(echo "$u" | grep -oE 'gpu[0-9]+')
      journalctl -n 100 -f -u "$u" | awk -v t="[$tag]" '{{print t" "$0; fflush()}}' &
      pids="$pids $!"
    done
    trap "kill $pids 2>/dev/null || true" EXIT INT TERM
    wait
    exit 0
  fi
fi
"""
    return f"""set -euo pipefail
{systemd_branch}paths=({quoted_paths})
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
if command -v systemctl >/dev/null 2>&1; then
  units=$(systemctl list-units --type=service --no-legend 2>/dev/null \
          | awk '{print $1}' | grep -E '^comfyui-gpu[0-9]+\.service$' | sort -u || true)
  if test -n "$units"; then
    echo "[comfy] tailing systemd units: $(echo $units | tr '\n' ' ')"
    pids=""
    for u in $units; do
      tag=$(echo "$u" | grep -oE 'gpu[0-9]+')
      journalctl -n 100 -f -u "$u" | awk -v t="[$tag]" '{print t" "$0; fflush()}' &
      pids="$pids $!"
    done
    trap "kill $pids 2>/dev/null || true" EXIT INT TERM
    wait
    exit 0
  fi
fi

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


@app.get("/api/verda/availability")
async def verda_availability(preference: str = "single", contract: str = "pay_as_go"):
    if not VERDA_CLI.exists():
        raise HTTPException(503, f"Verda CLI not found at {VERDA_CLI}")

    options: list[dict] = []
    errors: dict[str, str] = {}
    prices = _verda_instance_type_prices()
    contract = str(contract or "pay_as_go").lower()
    use_spot = contract == "spot"
    for location in VERDA_AVAILABILITY_LOCATIONS:
        try:
            command = [str(VERDA_CLI), "--agent", "availability", "--location", location]
            if use_spot:
                command.append("--spot")
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            if proc.returncode != 0:
                errors[location] = (proc.stderr or proc.stdout).strip()
                continue
            payload = json.loads(proc.stdout or "[]")
            rows = payload if isinstance(payload, list) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                loc = str(row.get("location_code") or location)
                for instance_type in row.get("instance_types") or []:
                    instance_type = str(instance_type)
                    if instance_type.upper().startswith("CPU"):
                        continue
                    rank = _verda_gpu_rank(instance_type, preference)
                    price = prices.get(instance_type) or {}
                    options.append({
                        "location": loc,
                        "instance_type": instance_type,
                        "label": _verda_instance_label(loc, instance_type, price),
                        "rank": rank[0],
                        "gpu_count": _verda_gpu_count(instance_type),
                        "price_per_hour": price.get("price_per_hour"),
                        "spot_price": price.get("spot_price"),
                        "currency": price.get("currency") or "usd",
                        "gpu_memory_gb": price.get("gpu_memory_gb"),
                        "ram_gb": price.get("ram_gb"),
                        "hardware_name": price.get("name") or "",
                    })
        except Exception as exc:
            errors[location] = str(exc)

    dedup: dict[tuple[str, str], dict] = {}
    for option in options:
        dedup[(option["location"], option["instance_type"])] = option
    ranked = sorted(dedup.values(), key=lambda item: (
        _verda_gpu_rank(item["instance_type"], preference),
        item["location"],
        item["instance_type"],
    ))
    return {"items": ranked[:5], "all_items": ranked, "errors": errors}


@app.get("/api/verda/cost-estimate")
async def verda_cost_estimate(
    instance_type: str,
    location: str = "FIN-01",
    os_volume_gb: int = 0,
    storage_gb: int = 0,
    storage_type: str = "NVMe",
    contract: str = "pay_as_go",
):
    if not VERDA_CLI.exists():
        raise HTTPException(503, f"Verda CLI not found at {VERDA_CLI}")

    command = [
        str(VERDA_CLI), "--agent", "cost", "estimate",
        "--type", instance_type,
        "--location", location,
        "-o", "json",
    ]
    if os_volume_gb > 0:
        command.extend(["--os-volume", str(os_volume_gb)])
    if storage_gb > 0:
        command.extend(["--storage", str(storage_gb), "--storage-type", storage_type or "NVMe"])
    if contract == "spot":
        command.append("--spot")
    proc = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
    if proc.returncode != 0:
        raise HTTPException(502, (proc.stderr or proc.stdout or "Verda cost estimate failed").strip())
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(502, f"Verda returned non-JSON cost estimate: {proc.stdout[:300]}") from exc
    return payload


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


@app.get("/api/runpod/pods")
async def list_runpod_pods():
    import asyncio, traceback
    rp = _runpod_client()
    if rp is None:
        print("[runpod] _runpod_client() returned None (no API key or import failed)")
        return []
    try:
        loop = asyncio.get_event_loop()
        pods = await loop.run_in_executor(None, rp.get_pods)
        return [
            _normalize_runpod_pod(p)
            for p in (pods or [])
            if _is_filmforge_pod(p)
        ]
    except Exception:
        print(f"[runpod] list_runpod_pods error:\n{traceback.format_exc()}")
        return []


@app.post("/api/runpod/pods/{pod_id}/resume")
async def resume_runpod_pod(pod_id: str):
    rp = _runpod_client()
    if rp is None:
        raise HTTPException(503, "RUNPOD_API_KEY not configured")
    try:
        loop = asyncio.get_event_loop()
        pods = await loop.run_in_executor(None, rp.get_pods)
        pod = next((p for p in (pods or []) if p.get("id") == pod_id), None)
        gpu_count = int((pod or {}).get("gpuCount") or 1)
        await loop.run_in_executor(None, rp.resume_pod, pod_id, gpu_count)
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "output": str(exc)}


@app.post("/api/runpod/pods/{pod_id}/stop")
async def stop_runpod_pod(pod_id: str):
    rp = _runpod_client()
    if rp is None:
        raise HTTPException(503, "RUNPOD_API_KEY not configured")
    try:
        rp.stop_pod(pod_id)
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "output": str(exc)}


@app.delete("/api/runpod/pods/{pod_id}")
async def terminate_runpod_pod(pod_id: str):
    rp = _runpod_client()
    if rp is None:
        raise HTTPException(503, "RUNPOD_API_KEY not configured")
    try:
        rp.terminate_pod(pod_id)
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "output": str(exc)}


@app.post("/api/provision-runpod")
async def start_runpod_provision(req: RunPodProvisionRequest):
    job_id = str(uuid.uuid4())
    q: queue.Queue[str | None] = queue.Queue()
    _jobs[job_id] = {
        "status": "running",
        "logs": [],
        "queue": q,
        "worker_url": None,
        "worker_urls": [],
        "ssh_command": None,
    }

    def _run() -> None:
        job = _jobs[job_id]
        try:
            from gpu_worker.deploy_gpu import extract_worker_url, extract_worker_urls
            runpod_python = _find_runpod_python()
            command = [
                runpod_python, "-u", str(SCRIPT_DIR / "deploy_gpu.py"),
                "--runpod",
                "--worker-port", str(req.worker_port),
                "--remote-root", req.remote_root,
                "--runpod-volume-gb", str(req.volume_gb),
                "--runpod-container-disk-gb", str(req.container_disk_gb),
                "--runpod-cloud-type", req.cloud_type,
                "--skip-warmup",
                "--update-backend-env",
            ]
            if req.gpu_type:
                command.extend(["--gpu", req.gpu_type])
            if req.pod_id:
                command.extend(["--pod-id", req.pod_id])
            for key, value in req.env_vars.items():
                if key and value:
                    command.extend(["--env", f"{key}={value}"])

            _put(job, "[runpod] Finding or creating pod, deploying worker…")
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

            worker_url = extract_worker_url(output_text)
            if not worker_url:
                m = _RUNPOD_PROXY_PATTERN.search(output_text)
                if m:
                    worker_url = m.group(0)
            job["worker_url"] = worker_url or None

            if proc.returncode == 0:
                job["status"] = "done"
                if worker_url:
                    _put(job, f"[runpod] Worker URL: {worker_url}")
                _put(job, "[runpod] Provision complete")
            else:
                job["status"] = "failed"
                _put(job, f"[runpod] Provision failed (exit {proc.returncode})")
        except Exception as exc:
            _put(job, f"[runpod] ERROR: {exc}")
            job["status"] = "failed"
        finally:
            job["queue"].put(None)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


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
    _jobs[job_id] = {
        "status": "running",
        "logs": [],
        "queue": q,
        "worker_url": None,
        "worker_urls": [],
    }

    def _run() -> None:
        job = _jobs[job_id]
        try:
            from gpu_worker.deploy_gpu import (
                parse_ssh_command, add_default_identity, add_default_host_key_policy,
                remote_script, vast_multi_gpu_script, extract_worker_url, extract_worker_urls,
                verda_fresh_install_script, verda_rehydrate_script,
                DEFAULT_WORKER_REPO_URL, DEFAULT_COMFY_REPO_URL, DEFAULT_PYTORCH_INDEX_URL,
            )
            _put(job, "[deploy] Parsing SSH command...")
            try:
                ssh_cmd, _scp_cmd, destination = parse_ssh_command(req.ssh_command)
            except ValueError as exc:
                _put(job, f"[deploy] ERROR: {exc}")
                job["status"] = "failed"
                return

            ssh_cmd = add_default_host_key_policy(add_default_identity(ssh_cmd))
            _put(job, f"[deploy] Waiting for SSH on {destination}…")
            if not _wait_for_ssh_ready(ssh_cmd, job, timeout=180):
                _put(job, "[deploy] ERROR: timed out waiting for SSH to become ready")
                job["status"] = "failed"
                return
            _put(job, "[deploy] SSH ready — cloning gpu_worker from GitHub…")
            clone_script = f"""set -euo pipefail
REMOTE_ROOT={shlex.quote(req.remote_root)}
mkdir -p "$REMOTE_ROOT"
if ! command -v git >/dev/null 2>&1; then
  apt-get update -qq 2>/dev/null || true
  apt-get install -y -qq git 2>/dev/null || true
fi
cd "$REMOTE_ROOT"
if test -d gpu_worker/.git; then
  echo "[deploy] Updating existing gpu_worker clone..."
  cd gpu_worker && git pull origin main && cd ..
else
  rm -rf gpu_worker
  echo "[deploy] Cloning gpu_worker from GitHub..."
  GIT_TERMINAL_PROMPT=0 git clone https://github.com/taxydriver/gpu_worker.git gpu_worker
fi
"""
            try:
                proc = subprocess.run(
                    [*ssh_cmd, "bash", "-s"],
                    input=clone_script, capture_output=True, text=True,
                )
                for line in (proc.stdout + proc.stderr).splitlines():
                    if line.strip():
                        _put(job, line)
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, ssh_cmd)
            except subprocess.CalledProcessError as exc:
                _put(job, f"[deploy] ERROR cloning from GitHub: {exc}")
                job["status"] = "failed"
                return

            env_exports = "\n".join(
                f"export {k}={shlex.quote(str(v))}" for k, v in req.env_vars.items() if k and v
            )
            env_vars = dict(req.env_vars or {})
            is_verda = str(env_vars.get("WORKER_PROVIDER") or "").lower() == "verda"
            direct_host = _host_from_ssh_destination(destination) if is_verda else None
            if is_verda:
                if direct_host and not env_vars.get("WORKER_PUBLIC_URL"):
                    env_vars["WORKER_PUBLIC_URL"] = f"http://{direct_host}:{req.worker_port}"
                    _put(job, f"[deploy] Verda direct worker URL: {env_vars['WORKER_PUBLIC_URL']}")
                if direct_host and int(req.worker_count or 0) > 1 and not env_vars.get("WORKER_PUBLIC_URLS"):
                    env_vars["WORKER_PUBLIC_URLS"] = ",".join(
                        f"http://{direct_host}:{req.worker_port + idx}"
                        for idx in range(int(req.worker_count or 0))
                    )
                    _put(job, f"[deploy] Verda direct worker URLs: {env_vars['WORKER_PUBLIC_URLS']}")
                env_exports = "\n".join(
                    f"export {k}={shlex.quote(str(v))}" for k, v in env_vars.items() if k and v
                )

                # Fresh Verda VMs ship with a base Ubuntu image — no ComfyUI.
                # The shared remote_script (Vast/RunPod path) only probes for an
                # existing ComfyUI and silently continues if absent, leaving the
                # worker unable to render. Detect missing ComfyUI and run the
                # CLI's fresh-install script to clone it and mount /dev/vdb.
                _put(job, "[deploy] Verda detected — checking for ComfyUI on remote VM...")
                comfy_check = subprocess.run(
                    [*ssh_cmd, "test -f /workspace/ComfyUI/main.py && test -x /workspace/ComfyUI/.venv/bin/python"],
                    capture_output=True, text=True, timeout=30, check=False,
                )
                if comfy_check.returncode != 0:
                    _put(job, "[deploy] ComfyUI not installed — running Verda fresh-install (clones ComfyUI, mounts /dev/vdb, installs deps). This takes ~5–10 min.")
                    install_script = verda_fresh_install_script(
                        worker_repo_url=DEFAULT_WORKER_REPO_URL,
                        comfy_repo_url=DEFAULT_COMFY_REPO_URL,
                        pytorch_index_url=DEFAULT_PYTORCH_INDEX_URL,
                        remote_root=req.remote_root,
                    )
                    if env_exports:
                        install_script = env_exports + "\n\n" + install_script
                    iproc = subprocess.Popen(
                        [*ssh_cmd, "bash", "-s"],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    )
                    assert iproc.stdin and iproc.stdout
                    iproc.stdin.write(install_script)
                    iproc.stdin.close()
                    for line in iproc.stdout:
                        line = line.rstrip()
                        if line:
                            _put(job, line)
                    iproc.wait()
                    if iproc.returncode != 0:
                        _put(job, f"[deploy] ERROR: Verda fresh-install failed (exit {iproc.returncode})")
                        job["status"] = "failed"
                        return
                    _put(job, "[deploy] Verda fresh-install complete — ComfyUI is now on the VM.")
                else:
                    _put(job, "[deploy] ComfyUI already present on remote — skipping fresh-install.")
            remote_gpu_count = 0
            try:
                gpu_probe = subprocess.run(
                    [*ssh_cmd, "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l | tr -d ' ' || true"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                remote_gpu_count = int((gpu_probe.stdout or "0").strip().splitlines()[-1] or "0")
            except Exception:
                remote_gpu_count = 0

            requested_worker_count = int(req.worker_count or 0)
            effective_worker_count = requested_worker_count if requested_worker_count > 0 else remote_gpu_count
            if is_verda:
                # Verda: use the rehydrate script so ComfyUI + worker run under
                # systemd (survives reboots) and the data volume binds re-mount
                # cleanly. This is the same path the CLI's --verda-fresh uses.
                _put(job, f"[deploy] Verda bootstrap: {max(effective_worker_count, 1)} worker(s) under systemd.")
                script = verda_rehydrate_script(
                    public_ip=direct_host or "",
                    worker_port=req.worker_port,
                    comfy_port=req.comfy_port,
                    worker_count=max(effective_worker_count, 1),
                    remote_root=req.remote_root,
                )
            elif effective_worker_count > 1:
                _put(job, f"[deploy] Detected {remote_gpu_count} GPUs; starting {effective_worker_count} workers.")
                script = vast_multi_gpu_script(
                    remote_root=req.remote_root,
                    worker_port=req.worker_port,
                    comfy_port=req.comfy_port,
                    worker_count=effective_worker_count,
                )
            else:
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
            worker_urls = extract_worker_urls(stdout_text)
            worker_url = worker_urls[0] if worker_urls else extract_worker_url(stdout_text)

            if worker_url:
                if worker_urls:
                    _put(job, f"[deploy] Worker URLs: {', '.join(worker_urls)}")
                else:
                    _put(job, f"[deploy] Worker URL: {worker_url}")
                _put(job, "[deploy] Worker registration is handled by FILMFORGE_BACKEND_URL / render broker heartbeat.")
                job["worker_url"] = worker_url
                job["worker_urls"] = worker_urls
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
    _jobs[job_id] = {
        "status": "running",
        "logs": [],
        "queue": q,
        "worker_url": None,
        "worker_urls": [],
        "ssh_command": None,
    }

    def _run() -> None:
        job = _jobs[job_id]
        try:
            from gpu_worker.deploy_gpu import extract_worker_url, extract_worker_urls
            command = [
                sys.executable, "-u", str(SCRIPT_DIR / "deploy_gpu.py"),
                "--vast",
                "--vast-gpu", req.gpu_type,
                "--vast-max-price", str(req.max_price),
                "--vast-min-vram-gb", str(req.min_vram_gb),
                "--vast-disk-gb", str(req.disk_gb),
                "--vast-image", req.image,
                "--worker-port", str(req.worker_port),
                "--vast-worker-count", str(req.worker_count),
                "--vast-comfy-port", str(req.comfy_port),
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
            worker_urls = extract_worker_urls(output_text)
            job["worker_urls"] = worker_urls
            job["worker_url"] = (worker_urls[0] if worker_urls else extract_worker_url(output_text)) or None
            job["ssh_command"] = _extract_ssh_command(output_text) or _last_ssh_command() or None

            if proc.returncode == 0:
                job["status"] = "done"
                if worker_urls:
                    _put(job, f"[vast] Worker URLs: {', '.join(worker_urls)}")
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


@app.post("/api/provision-verda")
async def start_verda_provision(req: VerdaProvisionRequest):
    if not req.fresh:
        try:
            _verda_validate_existing_volumes(req)
        except RuntimeError as exc:
            raise HTTPException(400, f"Existing-volume deploy blocked: {exc}") from exc

    job_id = str(uuid.uuid4())
    q: queue.Queue[str | None] = queue.Queue()
    _jobs[job_id] = {
        "status": "running",
        "logs": [],
        "queue": q,
        "worker_url": None,
        "worker_urls": [],
        "ssh_command": None,
    }

    def _run() -> None:
        job = _jobs[job_id]
        try:
            from gpu_worker.deploy_gpu import extract_worker_url, extract_worker_urls

            command = [
                sys.executable, "-u", str(SCRIPT_DIR / "deploy_gpu.py"),
                "--verda-fresh" if req.fresh else "--verda",
                "--verda-location", req.location,
                "--verda-instance-type", req.instance_type,
                "--verda-contract", req.contract,
                "--verda-ssh-key-id", req.ssh_key_id,
                "--verda-hostname", req.hostname,
                "--verda-worker-count", str(req.worker_count),
                "--verda-comfy-port", str(req.comfy_port),
                "--worker-port", str(req.worker_port),
                "--remote-root", req.remote_root,
            ]
            if req.fresh:
                command.extend([
                    "--verda-fresh-os-volume-size", str(req.fresh_os_volume_size),
                    "--verda-fresh-storage-size", str(req.fresh_storage_size),
                ])
                if req.fresh_os_volume_name:
                    command.extend(["--verda-fresh-os-volume-name", req.fresh_os_volume_name])
                if req.fresh_storage_name:
                    command.extend(["--verda-fresh-storage-name", req.fresh_storage_name])
                for asset_group in req.warm_asset_groups:
                    if asset_group:
                        command.extend(["--warm-asset-group", asset_group])
            else:
                command.extend([
                    "--verda-os-volume-id", req.os_volume_id,
                    "--verda-data-volume-id", req.data_volume_id,
                ])
            if req.skip_warmup:
                command.append("--skip-warmup")

            env_vars = dict(req.env_vars or {})
            env_vars.setdefault("WORKER_PROVIDER", "verda")
            for key, value in env_vars.items():
                if key and value:
                    command.extend(["--env", f"{key}={value}"])

            _put(
                job,
                (
                    f"[verda] Fresh install: creating {req.instance_type} in {req.location} "
                    f"with {req.fresh_storage_size}GB model storage ({req.contract})…"
                    if req.fresh
                    else f"[verda] Creating {req.instance_type} in {req.location} from OS volume "
                    f"{req.os_volume_id} with data volume {req.data_volume_id} ({req.contract})…"
                ),
            )
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
            worker_urls = extract_worker_urls(output_text)
            if not worker_urls:
                for line in output_text.splitlines():
                    if line.startswith("WORKER_URLS="):
                        worker_urls = [
                            url.strip()
                            for url in line.split("=", 1)[1].split(",")
                            if url.strip()
                        ]
                        break
            worker_url = worker_urls[0] if worker_urls else (extract_worker_url(output_text) or None)

            job["worker_urls"] = worker_urls
            job["worker_url"] = worker_url
            job["ssh_command"] = _extract_ssh_command(output_text) or _last_ssh_command() or None

            if proc.returncode == 0:
                job["status"] = "done"
                if worker_urls:
                    _put(job, f"[verda] Worker URLs: {', '.join(worker_urls)}")
                _put(job, "[verda] Provision complete")
            else:
                job["status"] = "failed"
                _put(job, f"[verda] Provision failed (exit {proc.returncode})")
        except Exception as exc:
            _put(job, f"[verda] ERROR: {exc}")
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
    return {
        "status": j["status"],
        "worker_url": j.get("worker_url"),
        "worker_urls": j.get("worker_urls") or [],
        "ssh_command": j.get("ssh_command"),
    }


@app.get("/api/remote-logs/{source}/stream")
async def stream_remote_log(source: str, ssh: str | None = None):
    if source != "downloads" and source not in REMOTE_LOG_SOURCES:
        raise HTTPException(404, "Unknown log source")
    label, paths = REMOTE_LOG_SOURCES.get(source, ("Downloads", ()))

    async def generate() -> AsyncGenerator[str, None]:
        ssh_command = ssh or _last_ssh_command()
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


@app.get("/api/remote-summary")
async def remote_summary(ssh: str):
    try:
        from gpu_worker.deploy_gpu import add_default_host_key_policy, add_default_identity, parse_ssh_command
        ssh_cmd, _, _ = parse_ssh_command(ssh)
        ssh_cmd = add_default_host_key_policy(add_default_identity(ssh_cmd))
        ssh_cmd = [ssh_cmd[0], "-q", "-o", "LogLevel=ERROR", *ssh_cmd[1:]]
    except Exception as exc:
        raise HTTPException(400, f"Invalid SSH command: {exc}") from exc

    script = r"""set -euo pipefail
fetch_stats() {
  local unit="$1" port="$2"
  local stats_json
  stats_json=$(curl -fsS --max-time 2 "http://127.0.0.1:${port}/stats" 2>/dev/null | tr '\n\t' '  ' || true)
  if test -n "$stats_json"; then
    printf 'WORKER_STATS\t%s\t%s\n' "$unit" "$stats_json"
  fi
}
if command -v systemctl >/dev/null 2>&1; then
  units=$(systemctl list-units --type=service --all --no-legend 'filmforge-worker-gpu*.service' 2>/dev/null \
          | awk '{print $1}' | sort -V || true)
  for unit in $units; do
    catout=$(systemctl cat "$unit" 2>/dev/null || true)
    active=$(systemctl is-active "$unit" 2>/dev/null || true)
    url=$(printf '%s\n' "$catout" | sed -n 's/^Environment=WORKER_PUBLIC_URL=//p' | tail -n 1 | sed 's/^"//;s/"$//')
    port=$(printf '%s\n' "$catout" | sed -n 's/^Environment=WORKER_PORT=//p' | tail -n 1 | sed 's/^"//;s/"$//')
    name=$(printf '%s\n' "$catout" | sed -n 's/^Environment=WORKER_NAME=//p' | tail -n 1 | sed 's/^"//;s/"$//')
    if test -n "$url"; then
      printf 'WORKER_URL\t%s\t%s\t%s\t%s\n' "$unit" "$active" "$url" "$name"
    elif test -n "$port"; then
      printf 'WORKER_PORT\t%s\t%s\t%s\t%s\n' "$unit" "$active" "$port" "$name"
    fi
    if test -n "$port"; then
      fetch_stats "$unit" "$port"
    fi
  done
  download=$(journalctl -u 'filmforge-worker-gpu*.service' --since '2 minutes ago' --no-pager 2>/dev/null \
    | grep -E 'Downloading asset progress:|Downloading asset:' \
    | tail -n 1 || true)
  if test -n "$download"; then
    printf 'DOWNLOAD\t%s\n' "$download"
  fi
fi

# Vast ComfyUI images usually run without systemd. In that case the multi-GPU
# deploy writes per-worker health files and tunnel logs under /tmp.
for health in /tmp/filmforge_worker_gpu*_health.json; do
  test -s "$health" || continue
  idx=$(printf '%s' "$health" | grep -oE 'gpu[0-9]+' | grep -oE '[0-9]+' | tail -n 1)
  port=$((9000 + idx))
  name=$(python3 - "$health" <<'PY' 2>/dev/null || true
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("worker_name") or "")
except Exception:
    pass
PY
)
  url=$(python3 - "$health" <<'PY' 2>/dev/null || true
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("public_url") or "")
except Exception:
    pass
PY
)
  if test -z "$url"; then
    log="/tmp/filmforge_gpu_worker_tunnel_gpu${idx}.log"
    url=$(grep -aEo 'https://[-a-z0-9]+\.trycloudflare\.com' "$log" 2>/dev/null | tail -n 1 || true)
  fi
  if test -n "$url"; then
    printf 'WORKER_URL\tfilmforge-worker-gpu%s\tunknown\t%s\t%s\n' "$idx" "$url" "$name"
  elif ss -ltn 2>/dev/null | grep -q ":${port} "; then
    printf 'WORKER_PORT\tfilmforge-worker-gpu%s\tactive\t%s\t%s\n' "$idx" "$port" "$name"
  fi
  fetch_stats "filmforge-worker-gpu${idx}" "$port"
done
"""
    proc = subprocess.run(
        [*ssh_cmd, "bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if proc.returncode != 0:
        raise HTTPException(502, (proc.stderr or proc.stdout or "Remote summary failed").strip())

    worker_urls: list[dict] = []
    stats_by_unit: dict[str, list[dict]] = {}
    download_line = ""
    for line in proc.stdout.splitlines():
        if line.startswith("WORKER_URL\t"):
            _kind, unit, active, url, name = (line.split("\t", 4) + ["", "", "", "", ""])[:5]
            worker_urls.append({"unit": unit, "active": active, "url": url, "name": name})
        elif line.startswith("WORKER_PORT\t"):
            _kind, unit, active, port, name = (line.split("\t", 4) + ["", "", "", "", ""])[:5]
            worker_urls.append({"unit": unit, "active": active, "port": port, "name": name})
        elif line.startswith("DOWNLOAD\t"):
            download_line = line.split("\t", 1)[1]
        elif line.startswith("WORKER_STATS\t"):
            parts = line.split("\t", 2)
            if len(parts) == 3:
                _kind, unit, raw_json = parts
                try:
                    parsed = json.loads(raw_json)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    groups = parsed.get("groups") or []
                    if isinstance(groups, list):
                        stats_by_unit[unit] = groups
    for entry in worker_urls:
        unit = entry.get("unit") or ""
        if unit in stats_by_unit:
            entry["stats"] = stats_by_unit[unit]
    return {"worker_urls": worker_urls, "download_line": download_line}


@app.get("/api/verda/hosts")
async def list_verda_hosts():
    return _verda_hosts_with_live_vms()


@app.get("/api/verda/volumes")
async def list_verda_volumes():
    if not VERDA_CLI.exists():
        raise HTTPException(503, f"Verda CLI not found at {VERDA_CLI}")
    try:
        return _verda_list_volumes()
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.get("/api/verda/vms")
async def list_verda_vms():
    if not VERDA_CLI.exists():
        raise HTTPException(503, f"Verda CLI not found at {VERDA_CLI}")
    try:
        return _verda_list_vms()
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/verda/hosts")
async def add_verda_host(req: VerdaHostRequest):
    name = req.name.strip()
    ssh = _normalize_ssh_command(req.ssh_command)
    if not name or not ssh:
        raise HTTPException(400, "name and ssh_command are required")
    hosts = _load_verda_hosts()
    hosts = [h for h in hosts if h["name"] != name]
    hosts.append({"name": name, "ssh_command": ssh})
    _save_verda_hosts(hosts)
    return {"name": name, "ssh_command": ssh}


@app.delete("/api/verda/hosts/{name}")
async def remove_verda_host(name: str):
    hosts = _load_verda_hosts()
    remaining = [h for h in hosts if h["name"] != name]
    if len(remaining) == len(hosts):
        raise HTTPException(404, f"No Verda host named {name!r}")
    _save_verda_hosts(remaining)
    return {"ok": True}


@app.post("/api/verda/teardown")
async def teardown_verda(req: VerdaTeardownRequest):
    if not VERDA_CLI.exists():
        raise HTTPException(503, f"Verda CLI not found at {VERDA_CLI}")
    try:
        vm = _verda_find_vm_for_teardown(req)
        instance_id = str(vm.get("id") or "")
        volume_ids = [str(v) for v in (vm.get("volume_ids") or []) if v]
        output_payload = _verda_delete_instance(instance_id, volume_ids, delete_volumes=req.delete_volumes)
        _verda_wait_for_vm_deleted(instance_id)
        if volume_ids and not req.delete_volumes:
            _verda_wait_for_detached(volume_ids)

        remaining_volumes = _verda_list_volumes()
        remaining_by_id = {
            str(v.get("id")): v for v in remaining_volumes if isinstance(v, dict) and v.get("id")
        }
        trash_proc = subprocess.run(
            [str(VERDA_CLI), "--agent", "volume", "trash"],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        trash: list[dict] = []
        if trash_proc.returncode == 0:
            try:
                payload = json.loads(trash_proc.stdout or "[]")
                trash = payload if isinstance(payload, list) else []
            except json.JSONDecodeError:
                trash = []
        trashed_ids = {
            str(v.get("id")) for v in trash if isinstance(v, dict) and v.get("id")
        }
        preserved = [remaining_by_id[vol_id] for vol_id in volume_ids if vol_id in remaining_by_id]
        deleted_or_trashed = [vol_id for vol_id in volume_ids if vol_id not in remaining_by_id or vol_id in trashed_ids]
        if volume_ids and not req.delete_volumes:
            unsafe = [vol_id for vol_id in volume_ids if vol_id not in remaining_by_id or vol_id in trashed_ids]
            if unsafe:
                raise RuntimeError(
                    "Verda delete completed, but expected preserved volume(s) are missing/trashed: "
                    + ", ".join(unsafe)
                )

        if req.name:
            hosts = [h for h in _load_verda_hosts() if h.get("name") != req.name]
            _save_verda_hosts(hosts)

        return {
            "ok": True,
            "instance_id": instance_id,
            "delete_volumes": req.delete_volumes,
            "volume_ids": volume_ids,
            "preserved_volumes": preserved,
            "deleted_or_trashed_volume_ids": deleted_or_trashed,
            "output": output_payload,
        }
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


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
.inst-card{background:#13161f;border:1px solid #252a38;border-radius:8px;padding:12px 14px;margin-bottom:8px;transition:border-color .15s;cursor:pointer}
.inst-card:last-child{margin-bottom:0}
.inst-card:hover{border-color:#3d4461}
.inst-card.selected{border-color:#7c3aed;background:#16102a}
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
.b-generating{background:#0d2b1e;color:#34d399;border:1px solid #1a4d35}.b-generating .dot{background:#34d399;animation:pulse 1s infinite}
.b-idle{background:#1c1f2a;color:#94a3b8;border:1px solid #2d3148}.b-idle .dot{background:#64748b}
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
.tab-close{margin-left:5px;opacity:.5;font-size:12px;line-height:1;vertical-align:middle;transition:opacity .15s}
.tab-close:hover{opacity:1;color:#f87171}

/* Config grid */
.cfg-grid{display:grid;grid-template-columns:120px 1fr;gap:8px 10px;align-items:center;margin-bottom:8px}
.cfg-label{font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.06em}
.cfg-custom{grid-column:2;margin-top:-4px;margin-bottom:8px}
.cfg-custom-input{width:100%;padding:7px 10px;background:#0c0f16;border:1px solid #2d3148;border-radius:5px;color:#e2e8f0;font-size:12px;font-family:inherit;transition:border-color .15s}
.cfg-custom-input:focus{outline:none;border-color:#7c3aed}
#sbar{font-size:11px;padding:3px 10px;border-radius:4px;display:none;margin-left:auto}
#sbar.running{background:#0f2744;color:#93c5fd;display:inline-block}
#sbar.done{background:#0d2b1e;color:#6ee7b7;display:inline-block}
#sbar.failed{background:#2d1010;color:#fca5a5;display:inline-block}
#log{flex:1;overflow-y:auto;padding:12px 16px;font-family:'Courier New',monospace;font-size:11.5px;line-height:1.7;background:#080b14;min-height:0}
.ll{color:#e2e8f0}
.ll.info{color:#93c5fd}.ll.ok{color:#94a3b8}.ll.signal{color:#6ee7b7}.ll.err{color:#fca5a5}.ll.warn{color:#fde68a}
.ll.gpu0{border-left:2px solid #60a5fa;padding-left:6px}
.ll.gpu1{border-left:2px solid #f472b6;padding-left:6px}
.ll.gpu2{border-left:2px solid #fbbf24;padding-left:6px}
.ll.gpu3{border-left:2px solid #34d399;padding-left:6px}

/* Workers */
.workers-pane{flex-shrink:0;border-top:1px solid #2d3148;max-height:200px;overflow-y:auto}
.workers-hdr{display:flex;align-items:center;padding:8px 16px;background:#0f1117;position:sticky;top:0;border-bottom:1px solid #1e2130}
.wrow{display:flex;align-items:center;gap:10px;padding:8px 16px;border-bottom:1px solid #13161f}
.wrow:last-child{border-bottom:none}
.wname{font-size:12px;color:#e2e8f0;font-weight:500}
.wurl{font-size:10px;color:#3d4461;font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:160px}

/* Instance header */
.inst-header{padding:10px 16px;border-bottom:1px solid #2d3148;flex-shrink:0;background:#0f1117}
.inst-header-info{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.inst-header-name{font-size:13px;font-weight:600;color:#e2e8f0}
.inst-header-meta{font-size:11px;color:#475569;font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:340px}
.inst-tools{margin-top:9px;display:grid;gap:6px}
.url-row{display:grid;grid-template-columns:1fr auto;gap:6px;align-items:center}
.url-pill{font-size:11px;color:#cbd5e1;background:#080b14;border:1px solid #1e2130;border-radius:5px;padding:6px 8px;font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.copy-btn{border:1px solid #2d3148;background:#171a24;color:#94a3b8;border-radius:5px;font-size:10px;padding:6px 8px;cursor:pointer}
.copy-btn:hover{color:#e2e8f0;border-color:#475569}
.dl-status{font-size:11px;color:#6ee7b7;background:#071512;border:1px solid #12372e;border-radius:5px;padding:6px 8px;font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.worker-stats{font-size:10px;color:#94a3b8;font-family:monospace;padding:2px 8px 0 2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.empty-tools{font-size:11px;color:#475569}

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

    <!-- RunPod Pods -->
    <div class="sec">
      <div class="sec-hdr">
        <span class="sec-title">RunPod Pods</span>
        <button class="btn btn-ghost btn-xs" onclick="loadRunpodPods()">↻ Refresh</button>
      </div>
      <div id="runpod-out"><span class="muted text-xs">Loading…</span></div>
    </div>

    <!-- Verda Hosts -->
    <div class="sec">
      <div class="sec-hdr">
        <span class="sec-title">Verda Hosts</span>
        <button class="btn btn-ghost btn-xs" onclick="loadVerdaHosts()">↻ Refresh</button>
      </div>
      <div id="verda-out"><span class="muted text-xs">Loading…</span></div>
      <div style="margin-top:10px;padding-top:10px;border-top:1px dashed #2d3148">
        <div class="fg" style="margin-bottom:6px"><label>Name</label><input id="verda-name" placeholder="filmforge-2a100-worker"></div>
        <div class="fg" style="margin-bottom:8px"><label>SSH</label><input id="verda-ssh" placeholder="ssh root@135.181.8.209"></div>
        <button class="btn btn-ghost btn-xs btn-full" onclick="addVerdaHost()">+ Add Verda host</button>
      </div>
    </div>

    <!-- Create & Deploy (Verda) -->
    <div class="sec">
      <div class="sec-hdr mb0">
        <span class="sec-title mb0">Create &amp; Deploy (Verda)</span>
      </div>
      <div class="fg">
        <label>Mode</label>
        <select id="v-mode" onchange="updateVerdaMode()">
          <option value="existing">Existing volumes</option>
          <option value="fresh">Fresh install + model download</option>
        </select>
      </div>
      <div class="fg">
        <label>GPU Preference</label>
        <select id="v-gpu-preference" onchange="loadVerdaAvailability()">
          <option value="single">Best single GPU</option>
          <option value="sprint">Multi-GPU sprint</option>
          <option value="four_plus">4+ GPU sprint</option>
          <option value="any">Any GPU</option>
        </select>
      </div>
      <div class="grid2" style="margin-bottom:10px">
        <div class="fg" style="margin-bottom:0">
          <label>Contract</label>
          <select id="v-contract" onchange="loadVerdaAvailability(); refreshVerdaCostEstimate()">
            <option value="pay_as_go">On-demand</option>
            <option value="spot">Spot</option>
          </select>
        </div>
        <div class="fg" style="margin-bottom:0">
          <label>GPU Options</label>
          <select id="v-option-limit" onchange="loadVerdaAvailability()">
            <option value="5">Top 5</option>
            <option value="10">Top 10</option>
            <option value="all">All available</option>
          </select>
        </div>
      </div>
      <div class="grid2" style="margin-bottom:10px">
        <div class="fg" style="margin-bottom:0">
          <label>Location</label>
          <input id="v-location" value="FIN-01" onchange="autofillVerdaVolumeIds(); validateVerdaExistingVolumes(); refreshVerdaCostEstimate()">
        </div>
        <div class="fg" style="margin-bottom:0">
          <label>Instance Type</label>
          <select id="v-instance-select" onchange="applyVerdaSelection()">
            <option value="FIN-01|2A100.44V">FIN-01 · 2A100.44V · fallback</option>
          </select>
          <input id="v-instance-type" value="2A100.44V" style="margin-top:6px">
        </div>
      </div>
      <div class="flex gap6 items-center" style="margin-bottom:10px">
        <button class="btn btn-ghost btn-xs" onclick="loadVerdaAvailability()">Refresh GPUs</button>
        <span id="v-availability-status" class="muted text-xs">Top Verda GPUs</span>
      </div>
      <div id="v-cost-estimate" class="muted text-xs" style="margin:-2px 0 10px 0">Cost estimate pending.</div>
      <div id="v-existing-fields">
        <div class="fg"><label>OS Volume</label><input id="v-os-volume" placeholder="Detached OS volume ID" onchange="validateVerdaExistingVolumes()"></div>
        <div class="fg"><label>Data Volume</label><input id="v-data-volume" placeholder="Detached model/data volume ID" onchange="validateVerdaExistingVolumes()"></div>
        <div class="flex gap6 items-center" style="margin-bottom:10px">
          <button class="btn btn-ghost btn-xs" onclick="loadVerdaVolumes()">Refresh Volumes</button>
          <span id="v-volume-status" class="muted text-xs">Volume status pending.</span>
        </div>
      </div>
      <div id="v-fresh-fields" style="display:none">
        <div class="grid2">
          <div class="fg"><label>OS GB</label><input id="v-fresh-os-size" type="number" value="100" min="50" onchange="refreshVerdaCostEstimate()"></div>
          <div class="fg"><label>Storage GB</label><input id="v-fresh-storage-size" type="number" value="250" min="100" onchange="refreshVerdaCostEstimate()"></div>
        </div>
        <div class="fg"><label>Preload Models</label>
          <select id="v-fresh-warm">
            <option value="flux_stills_v1,juggernaut_stills_v1,wan_i2v_v1,stable_audio_v1">Flux + Juggernaut + WAN + Audio</option>
            <option value="flux_stills_v1,wan_i2v_v1,stable_audio_v1">Flux + WAN + Audio</option>
            <option value="juggernaut_stills_v1,wan_i2v_v1,stable_audio_v1">Juggernaut + WAN + Audio</option>
            <option value="flux_stills_v1,wan_i2v_v1">Flux + WAN</option>
            <option value="juggernaut_stills_v1,wan_i2v_v1">Juggernaut + WAN</option>
            <option value="flux_stills_v1">Flux only</option>
            <option value="juggernaut_stills_v1">Juggernaut only</option>
            <option value="">Skip model preload</option>
          </select>
        </div>
      </div>
      <button class="adv-toggle" onclick="toggleVerdaAdv()">▶ Advanced options</button>
      <div id="v-adv-body" style="display:none;margin-top:12px;padding-top:12px;border-top:1px solid #1e2130">
        <div class="fg"><label>Hostname</label><input id="v-hostname" value="filmforge-verda-worker"></div>
        <div class="fg"><label>SSH Key ID</label><input id="v-ssh-key" value="11ee08a4-858a-4ee7-98c8-250aad99eb37"></div>
        <div class="grid2">
          <div class="fg"><label>Workers</label><input id="v-worker-count" type="number" value="0" min="0"></div>
          <div class="fg"><label>Comfy Port</label><input id="v-comfy-port" type="number" value="8188"></div>
        </div>
      </div>
      <button class="btn btn-success btn-full" id="verda-provision-btn" onclick="quickDeployVerda()" style="margin-top:8px">
        ⚡ Create &amp; Deploy (Verda)
      </button>
    </div>

    <!-- Create & Deploy (Vast) -->
    <div class="sec">
      <div class="sec-hdr mb0">
        <span class="sec-title mb0">Create &amp; Deploy (Vast)</span>
      </div>
      <div class="grid2" style="margin-bottom:10px">
        <div class="fg" style="margin-bottom:0">
          <label>GPU</label>
          <select id="c-gpu">
            <option>L40S</option><option>A100 SXM</option><option>A100 PCIe</option>
            <option>H100 PCIe</option><option>H100 SXM</option>
            <option>RTX 6000 Ada</option><option>RTX PRO 6000</option>
            <option>A6000</option><option>L4</option><option>RTX 4090</option><option>B200</option>
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
        <div class="grid2">
          <div class="fg"><label>Worker Port</label><input id="c-port" type="number" value="9000"></div>
          <div class="fg"><label>Workers (0 = auto)</label><input id="c-worker-count" type="number" value="0" min="0"></div>
        </div>
        <div class="fg"><label>Comfy Port</label><input id="c-comfy-port" type="number" value="18188"></div>
        <button class="btn btn-ghost btn-xs" onclick="autoDetectTemplate()" style="margin-bottom:6px">↻ Auto-detect template</button>
        <div id="tmpl-status" class="muted text-xs"></div>
      </div>
    </div>

    <!-- Create & Deploy (RunPod) -->
    <div class="sec">
      <div class="sec-hdr mb0">
        <span class="sec-title mb0">Create &amp; Deploy (RunPod)</span>
      </div>
      <div class="fg" style="margin-bottom:10px">
        <label>GPU Type</label>
        <select id="rp-gpu">
          <option value="NVIDIA L40S">L40S</option>
          <option value="NVIDIA A100-SXM4-80GB">A100 SXM 80GB</option>
          <option value="NVIDIA A100 80GB PCIe">A100 PCIe 80GB</option>
          <option value="NVIDIA H100 80GB HBM3">H100 SXM</option>
          <option value="NVIDIA H100 PCIe">H100 PCIe</option>
          <option value="NVIDIA GeForce RTX 4090">RTX 4090</option>
          <option value="NVIDIA GeForce RTX 5090">RTX 5090</option>
          <option value="NVIDIA RTX 6000 Ada Generation">RTX 6000 Ada</option>
          <option value="NVIDIA RTX A6000">RTX A6000</option>
        </select>
      </div>
      <button class="adv-toggle" onclick="toggleRpAdv()">▶ Advanced options</button>
      <div id="rp-adv-body" style="display:none;margin-top:12px;padding-top:12px;border-top:1px solid #1e2130">
        <div class="fg">
          <label>Cloud Type</label>
          <select id="rp-cloud">
            <option value="COMMUNITY">Community (cheaper, more available)</option>
            <option value="SECURE">Secure (dedicated, higher cost)</option>
          </select>
        </div>
        <div class="grid2">
          <div class="fg"><label>Volume (GB)</label><input id="rp-volume" type="number" value="150"></div>
          <div class="fg"><label>Container Disk (GB)</label><input id="rp-cdisk" type="number" value="50"></div>
        </div>
      </div>
      <button class="btn btn-success btn-full" id="runpod-provision-btn" onclick="quickDeployRunpod()" style="margin-top:8px">
        ⚡ Create &amp; Deploy (RunPod)
      </button>
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

    <!-- Selected instance header -->
    <div id="inst-header" class="inst-header">
      <span class="muted text-xs">← Select an instance from the left panel</span>
    </div>

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
const instData = {};   // instData[instId] = { ssh, logBuffers, evtSrc, status, workerUrl, workerUrls }
let selectedInstId = null;
let currentLogSrc = 'deploy';
let remoteEvt = null;
const MAX_LOG_LINES = 2000;

const LOG_LABELS = {
  deploy: 'Deploy', downloads: 'Downloads',
  worker: 'GPU Worker', comfy: 'ComfyUI', tunnel: 'Tunnel',
};
let verdaVolumes = [];

// ── Per-instance data ─────────────────────────────────────────────────────────
function getInstData(id) {
  if (!instData[id]) {
    instData[id] = {
      ssh: '',
      logBuffers: { deploy: [], downloads: [], worker: [], comfy: [], tunnel: [] },
      evtSrc: null,
      status: null,
      workerUrl: null,
      workerUrls: [],
      workerStats: {},
      downloadStatus: null,
      summaryLoaded: false,
    };
  }
  return instData[id];
}

// ── Instance selection ────────────────────────────────────────────────────────
function selectInstance(id, ssh) {
  selectedInstId = String(id);
  const idata = getInstData(selectedInstId);
  if (ssh) idata.ssh = ssh;
  document.querySelectorAll('.inst-card').forEach(c => {
    c.classList.toggle('selected', c.dataset.id === selectedInstId);
  });
  updateInstHeader();
  renderLog(currentLogSrc);
  refreshSelectedInstanceSummary();
  if (currentLogSrc !== 'deploy') reconnectRemoteLog(currentLogSrc);
}

function updateInstHeader() {
  const el = document.getElementById('inst-header');
  if (!selectedInstId) {
    el.innerHTML = '<span class="muted text-xs">← Select an instance from the left panel</span>';
    return;
  }
  const idata = getInstData(selectedInstId);
  const statusHtml = idata.status
    ? `<span class="badge ${idata.status === 'done' ? 'b-running' : 'b-exited'}" style="font-size:10px"><span class="dot"></span>${esc(idata.status)}</span>`
    : '';
  const urls = normalizedWorkerUrls(idata);
  const statsByUrl = idata.workerStats || {};
  const urlHtml = urls.length
    ? `<div class="inst-tools">
        <div class="empty-tools">Worker URLs</div>
        ${urls.map((url, idx) => {
          const statsLine = formatWorkerStats(statsByUrl[url]);
          const statsHtml = statsLine
            ? `<div class="worker-stats" title="${esc(statsLine)}">${esc(statsLine)}</div>`
            : '';
          return `<div class="url-row">
            <div class="url-pill" title="${esc(url)}">${esc(url)}</div>
            <button class="copy-btn" onclick='copyText(${JSON.stringify(url)}, this)'>Copy</button>
          </div>${statsHtml}`;
        }).join('')}
      </div>`
    : `<div class="inst-tools"><span class="empty-tools">No worker URL captured yet.</span></div>`;
  const activeDl = activeDownloadStatus(idata);
  const downloadHtml = activeDl
    ? `<div class="dl-status" title="${esc(activeDl.detail)}">${esc(activeDl.detail)}</div>`
    : '';
  el.innerHTML = `<div class="inst-header-info">
    <span class="inst-header-name">Instance #${esc(selectedInstId)}</span>
    ${idata.ssh ? `<span class="inst-header-meta">${esc(idata.ssh)}</span>` : ''}
    ${statusHtml}
  </div>
  ${urlHtml}
  ${downloadHtml}`;
}

function normalizedWorkerUrls(idata) {
  const urls = [];
  for (const url of idata.workerUrls || []) {
    if (url && !urls.includes(url)) urls.push(url);
  }
  if (idata.workerUrl && !urls.includes(idata.workerUrl)) urls.push(idata.workerUrl);
  return urls;
}

async function refreshSelectedInstanceSummary(force = false) {
  if (!selectedInstId) return;
  const capturedInstId = selectedInstId;
  const idata = getInstData(capturedInstId);
  if (!idata.ssh || (idata.summaryLoaded && !force)) return;
  idata.summaryLoaded = true;
  try {
    const summary = await fetch(`/api/remote-summary?ssh=${encodeURIComponent(idata.ssh)}`).then(r => {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    });
    const urls = [];
    const statsByUrl = {};
    for (const item of summary.worker_urls || []) {
      if (item.url) {
        urls.push(item.url);
        if (Array.isArray(item.stats) && item.stats.length) statsByUrl[item.url] = item.stats;
      }
    }
    if (urls.length) {
      idata.workerUrls = urls;
      idata.workerUrl = urls[0];
    }
    idata.workerStats = statsByUrl;
    if (summary.download_line) {
      updateDownloadStatusFromLine(summary.download_line, idata);
    } else if (idata.downloadStatus) {
      // No active download line in the recent journal → clear stale status.
      idata.downloadStatus = null;
    }
    if (capturedInstId === selectedInstId) updateInstHeader();
  } catch (e) {
    idata.summaryLoaded = false;
  }
}

// ── Instances ─────────────────────────────────────────────────────────────────
async function loadInstances() {
  const el = document.getElementById('instances-out');
  el.innerHTML = '<span class="muted text-xs">Loading…</span>';
  try {
    const list = await fetch('/api/instances').then(r => r.json());
    if (!list.length) {
      el.innerHTML = '<p class="muted text-xs">No instances — use Create &amp; Deploy below.</p>';
      return [];
    }
    el.innerHTML = list.map(inst => renderInstCard(inst)).join('');
    return list;
  } catch (e) {
    el.innerHTML = '<p class="muted text-xs" style="color:#f87171">Failed to load instances.</p>';
    return [];
  }
}

function renderInstCard(inst) {
  const ssh = getSsh(inst);
  const status = (inst.actual_status || '?').toLowerCase();
  const isRunning = status === 'running';
  const isStopped = status === 'exited' || status === 'stopped';
  const isSelected = String(inst.id) === selectedInstId;

  const badgeClass = isRunning ? 'b-running' : isStopped ? 'b-exited' : 'b-loading';
  const price = `$${(inst.dph_total || 0).toFixed(3)}/hr`;
  const meta = ssh ? `${price} · ${ssh}` : price;
  const sshJson = JSON.stringify(ssh);

  const actions = isRunning
    ? `<button class="btn btn-success btn-xs" onclick='event.stopPropagation();deployToInst(${sshJson},"${inst.id}")'>▶ Deploy</button>
       <button class="btn btn-ghost btn-xs" onclick='event.stopPropagation();destroyInst("${inst.id}")'>Destroy</button>`
    : isStopped
    ? `<button class="btn btn-primary btn-xs" onclick='event.stopPropagation();activateInst("${inst.id}")'>Activate</button>
       <button class="btn btn-ghost btn-xs" onclick='event.stopPropagation();destroyInst("${inst.id}")'>Destroy</button>`
    : `<span class="muted text-xs">${status}…</span>
       <button class="btn btn-ghost btn-xs" onclick='event.stopPropagation();destroyInst("${inst.id}")'>Destroy</button>`;

  return `<div class="inst-card${isSelected ? ' selected' : ''}" data-id="${inst.id}"
    onclick='selectInstance("${inst.id}",${sshJson})'>
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

// ── RunPod Pods ───────────────────────────────────────────────────────────────
async function loadRunpodPods() {
  const el = document.getElementById('runpod-out');
  const hasCards = el.querySelector('.inst-card') !== null;
  if (!hasCards) el.innerHTML = '<span class="muted text-xs">Loading…</span>';
  try {
    const list = await fetch('/api/runpod/pods').then(r => r.json());
    if (!list.length) {
      el.innerHTML = '<p class="muted text-xs">No pods — use Create &amp; Deploy below.</p>';
      return;
    }
    const next = list.map(pod => renderRunpodCard(pod)).join('');
    if (el.innerHTML !== next) el.innerHTML = next;
  } catch (e) {
    if (!hasCards) el.innerHTML = '<p class="muted text-xs" style="color:#f87171">Failed to load pods.</p>';
  }
}

function getRunpodSsh(pod) {
  if (!pod.ssh_ip || !pod.ssh_port) return '';
  return `ssh root@${pod.ssh_ip} -p ${pod.ssh_port}`;
}

function renderRunpodCard(pod) {
  const ssh = getRunpodSsh(pod);
  const status = (pod.status || '').toUpperCase();
  const isRunning = status === 'RUNNING';
  const isStopped = status === 'EXITED' || status === 'STOPPED';
  const podKey = `rp-${pod.id}`;
  const isSelected = podKey === selectedInstId;
  const badgeClass = isRunning ? 'b-running' : isStopped ? 'b-exited' : 'b-loading';
  const price = pod.cost_per_hr != null ? `$${(+pod.cost_per_hr).toFixed(3)}/hr` : '';
  const meta = [pod.name, price, ssh].filter(Boolean).join(' · ');
  const sshJson = JSON.stringify(ssh);

  const actions = isRunning && ssh
    ? `<button class="btn btn-success btn-xs" onclick='event.stopPropagation();quickDeployRunpod("${pod.id}")'>▶ Deploy</button>
       <button class="btn btn-warning btn-xs" onclick='event.stopPropagation();stopPod("${pod.id}")'>⏹ Stop</button>
       <button class="btn btn-danger btn-xs" onclick='event.stopPropagation();terminatePod("${pod.id}")'>Terminate</button>`
    : isStopped
    ? `<button class="btn btn-primary btn-xs" onclick='event.stopPropagation();resumePod("${pod.id}")'>▶ Resume</button>
       <button class="btn btn-danger btn-xs" onclick='event.stopPropagation();terminatePod("${pod.id}")'>Terminate</button>`
    : `<span class="muted text-xs">${status.toLowerCase()}…</span>
       <button class="btn btn-danger btn-xs" onclick='event.stopPropagation();terminatePod("${pod.id}")'>Terminate</button>`;

  return `<div class="inst-card${isSelected ? ' selected' : ''}" data-id="${podKey}"
    onclick='selectInstance("${podKey}",${sshJson})'>
    <div class="inst-top">
      <span class="inst-id">${esc(pod.name || pod.id)}</span>
      <span class="inst-gpu">${esc(pod.gpu_name || '?')}</span>
      <span class="badge ${badgeClass}"><span class="dot"></span>${esc(status)}</span>
    </div>
    <div class="inst-meta" title="${esc(meta)}">${esc(meta)}</div>
    <div class="inst-actions">${actions}</div>
  </div>`;
}

async function resumePod(podId) {
  const r = await fetch(`/api/runpod/pods/${podId}/resume`, {method: 'POST'});
  const d = await r.json();
  if (!d.ok) alert(`Failed to resume: ${d.output || ''}`);
  setTimeout(loadRunpodPods, 2000);
}

// ── Verda hosts ───────────────────────────────────────────────────────────────
async function loadVerdaHosts() {
  const el = document.getElementById('verda-out');
  try {
    const hosts = await fetch('/api/verda/hosts').then(r => r.json());
    if (!hosts.length) {
      el.innerHTML = '<span class="muted text-xs">No Verda hosts — add one below.</span>';
      return;
    }
    el.innerHTML = hosts.map(h => {
      const instId = `verda:${h.name}`;
      const isSelected = instId === selectedInstId;
      getInstData(instId).ssh = h.ssh_command;
      const safeName = esc(h.name);
      const safeSsh = esc(h.ssh_command);
      const nameJson = JSON.stringify(h.name);
      const sshJson = JSON.stringify(h.ssh_command);
      const instTypeJson = JSON.stringify(h.instance_type || '');
      const metaParts = [h.location, h.instance_type, h.status]
        .filter(Boolean)
        .map(v => String(v));
      if (h.price_per_hour !== undefined && h.price_per_hour !== null) {
        metaParts.push(`$${Number(h.price_per_hour).toFixed(3)}/hr`);
      }
      const liveBadge = h.source === 'live' ? '<span class="pill ok">live</span> ' : '';
      const meta = metaParts.length ? `${metaParts.join(' · ')} · ${safeSsh}` : safeSsh;
      const removeButton = h.source === 'live'
        ? ''
        : `<button class="btn btn-ghost btn-xs" onclick="event.stopPropagation(); removeVerdaHost('${safeName}')">Remove</button>`;
      return `<div class="inst-card${isSelected ? ' selected' : ''}" data-id="${esc(instId)}"
                   onclick="selectVerdaHost('${safeName}', '${safeSsh.replace(/'/g, "&#39;")}')">
        <div class="inst-header-name">${liveBadge}${safeName}</div>
        <div class="inst-meta" title="${esc(meta)}">${esc(meta)}</div>
        <div class="inst-actions">
          <button class="btn btn-success btn-xs" onclick='event.stopPropagation(); redeployVerdaHost(${nameJson}, ${sshJson}, ${instTypeJson})'>Redeploy</button>
          <button class="btn btn-warning btn-xs" onclick="event.stopPropagation(); teardownVerda('${safeName}', '${safeSsh.replace(/'/g, "&#39;")}', false)">Delete VM</button>
          <button class="btn btn-danger btn-xs" onclick="event.stopPropagation(); teardownVerda('${safeName}', '${safeSsh.replace(/'/g, "&#39;")}', true)">Delete VM+Volumes</button>
          ${removeButton}
        </div>
      </div>`;
    }).join('');
  } catch (e) {
    el.innerHTML = '<p class="muted text-xs" style="color:#f87171">Failed to load Verda hosts.</p>';
  }
}

function selectVerdaHost(name, ssh) {
  selectInstance(`verda:${name}`, ssh);
}

async function redeployVerdaHost(name, ssh, instanceType) {
  const verdaWorkerPort = Number((document.getElementById('v-worker-port') || {}).value);
  const fallbackWorkerPort = Number((document.getElementById('c-port') || {}).value);
  const verdaWorkerCount = Number((document.getElementById('v-worker-count') || {}).value);
  const verdaComfyPort = Number((document.getElementById('v-comfy-port') || {}).value);
  const workerPort = verdaWorkerPort || fallbackWorkerPort || 9000;
  const workerCount = verdaWorkerCount || 0;
  const host = sshHost(ssh);
  const env = {};
  if (host) {
    env.WORKER_PUBLIC_URL = `http://${host}:${workerPort}`;
    if (workerCount > 1) {
      env.WORKER_PUBLIC_URLS = Array.from({length: workerCount}, (_, idx) => `http://${host}:${workerPort + idx}`).join(',');
    }
  }
  await deployToInst(ssh, `verda:${name}`, {
    provider: 'verda',
    gpuName: instanceType || document.getElementById('v-instance-type')?.value || 'Verda',
    workerPort,
    workerCount,
    comfyPort: verdaComfyPort || 8188,
    remoteRoot: document.getElementById('remote-root')?.value || '/workspace/filmforge_gpu_worker',
    env,
  });
}

async function addVerdaHost() {
  const name = (document.getElementById('verda-name').value || '').trim();
  const ssh = (document.getElementById('verda-ssh').value || '').trim();
  if (!name || !ssh) { alert('Name and SSH are required.'); return; }
  const r = await fetch('/api/verda/hosts', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name, ssh_command: ssh}),
  });
  if (!r.ok) { alert(`Failed to add: ${(await r.json()).detail || r.status}`); return; }
  document.getElementById('verda-name').value = '';
  document.getElementById('verda-ssh').value = '';
  loadVerdaHosts();
}

async function removeVerdaHost(name) {
  if (!confirm(`Remove Verda host ${name}?`)) return;
  const r = await fetch(`/api/verda/hosts/${encodeURIComponent(name)}`, {method: 'DELETE'});
  if (!r.ok) { alert(`Failed to remove: ${(await r.json()).detail || r.status}`); return; }
  if (selectedInstId === `verda:${name}`) selectedInstId = null;
  loadVerdaHosts();
  updateInstHeader();
}

async function teardownVerda(name, ssh, deleteVolumes) {
  const mode = deleteVolumes ? 'delete the VM and its attached OS/model volumes' : 'detach volumes first, then delete only the VM';
  const warning = deleteVolumes
    ? 'This moves attached volumes toward deletion/trash. Use this only for disposable test volumes.'
    : 'This preserves the OS/model volumes for a later Existing volumes deploy.';
  if (!confirm(`Verda teardown: ${mode}?\n\n${warning}\n\nHost: ${name}`)) return;
  const expected = deleteVolumes ? 'DELETE VOLUMES' : 'DELETE VM';
  const typed = prompt(`Type ${expected} to confirm.`);
  if (typed !== expected) return;
  try {
    const r = await fetch('/api/verda/teardown', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name, ssh_command: ssh, delete_volumes: deleteVolumes}),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.status);
    const preserved = (d.preserved_volumes || []).map(v => `${v.name || v.id}: ${v.status}`).join(', ');
    const deleted = (d.deleted_or_trashed_volume_ids || []).join(', ');
    alert(`Verda teardown complete.\nInstance: ${d.instance_id}\nPreserved volumes: ${preserved || 'none'}\nDeleted/trashed volumes: ${deleted || 'none'}`);
    if (selectedInstId === `verda:${name}`) selectedInstId = null;
    await loadVerdaHosts();
    await loadVerdaVolumes();
    updateInstHeader();
  } catch (e) {
    alert(`Verda teardown failed: ${e}`);
  }
}

async function loadVerdaAvailability() {
  const sel = document.getElementById('v-instance-select');
  const status = document.getElementById('v-availability-status');
  const preference = document.getElementById('v-gpu-preference')?.value || 'single';
  const optionLimit = document.getElementById('v-option-limit')?.value || '5';
  const contract = document.getElementById('v-contract')?.value || 'pay_as_go';
  if (!sel || !status) return;
  status.textContent = 'Checking Verda GPUs…';
  try {
    const params = new URLSearchParams({preference, contract});
    const payload = await fetch(`/api/verda/availability?${params.toString()}`).then(r => {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    });
    const allItems = payload.all_items || payload.items || [];
    const items = optionLimit === 'all' ? allItems : allItems.slice(0, Number(optionLimit) || 5);
    if (!items.length) {
      status.textContent = 'No GPU capacity found';
      return;
    }
    sel.innerHTML = items.map(item => {
      const value = `${item.location}|${item.instance_type}`;
      return `<option value="${esc(value)}">${esc(item.label)}</option>`;
    }).join('');
    applyVerdaSelection();
    const label = {
      single: 'single-GPU',
      sprint: 'multi-GPU sprint',
      four_plus: '4+ GPU sprint',
      any: 'GPU',
    }[preference] || 'GPU';
    status.textContent = `${items.length} of ${allItems.length} ${label} option(s)`;
  } catch (e) {
    status.textContent = `Availability unavailable: ${e}`;
  }
}

function applyVerdaSelection() {
  const sel = document.getElementById('v-instance-select');
  if (!sel || !sel.value) return;
  const [location, instanceType] = sel.value.split('|');
  if (location) document.getElementById('v-location').value = location;
  if (instanceType) document.getElementById('v-instance-type').value = instanceType;
  refreshVerdaCostEstimate();
  validateVerdaExistingVolumes();
}

async function loadVerdaVolumes() {
  const status = document.getElementById('v-volume-status');
  if (status) status.textContent = 'Checking volumes…';
  try {
    verdaVolumes = await fetch('/api/verda/volumes').then(r => {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    });
    autofillVerdaVolumeIds();
    return validateVerdaExistingVolumes();
  } catch (e) {
    if (status) status.textContent = `Volume check unavailable: ${e}`;
    return false;
  }
}

function autofillVerdaVolumeIds() {
  if (!Array.isArray(verdaVolumes) || !verdaVolumes.length) return;
  const loc = (document.getElementById('v-location').value || '').trim();
  const osInput = document.getElementById('v-os-volume');
  const dataInput = document.getElementById('v-data-volume');
  const ids = new Set(verdaVolumes.map(v => v.id));
  const detached = v => String(v.status || '').toLowerCase() === 'detached';
  const byNewest = (a, b) => String(b.created_at || '').localeCompare(String(a.created_at || ''));
  const candidatesFor = location => verdaVolumes
    .filter(v => !location || v.location === location)
    .slice()
    .sort(byNewest);
  const findPair = volumes => {
    const os = volumes.find(v => detached(v) && v.is_os_volume);
    const data = volumes.find(v => detached(v) && !v.is_os_volume);
    return os && data && os.location === data.location ? {os, data} : null;
  };
  const sameLocationPair = findPair(candidatesFor(loc));
  const locations = [...new Set(verdaVolumes.map(v => v.location).filter(Boolean))];
  const anyLocationPair = locations
    .map(location => findPair(candidatesFor(location)))
    .find(Boolean);
  const pair = sameLocationPair || anyLocationPair;
  const candidates = candidatesFor(loc);
  const osCandidate = pair?.os || candidates.find(v => v.is_os_volume) || verdaVolumes.find(v => v.is_os_volume);
  const dataCandidate = pair?.data || candidates.find(v => !v.is_os_volume) || verdaVolumes.find(v => !v.is_os_volume);
  if (pair && pair.os.location && pair.os.location !== loc) {
    document.getElementById('v-location').value = pair.os.location;
  }
  if (osInput && osCandidate && (!osInput.value.trim() || !ids.has(osInput.value.trim()))) {
    osInput.value = osCandidate.id;
  }
  if (dataInput && dataCandidate && (!dataInput.value.trim() || !ids.has(dataInput.value.trim()))) {
    dataInput.value = dataCandidate.id;
  }
}

function validateVerdaExistingVolumes() {
  const status = document.getElementById('v-volume-status');
  if (!status || document.getElementById('v-mode').value === 'fresh') return true;
  const loc = (document.getElementById('v-location').value || '').trim();
  const osId = (document.getElementById('v-os-volume').value || '').trim();
  const dataId = (document.getElementById('v-data-volume').value || '').trim();
  if (!osId || !dataId) {
    status.textContent = 'Choose detached OS and model/data volumes.';
    status.style.color = '#fbbf24';
    return false;
  }
  if (!Array.isArray(verdaVolumes) || !verdaVolumes.length) {
    status.textContent = 'Volume status not loaded yet.';
    status.style.color = '#fbbf24';
    return false;
  }
  const regionSummary = [...verdaVolumes.reduce((acc, v) => {
    const region = v.location || 'unknown';
    acc.set(region, (acc.get(region) || 0) + 1);
    return acc;
  }, new Map())]
    .sort(([a], [b]) => String(a).localeCompare(String(b)))
    .map(([region, count]) => `${region}:${count}`)
    .join(', ');
  const osVol = verdaVolumes.find(v => v.id === osId);
  const dataVol = verdaVolumes.find(v => v.id === dataId);
  if (!osVol || !dataVol) {
    status.textContent = `One selected volume was not found in Verda. Refresh volumes. Seen regions: ${regionSummary || 'none'}.`;
    status.style.color = '#fca5a5';
    return false;
  }
  const problems = [];
  if (!osVol.is_os_volume) problems.push('OS field is not an OS volume');
  if (dataVol.is_os_volume) problems.push('data field is an OS volume');
  for (const [label, vol] of [['OS', osVol], ['data', dataVol]]) {
    if (String(vol.status || '').toLowerCase() !== 'detached') {
      problems.push(`${label} volume is ${vol.status}`);
    }
    if (loc && vol.location !== loc) {
      problems.push(`${label} volume is in ${vol.location}, not ${loc}`);
    }
  }
  const fmt = v => `${v.name || v.id.slice(0, 8)} ${v.status} ${v.location}`;
  if (problems.length) {
    status.textContent = `${problems.join('; ')}. Existing-volume deploy requires detached volumes in the selected region. Seen regions: ${regionSummary || 'none'}. (${fmt(osVol)} / ${fmt(dataVol)})`;
    status.style.color = '#fca5a5';
    return false;
  }
  status.textContent = `Volumes ready: ${fmt(osVol)} / ${fmt(dataVol)}. Seen regions: ${regionSummary || 'none'}.`;
  status.style.color = '#6ee7b7';
  return true;
}

async function refreshVerdaCostEstimate() {
  const el = document.getElementById('v-cost-estimate');
  if (!el) return;
  const instanceType = (document.getElementById('v-instance-type').value || '').trim();
  const location = (document.getElementById('v-location').value || 'FIN-01').trim();
  if (!instanceType) {
    el.textContent = 'Select a Verda GPU to see cost.';
    return;
  }
  const fresh = document.getElementById('v-mode').value === 'fresh';
  const contract = document.getElementById('v-contract')?.value || 'pay_as_go';
  const osGb = fresh ? (+(document.getElementById('v-fresh-os-size').value) || 100) : 0;
  const storageGb = fresh ? (+(document.getElementById('v-fresh-storage-size').value) || 250) : 0;
  const params = new URLSearchParams({
    instance_type: instanceType,
    location,
    os_volume_gb: String(osGb),
    storage_gb: String(storageGb),
    storage_type: 'NVMe',
    contract,
  });
  el.textContent = 'Estimating cost…';
  try {
    const estimate = await fetch(`/api/verda/cost-estimate?${params.toString()}`).then(r => {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    });
    const total = estimate.total || estimate.instance || {};
    const instance = estimate.instance || {};
    const hourly = money(total.hourly);
    const daily = money(total.daily);
    const monthly = money(total.monthly);
    const compute = money(instance.hourly);
    const storageNote = fresh ? ` incl. ${osGb}GB OS + ${storageGb}GB model storage` : ' compute only; existing volume storage is already billed separately';
    const contractNote = contract === 'spot' ? 'spot, interruptible' : 'on-demand';
    el.textContent = `${hourly}/hr · ${daily}/day · ${monthly}/mo (${contractNote}; ${compute}/hr compute${storageNote})`;
  } catch (e) {
    el.textContent = `Cost estimate unavailable: ${e}`;
  }
}

function money(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '$?';
  return `$${n.toFixed(n >= 10 ? 2 : 3)}`;
}

async function stopPod(podId) {
  if (!confirm('Stop this pod? It will preserve your volume and models. You can resume it later.')) return;
  const r = await fetch(`/api/runpod/pods/${podId}/stop`, {method: 'POST'});
  const d = await r.json();
  if (!d.ok) alert(`Failed to stop: ${d.output || ''}`);
  setTimeout(loadRunpodPods, 2000);
}

async function terminatePod(podId) {
  if (!confirm(`Terminate RunPod pod ${podId}?`)) return;
  const r = await fetch(`/api/runpod/pods/${podId}`, {method: 'DELETE'});
  const d = await r.json();
  if (!d.ok) alert(`Failed: ${d.output || ''}`);
  loadRunpodPods();
}

async function quickDeployRunpod(podId) {
  const provId = '_runpod_provision';
  selectInstance(provId, '');
  selectLog('deploy');
  clearLog(podId ? 'Deploying to pod…' : 'Provisioning RunPod…', 'deploy', provId);
  setStatus('running', '⟳ RunPod: deploying…');
  disableActions(true);

  try {
    const res = await fetch('/api/provision-runpod', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        gpu_type: document.getElementById('rp-gpu').value,
        cloud_type: document.getElementById('rp-cloud').value,
        volume_gb: +(document.getElementById('rp-volume').value) || 150,
        container_disk_gb: +(document.getElementById('rp-cdisk').value) || 50,
        worker_port: +(document.getElementById('c-port').value) || 9000,
        remote_root: document.getElementById('remote-root').value || '/workspace/filmforge_gpu_worker',
        pod_id: podId || null,
        env_vars: getEnv(),
      }),
    });
    const {job_id} = await res.json();
    streamRunpodJob(job_id, provId);
  } catch (e) {
    appendLog(`[runpod] ERROR: ${e}`, 'deploy', provId);
    setStatus('failed', '✗ Failed');
    disableActions(false);
  }
}

function streamRunpodJob(jobId, instId) {
  const es = new EventSource(`/api/deploy/${jobId}/stream`);
  getInstData(instId).evtSrc = es;
  es.onmessage = e => {
    if (e.data === '__DONE__') {
      es.close();
      getInstData(instId).evtSrc = null;
      onRunpodJobDone(jobId, instId);
      return;
    }
    appendLog(JSON.parse(e.data), 'deploy', instId);
  };
  es.onerror = () => {
    es.close();
    getInstData(instId).evtSrc = null;
    onRunpodJobDone(jobId, instId);
  };
}

async function onRunpodJobDone(jobId, instId) {
  disableActions(false);
  const d = await fetch(`/api/deploy/${jobId}`).then(r => r.json());
  const idata = getInstData(instId);
  idata.status = d.status;
  idata.workerUrl = d.worker_url;
  if (d.status === 'done') {
    setStatus('done', '✓ Done' + (d.worker_url ? ' — ' + d.worker_url : ''));
    const pods = await fetch('/api/runpod/pods').then(r => r.json()).catch(() => []);
    let pod = null;
    const m = (d.worker_url || '').match(/https:\/\/([a-z0-9]+)-\d+\.proxy\.runpod\.net/);
    if (m) pod = pods.find(p => p.id === m[1]);
    if (!pod && pods.length === 1) pod = pods[0];
    await loadRunpodPods();
    if (pod) {
      const newId = `rp-${pod.id}`;
      instData[newId] = instData[instId];
      delete instData[instId];
      const ssh = (pod.ssh_ip && pod.ssh_port) ? `ssh root@${pod.ssh_ip} -p ${pod.ssh_port}` : '';
      selectInstance(newId, ssh);
      selectLog(ssh ? 'worker' : 'deploy');
    }
    loadWorkers();
  } else {
    setStatus('failed', '✗ Failed — check log');
    updateInstHeader();
  }
}

async function quickDeployVerda() {
  const hostname = (document.getElementById('v-hostname').value || 'filmforge-verda-worker').trim();
  const fresh = document.getElementById('v-mode').value === 'fresh';
  if (!fresh) {
    const volumesReady = await loadVerdaVolumes();
    if (!volumesReady) {
      alert('Existing-volume deploy needs a detached OS volume and detached model/data volume in the selected region. Use Fresh install, or stop/detach the current instance volumes first.');
      return;
    }
  }
  const provId = `verda:${hostname}`;
  selectInstance(provId, '');
  selectLog('deploy');
  clearLog(fresh ? 'Provisioning fresh Verda install…' : 'Provisioning Verda…', 'deploy', provId);
  setStatus('running', fresh ? '⟳ Verda: installing + downloading…' : '⟳ Verda: creating + starting workers…');
  disableActions(true);

  const env = getEnv();
  env.WORKER_PROVIDER = 'verda';
  env.WORKER_GPU_NAME = (document.getElementById('v-instance-type').value || '2A100.44V').trim();
  if (!env.WORKER_REGISTRATION_TOKEN && !env.RENDER_BROKER_WORKER_TOKEN) {
    delete env.FILMFORGE_BACKEND_URL;
    delete env.RENDER_BROKER_BASE_URL;
  }

  try {
    const res = await fetch('/api/provision-verda', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        fresh,
        location: (document.getElementById('v-location').value || 'FIN-01').trim(),
        instance_type: (document.getElementById('v-instance-type').value || '2A100.44V').trim(),
        contract: document.getElementById('v-contract')?.value || 'pay_as_go',
        os_volume_id: (document.getElementById('v-os-volume').value || '').trim(),
        data_volume_id: (document.getElementById('v-data-volume').value || '').trim(),
        ssh_key_id: (document.getElementById('v-ssh-key').value || '').trim(),
        hostname,
        worker_count: +(document.getElementById('v-worker-count').value) || 0,
        worker_port: +(document.getElementById('c-port').value) || 9000,
        comfy_port: +(document.getElementById('v-comfy-port').value) || 8188,
        remote_root: document.getElementById('remote-root').value || '/workspace/filmforge_gpu_worker',
        fresh_os_volume_size: +(document.getElementById('v-fresh-os-size').value) || 100,
        fresh_storage_size: +(document.getElementById('v-fresh-storage-size').value) || 250,
        skip_warmup: fresh ? !document.getElementById('v-fresh-warm').value : true,
        warm_asset_groups: fresh
          ? document.getElementById('v-fresh-warm').value.split(',').map(s => s.trim()).filter(Boolean)
          : [],
        env_vars: env,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.status);
    const {job_id} = await res.json();
    streamVerdaJob(job_id, provId);
  } catch (e) {
    appendLog(`[verda] ERROR: ${e}`, 'deploy', provId);
    setStatus('failed', '✗ Failed');
    disableActions(false);
  }
}

function updateVerdaMode() {
  const fresh = document.getElementById('v-mode').value === 'fresh';
  document.getElementById('v-existing-fields').style.display = fresh ? 'none' : 'block';
  document.getElementById('v-fresh-fields').style.display = fresh ? 'block' : 'none';
  refreshVerdaCostEstimate();
  if (!fresh) loadVerdaVolumes();
}

function streamVerdaJob(jobId, instId) {
  const es = new EventSource(`/api/deploy/${jobId}/stream`);
  getInstData(instId).evtSrc = es;
  es.onmessage = e => {
    if (e.data === '__DONE__') {
      es.close();
      getInstData(instId).evtSrc = null;
      onVerdaJobDone(jobId, instId);
      return;
    }
    appendLog(JSON.parse(e.data), 'deploy', instId);
  };
  es.onerror = () => {
    es.close();
    getInstData(instId).evtSrc = null;
    onVerdaJobDone(jobId, instId);
  };
}

async function onVerdaJobDone(jobId, instId) {
  disableActions(false);
  const d = await fetch(`/api/deploy/${jobId}`).then(r => r.json());
  const idata = getInstData(instId);
  const urls = d.worker_urls && d.worker_urls.length ? d.worker_urls : (d.worker_url ? [d.worker_url] : []);
  idata.status = d.status;
  idata.workerUrl = d.worker_url;
  idata.workerUrls = urls;
  if (d.ssh_command) idata.ssh = d.ssh_command;

  if (d.status === 'done') {
    setStatus('done', '✓ Done' + (urls.length ? ' — ' + urls.join(', ') : ''));
    if (d.ssh_command) {
      const hostname = instId.replace(/^verda:/, '') || 'verda-worker';
      await fetch('/api/verda/hosts', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name: hostname, ssh_command: d.ssh_command}),
      }).catch(() => {});
      await loadVerdaHosts();
      selectInstance(instId, d.ssh_command);
      selectLog('worker');
    }
    loadWorkers();
  } else {
    setStatus('failed', '✗ Failed — check log');
    updateInstHeader();
  }
}

// ── Deploy to a running instance ──────────────────────────────────────────────
async function deployToInst(ssh, instId, options = {}) {
  if (!ssh) { alert('No SSH command available for this instance.'); return; }

  const provider = options.provider || 'vast';
  selectInstance(instId, ssh);
  selectLog('deploy');
  clearLog(provider === 'verda' ? 'Redeploying Verda worker…' : 'Deploying…', 'deploy', String(instId));
  setStatus('running', provider === 'verda' ? '⟳ Verda: redeploying…' : '⟳ Deploying…');
  disableActions(true);

  const env = getEnv();
  Object.assign(env, options.env || {});
  env.WORKER_PROVIDER = provider;
  env.WORKER_GPU_NAME = options.gpuName || document.getElementById('c-gpu').value;
  delete env.RENDER_BROKER_WORKER_ID;
  if (!env.FILMFORGE_BACKEND_URL || env.FILMFORGE_BACKEND_URL.includes('localhost')) {
    env.FILMFORGE_BACKEND_URL = 'https://filmforgepythonbackend.fly.dev';
  }
  if (!env.RENDER_BROKER_BASE_URL || env.RENDER_BROKER_BASE_URL.includes('localhost')) {
    env.RENDER_BROKER_BASE_URL = env.FILMFORGE_BACKEND_URL;
  }

  try {
    const res = await fetch('/api/deploy', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        ssh_command: ssh,
        env_vars: env,
        worker_port: options.workerPort || +document.getElementById('c-port').value || 9000,
        worker_count: options.workerCount ?? (+document.getElementById('c-worker-count').value || 0),
        comfy_port: options.comfyPort || +document.getElementById('c-comfy-port').value || 18188,
        remote_root: options.remoteRoot || document.getElementById('remote-root').value || '/workspace/filmforge_gpu_worker',
      }),
    });
    const {job_id} = await res.json();
    streamJob(job_id, false, String(instId));
  } catch (e) {
    appendLog(`[deploy] ERROR: ${e}`, 'deploy', String(instId));
    setStatus('failed', '✗ Failed');
    disableActions(false);
  }
}

// ── Create & Deploy (auto-provision) ─────────────────────────────────────────
async function quickDeploy() {
  const provId = '_provision';
  selectInstance(provId, '');
  selectLog('deploy');
  clearLog('Provisioning…', 'deploy', provId);
  setStatus('running', '⟳ Provisioning + deploying…');
  disableActions(true);

  const env = getEnv();
  env.WORKER_PROVIDER = 'vast';
  env.WORKER_GPU_NAME = document.getElementById('c-gpu').value;
  delete env.RENDER_BROKER_WORKER_ID;
  if (!env.FILMFORGE_BACKEND_URL || env.FILMFORGE_BACKEND_URL.includes('localhost')) {
    env.FILMFORGE_BACKEND_URL = 'https://filmforgepythonbackend.fly.dev';
  }
  if (!env.RENDER_BROKER_BASE_URL || env.RENDER_BROKER_BASE_URL.includes('localhost')) {
    env.RENDER_BROKER_BASE_URL = env.FILMFORGE_BACKEND_URL;
  }

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
        worker_count: +(document.getElementById('c-worker-count').value) || 0,
        comfy_port: +(document.getElementById('c-comfy-port').value) || 18188,
        remote_root: document.getElementById('remote-root').value || '/workspace/filmforge_gpu_worker',
        warm_asset_groups: [],
        env_vars: env,
      }),
    });
    const {job_id} = await res.json();
    streamJob(job_id, true, provId);
  } catch (e) {
    appendLog(`[vast] ERROR: ${e}`, 'deploy', provId);
    setStatus('failed', '✗ Failed');
    disableActions(false);
  }
}

// ── Job streaming ─────────────────────────────────────────────────────────────
function streamJob(jobId, isProvision, instId) {
  const es = new EventSource(`/api/deploy/${jobId}/stream`);
  getInstData(instId).evtSrc = es;
  es.onmessage = e => {
    if (e.data === '__DONE__') {
      es.close();
      getInstData(instId).evtSrc = null;
      onJobDone(jobId, isProvision, instId);
      return;
    }
    appendLog(JSON.parse(e.data), 'deploy', instId);
  };
  es.onerror = () => {
    es.close();
    getInstData(instId).evtSrc = null;
    onJobDone(jobId, isProvision, instId);
  };
}

async function onJobDone(jobId, isProvision, instId) {
  disableActions(false);
  const d = await fetch(`/api/deploy/${jobId}`).then(r => r.json());
  const idata = getInstData(instId);
  const urls = d.worker_urls && d.worker_urls.length ? d.worker_urls : (d.worker_url ? [d.worker_url] : []);
  idata.status = d.status;
  idata.workerUrl = d.worker_url || urls[0] || null;
  idata.workerUrls = urls;
  if (d.ssh_command) idata.ssh = d.ssh_command;

  if (d.status === 'done') {
    setStatus('done', '✓ Done' + (urls.length ? ' — ' + urls.join(', ') : ''));
    if (isProvision) {
      const instances = await loadInstances();
      if (instances && instances.length) {
        const newest = instances.reduce((a, b) => (+b.id > +a.id ? b : a));
        const newId = String(newest.id);
        const newSsh = getSsh(newest);
        instData[newId] = instData[instId];
        delete instData[instId];
        selectInstance(newId, newSsh || idata.ssh);
        selectLog('worker');
      }
    } else {
      loadInstances();
    }
    loadWorkers();
  } else {
    setStatus('failed', '✗ Failed — check log');
    updateInstHeader();
  }
}

function disableActions(on) {
  document.getElementById('provision-btn').disabled = on;
  const rpBtn = document.getElementById('runpod-provision-btn');
  if (rpBtn) rpBtn.disabled = on;
  const verdaBtn = document.getElementById('verda-provision-btn');
  if (verdaBtn) verdaBtn.disabled = on;
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
    const eta = formatWorkerEta(w.eta_by_asset_group || {});
    // ComfyUI queue depth is ground truth for "is the GPU generating right now".
    // It rides the worker heartbeat metadata; null means the worker isn't
    // reporting it (older build) so we fall back to plain Online/Offline.
    const running = (w.metadata && typeof w.metadata.comfy_queue_running === 'number')
      ? w.metadata.comfy_queue_running : null;
    const pending = (w.metadata && typeof w.metadata.comfy_queue_pending === 'number')
      ? w.metadata.comfy_queue_pending : 0;
    let badgeClass, badgeText;
    if (!alive) { badgeClass = 'b-exited'; badgeText = 'Offline'; }
    else if (running === null) { badgeClass = 'b-running'; badgeText = 'Online'; }
    else if (running > 0) { badgeClass = 'b-generating'; badgeText = pending > 0 ? `Generating +${pending}` : 'Generating'; }
    else { badgeClass = 'b-idle'; badgeText = 'Idle'; }
    return `<div class="wrow">
      <div class="flex1" style="min-width:0">
        <div class="wname">${esc(w.worker_name || w.id || '?')}</div>
        <div class="wurl" title="${esc(w.base_url || '')}">${esc(w.base_url || '—')}</div>
        <div class="wurl" title="${esc(caps)}">${esc(caps || 'no capabilities')}</div>
        ${eta ? `<div class="wurl" title="${esc(eta)}">${esc(eta)}</div>` : ''}
      </div>
      <span class="badge ${badgeClass}" style="flex-shrink:0">
        <span class="dot"></span>${badgeText}
      </span>
      <span class="muted text-xs" style="flex-shrink:0;min-width:44px;text-align:right">${age}</span>
    </div>`;
  }).join('');
}

function formatWorkerEta(etaByAssetGroup) {
  const labels = {
    flux_stills_v1: 'flux',
    juggernaut_stills_v1: 'jugg',
    wan_i2v_v1: 'wan',
    stable_audio_v1: 'audio',
  };
  return Object.entries(etaByAssetGroup)
    .map(([group, eta]) => {
      const seconds = eta && eta.estimated_total_sec;
      if (!seconds) return '';
      return `${labels[group] || group}: ${formatDuration(seconds)}`;
    })
    .filter(Boolean)
    .join(' · ');
}

function formatDuration(seconds) {
  const s = Math.round(Number(seconds) || 0);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return rem ? `${m}m ${rem}s` : `${m}m`;
}

function timeAgo(iso) {
  if (!iso) return '?';
  const s = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

// ── Log ───────────────────────────────────────────────────────────────────────
function selectLog(src) {
  currentLogSrc = src;
  document.querySelectorAll('.ltab').forEach(b => b.classList.toggle('active', b.dataset.src === src));
  if (remoteEvt) { remoteEvt.close(); remoteEvt = null; }
  renderLog(src);
  if (src === 'deploy') return;
  reconnectRemoteLog(src);
}

function reconnectRemoteLog(src) {
  if (remoteEvt) { remoteEvt.close(); remoteEvt = null; }
  if (!selectedInstId) return;
  const ssh = getInstData(selectedInstId).ssh;
  if (!ssh) {
    appendLog(`[remote-log] No SSH command for instance #${selectedInstId}. Deploy first.`, src, selectedInstId);
    return;
  }
  clearLog(`Connecting to ${LOG_LABELS[src] || src}…`, src, selectedInstId);
  const capturedInstId = selectedInstId;
  remoteEvt = new EventSource(`/api/remote-logs/${src}/stream?ssh=${encodeURIComponent(ssh)}`);
  remoteEvt.onmessage = e => {
    if (e.data === '__DONE__') { remoteEvt.close(); remoteEvt = null; return; }
    appendLog(JSON.parse(e.data), src, capturedInstId);
  };
  remoteEvt.onerror = () => {
    appendLog('[remote-log] disconnected', src, capturedInstId);
    remoteEvt.close(); remoteEvt = null;
  };
}

function updateDownloadStatusFromLine(line, idata) {
  let match = line.match(/Downloading asset progress:\s+([^\s]+)\s+(.+?)\s+\(([^)]+)\)\s+([0-9.]+\s+[KMGT]?B\/s)/);
  if (match) {
    idata.downloadStatus = {
      asset: match[1],
      detail: `${match[1]} ${match[3]} · ${match[4]} · ${match[2]}`,
      updatedAt: Date.now(),
    };
    return true;
  }
  if (/Download complete:/.test(line)) {
    // Download just finished — clear any lingering status so the UI hides it.
    if (idata.downloadStatus) {
      idata.downloadStatus = null;
      return true;
    }
    return false;
  }
  match = line.match(/Downloading asset:\s+([^\s]+)\s+total=([^ ]+\s+[KMGT]?B)/);
  if (match) {
    idata.downloadStatus = {
      asset: match[1],
      detail: `${match[1]} starting · ${match[2]}`,
      updatedAt: Date.now(),
    };
    return true;
  }
  return false;
}

const DOWNLOAD_STATUS_TTL_MS = 30000;

function activeDownloadStatus(idata) {
  const ds = idata.downloadStatus;
  if (!ds || !ds.updatedAt) return null;
  if (Date.now() - ds.updatedAt > DOWNLOAD_STATUS_TTL_MS) return null;
  return ds;
}

function formatWorkerStats(stats) {
  if (!Array.isArray(stats) || !stats.length) return '';
  const parts = stats.map(g => {
    const ag = g.asset_group || '?';
    const sec = (g.avg_total_sec ?? g.avg_comfy_run_sec ?? 0);
    return `${ag} ${Number(sec).toFixed(1)}s`;
  });
  const totalSamples = stats.reduce((acc, g) => acc + (g.sample_count || 0), 0);
  return `Avg gen: ${parts.join(' · ')} (n=${totalSamples})`;
}

function appendLog(line, src = currentLogSrc, instId = selectedInstId) {
  if (!line || !instId) return;
  const idata = getInstData(instId);
  if (!idata.logBuffers[src]) idata.logBuffers[src] = [];
  const isNoise = /"\s*(get|head)\s+\//i.test(line) || /http\/1\.1"\s+200/i.test(line);
  if (isNoise) return;
  const downloadChanged = updateDownloadStatusFromLine(line, idata);
  idata.logBuffers[src].push(line);
  if (idata.logBuffers[src].length > MAX_LOG_LINES) {
    idata.logBuffers[src].splice(0, idata.logBuffers[src].length - MAX_LOG_LINES);
  }
  if (downloadChanged && instId === selectedInstId) updateInstHeader();
  if (instId !== selectedInstId || src !== currentLogSrc) return;
  renderLog(src);
}

function renderLog(src = currentLogSrc) {
  const el = document.getElementById('log');
  if (!selectedInstId) {
    el.innerHTML = `<span class="muted text-xs">← Select an instance to view logs.</span>`;
    return;
  }
  const lines = getInstData(selectedInstId).logBuffers[src] || [];
  if (!lines.length) {
    el.innerHTML = `<span class="muted text-xs">No ${LOG_LABELS[src] || src} log yet for instance #${esc(selectedInstId)}.</span>`;
    return;
  }
  el.innerHTML = '';
  for (const line of lines) {
    const lo = line.toLowerCase();
    const d = document.createElement('div');
    const sev = lo.includes('error') || lo.includes('failed') ? 'err'
      : lo.includes('✓') || lo.includes(' done') ? 'ok'
      : lo.startsWith('[deploy]') || lo.startsWith('[vast]') || lo.startsWith('[runpod]') || lo.startsWith('[verda]') ? 'info'
      : lo.includes('download') || lo.includes('asset') || lo.includes('comfy') ? 'signal'
      : lo.includes('warn') ? 'warn' : '';
    const gpuMatch = line.match(/^\[(gpu\d+)\]/);
    const gpu = gpuMatch ? gpuMatch[1] : '';
    d.className = 'll' + (sev ? ' ' + sev : '') + (gpu ? ' ' + gpu : '');
    d.textContent = line;
    el.appendChild(d);
  }
  el.scrollTop = el.scrollHeight;
}

function clearLog(msg = 'Ready.', src = currentLogSrc, instId = selectedInstId) {
  if (instId) {
    const idata = getInstData(instId);
    idata.logBuffers[src] = [];
  }
  if (instId === selectedInstId && src === currentLogSrc) {
    document.getElementById('log').innerHTML = `<span class="muted text-xs">${esc(msg)}</span>`;
  }
}

function setStatus(type, msg) {
  const el = document.getElementById('sbar');
  el.className = type;
  el.textContent = msg;
}

// ── Env vars ──────────────────────────────────────────────────────────────────
const CONFIG_FIELDS = [
  {key: 'FILMFORGE_BACKEND_URL', label: 'Backend URL (Local=dev, Fly=remote)', type: 'select', selectId: 'cfg-backend', inputId: 'cfg-backend-custom', defaultValue: 'fly', options: [
    {value: 'local', label: 'Local (http://localhost:8000) — local testing only'},
    {value: 'fly', label: 'Fly.io (https://filmforgepythonbackend.fly.dev) — production'},
    {value: '__custom__', label: 'Custom'},
  ]},
  {key: 'RENDER_BROKER_BASE_URL', label: 'Broker URL (must match Backend)', type: 'select', selectId: 'cfg-broker', inputId: 'cfg-broker-custom', defaultValue: 'fly', options: [
    {value: 'local', label: 'Local (http://localhost:8000) — local testing only'},
    {value: 'fly', label: 'Fly.io (https://filmforgepythonbackend.fly.dev) — production'},
    {value: '__custom__', label: 'Custom'},
  ]},
  {key: 'WORKER_PROVIDER', label: 'Provider', type: 'select', selectId: 'cfg-provider', defaultValue: 'dedicated_worker', options: [
    {value: 'dedicated_worker', label: 'Dedicated Worker'},
    {value: 'vast', label: 'Vast'},
    {value: 'verda', label: 'Verda'},
  ]},
  {key: 'WORKER_MAX_CONCURRENT_JOBS', label: 'Max Concurrent Jobs', type: 'select', selectId: 'cfg-max-jobs', defaultValue: '1', options: [
    {value: '1', label: '1'},
    {value: '2', label: '2'},
    {value: '4', label: '4'},
  ]},
  {key: 'WORKER_HEARTBEAT_SECONDS', label: 'Heartbeat (sec)', type: 'select', selectId: 'cfg-heartbeat', defaultValue: '60', options: [
    {value: '30', label: '30s'},
    {value: '60', label: '60s'},
    {value: '120', label: '120s'},
    {value: '300', label: '300s'},
  ]},
  {key: 'WORKER_CAPABILITIES', label: 'Capabilities', type: 'select', selectId: 'cfg-capabilities', defaultValue: 'flux2_stills,juggernaut_stills,wan_i2v,stable_audio', options: [
    {value: 'flux2_stills,juggernaut_stills,wan_i2v,stable_audio', label: 'Flux + Juggernaut + WAN + Audio (all)'},
    {value: 'flux2_stills,wan_i2v,stable_audio', label: 'Flux + WAN + Audio'},
    {value: 'juggernaut_stills,wan_i2v,stable_audio', label: 'Juggernaut + WAN + Audio'},
    {value: 'flux2_stills,wan_i2v', label: 'Flux + WAN'},
    {value: 'juggernaut_stills,wan_i2v', label: 'Juggernaut + WAN'},
    {value: 'wan_i2v,stable_audio', label: 'WAN + Audio'},
    {value: 'flux2_stills', label: 'Flux (stills only)'},
    {value: 'juggernaut_stills', label: 'Juggernaut (stills only)'},
    {value: 'wan_i2v', label: 'WAN (video)'},
  ]},
  {key: 'WORKER_NAME', label: 'Worker Name', type: 'text', inputId: 'cfg-worker-name', placeholder: 'auto (hostname)'},
  {key: 'WORKER_GPU_NAME', label: 'GPU Type', type: 'select', selectId: 'cfg-gpu-name', inputId: 'cfg-gpu-custom', defaultValue: 'RTX 4090', options: [
    {value: 'RTX 4090', label: 'RTX 4090'},
    {value: 'L40S', label: 'L40S'},
    {value: 'A100 SXM', label: 'A100 SXM'},
    {value: 'A100 PCIe', label: 'A100 PCIe'},
    {value: '2A100.44V', label: 'Verda 2A100.44V'},
    {value: 'H100 SXM', label: 'H100 SXM'},
    {value: 'H100 PCIe', label: 'H100 PCIe'},
    {value: 'A6000', label: 'A6000'},
    {value: 'RTX 6000 Ada', label: 'RTX 6000 Ada'},
    {value: '__custom__', label: 'Custom'},
  ]},
  {key: 'WORKER_REGISTRATION_TOKEN', label: 'Registration Token', type: 'password', inputId: 'cfg-reg-token', placeholder: 'paste token or Load .env'},
  {key: 'RENDER_BROKER_WORKER_TOKEN', label: 'Worker Token', type: 'password', inputId: 'cfg-broker-token', placeholder: 'paste token or Load .env'},
];

function renderConfigPanel() {
  const container = document.getElementById('env-table');
  container.innerHTML = '';

  for (const field of CONFIG_FIELDS) {
    const row = document.createElement('div');
    row.className = 'cfg-grid';

    const label = document.createElement('div');
    label.className = 'cfg-label';
    label.textContent = field.label;

    let input;
    if (field.type === 'select') {
      input = document.createElement('select');
      input.id = field.selectId;
      field.options.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt.value;
        option.textContent = opt.label;
        input.appendChild(option);
      });
      input.value = field.defaultValue || '';
      if (field.options.some(o => o.value === '__custom__')) {
        input.addEventListener('change', () => {
          const custom = document.getElementById(field.inputId);
          if (custom) custom.style.display = input.value === '__custom__' ? 'block' : 'none';
        });
      }
    } else {
      input = document.createElement('input');
      input.id = field.inputId;
      input.type = field.type || 'text';
      input.placeholder = field.placeholder || '';
      input.value = field.defaultValue || '';
    }

    row.appendChild(label);
    row.appendChild(input);
    container.appendChild(row);

    // Custom text input for select fields (hidden by default)
    if (field.type === 'select' && field.inputId) {
      const customRow = document.createElement('div');
      customRow.id = field.inputId;
      customRow.className = 'cfg-custom';
      customRow.style.display = 'none';
      customRow.style.gridColumn = '2';
      customRow.style.marginBottom = '6px';
      const customInput = document.createElement('input');
      customInput.type = 'text';
      customInput.placeholder = 'enter custom value';
      customInput.className = 'cfg-custom-input';
      customRow.appendChild(customInput);
      container.appendChild(customRow);
    }
  }
}

function initEnv() {
  renderConfigPanel();
}

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

  // Collect from config fields
  for (const field of CONFIG_FIELDS) {
    let value = '';
    if (field.type === 'select') {
      const sel = document.getElementById(field.selectId);
      if (sel) {
        const selected = sel.value;
        if (selected === '__custom__') {
          // Get custom value from hidden input's nested input
          const customDiv = document.getElementById(field.inputId);
          if (customDiv) {
            const customInput = customDiv.querySelector('.cfg-custom-input');
            value = (customInput ? customInput.value : '').trim();
          }
        } else if (selected === 'local') {
          value = field.key === 'FILMFORGE_BACKEND_URL' || field.key === 'RENDER_BROKER_BASE_URL'
            ? 'http://localhost:8000' : '';
        } else if (selected === 'fly') {
          value = 'https://filmforgepythonbackend.fly.dev';
        } else {
          value = selected;
        }
      }
    } else {
      const inp = document.getElementById(field.inputId);
      if (inp) value = inp.value.trim();
    }
    if (field.key && value) o[field.key] = value;
  }

  // Collect from custom rows
  document.querySelectorAll('.env-row').forEach(r => {
    const k = r.querySelector('.ek').value.trim();
    const v = r.querySelector('.ev').value.trim();
    if (k && v) o[k] = v;
  });

  return o;
}

async function loadEnvFromBackend() {
  const data = await fetch('/api/backend-env').then(r => r.json()).catch(() => ({}));

  // Populate config fields
  for (const field of CONFIG_FIELDS) {
    if (data[field.key] !== undefined) {
      const val = data[field.key];
      if (field.type === 'select') {
        const sel = document.getElementById(field.selectId);
        if (sel) {
          // Check if value matches a known option
          const opt = field.options.find(o => o.value === val || o.value.replace(/https?:\/\//, '') === val);
          if (opt) {
            sel.value = opt.value;
            // Hide custom input if not selected
            if (field.inputId) {
              const customDiv = document.getElementById(field.inputId);
              if (customDiv) customDiv.style.display = 'none';
            }
          } else {
            // Unknown value — set to custom and populate custom input
            sel.value = '__custom__';
            if (field.inputId) {
              const customDiv = document.getElementById(field.inputId);
              if (customDiv) {
                const customInput = customDiv.querySelector('.cfg-custom-input');
                if (customInput) customInput.value = val;
                customDiv.style.display = 'block';
              }
            }
          }
        }
      } else {
        const inp = document.getElementById(field.inputId);
        if (inp) inp.value = val;
      }
    }
  }

  // Populate custom rows (for keys not in CONFIG_FIELDS)
  const configKeys = new Set(CONFIG_FIELDS.map(f => f.key));
  Object.entries(data).forEach(([k, v]) => {
    if (!configKeys.has(k)) {
      const exists = [...document.querySelectorAll('.ek')].some(i => i.value.trim() === k);
      if (!exists && v) addEnvRow(k, v);
    }
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

function toggleRpAdv() {
  const body = document.getElementById('rp-adv-body');
  const btn = event.target;
  const open = body.style.display === 'block';
  body.style.display = open ? 'none' : 'block';
  btn.textContent = (open ? '▶' : '▼') + ' Advanced options';
}

function toggleVerdaAdv() {
  const body = document.getElementById('v-adv-body');
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

function sshHost(ssh) {
  const parts = String(ssh || '').trim().split(/\s+/).filter(Boolean);
  for (let i = parts.length - 1; i >= 0; i--) {
    const part = parts[i];
    if (part.startsWith('-')) continue;
    if (part.includes('@')) return part.split('@').pop().replace(/^\[/, '').replace(/\]$/, '');
  }
  return '';
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
    if (btn) {
      const old = btn.textContent;
      btn.textContent = 'Copied';
      setTimeout(() => { btn.textContent = old; }, 900);
    }
  } catch {
    window.prompt('Copy worker URL', text);
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
initEnv();
updateVerdaMode();
loadVerdaAvailability();
loadVerdaVolumes();
loadInstances();
loadRunpodPods();
loadVerdaHosts();
loadWorkers();
updateInstHeader();
setInterval(loadInstances, 15000);
setInterval(loadRunpodPods, 15000);
setInterval(loadVerdaHosts, 30000);
setInterval(loadWorkers, 15000);
setInterval(() => refreshSelectedInstanceSummary(true), 30000);
</script>
</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    my_pid = os.getpid()
    try:
        out = subprocess.run(
            ["lsof", "-tiTCP:" + str(port), "-sTCP:LISTEN"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        pids = [int(p) for p in out.split() if p.isdigit() and int(p) != my_pid]
        for pid in pids:
            print(f"Killing existing deploy_ui on port {port} (pid {pid})…")
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                pass
        if pids:
            import time as _t
            _t.sleep(1)
    except FileNotFoundError:
        pass
    print(f"FilmForge GPU Deploy UI → http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
