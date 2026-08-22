#!/usr/bin/env python3
"""Deploy the Filmforge GPU worker to SSH, RunPod, or a freshly rented Vast box."""

from __future__ import annotations

import argparse
import ipaddress
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
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener, urlopen

try:
    from gpu_worker.worker_release import (
        WorkerReleaseBundle,
        WorkerReleaseError,
        build_worker_release_bundle,
        worker_release_activate_script,
        worker_release_install_script,
        worker_release_rollback_script,
    )
except ModuleNotFoundError:  # direct ``python deploy_gpu.py`` execution
    from worker_release import (  # type: ignore[no-redef]
        WorkerReleaseBundle,
        WorkerReleaseError,
        build_worker_release_bundle,
        worker_release_activate_script,
        worker_release_install_script,
        worker_release_rollback_script,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REMOTE_ROOT = "/workspace/filmforge_gpu_worker"
DEFAULT_BACKEND_ENV = SCRIPT_DIR.parent / "backend" / "app" / ".env"
DEFAULT_BACKEND_ROOT = SCRIPT_DIR.parent / "backend"
DEFAULT_SSH_IDENTITY = Path.home() / ".ssh" / "vast_deploy"
DEFAULT_SSH_IDENTITY_RUNPOD = Path.home() / ".ssh" / "runpod_deploy"
DEFAULT_VAST_IMAGE = "vastai/comfy:v0.15.1-cuda-12.9-py312"
DEFAULT_VAST_GPU = "RTX 4090"
DEFAULT_VAST_MAX_PRICE = 0.75
DEFAULT_VAST_MIN_VRAM_GB = 24
DEFAULT_VAST_LIMIT = 25
DEFAULT_VAST_BOOT_TIMEOUT = 900
_MAX_WORKER_WARMUP_RESPONSE_BYTES = 1024 * 1024
# Reliable-region allow-list for Vast offers. Chinese (CN) and other unlisted
# hosts are excluded because some hang forever on "loading" and never accept SSH
# (observed 2026-07-10: a CN box stuck loading past the boot timeout). Regions:
# North America, Europe, Australia/NZ, India. Override with --vast-country.
DEFAULT_VAST_COUNTRIES = [
    "US", "CA",  # North America
    "GB", "IE", "FR", "DE", "NL", "BE", "LU", "CH", "AT", "SE", "NO", "FI", "DK",
    "IS", "EE", "LT", "LV", "PL", "CZ", "SK", "HU", "RO", "BG", "SI", "HR", "IT",
    "ES", "PT", "GR", "UA",  # Europe
    "AU", "NZ", "IN",  # Australia + India
]
DEFAULT_VERDA_CLI = Path.home() / ".verda" / "bin" / "verda"
DEFAULT_VERDA_LOCATION = "FIN-01"
DEFAULT_VERDA_INSTANCE_TYPE = "2A100.44V"
DEFAULT_VERDA_OS_IMAGE = "ubuntu-24.04-cuda-12.8-open-docker"
DEFAULT_VERDA_OS_VOLUME_ID = "34ec939d-a8c1-4ee2-9637-533e324dfe39"
DEFAULT_VERDA_DATA_VOLUME_ID = "4ea18b04-564f-4218-ab79-e90d1ccc839b"
DEFAULT_VERDA_SSH_KEY_ID = "11ee08a4-858a-4ee7-98c8-250aad99eb37"
DEFAULT_VERDA_HOSTNAME = "filmforge-verda-worker"
DEFAULT_VERDA_FRESH_OS_VOLUME_SIZE = 100
DEFAULT_WORKER_RELEASES_ROOT = "/opt/filmforge-worker-releases"
DEFAULT_WORKER_RUNTIME_ROOT = "/opt/filmforge_gpu_worker"


def _validate_worker_public_url_env(env_vars: list[str]) -> None:
    """Fail deployment before advertising a credential-bearing cleartext URL."""

    rendered = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in env_vars
        if "=" in item
    }
    mode = str(
        rendered.get("WORKER_API_AUTH_MODE")
        or os.getenv("WORKER_API_AUTH_MODE")
        or "required"
    ).strip().lower()
    urls: list[str] = []
    if rendered.get("WORKER_PUBLIC_URL"):
        urls.append(rendered["WORKER_PUBLIC_URL"])
    if rendered.get("WORKER_PUBLIC_URLS"):
        urls.extend(rendered["WORKER_PUBLIC_URLS"].split(","))
    for raw_url in urls:
        raw_url = raw_url.strip()
        if not raw_url:
            continue
        try:
            parsed = urlsplit(raw_url)
            port = parsed.port
        except (TypeError, ValueError):
            raise RuntimeError("Worker public URL is invalid") from None
        hostname = str(parsed.hostname or "").lower().rstrip(".")
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            # Bare host for worker 0; /gpu{idx} path prefixes for indexed
            # workers behind the shared secure edge (ADR-0009). Nothing else.
            or not (parsed.path in {"", "/"} or re.fullmatch(r"/gpu\d+", parsed.path))
            or port is not None and not 1 <= port <= 65535
        ):
            raise RuntimeError("Worker public URL is invalid")
        loopback = hostname == "localhost"
        try:
            loopback = loopback or ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            pass
        if (
            parsed.scheme.lower() != "https"
            and not loopback
            and mode not in {"development", "test"}
        ):
            raise RuntimeError("Worker public URL requires HTTPS")


def _worker_env_map(env_vars: list[str]) -> dict[str, str]:
    """Parse deployment KEY=VALUE inputs without ever logging their values."""

    rendered: dict[str, str] = {}
    for item in env_vars:
        key, separator, value = str(item).partition("=")
        key = key.strip()
        if not separator or not key:
            continue
        if key in rendered and rendered[key] != value:
            raise RuntimeError(f"Conflicting deployment value for {key}")
        rendered[key] = value
    return rendered


_CREDENTIAL_ENV_KEY = re.compile(
    r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|API_KEY|PRIVATE_KEY)(?:_|$)"
)


def _load_worker_deploy_env_file(path: Path) -> list[str]:
    """Load deployment values from a protected file, never process argv."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError("--env-file must be a regular file")
    if path.stat().st_mode & 0o777 != 0o600:
        raise RuntimeError("--env-file must have mode 0600")
    values: list[str] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if (
            not separator
            or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key)
            or not value
            or any(character in value for character in "\r\n\x00")
        ):
            raise RuntimeError(f"invalid --env-file assignment at line {line_number}")
        values.append(f"{key}={value}")
    return values


def _preflight_complete_worker_contract(
    env_vars: list[str],
    *,
    worker_port: int,
    expected_worker_count: int | None = None,
) -> dict[str, list[str] | str]:
    """Refuse a worker deploy before GPU/provider mutation if any half is absent.

    This is a configuration preflight, not a readiness claim.  The subsequent
    secure-profile cutover still requires the independent, fresh receipt defined
    in ``worker_release.py``.
    """

    rendered = _worker_env_map(env_vars)
    missing: list[str] = []
    if not (rendered.get("GPU_WORKER_API_TOKEN") or rendered.get("WORKER_API_TOKEN")):
        missing.append("GPU_WORKER_API_TOKEN")
    for key in (
        "WORKER_REGISTRATION_TOKEN",
        "FILMFORGE_BACKEND_URL",
        "WORKER_API_AUTH_MODE",
        "WORKER_PUBLIC_URLS",
        "WORKER_TUNNEL_LOCAL_URLS",
        "WORKER_TUNNEL_UNITS",
        "WORKER_SECURITY_STAGE_RECEIPTS",
        "WORKER_DEPLOY_PHASE",
        "FILMFORGE_BACKEND_CLIENT_AUTH_MODE",
    ):
        if not rendered.get(key):
            missing.append(key)
    if missing:
        raise RuntimeError(
            "Worker security contract is incomplete before GPU setup: "
            + ", ".join(sorted(missing))
        )
    if rendered.get("WORKER_API_AUTH_MODE", "required").strip().lower() != "required":
        raise RuntimeError("Worker security contract requires WORKER_API_AUTH_MODE=required")
    if rendered["FILMFORGE_BACKEND_CLIENT_AUTH_MODE"].strip().lower() != "bearer":
        raise RuntimeError(
            "Worker security contract requires the backend bearer-sending client"
        )
    if rendered["WORKER_DEPLOY_PHASE"] not in {
        "stage-code",
        "provision-only",
        "activate",
    }:
        raise RuntimeError(
            "WORKER_DEPLOY_PHASE must be stage-code, provision-only, or activate"
        )
    if rendered.get("WORKER_EDGE_PROVIDER", "cloudflared") not in {"cloudflared", "caddy"}:
        raise RuntimeError("WORKER_EDGE_PROVIDER must be cloudflared or caddy")
    try:
        backend_url = urlsplit(rendered["FILMFORGE_BACKEND_URL"])
        backend_port = backend_url.port
    except (TypeError, ValueError):
        raise RuntimeError("FILMFORGE_BACKEND_URL is invalid") from None
    if (
        backend_url.scheme.lower() != "https"
        or not backend_url.hostname
        or backend_url.username is not None
        or backend_url.password is not None
        or backend_url.query
        or backend_url.fragment
        or backend_url.path not in {"", "/"}
        or backend_port is not None and not 1 <= backend_port <= 65535
    ):
        raise RuntimeError(
            "FILMFORGE_BACKEND_URL must be an origin-only HTTPS URL"
        )

    public_urls = [
        value.strip()
        for value in rendered["WORKER_PUBLIC_URLS"].split(",")
        if value.strip()
    ]
    local_urls = [
        value.strip()
        for value in rendered["WORKER_TUNNEL_LOCAL_URLS"].split(",")
        if value.strip()
    ]
    tunnel_units = [
        value.strip()
        for value in rendered["WORKER_TUNNEL_UNITS"].split(",")
        if value.strip()
    ]
    stage_receipts = [
        value.strip()
        for value in rendered["WORKER_SECURITY_STAGE_RECEIPTS"].split(",")
        if value.strip()
    ]
    # ADR-0009 (one edge, N workers): public and local URLs pair index-for-
    # index per worker, while the tunnel unit and stage receipt describe the
    # single shared edge — exactly one of each regardless of worker count.
    # The pre-ADR one-list-per-worker shape (equal cardinality across all
    # four) remains valid for its callers.
    shared_edge = len(tunnel_units) == 1 and len(stage_receipts) == 1
    if (
        not public_urls
        or len(public_urls) != len(local_urls)
        or (
            not shared_edge
            and (
                len(public_urls) != len(tunnel_units)
                or len(public_urls) != len(stage_receipts)
            )
        )
    ):
        raise RuntimeError(
            "Worker public URLs, tunnel local URLs, tunnel units, and stage receipts must have equal cardinality"
        )
    if len(set(public_urls)) != len(public_urls):
        raise RuntimeError("Worker public URLs must be unique per worker")
    if len(set(tunnel_units)) != len(tunnel_units):
        raise RuntimeError("Worker tunnel units must be unique per worker")
    if len(set(stage_receipts)) != len(stage_receipts):
        raise RuntimeError("Worker security stage receipts must be unique per worker")
    if expected_worker_count and len(public_urls) != expected_worker_count:
        raise RuntimeError(
            "Worker security contract URL count does not match the requested worker count"
        )

    _validate_worker_public_url_env(
        [
            "WORKER_API_AUTH_MODE=required",
            f"WORKER_PUBLIC_URLS={','.join(public_urls)}",
        ]
    )
    unit_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]*\.service$")
    for index, (local_url, tunnel_unit) in enumerate(zip(local_urls, tunnel_units)):
        try:
            parsed = urlsplit(local_url)
            parsed_port = parsed.port
        except (TypeError, ValueError):
            raise RuntimeError("Worker tunnel local URL is invalid") from None
        if (
            parsed.scheme.lower() != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed_port != worker_port + index
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise RuntimeError(
                "Worker tunnel local URLs must map exactly to loopback worker ports"
            )
        if not unit_pattern.fullmatch(tunnel_unit):
            raise RuntimeError("Worker tunnel unit name is invalid")
        receipt_path = stage_receipts[index]
        if (
            not receipt_path.startswith("/")
            or any(character.isspace() or ord(character) < 32 for character in receipt_path)
        ):
            raise RuntimeError("Worker security stage receipt path is invalid")
    return {
        "public_urls": public_urls,
        "local_urls": local_urls,
        "tunnel_units": tunnel_units,
        "stage_receipts": stage_receipts,
        "backend_auth_mode": "bearer",
        "deploy_phase": rendered["WORKER_DEPLOY_PHASE"],
        "edge_provider": rendered.get("WORKER_EDGE_PROVIDER", "cloudflared"),
    }


def _resolve_worker_security_env(args: argparse.Namespace) -> list[str]:
    """Resolve security-contract values without contacting or mutating a provider."""

    env_vars = list(getattr(args, "env_vars", []) or [])
    existing_keys = {item.split("=", 1)[0] for item in env_vars if "=" in item}
    for key in (
        "GPU_WORKER_API_TOKEN",
        "WORKER_API_TOKEN",
        "WORKER_REGISTRATION_TOKEN",
        "FILMFORGE_BACKEND_URL",
        "WORKER_API_AUTH_MODE",
        "WORKER_PUBLIC_URLS",
        "WORKER_TUNNEL_LOCAL_URLS",
        "WORKER_TUNNEL_UNITS",
        "WORKER_SECURITY_STAGE_RECEIPTS",
        "WORKER_DEPLOY_PHASE",
        "FILMFORGE_BACKEND_CLIENT_AUTH_MODE",
    ):
        if key in existing_keys:
            continue
        value = _read_env_value(args.backend_env, key) or os.getenv(key)
        if value:
            env_vars.append(f"{key}={value}")
            existing_keys.add(key)
            log(f"Resolved {key} for worker security preflight")
    args.env_vars = env_vars
    return env_vars
# Models on the data volume: default warm set (flux 71 + wan 38 + ltx 70)
# ≈ 180 GB; +juggernaut 10 ≈ 190 GB. 300 GB leaves ~100 GB for ComfyUI outputs/temp/.part
# download slack (LTX checkpoint alone is 46 GB) and the next model. See
# Filmforge/backend/docs/discoveries/ltx-2-3-gpu-worker-install-2026-05-30.md.
DEFAULT_VERDA_FRESH_STORAGE_SIZE = 300
DEFAULT_VERDA_CONTRACT = "pay_as_go"
DEFAULT_WORKER_REPO_URL = "https://github.com/taxydriver/gpu_worker.git"
DEFAULT_COMFY_REPO_URL = "https://github.com/comfyanonymous/ComfyUI.git"
DEFAULT_PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"
# Pinned to the last set that works on both cu128 (A100) and cu130 (B200).
# torch 2.12.0+cu130 broke B200 deploys because it requires ncclCommResume,
# a symbol absent from both nvidia-nccl-cu12 2.28.9 and nvidia-nccl-cu13 2.29.7.
_TORCH_PIN = "torch==2.11.0"
_TORCHVISION_PIN = "torchvision==0.26.0"
_TORCHAUDIO_PIN = "torchaudio==2.11.0"
DEFAULT_VERDA_FRESH_WARM_GROUPS = ["flux_stills_v1", "wan_i2v_v1", "ltx_i2v_v1", "character_loras_v1"]

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
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    log(f"+ {shlex.join(cmd)}")
    return subprocess.run(
        cmd,
        input=input_text,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=capture_output,
        check=check,
        timeout=timeout,
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
    # Ephemeral cloud GPUs (Vast/RunPod/Verda) reuse host:port across instances,
    # so a cached known_hosts entry from a destroyed rental will reject the new
    # one with "Host key verification failed". Bypass known_hosts entirely.
    if has_ssh_option(cmd, "StrictHostKeyChecking"):
        return cmd
    return [
        cmd[0],
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "GlobalKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        *cmd[1:],
    ]


def add_default_ssh_liveness_policy(cmd: list[str]) -> list[str]:
    """Bound connection setup and detect an interrupted cloud VM promptly."""
    defaults = (
        ("ConnectTimeout", "8"),
        ("BatchMode", "yes"),
        ("ServerAliveInterval", "15"),
        ("ServerAliveCountMax", "3"),
    )
    additions: list[str] = []
    for option, value in defaults:
        if not has_ssh_option(cmd, option):
            additions.extend(["-o", f"{option}={value}"])
    if not additions:
        return cmd
    return [cmd[0], *additions, *cmd[1:]]


def _run_verda_ssh_script(
    ssh_cmd: list[str],
    script: str,
    *,
    timeout_sec: int,
    capture_output: bool,
    operation: str,
) -> subprocess.CompletedProcess[str]:
    try:
        return run(
            [*ssh_cmd, "bash", "-s"],
            input_text=script,
            capture_output=capture_output,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out after {timeout_sec}s while {operation}; "
            "the Verda VM may have been interrupted or become unreachable"
        ) from exc


_VERDA_REHYDRATE_STATE_PROBE = r"""#!/usr/bin/env bash
set -u

data_state="missing"
if test -b /dev/vdb; then
  fs_type="$(blkid -s TYPE -o value /dev/vdb 2>/dev/null || true)"
  if test -n "$fs_type"; then
    data_state="filesystem:$fs_type"
  else
    signatures="$(wipefs -n --noheadings --output TYPE /dev/vdb 2>/dev/null | tr -d '[:space:]' || true)"
    size_bytes="$(blockdev --getsize64 /dev/vdb 2>/dev/null || true)"
    sample_bytes=$((4 * 1024 * 1024))
    samples_are_zero=0
    if test -z "$signatures" && test "${size_bytes:-0}" -ge "$sample_bytes"; then
      last_sample=$(((size_bytes - sample_bytes) / sample_bytes))
      if cmp -s <(dd if=/dev/vdb bs=4M count=1 status=none) \
                <(dd if=/dev/zero bs=4M count=1 status=none) \
          && cmp -s <(dd if=/dev/vdb bs=4M skip="$last_sample" count=1 status=none) \
                    <(dd if=/dev/zero bs=4M count=1 status=none); then
        samples_are_zero=1
      fi
    fi
    if test "$samples_are_zero" -eq 1; then
      data_state="blank"
    else
      data_state="unknown"
    fi
  fi
fi

worker_ready=0
comfy_ready=0
bootstrap_ready=0
# Worker code now arrives as an immutable content-addressed release with its
# own venv; the legacy runtime venv exists only on pre-refactor volumes.
test -x /opt/filmforge_gpu_worker/.venv/bin/python && worker_ready=1
for candidate_python in /opt/filmforge-worker-releases/releases/*/.venv/bin/python; do
  if test -x "$candidate_python"; then
    worker_ready=1
    break
  fi
done
test -x /workspace/ComfyUI/.venv/bin/python && comfy_ready=1
if test "$data_state" = "filesystem:ext4"; then
  if mountpoint -q /mnt/data 2>/dev/null; then
    test -f /mnt/data/.filmforge-bootstrap-complete && bootstrap_ready=1
  else
    probe_mount="/run/filmforge-data-probe"
    mkdir -p "$probe_mount"
    if mount -o ro,noload /dev/vdb "$probe_mount" 2>/dev/null; then
      test -f "$probe_mount/.filmforge-bootstrap-complete" && bootstrap_ready=1
      umount "$probe_mount" 2>/dev/null || true
    fi
    rmdir "$probe_mount" 2>/dev/null || true
  fi
fi

printf 'DATA_STATE=%s\n' "$data_state"
printf 'WORKER_READY=%s\n' "$worker_ready"
printf 'COMFY_READY=%s\n' "$comfy_ready"
printf 'BOOTSTRAP_READY=%s\n' "$bootstrap_ready"
"""


def _probe_verda_rehydrate_state(ssh_cmd: list[str]) -> dict[str, str]:
    """Read enough remote state to distinguish a reusable pair from a failed fresh install."""
    result = _run_verda_ssh_script(
        ssh_cmd,
        _VERDA_REHYDRATE_STATE_PROBE,
        timeout_sec=60,
        capture_output=True,
        operation="checking the attached Verda volumes",
    )
    state: dict[str, str] = {}
    for line in (result.stdout or "").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {
            "DATA_STATE", "WORKER_READY", "COMFY_READY", "BOOTSTRAP_READY"
        }:
            state[key] = value.strip()
    missing = {
        "DATA_STATE", "WORKER_READY", "COMFY_READY", "BOOTSTRAP_READY"
    } - state.keys()
    if missing:
        raise RuntimeError(
            "Could not determine whether the attached Verda volumes are reusable; "
            f"probe omitted {', '.join(sorted(missing))}"
        )
    return state


def _verda_pair_needs_bootstrap(state: dict[str, str]) -> bool:
    """Return true for a safe-to-initialize or partially installed Verda pair."""
    data_state = state["DATA_STATE"]
    if data_state == "blank":
        return True
    if data_state == "filesystem:ext4":
        return (
            state["BOOTSTRAP_READY"] != "1"
            or state["WORKER_READY"] != "1"
            or state["COMFY_READY"] != "1"
        )
    if data_state == "missing":
        raise RuntimeError("Verda data volume is not present as /dev/vdb; refusing to deploy")
    if data_state == "unknown":
        raise RuntimeError(
            "Verda data volume has no recognized filesystem but is not safely blank; "
            "refusing to format it automatically"
        )
    if data_state.startswith("filesystem:"):
        fs_type = data_state.split(":", 1)[1]
        raise RuntimeError(
            f"Verda data volume uses unsupported filesystem {fs_type!r}; expected ext4"
        )
    raise RuntimeError(f"Unexpected Verda data-volume state {data_state!r}")


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


def _stage_worker_release_over_ssh(
    *,
    ssh_cmd: list[str],
    scp_cmd: list[str],
    destination: str,
    releases_root: str,
    venv_path: str,
    bundle: WorkerReleaseBundle | None = None,
) -> tuple[str, str]:
    """Upload and atomically activate a content-addressed worker package.

    The remote archive is installed beside any historical checkout.  No file in
    that checkout is copied over, edited, reset, or pulled, so a dirty H100 box
    cannot turn a failed update into a partial service deployment.
    """

    if bundle is None:
        try:
            bundle_context = build_worker_release_bundle(
                SCRIPT_DIR,
                require_committed_source=True,
            )
        except WorkerReleaseError as exc:
            raise RuntimeError(str(exc)) from exc
        with bundle_context as local_bundle:
            return _stage_worker_release_over_ssh(
                ssh_cmd=ssh_cmd,
                scp_cmd=scp_cmd,
                destination=destination,
                releases_root=releases_root,
                venv_path=venv_path,
                bundle=local_bundle,
            )
    else:
        remote_archive = f"/tmp/filmforge-worker-{bundle.release_id}.tar.gz"
        run(
            [
                *scp_cmd,
                str(bundle.archive_path),
                f"{destination}:{remote_archive}",
            ]
        )
        install_script = worker_release_install_script(
            archive_path=remote_archive,
            archive_sha256=bundle.archive_sha256,
            source_sha256=bundle.source_sha256,
            release_id=bundle.release_id,
            releases_root=releases_root,
            venv_path=venv_path,
            git_commit=bundle.git_commit,
            tracked_manifest_sha256=bundle.tracked_manifest_sha256,
        )
        try:
            run(
                [*ssh_cmd, "bash", "-s"],
                input_text=install_script,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            # The installer is fail-closed and says exactly why it stopped —
            # but with capture_output=True that diagnosis dies in the captured
            # streams unless it is quoted here (its output is tar/pip/check
            # messages; secrets never flow through this script).
            stderr_tail = " ".join(str(exc.stderr or "")[-400:].split())
            stdout_tail = " ".join(str(exc.stdout or "")[-200:].split())
            raise RuntimeError(
                f"worker release install failed on the box (rc={exc.returncode}; "
                f"stderr ends: {stderr_tail!r}; stdout ends: {stdout_tail!r})"
            ) from None
        return (
            bundle.release_id,
            f"{releases_root}/releases/{bundle.release_id}/gpu_worker",
        )


def _prepare_worker_release_bundle(args: argparse.Namespace) -> WorkerReleaseBundle:
    """Build once before provider mutation and retain those exact bytes for SSH."""

    existing = getattr(args, "_prepared_worker_release_bundle", None)
    if isinstance(existing, WorkerReleaseBundle):
        return existing
    try:
        bundle = build_worker_release_bundle(
            SCRIPT_DIR,
            require_committed_source=True,
        )
    except WorkerReleaseError as exc:
        raise RuntimeError(str(exc)) from exc
    setattr(args, "_prepared_worker_release_bundle", bundle)
    return bundle


def _activate_worker_release_over_ssh(
    *,
    ssh_cmd: list[str],
    releases_root: str,
    release_id: str,
    worker_source_root: str,
    stage_receipt_paths: list[str],
) -> None:
    """Reverify profile CAS and promote code while holding the profile lock."""

    activation = worker_release_activate_script(
        releases_root=releases_root,
        release_id=release_id,
    )
    validator = f"""\
import json
import pathlib
import stat
import subprocess
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

failed_release_id = {release_id!r}
candidate_source = pathlib.Path({worker_source_root!r})
receipt_paths = {[str(path) for path in stage_receipt_paths]!r}
candidate_root = candidate_source.parent
if candidate_source.name != "gpu_worker" or candidate_root.name != failed_release_id:
    raise SystemExit("worker candidate path does not match finalization release")
if not receipt_paths:
    raise SystemExit("worker finalization requires secure-profile receipts")
for raw_path in receipt_paths:
    receipt_path = pathlib.Path(raw_path)
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise SystemExit("secure-profile receipt is not a regular file")
    if stat.S_IMODE(receipt_path.stat().st_mode) != 0o600:
        raise SystemExit("secure-profile receipt must have mode 0600")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "filmforge.worker-secure-stage.v1":
        raise SystemExit("secure-profile receipt schema is invalid")
    if receipt.get("cutover_performed") is not True:
        raise SystemExit("secure-profile cutover is not complete")
    if receipt.get("rollback_state") == "in_progress":
        raise SystemExit("secure-profile rollback is in progress")
    if receipt.get("worker_code_release_id") != failed_release_id:
        raise SystemExit("secure-profile receipt is bound to different worker code")
    if pathlib.Path(str(receipt.get("worker_module_dir"))) != candidate_root:
        raise SystemExit("secure-profile module directory is not the candidate")
    if pathlib.Path(str(receipt.get("worker_exec"))) != candidate_root / ".venv/bin/python":
        raise SystemExit("secure-profile executable is not the candidate")
    profile_release_id = receipt.get("release_id")
    worker_unit = receipt.get("worker_unit")
    tunnel_unit = receipt.get("tunnel_unit")
    if not all(isinstance(value, str) and value for value in (
        profile_release_id, worker_unit, tunnel_unit
    )):
        raise SystemExit("secure-profile identity is invalid")
    profile_dir = receipt_path.parent
    active = pathlib.Path("/etc/filmforge/worker-security/active") / worker_unit
    worker_dropin_dir = pathlib.Path("/etc/systemd/system") / f"{{worker_unit}}.d"
    tunnel_dropin_dir = pathlib.Path("/etc/systemd/system") / f"{{tunnel_unit}}.d"
    expected_links = {{
        active: profile_dir,
        worker_dropin_dir / "20-filmforge-secure-profile.conf": profile_dir / "worker-secure-profile.conf",
        tunnel_dropin_dir / "20-filmforge-secure-profile.conf": profile_dir / "tunnel-secure-profile.conf",
        pathlib.Path("/etc/systemd/system") / tunnel_unit: profile_dir / "filmforge-worker-tunnel.service",
    }}
    for managed, expected in expected_links.items():
        if not managed.is_symlink() or managed.resolve() != expected.resolve():
            raise SystemExit("secure-profile managed link changed before code finalization")
    authorization = profile_dir / "cutover-authorized"
    if (
        authorization.is_symlink()
        or not authorization.is_file()
        or stat.S_IMODE(authorization.stat().st_mode) != 0o600
        or authorization.read_text() != profile_release_id + "\\n"
    ):
        raise SystemExit("secure-profile cutover authorization drifted")
    if receipt.get("profile_mode") == "first-install":
        boot_authorization = profile_dir / "boot-authorized"
        if (
            boot_authorization.is_symlink()
            or not boot_authorization.is_file()
            or stat.S_IMODE(boot_authorization.stat().st_mode) != 0o600
            or boot_authorization.read_text() != profile_release_id + "\\n"
        ):
            raise SystemExit("first-install boot authorization drifted")
    for unit in (worker_unit, tunnel_unit):
        active_result = subprocess.run(
            ["systemctl", "is-active", "--quiet", unit],
            check=False,
            text=True,
            capture_output=True,
        )
        if active_result.returncode != 0:
            raise SystemExit("worker or tunnel became inactive before code finalization")
    worker_secret = profile_dir / "worker-secrets.env"
    if (
        worker_secret.is_symlink()
        or not worker_secret.is_file()
        or stat.S_IMODE(worker_secret.stat().st_mode) != 0o600
    ):
        raise SystemExit("worker secret drifted before code finalization")
    secret_values = {{}}
    for raw_line in worker_secret.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or key in secret_values:
            raise SystemExit("worker secret is invalid before code finalization")
        secret_values[key] = value
    worker_token = secret_values.get("GPU_WORKER_API_TOKEN") or secret_values.get(
        "WORKER_API_TOKEN"
    )
    if not worker_token:
        raise SystemExit("worker bearer is absent before code finalization")

    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    local_health_url = str(receipt.get("tunnel_local_url") or "").rstrip("/") + "/health"
    health_request = Request(
        local_health_url,
        headers={{"Authorization": f"Bearer {{worker_token}}"}},
        method="GET",
    )
    try:
        with build_opener(ProxyHandler({{}}), NoRedirect()).open(
            health_request,
            timeout=10,
        ) as health_response:
            if health_response.status != 200:
                raise SystemExit("worker health status changed before code finalization")
            health_body = health_response.read(65537)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise SystemExit("worker health failed before code finalization") from exc
    if len(health_body) > 65536:
        raise SystemExit("worker health response is too large before code finalization")
    try:
        health = json.loads(health_body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SystemExit("worker health is not JSON before code finalization") from None
    if (
        not isinstance(health, dict)
        or health.get("ok") is not True
        or health.get("worker_ok") is not True
        or health.get("code_release_id") != failed_release_id
        or str(health.get("public_url") or "").rstrip("/")
        != str(receipt.get("worker_public_url") or "").rstrip("/")
    ):
        raise SystemExit("worker health identity changed before code finalization")
    if (worker_dropin_dir / "99-public-url-override.conf").exists():
        raise SystemExit("public override reappeared before code finalization")
"""
    script = f"""#!/usr/bin/env bash
set -euo pipefail
install -d -m 0700 /etc/filmforge/worker-security
exec 9>/etc/filmforge/worker-security/.profile.lock
flock 9
python3 - <<'PY'
{validator}PY
{activation}
"""
    run(
        [*ssh_cmd, "bash", "-s"],
        input_text=script,
        capture_output=True,
    )


def _activate_or_rollback_worker_release_over_ssh(
    *,
    ssh_cmd: list[str],
    releases_root: str,
    release_id: str,
    worker_source_root: str,
    stage_receipt_paths: list[str],
) -> None:
    try:
        _activate_worker_release_over_ssh(
            ssh_cmd=ssh_cmd,
            releases_root=releases_root,
            release_id=release_id,
            worker_source_root=worker_source_root,
            stage_receipt_paths=stage_receipt_paths,
        )
    except Exception:
        try:
            _rollback_secure_profiles_over_ssh(
                ssh_cmd=ssh_cmd,
                worker_source_root=worker_source_root,
                failed_release_id=release_id,
                stage_receipt_paths=stage_receipt_paths,
            )
        except Exception:
            log(
                "Worker metadata activation failed and the active secure profile "
                "could not be rolled back; leaving the code pointer unchanged"
            )
            raise
        try:
            _rollback_worker_release_over_ssh(
                ssh_cmd=ssh_cmd,
                releases_root=releases_root,
                failed_release_id=release_id,
            )
        except Exception:
            log(
                "Secure profile rollback succeeded, but the code pointer rollback "
                "was unavailable"
            )
        raise


def _rollback_worker_release_over_ssh(
    *,
    ssh_cmd: list[str],
    releases_root: str,
    failed_release_id: str,
) -> None:
    """Best-effort CAS rollback after a post-package bootstrap failure."""

    rollback_script = worker_release_rollback_script(
        releases_root=releases_root,
        failed_release_id=failed_release_id,
    )
    run(
        [*ssh_cmd, "bash", "-s"],
        input_text=rollback_script,
        capture_output=True,
    )


def _rollback_secure_profiles_over_ssh(
    *,
    ssh_cmd: list[str],
    worker_source_root: str,
    failed_release_id: str,
    stage_receipt_paths: list[str],
) -> None:
    """Roll back receipt-pinned secure profiles before changing the code CAS.

    A secure drop-in pins the immutable candidate directly, so changing only
    ``current`` cannot restore service.  The remote helper imports the exact
    candidate's repo-managed rollback implementation and accepts only protected
    receipts bound to that candidate.  No credential is placed in argv.
    """

    if not stage_receipt_paths:
        raise RuntimeError("secure-profile rollback requires stage receipts")
    remote_helper = f"""\
import json
import pathlib
import stat
import sys

worker_source_root = pathlib.Path({worker_source_root!r})
failed_release_id = {failed_release_id!r}
receipt_paths = {[str(path) for path in stage_receipt_paths]!r}
candidate_root = worker_source_root.parent
if worker_source_root.name != "gpu_worker" or candidate_root.name != failed_release_id:
    raise SystemExit("worker candidate path does not match failed release")
if not (candidate_root / ".ready").is_file():
    raise SystemExit("worker candidate readiness marker is missing")
# The candidate is immutable, but this verifier runs as root.  Suppress import
# bytecode explicitly so a rollback cannot leave a writable ``__pycache__`` in
# the release and make the next idempotent resume fail closed.
sys.dont_write_bytecode = True
sys.path.insert(0, str(candidate_root))
from gpu_worker.worker_release import rollback_secure_profiles

# Validate the entire fleet before the first profile mutation.  A later
# per-profile failure is aggregated after every remaining profile gets its own
# rollback attempt; each profile journal makes retry safe after SSH loss.
profiles = []
already_rolled_back = []
for raw_path in receipt_paths:
    receipt_path = pathlib.Path(raw_path)
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise SystemExit("secure-profile receipt is not a regular file")
    if stat.S_IMODE(receipt_path.stat().st_mode) != 0o600:
        raise SystemExit("secure-profile receipt must have mode 0600")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("secure-profile receipt could not be read") from exc
    if receipt.get("worker_code_release_id") != failed_release_id:
        raise SystemExit("secure-profile receipt is bound to different worker code")
    rollback_state = receipt.get("rollback_state")
    if receipt.get("cutover_performed") is False and rollback_state == "complete":
        already_rolled_back.append(str(receipt.get("release_id")))
        continue
    if receipt.get("cutover_performed") is not True or rollback_state not in {{None, "in_progress"}}:
        raise SystemExit("secure-profile receipt is not cut over or retryable")
    profile_release_id = receipt.get("release_id")
    if not isinstance(profile_release_id, str) or not profile_release_id:
        raise SystemExit("secure-profile receipt release id is invalid")
    profiles.append(profile_release_id)

rolled_back = []
if profiles:
    try:
        rollback_secure_profiles(release_ids=profiles)
        rolled_back.extend(reversed(profiles))
    except Exception as exc:
        print(
            "WORKER_SECURITY_PROFILE_ROLLBACK_FAILED="
            + f"{{type(exc).__name__}}: {{exc}}",
            file=sys.stderr,
        )
        raise SystemExit("one or more secure profiles require rollback retry") from exc
completed = rolled_back + already_rolled_back
if not completed:
    raise SystemExit("no cut-over secure profile was available to roll back")
print("WORKER_SECURITY_PROFILES_ROLLED_BACK=" + ",".join(completed))
"""
    run(
        [*ssh_cmd, "python3", "-"],
        input_text=remote_helper,
        capture_output=True,
    )


def _rollback_failed_worker_transaction_over_ssh(
    *,
    ssh_cmd: list[str],
    releases_root: str,
    worker_source_root: str,
    failed_release_id: str,
    stage_receipt_paths: list[str],
    deploy_phase: str,
) -> bool:
    """Restore profile first, then code CAS; never claim a pointer-only fix."""

    if deploy_phase == "activate":
        try:
            _rollback_secure_profiles_over_ssh(
                ssh_cmd=ssh_cmd,
                worker_source_root=worker_source_root,
                failed_release_id=failed_release_id,
                stage_receipt_paths=stage_receipt_paths,
            )
            log("Restored the receipt-pinned worker security profiles")
        except Exception:
            log(
                "Active secure-profile rollback failed; the code pointer is left "
                "unchanged and no recovery claim is made"
            )
            return False
    try:
        _rollback_worker_release_over_ssh(
            ssh_cmd=ssh_cmd,
            releases_root=releases_root,
            failed_release_id=failed_release_id,
        )
        log("Rolled worker code pointer back after bootstrap failure")
    except Exception:
        # A first-ever install has no previous pointer. The immutable candidate
        # remains for inspection; profile rollback, when required, already ran.
        log("Worker code rollback was unavailable; no code recovery is claimed")
        return False
    return True


def build_env_exports(env_items: list[str]) -> str:
    exports: list[str] = []
    for item in env_items:
        key, sep, value = str(item).partition("=")
        key = key.strip()
        if not sep or not key:
            continue
        exports.append(f"export {key}={shlex.quote(value)}")
    return "\n".join(exports)


_WORKER_BOOTSTRAP_SECRET_KEYS = {
    "GPU_WORKER_API_TOKEN",
    "WORKER_API_TOKEN",
    "WORKER_REGISTRATION_TOKEN",
    "RENDER_BROKER_WORKER_TOKEN",
    "FILMFORGE_BACKEND_CUTOVER_PROBE_TOKEN",
}


def build_bootstrap_env_exports(env_items: list[str]) -> str:
    """Export non-worker configuration without leaking bearers to installers.

    Worker and registration credentials already live in the verified staged
    profile's mode-0600 EnvironmentFile. They must never be inherited by apt,
    pip, git, custom-node installers, or other third-party bootstrap children.
    """

    safe_items = []
    for item in env_items:
        key, separator, _value = str(item).partition("=")
        if separator and key.strip() not in _WORKER_BOOTSTRAP_SECRET_KEYS:
            safe_items.append(item)
    return build_env_exports(safe_items)


def worker_security_stage_gate_script() -> str:
    """Return a remote, read-only gate for the prepared profile receipts."""

    return r'''command -v flock >/dev/null 2>&1 || {
  echo "flock is required for coordinated worker/profile deployment" >&2
  exit 1
}
mkdir -p /etc/filmforge/worker-security
chmod 0700 /etc/filmforge/worker-security
exec 9>/etc/filmforge/worker-security/.profile.lock
chmod 0600 /etc/filmforge/worker-security/.profile.lock
flock -x 9

: "${WORKER_SECURITY_STAGE_RECEIPTS:?WORKER_SECURITY_STAGE_RECEIPTS is required}"
: "${WORKER_PUBLIC_URLS:?WORKER_PUBLIC_URLS is required}"
: "${WORKER_TUNNEL_LOCAL_URLS:?WORKER_TUNNEL_LOCAL_URLS is required}"
: "${WORKER_TUNNEL_UNITS:?WORKER_TUNNEL_UNITS is required}"
: "${WORKER_DEPLOY_PHASE:?WORKER_DEPLOY_PHASE is required}"
: "${WORKER_CODE_RELEASE_ID:?WORKER_CODE_RELEASE_ID is required}"
: "${WORKER_EDGE_PROVIDER:=cloudflared}"
WORKER_SECURITY_GATE_RESULT="$(python3 - \
  "$WORKER_SECURITY_STAGE_RECEIPTS" \
  "$WORKER_PUBLIC_URLS" \
  "$WORKER_TUNNEL_LOCAL_URLS" \
  "$WORKER_TUNNEL_UNITS" \
  "$WORKER_DEPLOY_PHASE" \
  "$WORKER_CODE_RELEASE_ID" \
  "$WORKER_EDGE_PROVIDER" <<'PY'
import hashlib
import json
import os
import pathlib
import socket
import stat
import subprocess
import sys
from urllib.parse import urlsplit

receipt_paths, public_urls, local_urls, tunnel_units = [
    [item.strip() for item in value.split(",") if item.strip()]
    for value in sys.argv[1:5]
]
deploy_phase = sys.argv[5]
code_release_id = sys.argv[6]
edge_provider = sys.argv[7]
if edge_provider not in {"cloudflared", "caddy"}:
    raise SystemExit("invalid worker edge provider")
if not receipt_paths or not (
    len(receipt_paths) == len(public_urls) == len(local_urls) == len(tunnel_units)
):
    raise SystemExit("prepared secure-profile receipt cardinality mismatch")
if deploy_phase == "stage-code":
    print("stage-code")
    raise SystemExit(0)
if deploy_phase not in {"activate", "provision-only"}:
    raise SystemExit("invalid worker deploy phase")
artifacts = {
    "worker_secret_sha256": ("worker-secrets.env", 0o600),
    "tunnel_secret_sha256": ("tunnel-secrets.env", 0o600),
    "backend_probe_secret_sha256": ("backend-cutover-probe.env", 0o600),
    "tunnel_config_sha256": ("cloudflared.yml", 0o600),
    "tunnel_credential_sha256": ("cloudflared-credential.json", 0o600),
    "tunnel_exec_sha256": ("filmforge-worker-tunnel", 0o755),
    "tunnel_binary_sha256": ("cloudflared", 0o755),
    "worker_dropin_sha256": ("worker-secure-profile.conf", 0o644),
    "tunnel_dropin_sha256": ("tunnel-secure-profile.conf", 0o644),
    "tunnel_unit_sha256": ("filmforge-worker-tunnel.service", 0o644),
    "worker_guard_sha256": ("worker-staged-guard.conf", 0o644),
}
all_cutover = True
all_first_install = True
any_cutover = False
for index, raw_path in enumerate(receipt_paths):
    path = pathlib.Path(raw_path)
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise SystemExit("secure-profile stage receipt is missing or not mode 0600")
    data = json.loads(path.read_text())
    if data.get("schema") != "filmforge.worker-secure-stage.v1":
        raise SystemExit("secure-profile stage receipt schema mismatch")
    if data.get("edge_provider", "cloudflared") != edge_provider:
        raise SystemExit("secure-profile edge provider does not match deploy contract")
    if edge_provider == "caddy":
        for field in ("tunnel_secret_sha256", "tunnel_config_sha256", "tunnel_credential_sha256", "tunnel_exec_sha256", "tunnel_binary_sha256"):
            artifacts.pop(field, None)
        artifacts.update({
            "caddy_secret_sha256": ("caddy-secrets.env", 0o600),
            "caddy_config_sha256": ("Caddyfile", 0o600),
            "caddy_exec_sha256": ("filmforge-worker-caddy", 0o755),
            "caddy_binary_sha256": ("caddy", 0o755),
        })
    if data.get("tunnel_prepared") is not True:
        raise SystemExit("secure-profile tunnel has not completed prepare")
    if data.get("worker_code_release_id") != code_release_id:
        raise SystemExit("secure-profile receipt pins a different worker code release")
    code_root = pathlib.Path(str(data.get("worker_module_dir") or ""))
    code_exec = pathlib.Path(str(data.get("worker_exec") or ""))
    source_marker = code_root / ".source-sha256"
    ready_marker = code_root / ".ready"
    dependency_freeze = code_root / ".dependency-freeze.txt"
    dependency_marker = code_root / ".dependency-freeze.sha256"
    if (
        code_root.name != code_release_id
        or code_root.parent.name != "releases"
        or code_exec != code_root / ".venv/bin/python"
        or code_root.is_symlink()
        or not code_root.is_dir()
        or code_exec.is_symlink()
        or not code_exec.is_file()
        or not os.access(code_exec, os.X_OK)
        or ready_marker.is_symlink()
        or not ready_marker.is_file()
        or stat.S_IMODE(ready_marker.stat().st_mode) != 0o444
        or source_marker.is_symlink()
        or not source_marker.is_file()
        or dependency_freeze.is_symlink()
        or not dependency_freeze.is_file()
        or dependency_marker.is_symlink()
        or not dependency_marker.is_file()
    ):
        raise SystemExit("secure-profile code candidate path or markers drifted")
    source_digest = source_marker.read_text().strip()
    dependency_digest = dependency_marker.read_text().strip()
    if (
        ready_marker.read_text().strip() != source_digest
        or code_release_id != f"sha256-{source_digest[:24]}"
        or hashlib.sha256(dependency_freeze.read_bytes()).hexdigest()
        != dependency_digest
        or data.get("worker_dependency_freeze_sha256") != dependency_digest
    ):
        raise SystemExit("secure-profile code candidate digest drifted")
    expected = (
        ("worker_public_url", public_urls[index].rstrip("/")),
        ("tunnel_local_url", local_urls[index].rstrip("/")),
        ("tunnel_unit", tunnel_units[index]),
    )
    if any(data.get(key) != value for key, value in expected):
        raise SystemExit("secure-profile stage receipt does not match deploy contract")
    release = path.parent
    for field, (relative, expected_mode) in artifacts.items():
        artifact = release / relative
        if artifact.is_symlink() or not artifact.is_file():
            raise SystemExit("secure-profile staged artifact is missing")
        if stat.S_IMODE(artifact.stat().st_mode) != expected_mode:
            raise SystemExit("secure-profile staged artifact mode drifted")
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != data.get(field):
            raise SystemExit("secure-profile staged artifact drifted")
    worker_unit = data["worker_unit"]
    managed = {
        pathlib.Path("/etc/systemd/system") / tunnel_units[index]: release / "filmforge-worker-tunnel.service",
        pathlib.Path("/etc/systemd/system") / f"{tunnel_units[index]}.d/20-filmforge-secure-profile.conf": release / "tunnel-secure-profile.conf",
    }
    worker_profile = pathlib.Path("/etc/systemd/system") / f"{worker_unit}.d/20-filmforge-secure-profile.conf"
    if data.get("cutover_performed") is True:
        managed[worker_profile] = release / "worker-secure-profile.conf"
        active_profile = (
            pathlib.Path("/etc/filmforge/worker-security/active") / worker_unit
        )
        if (
            not active_profile.is_symlink()
            or active_profile.resolve() != release.resolve()
        ):
            raise SystemExit("cutover secure-profile active pointer is incomplete")
    elif worker_profile.exists() or worker_profile.is_symlink():
        raise SystemExit("worker secure profile activated before cutover")
    if data.get("profile_mode") == "first-install":
        managed[
            pathlib.Path("/etc/systemd/system")
            / f"{worker_unit}.d/00-filmforge-staged-guard.conf"
        ] = release / "worker-staged-guard.conf"
    if any(
        not link.is_symlink() or link.resolve() != target.resolve()
        for link, target in managed.items()
    ):
        raise SystemExit("secure-profile managed systemd links are incomplete")
    override = pathlib.Path("/etc/systemd/system") / f"{worker_unit}.d/99-public-url-override.conf"
    mode = data.get("profile_mode")
    all_first_install = all_first_install and mode == "first-install"
    all_cutover = all_cutover and data.get("cutover_performed") is True
    any_cutover = any_cutover or data.get("cutover_performed") is True
    if mode == "migration" and data.get("cutover_performed") is not True:
        if override.is_symlink() or not override.is_file():
            raise SystemExit("migration public override disappeared before cutover")
        if hashlib.sha256(override.read_bytes()).hexdigest() != data.get("public_override_sha256"):
            raise SystemExit("migration public override drifted")
    elif override.exists() or override.is_symlink():
        raise SystemExit("unexpected public override in active/first-install profile")
    if data.get("cutover_performed") is True:
        authorization = release / "cutover-authorized"
        if (
            authorization.is_symlink()
            or not authorization.is_file()
            or stat.S_IMODE(authorization.stat().st_mode) != 0o600
            or authorization.read_text() != f"{data['release_id']}\n"
        ):
            raise SystemExit("cutover authorization drifted")
        if mode == "first-install":
            boot_authorization = release / "boot-authorized"
            if (
                boot_authorization.is_symlink()
                or not boot_authorization.is_file()
                or stat.S_IMODE(boot_authorization.stat().st_mode) != 0o600
                or boot_authorization.read_text() != f"{data['release_id']}\n"
            ):
                raise SystemExit("first-install boot authorization drifted")
    elif (release / "cutover-authorized").exists() or (
        release / "cutover-authorized"
    ).is_symlink():
        raise SystemExit("cutover authorization appeared before receipt commit")
if any_cutover and not all_cutover:
    raise SystemExit("mixed secure-profile cutover state is not atomic")
if deploy_phase == "provision-only":
    if not all_first_install:
        raise SystemExit("provision-only requires prepared first-install profiles")
    if any_cutover:
        raise SystemExit("provision-only refuses an already cut-over profile")
    active_units = subprocess.run(
        [
            "systemctl",
            "list-units",
            "--type=service",
            "--state=active",
            "--no-legend",
            "--no-pager",
        ],
        check=False,
        text=True,
        capture_output=True,
    ).stdout
    if any(
        "filmforge-worker-gpu" in line.lower() or "gpu_worker" in line.lower()
        for line in active_units.splitlines()
    ):
        raise SystemExit("provision-only refuses a live worker service")
    for process in pathlib.Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = process.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (OSError, PermissionError):
            continue
        if "gpu_worker.app:app" in command or "gpu_worker/app.py" in command:
            raise SystemExit("provision-only refuses a live legacy worker process")
    for local_url in local_urls:
        port = urlsplit(local_url).port
        if port is None:
            raise SystemExit("planned worker loopback URL has no port")
        try:
            connection = socket.create_connection(("127.0.0.1", port), timeout=0.2)
        except OSError:
            continue
        connection.close()
        raise SystemExit("provision-only refuses an occupied planned worker port")
    print("provision-only")
    raise SystemExit(0)
print("cutover" if all_cutover else "prepared")
PY
 )"
case "$WORKER_SECURITY_GATE_RESULT" in
  cutover) export WORKER_SECURITY_CUTOVER_COMPLETE=1 ;;
  prepared|provision-only|stage-code) export WORKER_SECURITY_CUTOVER_COMPLETE=0 ;;
  *) echo "unexpected worker security gate result" >&2; exit 1 ;;
esac
printf 'WORKER_SECURITY_GATE=%s\n' "$WORKER_SECURITY_GATE_RESULT"
if test "$WORKER_SECURITY_GATE_RESULT" = "prepared"; then
  echo "WORKER_RELEASE_STAGED_ONLY=${WORKER_CODE_RELEASE_ID}"
  echo "Prepared migration remains on its current worker until explicit cutover." >&2
  exit 0
fi
if test "$WORKER_SECURITY_GATE_RESULT" = "stage-code"; then
  echo "WORKER_RELEASE_STAGED_ONLY=$WORKER_CODE_RELEASE_ID"
  echo "Immutable worker candidate staged without GPU or service mutation." >&2
  exit 0
fi
if test "$WORKER_SECURITY_GATE_RESULT" = "cutover"; then
  python3 - "$WORKER_TUNNEL_LOCAL_URLS" "$WORKER_CODE_RELEASE_ID" <<'PY'
import json
import sys
from urllib.request import ProxyHandler, build_opener

urls = [item.strip().rstrip("/") for item in sys.argv[1].split(",") if item.strip()]
release_id = sys.argv[2]
opener = build_opener(ProxyHandler({}))
for url in urls:
    with opener.open(f"{url}/health", timeout=15) as response:
        health = json.load(response)
    if health.get("code_release_id") != release_id:
        raise SystemExit("cutover worker health came from a different code release")
PY
  echo "WORKER_RELEASE_VERIFIED=$WORKER_CODE_RELEASE_ID"
  echo "Receipt-gated cutover remains the final worker restart; no service files changed." >&2
  exit 0
fi
'''


def remote_script(
    remote_root: str,
    worker_port: int,
    worker_source_root: str | None = None,
) -> str:
    worker_source_root = worker_source_root or f"{remote_root.rstrip('/')}/gpu_worker"
    security_gate = worker_security_stage_gate_script()
    # Secure deployments use the systemd multi-worker path even for one GPU.
    # Keep this compatibility entry point fail-closed instead of embedding the
    # former nohup/env launcher, which exposed bearer values in process argv.
    return f"""#!/usr/bin/env bash
set -euo pipefail

{security_gate}

# GPU_WORKER_API_TOKEN, WORKER_API_AUTH_MODE, and
# WORKER_INPUT_URL_ALLOWED_HOSTS are supported only by the protected systemd
# EnvironmentFile path emitted by vast_multi_gpu_script.
echo "legacy single-worker bootstrap is disabled; use the atomic systemd deploy path" >&2
exit 1
"""

    # Historical implementation retained below temporarily for source-level
    # archaeology; it is unreachable and is not emitted by this function.
    return f"""#!/usr/bin/env bash
set -euo pipefail

{security_gate}

REMOTE_ROOT={shlex.quote(remote_root)}
WORKER_SOURCE_ROOT={shlex.quote(worker_source_root)}
WORKER_MODULE_DIR="$(dirname "$WORKER_SOURCE_ROOT")"
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
test -x "$WORKER_SOURCE_ROOT/.venv/bin/python" || {{
  echo "immutable worker candidate venv is missing" >&2
  exit 1
}}

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
ENV_VARS+=(PYTHONPATH="$WORKER_MODULE_DIR")
ENV_VARS+=(COMFY_BASE_URL="$COMFY_BASE_URL")
ENV_VARS+=(COMFY_OUTPUT_DIR="$COMFY_OUTPUT_DIR")
ENV_VARS+=(COMFY_TEMP_DIR="$COMFY_TEMP_DIR")
ENV_VARS+=(COMFY_INPUT_DIR="$COMFY_INPUT_DIR")
ENV_VARS+=(COMFY_STOP_CMD="$COMFY_STOP_CMD")
ENV_VARS+=(COMFY_START_CMD="$COMFY_START_CMD")
ENV_VARS+=(WORKER_PROVIDER="${{WORKER_PROVIDER:-dedicated_worker}}")
test -n "${{WORKER_INSTANCE_ID:-}}" && ENV_VARS+=(WORKER_INSTANCE_ID="${{WORKER_INSTANCE_ID}}")
ENV_VARS+=(WORKER_MAX_CONCURRENT_JOBS="${{WORKER_MAX_CONCURRENT_JOBS:-10}}")
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
ENV_VARS+=(WORKER_CAPABILITIES="${{WORKER_CAPABILITIES:-flux2_stills,wan_i2v,ltx_i2v,character_loras}}")
test -n "${{WORKER_REGISTRATION_TOKEN:-}}" && ENV_VARS+=(WORKER_REGISTRATION_TOKEN="${{WORKER_REGISTRATION_TOKEN}}")
test -n "${{WORKER_API_TOKEN:-}}" && ENV_VARS+=(WORKER_API_TOKEN="${{WORKER_API_TOKEN}}")
test -n "${{GPU_WORKER_API_TOKEN:-}}" && ENV_VARS+=(GPU_WORKER_API_TOKEN="${{GPU_WORKER_API_TOKEN}}")
test -n "${{WORKER_API_AUTH_MODE:-}}" && ENV_VARS+=(WORKER_API_AUTH_MODE="${{WORKER_API_AUTH_MODE}}")
test -n "${{WORKER_INPUT_URL_ALLOWED_HOSTS:-}}" && ENV_VARS+=(WORKER_INPUT_URL_ALLOWED_HOSTS="${{WORKER_INPUT_URL_ALLOWED_HOSTS}}")

nohup env "${{ENV_VARS[@]}}" \\
  "$WORKER_SOURCE_ROOT/.venv/bin/python" -m uvicorn gpu_worker.app:app --host 0.0.0.0 --port "$WORKER_PORT" \\
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
    "$WORKER_SOURCE_ROOT/.venv/bin/python" -m uvicorn gpu_worker.app:app --host 0.0.0.0 --port "$WORKER_PORT" \\
    >/tmp/gpu_worker.log 2>&1 </dev/null &
  for _ in $(seq 1 20); do
    if curl -fsS "http://127.0.0.1:$WORKER_PORT/health" >/tmp/gpu_worker_health.json 2>/dev/null; then
      break
    fi
    sleep 1
  done
  echo "Worker restarted with public URL" >&2
fi

# ── Optional Qwen3-VL vision sidecar (opt-in via ENABLE_QWEN_SIDECAR) ─────────
# Runs vLLM (OpenAI-compatible) alongside ComfyUI on the SAME GPU. VRAM is capped
# via --gpu-memory-utilization so the vision server never starves the render
# models. Exposed on its own cloudflared tunnel; the URL (with /v1) is printed as
# QWEN_URL for the deployer to write into the backend's QWEN_BASE_URL.
QWEN_URL=""
if test "${{ENABLE_QWEN_SIDECAR:-}}" = "true"; then
  if command -v docker >/dev/null 2>&1; then
    QWEN_MODEL="${{QWEN_MODEL:-Qwen/Qwen3-VL-8B-Instruct}}"
    QWEN_GPU_FRACTION="${{QWEN_GPU_FRACTION:-0.25}}"
    mkdir -p /workspace/hf_cache
    echo "Starting Qwen3-VL vision sidecar ($QWEN_MODEL, gpu-frac=$QWEN_GPU_FRACTION)..." >&2
    docker rm -f qwen-vision >/dev/null 2>&1 || true
    docker run -d --name qwen-vision --gpus all -p 8000:8000 \\
      -v /workspace/hf_cache:/root/.cache/huggingface \\
      --shm-size 8g --restart unless-stopped \\
      vllm/vllm-openai:latest \\
      "$QWEN_MODEL" \\
      --served-model-name "$QWEN_MODEL" \\
      --trust-remote-code \\
      --max-model-len 16384 \\
      --gpu-memory-utilization "$QWEN_GPU_FRACTION" \\
      --limit-mm-per-prompt '{{"image": 4}}' >/dev/null 2>&1 || true
    echo "Waiting for Qwen vision server (up to ~5 min for first model download)..." >&2
    for _ in $(seq 1 60); do
      if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "Qwen vision server healthy" >&2
        break
      fi
      sleep 5
    done
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1 && test -x /opt/instance-tools/bin/cloudflared; then
      pkill -f "cloudflared tunnel --url http://127.0.0.1:8000" || true
      sleep 2
      >/tmp/filmforge_qwen_tunnel.log
      nohup /opt/instance-tools/bin/cloudflared tunnel --url "http://127.0.0.1:8000" --no-autoupdate \\
        >/tmp/filmforge_qwen_tunnel.log 2>&1 </dev/null &
      for _ in $(seq 1 30); do
        if grep -aEo 'https://[-a-z0-9]+\\.trycloudflare\\.com' /tmp/filmforge_qwen_tunnel.log | tail -n 1 >/tmp/filmforge_qwen_tunnel_url.txt 2>/dev/null; then
          QWEN_URL="$(tr -d '\\000' </tmp/filmforge_qwen_tunnel_url.txt)"
          break
        fi
        sleep 1
      done
    fi
    if test -n "$QWEN_URL"; then
      QWEN_URL="${{QWEN_URL}}/v1"
      echo "Qwen vision sidecar public URL: $QWEN_URL" >&2
    else
      echo "Qwen vision sidecar did not produce a public URL (check: docker logs qwen-vision)." >&2
    fi
  else
    echo "ENABLE_QWEN_SIDECAR=true but docker unavailable on this box; skipping sidecar." >&2
  fi
fi

printf 'REMOTE_ROOT=%s\\n' "$REMOTE_ROOT"
printf 'COMFY_BASE_URL=%s\\n' "$COMFY_BASE_URL"
printf 'WORKER_PORT=%s\\n' "$WORKER_PORT"
printf 'WORKER_URL=%s\\n' "$WORKER_URL"
printf 'QWEN_URL=%s\\n' "$QWEN_URL"
printf 'WORKER_HEALTH=%s\\n' "$(cat /tmp/gpu_worker_health.json)"
"""


def vast_multi_gpu_script(
    *,
    remote_root: str,
    worker_port: int,
    comfy_port: int,
    worker_count: int,
    worker_source_root: str | None = None,
) -> str:
    worker_source_root = worker_source_root or f"{remote_root.rstrip('/')}/gpu_worker"
    security_gate = worker_security_stage_gate_script()
    return f"""#!/usr/bin/env bash
set -euo pipefail

{security_gate}

REMOTE_ROOT={shlex.quote(remote_root)}
WORKER_SOURCE_ROOT={shlex.quote(worker_source_root)}
WORKER_MODULE_DIR="$(dirname "$WORKER_SOURCE_ROOT")"
WORKER_PORT_BASE={worker_port}
COMFY_PORT_BASE={comfy_port}
WORKER_COUNT_REQUESTED={worker_count}
COMFY_ROOT="/workspace/ComfyUI"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; GPU drivers are not ready" >&2
  exit 1
fi

HAS_SYSTEMD=0
if test -d /run/systemd/system && systemctl list-units >/dev/null 2>&1; then
  HAS_SYSTEMD=1
fi
if test "$HAS_SYSTEMD" != "1"; then
  echo "atomic secure worker deployment requires systemd; refusing credential-bearing nohup fallback" >&2
  exit 1
fi

PHYSICAL_GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
GPU_COUNT="$PHYSICAL_GPU_COUNT"
# An explicit request may LOWER the worker count but must NEVER exceed the
# physical GPUs: each worker is pinned to CUDA_VISIBLE_DEVICES=<idx>, so a gpuN
# beyond the real GPU count crash-loops forever on "No CUDA GPUs are available".
if test "$WORKER_COUNT_REQUESTED" -gt 0 && test "$WORKER_COUNT_REQUESTED" -le "$PHYSICAL_GPU_COUNT"; then
  GPU_COUNT="$WORKER_COUNT_REQUESTED"
fi
if test "$GPU_COUNT" -lt 1; then
  echo "No GPUs detected" >&2
  exit 1
fi

if ! test -f "$COMFY_ROOT/main.py"; then
  echo "ComfyUI not found at $COMFY_ROOT/main.py" >&2
  exit 1
fi

mkdir -p "$REMOTE_ROOT"
cd "$REMOTE_ROOT"
test -x "$WORKER_SOURCE_ROOT/.venv/bin/python" || {{
  echo "immutable worker candidate venv is missing" >&2
  exit 1
}}

if ! command -v aria2c >/dev/null 2>&1; then
  echo "Installing aria2c..." >&2
  apt-get install -y -q aria2 2>/dev/null || true
fi

# Stop the template's single ComfyUI process; multi-GPU mode owns one ComfyUI
# process per GPU so each process can be pinned with CUDA_VISIBLE_DEVICES.
if supervisorctl status comfyui >/dev/null 2>&1; then
  supervisorctl stop comfyui >/dev/null 2>&1 || true
fi
pkill -f "uvicorn gpu_worker.app:app" || true
pkill -f "main.py.*--port $COMFY_PORT_BASE" || true

mkdir -p "$COMFY_ROOT/output" "$COMFY_ROOT/temp" "$COMFY_ROOT/input"

COMFY_PYTHON="python3"
if test -x "/venv/main/bin/python"; then
  COMFY_PYTHON="/venv/main/bin/python"
elif test -x "$COMFY_ROOT/.venv/bin/python"; then
  COMFY_PYTHON="$COMFY_ROOT/.venv/bin/python"
elif test -x "$COMFY_ROOT/venv/bin/python"; then
  COMFY_PYTHON="$COMFY_ROOT/venv/bin/python"
fi

# Keep ComfyUI deps current — ComfyUI updates its requirements.txt when it adds
# new features (e.g. sqlalchemy/alembic for its local DB). Without this, fresh
# VMs that rehydrate an existing volume can crash-loop with ModuleNotFoundError.
echo "Installing/updating ComfyUI requirements..." >&2
# Write to a log instead of piping to `tail`: under `set -o pipefail` a SIGPIPE
# from the closed `tail` read-end (141) propagated out and aborted the whole
# `ssh ... bash -s` deploy at this step. Tail the file afterwards (no pipe on
# the long-running command).
"$COMFY_PYTHON" -m pip install -q -r "$COMFY_ROOT/requirements.txt" "transformers<5" > /tmp/comfy_reqs_install.log 2>&1 || true
tail -5 /tmp/comfy_reqs_install.log >&2 || true

# Query GPU 0 only (-i 0) instead of piping all GPUs to `head -1`: on a multi-GPU
# box, `nvidia-smi | head -1` races — head closes after one line and SIGPIPEs
# nvidia-smi, which under `set -o pipefail` returns 141 and aborts the deploy.
# This is why the deploy worked on 1-2 GPU boxes but failed on 8.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | xargs)"
VRAM_GB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | awk '{{printf "%.0f", $1 / 1024}}')"
PUBLIC_URLS="${{WORKER_PUBLIC_URLS:-}}"
mkdir -p /etc/systemd/system

public_url_for_idx() {{
  idx="$1"
  python3 - "$PUBLIC_URLS" "$idx" <<'PY'
import sys
urls = [u.strip() for u in (sys.argv[1] or "").split(",")]
idx = int(sys.argv[2])
print(urls[idx] if idx < len(urls) else "")
PY
}}

write_worker_secret_env() {{
  idx="$1"
  secret_dir="/etc/filmforge"
  secret_path="$secret_dir/worker-gpu${{idx}}.env"
  mkdir -p "$secret_dir"
  chmod 0700 "$secret_dir"
  secret_tmp="$(mktemp "$secret_dir/.worker-gpu${{idx}}.XXXXXX")"
  chmod 0600 "$secret_tmp"
  for key in WORKER_REGISTRATION_TOKEN RENDER_BROKER_WORKER_TOKEN WORKER_API_TOKEN GPU_WORKER_API_TOKEN; do
    value="$(printenv "$key" 2>/dev/null || true)"
    test -n "$value" || continue
    case "$value" in
      *[[:space:]]*) echo "worker credential contains unsupported whitespace" >&2; rm -f "$secret_tmp"; exit 1 ;;
    esac
    printf '%s=%s\n' "$key" "$value" >> "$secret_tmp"
  done
  install -m 0600 "$secret_tmp" "$secret_path"
  rm -f "$secret_tmp"
  printf '%s\n' "$secret_path"
}}

start_worker_no_systemd() {{
  idx="$1"
  worker_public_url="$2"
  comfy_port=$((COMFY_PORT_BASE + idx))
  worker_port=$((WORKER_PORT_BASE + idx))
  pkill -f "uvicorn gpu_worker.app:app --host 0.0.0.0 --port ${{worker_port}}" || true
  declare -a worker_env
  worker_env+=(PYTHONPATH="$WORKER_MODULE_DIR")
  worker_env+=(CUDA_VISIBLE_DEVICES="$idx")
  worker_env+=(COMFY_BASE_URL="http://127.0.0.1:${{comfy_port}}")
  worker_env+=(COMFY_OUTPUT_DIR="$COMFY_ROOT/output")
  worker_env+=(COMFY_TEMP_DIR="$COMFY_ROOT/temp")
  worker_env+=(COMFY_INPUT_DIR="$COMFY_ROOT/input")
  worker_env+=(WORKER_HOST="0.0.0.0")
  worker_env+=(WORKER_PORT="$worker_port")
  worker_env+=(WORKER_NAME="filmforge-vast-${{HOSTNAME:-instance}}-gpu${{idx}}")
  worker_env+=(WORKER_PROVIDER="vast")
  worker_env+=(WORKER_GPU_NAME="$GPU_NAME")
  worker_env+=(WORKER_VRAM_GB="$VRAM_GB")
  worker_env+=(WORKER_CAPABILITIES="${{WORKER_CAPABILITIES:-flux2_stills,wan_i2v,ltx_i2v,character_loras}}")
  worker_env+=(WORKER_ID_FILE="/workspace/.filmforge_worker_gpu${{idx}}.id")
  worker_env+=(MODEL_DOWNLOAD_TIMEOUT_SEC="7200")
  worker_env+=(COMFY_HEALTH_TIMEOUT_SEC="180")
  worker_env+=(COMFY_STOP_CMD="pkill -f 'main.py.*--port ${{comfy_port}}' || true")
  worker_env+=(COMFY_START_CMD="cd $COMFY_ROOT && CUDA_VISIBLE_DEVICES=${{idx}} $COMFY_PYTHON main.py --listen 127.0.0.1 --port ${{comfy_port}} --enable-cors-header --user-directory $COMFY_ROOT/user_gpu${{idx}} --database-url sqlite:///$COMFY_ROOT/user_gpu${{idx}}/comfyui.db --temp-directory $COMFY_ROOT/temp/gpu${{idx}} >> /tmp/comfyui_gpu${{idx}}.log 2>&1 &")
  worker_env+=(WORKER_MAX_CONCURRENT_JOBS="${{WORKER_MAX_CONCURRENT_JOBS:-10}}")
  worker_env+=(WORKER_HEARTBEAT_SECONDS="30")
  worker_env+=(RENDER_BROKER_HEARTBEAT_SEC="30")
  worker_env+=(TMPDIR="/tmp")
  test -n "$worker_public_url" && worker_env+=(WORKER_PUBLIC_URL="$worker_public_url")
  test -n "${{FILMFORGE_BACKEND_URL:-}}" && worker_env+=(FILMFORGE_BACKEND_URL="${{FILMFORGE_BACKEND_URL}}")
  test -n "${{WORKER_REGISTRATION_TOKEN:-}}" && worker_env+=(WORKER_REGISTRATION_TOKEN="${{WORKER_REGISTRATION_TOKEN}}")
  test -n "${{RENDER_BROKER_WORKER_TOKEN:-}}" && worker_env+=(RENDER_BROKER_WORKER_TOKEN="${{RENDER_BROKER_WORKER_TOKEN}}")
  test -n "${{WORKER_API_TOKEN:-}}" && worker_env+=(WORKER_API_TOKEN="${{WORKER_API_TOKEN}}")
  test -n "${{GPU_WORKER_API_TOKEN:-}}" && worker_env+=(GPU_WORKER_API_TOKEN="${{GPU_WORKER_API_TOKEN}}")
  test -n "${{WORKER_API_AUTH_MODE:-}}" && worker_env+=(WORKER_API_AUTH_MODE="${{WORKER_API_AUTH_MODE}}")
  test -n "${{WORKER_INPUT_URL_ALLOWED_HOSTS:-}}" && worker_env+=(WORKER_INPUT_URL_ALLOWED_HOSTS="${{WORKER_INPUT_URL_ALLOWED_HOSTS}}")
  echo "non-systemd worker startup is disabled for atomic secure deploys" >&2
  return 1
}}

for idx in $(seq 0 $((GPU_COUNT - 1))); do
  comfy_port=$((COMFY_PORT_BASE + idx))
  worker_port=$((WORKER_PORT_BASE + idx))
  comfy_user_dir="$COMFY_ROOT/user_gpu${{idx}}"
  worker_public_url="$(public_url_for_idx "$idx")"
  comfy_temp_dir="$COMFY_ROOT/temp/gpu${{idx}}"
  mkdir -p "$comfy_user_dir" "$comfy_temp_dir"

  cat > "/etc/systemd/system/comfyui-gpu${{idx}}.service" <<UNIT
[Unit]
Description=ComfyUI GPU ${{idx}}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$COMFY_ROOT
Environment=CUDA_VISIBLE_DEVICES=${{idx}}
ExecStart=$COMFY_PYTHON main.py --listen 127.0.0.1 --port ${{comfy_port}} --enable-cors-header --user-directory ${{comfy_user_dir}} --database-url sqlite:///${{comfy_user_dir}}/comfyui.db --temp-directory ${{comfy_temp_dir}}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

  worker_public_url="$(public_url_for_idx "$idx")"
  if test -z "$worker_public_url" && test -n "${{WORKER_PUBLIC_URL:-}}" && test "$idx" = "0"; then
    worker_public_url="$WORKER_PUBLIC_URL"
  fi
  worker_secret_env="$(write_worker_secret_env "$idx")"

  cat > "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service" <<UNIT
[Unit]
Description=FilmForge GPU Worker API GPU ${{idx}}
After=network-online.target comfyui-gpu${{idx}}.service
Wants=network-online.target comfyui-gpu${{idx}}.service

[Service]
Type=simple
WorkingDirectory=$WORKER_MODULE_DIR
Environment=PYTHONPATH=$WORKER_MODULE_DIR
EnvironmentFile=$worker_secret_env
Environment=CUDA_VISIBLE_DEVICES=${{idx}}
Environment=COMFY_BASE_URL=http://127.0.0.1:${{comfy_port}}
Environment=COMFY_OUTPUT_DIR=$COMFY_ROOT/output
Environment=COMFY_TEMP_DIR=$COMFY_ROOT/temp
Environment=COMFY_INPUT_DIR=$COMFY_ROOT/input
Environment=WORKER_HOST=127.0.0.1
Environment=WORKER_PORT=${{worker_port}}
Environment=WORKER_NAME=filmforge-vast-${{HOSTNAME:-instance}}-gpu${{idx}}
Environment=WORKER_PROVIDER=vast
Environment=WORKER_CODE_RELEASE_ID=${{WORKER_CODE_RELEASE_ID}}
Environment=PYTHONDONTWRITEBYTECODE=1
Environment="WORKER_GPU_NAME=${{GPU_NAME}}"
Environment=WORKER_VRAM_GB=${{VRAM_GB}}
Environment="WORKER_CAPABILITIES=${{WORKER_CAPABILITIES:-flux2_stills,wan_i2v,ltx_i2v,character_loras}}"
Environment=WORKER_ID_FILE=/workspace/.filmforge_worker_gpu${{idx}}.id
Environment=MODEL_DOWNLOAD_TIMEOUT_SEC=7200
Environment=COMFY_HEALTH_TIMEOUT_SEC=180
Environment="COMFY_STOP_CMD=systemctl stop comfyui-gpu${{idx}}.service"
Environment="COMFY_START_CMD=systemctl start comfyui-gpu${{idx}}.service"
Environment=WORKER_MAX_CONCURRENT_JOBS=${{WORKER_MAX_CONCURRENT_JOBS:-10}}
Environment=WORKER_HEARTBEAT_SECONDS=30
Environment=RENDER_BROKER_HEARTBEAT_SEC=30
UNIT

  if test -n "$worker_public_url"; then
    echo "Environment=WORKER_PUBLIC_URL=$worker_public_url" >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
  fi
  if test -n "${{FILMFORGE_BACKEND_URL:-}}"; then
    echo "Environment=FILMFORGE_BACKEND_URL=${{FILMFORGE_BACKEND_URL}}" >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
  fi
  if test -n "${{WORKER_INSTANCE_ID:-}}"; then
    echo "Environment=WORKER_INSTANCE_ID=${{WORKER_INSTANCE_ID}}" >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
  fi
  if test -n "${{WORKER_API_AUTH_MODE:-}}"; then
    echo "Environment=WORKER_API_AUTH_MODE=${{WORKER_API_AUTH_MODE}}" >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
  fi
  if test -n "${{WORKER_INPUT_URL_ALLOWED_HOSTS:-}}"; then
    echo "Environment=WORKER_INPUT_URL_ALLOWED_HOSTS=${{WORKER_INPUT_URL_ALLOWED_HOSTS}}" >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
  fi

  cat >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service" <<UNIT
ExecStart=$WORKER_SOURCE_ROOT/.venv/bin/python -m uvicorn gpu_worker.app:app --host 127.0.0.1 --port ${{worker_port}}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
done

if test "$HAS_SYSTEMD" = "1"; then
  systemctl daemon-reload
  for idx in $(seq 0 $((GPU_COUNT - 1))); do
    systemctl enable --now "comfyui-gpu${{idx}}.service"
  done
else
  for idx in $(seq 0 $((GPU_COUNT - 1))); do
    comfy_port=$((COMFY_PORT_BASE + idx))
    comfy_user_dir="$COMFY_ROOT/user_gpu${{idx}}"
    comfy_temp_dir="$COMFY_ROOT/temp/gpu${{idx}}"
    nohup env CUDA_VISIBLE_DEVICES="$idx" "$COMFY_PYTHON" "$COMFY_ROOT/main.py" \\
      --listen 127.0.0.1 --port "$comfy_port" --enable-cors-header \\
      --user-directory "$comfy_user_dir" --database-url "sqlite:///$comfy_user_dir/comfyui.db" --temp-directory "$comfy_temp_dir" \\
      >/tmp/comfyui_gpu${{idx}}.log 2>&1 </dev/null &
  done
fi

for idx in $(seq 0 $((GPU_COUNT - 1))); do
  port=$((COMFY_PORT_BASE + idx))
  for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:${{port}}/system_stats" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
done

if test "${{WORKER_SECURITY_CUTOVER_COMPLETE:-0}}" != "1"; then
  echo "WORKER_RELEASE_STAGED_ONLY=${{WORKER_CODE_RELEASE_ID}}"
  echo "Worker code/GPU stack staged; worker start waits for receipt-gated cutover." >&2
  exit 0
fi

if test "$HAS_SYSTEMD" = "1"; then
  for idx in $(seq 0 $((GPU_COUNT - 1))); do
    systemctl enable --now "filmforge-worker-gpu${{idx}}.service"
  done
else
  for idx in $(seq 0 $((GPU_COUNT - 1))); do
    start_worker_no_systemd "$idx" "$(public_url_for_idx "$idx")"
  done
fi

declare -a WORKER_URLS
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

  python3 - "/tmp/filmforge_worker_gpu${{idx}}_health.json" "$WORKER_CODE_RELEASE_ID" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    health = json.load(stream)
if health.get("code_release_id") != sys.argv[2]:
    raise SystemExit("worker health came from a different code release")
PY

  worker_url="$(public_url_for_idx "$idx")"
  if test -z "$worker_url" && test -x /opt/instance-tools/bin/cloudflared; then
    >/tmp/filmforge_gpu_worker_tunnel_gpu${{idx}}.log
    pkill -f "cloudflared tunnel --url http://127.0.0.1:${{port}}" || true
    nohup /opt/instance-tools/bin/cloudflared tunnel --url "http://127.0.0.1:${{port}}" --protocol http2 --no-autoupdate \\
      >/tmp/filmforge_gpu_worker_tunnel_gpu${{idx}}.log 2>&1 </dev/null &
    for _ in $(seq 1 30); do
      worker_url="$(grep -aEo 'https://[-a-z0-9]+\\.trycloudflare\\.com' /tmp/filmforge_gpu_worker_tunnel_gpu${{idx}}.log | tail -n 1 || true)"
      if test -n "$worker_url"; then
        if test "$HAS_SYSTEMD" = "1"; then
          systemctl set-environment WORKER_PUBLIC_URL_GPU${{idx}}="$worker_url" >/dev/null 2>&1 || true
          sed -i "/^Environment=WORKER_PUBLIC_URL=/d" "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
          sed -i "/^Environment=WORKER_ID_FILE=/i Environment=WORKER_PUBLIC_URL=$worker_url" "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
          systemctl daemon-reload
          systemctl restart "filmforge-worker-gpu${{idx}}.service"
        else
          start_worker_no_systemd "$idx" "$worker_url"
        fi
        break
      fi
      sleep 1
    done
  fi

  if test -n "$worker_url"; then
    WORKER_URLS+=("$worker_url")
    echo "WORKER_URL=$worker_url"
  fi
  echo "WORKER_HEALTH_GPU${{idx}}=$(cat /tmp/filmforge_worker_gpu${{idx}}_health.json)"
done

echo "WORKER_RELEASE_VERIFIED=${{WORKER_CODE_RELEASE_ID}}"

if test "${{#WORKER_URLS[@]}}" -gt 0; then
  (IFS=,; echo "WORKER_URLS=${{WORKER_URLS[*]}}")
fi
echo "GPU_COUNT=${{GPU_COUNT}}"
"""


def _upsert_env_line(content: str, key: str, value: str) -> str:
    line = f"{key}={value}"
    if f"{key}=" in content:
        return re.sub(rf"^{re.escape(key)}=.*$", line, content, flags=re.MULTILINE)
    suffix = "" if content.endswith("\n") else "\n"
    return f"{content}{suffix}{line}\n"


def update_env_file(
    env_path: Path,
    worker_url: str,
    semantic_url: str | None = None,
    qwen_url: str | None = None,
) -> None:
    content = env_path.read_text()
    content = _upsert_env_line(content, "GPU_WORKER_BASE_URL", worker_url)
    if semantic_url:
        content = _upsert_env_line(content, "SEMANTIC_SEARCH_URL", semantic_url)
    if qwen_url:
        content = _upsert_env_line(content, "QWEN_BASE_URL", qwen_url)
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


def extract_qwen_url(remote_output: str) -> str:
    for line in remote_output.splitlines():
        if line.startswith("QWEN_URL="):
            return line.split("=", 1)[1].strip()
    return ""


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


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value >= 1 else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return int(number) if number >= 1 else None
    if isinstance(value, (list, tuple, set)):
        return len(value) if value else None
    if isinstance(value, dict):
        return len(value) if value else None
    return None


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if float(value) > 0 else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return number if number > 0 else None
    return None


def _select_vast_offer(args: argparse.Namespace) -> dict:
    # An explicit offer id (from the rent UI's "pick this box") wins over the
    # search heuristics — rent the exact box the operator clicked.
    pinned_id = getattr(args, "vast_offer_id", None)
    if pinned_id:
        # The rent UI already picked a specific offer. Vast's `search offers` is
        # non-deterministic (identical queries return different offer sets), so
        # there's no way to re-fetch this id — but `create instance <id>` accepts
        # the id directly. Synthesize a minimal offer dict from what the caller
        # passed; gpu count is corrected post-SSH by _vast_remote_gpu_count.
        selected = {
            "id": int(pinned_id),
            "gpu_name": getattr(args, "vast_gpu", "") or "",
            "num_gpus": getattr(args, "vast_worker_count", 0) or 0,
            "dph_total": float(getattr(args, "vast_max_price", 0.0) or 0.0),
        }
        log(f"Using pinned Vast offer id={pinned_id} gpu={selected['gpu_name'] or 'unknown'}")
        return selected
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
    countries = [c.strip().upper() for c in getattr(args, "vast_country", None) or [] if c.strip()]
    if countries:
        query = f"{query} geolocation in [{','.join(countries)}]"
        log(f"Restricting Vast offers to regions: {', '.join(countries)}")
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
    requested_worker_count = int(getattr(args, "vast_worker_count", 0) or 0)
    if requested_worker_count > 1:
        multi_gpu_candidates = [
            offer for offer in candidates
            if _vast_offer_gpu_count(offer) >= requested_worker_count
        ]
        if multi_gpu_candidates:
            candidates = multi_gpu_candidates
        else:
            log(
                f"No Vast offers in the current result set reported "
                f"{requested_worker_count}+ GPUs; falling back to single-worker selection."
            )
    selected = sorted(candidates, key=_vast_preferred_offer_sort_key)[0]
    selected_gpu_count = _vast_offer_gpu_count(selected)
    selected_price = float(selected.get("dph_total") or selected.get("dph") or 0.0)
    selected_worker_count = max(1, min(selected_gpu_count, requested_worker_count or selected_gpu_count))
    per_worker_price = selected_price / selected_worker_count if selected_worker_count else selected_price
    log(
        "Selected Vast offer "
        f"id={selected.get('id')} gpu={selected.get('gpu_name')} "
        f"gpus={selected_gpu_count} "
        f"price=${selected_price:.3f}/hr "
        f"per_worker=${per_worker_price:.3f}/hr "
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
    """Never advertise Vast's cleartext public port as a worker API URL.

    The port may remain published for provider diagnostics, but render traffic
    carries a replayable bearer plus prompts/reference bytes and must use the
    TLS tunnel created by the remote bootstrap.
    """
    del instance_id, container_port, timeout_sec
    return None


def _vast_offer_gpu_count(offer: dict) -> int:
    """Best-effort GPU count from a Vast offer payload."""

    for key in (
        "num_gpus",
        "gpu_count",
        "num_gpus_total",
        "gpu_count_total",
        "total_gpus",
        "gpu_quantity",
        "gpus",
        "gpu_ids",
        "gpu_names",
    ):
        count = _positive_int(offer.get(key))
        if count:
            return count

    per_gpu_ram = _positive_float(offer.get("gpu_ram") or offer.get("gpu_memory"))
    total_gpu_ram = _positive_float(
        offer.get("gpu_total_ram")
        or offer.get("gpu_totalram")
        or offer.get("gpu_ram_total")
        or offer.get("total_gpu_ram")
        or offer.get("gpu_memory_total")
    )
    if per_gpu_ram and total_gpu_ram and total_gpu_ram >= per_gpu_ram:
        count = round(total_gpu_ram / per_gpu_ram)
        if count > 0:
            return count

    for key in ("gpu_name", "gpu_display_name", "bundle_id"):
        value = str(offer.get(key) or "")
        for pattern in (
            r"\b(\d+)\s*x\b",
            r"\bx\s*(\d+)\b",
            r"\b(\d+)\s+gpus?\b",
        ):
            match = re.search(pattern, value, flags=re.IGNORECASE)
            if match:
                return max(1, int(match.group(1)))

    return 0


def _vast_remote_gpu_count(args: argparse.Namespace) -> int | None:
    """Read the actual GPU count from the rented Vast instance over SSH."""

    try:
        ssh_cmd, _, _ = parse_ssh_command(args.ssh_command)
        ssh_cmd = add_default_identity(ssh_cmd, override=args.ssh_identity)
        ssh_cmd = add_default_host_key_policy(ssh_cmd)
        result = subprocess.run(
            [
                *ssh_cmd,
                "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l | tr -d ' ' || true",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        log(f"Could not query remote Vast GPU count: {exc}")
        return None

    count = _positive_int((result.stdout or "").strip().splitlines()[-1] if result.stdout else None)
    if count:
        return count
    if result.stderr.strip():
        log(f"Remote Vast GPU count query returned no count: {result.stderr.strip().splitlines()[-1]}")
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


class _RejectWorkerRedirectHandler(HTTPRedirectHandler):
    """Keep a deploy-time worker bearer on its validated origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _validated_worker_warmup_base_url(worker_url: str) -> str:
    raw_url = str(worker_url or "")
    if (
        not raw_url
        or raw_url != raw_url.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in raw_url)
    ):
        raise RuntimeError("Worker warmup URL is invalid")
    # Warmup carries the worker bearer, so its transport policy is always the
    # production policy even if the caller's ambient shell says test/dev.
    _validate_worker_public_url_env(
        [
            "WORKER_API_AUTH_MODE=required",
            f"WORKER_PUBLIC_URL={raw_url}",
        ]
    )
    return raw_url.rstrip("/")


def warm_remote_worker(
    worker_url: str,
    asset_groups: list[str],
    *,
    api_token: str | None = None,
    timeout_sec: int = 3600,
) -> dict:
    base_url = _validated_worker_warmup_base_url(worker_url)
    payload = json.dumps({"asset_groups": asset_groups}).encode()
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    request = Request(
        f"{base_url}/assets/ensure",
        data=payload,
        headers=headers,
        method="POST",
    )
    # Disable both environment proxies and redirects before attaching the
    # bearer. A hostile rented worker must not induce a second credentialed
    # request to another origin.
    opener = build_opener(ProxyHandler({}), _RejectWorkerRedirectHandler())
    try:
        with opener.open(request, timeout=timeout_sec) as response:
            status = int(getattr(response, "status", 200) or 200)
            body = response.read(_MAX_WORKER_WARMUP_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        if 300 <= int(exc.code) < 400:
            raise RuntimeError("Worker warmup redirect rejected") from None
        raise RuntimeError(f"Worker warmup request failed HTTP {exc.code}") from None
    except Exception:
        raise RuntimeError("Worker warmup request failed") from None
    if 300 <= status < 400:
        raise RuntimeError("Worker warmup redirect rejected")
    if not 200 <= status < 300:
        raise RuntimeError(f"Worker warmup request failed HTTP {status}")
    if len(body) > _MAX_WORKER_WARMUP_RESPONSE_BYTES:
        raise RuntimeError("Worker warmup response exceeded byte limit")
    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("Worker warmup returned invalid JSON") from None
    if not isinstance(result, dict):
        raise RuntimeError("Worker warmup returned invalid JSON")
    return result


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


def _verda_available_instance_types(args: argparse.Namespace) -> set[str]:
    command = ["availability", "--location", args.verda_location]
    if _verda_contract(args) == "spot":
        command.append("--spot")
    availability = _verda_json(args, *command, timeout=60)
    available_types: set[str] = set()
    if isinstance(availability, list):
        for row in availability:
            if isinstance(row, dict):
                available_types.update(str(item) for item in (row.get("instance_types") or []))
    return available_types


def _verda_check_gpu_availability(args: argparse.Namespace) -> None:
    log("Checking Verda GPU availability...")
    available_types = _verda_available_instance_types(args)
    if args.verda_instance_type not in available_types:
        types = ", ".join(sorted(available_types)) or "(none)"
        raise RuntimeError(
            f"Verda instance type {args.verda_instance_type!r} is not currently available in "
            f"{args.verda_location} for contract {_verda_contract(args)!r}. Available: {types}"
        )


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

    _verda_check_gpu_availability(args)


def _verda_fresh_preflight(args: argparse.Namespace) -> None:
    cli = args.verda_cli.expanduser()
    if not cli.exists():
        raise RuntimeError(f"Verda CLI not found at {cli}")

    log("Checking Verda auth...")
    _verda_check(args, "auth", "show", timeout=30)

    _verda_check_gpu_availability(args)


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


def _resume_verda_instance(
    args: argparse.Namespace,
    instance_id: str,
) -> tuple[str, str]:
    """Resolve one exact running VM for a later secure-deploy phase.

    The one-click state machine creates the paid VM only during ``stage-code``.
    ``provision-only`` and ``activate`` must resume that exact instance rather
    than performing another provider create.  Match every stable identity field
    the Verda inventory exposes and fail closed on ambiguity or drift.
    """

    expected_id = str(instance_id or "").strip()
    if not re.fullmatch(r"[0-9a-fA-F-]{16,64}", expected_id):
        raise RuntimeError("Verda resume instance id is invalid")
    cli = args.verda_cli.expanduser()
    if not cli.exists():
        raise RuntimeError(f"Verda CLI not found at {cli}")
    log("Checking Verda auth for exact-instance resume...")
    _verda_check(args, "auth", "show", timeout=30)
    instances = _verda_json(args, "vm", "list", timeout=60)
    if not isinstance(instances, list):
        raise RuntimeError("Unexpected Verda VM inventory response")
    matches = [
        row
        for row in instances
        if isinstance(row, dict) and str(row.get("id") or "") == expected_id
    ]
    if len(matches) != 1:
        raise RuntimeError("Exact Verda resume instance is absent or ambiguous")
    instance = matches[0]
    hostname = str(instance.get("hostname") or "").strip()
    status = str(instance.get("status") or "").strip().lower()
    ip = str(instance.get("ip") or "").strip()
    location = str(instance.get("location") or "").strip()
    instance_type = str(
        instance.get("instance_type")
        or instance.get("instanceType")
        or instance.get("type")
        or ""
    ).strip()
    if hostname != args.verda_hostname:
        raise RuntimeError("Exact Verda resume instance hostname drifted")
    if status != "running" or not ip:
        raise RuntimeError("Exact Verda resume instance is not running with an IP")
    if location and location != args.verda_location:
        raise RuntimeError("Exact Verda resume instance location drifted")
    if instance_type and instance_type != args.verda_instance_type:
        raise RuntimeError("Exact Verda resume instance type drifted")
    log(f"Resuming exact Verda instance id={expected_id} ip={ip}")
    return ip, expected_id


def _wait_for_verda_ssh(ip: str, identity: Path, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        result = subprocess.run(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "GlobalKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
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


# Departments a single GPU slot can serve on a multi-GPU box. One GPU = one
# department stays the law (the resident co-tenants starved WAN); a 4-GPU box
# just gets to run several departments side by side, one per card.
WORKER_PLAN_DEPARTMENTS = ("generation", "vision", "audio", "none")

# Vamsee's 4-GPU Verda layout (2026-07-26): two generation workers (FLUX/WAN
# share the ComfyUI process and evict between phases), one vision worker
# (resident vLLM/Qwen3-VL), one audio worker (resident Parler + SA3).
DEFAULT_4GPU_WORKER_PLAN = ("generation", "generation", "vision", "audio")


def parse_worker_plan(spec: str) -> list[str]:
    """Parse a --*-worker-plan spec ("generation,generation,vision,audio").

    Index in the list == GPU index. Returns [] for an empty spec (homogeneous
    box, every worker gets the same capabilities — the pre-plan behavior).
    """
    if not (spec or "").strip():
        return []
    plan = [token.strip().lower() for token in spec.split(",") if token.strip()]
    unknown = sorted(set(plan) - set(WORKER_PLAN_DEPARTMENTS))
    if unknown:
        raise RuntimeError(
            f"Unknown worker-plan department(s): {', '.join(unknown)}. "
            f"Valid: {', '.join(WORKER_PLAN_DEPARTMENTS)}"
        )
    if plan.count("vision") > 1 or plan.count("audio") > 1:
        raise RuntimeError(
            "At most one vision and one audio GPU per box: their servers are "
            "resident singletons bound to a fixed port, so a second copy would "
            "fight for the port and the same model cache."
        )
    return plan


def verda_rehydrate_script(
    *,
    public_ip: str,
    worker_port: int,
    comfy_port: int,
    worker_count: int,
    remote_root: str,
    patch_content: str = "",
    semantic_port: int = 8082,
    worker_plan: list[str] | None = None,
    vllm_port: int = 8100,
    worker_source_root: str | None = None,
) -> str:
    worker_source_root = worker_source_root or f"{DEFAULT_WORKER_RELEASES_ROOT}/current/gpu_worker"
    security_gate = worker_security_stage_gate_script()
    if patch_content:
        _rehydrate_patch_block = (
            "mkdir -p \"$COMFY_ROOT/custom_nodes/filmforge_cuda_patch\"\n"
            "cat > \"$COMFY_ROOT/custom_nodes/filmforge_cuda_patch/__init__.py\" << 'FILMFORGE_PATCH_EOF'\n"
            + patch_content.rstrip("\n")
            + "\nFILMFORGE_PATCH_EOF\n"
            "echo \"[verda] cuDNN patch custom node installed\" >&2"
        )
    else:
        _rehydrate_patch_block = (
            "echo \"[verda] WARNING: cuDNN patch not embedded — Blackwell SDP errors may occur\" >&2"
        )

    return f"""#!/usr/bin/env bash
set -euo pipefail

{security_gate}

PUBLIC_IP={shlex.quote(public_ip)}
WORKER_PORT_BASE={worker_port}
COMFY_PORT_BASE={comfy_port}
WORKER_COUNT_REQUESTED={worker_count}
WORKER_PLAN_SPEC={shlex.quote(",".join(worker_plan or []))}
VLLM_PORT={vllm_port}
REMOTE_ROOT={shlex.quote(remote_root)}
COMFY_ROOT="/workspace/ComfyUI"
WORKER_ROOT={shlex.quote(worker_source_root)}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; GPU drivers are not ready" >&2
  exit 1
fi

PHYSICAL_GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
GPU_COUNT="$PHYSICAL_GPU_COUNT"
# An explicit request may LOWER the worker count but must NEVER exceed the
# physical GPUs: each worker is pinned to CUDA_VISIBLE_DEVICES=<idx>, so a gpuN
# beyond the real GPU count crash-loops forever on "No CUDA GPUs are available".
if test "$WORKER_COUNT_REQUESTED" -gt 0 && test "$WORKER_COUNT_REQUESTED" -le "$PHYSICAL_GPU_COUNT"; then
  GPU_COUNT="$WORKER_COUNT_REQUESTED"
fi

# ── Per-GPU department plan ───────────────────────────────────────────────────
# A plan assigns each GPU index its own department ("generation,generation,
# vision,audio"). Empty spec = the old homogeneous box: every worker gets the
# same WORKER_CAPABILITIES. The plan sizes the box: a 4-entry plan on a 4-GPU
# machine wins over WORKER_COUNT_REQUESTED, and is truncated (never padded) to
# the physical GPU count so no worker is pinned to a card that doesn't exist.
WORKER_PLAN=()
if test -n "$WORKER_PLAN_SPEC"; then
  IFS=',' read -r -a WORKER_PLAN <<< "$WORKER_PLAN_SPEC"
  PLAN_COUNT="${{#WORKER_PLAN[@]}}"
  if test "$PLAN_COUNT" -gt "$PHYSICAL_GPU_COUNT"; then
    echo "[verda] WARNING: worker plan has $PLAN_COUNT entries but the box has $PHYSICAL_GPU_COUNT GPU(s); truncating." >&2
    PLAN_COUNT="$PHYSICAL_GPU_COUNT"
  fi
  GPU_COUNT="$PLAN_COUNT"
  echo "[verda] worker plan: ${{WORKER_PLAN[*]:0:$GPU_COUNT}}"
fi

dept_for_idx() {{
  if test "${{#WORKER_PLAN[@]}}" -eq 0; then
    echo "generation"
  else
    echo "${{WORKER_PLAN[$1]}}"
  fi
}}

# Capabilities a GPU advertises, by department. Generation honors an explicit
# WORKER_CAPABILITIES override (that's how the rent UI narrows a render box);
# vision/audio are fixed sets — their capability IS their department.
caps_for_dept() {{
  case "$1" in
    vision) echo "qwen_vision" ;;
    audio)  echo "tts_dialogue,stable_audio3" ;;
    *)      echo "${{WORKER_CAPABILITIES:-flux2_stills,wan_i2v,ltx_i2v,character_loras}}" ;;
  esac
}}

if test "$GPU_COUNT" -lt 1; then
  echo "[verda] ERROR: No GPUs detected by nvidia-smi" >&2
  exit 1
fi

# ── Stop resident department services this deploy did NOT ask for ─────────────
# filmforge-vllm / -parler / -sa3 are systemd units on the OS VOLUME, which is
# preserved and reattached on every deploy — so they come back at boot even when
# the new deploy has no vision/audio card. Observed twice on 2026-07-26: an 83 GB
# vLLM sat on gpu2 while that card's worker advertised wan_i2v, i.e. a guaranteed
# OOM on the next WAN job, and the drift guard cannot catch it (with no plan the
# guard is a no-op, and it only ever rewrites units the plan names).
#
# What this box should host is the union of the plan and the capability
# broadcast, so a legacy caps-only audio box keeps its servers. Runs BEFORE
# ComfyUI starts, so a reclaimed card is already free when the render process
# loads.
_wants_vision=""
_wants_audio=""
case ",${{WORKER_PLAN_SPEC}}," in *,vision,*) _wants_vision=1 ;; esac
case ",${{WORKER_PLAN_SPEC}}," in *,audio,*) _wants_audio=1 ;; esac
case ",${{WORKER_CAPABILITIES:-}}," in *,qwen_vision,*) _wants_vision=1 ;; esac
case ",${{WORKER_CAPABILITIES:-}}," in *,tts_dialogue,*|*,stable_audio3,*) _wants_audio=1 ;; esac

_stop_stale_resident() {{
  unit="$1"
  reason="$2"
  if systemctl is-active --quiet "$unit" 2>/dev/null; then
    echo "[verda] stopping $unit — $reason (it was holding VRAM on a card this deploy renders on)" >&2
    systemctl disable --now "$unit" >/dev/null 2>&1 || true
  fi
}}

if test -z "$_wants_vision"; then
  _stop_stale_resident filmforge-vllm "no vision card in this deploy"
fi
if test -z "$_wants_audio"; then
  _stop_stale_resident filmforge-parler "no audio card in this deploy"
  _stop_stale_resident filmforge-sa3 "no audio card in this deploy"
fi

# GPU firmware check — must happen before any CUDA call.
# If GPU Firmware shows N/A the kernel module loaded but the GSP firmware
# binary is absent. We try a one-shot apt repair (install firmware package +
# reload the module) before failing hard. Safe here: ComfyUI has not started,
# no GPU processes are running, so rmmod/modprobe is safe.
_try_repair_nvidia_firmware() {{
  local driver_version major fw_dir
  driver_version="$(modinfo nvidia 2>/dev/null | awk '/^version:/ {{print $2; exit}}')"
  test -n "$driver_version" || {{ echo "[verda]   Cannot determine driver version" >&2; return 1; }}
  major="${{driver_version%%.*}}"
  fw_dir="/lib/firmware/nvidia/$driver_version"
  echo "[verda]   Driver $driver_version — trying apt-get install nvidia-firmware-${{major}}-open ..." >&2
  apt-get install -y -q "nvidia-firmware-${{major}}-open" 2>&1 | tail -3 >&2 \
    || apt-get install -y -q "libnvidia-extra-${{major}}" 2>&1 | tail -3 >&2 \
    || true
  ls "$fw_dir"/gsp*.bin >/dev/null 2>&1 \
    || {{ echo "[verda]   No gsp*.bin in $fw_dir after apt — cannot repair" >&2; return 1; }}
  echo "[verda]   Firmware files present; reloading NVIDIA kernel module ..." >&2
  rmmod nvidia_drm 2>/dev/null || true
  rmmod nvidia_modeset 2>/dev/null || true
  rmmod nvidia_uvm 2>/dev/null || true
  rmmod nvidia 2>/dev/null \
    || {{ echo "[verda]   Cannot unload nvidia module (GPU may be in use)" >&2; return 1; }}
  sleep 2
  modprobe nvidia 2>/dev/null \
    || {{ echo "[verda]   modprobe nvidia failed after unload — driver unrecoverable" >&2; return 1; }}
  sleep 3
  modprobe nvidia_uvm 2>/dev/null || true
  modprobe nvidia_modeset 2>/dev/null || true
  modprobe nvidia_drm 2>/dev/null || true
}}

echo "[verda] Checking GPU firmware compatibility..." >&2
_fw_ok=1
for _gpu_dir in /proc/driver/nvidia/gpus/*/; do
  test -f "$_gpu_dir/information" || continue
  _fw="$(grep "^GPU Firmware:" "$_gpu_dir/information" 2>/dev/null | awk '{{print $3}}')"
  _model="$(grep "^Model:" "$_gpu_dir/information" 2>/dev/null | sed 's/^Model:[[:space:]]*//')"
  if test "${{_fw:-N/A}}" = "N/A"; then
    echo "[verda] ✗ GPU firmware missing: ${{_model:-unknown GPU}} — attempting repair..." >&2
    if _try_repair_nvidia_firmware; then
      _fw="$(grep "^GPU Firmware:" "$_gpu_dir/information" 2>/dev/null | awk '{{print $3}}')"
      if test "${{_fw:-N/A}}" != "N/A"; then
        echo "[verda] ✓ GPU firmware repaired: ${{_model:-unknown}} (fw=${{_fw}})" >&2
      else
        echo "[verda] ✗ Firmware repair did not resolve (still N/A): ${{_model:-unknown GPU}}" >&2
        echo "[verda]   The installed driver does not support this GPU architecture." >&2
        echo "[verda]   FIX: Use 1A100.22V, 1H100.80S.32V, or 1B200.30V." >&2
        _fw_ok=0
      fi
    else
      echo "[verda] ✗ Firmware repair failed: ${{_model:-unknown GPU}}" >&2
      echo "[verda]   FIX: Use 1A100.22V, 1H100.80S.32V, or 1B200.30V." >&2
      _fw_ok=0
    fi
  else
    echo "[verda] ✓ GPU firmware OK: ${{_model:-unknown}} (fw=${{_fw}})" >&2
  fi
done
if test "$_fw_ok" -eq 0; then
  exit 1
fi

# Confidential Computing check — CC PRODUCTION mode blocks cuInit for all
# normal CUDA applications (ComfyUI, PyTorch). Detect it early and fail
# with a clear message rather than burning retries. Instance types ending
# in ".CC" from Verda are always CC PRODUCTION mode; there is no in-VM toggle.
if nvidia-smi conf-compute -f 2>/dev/null | grep -q "CC status: ON"; then
  echo "[verda] ✗ GPU is in Confidential Computing PRODUCTION mode (CC status: ON)" >&2
  echo "[verda]   Normal CUDA (PyTorch, ComfyUI) cannot run in CC mode." >&2
  echo "[verda]   FIX: Use a non-CC instance type — 1A100.22V, 1H100.80S.32V, or 1B200.30V." >&2
  exit 1
fi

mkdir -p /mnt/data
if ! mountpoint -q /mnt/data; then
  VDB_FS="$(blkid -s TYPE -o value /dev/vdb 2>/dev/null || true)"
  if test "$VDB_FS" != "ext4"; then
    echo "[verda] ERROR: /dev/vdb is not an initialized ext4 data volume (detected: ${{VDB_FS:-none}})." >&2
    echo "[verda] The deploy preflight should bootstrap an incomplete volume pair before rehydration." >&2
    exit 1
  fi
  mount /dev/vdb /mnt/data
fi
VDB_UUID="$(blkid -s UUID -o value /dev/vdb 2>/dev/null || true)"

mkdir -p /mnt/data/ComfyUI/models /mnt/data/ComfyUI/input /mnt/data/ComfyUI/output /mnt/data/ComfyUI/temp
mkdir -p "$COMFY_ROOT"

root_avail_kb() {{
  df -Pk / | awk 'NR == 2 {{print $4}}'
}}

require_root_space_kb() {{
  required="$1"
  available="$(root_avail_kb)"
  if test "${{available:-0}}" -lt "$required"; then
    echo "Root filesystem has only ${{available:-0}} KiB free; need at least $required KiB" >&2
    df -h / /mnt/data >&2 || true
    exit 1
  fi
}}

cleanup_hidden_comfy_dir() {{
  src="$1"
  dst="$2"
  test -d "$src" || return 0
  test -d "$dst" || return 0
  mountpoint -q "$dst" || return 0

  root_view="/mnt/data/.filmforge_root_view"
  mkdir -p "$root_view"
  if ! mountpoint -q "$root_view"; then
    mount --bind / "$root_view"
  fi

  hidden="$root_view$dst"
  test -d "$hidden" || return 0
  if ! test -n "$(find "$hidden" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)"; then
    return 0
  fi

  missing=0
  while IFS= read -r hidden_file; do
    rel="$(printf '%s\n' "$hidden_file" | sed "s#^$hidden/##")"
    if ! test -e "$src/$rel"; then
      missing=1
      break
    fi
  done < <(find "$hidden" -type f -print 2>/dev/null)

  if test "$missing" -eq 0; then
    echo "Removing hidden OS-volume copy under $dst; active data lives in $src" >&2
    find "$hidden" -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +
  else
    echo "Found hidden files under $dst that are absent from $src; leaving them in place" >&2
  fi
}}

move_visible_comfy_dir_to_data() {{
  src="$1"
  dst="$2"
  test -d "$dst" || return 0
  mountpoint -q "$dst" && return 0
  if ! test -n "$(find "$dst" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)"; then
    return 0
  fi
  echo "Moving existing $dst contents to $src before bind mount" >&2
  mkdir -p "$src"
  cp -an "$dst"/. "$src"/
  find "$dst" -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +
}}

bind_comfy_dir() {{
  src="$1"
  dst="$2"
  mkdir -p "$src"
  if mountpoint -q "$dst"; then
    cleanup_hidden_comfy_dir "$src" "$dst"
    return
  fi
  if test -L "$dst"; then
    rm "$dst"
  elif test -e "$dst" && ! test -d "$dst"; then
    rm -rf "$dst"
  fi
  move_visible_comfy_dir_to_data "$src" "$dst"
  mkdir -p "$dst"
  mount --bind "$src" "$dst"
}}

for dir in models input output temp; do
  bind_comfy_dir "/mnt/data/ComfyUI/$dir" "$COMFY_ROOT/$dir"
done

# Belt-and-suspenders: ComfyUI must see the models that live on the data volume.
# The bind above can be silently lost (e.g. ComfyUI recreates models/ on the OS
# disk after boot), which leaves an empty /workspace/ComfyUI/models and triggers a
# full re-download even though /mnt/data is 69% full. So (1) fail loud if the
# models bind did not take, and (2) ALWAYS write extra_model_paths.yaml pointing
# directly at /mnt/data so ComfyUI finds the models regardless of the bind state.
if ! mountpoint -q "$COMFY_ROOT/models"; then
  echo "WARNING: $COMFY_ROOT/models is not bind-mounted from /mnt/data/ComfyUI/models" >&2
  echo "         Relying on extra_model_paths.yaml below to point ComfyUI at /mnt/data." >&2
  mount | grep -iE "comfy|mnt/data" >&2 || true
else
  echo "$COMFY_ROOT/models bind-mounted from /mnt/data/ComfyUI/models (used: $(df -h /mnt/data | awk 'NR==2{{print $3}}'))" >&2
fi

cat > "$COMFY_ROOT/extra_model_paths.yaml" <<'YAML'
filmforge_worker:
    base_path: /mnt/data/ComfyUI/models
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
echo "Wrote extra_model_paths.yaml → ComfyUI scans /mnt/data/ComfyUI/models directly" >&2

require_root_space_kb 65536

if test -n "$VDB_UUID" && ! grep -q "$VDB_UUID" /etc/fstab; then
  echo "UUID=$VDB_UUID /mnt/data ext4 defaults,nofail 0 2" >> /etc/fstab
fi

if ! test -f "$COMFY_ROOT/main.py"; then
  echo "ComfyUI not found at $COMFY_ROOT/main.py; OS volume is not worker-ready" >&2
  exit 1
fi
if ! test -x "$COMFY_ROOT/.venv/bin/python"; then
  echo "ComfyUI venv not found at $COMFY_ROOT/.venv/bin/python" >&2
  exit 1
fi

# Enable persistence mode so the NVIDIA driver stays warm between calls.
# Without this, Blackwell GPUs can return cudaErrorSystemNotReady (802) on
# the first CUDA call because compute initialization is still in progress.
nvidia-smi -pm 1 >/dev/null 2>&1 || true

# Bidirectional CUDA wheel repair: the OS volume persists torch/torchvision
# across rehydrates. If the last deploy was on a different CUDA major (e.g.
# CUDA 13 B200 → CUDA 12 A100 or vice versa), the C++ extensions fail to
# load and ComfyUI crashes with "operator torchvision::nms does not exist".
# Detect the mismatch and repair in both directions.
_cuda_driver_major="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+' | grep -oE '[0-9]+$' | head -1 || echo '')"
_torch_cuda_ver="$("$COMFY_ROOT/.venv/bin/python" -c \
  'import torch; v=getattr(torch.version,"cuda","") or ""; print(v.split(".")[0] if v else "")' \
  2>/dev/null || echo '')"
_torch_cuda_ver="$(echo "$_torch_cuda_ver" | tr -d '[:space:]')"
_cuda_driver_major="$(echo "$_cuda_driver_major" | tr -d '[:space:]')"

case "$_cuda_driver_major" in
  13) _pytorch_index="https://download.pytorch.org/whl/cu130" ;;
  12) _pytorch_index="https://download.pytorch.org/whl/cu128" ;;
  *)  _pytorch_index="" ;;
esac

if test "$_cuda_driver_major" = "13" && test "$_torch_cuda_ver" != "13"; then
  echo "[verda] CUDA mismatch: driver=13, torch_cuda=$_torch_cuda_ver — repairing to cu130..." >&2
  "$COMFY_ROOT/.venv/bin/python" -m pip install --force-reinstall --no-deps \
    --index-url "$_pytorch_index" '{_TORCH_PIN}' '{_TORCHVISION_PIN}' '{_TORCHAUDIO_PIN}'
elif test "$_cuda_driver_major" = "12" && test "$_torch_cuda_ver" = "13"; then
  echo "[verda] CUDA mismatch: driver=12, torch_cuda=$_torch_cuda_ver — repairing to cu128..." >&2
  "$COMFY_ROOT/.venv/bin/python" -m pip install --force-reinstall --no-deps \
    --index-url "$_pytorch_index" '{_TORCH_PIN}' '{_TORCHVISION_PIN}' '{_TORCHAUDIO_PIN}'
elif test -n "$_pytorch_index"; then
  # Even when the torch CUDA major matches, torchvision/torchaudio may have
  # been upgraded to a different CUDA build (e.g. cu130 while torch is cu128).
  # Check each package's CUDA suffix and re-pin if mismatched.
  _tv_ver="$("$COMFY_ROOT/.venv/bin/pip" show torchvision 2>/dev/null | grep '^Version:' | awk '{{print $2}}')"
  _ta_ver="$("$COMFY_ROOT/.venv/bin/pip" show torchaudio 2>/dev/null | grep '^Version:' | awk '{{print $2}}')"
  _need_repair=0
  echo "$_tv_ver" | grep -qE "cu${{_torch_cuda_ver}}" || _need_repair=1
  echo "$_ta_ver" | grep -qE "cu${{_torch_cuda_ver}}" || _need_repair=1
  if test "$_need_repair" = "1"; then
    echo "[verda] torchvision=$_tv_ver torchaudio=$_ta_ver mismatched with torch cuda=$_torch_cuda_ver — reinstalling from $_pytorch_index..." >&2
    "$COMFY_ROOT/.venv/bin/python" -m pip install --force-reinstall --no-deps \
      --index-url "$_pytorch_index" torchvision torchaudio
  else
    echo "[verda] CUDA OK: driver=$_cuda_driver_major torch_cuda=$_torch_cuda_ver torchvision=$_tv_ver torchaudio=$_ta_ver" >&2
  fi
else
  echo "[verda] CUDA OK: driver=$_cuda_driver_major torch_cuda=$_torch_cuda_ver (no index available for repair)" >&2
fi

# NVSwitch boxes (A100/H100 SXM) require Fabric Manager before CUDA can create
# a context. Verda occasionally starts it before the assigned GPUs are online;
# one bounded restart repairs that race. A restart that fails or hangs indicates
# a provider-side NVLink/NVSwitch fault, so keep the volume pair and replace the
# VM instead of registering workers that will fail every job with CUDA error 802.
if test -e /dev/nvidia-nvswitchctl \
   && ! "$COMFY_ROOT/.venv/bin/python" -c \
        "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
        >/dev/null 2>&1; then
  echo "[verda] CUDA unavailable on NVSwitch box; restarting NVIDIA Fabric Manager once..." >&2
  systemctl reset-failed nvidia-fabricmanager.service 2>/dev/null || true
  if ! timeout 180 systemctl restart nvidia-fabricmanager.service; then
    systemctl kill --kill-whom=all nvidia-fabricmanager.service 2>/dev/null || true
    echo "[verda] ERROR: NVIDIA Fabric Manager could not initialize the NVSwitch topology." >&2
    echo "[verda] This VM has a provider-side GPU fabric fault; preserve the volumes and replace the VM." >&2
    journalctl -u nvidia-fabricmanager.service -n 30 --no-pager >&2 || true
    exit 1
  fi
fi

# GPUs can be slow to enter compute-ready state after module/fabric init.
# Retry up to 5× (50s max) before treating unavailability as fatal.
for _cuda_try in 1 2 3 4 5; do
  if "$COMFY_ROOT/.venv/bin/python" -c \
       "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
     >/dev/null 2>&1; then
    break
  fi
  if test "$_cuda_try" -lt 5; then
    echo "[verda] CUDA not ready (attempt $_cuda_try/5) — waiting 10s for compute init..." >&2
    sleep 10
  fi
done

if ! "$COMFY_ROOT/.venv/bin/python" - <<'PY'
import torch
print("ComfyUI torch=" + torch.__version__ + " cuda=" + str(torch.version.cuda) + " available=" + str(torch.cuda.is_available()))
if not torch.cuda.is_available():
    raise SystemExit("ComfyUI PyTorch cannot initialize CUDA")
print("ComfyUI CUDA device=" + torch.cuda.get_device_name(0))
PY
then
  if test -e /dev/nvidia-nvswitchctl; then
    echo "[verda] NVIDIA Fabric Manager state: $(systemctl is-active nvidia-fabricmanager.service 2>/dev/null || true)" >&2
    journalctl -u nvidia-fabricmanager.service -n 30 --no-pager >&2 || true
  fi
  echo "ComfyUI PyTorch CUDA validation failed; refusing to register an unusable worker" >&2
  exit 1
fi

WORKER_MODULE_DIR="$(dirname "$WORKER_ROOT")"
if ! test -f "$WORKER_ROOT/app.py"; then
  echo "immutable gpu_worker release is missing at $WORKER_ROOT" >&2
  exit 1
fi
if ! test -x "$WORKER_ROOT/.venv/bin/python"; then
  echo "gpu_worker venv not found at $WORKER_ROOT/.venv/bin/python" >&2
  exit 1
fi

# Query GPU 0 only (-i 0) instead of piping all GPUs to `head -1`: on a multi-GPU
# box, `nvidia-smi | head -1` races — head closes after one line and SIGPIPEs
# nvidia-smi, which under `set -o pipefail` returns 141 and aborts the deploy.
# This is why the deploy worked on 1-2 GPU boxes but failed on 8.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | xargs)"
VRAM_GB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | awk '{{printf "%.0f", $1 / 1024}}')"
PUBLIC_URLS="${{WORKER_PUBLIC_URLS:-${{WORKER_PUBLIC_URL:-}}}}"

public_url_for_idx() {{
  idx="$1"
  python3 - "$PUBLIC_URLS" "$idx" <<'PY'
import sys
urls = [u.strip() for u in (sys.argv[1] or "").split(",")]
idx = int(sys.argv[2])
print(urls[idx] if idx < len(urls) else "")
PY
}}

write_worker_secret_env() {{
  idx="$1"
  secret_dir="/etc/filmforge"
  secret_path="$secret_dir/worker-gpu${{idx}}.env"
  mkdir -p "$secret_dir"
  chmod 0700 "$secret_dir"
  secret_tmp="$(mktemp "$secret_dir/.worker-gpu${{idx}}.XXXXXX")"
  chmod 0600 "$secret_tmp"
  for key in WORKER_REGISTRATION_TOKEN RENDER_BROKER_WORKER_TOKEN WORKER_API_TOKEN GPU_WORKER_API_TOKEN; do
    value="$(printenv "$key" 2>/dev/null || true)"
    test -n "$value" || continue
    case "$value" in
      *[[:space:]]*) echo "worker credential contains unsupported whitespace" >&2; rm -f "$secret_tmp"; exit 1 ;;
    esac
    printf '%s=%s\n' "$key" "$value" >> "$secret_tmp"
  done
  install -m 0600 "$secret_tmp" "$secret_path"
  rm -f "$secret_tmp"
  printf '%s\n' "$secret_path"
}}

# Choose ComfyUI VRAM flag based on detected VRAM.
# Total model footprint (Flux dev ~24 GB + WAN21 ~14 GB + encoders ~7 GB) is ~45 GB.
# On cards with >= 64 GB we keep all models resident (--highvram) to eliminate the
# per-job staging overhead (~15 s / job) that ComfyUI's default dynamic-VRAM mode
# incurs. Below 64 GB let ComfyUI manage dynamically to avoid OOM on tighter cards.
if [ "${{VRAM_GB}}" -ge 64 ] 2>/dev/null; then
  COMFY_VRAM_FLAG="--highvram"
else
  COMFY_VRAM_FLAG=""
fi
echo "VRAM: ${{VRAM_GB}} GB → ComfyUI flag: ${{COMFY_VRAM_FLAG:-'(none, dynamic)'}}"

# Install FilmForge cuDNN patch as a ComfyUI custom node (before services start).
# Disables cuDNN SDP backend to prevent "No valid execution plans built" on
# Blackwell (B300/SM 10.x) and future architectures where cuDNN has no plan.
{_rehydrate_patch_block}

for idx in $(seq 0 $((GPU_COUNT - 1))); do
  dept="$(dept_for_idx "$idx")"
  comfy_port=$((COMFY_PORT_BASE + idx))
  worker_port=$((WORKER_PORT_BASE + idx))
  comfy_user_dir="$COMFY_ROOT/user_gpu${{idx}}"
  worker_public_url="$(public_url_for_idx "$idx")"

  if test "$dept" = "none"; then
    echo "[verda] gpu${{idx}}: department=none — no worker started"
    continue
  fi
  if test -z "$worker_public_url"; then
    echo "[verda] gpu${{idx}}: TLS WORKER_PUBLIC_URLS entry is required" >&2
    exit 1
  fi
  case "$worker_public_url" in
    https://*) ;;
    *)
      case "${{WORKER_API_AUTH_MODE:-required}}" in
        development|test) ;;
        *) echo "[verda] gpu${{idx}}: public worker URL must use HTTPS" >&2; exit 1 ;;
      esac
      ;;
  esac

  # ComfyUI is the generation runtime only. A vision/audio GPU runs a resident
  # server instead (vLLM / Parler+SA3), and a second ComfyUI process on that card
  # would just hold VRAM the resident model needs — the co-tenancy that caused
  # the WAN OOMs in the first place.
  if test "$dept" = "generation"; then
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
ExecStart=$COMFY_ROOT/.venv/bin/python main.py --listen 127.0.0.1 --port ${{comfy_port}} --enable-cors-header --use-pytorch-cross-attention ${{COMFY_VRAM_FLAG}} --user-directory ${{comfy_user_dir}} --database-url sqlite:///${{comfy_user_dir}}/comfyui.db --temp-directory $COMFY_ROOT/temp/gpu${{idx}}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
    unit_after="network-online.target comfyui-gpu${{idx}}.service"
  else
    unit_after="network-online.target"
  fi
  worker_secret_env="$(write_worker_secret_env "$idx")"

  cat > "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service" <<UNIT
[Unit]
Description=FilmForge GPU Worker API GPU ${{idx}} (${{dept}})
After=${{unit_after}}
Wants=${{unit_after}}

[Service]
Type=simple
WorkingDirectory=$WORKER_MODULE_DIR
EnvironmentFile=$worker_secret_env
Environment=CUDA_VISIBLE_DEVICES=${{idx}}
Environment=WORKER_HOST=127.0.0.1
Environment=WORKER_PORT=${{worker_port}}
Environment=WORKER_NAME=filmforge-verda-${{PUBLIC_IP}}-gpu${{idx}}-${{dept}}
Environment=WORKER_PROVIDER=verda
Environment=WORKER_CODE_RELEASE_ID=${{WORKER_CODE_RELEASE_ID}}
Environment=PYTHONDONTWRITEBYTECODE=1
Environment="WORKER_GPU_NAME=${{GPU_NAME}}"
Environment=WORKER_VRAM_GB=${{VRAM_GB}}
Environment="WORKER_CAPABILITIES=$(caps_for_dept "$dept")"
Environment="WORKER_PUBLIC_URL=${{worker_public_url}}"
Environment=WORKER_ID_FILE=/workspace/.filmforge_worker_gpu${{idx}}.id
Environment=MODEL_DOWNLOAD_TIMEOUT_SEC=7200
Environment=WORKER_MAX_CONCURRENT_JOBS=${{WORKER_MAX_CONCURRENT_JOBS:-10}}
Environment=WORKER_HEARTBEAT_SECONDS=30
Environment=RENDER_BROKER_HEARTBEAT_SEC=30
UNIT

  if test "$dept" = "generation"; then
    cat >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service" <<UNIT
Environment=COMFY_BASE_URL=http://127.0.0.1:${{comfy_port}}
Environment=COMFY_OUTPUT_DIR=$COMFY_ROOT/output
Environment=COMFY_TEMP_DIR=$COMFY_ROOT/temp
Environment=COMFY_INPUT_DIR=$COMFY_ROOT/input
Environment=COMFY_HEALTH_TIMEOUT_SEC=180
Environment="COMFY_STOP_CMD=systemctl stop comfyui-gpu${{idx}}.service"
Environment="COMFY_START_CMD=systemctl start comfyui-gpu${{idx}}.service"
UNIT
  fi

  # Vision worker: registration metadata carries the vLLM base URL, which is how
  # the backend's GET /api/render-broker/vision-worker hands it to the LLM
  # gateway (no .env edit, no backend restart). Verda boxes have a public IP, so
  # this is the direct address — no cloudflared hop like the Vast/SkyPilot yaml.
  if test "$dept" = "vision"; then
    echo "Environment=WORKER_VISION_BASE_URL=http://${{PUBLIC_IP}}:${{VLLM_PORT}}/v1" \
      >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
  fi

  # Non-generation workers get a deliberately dead ComfyUI address: the default
  # (:8188) is gpu0's ComfyUI, so leaving it unset would make an audio/vision
  # worker report comfy_reachable=true and — if a render ever slipped past the
  # capability gate — execute it on somebody else's GPU.
  if test "$dept" != "generation"; then
    echo "Environment=COMFY_BASE_URL=http://127.0.0.1:1" \
      >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
  fi

  if test -n "${{FILMFORGE_BACKEND_URL:-}}"; then
    echo "Environment=FILMFORGE_BACKEND_URL=${{FILMFORGE_BACKEND_URL}}" >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
  fi
  if test -n "${{WORKER_INSTANCE_ID:-}}"; then
    echo "Environment=WORKER_INSTANCE_ID=${{WORKER_INSTANCE_ID}}" >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
  fi
  if test -n "${{WORKER_API_AUTH_MODE:-}}"; then
    echo "Environment=WORKER_API_AUTH_MODE=${{WORKER_API_AUTH_MODE}}" >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
  fi
  if test -n "${{WORKER_INPUT_URL_ALLOWED_HOSTS:-}}"; then
    echo "Environment=WORKER_INPUT_URL_ALLOWED_HOSTS=${{WORKER_INPUT_URL_ALLOWED_HOSTS}}" >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
  fi

  cat >> "/etc/systemd/system/filmforge-worker-gpu${{idx}}.service" <<UNIT
ExecStart=$WORKER_ROOT/.venv/bin/python -m uvicorn gpu_worker.app:app --host 127.0.0.1 --port ${{worker_port}}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
done

systemctl daemon-reload
for idx in $(seq 0 $((GPU_COUNT - 1))); do
  test "$(dept_for_idx "$idx")" = "generation" || continue
  systemctl enable --now "comfyui-gpu${{idx}}.service"
done

wait_comfy_healthy() {{
  for idx in $(seq 0 $((GPU_COUNT - 1))); do
    test "$(dept_for_idx "$idx")" = "generation" || continue
    port=$((COMFY_PORT_BASE + idx))
    stats_file="/tmp/comfyui_gpu${{idx}}_stats.json"
    rm -f "$stats_file"
    for _ in $(seq 1 60); do
      if curl -fsS "http://127.0.0.1:${{port}}/system_stats" >"$stats_file" 2>/dev/null; then
        break
      fi
      sleep 2
    done
    if ! test -s "$stats_file"; then
      echo "ComfyUI gpu${{idx}} failed system_stats health check" >&2
      journalctl -u "comfyui-gpu${{idx}}.service" -n 120 --no-pager >&2 || true
      exit 1
    fi
  done
}}
wait_comfy_healthy

# ── InfiniteTalk Comfy custom-node provisioner ────────────────────────────────
# This is GPU-stack work and must run before the secure provision-only exit.
# The cutover pass returns through the security stage gate at the top of this
# script, so anything after the exit is dead code on the one-click rent path.
# Comfy discovers custom nodes only at start: restart generation instances and
# prove them healthy again before releasing the staged receipt.
_infinitetalk_wanted=""
_infinitetalk_asset_group="infinitetalk_v1"
_infinitetalk_require_two_person="0"
case ",${{WORKER_CAPABILITIES:-}}," in
  *,infinitetalk_two_person_v1,*)
    _infinitetalk_wanted=1
    _infinitetalk_asset_group="infinitetalk_two_person_v1"
    _infinitetalk_require_two_person="1"
    ;;
  *,infinitetalk,*|*,infinitetalk_v1,*|*,talking_shot,*|*,multitalk,*)
    _infinitetalk_wanted=1
    ;;
esac
if test -n "$_infinitetalk_wanted"; then
  echo "[infinitetalk] provisioning Comfy talking-shot runtime"
  cd "$WORKER_ROOT"
  bash provision_infinitetalk.sh
  for idx in $(seq 0 $((GPU_COUNT - 1))); do
    dept="$(dept_for_idx "$idx")"
    test "$dept" = "generation" || continue
    echo "[infinitetalk] restarting ComfyUI on generation gpu$idx after provisioning"
    systemctl restart "comfyui-gpu${{idx}}.service"
  done
  wait_comfy_healthy

  # Secure one-click confirms the advertised capability immediately after the
  # staged receipt. Download the exact canonical group now, then prove the
  # restarted Comfy runtime can actually load its nodes and audio dependencies.
  # A fresh box therefore fails closed before broker registration/cutover rather
  # than advertising a capability that is still warming in the background.
  _infinitetalk_comfy_port=""
  for idx in $(seq 0 $((GPU_COUNT - 1))); do
    dept="$(dept_for_idx "$idx")"
    test "$dept" = "generation" || continue
    _infinitetalk_comfy_port=$((COMFY_PORT_BASE + idx))
    break
  done
  if test -z "$_infinitetalk_comfy_port"; then
    echo "InfiniteTalk requires at least one generation worker" >&2
    exit 1
  fi
  echo "[infinitetalk] materializing exact talking-shot asset group before cutover"
  cd "$WORKER_MODULE_DIR"
  COMFY_BASE_URL="http://127.0.0.1:${{_infinitetalk_comfy_port}}" \
  COMFY_DIR="$COMFY_ROOT" \
  INFINITETALK_PREFLIGHT_GROUP="$_infinitetalk_asset_group" \
  INFINITETALK_REQUIRE_TWO_PERSON="$_infinitetalk_require_two_person" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$WORKER_MODULE_DIR" \
    "$WORKER_ROOT/.venv/bin/python" - <<'PY'
import os

from gpu_worker.asset_manager import ensure_asset_group
from gpu_worker.infinitetalk import check_infinitetalk_readiness

asset_group = os.environ["INFINITETALK_PREFLIGHT_GROUP"]
require_two_person = os.environ["INFINITETALK_REQUIRE_TWO_PERSON"] == "1"
result = ensure_asset_group(asset_group)
readiness = check_infinitetalk_readiness(require_two_person=require_two_person)
if not readiness.ready:
    raise SystemExit(
        "InfiniteTalk readiness failed before secure cutover: "
        + str(readiness.as_dict())
    )
print(
    "[infinitetalk] exact asset group " + asset_group
    + " ready before secure cutover; downloaded="
    + str(len(result.downloaded_assets))
)
PY
fi

# Provision-only code executes as root, which can bypass the immutable
# candidate's read-only directory modes.  Prove that no runtime import or
# provisioner dirtied the candidate before emitting the receipt cutover trusts.
if ! python3 - "$WORKER_MODULE_DIR" "$WORKER_CODE_RELEASE_ID" <<'PY'
import hashlib
import itertools
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
release_id = sys.argv[2]
problems = []

ready = root / ".ready"
source_marker = root / ".source-sha256"
dependency_freeze = root / ".dependency-freeze.txt"
dependency_marker = root / ".dependency-freeze.sha256"

def regular_file(path):
    return not path.is_symlink() and path.is_file()

try:
    ready_mode = stat.S_IMODE(ready.lstat().st_mode)
except OSError:
    ready_mode = None
if not regular_file(ready) or ready_mode != 0o444:
    problems.append("readiness marker missing or mode drifted")

source_digest = source_marker.read_text().strip() if regular_file(source_marker) else ""
if not source_digest or release_id != "sha256-" + source_digest[:24]:
    problems.append("source marker does not match release id")
elif regular_file(ready) and ready.read_text().strip() != source_digest:
    problems.append("readiness marker does not match source digest")

if regular_file(dependency_freeze) and regular_file(dependency_marker):
    dependency_digest = hashlib.sha256(dependency_freeze.read_bytes()).hexdigest()
    recorded_dependency_digest = dependency_marker.read_text().strip()
else:
    dependency_digest = ""
    recorded_dependency_digest = ""
if not dependency_digest or recorded_dependency_digest != dependency_digest:
    problems.append("dependency snapshot missing or drifted")

writable = next(
    (
        path
        for path in itertools.chain((root,), root.rglob("*"))
        if not path.is_symlink()
        and stat.S_IMODE(path.lstat().st_mode) & 0o222
    ),
    None,
)
if writable is not None:
    problems.append("writable path present: " + str(writable.relative_to(root) or "."))

if problems:
    raise SystemExit("immutable worker candidate failed pre-cutover seal: " + "; ".join(problems))
PY
then
  echo "worker release candidate is incomplete; refusing staged receipt" >&2
  exit 1
fi

if test "${{WORKER_SECURITY_CUTOVER_COMPLETE:-0}}" != "1"; then
  echo "WORKER_RELEASE_STAGED_ONLY=${{WORKER_CODE_RELEASE_ID}}"
  echo "Worker code/GPU stack staged; worker start waits for receipt-gated cutover." >&2
  exit 0
fi

for idx in $(seq 0 $((GPU_COUNT - 1))); do
  test "$(dept_for_idx "$idx")" != "none" || continue
  systemctl enable --now "filmforge-worker-gpu${{idx}}.service"
done

for idx in $(seq 0 $((GPU_COUNT - 1))); do
  test "$(dept_for_idx "$idx")" != "none" || continue
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
  python3 - "/tmp/filmforge_worker_gpu${{idx}}_health.json" "$WORKER_CODE_RELEASE_ID" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    health = json.load(stream)
if health.get("code_release_id") != sys.argv[2]:
    raise SystemExit("worker health came from a different code release")
PY
  echo "WORKER_URL=$(public_url_for_idx "$idx")"
  echo "WORKER_HEALTH_GPU${{idx}}=$(cat /tmp/filmforge_worker_gpu${{idx}}_health.json)"
done

echo "WORKER_RELEASE_VERIFIED=${{WORKER_CODE_RELEASE_ID}}"

echo "GPU_COUNT=${{GPU_COUNT}}"
df -h /mnt/data

# Which GPU (if any) the plan assigned to each resident department. Empty when
# there is no plan — then the legacy WORKER_CAPABILITIES check below decides.
VISION_GPU_IDX=""
AUDIO_GPU_IDX=""
for idx in $(seq 0 $((GPU_COUNT - 1))); do
  case "$(dept_for_idx "$idx")" in
    vision) VISION_GPU_IDX="$idx" ;;
    audio)  AUDIO_GPU_IDX="$idx" ;;
  esac
done

# ── Vision department (plan: a GPU assigned "vision") ─────────────────────────
# Resident vLLM serving Qwen3-VL, pinned to its own card. The venv and the HF
# cache live on /mnt/data so a rehydrate of the cached volume pair skips both the
# ~2min vllm install and the ~17GB weight download. Guarded subshell: a vision
# failure must never kill a render deploy — the LLM gateway falls back to OpenAI.
(
set +e
if test -n "$VISION_GPU_IDX"; then
  echo "[vision] gpu${{VISION_GPU_IDX}} → vLLM ${{QWEN_VISION_MODEL:-Qwen/Qwen3-VL-8B-Instruct}} on :${{VLLM_PORT}}"
  VLLM_VENV=/mnt/data/vllm_venv
  if ! test -x "$VLLM_VENV/bin/vllm"; then
    python3 -m venv "$VLLM_VENV"
    "$VLLM_VENV/bin/pip" install -q --upgrade pip
    # 0.11.x = the Qwen3-VL support line incl. video input; pinned so a box
    # rebuilt months later serves the same stack (matches ff_worker_vision_vast.yaml).
    "$VLLM_VENV/bin/pip" install -q "vllm==0.11.2" || echo "[vision] WARN: vllm install failed" >&2
  fi
  if test -x "$VLLM_VENV/bin/vllm"; then
    cat > /etc/systemd/system/filmforge-vllm.service <<UNIT
[Unit]
Description=FilmForge resident vLLM vision server (Qwen3-VL)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=CUDA_VISIBLE_DEVICES=${{VISION_GPU_IDX}}
Environment=HF_HOME=/mnt/data/hf_cache
Environment=HF_HUB_ENABLE_HF_TRANSFER=0
ExecStart=$VLLM_VENV/bin/vllm serve ${{QWEN_VISION_MODEL:-Qwen/Qwen3-VL-8B-Instruct}} --served-model-name ${{QWEN_VISION_MODEL:-Qwen/Qwen3-VL-8B-Instruct}} --host 0.0.0.0 --port ${{VLLM_PORT}} --trust-remote-code --max-model-len 16384 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{{"image":4,"video":1}}'
Restart=always
RestartSec=15

[Install]
WantedBy=multi-user.target
UNIT
    systemctl daemon-reload
    systemctl enable -q filmforge-vllm
    systemctl restart filmforge-vllm
    # Bounded wait: a warm volume is up in ~60s, a first-ever boot downloads
    # ~17GB. Don't hold the deploy hostage for that — the worker already
    # advertises the URL and the gateway falls back to OpenAI until it answers.
    VLLM_UP=""
    for _ in $(seq 1 36); do
      if curl -fsS "http://127.0.0.1:${{VLLM_PORT}}/health" >/dev/null 2>&1; then VLLM_UP=1; break; fi
      sleep 5
    done
    if test -n "$VLLM_UP"; then
      echo "[vision] vLLM healthy — QWEN_BASE_URL=http://${{PUBLIC_IP}}:${{VLLM_PORT}}/v1"
    else
      echo "[vision] WARN: vLLM not healthy after 3min (still downloading weights?) — journalctl -u filmforge-vllm" >&2
    fi
  fi
fi
)

# ── Audio department ──────────────────────────────────────────────────────────
# The sound stage (resident Parler voice + SA3 music): provision the model
# stacks (idempotent — instant when the /mnt/data volume already carries them)
# and install the servers via setup_audio_services.sh. Selected either by the
# per-GPU plan (AUDIO_GPU_IDX, which also pins the servers to that card) or, on
# a homogeneous box, by tts_dialogue/stable_audio3 in WORKER_CAPABILITIES.
# Guarded subshell: audio failures degrade to warnings, never kill a deploy.
(
set +e
_audio_wanted=""
if test -n "$AUDIO_GPU_IDX"; then
  _audio_wanted=1
elif test "${{#WORKER_PLAN[@]}}" -eq 0; then
  case ",${{WORKER_CAPABILITIES:-}}," in
    *,tts_dialogue,*|*,stable_audio3,*) _audio_wanted=1 ;;
  esac
fi
if test -n "$_audio_wanted"; then
    echo "[audio] setting up the sound stage${{AUDIO_GPU_IDX:+ on gpu$AUDIO_GPU_IDX}}"
    cd "$WORKER_ROOT"
    export WORKSPACE=/mnt/data
    export HF_HUB_ENABLE_HF_TRANSFER=0
    # Pin the resident servers to the plan's audio card, and leave the worker
    # unit alone — the plan already wrote its capabilities.
    if test -n "$AUDIO_GPU_IDX"; then
      export AUDIO_GPU_INDEX="$AUDIO_GPU_IDX"
      export AUDIO_SKIP_WORKER_CAPS=1
    fi
    if [ -n "${{HF_TOKEN:-}}" ]; then
      bash provision_tts.sh || echo "[audio] WARN: provision_tts.sh failed" >&2
      bash provision_sa3.sh || echo "[audio] WARN: provision_sa3.sh failed" >&2
    else
      echo "[audio] HF_TOKEN not set — skipping model downloads (volume assumed provisioned)"
    fi
    bash setup_audio_services.sh || echo "[audio] WARN: setup_audio_services.sh failed" >&2
fi
)

# ── Re-assert the plan's capabilities (drift guard) ───────────────────────────
# The provisioners above are shell scripts read from the box's git checkout of
# gpu_worker, NOT from the deploy we just piped in — so an older checkout can
# still be carrying the pre-plan setup_audio_services.sh, which bolts
# tts_dialogue/stable_audio3 onto filmforge-worker-gpu0 and restarts it. That
# silently recreates the exact department mix this plan exists to prevent
# (observed on the first live 4-GPU deploy, 2026-07-26). The plan is the truth:
# rewrite any unit whose capabilities drifted from it and say so loudly.
if test "${{#WORKER_PLAN[@]}}" -gt 0; then
  for idx in $(seq 0 $((GPU_COUNT - 1))); do
    dept="$(dept_for_idx "$idx")"
    test "$dept" != "none" || continue
    unit="/etc/systemd/system/filmforge-worker-gpu${{idx}}.service"
    test -f "$unit" || continue
    want="$(caps_for_dept "$dept")"
    have="$(grep -m1 '^Environment=.*WORKER_CAPABILITIES=' "$unit" | sed -e 's/.*WORKER_CAPABILITIES=//' -e 's/"$//')"
    if test "$have" != "$want"; then
      echo "[verda] WARNING: gpu${{idx}} ($dept) capabilities drifted from the plan — was '$have', restoring '$want'" >&2
      sed -i "s|^Environment=\\"WORKER_CAPABILITIES=.*\\"$|Environment=\\"WORKER_CAPABILITIES=${{want}}\\"|" "$unit"
      systemctl daemon-reload
      systemctl restart "filmforge-worker-gpu${{idx}}.service"
    fi
  done

  # Same drift, second half: an old setup_audio_services.sh writes the resident
  # Parler/SA3 units with NO CUDA_VISIBLE_DEVICES, so both load onto GPU 0 and
  # sit in a render card's VRAM. Re-pin them to the plan's audio card.
  if test -n "$AUDIO_GPU_IDX"; then
    for svc in filmforge-parler filmforge-sa3; do
      unit="/etc/systemd/system/$svc.service"
      test -f "$unit" || continue
      want="Environment=CUDA_VISIBLE_DEVICES=$AUDIO_GPU_IDX"
      if ! grep -qxF "$want" "$unit"; then
        echo "[verda] WARNING: $svc is not pinned to gpu${{AUDIO_GPU_IDX}} — re-pinning" >&2
        sed -i '/^Environment=CUDA_VISIBLE_DEVICES=/d' "$unit"
        sed -i "/^\\[Service\\]/a $want" "$unit"
        systemctl daemon-reload
        systemctl restart "$svc"
      fi
    done
  fi
fi

# ── Semantic search service ───────────────────────────────────────────────────
# Run the whole block in a guarded subshell: the outer script is `set -e`, so any
# failure here (pip, embedding, systemd) would otherwise abort the entire worker
# deploy. Semantic search is non-essential to the GPU worker — degrade to a
# warning instead of killing the deploy.
(
set +e
SEMANTIC_DIR="/mnt/data/semantic_search"
SEMANTIC_ART="$SEMANTIC_DIR/artifacts/semantics"
SEMANTIC_PORT={semantic_port}
SEMANTIC_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

mkdir -p "$SEMANTIC_ART"

# Install dependencies if not already present
if ! python3 -c "import faiss, flask, sentence_transformers" 2>/dev/null; then
  # --ignore-installed blinker: flask depends on blinker, but the Debian-shipped
  # blinker 1.7.0 has no RECORD file, so pip's uninstall step fails and aborts the
  # whole deploy. Skip uninstalling it and install our own copy alongside.
  pip3 install --quiet --break-system-packages --ignore-installed blinker flask sentence-transformers faiss-gpu numpy tqdm
fi

# Write the serve app (idempotent)
cat > "$SEMANTIC_DIR/app.py" << 'SEMANTIC_APP_EOF'
import os, json, faiss, numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

MODEL_DIR = os.getenv("MODEL_DIR", "/mnt/data/semantic_search/artifacts/semantics")
ST_MODEL  = os.getenv("ST_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TMDB_BASE = "https://image.tmdb.org/t/p/w500"

def _title_from(m):  return m.get("title") or m.get("name") or m.get("original_title")
def _year_from(m):   return m.get("year") or m.get("release_year")
def _poster_from(m):
    p = m.get("poster") or m.get("posterUrl") or m.get("poster_path")
    if p and isinstance(p, str) and not p.startswith("http"):
        return TMDB_BASE + p
    return p

index = faiss.read_index(os.path.join(MODEL_DIR, "movies.faiss"))
ids   = np.load(os.path.join(MODEL_DIR, "ids.npy"))
with open(os.path.join(MODEL_DIR, "meta.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)
model = SentenceTransformer(ST_MODEL)
print(f"[semantic] Loaded {{index.ntotal}} vectors, {{len(meta)}} meta entries", flush=True)

app = Flask(__name__)

@app.get("/ping")
def ping(): return "ok", 200

@app.post("/invocations")
def invocations():
    payload = request.get_json(force=True)
    q = (payload.get("q") or "").strip()
    k = int(payload.get("k", 20))
    if not q: return jsonify({{"error": "Missing q"}}), 400
    qv = model.encode([q], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(qv, k)
    items = []
    for i, s in zip(idxs[0], scores[0]):
        if i < 0: continue
        mid = str(ids[int(i)])
        m = meta.get(mid, {{}})
        poster = _poster_from(m)
        if poster and not poster.startswith("http"):
            poster = TMDB_BASE + poster
        items.append({{"item_id": mid, "title": _title_from(m), "year": _year_from(m), "posterUrl": poster, "score": float(s)}})
    return jsonify({{"items": items}})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8082")), debug=False)
SEMANTIC_APP_EOF

# Build artifacts if they don't exist yet (first boot after fresh deploy)
if [ ! -f "$SEMANTIC_ART/movies.faiss" ]; then
  echo "[semantic] No FAISS index found — running embedding job..." >&2
  DATA_FILE="/mnt/data/semantic_search/movies_colab_last.ndjson"
  if [ ! -f "$DATA_FILE" ]; then
    echo "[semantic] ERROR: movie data not found at $DATA_FILE" >&2
    echo "[semantic] Upload it with: scp movies_colab_last.ndjson root@$PUBLIC_IP:$DATA_FILE" >&2
  else
    cat > /tmp/embed_and_index.py << 'EMBED_EOF'
import sys, os, json, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch, faiss

MODEL_NAME = os.environ.get("ST_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
in_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

ids, texts, meta = [], [], {{}}
with open(in_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        m = json.loads(line)
        ov = (m.get("overview") or "").strip()
        if not ov: continue
        mid = int(m["id"])
        title = m.get("title", "")
        lang = m.get("original_language", "")
        genres = " ".join(m.get("genres", []))
        ids.append(mid)
        texts.append(f"{{title}}\nLANG:{{lang}}\nGENRES:{{genres}}\n{{ov}}".strip())
        meta[str(mid)] = {{"title": title, "year": m.get("year"), "poster": m.get("poster_path") or m.get("poster"), "genres": m.get("genres", []), "overview": ov}}

print(f"Encoding {{len(ids)}} movies...", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)
vecs = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True).astype("float32")

np.save(out_dir / "item_ids.npy", np.array(ids, dtype=np.int64))
np.save(out_dir / "item_vecs_norm.npy", vecs)
import shutil; shutil.copy(out_dir / "item_ids.npy", out_dir / "ids.npy")
with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False)

index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
faiss.write_index(index, str(out_dir / "movies.faiss"))
print(f"Done: {{index.ntotal}} vectors indexed", flush=True)
EMBED_EOF
    ST_MODEL="$SEMANTIC_MODEL" python3 /tmp/embed_and_index.py "$DATA_FILE" "$SEMANTIC_ART" \
      >> /var/log/semantic_embed.log 2>&1 \
      && echo "[semantic] Embedding complete" >&2 \
      || echo "[semantic] Embedding failed — check /var/log/semantic_embed.log" >&2
  fi
fi

# Write systemd unit for semantic search service
cat > /etc/systemd/system/semantic-search.service << SEMANTIC_UNIT_EOF
[Unit]
Description=FilmForge Semantic Search
After=network.target

[Service]
Type=simple
Environment=MODEL_DIR=$SEMANTIC_ART
Environment=ST_MODEL=$SEMANTIC_MODEL
Environment=PORT=$SEMANTIC_PORT
ExecStart=python3 $SEMANTIC_DIR/app.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SEMANTIC_UNIT_EOF

systemctl daemon-reload
systemctl enable --now semantic-search.service

# Wait for it to be ready
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$SEMANTIC_PORT/ping" >/dev/null 2>&1; then
    echo "[semantic] Server ready on port $SEMANTIC_PORT" >&2
    break
  fi
  sleep 3
done

echo "SEMANTIC_URL=http://${{PUBLIC_IP}}:${{SEMANTIC_PORT}}"
) || echo "[semantic] WARNING: semantic-search setup failed — worker deploy continuing without it" >&2
"""


def verda_fresh_install_script(
    *,
    worker_repo_url: str,
    comfy_repo_url: str,
    pytorch_index_url: str,
    remote_root: str,
    patch_content: str = "",
    worker_source_root: str | None = None,
) -> str:
    """Install ComfyUI and gpu_worker onto a fresh Verda base image."""

    # Retained as an API/CLI compatibility argument. Worker code now arrives as
    # a verified immutable package; this installer never clones or pulls it.
    del worker_repo_url
    worker_source_root = worker_source_root or f"{DEFAULT_WORKER_RELEASES_ROOT}/current/gpu_worker"

    if patch_content:
        _cuda_patch_block = (
            "mkdir -p \"$COMFY_ROOT/custom_nodes/filmforge_cuda_patch\"\n"
            "cat > \"$COMFY_ROOT/custom_nodes/filmforge_cuda_patch/__init__.py\" << 'FILMFORGE_PATCH_EOF'\n"
            + patch_content.rstrip("\n")
            + "\nFILMFORGE_PATCH_EOF\n"
            "echo \"FilmForge cuDNN patch custom node installed\""
        )
    else:
        _cuda_patch_block = (
            "echo \"[verda] WARNING: cuDNN patch not embedded — Blackwell SDP errors may occur\" >&2"
        )

    security_gate = worker_security_stage_gate_script()
    return f"""#!/usr/bin/env bash
set -euo pipefail

{security_gate}

export DEBIAN_FRONTEND=noninteractive
REMOTE_ROOT={shlex.quote(remote_root)}
WORKER_ROOT={shlex.quote(worker_source_root)}
WORKER_RUNTIME_ROOT={shlex.quote(DEFAULT_WORKER_RUNTIME_ROOT)}
COMFY_ROOT="/workspace/ComfyUI"
COMFY_REPO_URL={shlex.quote(comfy_repo_url)}
PYTORCH_INDEX_URL={shlex.quote(pytorch_index_url)}

wait_for_apt_locks() {{
  local waited=0
  local timeout="${{APT_LOCK_TIMEOUT_SEC:-600}}"
  while fuser /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock /var/cache/apt/archives/lock >/dev/null 2>&1; do
    if test "$waited" -ge "$timeout"; then
      echo "Timed out waiting for apt/dpkg locks after ${{timeout}}s" >&2
      fuser -v /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock /var/cache/apt/archives/lock >&2 || true
      return 1
    fi
    echo "Waiting for apt/dpkg locks to clear... (${{waited}}s)" >&2
    sleep 5
    waited=$((waited + 5))
  done
}}

apt_retry() {{
  local attempt
  for attempt in 1 2 3; do
    wait_for_apt_locks
    if apt-get "$@"; then
      return 0
    fi
    echo "apt-get $* failed on attempt ${{attempt}}; retrying..." >&2
    sleep 10
  done
  wait_for_apt_locks
  apt-get "$@"
}}

BOOTSTRAP_PACKAGES=(
  ca-certificates curl git rsync python3 python3-dev python3-pip python3-venv
  build-essential e2fsprogs ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
)

repair_ubuntu_package_sources() {{
  # Some Verda base images point at a regional cloud mirror whose InRelease
  # files are current but whose noble-updates/security Packages indexes are
  # empty. apt-get update then succeeds while every install is unsatisfiable.
  # Move only Ubuntu's own sources to the canonical HTTPS endpoints; leave
  # NVIDIA CUDA and Docker repositories untouched.
  test -f /etc/os-release || return 1
  . /etc/os-release
  test "${{ID:-}}" = "ubuntu" || return 1

  local source_file backup_dir backup_file
  backup_dir="/var/backups/filmforge-apt"
  mkdir -p "$backup_dir"
  for source_file in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
    test -f "$source_file" || continue
    if grep -Eq '(clouds\\.archive\\.ubuntu\\.com/ubuntu|security\\.ubuntu\\.com/ubuntu)' "$source_file"; then
      backup_file="$backup_dir/${{source_file##*/}}.original"
      test -f "$backup_file" || cp -a "$source_file" "$backup_file"
      sed -Ei \
        -e 's#https?://[^[:space:]]*clouds\\.archive\\.ubuntu\\.com/ubuntu/?#https://archive.ubuntu.com/ubuntu/#g' \
        -e 's#http://security\\.ubuntu\\.com/ubuntu/?#https://security.ubuntu.com/ubuntu/#g' \
        "$source_file"
    fi
  done

  # Zero-byte indexes from the broken mirror otherwise remain valid cache
  # entries because its release metadata itself was successfully downloaded.
  find /var/lib/apt/lists -maxdepth 1 -type f -size 0 -delete 2>/dev/null || true
  apt_retry update -o Acquire::Retries=3
}}

apt_retry update
if ! apt-get install --simulate --no-install-recommends "${{BOOTSTRAP_PACKAGES[@]}}" >/dev/null 2>&1; then
  echo "[verda] Ubuntu package catalog is inconsistent; switching to canonical HTTPS mirrors..." >&2
  repair_ubuntu_package_sources
fi
if ! apt-get install --simulate --no-install-recommends "${{BOOTSTRAP_PACKAGES[@]}}" >/dev/null 2>&1; then
  echo "[verda] ERROR: Ubuntu package dependencies remain unsatisfiable after mirror repair." >&2
  apt-get install --simulate --no-install-recommends "${{BOOTSTRAP_PACKAGES[@]}}" >&2 || true
  exit 1
fi
apt_retry install -y --no-install-recommends "${{BOOTSTRAP_PACKAGES[@]}}"

mkdir -p /opt /workspace "$REMOTE_ROOT" "$WORKER_RUNTIME_ROOT"

# Mount the persistent data volume at /mnt/data early so it's available — but do
# NOT bind-mount onto /workspace/ComfyUI/* yet. Cloning ComfyUI needs to
# `rm -rf "$COMFY_ROOT"` if a stub exists, and that fails on bind mounts
# with "Device or resource busy". We move the asset dirs onto the data
# volume AFTER ComfyUI is installed (see _bind_comfy_asset_dirs below).
if test -b /dev/vdb; then
  VDB_FS="$(blkid -s TYPE -o value /dev/vdb 2>/dev/null || true)"
  if test -z "$VDB_FS"; then
    # A previous fresh deploy can fail before its new data volume is formatted.
    # Only initialize a volume that has no known signatures and whose leading
    # and trailing 4 MiB samples are zero. Anything else requires inspection.
    VDB_SIGNATURES="$(wipefs -n --noheadings --output TYPE /dev/vdb 2>/dev/null | tr -d '[:space:]' || true)"
    VDB_SIZE="$(blockdev --getsize64 /dev/vdb 2>/dev/null || true)"
    SAMPLE_BYTES=$((4 * 1024 * 1024))
    if test -n "$VDB_SIGNATURES" || test "${{VDB_SIZE:-0}}" -lt "$SAMPLE_BYTES"; then
      echo "[verda] ERROR: /dev/vdb has no mountable filesystem and cannot be proven blank; refusing to format." >&2
      exit 1
    fi
    LAST_SAMPLE=$(((VDB_SIZE - SAMPLE_BYTES) / SAMPLE_BYTES))
    if ! cmp -s <(dd if=/dev/vdb bs=4M count=1 status=none) \
               <(dd if=/dev/zero bs=4M count=1 status=none) \
        || ! cmp -s <(dd if=/dev/vdb bs=4M skip="$LAST_SAMPLE" count=1 status=none) \
                  <(dd if=/dev/zero bs=4M count=1 status=none); then
      echo "[verda] ERROR: /dev/vdb contains unrecognized data; refusing to format it automatically." >&2
      exit 1
    fi
    echo "[verda] Initializing blank data volume /dev/vdb as ext4..." >&2
    mkfs.ext4 -F /dev/vdb
    VDB_FS="ext4"
  elif test "$VDB_FS" != "ext4"; then
    echo "[verda] ERROR: Unsupported filesystem '$VDB_FS' on /dev/vdb; expected ext4." >&2
    exit 1
  fi
  mkdir -p /mnt/data
  if ! mountpoint -q /mnt/data; then
    mount /dev/vdb /mnt/data
  fi
  VDB_UUID="$(blkid -s UUID -o value /dev/vdb 2>/dev/null || true)"
  if test -n "$VDB_UUID" && ! grep -q "$VDB_UUID" /etc/fstab; then
    echo "UUID=$VDB_UUID /mnt/data ext4 defaults,nofail 0 2" >> /etc/fstab
  fi
  # Tear down any stale bind mounts from a prior failed run so the rm -rf
  # ahead of git clone can succeed.
  for dir in models input output temp; do
    if mountpoint -q "$COMFY_ROOT/$dir" 2>/dev/null; then
      umount "$COMFY_ROOT/$dir" || true
    fi
  done
fi

test -f "$WORKER_ROOT/app.py" || {{
  echo "immutable gpu_worker release is missing at $WORKER_ROOT" >&2
  exit 1
}}
ln -sfn "$WORKER_ROOT" /opt/gpu_worker

test -x "$WORKER_ROOT/.venv/bin/python" || {{
  echo "immutable worker candidate venv is missing" >&2
  exit 1
}}

if test -d "$COMFY_ROOT/.git"; then
  # A dirty or divergent runtime checkout is an operator-visible conflict, not
  # permission to continue into service rewrites with unknown code.
  GIT_TERMINAL_PROMPT=0 GIT_PAGER=cat git -C "$COMFY_ROOT" pull --ff-only </dev/null
else
  rm -rf "$COMFY_ROOT"
  GIT_TERMINAL_PROMPT=0 git clone "$COMFY_REPO_URL" "$COMFY_ROOT"
fi

if ! test -x "$COMFY_ROOT/.venv/bin/python"; then
  python3 -m venv "$COMFY_ROOT/.venv"
fi
"$COMFY_ROOT/.venv/bin/python" -m pip install --upgrade pip wheel setuptools
"$COMFY_ROOT/.venv/bin/python" -m pip install --index-url "$PYTORCH_INDEX_URL" \
  '{_TORCH_PIN}' '{_TORCHVISION_PIN}' '{_TORCHAUDIO_PIN}'
# ComfyUI pins only transformers>=4.50.3; v5 changed the MistralConverter API
# its FLUX.2 tokenizer path calls, so a fresh resolve to 5.x breaks flux
# stills at text-encoder load (proven live on the first FIN-02 render).
"$COMFY_ROOT/.venv/bin/python" -m pip install -r "$COMFY_ROOT/requirements.txt" "transformers<5"

mkdir -p "$COMFY_ROOT/custom_nodes"
if ! test -d "$COMFY_ROOT/custom_nodes/ComfyUI-VideoHelperSuite"; then
  GIT_TERMINAL_PROMPT=0 git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "$COMFY_ROOT/custom_nodes/ComfyUI-VideoHelperSuite" || true
fi
if ! test -d "$COMFY_ROOT/custom_nodes/ComfyUI_IPAdapter_plus"; then
  GIT_TERMINAL_PROMPT=0 git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git "$COMFY_ROOT/custom_nodes/ComfyUI_IPAdapter_plus" || true
fi

# Install FilmForge cuDNN patch custom node (disables cuDNN SDP to prevent
# "No valid execution plans built" on Blackwell and other unsupported GPUs).
{_cuda_patch_block}

for req in "$COMFY_ROOT"/custom_nodes/*/requirements.txt; do
  if test -f "$req"; then
    "$COMFY_ROOT/.venv/bin/python" -m pip install -r "$req" || true
  fi
done
# Custom node requirements.txt files often list `torchvision` without an
# index URL, which lets pip install the CPU-only PyPI wheel and clobber the
# CUDA version we just installed. Re-pin torch/torchvision/torchaudio after
# the loop to guarantee the CUDA wheels always win.
"$COMFY_ROOT/.venv/bin/python" -m pip install --force-reinstall --no-deps \
  --index-url "$PYTORCH_INDEX_URL" '{_TORCH_PIN}' '{_TORCHVISION_PIN}' '{_TORCHAUDIO_PIN}'

# ComfyUI-LTXVideo currently imports `pad` from kornia.geometry.transform.pyramid,
# but kornia 0.8.x no longer exports it there. Patch the custom node to use
# torch.nn.functional.pad so LTX nodes import after fresh deploys/redeploys.
if test -f "$COMFY_ROOT/custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py"; then
  "$COMFY_ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path

p = Path("/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py")
if p.exists():
    s = p.read_text()
    if "from torch.nn.functional import pad" not in s:
        s = s.replace(
            "import torch\nimport torch.nn.functional as F\n",
            "import torch\nimport torch.nn.functional as F\nfrom torch.nn.functional import pad\n",
        )
    s = s.replace("    is_powerof_two,\n    pad,\n)", "    is_powerof_two,\n)")
    p.write_text(s)
PY
fi

# Now that ComfyUI is fully installed, move its asset dirs onto the model/data
# data volume and bind-mount them back. Doing this AFTER install avoids the
# rm-on-busy-mount problem (the install needs to `rm -rf "$COMFY_ROOT"` for
# a clean clone, which fails if the subdirs are bind mounts).
if mountpoint -q /mnt/data; then
  mkdir -p /mnt/data/ComfyUI/models /mnt/data/ComfyUI/input \
           /mnt/data/ComfyUI/output /mnt/data/ComfyUI/temp
  for dir in models input output temp; do
    if mountpoint -q "$COMFY_ROOT/$dir"; then
      continue
    fi
    mkdir -p "$COMFY_ROOT/$dir"
    # Seed the data volume with whatever ComfyUI shipped (usually empty
    # placeholder subdirs like models/checkpoints, models/vae, etc.).
    if test -n "$(ls -A "$COMFY_ROOT/$dir" 2>/dev/null || true)"; then
      cp -aRT "$COMFY_ROOT/$dir/" "/mnt/data/ComfyUI/$dir/"
      rm -rf "$COMFY_ROOT/$dir"
      mkdir -p "$COMFY_ROOT/$dir"
    fi
    mount --bind "/mnt/data/ComfyUI/$dir" "$COMFY_ROOT/$dir"
  done
  touch /mnt/data/.filmforge-bootstrap-complete
fi

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


def _remote_verified_worker_release(remote_output: str, release_id: str) -> bool:
    return any(
        line.strip() == f"WORKER_RELEASE_VERIFIED={release_id}"
        for line in (remote_output or "").splitlines()
    )


def _verified_receipt_worker_urls(
    remote_output: str,
    *,
    release_id: str,
    public_urls: list[str],
) -> list[str]:
    """Return only contract-pinned URLs after the receipt gate proves cutover."""

    if not _remote_verified_worker_release(remote_output, release_id):
        return []
    return list(public_urls)


def _verda_env_vars(
    args: argparse.Namespace,
    *,
    instance_id: str | None = None,
) -> list[str]:
    env_vars = list(getattr(args, "env_vars", []) or [])
    existing_keys = {item.split("=", 1)[0] for item in env_vars if "=" in item}
    if instance_id and "WORKER_INSTANCE_ID" not in existing_keys:
        env_vars.append(f"WORKER_INSTANCE_ID={instance_id}")
        existing_keys.add("WORKER_INSTANCE_ID")
    for key in (
        "WORKER_REGISTRATION_TOKEN",
        "RENDER_BROKER_WORKER_TOKEN",
        "WORKER_API_TOKEN",
        "GPU_WORKER_API_TOKEN",
        "WORKER_API_AUTH_MODE",
        "WORKER_INPUT_URL_ALLOWED_HOSTS",
        "WORKER_PUBLIC_URLS",
        "WORKER_TUNNEL_LOCAL_URLS",
        "WORKER_TUNNEL_UNITS",
        "WORKER_SECURITY_STAGE_RECEIPTS",
        "WORKER_DEPLOY_PHASE",
        "FILMFORGE_BACKEND_CLIENT_AUTH_MODE",
    ):
        if key not in existing_keys:
            value = _read_env_value(args.backend_env, key) or os.getenv(key)
            if value:
                env_vars.append(f"{key}={value}")
                existing_keys.add(key)
                log(f"Auto-injected {key} from backend .env")
    has_registration_token = bool(
        {"WORKER_REGISTRATION_TOKEN", "RENDER_BROKER_WORKER_TOKEN"} & existing_keys
    )
    if has_registration_token and "FILMFORGE_BACKEND_URL" not in existing_keys:
        value = _read_env_value(args.backend_env, "FILMFORGE_BACKEND_URL") or os.getenv("FILMFORGE_BACKEND_URL")
        if value:
            env_vars.append(f"FILMFORGE_BACKEND_URL={value}")
            log("Auto-injected FILMFORGE_BACKEND_URL from backend .env")
    # Audio deploys (tts_dialogue / stable_audio3 capability) need HF_TOKEN on the
    # box for the gated repos (SA3 license, Indic Parler terms). Fallback chain:
    # backend .env → local env → the hf CLI's token file.
    caps = next(
        (item.split("=", 1)[1] for item in env_vars if item.startswith("WORKER_CAPABILITIES=")),
        os.getenv("WORKER_CAPABILITIES", ""),
    )
    # A per-GPU plan with an "audio" card needs HF_TOKEN just the same, and its
    # audio capabilities never appear in the broadcast WORKER_CAPABILITIES.
    plan_wants_audio = "audio" in parse_worker_plan(getattr(args, "verda_worker_plan", "") or "")
    if (plan_wants_audio or "tts_dialogue" in caps or "stable_audio3" in caps) and "HF_TOKEN" not in existing_keys:
        value = _read_env_value(args.backend_env, "HF_TOKEN") or os.getenv("HF_TOKEN")
        if not value:
            token_file = Path.home() / ".cache" / "huggingface" / "token"
            if token_file.exists():
                value = token_file.read_text().strip()
        if value:
            env_vars.append(f"HF_TOKEN={value}")
            log("Auto-injected HF_TOKEN for audio provisioning")
    _validate_worker_public_url_env(env_vars)
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


def _wait_for_interrupted_verda_pair(
    args: argparse.Namespace,
    *,
    timeout_sec: int = 180,
) -> bool:
    """Wait until a discontinued spot VM is gone and both reusable volumes detach."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            instance = _verda_instance_by_hostname(args, args.verda_hostname)
            volumes = _verda_json(args, "volume", "list", timeout=60)
            os_volume = _verda_find_volume(volumes, args.verda_os_volume_id)
            data_volume = _verda_find_volume(volumes, args.verda_data_volume_id)
            detached = all(
                str(volume.get("status") or "").lower() == "detached"
                for volume in (os_volume, data_volume)
            )
            if instance is None and detached:
                return True
        except RuntimeError as exc:
            log(f"  Waiting for interrupted Verda resources ({exc})...")
        time.sleep(5)
    return False


def verda_deploy_with_spot_retries(args: argparse.Namespace) -> int:
    """Resume a reusable-volume deploy when Verda preempts spot during SSH work."""
    try:
        max_attempts = max(1, int(os.getenv("VERDA_SPOT_DEPLOY_ATTEMPTS", "3")))
    except ValueError:
        max_attempts = 3
    if _verda_contract(args) != "spot":
        max_attempts = 1

    for attempt in range(1, max_attempts + 1):
        try:
            return verda_deploy(args)
        except subprocess.CalledProcessError as exc:
            # ssh returns 255 when the remote host disappears. With spot, Verda
            # detaches the preserved OS+data pair, and the idempotent bootstrap
            # can continue on a replacement VM without redoing completed work.
            if exc.returncode != 255 or attempt >= max_attempts:
                raise
            log(
                f"[verda] Spot VM disconnected during deploy (attempt {attempt}/{max_attempts}); "
                "waiting for preserved volumes to detach..."
            )
            if not _wait_for_interrupted_verda_pair(args):
                raise RuntimeError(
                    "Verda spot VM disconnected, but its reusable volumes did not "
                    "become detached within 180s"
                ) from exc
            log(
                f"[verda] Preserved volumes are detached; provisioning replacement "
                f"attempt {attempt + 1}/{max_attempts}..."
            )

    raise RuntimeError("Verda spot deployment attempts exhausted")


def verda_deploy(args: argparse.Namespace) -> int:
    preflight_env_vars = _verda_env_vars(args)
    worker_plan = parse_worker_plan(getattr(args, "verda_worker_plan", ""))
    requested_worker_count = len(worker_plan) or int(args.verda_worker_count or 0)
    if requested_worker_count < 1:
        raise RuntimeError(
            "Secure Verda deploy requires an explicit --verda-worker-count or worker plan"
        )
    security_contract = _preflight_complete_worker_contract(
        preflight_env_vars,
        worker_port=args.worker_port,
        expected_worker_count=requested_worker_count or None,
    )
    contract_stage_receipts = list(security_contract["stage_receipts"])
    contract_public_urls = list(security_contract["public_urls"])
    deploy_phase = str(security_contract["deploy_phase"])
    _prepare_worker_release_bundle(args)

    identity = (args.ssh_identity or Path.home() / ".ssh" / "id_ed25519").expanduser()
    if not identity.exists():
        raise RuntimeError(f"SSH private key not found at {identity}")

    hostname = args.verda_hostname
    resume_id = str(getattr(args, "verda_existing_instance_id", "") or "").strip()
    if resume_id:
        ip, instance_id = _resume_verda_instance(args, resume_id)
    else:
        _verda_preflight(args)
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
    ssh_cmd, scp_cmd, destination = parse_ssh_command(ssh_command)
    ssh_cmd = add_default_host_key_policy(ssh_cmd)
    scp_cmd = add_default_host_key_policy(scp_cmd)
    ssh_cmd = add_default_ssh_liveness_policy(ssh_cmd)

    pair_state = _probe_verda_rehydrate_state(ssh_cmd)
    log(
        "Verda reusable-pair state: "
        f"data={pair_state['DATA_STATE']} "
        f"worker_ready={pair_state['WORKER_READY']} "
        f"comfy_ready={pair_state['COMFY_READY']} "
        f"bootstrap_ready={pair_state['BOOTSTRAP_READY']}"
    )
    release_id, worker_source_root = _stage_worker_release_over_ssh(
        ssh_cmd=ssh_cmd,
        scp_cmd=scp_cmd,
        destination=destination,
        releases_root=DEFAULT_WORKER_RELEASES_ROOT,
        venv_path=f"{DEFAULT_WORKER_RUNTIME_ROOT}/.venv",
        bundle=getattr(args, "_prepared_worker_release_bundle", None),
    )
    setattr(args, "_verda_active_instance_id", instance_id)
    setattr(args, "_verda_active_ip", ip)
    setattr(args, "_verda_worker_release_id", release_id)
    setattr(args, "_verda_worker_source_root", worker_source_root)
    setattr(args, "_verda_ssh_cmd", list(ssh_cmd))
    setattr(args, "_verda_scp_cmd", list(scp_cmd))
    setattr(args, "_verda_destination", destination)
    preflight_env_vars.append(f"WORKER_CODE_RELEASE_ID={release_id}")
    if deploy_phase == "stage-code":
        log(
            "Verda worker code candidate staged; skipping volume bootstrap, GPU, "
            "unit, process, dataset, and backend work"
        )
        return 0
    if deploy_phase != "activate" and _verda_pair_needs_bootstrap(pair_state):
        log(
            "Incomplete reusable Verda volume pair detected; "
            "bootstrapping the attached OS and data volumes before rehydration..."
        )
        _patch_path = Path(__file__).parent / "comfy_torch_patch.py"
        install_script = verda_fresh_install_script(
            worker_repo_url=args.verda_worker_repo_url,
            comfy_repo_url=args.verda_comfy_repo_url,
            pytorch_index_url=args.verda_pytorch_index_url,
            remote_root=args.remote_root,
            patch_content=_patch_path.read_text() if _patch_path.exists() else "",
            worker_source_root=worker_source_root,
        )
        install_exports = build_bootstrap_env_exports(preflight_env_vars)
        if install_exports:
            install_script = f"{install_exports}\n\n{install_script}"
        _run_verda_ssh_script(
            ssh_cmd,
            install_script,
            timeout_sec=args.verda_install_timeout,
            capture_output=False,
            operation="bootstrapping the incomplete Verda volume pair",
        )
        verified_state = _probe_verda_rehydrate_state(ssh_cmd)
        if _verda_pair_needs_bootstrap(verified_state):
            raise RuntimeError(
                "Verda bootstrap command completed but the worker stack is still incomplete"
            )
        log("Verda volume pair bootstrap completed; continuing with worker startup.")

    env_vars = _verda_env_vars(args, instance_id=instance_id)
    env_vars.append(f"WORKER_CODE_RELEASE_ID={release_id}")

    _patch_path = Path(__file__).parent / "comfy_torch_patch.py"
    script = verda_rehydrate_script(
        public_ip=ip,
        worker_port=args.worker_port,
        comfy_port=args.verda_comfy_port,
        worker_count=args.verda_worker_count,
        remote_root=args.remote_root,
        patch_content=_patch_path.read_text() if _patch_path.exists() else "",
        worker_plan=worker_plan,
        vllm_port=getattr(args, "verda_vllm_port", 8100),
        worker_source_root=worker_source_root,
    )
    env_exports = build_bootstrap_env_exports(env_vars)
    if env_exports:
        script = f"{env_exports}\n\n{script}"

    # Upload movie data for semantic search if not already on the volume
    _movie_data = Path(__file__).parent.parent / "duku 1.0" / "duku-recs" / "data" / "tmdb_raw" / "movies_colab_last.ndjson"
    if deploy_phase != "activate" and _movie_data.exists():
        log(f"Uploading movie dataset ({_movie_data.stat().st_size // 1_000_000}MB) for semantic search...")
        run([*ssh_cmd, "mkdir -p /mnt/data/semantic_search"], check=False)
        scp_cmd = ["scp", f"-i{identity}", "-o", "StrictHostKeyChecking=no",
                   "-o", "UserKnownHostsFile=/dev/null", "-o", "GlobalKnownHostsFile=/dev/null",
                   str(_movie_data), f"root@{ip}:/mnt/data/semantic_search/movies_colab_last.ndjson"]
        run(scp_cmd, check=False)
    else:
        log(f"Movie dataset not found at {_movie_data} — semantic embedding will be skipped on GPU")

    log("Running Verda post-boot worker fixup...")
    try:
        remote_result = _run_verda_ssh_script(
            ssh_cmd,
            script,
            timeout_sec=args.verda_install_timeout,
            capture_output=True,
            operation="starting workers on the rehydrated Verda VM",
        )
    except Exception as exc:
        if getattr(exc, "stdout", None):
            print(exc.stdout, end="")
        if getattr(exc, "stderr", None):
            print(exc.stderr, file=sys.stderr, end="")
        _rollback_failed_worker_transaction_over_ssh(
            ssh_cmd=ssh_cmd,
            releases_root=DEFAULT_WORKER_RELEASES_ROOT,
            worker_source_root=worker_source_root,
            failed_release_id=release_id,
            stage_receipt_paths=contract_stage_receipts,
            deploy_phase=deploy_phase,
        )
        raise
    if remote_result.stdout:
        print(remote_result.stdout, end="")
    if remote_result.stderr:
        print(remote_result.stderr, file=sys.stderr, end="")

    if _remote_verified_worker_release(remote_result.stdout, release_id):
        _activate_or_rollback_worker_release_over_ssh(
            ssh_cmd=ssh_cmd,
            releases_root=DEFAULT_WORKER_RELEASES_ROOT,
            release_id=release_id,
            worker_source_root=worker_source_root,
            stage_receipt_paths=contract_stage_receipts,
        )
    else:
        log("Worker code remains an unactivated candidate pending secure cutover")

    worker_urls = _verified_receipt_worker_urls(
        remote_result.stdout,
        release_id=release_id,
        public_urls=contract_public_urls,
    )
    if worker_urls:
        _validate_worker_public_url_env(
            [
                *[
                    item
                    for item in env_vars
                    if item.startswith("WORKER_API_AUTH_MODE=")
                ],
                f"WORKER_PUBLIC_URLS={','.join(worker_urls)}",
            ]
        )
        log("Verda worker URLs:")
        for url in worker_urls:
            log(f"  {url}")
        print(f"WORKER_URLS={','.join(worker_urls)}")
    else:
        log("No Verda worker URLs found in post-boot output.")

    # Extract SEMANTIC_URL from script output and update .env
    semantic_url: str | None = None
    for line in (remote_result.stdout or "").splitlines():
        if line.startswith("SEMANTIC_URL="):
            semantic_url = line.split("=", 1)[1].strip()
            break
    if semantic_url:
        log(f"Semantic search URL: {semantic_url}")
        print(f"SEMANTIC_URL={semantic_url}")

    (SCRIPT_DIR / ".last_ssh_dest").write_text(f"-i {identity} root@{ip}\n")

    if worker_urls:
        log(
            "Worker URLs are candidates only; backend routing remains unchanged "
            "until prepare + receipt + authenticated cutover succeed"
        )

    return 0


def verda_fresh_deploy(args: argparse.Namespace) -> int:
    preflight_env_vars = _verda_env_vars(args)
    worker_plan = parse_worker_plan(getattr(args, "verda_worker_plan", ""))
    requested_worker_count = len(worker_plan) or int(args.verda_worker_count or 0)
    if requested_worker_count < 1:
        raise RuntimeError(
            "Secure Verda deploy requires an explicit --verda-worker-count or worker plan"
        )
    security_contract = _preflight_complete_worker_contract(
        preflight_env_vars,
        worker_port=args.worker_port,
        expected_worker_count=requested_worker_count or None,
    )
    contract_stage_receipts = list(security_contract["stage_receipts"])
    contract_public_urls = list(security_contract["public_urls"])
    deploy_phase = str(security_contract["deploy_phase"])
    _prepare_worker_release_bundle(args)

    identity = (args.ssh_identity or Path.home() / ".ssh" / "id_ed25519").expanduser()
    if not identity.exists():
        raise RuntimeError(f"SSH private key not found at {identity}")

    hostname = args.verda_hostname
    os_volume_name = args.verda_fresh_os_volume_name or f"{hostname}-os"
    storage_name = args.verda_fresh_storage_name or f"{hostname}-models"
    resume_id = str(getattr(args, "verda_existing_instance_id", "") or "").strip()
    if resume_id:
        ip, instance_id = _resume_verda_instance(args, resume_id)
    else:
        _verda_fresh_preflight(args)
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
    ssh_cmd, scp_cmd, destination = parse_ssh_command(ssh_command)
    ssh_cmd = add_default_host_key_policy(ssh_cmd)
    scp_cmd = add_default_host_key_policy(scp_cmd)
    ssh_cmd = add_default_ssh_liveness_policy(ssh_cmd)

    release_id, worker_source_root = _stage_worker_release_over_ssh(
        ssh_cmd=ssh_cmd,
        scp_cmd=scp_cmd,
        destination=destination,
        releases_root=DEFAULT_WORKER_RELEASES_ROOT,
        venv_path=f"{DEFAULT_WORKER_RUNTIME_ROOT}/.venv",
        bundle=getattr(args, "_prepared_worker_release_bundle", None),
    )
    setattr(args, "_verda_active_instance_id", instance_id)
    setattr(args, "_verda_active_ip", ip)
    setattr(args, "_verda_worker_release_id", release_id)
    setattr(args, "_verda_worker_source_root", worker_source_root)
    setattr(args, "_verda_ssh_cmd", list(ssh_cmd))
    setattr(args, "_verda_scp_cmd", list(scp_cmd))
    setattr(args, "_verda_destination", destination)
    preflight_env_vars.append(f"WORKER_CODE_RELEASE_ID={release_id}")
    if deploy_phase == "stage-code":
        log(
            "Fresh Verda worker code candidate staged; skipping GPU stack, unit, "
            "process, dataset, and backend work"
        )
        return 0

    _patch_path = Path(__file__).parent / "comfy_torch_patch.py"
    if deploy_phase != "activate":
        install_script = verda_fresh_install_script(
            worker_repo_url=args.verda_worker_repo_url,
            comfy_repo_url=args.verda_comfy_repo_url,
            pytorch_index_url=args.verda_pytorch_index_url,
            remote_root=args.remote_root,
            patch_content=_patch_path.read_text() if _patch_path.exists() else "",
            worker_source_root=worker_source_root,
        )
        preflight_exports = build_bootstrap_env_exports(preflight_env_vars)
        if preflight_exports:
            install_script = f"{preflight_exports}\n\n{install_script}"
        log("Installing worker stack on fresh Verda OS volume...")
        try:
            # Stream the long install directly into the deploy log. Besides making
            # progress visible, this avoids buffering tens of minutes of output in
            # memory while the dashboard appears frozen.
            install_result = _run_verda_ssh_script(
                ssh_cmd,
                install_script,
                timeout_sec=args.verda_install_timeout,
                capture_output=False,
                operation="installing the worker stack on the fresh Verda VM",
            )
        except Exception as exc:
            if getattr(exc, "stdout", None):
                print(exc.stdout, end="")
            if getattr(exc, "stderr", None):
                print(exc.stderr, file=sys.stderr, end="")
            _rollback_failed_worker_transaction_over_ssh(
                ssh_cmd=ssh_cmd,
                releases_root=DEFAULT_WORKER_RELEASES_ROOT,
                worker_source_root=worker_source_root,
                failed_release_id=release_id,
                stage_receipt_paths=contract_stage_receipts,
                deploy_phase=deploy_phase,
            )
            raise
        if install_result.stdout:
            print(install_result.stdout, end="")
        if install_result.stderr:
            print(install_result.stderr, file=sys.stderr, end="")

    env_vars = _verda_env_vars(args, instance_id=instance_id)
    env_vars.append(f"WORKER_CODE_RELEASE_ID={release_id}")
    _patch_path = Path(__file__).parent / "comfy_torch_patch.py"
    script = verda_rehydrate_script(
        public_ip=ip,
        worker_port=args.worker_port,
        comfy_port=args.verda_comfy_port,
        worker_count=args.verda_worker_count,
        remote_root=args.remote_root,
        patch_content=_patch_path.read_text() if _patch_path.exists() else "",
        worker_plan=worker_plan,
        vllm_port=getattr(args, "verda_vllm_port", 8100),
        worker_source_root=worker_source_root,
    )
    env_exports = build_bootstrap_env_exports(env_vars)
    if env_exports:
        script = f"{env_exports}\n\n{script}"

    log("Starting workers on fresh Verda VM...")
    try:
        remote_result = _run_verda_ssh_script(
            ssh_cmd,
            script,
            timeout_sec=args.verda_install_timeout,
            capture_output=True,
            operation="starting workers on the fresh Verda VM",
        )
    except Exception as exc:
        if getattr(exc, "stdout", None):
            print(exc.stdout, end="")
        if getattr(exc, "stderr", None):
            print(exc.stderr, file=sys.stderr, end="")
        _rollback_failed_worker_transaction_over_ssh(
            ssh_cmd=ssh_cmd,
            releases_root=DEFAULT_WORKER_RELEASES_ROOT,
            worker_source_root=worker_source_root,
            failed_release_id=release_id,
            stage_receipt_paths=contract_stage_receipts,
            deploy_phase=deploy_phase,
        )
        raise
    if remote_result.stdout:
        print(remote_result.stdout, end="")
    if remote_result.stderr:
        print(remote_result.stderr, file=sys.stderr, end="")

    if _remote_verified_worker_release(remote_result.stdout, release_id):
        _activate_or_rollback_worker_release_over_ssh(
            ssh_cmd=ssh_cmd,
            releases_root=DEFAULT_WORKER_RELEASES_ROOT,
            release_id=release_id,
            worker_source_root=worker_source_root,
            stage_receipt_paths=contract_stage_receipts,
        )
    else:
        log("Worker code remains an unactivated candidate pending secure cutover")

    worker_urls = _verified_receipt_worker_urls(
        remote_result.stdout,
        release_id=release_id,
        public_urls=contract_public_urls,
    )
    if worker_urls:
        _validate_worker_public_url_env(
            [
                *[
                    item
                    for item in env_vars
                    if item.startswith("WORKER_API_AUTH_MODE=")
                ],
                f"WORKER_PUBLIC_URLS={','.join(worker_urls)}",
            ]
        )
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


def _wait_for_ssh(ip: str, port: int, identity: Path, timeout: int) -> bool:
    """Poll until SSH accepts connections. Returns True on success."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "GlobalKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
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
    _preflight_complete_worker_contract(
        _resolve_worker_security_env(args),
        worker_port=args.worker_port,
        expected_worker_count=1,
    )
    _prepare_worker_release_bundle(args)

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
        _read_env_value(args.backend_env, "GPU_WORKER_API_TOKEN")
        or _read_env_value(args.backend_env, "WORKER_API_TOKEN")
        or _read_env_value(args.backend_env, "RENDER_BROKER_WORKER_TOKEN")
        or os.getenv("GPU_WORKER_API_TOKEN")
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
    offer_gpu_count = _vast_offer_gpu_count(offer)
    if offer_gpu_count < 1:
        raise RuntimeError(
            "Selected Vast offer has no authoritative GPU count; refusing secure deploy"
        )
    requested_worker_count = int(getattr(args, "vast_worker_count", 0) or 0)
    effective_worker_count = requested_worker_count if requested_worker_count > 0 else offer_gpu_count
    effective_worker_count = max(1, effective_worker_count)
    _preflight_complete_worker_contract(
        _resolve_worker_security_env(args),
        worker_port=args.worker_port,
        expected_worker_count=effective_worker_count,
    )
    _prepare_worker_release_bundle(args)
    # When Vast's offer payload omits the GPU count, still expose the first two
    # worker ports so a 2-GPU host can use direct URLs after SSH confirms it.
    port_publish_count = effective_worker_count
    if requested_worker_count == 0:
        port_publish_count = max(port_publish_count, 2)

    port_publish_flags = " ".join(
        f"-p {args.worker_port + idx}:{args.worker_port + idx}"
        for idx in range(port_publish_count)
    )
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
            port_publish_flags,
        ]
    )
    # Inject our public key into authorized_keys at boot. Vast's `--ssh` relies on
    # syncing the *account* keys into the container, which can lag or silently fail
    # (observed 2026-07-10: a running box left the key unsynced for >13 min and
    # rejected every login). This onstart append makes SSH auth deterministic and
    # runs alongside the image's own startup (it does not replace the container
    # CMD/supervisord that boots ComfyUI). Guarded so a restart can't duplicate it.
    pubkey_path = identity.with_name(identity.name + ".pub")
    if pubkey_path.exists():
        pubkey = pubkey_path.read_text().strip()
        # sshd StrictModes refuses authorized_keys if /root or ~/.ssh is group/
        # world-writable — the vastai/comfy image ships /root writable, so sshd
        # rejected BOTH Vast's account key and ours ("bad ownership or modes",
        # observed 2026-07-10). Fix ownership + strip group/other write on the
        # whole chain, not just the file.
        onstart_cmd = (
            "mkdir -p /root/.ssh && "
            "chmod go-w /root && chmod 700 /root/.ssh && "
            f"(grep -qF '{pubkey}' /root/.ssh/authorized_keys 2>/dev/null || "
            f"echo '{pubkey}' >> /root/.ssh/authorized_keys) && "
            "chown -R root:root /root/.ssh && chmod 600 /root/.ssh/authorized_keys"
        )
        create_args.extend(["--onstart-cmd", onstart_cmd])
    else:
        log(f"WARNING: public key {pubkey_path} not found; relying on Vast account-key sync for SSH.")
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

    # Start polling for direct port mappings in parallel with SSH wait.
    import threading
    direct_url_results: list[str | None] = [None] * port_publish_count

    def _poll_direct_url(idx: int) -> None:
        direct_url_results[idx] = _vast_direct_worker_url(
            instance_id, args.worker_port + idx, timeout_sec=args.vast_boot_timeout
        )

    poll_threads = [
        threading.Thread(target=_poll_direct_url, args=(idx,), daemon=True)
        for idx in range(port_publish_count)
    ]
    for thread in poll_threads:
        thread.start()

    args.ssh_command = _wait_for_vast_ssh_command(
        instance_id,
        identity=identity,
        timeout_sec=args.vast_boot_timeout,
    )

    remote_gpu_count = _vast_remote_gpu_count(args)
    if remote_gpu_count:
        if requested_worker_count > 0:
            effective_worker_count = min(requested_worker_count, remote_gpu_count)
            if effective_worker_count < requested_worker_count:
                log(
                    f"Requested {requested_worker_count} Vast workers, but the rented "
                    f"instance exposes {remote_gpu_count} GPU(s); using {effective_worker_count}."
                )
        else:
            effective_worker_count = remote_gpu_count
        effective_worker_count = max(1, effective_worker_count)
    args._vast_effective_worker_count = effective_worker_count
    if effective_worker_count > 1:
        log(f"Vast multi-GPU mode: starting {effective_worker_count} worker services on one instance.")
    else:
        log("Vast single-GPU mode: using legacy worker bootstrap.")

    # Give the poll threads a moment to finish if they haven't already.
    for thread in poll_threads:
        thread.join(timeout=30)
    if effective_worker_count > len(direct_url_results):
        direct_url_results.extend([None] * (effective_worker_count - len(direct_url_results)))
    direct_url_results = direct_url_results[:effective_worker_count]
    direct_urls = [url for url in direct_url_results if url]
    direct_url = direct_url_results[0] if direct_url_results else None

    if direct_urls:
        for idx, url in enumerate(direct_url_results):
            if url:
                log(f"Vast direct worker URL gpu{idx}: {url}")
        env_vars = list(getattr(args, "env_vars", []) or [])
        if effective_worker_count > 1 and not any(v.startswith("WORKER_PUBLIC_URLS=") for v in env_vars):
            env_vars.append(f"WORKER_PUBLIC_URLS={','.join(url or '' for url in direct_url_results)}")
            args.env_vars = env_vars
        elif direct_url and not any(v.startswith("WORKER_PUBLIC_URL=") for v in env_vars):
            env_vars.append(f"WORKER_PUBLIC_URL={direct_url}")
            args.env_vars = env_vars
    else:
        log(
            "Vast cleartext public worker URLs are disabled; using the "
            "TLS cloudflared worker endpoint."
        )

    exit_code, worker_url = _do_deploy(args)
    worker_urls = getattr(args, "_last_worker_urls", None) or ([worker_url] if worker_url else [])
    if exit_code == 0 and worker_urls and args.warm_asset_groups and not args.skip_warmup:
        for url in worker_urls:
            _run_worker_warmup(args, url)
    return exit_code


def _do_deploy(args: argparse.Namespace, pod_id: str | None = None) -> tuple[int, str | None]:
    """Core deploy: scp worker + run remote bootstrap + update env."""
    ssh_cmd, scp_cmd, destination = parse_ssh_command(args.ssh_command)
    ssh_cmd = add_default_identity(ssh_cmd, override=args.ssh_identity)
    scp_cmd = add_default_identity(scp_cmd, override=args.ssh_identity)
    ssh_cmd = add_default_host_key_policy(ssh_cmd)
    scp_cmd = add_default_host_key_policy(scp_cmd)

    # Resolve and validate the *whole* worker/tunnel/backend contract before the
    # first remote mkdir, package upload, dependency install, GPU probe, restart,
    # or provider-side setup. This is the H100 regression gate: a missing TLS
    # URL can no longer surface only after a tolerated git-pull conflict.
    env_vars = _resolve_worker_security_env(args)
    existing_keys = {v.split("=", 1)[0] for v in env_vars if "=" in v}
    requested_vast_workers = int(getattr(args, "vast_worker_count", 0) or 0)
    effective_vast_workers = int(
        getattr(args, "_vast_effective_worker_count", 0) or requested_vast_workers or 1
    )
    is_vast_multi = effective_vast_workers > 1
    worker_count_requested = int(getattr(args, "worker_count", -1))
    # Secure deployment always uses the systemd path, even for one GPU. The
    # legacy nohup path exposed credentials in argv and has no atomic profile
    # or process rollback contract.
    is_ssh_multi = True
    is_multi = is_vast_multi or is_ssh_multi
    if not pod_id and not getattr(args, "vast", False) and worker_count_requested < 1:
        raise RuntimeError(
            "Secure SSH deploy requires an explicit --worker-count before remote mutation"
        )
    auto_env_keys = [
        "FILMFORGE_BACKEND_URL",
        "WORKER_REGISTRATION_TOKEN",
        "WORKER_API_TOKEN",
        "GPU_WORKER_API_TOKEN",
        "WORKER_API_AUTH_MODE",
        "WORKER_INPUT_URL_ALLOWED_HOSTS",
        "WORKER_PUBLIC_URLS",
        "WORKER_TUNNEL_LOCAL_URLS",
        "WORKER_TUNNEL_UNITS",
        "WORKER_SECURITY_STAGE_RECEIPTS",
        "WORKER_DEPLOY_PHASE",
        "FILMFORGE_BACKEND_CLIENT_AUTH_MODE",
        "WORKER_CAPABILITIES",
    ]
    if not is_multi:
        auto_env_keys.append("RENDER_BROKER_WORKER_ID")
    for key in auto_env_keys:
        if key not in existing_keys:
            value = _read_env_value(args.backend_env, key)
            if value:
                env_vars.append(f"{key}={value}")
                log(f"Auto-injected {key} from backend .env")
    existing_keys = {v.split("=", 1)[0] for v in env_vars if "=" in v}
    expected_worker_count = (
        effective_vast_workers
        if getattr(args, "vast", False)
        else 1
        if pod_id
        else worker_count_requested
        if worker_count_requested > 0
        else None
    )
    security_contract = _preflight_complete_worker_contract(
        env_vars,
        worker_port=args.worker_port,
        expected_worker_count=expected_worker_count,
    )
    contract_public_urls = list(security_contract["public_urls"])
    contract_stage_receipts = list(security_contract["stage_receipts"])
    deploy_phase = str(security_contract["deploy_phase"])

    releases_root = f"{args.remote_root.rstrip('/')}/worker_releases"
    release_id, worker_source_root = _stage_worker_release_over_ssh(
        ssh_cmd=ssh_cmd,
        scp_cmd=scp_cmd,
        destination=destination,
        releases_root=releases_root,
        venv_path=f"{args.remote_root.rstrip('/')}/.venv",
        bundle=getattr(args, "_prepared_worker_release_bundle", None),
    )
    env_vars.append(f"WORKER_CODE_RELEASE_ID={release_id}")

    try:

        if is_vast_multi:
            script = vast_multi_gpu_script(
                remote_root=args.remote_root,
                worker_port=args.worker_port,
                comfy_port=args.vast_comfy_port,
                worker_count=effective_vast_workers,
                worker_source_root=worker_source_root,
            )
        elif is_ssh_multi:
            # worker_count_requested == -1 means auto-detect: pass 0 to the script
            # so it uses the box's PHYSICAL_GPU_COUNT.
            script_worker_count = max(worker_count_requested, 0)
            if "WORKER_PUBLIC_URLS" not in existing_keys:
                log(
                    "SSH multi-GPU: no preset TLS worker URLs; remote bootstrap "
                    "will create cloudflared endpoints."
                )
            log(
                "SSH multi-GPU deploy: using systemd multi-GPU script "
                f"(worker_count={'auto' if script_worker_count == 0 else script_worker_count})"
            )
            script = vast_multi_gpu_script(
                remote_root=args.remote_root,
                worker_port=args.worker_port,
                comfy_port=args.vast_comfy_port,
                worker_count=script_worker_count,
                worker_source_root=worker_source_root,
            )
        else:
            script = remote_script(
                args.remote_root,
                args.worker_port,
                worker_source_root=worker_source_root,
            )
        if getattr(args, "qwen_sidecar", False):
            # One GPU = one department: the Qwen vision sidecar pins ~25% of VRAM
            # and starved the render models (an OOM cause), so it must not share a
            # generation GPU. If this box carries any render capability, refuse —
            # give vision its own box.
            _caps_str = next(
                (v.split("=", 1)[1] for v in env_vars if v.startswith("WORKER_CAPABILITIES=")),
                os.getenv("WORKER_CAPABILITIES", "flux2_stills,wan_i2v,ltx_i2v,character_loras"),
            )
            _gen = {"flux2_stills", "wan_i2v", "ltx_i2v"}
            if _gen & {c.strip() for c in _caps_str.split(",")}:
                raise SystemExit(
                    "--qwen-sidecar cannot share a generation GPU (one box = one "
                    "department). Deploy Qwen vision on its own box, or set "
                    "WORKER_CAPABILITIES to a non-render set."
                )
            env_vars.append("ENABLE_QWEN_SIDECAR=true")
            env_vars.append(f"QWEN_MODEL={args.qwen_model}")
            env_vars.append(f"QWEN_GPU_FRACTION={args.qwen_gpu_fraction}")
            log("Qwen3-VL vision sidecar enabled for this deploy")
        _validate_worker_public_url_env(env_vars)
        env_exports = build_bootstrap_env_exports(env_vars)
        if env_exports:
            script = f"{env_exports}\n\n{script}"
        remote_result = run(
            [*ssh_cmd, "bash", "-s"],
            input_text=script,
            capture_output=True,
        )
    except Exception as exc:
        if getattr(exc, "stdout", None):
            print(exc.stdout, end="")
        if getattr(exc, "stderr", None):
            print(exc.stderr, file=sys.stderr, end="")
        _rollback_failed_worker_transaction_over_ssh(
            ssh_cmd=ssh_cmd,
            releases_root=releases_root,
            worker_source_root=worker_source_root,
            failed_release_id=release_id,
            stage_receipt_paths=contract_stage_receipts,
            deploy_phase=deploy_phase,
        )
        return_code = int(getattr(exc, "returncode", 1) or 1)
        log(f"Remote bootstrap failed with exit code {return_code}")
        return return_code, None

    if remote_result.stdout:
        print(remote_result.stdout, end="")
    if remote_result.stderr:
        print(remote_result.stderr, file=sys.stderr, end="")

    if _remote_verified_worker_release(remote_result.stdout, release_id):
        _activate_or_rollback_worker_release_over_ssh(
            ssh_cmd=ssh_cmd,
            releases_root=releases_root,
            release_id=release_id,
            worker_source_root=worker_source_root,
            stage_receipt_paths=contract_stage_receipts,
        )
    else:
        log("Worker code remains an unactivated candidate pending secure cutover")

    worker_urls = _verified_receipt_worker_urls(
        remote_result.stdout,
        release_id=release_id,
        public_urls=contract_public_urls,
    )
    if worker_urls:
        _validate_worker_public_url_env(
            [
                *[
                    item
                    for item in env_vars
                    if item.startswith("WORKER_API_AUTH_MODE=")
                ],
                f"WORKER_PUBLIC_URLS={','.join(worker_urls)}",
            ]
        )
        log("Worker URLs:")
        for url in worker_urls:
            log(f"  {url}")
        setattr(args, "_last_worker_urls", worker_urls)
    worker_url = worker_urls[0] if worker_urls else None
    if worker_url:
        _validate_worker_public_url_env(
            [
                *[
                    item
                    for item in env_vars
                    if item.startswith("WORKER_API_AUTH_MODE=")
                ],
                f"WORKER_PUBLIC_URL={worker_url}",
            ]
        )

    qwen_url = extract_qwen_url(remote_result.stdout)
    if qwen_url:
        log(f"Qwen vision sidecar URL: {qwen_url}")

    if worker_url:
        log(
            "Worker URL is a candidate only; this deploy does not mutate or restart "
            "the backend before receipt-gated cutover"
        )

    # Save SSH dest so --logs can reconnect without re-specifying the host
    ssh_dest = args.ssh_command
    if ssh_dest.startswith("ssh "):
        ssh_dest = ssh_dest[4:]
    (SCRIPT_DIR / ".last_ssh_dest").write_text(ssh_dest + "\n")

    return 0, worker_url or None


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Python 3.13 does not allow positional args inside mutually_exclusive_group.
    # ssh_command is optional/positional; the flags below are the other modes.
    # Manual conflict check is done after parse_args() below.
    parser.add_argument(
        "ssh_command",
        nargs="?",
        default=None,
        help="Full SSH command, e.g.: ssh -i ~/.ssh/id_ed25519 -p 22981 root@61.206.39.5",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
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
        "--vast-offer-id",
        default=None,
        help="Rent this exact Vast offer id (bypasses search heuristics).",
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
        "--vast-country",
        type=lambda s: [x.strip().upper() for x in s.split(",") if x.strip()],
        default=DEFAULT_VAST_COUNTRIES,
        help=(
            "Comma-separated ISO country codes to restrict Vast offers to "
            "(reliable regions). Pass an empty string to disable the filter. "
            f"Default: {','.join(DEFAULT_VAST_COUNTRIES)}"
        ),
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
        "--vast-comfy-port",
        type=int,
        default=int(os.getenv("VAST_COMFY_PORT", "18188")),
        help="Base ComfyUI port for Vast multi-GPU workers. Default: 18188",
    )
    parser.add_argument(
        "--vast-worker-count",
        type=int,
        default=int(os.getenv("VAST_WORKER_COUNT", "0")),
        help=(
            "Number of worker services to start on a Vast instance. "
            "0 auto-detects from the selected offer; 1 keeps the legacy single-worker flow."
        ),
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
        "--verda-existing-instance-id",
        default="",
        help=(
            "Internal secure-state-machine resume target. Later phases reuse "
            "this exact VM instead of creating a second paid instance."
        ),
    )
    parser.add_argument(
        "--secure-one-click",
        action="store_true",
        help=(
            "Run the complete first-install state machine automatically: Fly "
            "bearer secrets, Vercel DNS, Caddy TLS, provision-only, receipt "
            "cutover, activation, and rollback. Verda only."
        ),
    )
    parser.add_argument(
        "--secure-resume-existing-fly-contract",
        action="store_true",
        help=(
            "Recover an interrupted secure first install without mutating Fly: "
            "verify the already-deployed secret names and cutover credential, "
            "and leave production GPU dispatch disabled. Requires --secure-one-click."
        ),
    )
    parser.add_argument(
        "--worker-edge-hostname",
        default=os.getenv("WORKER_EDGE_HOSTNAME", "gpu-worker.anapana.ai"),
        help="Stable DNS hostname for the one-click Caddy worker edge.",
    )
    parser.add_argument(
        "--worker-edge-domain",
        default=os.getenv("WORKER_EDGE_DOMAIN", "anapana.ai"),
        help="Vercel-managed DNS zone for the one-click worker edge.",
    )
    parser.add_argument(
        "--fly-app",
        default=os.getenv("FILMFORGE_FLY_APP", "filmforgepythonbackend"),
        help="Fly backend app that receives worker registration and cutover proof.",
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
        help="Number of GPU worker services to start, one per GPU (each pinned to its own CUDA device). 0 (default) auto-detects and uses every physical GPU on the box, so a multi-GPU box uses all its GPUs. Set a lower value to leave some GPUs unused. Per-GPU inference concurrency is a separate setting, not this.",
    )
    parser.add_argument(
        "--verda-worker-plan",
        default=os.getenv("VERDA_WORKER_PLAN", ""),
        help=(
            "Per-GPU department plan for a multi-GPU box, comma-separated, index = GPU index. "
            "Valid entries: " + ", ".join(WORKER_PLAN_DEPARTMENTS) + ". "
            "Example: '" + ",".join(DEFAULT_4GPU_WORKER_PLAN) + "' puts FLUX/WAN on GPUs 0-1, "
            "the Qwen3-VL vLLM server on GPU 2 and the Parler+SA3 sound stage on GPU 3, all sharing "
            "the one /mnt/data volume. Sizes the box (overrides --verda-worker-count). "
            "Empty (default) = every worker gets the same WORKER_CAPABILITIES."
        ),
    )
    parser.add_argument(
        "--verda-vllm-port",
        type=int,
        default=int(os.getenv("VERDA_VLLM_PORT", "8100")),
        help="Port for the resident vLLM vision server when the plan has a 'vision' GPU. Default: 8100",
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
        help=(
            "Deprecated compatibility option; worker code is deployed from the "
            "local content-addressed release package and no remote worker checkout is pulled."
        ),
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
        "--verda-install-timeout",
        type=int,
        default=int(os.getenv("VERDA_INSTALL_TIMEOUT_SEC", "3600")),
        help="Maximum seconds for each Verda remote install/start command. Default: 3600",
    )
    parser.add_argument(
        "--remote-root",
        default=DEFAULT_REMOTE_ROOT,
        help=f"Remote install root. Default: {DEFAULT_REMOTE_ROOT}",
    )
    parser.add_argument(
        "--qwen-sidecar",
        action="store_true",
        help="Also run a Qwen3-VL vision server (vLLM) alongside ComfyUI on the worker "
             "GPU, exposed via its own cloudflared tunnel and written to the backend's "
             "QWEN_BASE_URL. Requires docker on the worker box.",
    )
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Served model for the Qwen vision sidecar. Default: Qwen/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument(
        "--qwen-gpu-fraction",
        default="0.25",
        help="Fraction of GPU VRAM the vision sidecar may use, so it does not starve "
             "the render models. Default: 0.25 (~20GB on an 80GB A100).",
    )
    parser.add_argument(
        "--worker-port",
        type=int,
        default=9000,
        help="Remote local port for the GPU worker. Default: 9000",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=int(os.getenv("WORKER_COUNT", "-1")),
        help=(
            "Number of GPU workers to provision over an SSH deploy. "
            "-1 (default) auto-detects the box's physical GPUs and uses the "
            "multi-GPU systemd script when more than one GPU is present; "
            "1 forces the legacy single-worker flow."
        ),
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
        help=(
            "Non-credential environment variable for remote bootstrap. "
            "Credential-shaped keys are rejected; use --env-file."
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Mode-0600 deployment env file; keeps credentials out of process argv.",
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

    for item in args.env_vars:
        key = item.partition("=")[0].strip()
        if _CREDENTIAL_ENV_KEY.search(key):
            parser.error(
                f"credential key {key} is forbidden in --env; use --env-file"
            )
    if args.env_file:
        try:
            args.env_vars.extend(_load_worker_deploy_env_file(args.env_file))
        except RuntimeError as exc:
            parser.error(str(exc))

    # Manual mutual-exclusion check (Python 3.13 disallows positional in exclusive group)
    flags = [args.runpod, args.vast, args.verda, args.verda_fresh]
    if args.ssh_command and any(flags):
        parser.error("ssh_command cannot be used together with --runpod/--vast/--verda/--verda-fresh")
    if not args.ssh_command and not any(flags):
        parser.error("one of ssh_command, --runpod, --vast, --verda, or --verda-fresh is required")

    return args


def main() -> int:
    args = parse_args()

    try:
        if args.secure_resume_existing_fly_contract and not args.secure_one_click:
            raise RuntimeError(
                "--secure-resume-existing-fly-contract requires --secure-one-click"
            )
        if args.secure_one_click:
            if not (args.verda or args.verda_fresh):
                raise RuntimeError("--secure-one-click currently supports Verda only")
            if __package__:
                from gpu_worker.secure_one_click import run_secure_verda_first_install
            else:
                from secure_one_click import run_secure_verda_first_install

            return run_secure_verda_first_install(args, deploy_api=sys.modules[__name__])
        if args.runpod:
            return runpod_deploy(args)
        if args.vast:
            return vast_deploy(args)
        if args.verda:
            return verda_deploy_with_spot_retries(args)
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
