"""Automatic, receipt-gated first installation for one Verda GPU worker.

This module is the server-side implementation behind FilmForge's Rent GPU
button.  It deliberately keeps provider spend after every locally provable
precondition, then drives the existing worker release state machine in its
documented order:

``stage-code -> DNS -> stage/prepare edge -> provision-only -> cutover -> activate``.

The first implementation supports one worker per VM.  A shared TLS edge for
multiple worker ports needs a distinct fleet contract; silently pretending the
single-hostname Caddy profile supports that would recreate the partial-deploy
incident this code is intended to prevent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import ssl
import stat
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


CADDY_VERSION = "2.11.4"
CADDY_ARCHIVE_SHA256 = "527fbf917c39189a1e3b31d34fa955601680b2d5c8055d2a87b8b9588dec7bb9"
CADDY_ARCHIVE_URL = (
    "https://github.com/caddyserver/caddy/releases/download/"
    f"v{CADDY_VERSION}/caddy_{CADDY_VERSION}_linux_amd64.tar.gz"
)
DEFAULT_FLY_BACKEND_URL = "https://filmforgepythonbackend.fly.dev"
DEFAULT_PROFILE_ROOT = "/etc/filmforge/worker-security"
DEFAULT_STAGING_ROOT = "/etc/filmforge/one-click-staging"
MAX_CADDY_ARCHIVE_BYTES = 64 * 1024 * 1024
_SAFE_HOSTNAME = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class OneClickDeploymentError(RuntimeError):
    """The automatic deployment could not prove a safe next transition."""


class DeployApi(Protocol):
    def _prepare_worker_release_bundle(self, args: Any) -> Any: ...

    def verda_deploy(self, args: Any) -> int: ...

    def verda_fresh_deploy(self, args: Any) -> int: ...


class CommandRunner:
    """Small injectable subprocess boundary used by provider-free tests."""

    def run(
        self,
        command: Sequence[str],
        *,
        input_text: str | None = None,
        timeout: int = 120,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(command),
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=check,
            env={**os.environ, "NO_COLOR": "1"},
        )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


@dataclass(frozen=True)
class DeploymentSecrets:
    worker_api_token: str
    registration_token: str
    cutover_probe_token: str
    backend_url: str


@dataclass(frozen=True)
class DnsRecord:
    record_id: str
    name: str
    record_type: str
    value: str


@dataclass
class DnsMutation:
    previous: DnsRecord | None
    created: DnsRecord | None = None
    changed: bool = False


def _validated_hostname(hostname: str, domain: str) -> tuple[str, str]:
    normalized_host = str(hostname or "").strip().lower().rstrip(".")
    normalized_domain = str(domain or "").strip().lower().rstrip(".")
    if not _SAFE_HOSTNAME.fullmatch(normalized_host):
        raise OneClickDeploymentError("Worker edge hostname is invalid")
    if not _SAFE_HOSTNAME.fullmatch(normalized_domain):
        raise OneClickDeploymentError("Worker edge DNS domain is invalid")
    suffix = f".{normalized_domain}"
    if not normalized_host.endswith(suffix):
        raise OneClickDeploymentError("Worker edge hostname is outside its DNS domain")
    label = normalized_host[: -len(suffix)]
    if not label or "." in label:
        raise OneClickDeploymentError(
            "One-click v1 requires one direct DNS label below the configured domain"
        )
    return normalized_host, normalized_domain


def _strict_env(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise OneClickDeploymentError("Backend environment file is missing or unsafe")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise OneClickDeploymentError("Backend environment file must have mode 0600")
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            # Existing application env files may contain shell conveniences.
            # Ignore unrelated lines, but never accept them for deployment keys.
            continue
        if key in values and key in {
            "GPU_WORKER_API_TOKEN",
            "WORKER_API_TOKEN",
            "WORKER_REGISTRATION_TOKEN",
            "RENDER_BROKER_WORKER_TOKEN",
            "FILMFORGE_BACKEND_URL",
            "FILMFORGE_WORKER_CUTOVER_PROBE_TOKEN",
        }:
            raise OneClickDeploymentError(
                f"Backend environment contains duplicate key {key} at line {line_number}"
            )
        values[key] = value.strip()
    return values


def _atomic_set_env(path: Path, assignments: dict[str, str]) -> None:
    """Update a protected env file without weakening its permissions."""

    existing = path.read_text().splitlines()
    remaining = dict(assignments)
    rendered: list[str] = []
    for raw in existing:
        key = raw.partition("=")[0].strip()
        if key in remaining:
            rendered.append(f"{key}={remaining.pop(key)}")
        else:
            rendered.append(raw)
    if rendered and rendered[-1] != "":
        rendered.append("")
    rendered.extend(f"{key}={value}" for key, value in remaining.items())
    content = "\n".join(rendered).rstrip("\n") + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.one-click")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_or_create_secrets(backend_env: Path) -> DeploymentSecrets:
    values = _strict_env(backend_env)
    worker_token = (
        values.get("GPU_WORKER_API_TOKEN")
        or values.get("WORKER_API_TOKEN")
        or ""
    ).strip()
    registration_token = (
        values.get("WORKER_REGISTRATION_TOKEN")
        or values.get("RENDER_BROKER_WORKER_TOKEN")
        or ""
    ).strip()
    backend_url = (
        values.get("FILMFORGE_BACKEND_URL") or DEFAULT_FLY_BACKEND_URL
    ).strip().rstrip("/")
    probe_token = values.get("FILMFORGE_WORKER_CUTOVER_PROBE_TOKEN", "").strip()
    if len(worker_token) < 32:
        raise OneClickDeploymentError(
            "GPU worker bearer is missing or too short in the protected backend env"
        )
    if len(registration_token) < 16:
        raise OneClickDeploymentError(
            "Worker registration token is missing or too short in the protected backend env"
        )
    parsed_backend = urlsplit(backend_url)
    if (
        parsed_backend.scheme != "https"
        or not parsed_backend.hostname
        or parsed_backend.path not in {"", "/"}
        or parsed_backend.query
        or parsed_backend.fragment
        or parsed_backend.username is not None
        or parsed_backend.password is not None
    ):
        raise OneClickDeploymentError("FilmForge backend URL must be an HTTPS origin")
    if not probe_token:
        probe_token = secrets.token_urlsafe(48)
        _atomic_set_env(
            backend_env,
            {"FILMFORGE_WORKER_CUTOVER_PROBE_TOKEN": probe_token},
        )
    if len(probe_token) < 32:
        raise OneClickDeploymentError("Worker cutover probe token is too short")
    return DeploymentSecrets(
        worker_api_token=worker_token,
        registration_token=registration_token,
        cutover_probe_token=probe_token,
        backend_url=backend_url,
    )


def _command_path(name: str, *, fallback: str | None = None) -> str:
    found = shutil.which(name)
    if found:
        return found
    if fallback and Path(fallback).is_file():
        return fallback
    raise OneClickDeploymentError(f"Required one-click command is unavailable: {name}")


def _json_output(result: subprocess.CompletedProcess[str], *, label: str) -> Any:
    try:
        return json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        raise OneClickDeploymentError(f"{label} returned invalid JSON") from None


class VercelDns:
    def __init__(
        self,
        *,
        runner: CommandRunner,
        hostname: str,
        domain: str,
        executable: str | None = None,
    ) -> None:
        self.runner = runner
        self.hostname, self.domain = _validated_hostname(hostname, domain)
        self.label = self.hostname[: -(len(self.domain) + 1)]
        self.executable = executable or _command_path(
            "vercel", fallback=str(Path.home() / ".local/bin/vercel")
        )

    @property
    def endpoint(self) -> str:
        return f"/v3/domains/{self.domain}/records"

    def _records(self) -> list[DnsRecord]:
        result = self.runner.run(
            [self.executable, "api", f"{self.endpoint}?limit=100", "--raw"],
            timeout=60,
        )
        raw = _json_output(result, label="Vercel DNS inventory")
        rows = raw.get("records") if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            raise OneClickDeploymentError("Vercel DNS inventory has an unexpected shape")
        matched: list[DnsRecord] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip().lower().rstrip(".")
            full_name = name if name.endswith(f".{self.domain}") else f"{name}.{self.domain}"
            if full_name != self.hostname:
                continue
            matched.append(
                DnsRecord(
                    record_id=str(row.get("id") or ""),
                    name=name,
                    record_type=str(row.get("type") or "").upper(),
                    value=str(row.get("value") or "").strip(),
                )
            )
        if len(matched) > 1:
            raise OneClickDeploymentError("Worker DNS hostname has multiple records")
        if matched and (
            matched[0].record_type != "A" or not matched[0].record_id
        ):
            raise OneClickDeploymentError(
                "Worker DNS hostname conflicts with a non-A or unidentifiable record"
            )
        return matched

    def preflight(self) -> DnsRecord | None:
        records = self._records()
        return records[0] if records else None

    def _delete(self, record: DnsRecord) -> None:
        self.runner.run(
            [
                self.executable,
                "api",
                f"{self.endpoint}/{record.record_id}",
                "-X",
                "DELETE",
                "--dangerously-skip-permissions",
                "--raw",
            ],
            timeout=60,
        )

    def _create(self, value: str) -> DnsRecord:
        result = self.runner.run(
            [
                self.executable,
                "api",
                self.endpoint,
                "-X",
                "POST",
                "-f",
                f"name={self.label}",
                "-f",
                "type=A",
                "-f",
                f"value={value}",
                "--raw",
            ],
            timeout=60,
        )
        raw = _json_output(result, label="Vercel DNS create")
        record = raw.get("record") if isinstance(raw, dict) and isinstance(raw.get("record"), dict) else raw
        if not isinstance(record, dict):
            raise OneClickDeploymentError("Vercel DNS create returned an invalid record")
        created = DnsRecord(
            record_id=str(record.get("id") or ""),
            name=str(record.get("name") or self.label),
            record_type=str(record.get("type") or "").upper(),
            value=str(record.get("value") or ""),
        )
        if not created.record_id or created.record_type != "A" or created.value != value:
            raise OneClickDeploymentError("Vercel DNS create did not return the requested A record")
        return created

    def point_to(self, ip: str, previous: DnsRecord | None) -> DnsMutation:
        try:
            socket.inet_aton(ip)
        except OSError:
            raise OneClickDeploymentError("Verda worker has no valid public IPv4 address") from None
        if previous and previous.value == ip:
            return DnsMutation(previous=previous, created=previous, changed=False)
        mutation = DnsMutation(previous=previous, changed=True)
        if previous:
            self._delete(previous)
        try:
            mutation.created = self._create(ip)
        except Exception:
            if previous:
                self._create(previous.value)
            raise
        return mutation

    def rollback(self, mutation: DnsMutation | None) -> None:
        if mutation is None or not mutation.changed:
            return
        current = self.preflight()
        if mutation.created and current and current.record_id == mutation.created.record_id:
            self._delete(current)
        elif current is not None:
            raise OneClickDeploymentError(
                "Worker DNS changed outside the one-click transaction; refusing rollback overwrite"
            )
        if mutation.previous:
            self._create(mutation.previous.value)


class FlyBackend:
    def __init__(
        self,
        *,
        runner: CommandRunner,
        app: str,
        backend_url: str,
        executable: str | None = None,
    ) -> None:
        self.runner = runner
        self.app = str(app or "").strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", self.app):
            raise OneClickDeploymentError("Fly application name is invalid")
        self.backend_url = backend_url.rstrip("/")
        self.executable = executable or _command_path("flyctl", fallback="/opt/homebrew/bin/flyctl")

    def preflight(self) -> None:
        result = self.runner.run(
            [self.executable, "status", "-a", self.app, "--json"],
            timeout=60,
        )
        value = _json_output(result, label="Fly application status")
        if not isinstance(value, dict):
            raise OneClickDeploymentError("Fly application status is unavailable")

    def sync_fail_closed_secrets(self, values: DeploymentSecrets) -> None:
        assignments = {
            "GPU_WORKER_API_TOKEN": values.worker_api_token,
            "GPU_WORKER_API_AUTH_MODE": "required",
            "WORKER_REGISTRATION_TOKEN": values.registration_token,
            "FILMFORGE_WORKER_CUTOVER_PROBE_TOKEN": values.cutover_probe_token,
            "GPU_WORKER_ENABLED": "false",
        }
        payload = "".join(f"{key}={value}\n" for key, value in assignments.items())
        self.runner.run(
            [self.executable, "secrets", "import", "-a", self.app],
            input_text=payload,
            timeout=600,
        )
        self._wait_health()
        self._assert_probe_token_gate()

    def enable_worker_dispatch(self) -> None:
        self.runner.run(
            [self.executable, "secrets", "import", "-a", self.app],
            input_text="GPU_WORKER_ENABLED=true\n",
            timeout=600,
        )
        self._wait_health()

    def disable_worker_dispatch(self) -> None:
        """Return Fly to a fail-closed no-dispatch state during rollback."""

        self.runner.run(
            [self.executable, "secrets", "import", "-a", self.app],
            input_text="GPU_WORKER_ENABLED=false\n",
            timeout=600,
        )
        self._wait_health()

    def _wait_health(self, timeout: int = 180) -> None:
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with opener.open(f"{self.backend_url}/healthz", timeout=10) as response:
                    if response.status == 200:
                        value = json.loads(response.read(65537))
                        if isinstance(value, dict) and value.get("ok") is True:
                            return
            except Exception:
                pass
            time.sleep(2)
        raise OneClickDeploymentError("Fly backend did not become healthy after secret sync")

    def _assert_probe_token_gate(self) -> None:
        body = json.dumps(
            {
                "schema": "filmforge.worker-cutover-probe.v1",
                "release_id": "preflight",
                "worker_code_release_id": "preflight",
                "worker_dependency_freeze_sha256": "0" * 64,
                "worker_public_url": "https://gpu-worker.invalid",
            }
        ).encode()
        request = Request(
            f"{self.backend_url}/api/internal/worker-cutover-probe",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        try:
            opener.open(request, timeout=15)
        except HTTPError as exc:
            if exc.code == 401:
                return
        except Exception:
            pass
        raise OneClickDeploymentError("Fly worker cutover endpoint is not fail-closed")


def _download_caddy(runner: CommandRunner) -> tempfile.TemporaryDirectory[str]:
    directory = tempfile.TemporaryDirectory(prefix="filmforge-caddy-")
    root = Path(directory.name)
    archive = root / "caddy.tar.gz"
    runner.run(
        [
            _command_path("curl"),
            "-fsSL",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--max-filesize",
            str(MAX_CADDY_ARCHIVE_BYTES),
            CADDY_ARCHIVE_URL,
            "-o",
            str(archive),
        ],
        timeout=180,
    )
    if not archive.is_file() or archive.stat().st_size > MAX_CADDY_ARCHIVE_BYTES:
        directory.cleanup()
        raise OneClickDeploymentError("Pinned Caddy archive is missing or oversized")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != CADDY_ARCHIVE_SHA256:
        directory.cleanup()
        raise OneClickDeploymentError("Pinned Caddy archive digest mismatch")
    with tarfile.open(archive, mode="r:gz") as bundle:
        candidates = [member for member in bundle.getmembers() if member.name == "caddy"]
        if len(candidates) != 1 or not candidates[0].isfile():
            directory.cleanup()
            raise OneClickDeploymentError("Pinned Caddy archive has an unsafe layout")
        stream = bundle.extractfile(candidates[0])
        if stream is None:
            directory.cleanup()
            raise OneClickDeploymentError("Pinned Caddy binary could not be extracted")
        binary = root / "caddy"
        descriptor = os.open(binary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o755)
        with os.fdopen(descriptor, "wb") as output:
            shutil.copyfileobj(stream, output)
        os.chmod(binary, 0o755)
    return directory


def _set_deploy_env(args: Any, key: str, value: str) -> None:
    current = list(getattr(args, "env_vars", []) or [])
    current = [item for item in current if item.partition("=")[0] != key]
    current.append(f"{key}={value}")
    args.env_vars = current


def _phase(args: Any, name: str) -> None:
    _set_deploy_env(args, "WORKER_DEPLOY_PHASE", name)


def _deploy_phase(args: Any, deploy_api: DeployApi) -> int:
    result = (
        deploy_api.verda_fresh_deploy(args)
        if bool(getattr(args, "verda_fresh", False))
        else deploy_api.verda_deploy(args)
    )
    if int(result) != 0:
        raise OneClickDeploymentError(f"Worker deploy phase failed with rc={result}")
    return int(result)


def _write_mode(path: Path, content: str, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(content)
    os.chmod(path, mode)


def _remote_run(
    runner: CommandRunner,
    ssh_cmd: Sequence[str],
    command: Sequence[str],
    *,
    input_text: str | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    return runner.run([*ssh_cmd, *command], input_text=input_text, timeout=timeout)


def _stage_profile_sources(
    *,
    args: Any,
    runner: CommandRunner,
    secrets_value: DeploymentSecrets,
    public_url: str,
    profile_id: str,
    caddy_binary: Path,
) -> tuple[str, str]:
    ssh_cmd = list(getattr(args, "_verda_ssh_cmd"))
    scp_cmd = list(getattr(args, "_verda_scp_cmd"))
    destination = str(getattr(args, "_verda_destination"))
    code_release_id = str(getattr(args, "_verda_worker_release_id"))
    worker_source = str(getattr(args, "_verda_worker_source_root"))
    candidate_root = str(Path(worker_source).parent)
    remote_staging = f"{DEFAULT_STAGING_ROOT}/{profile_id}"
    stage_receipt = f"{DEFAULT_PROFILE_ROOT}/releases/{profile_id}/stage-receipt.json"
    cutover_receipt = f"{DEFAULT_PROFILE_ROOT}/releases/{profile_id}/cutover-receipt.json"
    local_url = "http://127.0.0.1:9000"
    edge_unit = "filmforge-worker-edge-gpu0.service"
    probe_url = f"{secrets_value.backend_url}/api/internal/worker-cutover-probe"
    with tempfile.TemporaryDirectory(prefix="filmforge-one-click-profile-") as temporary:
        root = Path(temporary)
        os.chmod(root, 0o700)
        files = {
            "worker.env": (
                "\n".join(
                    (
                        f"GPU_WORKER_API_TOKEN={secrets_value.worker_api_token}",
                        f"WORKER_REGISTRATION_TOKEN={secrets_value.registration_token}",
                        f"FILMFORGE_BACKEND_URL={secrets_value.backend_url}",
                        "WORKER_API_AUTH_MODE=required",
                        f"WORKER_PUBLIC_URL={public_url}",
                    )
                )
                + "\n"
            ),
            "caddy.env": (
                f"FILMFORGE_CADDY_CONFIG_FILE={remote_staging}/Caddyfile\n"
                f"FILMFORGE_CADDY_LOCAL_URL={local_url}\n"
                f"FILMFORGE_CADDY_PUBLIC_URL={public_url}\n"
                f"FILMFORGE_CADDY_UNIT={edge_unit}\n"
            ),
            "backend-probe.env": (
                f"FILMFORGE_BACKEND_CUTOVER_PROBE_URL={probe_url}\n"
                f"FILMFORGE_BACKEND_CUTOVER_PROBE_TOKEN={secrets_value.cutover_probe_token}\n"
            ),
            "Caddyfile": (
                f"{urlsplit(public_url).hostname} {{\n"
                f"    reverse_proxy {local_url}\n"
                "}\n"
            ),
        }
        for name, content in files.items():
            _write_mode(root / name, content, 0o600)
        _remote_run(
            runner,
            ssh_cmd,
            ["install", "-d", "-m", "0700", remote_staging],
        )
        _remote_run(
            runner,
            ssh_cmd,
            ["install", "-d", "-m", "0755", "/opt/instance-tools/bin"],
        )
        for name in files:
            runner.run(
                [*scp_cmd, str(root / name), f"{destination}:{remote_staging}/{name}"],
                timeout=120,
            )
        runner.run(
            [*scp_cmd, str(caddy_binary), f"{destination}:/opt/instance-tools/bin/caddy"],
            timeout=120,
        )
        _remote_run(
            runner,
            ssh_cmd,
            [
                "chmod",
                "0600",
                f"{remote_staging}/worker.env",
                f"{remote_staging}/caddy.env",
                f"{remote_staging}/backend-probe.env",
                f"{remote_staging}/Caddyfile",
            ],
        )
        _remote_run(
            runner,
            ssh_cmd,
            ["chmod", "0755", "/opt/instance-tools/bin/caddy"],
        )
        manage = f"{worker_source}/manage_worker_release.py"
        python = f"{candidate_root}/.venv/bin/python"
        _remote_run(
            runner,
            ssh_cmd,
            [
                python,
                manage,
                "stage",
                "--edge-provider",
                "caddy",
                "--profile-mode",
                "first-install",
                "--release-id",
                profile_id,
                "--worker-code-release-id",
                code_release_id,
                "--worker-unit",
                "filmforge-worker-gpu0.service",
                "--tunnel-unit",
                edge_unit,
                "--worker-port",
                "9000",
                "--worker-public-url",
                public_url,
                "--tunnel-local-url",
                local_url,
                "--worker-exec",
                f"{candidate_root}/.venv/bin/python",
                "--worker-module-dir",
                candidate_root,
                "--worker-secret-source",
                f"{remote_staging}/worker.env",
                "--tunnel-secret-source",
                f"{remote_staging}/caddy.env",
                "--backend-probe-secret-source",
                f"{remote_staging}/backend-probe.env",
                "--tunnel-exec-source",
                f"{worker_source}/deploy/bin/filmforge-worker-caddy",
                "--tunnel-binary-source",
                "/opt/instance-tools/bin/caddy",
            ],
            timeout=300,
        )
        _remote_run(
            runner,
            ssh_cmd,
            [
                python,
                manage,
                "prepare",
                "--release-id",
                profile_id,
                "--receipt-template",
                cutover_receipt,
            ],
            timeout=300,
        )
    return stage_receipt, cutover_receipt


def _wait_for_tls_hostname(hostname: str, expected_ip: str, timeout: int = 240) -> None:
    deadline = time.monotonic() + timeout
    context = ssl.create_default_context()
    last_error = "not ready"
    while time.monotonic() < deadline:
        try:
            addresses = {
                row[4][0]
                for row in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
            }
            if expected_ip not in addresses:
                last_error = "DNS has not reached the new VM"
                time.sleep(3)
                continue
            with socket.create_connection((hostname, 443), timeout=10) as raw:
                with context.wrap_socket(raw, server_hostname=hostname) as tls:
                    certificate = tls.getpeercert()
                    ssl.match_hostname(certificate, hostname)
                    return
        except Exception as exc:
            last_error = type(exc).__name__
        time.sleep(3)
    raise OneClickDeploymentError(
        f"Caddy did not present a valid TLS certificate for the worker hostname ({last_error})"
    )


def _authorize_cutover_receipt(
    *,
    args: Any,
    runner: CommandRunner,
    receipt_path: str,
) -> None:
    script = r'''import json, os, pathlib, sys, time
p = pathlib.Path(sys.argv[1])
if p.is_symlink() or not p.is_file() or (p.stat().st_mode & 0o777) != 0o600:
    raise SystemExit("cutover receipt is missing or unsafe")
v = json.loads(p.read_text())
if v.get("profile_mode") != "first-install" or v.get("edge_provider") != "caddy":
    raise SystemExit("cutover receipt is not the expected first-install Caddy profile")
v["issued_at_epoch"] = int(time.time())
v["tunnel_ready"] = True
v["worker_secret_fingerprint_match"] = True
v["edge_tls_hostname_ready"] = True
v["backend_bearer_client_ready"] = False
v["backend_registration_ready"] = False
t = p.with_name("." + p.name + ".one-click")
fd = os.open(t, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as stream:
    json.dump(v, stream, sort_keys=True, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
os.chmod(t, 0o600); os.replace(t, p)
'''
    _remote_run(
        runner,
        list(getattr(args, "_verda_ssh_cmd")),
        ["python3", "-", receipt_path],
        input_text=script,
    )


def _remote_profile_operation(
    *,
    args: Any,
    runner: CommandRunner,
    profile_id: str,
    operation: str,
    receipt_path: str | None = None,
) -> None:
    worker_source = str(getattr(args, "_verda_worker_source_root"))
    candidate_root = str(Path(worker_source).parent)
    command = [
        f"{candidate_root}/.venv/bin/python",
        f"{worker_source}/manage_worker_release.py",
        operation,
        "--release-id",
        profile_id,
    ]
    if receipt_path is not None:
        command.extend(["--receipt", receipt_path])
    _remote_run(
        runner,
        list(getattr(args, "_verda_ssh_cmd")),
        command,
        timeout=360,
    )


def _finalize_local_backend_env(path: Path, public_url: str) -> None:
    _atomic_set_env(
        path,
        {
            "GPU_WORKER_ENABLED": "true",
            "GPU_WORKER_API_AUTH_MODE": "required",
            "GPU_WORKER_BASE_URL": public_url,
        },
    )


def _validate_one_worker(args: Any) -> None:
    plan = [item for item in str(getattr(args, "verda_worker_plan", "") or "").split(",") if item.strip()]
    count = len(plan) or int(getattr(args, "verda_worker_count", 0) or 0)
    if count != 1:
        raise OneClickDeploymentError(
            "Automatic secure deployment currently supports exactly one GPU worker per VM"
        )


def run_secure_verda_first_install(
    args: Any,
    *,
    deploy_api: DeployApi,
    runner: CommandRunner | None = None,
) -> int:
    """Drive one complete secure first-install transaction.

    This function intentionally emits no secret values.  The sole machine-
    readable success line is the verified HTTPS worker URL; the capacity
    manager consumes that only after ``activate`` returns successfully.
    """

    command_runner = runner or CommandRunner()
    _validate_one_worker(args)
    hostname, domain = _validated_hostname(
        getattr(args, "worker_edge_hostname", ""),
        getattr(args, "worker_edge_domain", ""),
    )
    public_url = f"https://{hostname}"
    backend_env = Path(getattr(args, "backend_env")).expanduser()
    print("[one-click] Preflight: immutable code, protected secrets, Fly, DNS, and Caddy")
    # Build and retain the exact committed package before any external mutation.
    deploy_api._prepare_worker_release_bundle(args)
    secrets_value = _load_or_create_secrets(backend_env)
    dns = VercelDns(runner=command_runner, hostname=hostname, domain=domain)
    previous_dns = dns.preflight()
    fly = FlyBackend(
        runner=command_runner,
        app=getattr(args, "fly_app", ""),
        backend_url=secrets_value.backend_url,
    )
    fly.preflight()
    caddy_directory = _download_caddy(command_runner)

    profile_id = f"auto-{int(time.time())}-{secrets.token_hex(4)}-gpu0"
    if not _SAFE_ID.fullmatch(profile_id):
        caddy_directory.cleanup()
        raise OneClickDeploymentError("Generated secure profile id is invalid")
    stage_receipt = f"{DEFAULT_PROFILE_ROOT}/releases/{profile_id}/stage-receipt.json"
    local_url = "http://127.0.0.1:9000"
    edge_unit = "filmforge-worker-edge-gpu0.service"
    for key, value in (
        ("WORKER_API_AUTH_MODE", "required"),
        ("FILMFORGE_BACKEND_CLIENT_AUTH_MODE", "bearer"),
        ("WORKER_PUBLIC_URLS", public_url),
        ("WORKER_TUNNEL_LOCAL_URLS", local_url),
        ("WORKER_TUNNEL_UNITS", edge_unit),
        ("WORKER_SECURITY_STAGE_RECEIPTS", stage_receipt),
        ("WORKER_EDGE_PROVIDER", "caddy"),
        ("FILMFORGE_BACKEND_URL", secrets_value.backend_url),
    ):
        _set_deploy_env(args, key, value)
    _phase(args, "stage-code")

    dns_mutation: DnsMutation | None = None
    profile_staged = False
    fly_secrets_synced = False
    try:
        # Fly is made bearer-ready and explicitly dispatch-disabled before the
        # paid VM exists.  This is a safe, no-GPU precondition.
        fly.sync_fail_closed_secrets(secrets_value)
        fly_secrets_synced = True
        print("[one-click] Starting one paid VM only after every preflight passed")
        _deploy_phase(args, deploy_api)
        instance_id = str(getattr(args, "_verda_active_instance_id", ""))
        instance_ip = str(getattr(args, "_verda_active_ip", ""))
        if not instance_id or not instance_ip:
            raise OneClickDeploymentError("Stage-code did not return an exact Verda instance")
        print(f"VERDA_INSTANCE_ID={instance_id}")
        args.verda_existing_instance_id = instance_id

        dns_mutation = dns.point_to(instance_ip, previous_dns)
        returned_stage_receipt, cutover_receipt = _stage_profile_sources(
            args=args,
            runner=command_runner,
            secrets_value=secrets_value,
            public_url=public_url,
            profile_id=profile_id,
            caddy_binary=Path(caddy_directory.name) / "caddy",
        )
        if returned_stage_receipt != stage_receipt:
            raise OneClickDeploymentError("Secure profile receipt path drifted")
        profile_staged = True
        _wait_for_tls_hostname(hostname, instance_ip)

        print("[one-click] TLS ready; installing GPU stack with worker still disabled")
        _phase(args, "provision-only")
        _deploy_phase(args, deploy_api)
        _wait_for_tls_hostname(hostname, instance_ip)
        _authorize_cutover_receipt(
            args=args,
            runner=command_runner,
            receipt_path=cutover_receipt,
        )
        _remote_profile_operation(
            args=args,
            runner=command_runner,
            profile_id=profile_id,
            operation="cutover",
            receipt_path=cutover_receipt,
        )

        print("[one-click] Authenticated cutover passed; activating immutable code")
        _phase(args, "activate")
        _deploy_phase(args, deploy_api)
        fly.enable_worker_dispatch()
        _finalize_local_backend_env(backend_env, public_url)
        print(f"WORKER_URLS={public_url}")
        print("[one-click] Complete: one secure worker is ready")
        return 0
    except Exception:
        if profile_staged:
            try:
                _remote_profile_operation(
                    args=args,
                    runner=command_runner,
                    profile_id=profile_id,
                    operation="rollback",
                )
            except Exception:
                print("[one-click] WARNING: remote profile rollback requires operator review")
        if dns_mutation is not None:
            try:
                dns.rollback(dns_mutation)
            except Exception:
                print("[one-click] WARNING: DNS rollback requires operator review")
        if fly_secrets_synced:
            try:
                fly.disable_worker_dispatch()
            except Exception:
                print("[one-click] WARNING: Fly dispatch rollback requires operator review")
        raise
    finally:
        caddy_directory.cleanup()


__all__ = [
    "CADDY_ARCHIVE_SHA256",
    "CADDY_ARCHIVE_URL",
    "CADDY_VERSION",
    "CommandRunner",
    "DnsMutation",
    "DnsRecord",
    "FlyBackend",
    "OneClickDeploymentError",
    "VercelDns",
    "run_secure_verda_first_install",
]
