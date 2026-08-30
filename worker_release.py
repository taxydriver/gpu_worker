"""Immutable worker packages and receipt-gated secure-profile cutovers.

This module deliberately separates two operations which used to be interleaved:

* a worker *code release* is a content-addressed archive installed under a
  versioned directory; it never edits or pulls a remote git checkout;
* a worker *network cutover* is a staged systemd profile.  The existing
  ``99-public-url-override.conf`` remains authoritative until an explicit,
  fresh readiness receipt proves that the tunnel and backend bearer client are
  ready together.

No credential value is rendered into a systemd unit, manifest, log message, or
repository file.  Credentials are copied from operator-provided mode-0600 env
files into the versioned release directory and referenced with EnvironmentFile.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
import uuid
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Protocol, Sequence
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


PROFILE_DROPIN_NAME = "20-filmforge-secure-profile.conf"
STAGED_GUARD_DROPIN_NAME = "00-filmforge-staged-guard.conf"
PUBLIC_OVERRIDE_NAME = "99-public-url-override.conf"
RECEIPT_SCHEMA = "filmforge.worker-secure-cutover.v1"
STAGE_SCHEMA = "filmforge.worker-secure-stage.v1"
BACKEND_PROBE_SCHEMA = "filmforge.worker-cutover-probe.v1"
MAX_RECEIPT_AGE_SECONDS = 15 * 60
MAX_BACKEND_PROBE_RESPONSE_BYTES = 64 * 1024
BACKEND_PROBE_REGISTRATION_RETRY_SECONDS = 150
BACKEND_PROBE_REGISTRATION_RETRY_INTERVAL_SECONDS = 2

_SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_SAFE_UNIT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.@-]*\.service$")
_ENV_KEY = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_PACKAGE_EXCLUDED_NAMES = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".last_ssh_dest",
    ".env",
}
_PACKAGE_SECRET_SUFFIXES = {
    ".env",
    ".key",
    ".pem",
    ".p12",
    ".pfx",
    ".jwk",
}
_PACKAGE_SECRET_NAME = re.compile(
    r"(?:^|[-_.])(credential|credentials|secret|secrets|token|tokens)(?:[-_.]|$)",
    re.IGNORECASE,
)


class WorkerReleaseError(RuntimeError):
    """A release/profile operation was unsafe or incomplete."""


@dataclass
class WorkerReleaseBundle:
    """A temporary, content-addressed worker source archive."""

    release_id: str
    source_sha256: str
    archive_sha256: str
    archive_path: Path
    git_commit: str | None
    tracked_manifest_sha256: str | None
    _temporary_directory: tempfile.TemporaryDirectory[str]

    def cleanup(self) -> None:
        self._temporary_directory.cleanup()

    def __enter__(self) -> "WorkerReleaseBundle":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.cleanup()


@dataclass(frozen=True)
class SecureProfileLayout:
    """Filesystem roots used by the host-side secure-profile installer."""

    systemd_root: Path = Path("/etc/systemd/system")
    state_root: Path = Path("/etc/filmforge/worker-security")

    @property
    def releases_root(self) -> Path:
        return self.state_root / "releases"


@dataclass(frozen=True)
class SecureWorkerContract:
    """One indivisible worker/tunnel/auth deployment contract."""

    release_id: str
    worker_code_release_id: str
    worker_unit: str
    tunnel_unit: str
    worker_port: int
    worker_public_url: str
    tunnel_local_url: str
    worker_exec: Path
    worker_module_dir: Path
    worker_secret_source: Path
    tunnel_secret_source: Path
    backend_probe_secret_source: Path
    tunnel_exec_source: Path
    tunnel_binary_source: Path
    # ``cloudflared`` is the original named-tunnel profile.  ``caddy`` is a
    # direct, DNS-backed TLS edge.  Keep the tunnel-named fields for backwards
    # compatible callers; the on-disk receipt records the provider explicitly.
    edge_provider: str = "cloudflared"
    profile_mode: str = "migration"
    # ADR-0009 (one edge, N workers): number of per-GPU worker services behind
    # the single secure edge. The contract stays singular — one hostname, one
    # tunnel unit, one cutover proof — and workers 1..N-1 ride as additional
    # staged drop-ins (worker-secure-profile-gpu{i}.conf) that differ from
    # worker 0's only by loopback port and a path-suffixed WORKER_PUBLIC_URL.
    # Default 1 keeps every existing receipt and caller byte-identical.
    worker_count: int = 1


@dataclass(frozen=True)
class StagedSecureProfile:
    release_id: str
    release_dir: Path
    worker_dropin: Path
    tunnel_dropin: Path
    stage_receipt: Path


class ServiceController(Protocol):
    """Minimal host operations, injectable for deterministic tests."""

    def assert_active(self, unit: str) -> None: ...

    def assert_inactive(self, unit: str) -> None: ...

    def assert_disabled(self, unit: str) -> None: ...

    def assert_enabled(self, unit: str) -> None: ...

    def daemon_reload(self) -> None: ...

    def restart(self, unit: str) -> None: ...

    def enable(self, unit: str) -> None: ...

    def disable(self, unit: str) -> None: ...

    def stop(self, unit: str) -> None: ...

    def assert_loopback_only(self, port: int) -> None: ...

    def assert_public_listener(self, port: int) -> None: ...

    def assert_unit_loaded(
        self,
        unit: str,
        *,
        fragment_path: Path,
        dropin_paths: Sequence[Path],
    ) -> None: ...

    def assert_authenticated_backend_route(
        self,
        *,
        probe_url: str,
        probe_token: str,
        release_id: str,
        worker_code_release_id: str,
        worker_dependency_freeze_sha256: str,
        worker_public_url: str,
    ) -> None: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class SystemdServiceController:
    """Production controller used by the installed CLI."""

    def _run(
        self,
        args: Sequence[str],
        *,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(args),
            check=True,
            text=True,
            capture_output=capture_output,
        )

    def assert_active(self, unit: str) -> None:
        self._run(["systemctl", "is-active", "--quiet", unit])

    def assert_inactive(self, unit: str) -> None:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", unit],
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            raise WorkerReleaseError(
                "first-install profile refuses an already-active worker"
            )

    def assert_disabled(self, unit: str) -> None:
        result = subprocess.run(
            ["systemctl", "is-enabled", unit],
            check=False,
            text=True,
            capture_output=True,
        )
        state = result.stdout.strip()
        if state in {"enabled", "enabled-runtime"}:
            raise WorkerReleaseError(
                "first-install profile refuses an already-enabled service"
            )
        # ``stage`` deliberately installs the edge unit as an exact symlink.
        # systemd reports that unit-file state as ``alias`` (and returns zero),
        # even though it has no boot target Wants/Requires link.  Treat only
        # the two genuinely enabled states as enabled; keep unknown states
        # fail-closed so a systemd/runtime change cannot weaken this gate.
        if state not in {
            "alias",
            "disabled",
            "generated",
            "indirect",
            "linked",
            "linked-runtime",
            "masked",
            "masked-runtime",
            "not-found",
            "static",
            "transient",
        }:
            raise WorkerReleaseError(
                "first-install profile could not prove service is disabled"
            )

    def assert_enabled(self, unit: str) -> None:
        self._run(["systemctl", "is-enabled", "--quiet", unit])

    def daemon_reload(self) -> None:
        self._run(["systemctl", "daemon-reload"])

    def restart(self, unit: str) -> None:
        self._run(["systemctl", "restart", unit])

    def enable(self, unit: str) -> None:
        self._run(["systemctl", "enable", unit])

    def disable(self, unit: str) -> None:
        self._run(["systemctl", "disable", unit])

    def stop(self, unit: str) -> None:
        self._run(["systemctl", "stop", unit])

    def assert_unit_loaded(
        self,
        unit: str,
        *,
        fragment_path: Path,
        dropin_paths: Sequence[Path],
    ) -> None:
        result = self._run(
            [
                "systemctl",
                "show",
                unit,
                "--property=FragmentPath",
                "--property=DropInPaths",
            ],
            capture_output=True,
        )
        fields: dict[str, str] = {}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                fields[key] = value
        fragment_candidates = {str(fragment_path), str(fragment_path.resolve())}
        loaded_fragment = fields.get("FragmentPath")
        if loaded_fragment not in fragment_candidates:
            raise WorkerReleaseError("systemd did not load the exact managed unit")
        if Path(str(loaded_fragment)).name != unit:
            raise WorkerReleaseError(
                "systemd loaded the managed unit as an alias instead of its exact name"
            )
        loaded_dropins = fields.get("DropInPaths", "").split()
        if len(loaded_dropins) != len(dropin_paths):
            raise WorkerReleaseError("systemd loaded an unexpected unit drop-in set")
        unmatched = list(loaded_dropins)
        for expected_path in dropin_paths:
            candidates = {str(expected_path), str(expected_path.resolve())}
            match = next((item for item in unmatched if item in candidates), None)
            if match is None:
                raise WorkerReleaseError("systemd did not load the exact managed drop-ins")
            unmatched.remove(match)

    def assert_loopback_only(self, port: int) -> None:
        public_hosts = {"0.0.0.0", "*", "::", "[::]"}
        # uvicorn imports and completes its lifespan after systemd reports the
        # service started.  Wait for that bounded startup window, while still
        # rejecting a public or unexpected bind immediately if one appears.
        for attempt in range(150):
            result = self._run(["ss", "-H", "-ltn"], capture_output=True)
            listeners: list[str] = []
            for line in result.stdout.splitlines():
                fields = line.split()
                if len(fields) < 4:
                    continue
                local = fields[3]
                if local.rsplit(":", 1)[-1] == str(port):
                    listeners.append(local)
            if listeners:
                for local in listeners:
                    host = local.rsplit(":", 1)[0].strip("[]")
                    if host in public_hosts:
                        raise WorkerReleaseError(
                            f"worker port {port} remains publicly bound after cutover"
                        )
                    if host not in {"127.0.0.1", "::1"}:
                        raise WorkerReleaseError(
                            f"worker port {port} is bound to unexpected address"
                        )
                return
            if attempt < 149:
                time.sleep(0.1)
        raise WorkerReleaseError(
            f"worker port {port} has no listener after secure cutover"
        )

    def assert_public_listener(self, port: int) -> None:
        public_hosts = {"0.0.0.0", "*", "::", "[::]"}
        # systemctl considers the launcher started before Caddy has completed
        # config validation and bound its sockets.  Give that short, bounded
        # startup window time to settle instead of misclassifying a healthy
        # edge as absent immediately after restart.
        for attempt in range(50):
            result = self._run(["ss", "-H", "-ltn"], capture_output=True)
            listeners: list[str] = []
            for line in result.stdout.splitlines():
                fields = line.split()
                if len(fields) >= 4 and fields[3].rsplit(":", 1)[-1] == str(port):
                    listeners.append(fields[3])
            if any(
                local.rsplit(":", 1)[0].strip("[]") in public_hosts
                for local in listeners
            ):
                return
            if attempt < 49:
                time.sleep(0.1)
        raise WorkerReleaseError(f"edge port {port} has no public listener")

    def assert_authenticated_backend_route(
        self,
        *,
        probe_url: str,
        probe_token: str,
        release_id: str,
        worker_code_release_id: str,
        worker_dependency_freeze_sha256: str,
        worker_public_url: str,
    ) -> None:
        payload = json.dumps(
            {
                "schema": BACKEND_PROBE_SCHEMA,
                "release_id": release_id,
                "worker_code_release_id": worker_code_release_id,
                "worker_dependency_freeze_sha256": worker_dependency_freeze_sha256,
                "worker_public_url": worker_public_url,
            },
            sort_keys=True,
        ).encode("utf-8")
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        deadline = time.monotonic() + BACKEND_PROBE_REGISTRATION_RETRY_SECONDS
        while True:
            request = Request(
                probe_url,
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {probe_token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            try:
                with opener.open(request, timeout=30) as response:
                    raw = response.read(MAX_BACKEND_PROBE_RESPONSE_BYTES + 1)
                    if len(raw) > MAX_BACKEND_PROBE_RESPONSE_BYTES:
                        raise WorkerReleaseError(
                            "backend cutover probe response is too large"
                        )
                    value = json.loads(raw)
                break
            except HTTPError as exc:
                # A freshly restarted worker registers before its background
                # preload has rebuilt process-local checksum facts. The backend
                # deliberately returns 409 until the next capability-confirmed
                # heartbeat promotes that exact URL online. No other response
                # is transient: auth, contract, identity and route failures stay
                # immediate fail-closed errors.
                if exc.code != 409:
                    exc.close()
                    raise WorkerReleaseError("backend cutover probe failed") from exc
                exc.close()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WorkerReleaseError(
                        "backend cutover registration did not become ready"
                    ) from exc
                time.sleep(
                    min(BACKEND_PROBE_REGISTRATION_RETRY_INTERVAL_SECONDS, remaining)
                )
            except WorkerReleaseError:
                raise
            except Exception as exc:
                raise WorkerReleaseError("backend cutover probe failed") from exc
        expected = {
            "schema": BACKEND_PROBE_SCHEMA,
            "ok": True,
            "release_id": release_id,
            "worker_code_release_id": worker_code_release_id,
            "worker_dependency_freeze_sha256": worker_dependency_freeze_sha256,
            "worker_public_url": worker_public_url,
            "tunnel_ready": True,
            "worker_auth_ready": True,
            "registration_ready": True,
        }
        if not isinstance(value, dict) or any(value.get(key) != item for key, item in expected.items()):
            raise WorkerReleaseError(
                "backend cutover probe did not prove the authenticated route"
            )


@contextmanager
def _profile_lock(layout: SecureProfileLayout):
    layout.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = layout.state_root / ".profile.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.chmod(lock_path, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _locked_profile_operation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if "layout" in kwargs:
            layout = kwargs["layout"]
        elif len(args) >= 2 and isinstance(args[1], SecureProfileLayout):
            layout = args[1]
        else:
            layout = SecureProfileLayout()
        with _profile_lock(layout):
            return function(*args, **kwargs)

    return wrapped


def _is_package_path(path: Path, source_dir: Path) -> bool:
    relative = path.relative_to(source_dir)
    if any(part in _PACKAGE_EXCLUDED_NAMES for part in relative.parts):
        return False
    if path.name.endswith((".pyc", ".pyo")):
        return False
    # Never package an operator secret by accident. Examples are inert and are
    # the only credential-shaped files accepted by the release builder.
    if path.name.endswith(".example"):
        return True
    if path.suffix.lower() in _PACKAGE_SECRET_SUFFIXES:
        return False
    if path.name.startswith(".env."):
        return False
    if _PACKAGE_SECRET_NAME.search(path.name):
        return False
    return True


def _package_entries(source_dir: Path) -> list[Path]:
    entries: list[Path] = []
    for path in source_dir.rglob("*"):
        if not _is_package_path(path, source_dir):
            continue
        if path.is_symlink():
            target = path.resolve(strict=False)
            try:
                target.relative_to(source_dir)
            except ValueError:
                raise WorkerReleaseError(
                    f"release symlink escapes source tree: {path.relative_to(source_dir)}"
                ) from None
        if path.is_file() or path.is_symlink():
            entries.append(path)
    return sorted(entries, key=lambda item: item.relative_to(source_dir).as_posix())


def _tracked_package_entries(source_dir: Path) -> tuple[list[Path], str, str]:
    """Return the clean, tracked release manifest for a production deploy.

    The incident deploy copied a dirty checkout, so content addressing alone is
    not sufficient provenance: it would merely give unreviewed working-tree
    bytes a stable name. Production staging therefore packages only files in
    the git index and refuses any tracked or untracked change under the worker
    tree. Ignored operator secrets remain outside the manifest.
    """

    try:
        repository_root = Path(
            subprocess.run(
                ["git", "-C", str(source_dir), "rev-parse", "--show-toplevel"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
        ).resolve(strict=True)
        source_prefix = source_dir.relative_to(repository_root)
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                str(source_prefix) if source_prefix.parts else ".",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        if status.strip():
            raise WorkerReleaseError(
                "production worker release refuses dirty or untracked source files"
            )
        raw_manifest = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "ls-files",
                "-z",
                "--",
                str(source_prefix) if source_prefix.parts else ".",
            ],
            check=True,
            capture_output=True,
        ).stdout
        commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except WorkerReleaseError:
        raise
    except (OSError, subprocess.CalledProcessError, ValueError):
        raise WorkerReleaseError(
            "production worker release source must be a committed git tree"
        ) from None

    relative_names: list[str] = []
    entries: list[Path] = []
    for encoded_name in raw_manifest.split(b"\0"):
        if not encoded_name:
            continue
        repository_relative = Path(os.fsdecode(encoded_name))
        try:
            worker_relative = repository_relative.relative_to(source_prefix)
        except ValueError:
            continue
        path = source_dir / worker_relative
        if not _is_package_path(path, source_dir):
            continue
        if not (path.is_file() or path.is_symlink()):
            raise WorkerReleaseError(f"tracked release file is missing: {worker_relative}")
        if path.is_symlink():
            try:
                path.resolve(strict=False).relative_to(source_dir)
            except ValueError:
                raise WorkerReleaseError(
                    f"release symlink escapes source tree: {worker_relative}"
                ) from None
        relative_names.append(worker_relative.as_posix())
        entries.append(path)
    if not entries:
        raise WorkerReleaseError("production worker release manifest is empty")
    manifest_bytes = ("\n".join(sorted(relative_names)) + "\n").encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    return (
        sorted(entries, key=lambda item: item.relative_to(source_dir).as_posix()),
        commit,
        manifest_sha256,
    )


def _source_digest(source_dir: Path, entries: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in entries:
        relative = path.relative_to(source_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        if path.is_symlink():
            kind = b"L"
            normalized_mode = 0o777
            data = os.readlink(path).encode("utf-8")
        else:
            kind = b"F"
            # Installed releases are made read-only. Encode the read/execute
            # contract so an executable-bit change gets a new release id while
            # source-tree write bits do not fight the immutable target mode.
            normalized_mode = stat.S_IMODE(path.lstat().st_mode) & 0o555
            data = path.read_bytes()
        digest.update(kind)
        digest.update(normalized_mode.to_bytes(4, "big"))
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def build_worker_release_bundle(
    source_dir: Path,
    *,
    require_committed_source: bool = False,
) -> WorkerReleaseBundle:
    """Build a versioned archive without ``.git``, virtualenvs, or secrets."""

    source_dir = source_dir.resolve(strict=True)
    if not (source_dir / "app.py").is_file():
        raise WorkerReleaseError("worker release source is missing app.py")
    if not (source_dir / "requirements.txt").is_file():
        raise WorkerReleaseError("worker release source is missing requirements.txt")
    if not (source_dir / "requirements.lock").is_file():
        raise WorkerReleaseError("worker release source is missing requirements.lock")

    git_commit: str | None = None
    tracked_manifest_sha256: str | None = None
    if require_committed_source:
        entries, git_commit, tracked_manifest_sha256 = _tracked_package_entries(source_dir)
    else:
        entries = _package_entries(source_dir)
    source_sha256 = _source_digest(source_dir, entries)
    release_id = f"sha256-{source_sha256[:24]}"
    temporary_directory = tempfile.TemporaryDirectory(prefix="filmforge-worker-release-")
    archive_path = Path(temporary_directory.name) / f"{release_id}.tar.gz"
    try:
        with tarfile.open(archive_path, "w:gz", dereference=False) as archive:
            for path in entries:
                arcname = Path("gpu_worker") / path.relative_to(source_dir)
                archive.add(path, arcname=arcname, recursive=False)
        archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    except Exception:
        temporary_directory.cleanup()
        raise
    return WorkerReleaseBundle(
        release_id=release_id,
        source_sha256=source_sha256,
        archive_sha256=archive_sha256,
        archive_path=archive_path,
        git_commit=git_commit,
        tracked_manifest_sha256=tracked_manifest_sha256,
        _temporary_directory=temporary_directory,
    )


def worker_release_install_script(
    *,
    archive_path: str,
    archive_sha256: str,
    source_sha256: str,
    release_id: str,
    releases_root: str,
    venv_path: str,
    git_commit: str | None = None,
    tracked_manifest_sha256: str | None = None,
) -> str:
    """Return a fail-closed remote installer for one immutable code package."""

    for value, label in (
        (release_id, "release id"),
        (archive_sha256, "archive digest"),
        (source_sha256, "source digest"),
    ):
        if not _SAFE_ID.fullmatch(value):
            raise WorkerReleaseError(f"invalid {label}")
    if (git_commit is None) != (tracked_manifest_sha256 is None):
        raise WorkerReleaseError("git provenance fields must be supplied together")
    if git_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise WorkerReleaseError("invalid git commit provenance")
    if tracked_manifest_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", tracked_manifest_sha256
    ):
        raise WorkerReleaseError("invalid tracked manifest provenance")
    quoted = {  # shell arguments are data, never executable fragments
        "archive": _shell_quote(archive_path),
        "archive_sha": _shell_quote(archive_sha256),
        "source_sha": _shell_quote(source_sha256),
        "release": _shell_quote(release_id),
        "root": _shell_quote(releases_root),
        "venv": _shell_quote(venv_path),
        "git_commit": _shell_quote(git_commit or ""),
        "manifest_sha": _shell_quote(tracked_manifest_sha256 or ""),
    }
    return f"""#!/usr/bin/env bash
set -euo pipefail
umask 077

ARCHIVE={quoted['archive']}
EXPECTED_ARCHIVE_SHA256={quoted['archive_sha']}
EXPECTED_SOURCE_SHA256={quoted['source_sha']}
RELEASE_ID={quoted['release']}
RELEASES_ROOT={quoted['root']}
VENV_SEED={quoted['venv']}
EXPECTED_GIT_COMMIT={quoted['git_commit']}
EXPECTED_TRACKED_MANIFEST_SHA256={quoted['manifest_sha']}
TARGET="$RELEASES_ROOT/releases/$RELEASE_ID"
STAGING="$RELEASES_ROOT/releases/.${{RELEASE_ID}}.stage.$$"

test -f "$ARCHIVE" || {{ echo "worker release archive is missing" >&2; exit 1; }}
ACTUAL_ARCHIVE_SHA256="$(sha256sum "$ARCHIVE" | awk '{{print $1}}')"
test "$ACTUAL_ARCHIVE_SHA256" = "$EXPECTED_ARCHIVE_SHA256" || {{
  echo "worker release archive digest mismatch" >&2
  exit 1
}}

# Validate every member before extraction.  A content-addressed package may not
# escape its release directory through absolute paths, dot-dot, or link targets.
python3 - "$ARCHIVE" <<'PY'
import pathlib
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:gz") as archive:
    for member in archive.getmembers():
        path = pathlib.PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "gpu_worker":
            raise SystemExit("unsafe worker release archive member")
        if member.issym() or member.islnk():
            target = pathlib.PurePosixPath(member.linkname)
            if target.is_absolute() or ".." in target.parts:
                raise SystemExit("unsafe worker release archive link")
PY

mkdir -p "$RELEASES_ROOT/releases"
LOCK_FILE="$RELEASES_ROOT/.release.lock"
(umask 077; set -o noclobber; : > "$LOCK_FILE") 2>/dev/null || true
test -f "$LOCK_FILE" && ! test -L "$LOCK_FILE" || {{
  echo "worker release lock file is unsafe" >&2
  exit 1
}}
chmod 0600 "$LOCK_FILE"
exec 8<>"$LOCK_FILE"
python3 -c 'import fcntl; fcntl.flock(8, fcntl.LOCK_EX)' 8>&8
candidate_has_writable_paths() {{
  python3 - "$1" <<'PY'
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
raise SystemExit(
    0
    if any(
        not path.is_symlink() and stat.S_IMODE(path.lstat().st_mode) & 0o222
        for path in (root, *root.rglob("*"))
    )
    else 1
)
PY
}}
candidate_incomplete=0
if test -d "$TARGET"; then
  ready_mode="$(python3 -c 'import os, stat, sys; print(oct(stat.S_IMODE(os.stat(sys.argv[1]).st_mode))[2:])' "$TARGET/.ready" 2>/dev/null || true)"
  if ! test -f "$TARGET/.ready" || test "$ready_mode" != "444"; then
    echo "candidate incomplete: readiness marker missing or mode drifted (mode=$ready_mode)" >&2
    candidate_incomplete=1
  fi
  if candidate_has_writable_paths "$TARGET"; then
    echo "candidate incomplete: writable paths present" >&2
    candidate_incomplete=1
  fi
  if ! test -f "$TARGET/.dependency-freeze.txt" || \
      ! test -f "$TARGET/.dependency-freeze.sha256" || \
      test "$(cat "$TARGET/.dependency-freeze.sha256" 2>/dev/null || true)" != \
        "$(sha256sum "$TARGET/.dependency-freeze.txt" 2>/dev/null | awk '{{print $1}}')"; then
    echo "candidate incomplete: dependency snapshot missing or drifted" >&2
    candidate_incomplete=1
  fi
  if ! test -f "$TARGET/.source-sha256" || \
      test "$(cat "$TARGET/.source-sha256" 2>/dev/null || true)" != "$EXPECTED_SOURCE_SHA256"; then
    echo "candidate incomplete: source digest mismatch" >&2
    candidate_incomplete=1
  fi
  if test -n "$EXPECTED_GIT_COMMIT" && {{
      ! test -f "$TARGET/.git-commit" ||
      ! test -f "$TARGET/.tracked-manifest-sha256" ||
      test "$(cat "$TARGET/.git-commit" 2>/dev/null || true)" != "$EXPECTED_GIT_COMMIT" ||
      test "$(cat "$TARGET/.tracked-manifest-sha256" 2>/dev/null || true)" != "$EXPECTED_TRACKED_MANIFEST_SHA256"
    }}; then
    echo "candidate incomplete: git provenance mismatch" >&2
    candidate_incomplete=1
  fi
fi
if test "$candidate_incomplete" = "1"; then
  if test -L "$RELEASES_ROOT/current" && test "$(readlink "$RELEASES_ROOT/current")" = "$TARGET"; then
    echo "incomplete worker candidate is current; refusing cleanup" >&2
    exit 1
  fi
  if grep -R -F -l "$TARGET" \
      /etc/systemd/system \
      /etc/filmforge/worker-security >/dev/null 2>&1; then
    echo "incomplete worker candidate is referenced by systemd; refusing cleanup" >&2
    exit 1
  fi
  if grep -a -F -l "$TARGET" /proc/[0-9]*/cmdline >/dev/null 2>&1; then
    echo "incomplete worker candidate is still running; refusing cleanup" >&2
    exit 1
  fi
  chmod -R u+w "$TARGET" 2>/dev/null || true
  rm -rf "$TARGET"
fi
if test -d "$TARGET"; then
  test -f "$TARGET/.ready" || {{
    echo "existing worker candidate is incomplete; refusing to reuse it" >&2
    exit 1
  }}
  test "$(cat "$TARGET/.dependency-freeze.sha256" 2>/dev/null || true)" = \
    "$(sha256sum "$TARGET/.dependency-freeze.txt" 2>/dev/null | awk '{{print $1}}')" || {{
    echo "existing worker candidate dependency snapshot drifted" >&2
    exit 1
  }}
  test "$(python3 -c 'import os, stat, sys; print(oct(stat.S_IMODE(os.stat(sys.argv[1]).st_mode))[2:])' "$TARGET/.ready")" = "444" || {{
    echo "existing worker candidate readiness marker mode drifted" >&2
    exit 1
  }}
  if candidate_has_writable_paths "$TARGET"; then
    echo "existing worker candidate is writable" >&2
    exit 1
  fi
  test "$(cat "$TARGET/.source-sha256" 2>/dev/null || true)" = "$EXPECTED_SOURCE_SHA256" || {{
    echo "existing immutable release does not match requested digest" >&2
    exit 1
  }}
  if test -n "$EXPECTED_GIT_COMMIT"; then
    test "$(cat "$TARGET/.git-commit" 2>/dev/null || true)" = "$EXPECTED_GIT_COMMIT" || {{
      echo "existing worker candidate git provenance drifted" >&2
      exit 1
    }}
    test "$(cat "$TARGET/.tracked-manifest-sha256" 2>/dev/null || true)" = "$EXPECTED_TRACKED_MANIFEST_SHA256" || {{
      echo "existing worker candidate tracked manifest drifted" >&2
      exit 1
    }}
  fi
else
  rm -rf "$STAGING"
  mkdir -p "$STAGING"
  trap 'rm -rf "$STAGING"' EXIT
  tar -xzf "$ARCHIVE" --no-same-owner --same-permissions -C "$STAGING"
  test -f "$STAGING/gpu_worker/app.py" || {{ echo "worker release missing app.py" >&2; exit 1; }}
  test -f "$STAGING/gpu_worker/requirements.txt" || {{ echo "worker release missing requirements.txt" >&2; exit 1; }}
  test -f "$STAGING/gpu_worker/requirements.lock" || {{ echo "worker release missing requirements.lock" >&2; exit 1; }}
  printf '%s\n' "$EXPECTED_SOURCE_SHA256" > "$STAGING/.source-sha256"
  ln -s ../.venv "$STAGING/gpu_worker/.venv"
  python3 - "$STAGING/gpu_worker" <<'PY'
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
for path in sorted(root.rglob("*"), reverse=True):
    if path.is_symlink():
        continue
    mode = path.lstat().st_mode
    if path.is_dir():
        os.chmod(path, 0o555)
    elif path.is_file():
        os.chmod(path, 0o555 if mode & 0o111 else 0o444)
os.chmod(root, 0o555)
PY
  mv "$STAGING" "$TARGET"
  trap - EXIT
  # A raw provider image ships python3 without ensurepip; `python3 -m venv`
  # then creates a pipless venv and aborts asking for the python3-venv
  # package. Bootstrapped OS volumes already have it, making this a no-op.
  if ! python3 -m ensurepip --version >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >&2 || true
    apt-get install -y -qq python3-venv python3-pip >&2
  fi
  if test -x "$VENV_SEED/bin/python"; then
    "$VENV_SEED/bin/python" -m venv --copies "$TARGET/.venv"
  else
    python3 -m venv --copies "$TARGET/.venv"
  fi
  # requirements.lock is committed and part of the release digest. Resolving
  # unpinned requirements.txt on the target would make one release id produce
  # different runtimes on different days.
  "$TARGET/.venv/bin/python" -m pip install -r "$TARGET/gpu_worker/requirements.lock"
  "$TARGET/.venv/bin/python" -m pip freeze --all > "$TARGET/.dependency-freeze.txt"
  sha256sum "$TARGET/.dependency-freeze.txt" | awk '{{print $1}}' > "$TARGET/.dependency-freeze.sha256"
  if test -n "$EXPECTED_GIT_COMMIT"; then
    printf '%s\n' "$EXPECTED_GIT_COMMIT" > "$TARGET/.git-commit"
    printf '%s\n' "$EXPECTED_TRACKED_MANIFEST_SHA256" > "$TARGET/.tracked-manifest-sha256"
  fi
  printf '%s\n' "$EXPECTED_SOURCE_SHA256" > "$TARGET/.ready"
  chmod 0444 "$TARGET/.ready"
  python3 - "$TARGET" <<'PY'
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
    if path.is_symlink():
        continue
    mode = path.lstat().st_mode
    if path.is_dir():
        os.chmod(path, 0o555)
    elif path.is_file():
        os.chmod(path, 0o555 if mode & 0o111 else 0o444)
os.chmod(root, 0o555)
PY
fi

# Recompute the source digest even when the release id already exists. Root can
# alter read-only files, so trusting only .source-sha256 would make idempotent
# redeploy reuse a tampered directory.
ACTUAL_SOURCE_SHA256="$(python3 - "$TARGET/gpu_worker" <<'PY'
import hashlib
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
entries = sorted(
    (
        path
        for path in root.rglob("*")
        if ".venv" not in path.relative_to(root).parts
        and (path.is_file() or path.is_symlink())
    ),
    key=lambda path: path.relative_to(root).as_posix(),
)
digest = hashlib.sha256()
for path in entries:
    relative = path.relative_to(root).as_posix().encode("utf-8")
    digest.update(len(relative).to_bytes(4, "big"))
    digest.update(relative)
    if path.is_symlink():
        kind = b"L"
        normalized_mode = 0o777
        data = os.readlink(path).encode("utf-8")
    else:
        kind = b"F"
        normalized_mode = stat.S_IMODE(path.lstat().st_mode) & 0o555
        data = path.read_bytes()
    digest.update(kind)
    digest.update(normalized_mode.to_bytes(4, "big"))
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)
print(digest.hexdigest())
PY
)"
test "$ACTUAL_SOURCE_SHA256" = "$EXPECTED_SOURCE_SHA256" || {{
  echo "installed worker release content digest mismatch" >&2
  exit 1
}}

rm -f "$ARCHIVE"
printf 'WORKER_RELEASE_ID=%s\n' "$RELEASE_ID"
printf 'WORKER_RELEASE_CANDIDATE_ROOT=%s\n' "$TARGET/gpu_worker"
"""


def worker_release_activate_script(*, releases_root: str, release_id: str) -> str:
    """Promote a fully bootstrapped candidate; never run before worker health."""

    if not _SAFE_ID.fullmatch(release_id):
        raise WorkerReleaseError("invalid release id")
    return f"""#!/usr/bin/env bash
set -euo pipefail
RELEASES_ROOT={_shell_quote(releases_root)}
RELEASE_ID={_shell_quote(release_id)}
TARGET="$RELEASES_ROOT/releases/$RELEASE_ID"
CURRENT="$RELEASES_ROOT/current"
PREVIOUS="$RELEASES_ROOT/previous"
mkdir -p "$RELEASES_ROOT"
LOCK_FILE="$RELEASES_ROOT/.release.lock"
(umask 077; set -o noclobber; : > "$LOCK_FILE") 2>/dev/null || true
test -f "$LOCK_FILE" && ! test -L "$LOCK_FILE" || {{
  echo "worker release lock file is unsafe" >&2
  exit 1
}}
chmod 0600 "$LOCK_FILE"
exec 8<>"$LOCK_FILE"
python3 -c 'import fcntl; fcntl.flock(8, fcntl.LOCK_EX)' 8>&8
candidate_has_writable_paths() {{
  python3 - "$1" <<'PY'
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
raise SystemExit(
    0
    if any(
        not path.is_symlink() and stat.S_IMODE(path.lstat().st_mode) & 0o222
        for path in (root, *root.rglob("*"))
    )
    else 1
)
PY
}}
test -f "$TARGET/.ready" || {{ echo "worker candidate is not ready" >&2; exit 1; }}
test "$(cat "$TARGET/.dependency-freeze.sha256" 2>/dev/null || true)" = \
  "$(sha256sum "$TARGET/.dependency-freeze.txt" 2>/dev/null | awk '{{print $1}}')" || {{
  echo "worker candidate dependency snapshot drifted" >&2
  exit 1
}}
if candidate_has_writable_paths "$TARGET"; then
  echo "worker candidate became writable" >&2
  exit 1
fi
if test -L "$CURRENT" && test "$(basename "$(readlink "$CURRENT")")" = "$RELEASE_ID"; then
  printf 'WORKER_RELEASE_ACTIVE=%s\n' "$RELEASE_ID"
  exit 0
fi
if test -L "$CURRENT"; then
  old_target="$(readlink "$CURRENT")"
  ln -sfn "$old_target" "$PREVIOUS.next"
  mv -Tf "$PREVIOUS.next" "$PREVIOUS"
fi
ln -sfn "$TARGET" "$CURRENT.next"
mv -Tf "$CURRENT.next" "$CURRENT"
printf 'WORKER_RELEASE_ACTIVE=%s\n' "$RELEASE_ID"
"""


def worker_release_rollback_script(*, releases_root: str, failed_release_id: str) -> str:
    """Return a locked rollback that removes every failed candidate ExecStart."""

    if not _SAFE_ID.fullmatch(failed_release_id):
        raise WorkerReleaseError("invalid failed release id")
    return f"""#!/usr/bin/env bash
set -euo pipefail
RELEASES_ROOT={_shell_quote(releases_root)}
FAILED_RELEASE_ID={_shell_quote(failed_release_id)}
CURRENT="$RELEASES_ROOT/current"
PREVIOUS="$RELEASES_ROOT/previous"
FAILED_TARGET="$RELEASES_ROOT/releases/$FAILED_RELEASE_ID"
mkdir -p "$RELEASES_ROOT"
LOCK_FILE="$RELEASES_ROOT/.release.lock"
(umask 077; set -o noclobber; : > "$LOCK_FILE") 2>/dev/null || true
test -f "$LOCK_FILE" && ! test -L "$LOCK_FILE" || {{
  echo "worker release lock file is unsafe" >&2
  exit 1
}}
chmod 0600 "$LOCK_FILE"
exec 8<>"$LOCK_FILE"
python3 -c 'import fcntl; fcntl.flock(8, fcntl.LOCK_EX)' 8>&8
if test -L "$CURRENT" && test "$(basename "$(readlink "$CURRENT")")" = "$FAILED_RELEASE_ID"; then
  test -L "$PREVIOUS" || {{ echo "previous worker release is missing" >&2; exit 1; }}
  previous_target="$(readlink "$PREVIOUS")"
  test -f "$previous_target/.ready" || {{ echo "previous worker release is incomplete" >&2; exit 1; }}
  ln -sfn "$previous_target" "$CURRENT.rollback"
  mv -Tf "$CURRENT.rollback" "$CURRENT"
fi

# Any partially written base unit must stop naming the failed immutable path.
# Validate every target before replacement; secure-profile drop-ins are handled
# by their own CAS rollback and are deliberately outside this code transaction.
TOUCHED="$RELEASES_ROOT/.rollback-units-$FAILED_RELEASE_ID"
set +e
python3 - "$FAILED_TARGET" "$CURRENT" "$TOUCHED" <<'PY'
import glob
import os
import pathlib
import sys

failed = str(pathlib.Path(sys.argv[1]))
current_link = pathlib.Path(sys.argv[2])
touched_path = pathlib.Path(sys.argv[3])
replacement = str(current_link / "gpu_worker") if current_link.is_symlink() else ""
candidates = sorted(
    pathlib.Path(path)
    for path in glob.glob("/etc/systemd/system/filmforge-worker-gpu*.service")
)
plans = []
for path in candidates:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"unsafe worker unit during rollback: {{path}}")
    source = path.read_text()
    if failed not in source:
        continue
    plans.append((path, source.replace(failed + "/gpu_worker", replacement)))
if plans and not replacement:
    touched_path.write_text("\n".join(path.name for path, _ in plans) + "\n")
    raise SystemExit(75)
for path, rendered in plans:
    temporary = path.with_name(f".{{path.name}}.rollback.{{os.getpid()}}")
    temporary.write_text(rendered)
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)
touched_path.write_text("\n".join(path.name for path, _ in plans) + "\n")
PY
rollback_status="$?"
set -e
if test "$rollback_status" = "75"; then
  while IFS= read -r unit; do
    test -n "$unit" && systemctl stop "$unit" || true
  done < "$TOUCHED"
  echo "failed first-install candidate stopped; no previous release existed" >&2
elif test "$rollback_status" != "0"; then
  exit "$rollback_status"
else
  systemctl daemon-reload
  while IFS= read -r unit; do
    test -n "$unit" || continue
    if systemctl is-active --quiet "$unit"; then
      systemctl restart "$unit"
    fi
  done < "$TOUCHED"
fi
printf 'WORKER_RELEASE_ROLLBACK=%s\n' "$FAILED_RELEASE_ID"
"""


def _shell_quote(value: str) -> str:
    # Local copy of shlex.quote keeps this module import-light on minimal boxes.
    if value and re.fullmatch(r"[a-zA-Z0-9_@%+=:,./-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _strict_secret_env(path: Path) -> dict[str, str]:
    _strict_mode_0600_file(path, label="secret env")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            raise WorkerReleaseError(
                f"systemd EnvironmentFile does not accept export syntax at {path}:{line_number}"
            )
        key, separator, value = line.partition("=")
        if not separator or not _ENV_KEY.fullmatch(key):
            raise WorkerReleaseError(
                f"invalid secret env assignment at {path}:{line_number}"
            )
        if key in values:
            raise WorkerReleaseError(f"duplicate secret env key: {key}")
        if value.startswith(("'", '"')) and value.endswith(value[:1]) and len(value) >= 2:
            value = value[1:-1]
        if not value or "\x00" in value or "\n" in value or "\r" in value:
            raise WorkerReleaseError(f"secret env key is empty or invalid: {key}")
        if any(character.isspace() for character in value) or any(
            character in value for character in ("'", '"', "\\")
        ):
            raise WorkerReleaseError(
                f"secret env value is not systemd-safe for key: {key}"
            )
        values[key] = value
    return values


def _strict_mode_0600_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise WorkerReleaseError(f"{label} must be a regular file: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise WorkerReleaseError(f"{label} must have mode 0600: {path}")


def _validated_https_url(raw_url: str, *, label: str) -> str:
    try:
        parsed = urlsplit(raw_url)
        port = parsed.port
    except (TypeError, ValueError):
        raise WorkerReleaseError(f"{label} is invalid") from None
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or port is not None and not 1 <= port <= 65535
    ):
        raise WorkerReleaseError(f"{label} must be an origin-only HTTPS URL")
    return raw_url.rstrip("/")


def _validated_https_endpoint(raw_url: str, *, label: str) -> str:
    try:
        parsed = urlsplit(raw_url)
        port = parsed.port
    except (TypeError, ValueError):
        raise WorkerReleaseError(f"{label} is invalid") from None
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port is not None and not 1 <= port <= 65535
    ):
        raise WorkerReleaseError(f"{label} must be an HTTPS URL without credentials")
    return raw_url


def _validated_loopback_url(raw_url: str, *, port: int) -> str:
    try:
        parsed = urlsplit(raw_url)
        parsed_port = parsed.port
    except (TypeError, ValueError):
        raise WorkerReleaseError("tunnel local URL is invalid") from None
    if (
        parsed.scheme.lower() != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed_port != port
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise WorkerReleaseError(
            "tunnel local URL must target the exact loopback worker port"
        )
    return raw_url.rstrip("/")


def _validated_systemd_path(path: Path, *, label: str) -> None:
    rendered = str(path)
    if (
        not path.is_absolute()
        or not rendered
        or any(character.isspace() or ord(character) < 32 for character in rendered)
    ):
        raise WorkerReleaseError(f"{label} must be an absolute whitespace-free path")


def _render_secret_env(values: dict[str, str]) -> str:
    """Render the already-restricted values in systemd EnvironmentFile syntax."""

    return "".join(f"{key}={values[key]}\n" for key in sorted(values))


def _render_tunnel_config(
    source: Path,
    *,
    public_url: str,
    local_url: str,
    source_credential: Path,
    staged_credential: Path,
) -> str:
    """Validate a named-tunnel ingress and rewrite its credential atomically."""

    source_text = source.read_text()
    normalized = [
        raw_line.strip()
        for raw_line in source_text.splitlines()
        if raw_line.strip() and not raw_line.lstrip().startswith("#")
    ]
    hostname = urlsplit(public_url).hostname
    ingress: list[dict[str, str]] = []
    in_ingress = False
    current: dict[str, str] | None = None
    for raw_line in source_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        if stripped == "ingress:":
            in_ingress = True
            continue
        if not in_ingress:
            continue
        if indent == 0:
            break
        if stripped.startswith("- "):
            if current is not None:
                ingress.append(current)
            current = {}
            stripped = stripped[2:].strip()
        if current is not None and ":" in stripped:
            key, _, value = stripped.partition(":")
            current[key.strip()] = value.strip()
    if current is not None:
        ingress.append(current)
    if (
        ingress
        != [
            {"hostname": hostname or "", "service": local_url},
            {"service": "http_status:404"},
        ]
        or f"credentials-file: {source_credential}" not in normalized
        or not any(
        line.startswith("tunnel: ") and len(line.partition(":")[2].strip()) > 0
        for line in normalized
        )
    ):
        raise WorkerReleaseError(
            "tunnel config does not match the public/local/credential contract"
        )
    credential_pattern = re.compile(r"^(\s*credentials-file:\s*)(\S+)(\s*)$")
    rewritten: list[str] = []
    replacements = 0
    for raw_line in source_text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        ending = raw_line[len(line) :]
        match = credential_pattern.fullmatch(line)
        if match and match.group(2) == str(source_credential):
            line = f"{match.group(1)}{staged_credential}{match.group(3)}"
            replacements += 1
        rewritten.append(line + ending)
    if replacements != 1:
        raise WorkerReleaseError("tunnel config must contain one credential file")
    return "".join(rewritten)


_MAX_WORKERS_PER_EDGE = 8


def _validated_worker_count(worker_count: int, worker_unit: str) -> int:
    count = int(worker_count)
    if not 1 <= count <= _MAX_WORKERS_PER_EDGE:
        raise WorkerReleaseError(
            f"worker_count must be 1..{_MAX_WORKERS_PER_EDGE}, got {count}"
        )
    if count > 1 and "gpu0" not in worker_unit:
        raise WorkerReleaseError(
            "multi-worker contracts require a gpu0-indexed worker unit name"
        )
    return count


def _indexed_worker_units(worker_unit: str, worker_count: int) -> list[str]:
    """Sibling systemd units for workers 1..N-1, derived from the gpu0 unit."""
    count = _validated_worker_count(worker_count, worker_unit)
    return [worker_unit.replace("gpu0", f"gpu{i}") for i in range(1, count)]


def _indexed_public_url(public_url: str, idx: int) -> str:
    return f"{public_url.rstrip('/')}/gpu{idx}"


def _indexed_local_url(local_url: str, idx: int) -> str:
    parsed = urlsplit(local_url.rstrip("/"))
    if parsed.port is None:
        raise WorkerReleaseError("tunnel local URL must carry an explicit port")
    return f"{parsed.scheme}://{parsed.hostname}:{parsed.port + idx}"


def expected_caddy_config(*, public_url: str, local_url: str, worker_count: int = 1) -> str:
    """The one exact Caddyfile this profile accepts, for N workers behind one edge.

    Worker 0 stays the bare-hostname reverse proxy — a single-worker box
    produces the byte-identical config this validator has always required.
    Workers 1..N-1 are ``handle_path /gpu{i}/*`` blocks (ADR-0009): the prefix
    is stripped by Caddy, so every worker serves its API at ``/`` and never
    learns its own path. Ports ascend from worker 0's.
    """
    hostname = urlsplit(public_url).hostname
    local = local_url.rstrip("/")
    blocks = "".join(
        (
            f"    handle_path /gpu{i}/* {{\n"
            f"        reverse_proxy {_indexed_local_url(local, i)}\n"
            "    }\n"
        )
        for i in range(1, int(worker_count))
    )
    return (
        f"{hostname} {{\n"
        f"{blocks}"
        f"    reverse_proxy {local}\n"
        "}\n"
    )


def _render_caddy_config(
    source: Path, *, public_url: str, local_url: str, worker_count: int = 1
) -> str:
    """Accept only the small, deterministic Caddyfile used by this profile.

    This deliberately does not accept arbitrary snippets.  The immutable
    release must prove that the hostname terminates TLS and proxies only to the
    loopback worker(s); optional Caddy directives would make that claim
    ambiguous.
    """
    expected = expected_caddy_config(
        public_url=public_url, local_url=local_url, worker_count=worker_count
    )
    if source.read_text() != expected:
        raise WorkerReleaseError(
            "Caddy config must exactly terminate the public hostname to the loopback worker"
        )
    return expected


def validate_secure_worker_contract(
    contract: SecureWorkerContract,
    layout: SecureProfileLayout,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Refuse every known half-configured security-profile state."""

    if not _SAFE_ID.fullmatch(contract.release_id):
        raise WorkerReleaseError("invalid secure-profile release id")
    if not _SAFE_ID.fullmatch(contract.worker_code_release_id):
        raise WorkerReleaseError("invalid worker code release id")
    if contract.profile_mode not in {"migration", "first-install"}:
        raise WorkerReleaseError("secure profile mode must be migration or first-install")
    if contract.edge_provider not in {"cloudflared", "caddy"}:
        raise WorkerReleaseError("secure edge provider must be cloudflared or caddy")
    if not _SAFE_UNIT.fullmatch(contract.worker_unit):
        raise WorkerReleaseError("invalid worker systemd unit")
    if not _SAFE_UNIT.fullmatch(contract.tunnel_unit):
        raise WorkerReleaseError("invalid tunnel systemd unit")
    if contract.worker_unit == contract.tunnel_unit:
        raise WorkerReleaseError("worker and tunnel systemd units must be different")
    if not 1 <= contract.worker_port <= 65535:
        raise WorkerReleaseError("invalid worker port")
    _validated_worker_count(contract.worker_count, contract.worker_unit)
    if contract.worker_port + contract.worker_count - 1 > 65535:
        raise WorkerReleaseError("indexed worker ports exceed the valid range")
    for sibling in _indexed_worker_units(contract.worker_unit, contract.worker_count):
        if not _SAFE_UNIT.fullmatch(sibling):
            raise WorkerReleaseError("invalid indexed worker systemd unit")
        if sibling == contract.tunnel_unit:
            raise WorkerReleaseError("indexed worker unit collides with the tunnel unit")
    public_url = _validated_https_url(
        contract.worker_public_url,
        label="worker public URL",
    )
    local_url = _validated_loopback_url(
        contract.tunnel_local_url,
        port=contract.worker_port,
    )
    _validated_systemd_path(contract.worker_exec, label="worker executable")
    _validated_systemd_path(contract.worker_module_dir, label="worker module directory")
    code_release_root = contract.worker_module_dir
    expected_exec = code_release_root / ".venv" / "bin" / "python"
    if (
        code_release_root.name != contract.worker_code_release_id
        or code_release_root.parent.name != "releases"
        or contract.worker_exec != expected_exec
        or code_release_root.is_symlink()
        or not code_release_root.is_dir()
        or contract.worker_exec.is_symlink()
        or not contract.worker_exec.is_file()
        or not os.access(contract.worker_exec, os.X_OK)
    ):
        raise WorkerReleaseError(
            "worker executable and module directory must pin the exact code release"
        )
    ready = code_release_root / ".ready"
    source_marker = code_release_root / ".source-sha256"
    dependency_freeze = code_release_root / ".dependency-freeze.txt"
    dependency_marker = code_release_root / ".dependency-freeze.sha256"
    if (
        ready.is_symlink()
        or not ready.is_file()
        or stat.S_IMODE(ready.stat().st_mode) != 0o444
    ):
        raise WorkerReleaseError("worker code candidate is not ready")
    if source_marker.is_symlink() or not source_marker.is_file():
        raise WorkerReleaseError("worker code candidate source marker is missing")
    source_digest = source_marker.read_text().strip()
    if (
        not re.fullmatch(r"[0-9a-f]{64}", source_digest)
        or contract.worker_code_release_id != f"sha256-{source_digest[:24]}"
        or ready.read_text().strip() != source_digest
    ):
        raise WorkerReleaseError("worker code candidate source marker does not match")
    if (
        dependency_freeze.is_symlink()
        or not dependency_freeze.is_file()
        or dependency_marker.is_symlink()
        or not dependency_marker.is_file()
    ):
        raise WorkerReleaseError("worker code candidate dependency snapshot is missing")
    dependency_digest = dependency_marker.read_text().strip()
    if (
        not re.fullmatch(r"[0-9a-f]{64}", dependency_digest)
        or _sha256_file(dependency_freeze) != dependency_digest
    ):
        raise WorkerReleaseError("worker code candidate dependency snapshot drifted")
    _validated_systemd_path(layout.systemd_root, label="systemd root")
    _validated_systemd_path(layout.state_root, label="secure-profile state root")

    if contract.tunnel_exec_source.is_symlink() or not contract.tunnel_exec_source.is_file():
        raise WorkerReleaseError("repo-managed stable tunnel executable is missing")
    tunnel_exec_source = contract.tunnel_exec_source.read_text()
    provider_prefix = "FILMFORGE_TUNNEL" if contract.edge_provider == "cloudflared" else "FILMFORGE_CADDY"
    if f"{provider_prefix}_LOCAL_URL" not in tunnel_exec_source:
        raise WorkerReleaseError("stable edge executable does not enforce the local URL contract")
    if f"{provider_prefix}_PUBLIC_URL" not in tunnel_exec_source:
        raise WorkerReleaseError("stable edge executable does not enforce the public URL contract")
    if "FILMFORGE_EDGE_WORKER_SECRET_FILE" not in tunnel_exec_source and "FILMFORGE_TUNNEL_WORKER_SECRET_FILE" not in tunnel_exec_source:
        raise WorkerReleaseError("stable edge executable has no authenticated route watchdog")
    if "curl --config" not in tunnel_exec_source:
        raise WorkerReleaseError("stable edge executable has no continuous route probe")
    if re.search(r"(?:TOKEN|SECRET|PASSWORD)\s*=\s*[^\"'$\s]", tunnel_exec_source):
        raise WorkerReleaseError("stable edge executable contains an inline credential")
    if contract.tunnel_binary_source.is_symlink() or not contract.tunnel_binary_source.is_file():
        raise WorkerReleaseError("concrete edge executable is missing")
    if not os.access(contract.tunnel_binary_source, os.X_OK):
        raise WorkerReleaseError("concrete edge executable is not executable")

    worker_env = _strict_secret_env(contract.worker_secret_source)
    tunnel_env = _strict_secret_env(contract.tunnel_secret_source)
    required_worker = {
        "GPU_WORKER_API_TOKEN",
        "WORKER_REGISTRATION_TOKEN",
        "FILMFORGE_BACKEND_URL",
        "WORKER_API_AUTH_MODE",
        "WORKER_PUBLIC_URL",
    }
    missing_worker = sorted(required_worker - worker_env.keys())
    if missing_worker:
        raise WorkerReleaseError(
            "worker secret env is incomplete: " + ", ".join(missing_worker)
        )
    if worker_env["WORKER_API_AUTH_MODE"].strip().lower() != "required":
        raise WorkerReleaseError("secure profile requires WORKER_API_AUTH_MODE=required")
    if worker_env["WORKER_PUBLIC_URL"].rstrip("/") != public_url:
        raise WorkerReleaseError("worker and tunnel public URLs do not match")
    _validated_https_url(
        worker_env["FILMFORGE_BACKEND_URL"],
        label="FilmForge backend URL",
    )

    if contract.edge_provider == "cloudflared":
        required_tunnel = {
            "FILMFORGE_TUNNEL_CONFIG_FILE", "FILMFORGE_TUNNEL_CREDENTIAL_FILE",
            "FILMFORGE_TUNNEL_LOCAL_URL", "FILMFORGE_TUNNEL_PUBLIC_URL", "FILMFORGE_TUNNEL_UNIT",
        }
    else:
        required_tunnel = {
            "FILMFORGE_CADDY_CONFIG_FILE", "FILMFORGE_CADDY_LOCAL_URL",
            "FILMFORGE_CADDY_PUBLIC_URL", "FILMFORGE_CADDY_UNIT",
        }
    missing_tunnel = sorted(required_tunnel - tunnel_env.keys())
    if missing_tunnel:
        raise WorkerReleaseError(
            "tunnel secret env is incomplete: " + ", ".join(missing_tunnel)
        )
    local_key = "FILMFORGE_TUNNEL_LOCAL_URL" if contract.edge_provider == "cloudflared" else "FILMFORGE_CADDY_LOCAL_URL"
    public_key = "FILMFORGE_TUNNEL_PUBLIC_URL" if contract.edge_provider == "cloudflared" else "FILMFORGE_CADDY_PUBLIC_URL"
    unit_key = "FILMFORGE_TUNNEL_UNIT" if contract.edge_provider == "cloudflared" else "FILMFORGE_CADDY_UNIT"
    if tunnel_env[local_key].rstrip("/") != local_url:
        raise WorkerReleaseError("tunnel local URL does not match worker bind contract")
    if tunnel_env[public_key].rstrip("/") != public_url:
        raise WorkerReleaseError("worker and tunnel public URLs do not match")
    if tunnel_env[unit_key] != contract.tunnel_unit:
        raise WorkerReleaseError("tunnel env names a different systemd unit")

    unexpected_tunnel = sorted(
        set(tunnel_env)
        - required_tunnel
        - ({"CLOUDFLARED_BIN"} if contract.edge_provider == "cloudflared" else {"CADDY_BIN"})
    )
    if unexpected_tunnel:
        raise WorkerReleaseError(
            "tunnel env has unsupported keys: " + ", ".join(unexpected_tunnel)
        )

    config_key = "FILMFORGE_TUNNEL_CONFIG_FILE" if contract.edge_provider == "cloudflared" else "FILMFORGE_CADDY_CONFIG_FILE"
    _strict_mode_0600_file(Path(tunnel_env[config_key]), label="edge config file")
    _validated_systemd_path(Path(tunnel_env[config_key]), label="edge config file")
    if contract.edge_provider == "cloudflared":
        _strict_mode_0600_file(Path(tunnel_env["FILMFORGE_TUNNEL_CREDENTIAL_FILE"]), label="tunnel credential file")
        _validated_systemd_path(Path(tunnel_env["FILMFORGE_TUNNEL_CREDENTIAL_FILE"]), label="tunnel credential file")

    probe_env = _strict_secret_env(contract.backend_probe_secret_source)
    required_probe = {
        "FILMFORGE_BACKEND_CUTOVER_PROBE_URL",
        "FILMFORGE_BACKEND_CUTOVER_PROBE_TOKEN",
    }
    missing_probe = sorted(required_probe - probe_env.keys())
    if missing_probe:
        raise WorkerReleaseError(
            "backend cutover probe env is incomplete: " + ", ".join(missing_probe)
        )
    if set(probe_env) != required_probe:
        raise WorkerReleaseError("backend cutover probe env has unsupported keys")
    _validated_https_endpoint(
        probe_env["FILMFORGE_BACKEND_CUTOVER_PROBE_URL"],
        label="backend cutover probe URL",
    )
    probe_token = probe_env["FILMFORGE_BACKEND_CUTOVER_PROBE_TOKEN"]
    if len(probe_token) < 16:
        raise WorkerReleaseError("backend cutover probe token is too short")

    if contract.edge_provider == "cloudflared":
        _render_tunnel_config(Path(tunnel_env[config_key]), public_url=public_url, local_url=local_url, source_credential=Path(tunnel_env["FILMFORGE_TUNNEL_CREDENTIAL_FILE"]), staged_credential=Path("/validated/staged-tunnel-credential.json"))
    else:
        _render_caddy_config(Path(tunnel_env[config_key]), public_url=public_url, local_url=local_url, worker_count=contract.worker_count)
    return worker_env, tunnel_env, probe_env


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_text(path: Path, content: str, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _copy_secret(source: Path, destination: Path) -> None:
    data = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _copy_executable(source: Path, destination: Path) -> None:
    data = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o755)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o755)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.{uuid.uuid4().hex}.tmp")
    temporary.symlink_to(target)
    os.replace(temporary, link)


def _is_managed_profile_path(path: Path, layout: SecureProfileLayout) -> bool:
    if not path.is_symlink():
        return False
    try:
        path.resolve(strict=True).relative_to(layout.releases_root.resolve())
    except (FileNotFoundError, ValueError):
        return False
    return True


def _validate_profile_dropins(
    *,
    worker_dropin_dir: Path,
    tunnel_dropin_dir: Path,
    tunnel_unit_path: Path,
    layout: SecureProfileLayout,
    profile_mode: str,
    allow_legacy_loopback: bool = False,
    indexed_worker_units: tuple[str, ...] = (),
) -> Path | None:
    if PROFILE_DROPIN_NAME >= PUBLIC_OVERRIDE_NAME:
        raise WorkerReleaseError("secure profile must sort before public override")
    worker_dropin_dir.mkdir(parents=True, exist_ok=True)
    tunnel_dropin_dir.mkdir(parents=True, exist_ok=True)
    # ADR-0009: indexed workers (gpu1..N-1) live under the same refusal rules
    # as worker 0 — their drop-in dirs may hold only the managed profile link
    # and, on first-install, the staged guard. No override, no legacy shims:
    # indexed workers postdate both mechanisms.
    for indexed_unit in indexed_worker_units:
        indexed_dir = layout.systemd_root / f"{indexed_unit}.d"
        indexed_dir.mkdir(parents=True, exist_ok=True)
        allowed = {
            PROFILE_DROPIN_NAME,
            *({STAGED_GUARD_DROPIN_NAME} if profile_mode == "first-install" else set()),
        }
        unknown_indexed = [
            path.name
            for path in sorted(indexed_dir.glob("*.conf"))
            if path.name not in allowed
        ]
        if unknown_indexed:
            raise WorkerReleaseError(
                f"unmanaged worker drop-ins would survive secure cutover ({indexed_unit}): "
                + ", ".join(unknown_indexed)
            )
        for name in allowed:
            candidate = indexed_dir / name
            if (candidate.exists() or candidate.is_symlink()) and not _is_managed_profile_path(
                candidate, layout
            ):
                raise WorkerReleaseError(
                    f"existing indexed worker drop-in is not repo-managed ({indexed_unit})"
                )
    worker_dropins = sorted(path for path in worker_dropin_dir.glob("*.conf"))
    unknown_worker = [
        path.name
        for path in worker_dropins
        if path.name not in {
            PROFILE_DROPIN_NAME,
            *( {STAGED_GUARD_DROPIN_NAME} if profile_mode == "first-install" else set() ),
            PUBLIC_OVERRIDE_NAME,
            *( {"10-secure-loopback.conf"} if allow_legacy_loopback else set() ),
        }
    ]
    if unknown_worker:
        raise WorkerReleaseError(
            "unmanaged worker drop-ins would survive secure cutover: "
            + ", ".join(unknown_worker)
        )
    current_profile = worker_dropin_dir / PROFILE_DROPIN_NAME
    if current_profile.exists() or current_profile.is_symlink():
        if not _is_managed_profile_path(current_profile, layout):
            raise WorkerReleaseError("existing worker secure drop-in is not repo-managed")
    staged_guard = worker_dropin_dir / STAGED_GUARD_DROPIN_NAME
    if staged_guard.exists() or staged_guard.is_symlink():
        if profile_mode != "first-install" or not _is_managed_profile_path(
            staged_guard, layout
        ):
            raise WorkerReleaseError("existing staged worker guard is not repo-managed")

    tunnel_dropins = sorted(path for path in tunnel_dropin_dir.glob("*.conf"))
    unknown_tunnel = [
        path.name for path in tunnel_dropins if path.name != PROFILE_DROPIN_NAME
    ]
    if unknown_tunnel:
        raise WorkerReleaseError(
            "unmanaged tunnel drop-ins would alter the staged contract: "
            + ", ".join(unknown_tunnel)
        )
    current_tunnel_profile = tunnel_dropin_dir / PROFILE_DROPIN_NAME
    if current_tunnel_profile.exists() or current_tunnel_profile.is_symlink():
        if not _is_managed_profile_path(current_tunnel_profile, layout):
            raise WorkerReleaseError("existing tunnel secure drop-in is not repo-managed")
    if tunnel_unit_path.exists() or tunnel_unit_path.is_symlink():
        if not _is_managed_profile_path(tunnel_unit_path, layout):
            raise WorkerReleaseError("existing tunnel unit is not repo-managed")

    override = worker_dropin_dir / PUBLIC_OVERRIDE_NAME
    if profile_mode == "migration":
        if override.is_symlink() or not override.is_file():
            raise WorkerReleaseError(
                "99-public-url-override.conf must remain installed until verified cutover"
            )
        return override
    if override.exists() or override.is_symlink():
        raise WorkerReleaseError(
            "first-install mode refuses a legacy public override"
        )
    return None


def _inspect_legacy_loopback_profile(
    worker_dropin_dir: Path,
    *,
    worker_port: int,
    layout: SecureProfileLayout,
) -> tuple[Path, Path] | None:
    """Recognize only the incident's known half-shipped 10-* profile."""

    legacy = worker_dropin_dir / "10-secure-loopback.conf"
    if not (legacy.exists() or legacy.is_symlink()):
        return None
    if legacy.is_symlink() or not legacy.is_file():
        raise WorkerReleaseError("legacy 10-secure-loopback.conf is not a regular file")
    content = legacy.read_text()
    matches = re.findall(r"^EnvironmentFile=-?(\S+)\s*$", content, re.MULTILINE)
    if len(matches) != 1:
        raise WorkerReleaseError("legacy secure profile has an ambiguous EnvironmentFile")
    env_path = Path(matches[0])
    unit_match = re.fullmatch(
        r"filmforge-worker-gpu(\d+)\.service\.d",
        worker_dropin_dir.name,
    )
    if unit_match is None:
        raise WorkerReleaseError("legacy secure profile is attached to an unknown unit")
    expected_env = (
        layout.state_root.parent
        / f"worker-gpu{unit_match.group(1)}-secure.env"
    )
    if env_path != expected_env:
        raise WorkerReleaseError(
            "legacy secure profile references an unexpected env file"
        )
    _validated_systemd_path(env_path, label="legacy worker secure env")
    _strict_mode_0600_file(env_path, label="legacy worker secure env")
    env_values = _strict_secret_env(env_path)
    if (
        env_values.get("WORKER_API_AUTH_MODE") != "required"
        or env_values.get("WORKER_PUBLIC_URL") != "http://127.0.0.1:19000"
    ):
        raise WorkerReleaseError("legacy secure env is not the known incident profile")
    directives: list[tuple[str, str]] = []
    sections: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            sections.append(line)
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise WorkerReleaseError("legacy secure drop-in has invalid syntax")
        directives.append((key, value))
    exec_values = [value for key, value in directives if key == "ExecStart"]
    if (
        sections != ["[Service]"]
        or any(key not in {"EnvironmentFile", "WorkingDirectory", "ExecStart"} for key, _ in directives)
        or len([value for key, value in directives if key == "EnvironmentFile"]) != 1
        or len(exec_values) != 2
        or exec_values[0] != ""
        or not re.search(
            rf"(?:^|\s)--host\s+127\.0\.0\.1(?=\s|$).*?"
            rf"\s--port\s+{worker_port}(?:\s|$)",
            exec_values[1],
        )
    ):
        raise WorkerReleaseError("legacy 10-secure-loopback.conf is not the known profile")
    return legacy, env_path


@_locked_profile_operation
def stage_secure_profile(
    contract: SecureWorkerContract,
    layout: SecureProfileLayout = SecureProfileLayout(),
) -> StagedSecureProfile:
    """Stage one complete profile without restarting the worker or tunnel."""

    worker_env, tunnel_env, probe_env = validate_secure_worker_contract(contract, layout)
    worker_dropin_dir = layout.systemd_root / f"{contract.worker_unit}.d"
    tunnel_dropin_dir = layout.systemd_root / f"{contract.tunnel_unit}.d"
    tunnel_unit_path = layout.systemd_root / contract.tunnel_unit
    live_worker_profile = worker_dropin_dir / PROFILE_DROPIN_NAME
    live_active_pointer = layout.state_root / "active" / contract.worker_unit
    if (
        live_worker_profile.exists()
        or live_worker_profile.is_symlink()
        or live_active_pointer.exists()
        or live_active_pointer.is_symlink()
    ):
        expected_release = layout.releases_root / contract.release_id
        if (
            not live_worker_profile.is_symlink()
            or live_worker_profile.resolve()
            != (expected_release / "worker-secure-profile.conf").resolve()
            or not live_active_pointer.is_symlink()
            or live_active_pointer.resolve() != expected_release.resolve()
        ):
            raise WorkerReleaseError(
                "secure-to-secure profile replacement is unsupported; active profile unchanged"
            )
    legacy_profile = (
        _inspect_legacy_loopback_profile(
            worker_dropin_dir,
            worker_port=contract.worker_port,
            layout=layout,
        )
        if contract.profile_mode == "migration"
        else None
    )
    override = _validate_profile_dropins(
        worker_dropin_dir=worker_dropin_dir,
        tunnel_dropin_dir=tunnel_dropin_dir,
        tunnel_unit_path=tunnel_unit_path,
        layout=layout,
        profile_mode=contract.profile_mode,
        allow_legacy_loopback=legacy_profile is not None,
        indexed_worker_units=tuple(
            _indexed_worker_units(contract.worker_unit, contract.worker_count)
        ),
    )

    contract_data: dict[str, object] = {
        "edge_provider": contract.edge_provider,
        "profile_mode": contract.profile_mode,
        "release_id": contract.release_id,
        "worker_code_release_id": contract.worker_code_release_id,
        "worker_unit": contract.worker_unit,
        "tunnel_unit": contract.tunnel_unit,
        "worker_port": contract.worker_port,
        "worker_public_url": contract.worker_public_url.rstrip("/"),
        "tunnel_local_url": contract.tunnel_local_url.rstrip("/"),
        "worker_exec": str(contract.worker_exec),
        "worker_module_dir": str(contract.worker_module_dir),
        "worker_dependency_freeze_sha256": (
            contract.worker_module_dir / ".dependency-freeze.sha256"
        ).read_text().strip(),
    }
    contract_sha256 = hashlib.sha256(
        json.dumps(contract_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    config_key = "FILMFORGE_TUNNEL_CONFIG_FILE" if contract.edge_provider == "cloudflared" else "FILMFORGE_CADDY_CONFIG_FILE"
    tunnel_config_source = Path(tunnel_env[config_key])
    tunnel_credential_source = (
        Path(tunnel_env["FILMFORGE_TUNNEL_CREDENTIAL_FILE"])
        if contract.edge_provider == "cloudflared" else None
    )
    source_fingerprints = {
        "worker_secret_source_sha256": _sha256_file(contract.worker_secret_source),
        "tunnel_secret_source_sha256": _sha256_file(contract.tunnel_secret_source),
        "backend_probe_secret_source_sha256": _sha256_file(
            contract.backend_probe_secret_source
        ),
        "tunnel_config_source_sha256": _sha256_file(tunnel_config_source),
        "tunnel_exec_source_sha256": _sha256_file(contract.tunnel_exec_source),
        "tunnel_binary_source_sha256": _sha256_file(contract.tunnel_binary_source),
    }
    if tunnel_credential_source is not None:
        source_fingerprints["tunnel_credential_source_sha256"] = _sha256_file(tunnel_credential_source)
    if legacy_profile is not None:
        source_fingerprints.update(
            {
                "legacy_loopback_dropin_sha256": _sha256_file(legacy_profile[0]),
                "legacy_loopback_env_sha256": _sha256_file(legacy_profile[1]),
            }
        )

    release_dir = layout.releases_root / contract.release_id
    if release_dir.exists():
        stage_receipt = release_dir / "stage-receipt.json"
        if not stage_receipt.is_file():
            raise WorkerReleaseError("existing secure-profile release is incomplete")
        data = _stage_data(release_dir)
        if data.get("contract_sha256") != contract_sha256:
            raise WorkerReleaseError(
                "existing secure-profile release id was reused with a different contract"
            )
        for field, expected in source_fingerprints.items():
            if data.get(field) != expected:
                raise WorkerReleaseError(
                    "existing secure-profile release id was reused with different inputs"
                )
        _verify_staged_release(release_dir, data)
        if data.get("rollback_state") == "complete" and not (
            live_worker_profile.exists() or live_worker_profile.is_symlink()
        ):
            for field in (
                "rollback_state",
                "rollback_started_at_epoch",
                "rolled_back_at_epoch",
                "cutover_state",
                "cutover_started_at_epoch",
                "cutover_failure_at_epoch",
                "cutover_at_epoch",
            ):
                data.pop(field, None)
            data["tunnel_prepared"] = False
            data["cutover_performed"] = False
            _write_text(
                release_dir / "stage-receipt.json",
                json.dumps(data, sort_keys=True, indent=2) + "\n",
                mode=0o600,
            )
    else:
        layout.releases_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        staging = layout.releases_root / f".{contract.release_id}.{uuid.uuid4().hex}.stage"
        staging.mkdir(mode=0o700)
        try:
            tunnel_unit_artifact = contract.tunnel_unit
            worker_env_path = staging / "worker-secrets.env"
            tunnel_env_path = staging / ("tunnel-secrets.env" if contract.edge_provider == "cloudflared" else "caddy-secrets.env")
            backend_probe_env_path = staging / "backend-cutover-probe.env"
            tunnel_config_path = staging / ("cloudflared.yml" if contract.edge_provider == "cloudflared" else "Caddyfile")
            tunnel_credential_path = staging / "cloudflared-credential.json"
            tunnel_exec_path = staging / ("filmforge-worker-tunnel" if contract.edge_provider == "cloudflared" else "filmforge-worker-caddy")
            tunnel_binary_path = staging / ("cloudflared" if contract.edge_provider == "cloudflared" else "caddy")

            final_worker_env = release_dir / worker_env_path.name
            final_tunnel_env = release_dir / tunnel_env_path.name
            final_probe_env = release_dir / backend_probe_env_path.name
            final_tunnel_config = release_dir / tunnel_config_path.name
            final_tunnel_credential = release_dir / tunnel_credential_path.name
            final_tunnel_exec = release_dir / tunnel_exec_path.name
            final_tunnel_binary = release_dir / tunnel_binary_path.name

            _write_text(
                worker_env_path,
                _render_secret_env(worker_env),
                mode=0o600,
            )
            _write_text(
                backend_probe_env_path,
                _render_secret_env(probe_env),
                mode=0o600,
            )
            if contract.edge_provider == "cloudflared":
                assert tunnel_credential_source is not None
                _copy_secret(tunnel_credential_source, tunnel_credential_path)
                _write_text(tunnel_config_path, _render_tunnel_config(tunnel_config_source, public_url=contract.worker_public_url.rstrip("/"), local_url=contract.tunnel_local_url.rstrip("/"), source_credential=tunnel_credential_source, staged_credential=final_tunnel_credential), mode=0o600)
            else:
                _write_text(tunnel_config_path, _render_caddy_config(tunnel_config_source, public_url=contract.worker_public_url.rstrip("/"), local_url=contract.tunnel_local_url.rstrip("/"), worker_count=contract.worker_count), mode=0o600)
            staged_tunnel_env = dict(tunnel_env)
            if contract.edge_provider == "cloudflared":
                staged_tunnel_env.update({"FILMFORGE_TUNNEL_CONFIG_FILE": str(final_tunnel_config), "FILMFORGE_TUNNEL_CREDENTIAL_FILE": str(final_tunnel_credential), "FILMFORGE_TUNNEL_WORKER_SECRET_FILE": str(final_worker_env), "FILMFORGE_TUNNEL_CUTOVER_AUTHORIZATION_FILE": str(release_dir / "cutover-authorized"), "FILMFORGE_TUNNEL_WORKER_CODE_RELEASE_ID": contract.worker_code_release_id, "CLOUDFLARED_BIN": str(final_tunnel_binary)})
            else:
                staged_tunnel_env.update({"FILMFORGE_CADDY_CONFIG_FILE": str(final_tunnel_config), "FILMFORGE_CADDY_LOCAL_URL": contract.tunnel_local_url.rstrip("/"), "FILMFORGE_CADDY_PUBLIC_URL": contract.worker_public_url.rstrip("/"), "FILMFORGE_CADDY_UNIT": contract.tunnel_unit, "FILMFORGE_EDGE_WORKER_SECRET_FILE": str(final_worker_env), "FILMFORGE_EDGE_CUTOVER_AUTHORIZATION_FILE": str(release_dir / "cutover-authorized"), "FILMFORGE_EDGE_WORKER_CODE_RELEASE_ID": contract.worker_code_release_id, "CADDY_BIN": str(final_tunnel_binary)})
            _write_text(
                tunnel_env_path,
                _render_secret_env(staged_tunnel_env),
                mode=0o600,
            )
            _copy_executable(contract.tunnel_exec_source, tunnel_exec_path)
            _copy_executable(contract.tunnel_binary_source, tunnel_binary_path)
            worker_dropin_content = (
                "[Unit]\n"
                f"BindsTo={contract.tunnel_unit}\n"
                f"Requires={contract.tunnel_unit}\n"
                f"After={contract.tunnel_unit}\n\n"
                "[Service]\n"
                f"EnvironmentFile={final_worker_env}\n"
                "Environment=PYTHONDONTWRITEBYTECODE=1\n"
                f"Environment=WORKER_CODE_RELEASE_ID={contract.worker_code_release_id}\n"
                f"WorkingDirectory={contract.worker_module_dir}\n"
                "ExecStart=\n"
                f"ExecStart={contract.worker_exec} -m uvicorn gpu_worker.app:app "
                f"--host 127.0.0.1 --port {contract.worker_port}\n"
            )
            tunnel_dropin_content = (
                "[Service]\n"
                f"EnvironmentFile={final_tunnel_env}\n"
            )
            tunnel_unit_content = (
                "[Unit]\n"
                f"Description=FilmForge stable authenticated worker {contract.edge_provider} edge\n"
                "After=network-online.target\n"
                "Wants=network-online.target\n\n"
                "[Service]\n"
                "Type=simple\n"
                f"ExecStart={final_tunnel_exec}\n"
                "Restart=always\n"
                "RestartSec=5\n\n"
                "[Install]\n"
                "WantedBy=multi-user.target\n"
            )
            staged_guard_content = (
                "[Unit]\n"
                f"ConditionPathExists={release_dir / 'cutover-authorized'}\n"
            )
            _write_text(
                staging / "worker-secure-profile.conf",
                worker_dropin_content,
                mode=0o644,
            )
            # ADR-0009: one staged drop-in per additional worker. Identical to
            # worker 0's except the loopback port and a WORKER_PUBLIC_URL set
            # AFTER the EnvironmentFile line — systemd's last-assignment-wins
            # gives each indexed worker its /gpu{i} path-suffixed identity
            # while every secret still comes from the one staged env file.
            for _idx in range(1, contract.worker_count):
                indexed_content = (
                    "[Unit]\n"
                    f"BindsTo={contract.tunnel_unit}\n"
                    f"Requires={contract.tunnel_unit}\n"
                    f"After={contract.tunnel_unit}\n\n"
                    "[Service]\n"
                    f"EnvironmentFile={final_worker_env}\n"
                    f"Environment=WORKER_PUBLIC_URL={_indexed_public_url(contract.worker_public_url, _idx)}\n"
                    "Environment=PYTHONDONTWRITEBYTECODE=1\n"
                    f"Environment=WORKER_CODE_RELEASE_ID={contract.worker_code_release_id}\n"
                    f"WorkingDirectory={contract.worker_module_dir}\n"
                    "ExecStart=\n"
                    f"ExecStart={contract.worker_exec} -m uvicorn gpu_worker.app:app "
                    f"--host 127.0.0.1 --port {contract.worker_port + _idx}\n"
                )
                _write_text(
                    staging / f"worker-secure-profile-gpu{_idx}.conf",
                    indexed_content,
                    mode=0o644,
                )
            _write_text(
                staging / "tunnel-secure-profile.conf",
                tunnel_dropin_content,
                mode=0o644,
            )
            _write_text(
                staging / tunnel_unit_artifact,
                tunnel_unit_content,
                mode=0o644,
            )
            _write_text(
                staging / "worker-staged-guard.conf",
                staged_guard_content,
                mode=0o644,
            )
            rollback_dir = staging / "rollback"
            rollback_dir.mkdir(mode=0o700)
            if override is not None:
                shutil.copy2(override, rollback_dir / PUBLIC_OVERRIDE_NAME)
                os.chmod(rollback_dir / PUBLIC_OVERRIDE_NAME, 0o600)
            if legacy_profile is not None:
                _copy_secret(
                    legacy_profile[0],
                    rollback_dir / "10-secure-loopback.conf.original",
                )
                _copy_secret(
                    legacy_profile[1],
                    rollback_dir / "legacy-worker-secure.env.original",
                )

            stage_data = {
                "schema": STAGE_SCHEMA,
                **contract_data,
                "contract_sha256": contract_sha256,
                **source_fingerprints,
                "public_override_name": PUBLIC_OVERRIDE_NAME,
                "public_override_sha256": (
                    _sha256_file(override) if override is not None else None
                ),
                "legacy_loopback_env_path": (
                    str(legacy_profile[1]) if legacy_profile is not None else None
                ),
                "legacy_loopback_dropin_mode": (
                    stat.S_IMODE(legacy_profile[0].stat().st_mode)
                    if legacy_profile is not None
                    else None
                ),
                "worker_secret_sha256": _sha256_file(worker_env_path),
                "tunnel_secret_sha256": _sha256_file(tunnel_env_path),
                "backend_probe_secret_sha256": _sha256_file(backend_probe_env_path),
                "tunnel_config_sha256": _sha256_file(tunnel_config_path),
                "tunnel_exec_sha256": _sha256_file(tunnel_exec_path),
                "tunnel_binary_sha256": _sha256_file(tunnel_binary_path),
                "worker_dropin_sha256": _sha256_file(
                    staging / "worker-secure-profile.conf"
                ),
                "tunnel_dropin_sha256": _sha256_file(
                    staging / "tunnel-secure-profile.conf"
                ),
                "tunnel_unit_sha256": _sha256_file(
                    staging / tunnel_unit_artifact
                ),
                "tunnel_unit_artifact": tunnel_unit_artifact,
                "worker_guard_sha256": _sha256_file(
                    staging / "worker-staged-guard.conf"
                ),
                **{
                    f"worker_dropin_gpu{_idx}_sha256": _sha256_file(
                        staging / f"worker-secure-profile-gpu{_idx}.conf"
                    )
                    for _idx in range(1, contract.worker_count)
                },
                "staged_at_epoch": int(time.time()),
                "tunnel_prepared": False,
                "cutover_performed": False,
            }
            if contract.edge_provider == "cloudflared":
                stage_data["tunnel_credential_sha256"] = _sha256_file(tunnel_credential_path)
            else:
                stage_data.update({"caddy_secret_sha256": stage_data["tunnel_secret_sha256"], "caddy_config_sha256": stage_data["tunnel_config_sha256"], "caddy_exec_sha256": stage_data["tunnel_exec_sha256"], "caddy_binary_sha256": stage_data["tunnel_binary_sha256"], "caddy_tls_hostname": urlsplit(contract.worker_public_url).hostname, "caddy_https_ports": [80, 443]})
            _write_text(
                staging / "stage-receipt.json",
                json.dumps(stage_data, sort_keys=True, indent=2) + "\n",
                mode=0o600,
            )
            os.replace(staging, release_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    worker_dropin = worker_dropin_dir / PROFILE_DROPIN_NAME
    tunnel_dropin = tunnel_dropin_dir / PROFILE_DROPIN_NAME
    staged_data = _stage_data(release_dir)
    _atomic_symlink(
        _tunnel_unit_artifact(release_dir, staged_data), tunnel_unit_path
    )
    _atomic_symlink(release_dir / "tunnel-secure-profile.conf", tunnel_dropin)
    if contract.profile_mode == "first-install":
        _atomic_symlink(
            release_dir / "worker-staged-guard.conf",
            worker_dropin_dir / STAGED_GUARD_DROPIN_NAME,
        )
        # ADR-0009: indexed workers get the same start guard — the deploy layer
        # enables their base units, and none may start before the receipt-gated
        # cutover authorizes the release, exactly like worker 0.
        for _unit in _indexed_worker_units(contract.worker_unit, contract.worker_count):
            _atomic_symlink(
                release_dir / "worker-staged-guard.conf",
                layout.systemd_root / f"{_unit}.d" / STAGED_GUARD_DROPIN_NAME,
            )
    _atomic_symlink(release_dir, layout.state_root / "staged" / contract.worker_unit)
    # Re-check after tunnel-only link installation. The worker dependency and
    # ExecStart are intentionally not linked until the receipt-gated cutover;
    # restarting the prepared tunnel therefore cannot stop the live worker.
    _validate_profile_dropins(
        worker_dropin_dir=worker_dropin_dir,
        tunnel_dropin_dir=tunnel_dropin_dir,
        tunnel_unit_path=tunnel_unit_path,
        layout=layout,
        profile_mode=contract.profile_mode,
        allow_legacy_loopback=legacy_profile is not None,
        indexed_worker_units=tuple(
            _indexed_worker_units(contract.worker_unit, contract.worker_count)
        ),
    )
    _verify_staged_release(release_dir, _stage_data(release_dir))
    return StagedSecureProfile(
        release_id=contract.release_id,
        release_dir=release_dir,
        worker_dropin=worker_dropin,
        tunnel_dropin=tunnel_dropin,
        stage_receipt=release_dir / "stage-receipt.json",
    )


_STAGED_ARTIFACTS: dict[str, tuple[str, int]] = {
    "worker_secret_sha256": ("worker-secrets.env", 0o600),
    "tunnel_secret_sha256": ("tunnel-secrets.env", 0o600),
    "backend_probe_secret_sha256": ("backend-cutover-probe.env", 0o600),
    "tunnel_config_sha256": ("cloudflared.yml", 0o600),
    "tunnel_credential_sha256": ("cloudflared-credential.json", 0o600),
    "tunnel_exec_sha256": ("filmforge-worker-tunnel", 0o755),
    "tunnel_binary_sha256": ("cloudflared", 0o755),
    "worker_dropin_sha256": ("worker-secure-profile.conf", 0o644),
    "tunnel_dropin_sha256": ("tunnel-secure-profile.conf", 0o644),
    "worker_guard_sha256": ("worker-staged-guard.conf", 0o644),
}


def _tunnel_unit_artifact(
    release_dir: Path,
    stage_data: dict[str, object],
) -> Path:
    """Resolve and validate the tunnel unit owned by this immutable receipt.

    Early Caddy releases stored the unit under its concrete systemd unit name
    (for example ``filmforge-worker-edge-gpu0.service``). New releases use the
    receipt-bound ``tunnel_unit_artifact`` identity. The recorded hash remains
    authoritative in both cases.
    """

    value = stage_data.get("tunnel_unit_artifact")
    if value is not None:
        if (
            not isinstance(value, str)
            or not _SAFE_UNIT.fullmatch(value)
            or value != stage_data.get("tunnel_unit")
        ):
            raise WorkerReleaseError("staged tunnel unit artifact identity is invalid")
        return release_dir / value

    canonical = release_dir / "filmforge-worker-tunnel.service"
    if canonical.is_file() and not canonical.is_symlink():
        return canonical
    legacy_name = str(stage_data.get("tunnel_unit") or "")
    if _SAFE_UNIT.fullmatch(legacy_name):
        legacy = release_dir / legacy_name
        if legacy.is_file() and not legacy.is_symlink():
            return legacy
    return canonical


def _staged_artifacts(
    stage_data: dict[str, object],
    release_dir: Path,
) -> dict[str, tuple[str, int]]:
    artifacts = dict(_STAGED_ARTIFACTS)
    artifacts["tunnel_unit_sha256"] = (
        _tunnel_unit_artifact(release_dir, stage_data).name,
        0o644,
    )
    if stage_data.get("edge_provider") == "caddy":
        for field in ("tunnel_secret_sha256", "tunnel_config_sha256", "tunnel_credential_sha256", "tunnel_exec_sha256", "tunnel_binary_sha256"):
            artifacts.pop(field)
        artifacts["caddy_secret_sha256"] = ("caddy-secrets.env", 0o600)
        artifacts["caddy_config_sha256"] = ("Caddyfile", 0o600)
        artifacts["caddy_exec_sha256"] = ("filmforge-worker-caddy", 0o755)
        artifacts["caddy_binary_sha256"] = ("caddy", 0o755)
    for _idx in range(1, int(stage_data.get("worker_count") or 1)):
        artifacts[f"worker_dropin_gpu{_idx}_sha256"] = (
            f"worker-secure-profile-gpu{_idx}.conf",
            0o644,
        )
    return artifacts


def _stage_worker_count(stage_data: dict[str, object]) -> int:
    """worker_count from a receipt; pre-ADR-0009 receipts read as 1."""
    return int(stage_data.get("worker_count") or 1)


def _stage_indexed_units(stage_data: dict[str, object]) -> list[str]:
    return _indexed_worker_units(
        str(stage_data["worker_unit"]), _stage_worker_count(stage_data)
    )


def _verify_staged_release(
    release_dir: Path,
    stage_data: dict[str, object],
) -> None:
    for field, (relative, expected_mode) in _staged_artifacts(
        stage_data, release_dir
    ).items():
        path = release_dir / relative
        if path.is_symlink() or not path.is_file():
            raise WorkerReleaseError(f"staged secure-profile artifact is missing: {relative}")
        if stat.S_IMODE(path.stat().st_mode) != expected_mode:
            raise WorkerReleaseError(
                f"staged secure-profile artifact mode drifted: {relative}"
            )
        if _sha256_file(path) != stage_data.get(field):
            raise WorkerReleaseError(
                f"staged secure-profile artifact content drifted: {relative}"
            )
    stage_receipt = release_dir / "stage-receipt.json"
    if stage_receipt.is_symlink() or stat.S_IMODE(stage_receipt.stat().st_mode) != 0o600:
        raise WorkerReleaseError("staged secure-profile receipt permissions drifted")


def _assert_active_profile_links(
    *,
    release_dir: Path,
    stage_data: dict[str, object],
    layout: SecureProfileLayout,
    require_worker_profile: bool = True,
) -> tuple[Path, Path, Path]:
    worker_unit = str(stage_data["worker_unit"])
    tunnel_unit = str(stage_data["tunnel_unit"])
    worker_dropin = (
        layout.systemd_root / f"{worker_unit}.d" / PROFILE_DROPIN_NAME
    )
    tunnel_dropin = (
        layout.systemd_root / f"{tunnel_unit}.d" / PROFILE_DROPIN_NAME
    )
    tunnel_unit_path = layout.systemd_root / tunnel_unit
    expected = {
        tunnel_dropin: release_dir / "tunnel-secure-profile.conf",
        tunnel_unit_path: _tunnel_unit_artifact(release_dir, stage_data),
    }
    indexed_dropins = {
        layout.systemd_root / f"{unit}.d" / PROFILE_DROPIN_NAME:
            release_dir / f"worker-secure-profile-gpu{idx}.conf"
        for idx, unit in enumerate(_stage_indexed_units(stage_data), start=1)
    }
    if require_worker_profile:
        expected[worker_dropin] = release_dir / "worker-secure-profile.conf"
        expected.update(indexed_dropins)
    else:
        for pending in (worker_dropin, *indexed_dropins):
            if pending.exists() or pending.is_symlink():
                raise WorkerReleaseError(
                    "worker secure profile became active before receipt-gated cutover"
                )
    for managed_path, target in expected.items():
        if not managed_path.is_symlink() or managed_path.resolve() != target.resolve():
            raise WorkerReleaseError("staged secure profile is not the active managed set")
    if stage_data.get("profile_mode") == "first-install":
        guard = worker_dropin.parent / STAGED_GUARD_DROPIN_NAME
        expected_guard = release_dir / "worker-staged-guard.conf"
        if not guard.is_symlink() or guard.resolve() != expected_guard.resolve():
            raise WorkerReleaseError("first-install staged worker guard is missing")
    return worker_dropin, tunnel_dropin, tunnel_unit_path


@_locked_profile_operation
def prepare_secure_profile(
    *,
    release_id: str,
    layout: SecureProfileLayout = SecureProfileLayout(),
    controller: ServiceController | None = None,
    now_epoch: int | None = None,
) -> None:
    """Load and restart the staged tunnel while the worker remains safe."""

    if not _SAFE_ID.fullmatch(release_id):
        raise WorkerReleaseError("invalid secure-profile release id")
    release_dir = layout.releases_root / release_id
    stage_data = _stage_data(release_dir)
    if stage_data.get("release_id") != release_id:
        raise WorkerReleaseError("staged release id does not match")
    _verify_staged_release(release_dir, stage_data)
    worker_dropin, tunnel_dropin, tunnel_unit_path = _assert_active_profile_links(
        release_dir=release_dir,
        stage_data=stage_data,
        layout=layout,
        require_worker_profile=False,
    )
    profile_mode = str(stage_data["profile_mode"])
    override = _validate_profile_dropins(
        worker_dropin_dir=worker_dropin.parent,
        tunnel_dropin_dir=tunnel_dropin.parent,
        tunnel_unit_path=tunnel_unit_path,
        layout=layout,
        profile_mode=profile_mode,
        allow_legacy_loopback=stage_data.get("legacy_loopback_dropin_sha256") is not None,
        indexed_worker_units=tuple(_stage_indexed_units(stage_data)),
    )
    if override is not None and _sha256_file(override) != stage_data.get(
        "public_override_sha256"
    ):
        raise WorkerReleaseError("public override changed after secure profile was staged")

    service_controller = controller or SystemdServiceController()
    if profile_mode == "first-install":
        service_controller.assert_inactive(str(stage_data["worker_unit"]))
        service_controller.assert_disabled(str(stage_data["worker_unit"]))
        service_controller.assert_disabled(str(stage_data["tunnel_unit"]))
    service_controller.daemon_reload()
    service_controller.restart(str(stage_data["tunnel_unit"]))
    service_controller.assert_active(str(stage_data["tunnel_unit"]))
    if stage_data.get("edge_provider") == "caddy":
        # Caddy is the deliberately public TLS edge.  This is the only place
        # public listeners are permitted; the worker itself remains loopback.
        service_controller.assert_public_listener(80)
        service_controller.assert_public_listener(443)
    service_controller.assert_unit_loaded(
        str(stage_data["tunnel_unit"]),
        fragment_path=tunnel_unit_path,
        dropin_paths=[tunnel_dropin],
    )
    stage_data["tunnel_prepared"] = True
    stage_data["tunnel_prepared_at_epoch"] = (
        int(time.time()) if now_epoch is None else now_epoch
    )
    _write_text(
        release_dir / "stage-receipt.json",
        json.dumps(stage_data, sort_keys=True, indent=2) + "\n",
        mode=0o600,
    )


def _load_cutover_receipt(
    receipt_path: Path,
    *,
    stage_data: dict[str, object],
    now_epoch: int,
) -> dict[str, object]:
    receipt_values = _strict_secret_env_file_json(receipt_path)
    if receipt_values.get("schema") != RECEIPT_SCHEMA:
        raise WorkerReleaseError("cutover receipt schema is invalid")
    exact_fields = (
        "profile_mode",
        "release_id",
        "worker_code_release_id",
        "worker_dependency_freeze_sha256",
        "worker_unit",
        "tunnel_unit",
        "worker_port",
        "worker_public_url",
        "tunnel_local_url",
        "worker_secret_sha256",
        "backend_probe_secret_sha256",
        "worker_dropin_sha256",
        "tunnel_dropin_sha256",
        "tunnel_unit_sha256",
        "worker_guard_sha256",
        "contract_sha256",
    )
    if stage_data.get("edge_provider") == "caddy":
        exact_fields += ("edge_provider", "caddy_secret_sha256", "caddy_config_sha256", "caddy_exec_sha256", "caddy_binary_sha256", "caddy_tls_hostname", "caddy_https_ports")
    else:
        exact_fields += ("tunnel_secret_sha256", "tunnel_config_sha256", "tunnel_credential_sha256", "tunnel_exec_sha256", "tunnel_binary_sha256")
    for field in exact_fields:
        if receipt_values.get(field) != stage_data.get(field):
            raise WorkerReleaseError(f"cutover receipt {field} does not match staged release")
    for field in ("tunnel_ready", "worker_secret_fingerprint_match"):
        if receipt_values.get(field) is not True:
            raise WorkerReleaseError(f"cutover receipt does not prove {field}")
    if stage_data.get("profile_mode") == "migration":
        # A live worker must prove that its existing backend relationship is
        # ready before we remove the incident-era public override.
        for field in ("backend_bearer_client_ready", "backend_registration_ready"):
            if receipt_values.get(field) is not True:
                raise WorkerReleaseError(f"cutover receipt does not prove {field}")
    else:
        # A first-install worker is deliberately inactive until this function
        # starts it.  Its pre-start receipt cannot truthfully claim a client or
        # registration; the mandatory post-start backend probe below is the
        # authoritative proof and is before enable/active publication.
        for field in ("backend_bearer_client_ready", "backend_registration_ready"):
            if receipt_values.get(field) is not False:
                raise WorkerReleaseError(
                    f"first-install receipt must leave {field} false until post-start probe"
                )
    if stage_data.get("edge_provider") == "caddy" and receipt_values.get("edge_tls_hostname_ready") is not True:
        raise WorkerReleaseError("cutover receipt does not prove edge TLS hostname readiness")
    if stage_data.get("tunnel_prepared") is not True:
        raise WorkerReleaseError("staged tunnel has not completed the prepare phase")
    issued_at = receipt_values.get("issued_at_epoch")
    if not isinstance(issued_at, int):
        raise WorkerReleaseError("cutover receipt issued_at_epoch is invalid")
    age = now_epoch - issued_at
    if age < -60 or age > MAX_RECEIPT_AGE_SECONDS:
        raise WorkerReleaseError("cutover receipt is stale")
    return receipt_values


def _strict_secret_env_file_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise WorkerReleaseError("cutover receipt must be a regular file")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise WorkerReleaseError("cutover receipt must have mode 0600")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        raise WorkerReleaseError("cutover receipt is not valid JSON") from None
    if not isinstance(value, dict):
        raise WorkerReleaseError("cutover receipt must be a JSON object")
    return value


def _stage_data(release_dir: Path) -> dict[str, object]:
    path = release_dir / "stage-receipt.json"
    try:
        value = _strict_secret_env_file_json(path)
    except WorkerReleaseError:
        raise WorkerReleaseError(
            "staged secure-profile receipt is missing, invalid, or not mode 0600"
        ) from None
    if not isinstance(value, dict) or value.get("schema") != STAGE_SCHEMA:
        raise WorkerReleaseError("staged secure-profile receipt schema is invalid")
    return value


def _verified_public_override_backup(
    release_dir: Path,
    stage_data: dict[str, object],
) -> Path:
    backup = release_dir / "rollback" / PUBLIC_OVERRIDE_NAME
    if backup.is_symlink() or not backup.is_file():
        raise WorkerReleaseError("public override rollback copy is missing")
    if stat.S_IMODE(backup.stat().st_mode) != 0o600:
        raise WorkerReleaseError("public override rollback copy permissions drifted")
    if _sha256_file(backup) != stage_data.get("public_override_sha256"):
        raise WorkerReleaseError("public override rollback copy content drifted")
    return backup


def _legacy_profile_paths(
    *,
    release_dir: Path,
    stage_data: dict[str, object],
    worker_dropin_dir: Path,
    layout: SecureProfileLayout,
) -> tuple[Path, Path] | None:
    expected_dropin_sha = stage_data.get("legacy_loopback_dropin_sha256")
    expected_env_sha = stage_data.get("legacy_loopback_env_sha256")
    if expected_dropin_sha is None and expected_env_sha is None:
        return None
    if not isinstance(expected_dropin_sha, str) or not isinstance(expected_env_sha, str):
        raise WorkerReleaseError("legacy secure-profile rollback metadata is incomplete")
    unit_match = re.fullmatch(
        r"filmforge-worker-gpu(\d+)\.service\.d",
        worker_dropin_dir.name,
    )
    if unit_match is None:
        raise WorkerReleaseError("legacy secure-profile worker unit is invalid")
    dropin = worker_dropin_dir / "10-secure-loopback.conf"
    env_path = layout.state_root.parent / f"worker-gpu{unit_match.group(1)}-secure.env"
    if stage_data.get("legacy_loopback_env_path") != str(env_path):
        raise WorkerReleaseError("legacy secure-profile env path metadata drifted")
    original_dropin = release_dir / "rollback" / "10-secure-loopback.conf.original"
    original_env = release_dir / "rollback" / "legacy-worker-secure.env.original"
    for original, expected_sha, label in (
        (original_dropin, expected_dropin_sha, "legacy drop-in"),
        (original_env, expected_env_sha, "legacy env"),
    ):
        if original.is_symlink() or not original.is_file():
            raise WorkerReleaseError(f"{label} rollback copy is missing")
        if stat.S_IMODE(original.stat().st_mode) != 0o600:
            raise WorkerReleaseError(f"{label} rollback copy permissions drifted")
        if _sha256_file(original) != expected_sha:
            raise WorkerReleaseError(f"{label} rollback copy content drifted")
    return dropin, env_path


def _disable_legacy_profile(
    *,
    release_dir: Path,
    stage_data: dict[str, object],
    worker_dropin_dir: Path,
    layout: SecureProfileLayout,
) -> bool:
    paths = _legacy_profile_paths(
        release_dir=release_dir,
        stage_data=stage_data,
        worker_dropin_dir=worker_dropin_dir,
        layout=layout,
    )
    if paths is None:
        return False
    dropin, env_path = paths
    disabled_dropin = release_dir / "rollback" / "10-secure-loopback.conf.disabled"
    disabled_env = release_dir / "rollback" / "legacy-worker-secure.env.disabled"
    live_state = (dropin.is_file() and not dropin.is_symlink(), env_path.is_file() and not env_path.is_symlink())
    disabled_state = (
        disabled_dropin.is_file() and not disabled_dropin.is_symlink(),
        disabled_env.is_file() and not disabled_env.is_symlink(),
    )
    if live_state == (False, False) and disabled_state == (True, True):
        if (
            stat.S_IMODE(disabled_dropin.stat().st_mode) != 0o600
            or stat.S_IMODE(disabled_env.stat().st_mode) != 0o600
            or _sha256_file(disabled_dropin) != stage_data["legacy_loopback_dropin_sha256"]
            or _sha256_file(disabled_env) != stage_data["legacy_loopback_env_sha256"]
        ):
            raise WorkerReleaseError("disabled legacy secure profile drifted")
        return True
    if live_state == (False, True) and disabled_state == (True, False):
        if (
            _sha256_file(disabled_dropin) != stage_data["legacy_loopback_dropin_sha256"]
            or _sha256_file(env_path) != stage_data["legacy_loopback_env_sha256"]
        ):
            raise WorkerReleaseError("partially disabled legacy secure profile drifted")
        os.replace(env_path, disabled_env)
        os.chmod(disabled_env, 0o600)
        return True
    if live_state == (True, False) and disabled_state == (False, True):
        if (
            _sha256_file(dropin) != stage_data["legacy_loopback_dropin_sha256"]
            or _sha256_file(disabled_env) != stage_data["legacy_loopback_env_sha256"]
        ):
            raise WorkerReleaseError("partially disabled legacy secure profile drifted")
        os.replace(dropin, disabled_dropin)
        os.chmod(disabled_dropin, 0o600)
        return True
    if live_state != (True, True) or disabled_state != (False, False):
        raise WorkerReleaseError("legacy secure profile is in a partial transition state")
    if (
        _sha256_file(dropin) != stage_data["legacy_loopback_dropin_sha256"]
        or _sha256_file(env_path) != stage_data["legacy_loopback_env_sha256"]
    ):
        raise WorkerReleaseError("legacy secure profile changed before cutover")
    os.replace(dropin, disabled_dropin)
    try:
        os.replace(env_path, disabled_env)
    except Exception:
        os.replace(disabled_dropin, dropin)
        raise
    os.chmod(disabled_dropin, 0o600)
    os.chmod(disabled_env, 0o600)
    return True


def _restore_legacy_profile(
    *,
    release_dir: Path,
    stage_data: dict[str, object],
    worker_dropin_dir: Path,
    layout: SecureProfileLayout,
) -> None:
    paths = _legacy_profile_paths(
        release_dir=release_dir,
        stage_data=stage_data,
        worker_dropin_dir=worker_dropin_dir,
        layout=layout,
    )
    if paths is None:
        return
    dropin, env_path = paths
    original_dropin = release_dir / "rollback" / "10-secure-loopback.conf.original"
    original_env = release_dir / "rollback" / "legacy-worker-secure.env.original"
    _copy_secret(original_dropin, dropin)
    os.chmod(dropin, int(stage_data.get("legacy_loopback_dropin_mode") or 0o644))
    _copy_secret(original_env, env_path)
    if (
        _sha256_file(dropin) != stage_data["legacy_loopback_dropin_sha256"]
        or _sha256_file(env_path) != stage_data["legacy_loopback_env_sha256"]
    ):
        raise WorkerReleaseError("restored legacy secure profile failed verification")
    for disabled in (
        release_dir / "rollback" / "10-secure-loopback.conf.disabled",
        release_dir / "rollback" / "legacy-worker-secure.env.disabled",
    ):
        disabled.unlink(missing_ok=True)


@_locked_profile_operation
def cutover_secure_profile(
    *,
    release_id: str,
    receipt_path: Path,
    layout: SecureProfileLayout = SecureProfileLayout(),
    controller: ServiceController | None = None,
    now_epoch: int | None = None,
) -> None:
    """Remove the public override only after the coordinated readiness receipt."""

    if not _SAFE_ID.fullmatch(release_id):
        raise WorkerReleaseError("invalid secure-profile release id")
    release_dir = layout.releases_root / release_id
    stage_data = _stage_data(release_dir)
    if stage_data.get("release_id") != release_id:
        raise WorkerReleaseError("staged release id does not match")
    if stage_data.get("rollback_state") == "in_progress":
        try:
            _rollback_secure_profile_unlocked(
                release_id=release_id,
                layout=layout,
                controller=controller,
            )
        except Exception as rollback_exc:
            raise WorkerReleaseError(
                "prior secure cutover rollback still requires retry"
            ) from rollback_exc
        raise WorkerReleaseError(
            "prior secure cutover failure recovery completed; safe state restored"
        )
    boot_authorization = release_dir / "boot-authorized"
    boot_recovery = False
    if boot_authorization.exists() or boot_authorization.is_symlink():
        if (
            boot_authorization.is_symlink()
            or not boot_authorization.is_file()
            or stat.S_IMODE(boot_authorization.stat().st_mode) != 0o600
            or boot_authorization.read_text() != release_id + "\n"
        ):
            raise WorkerReleaseError("boot authorization drifted")
        boot_recovery = stage_data.get("profile_mode") == "first-install"
    cutover_recovery = stage_data.get("cutover_state") == "in_progress"
    if not boot_recovery and not cutover_recovery:
        _load_cutover_receipt(
            receipt_path,
            stage_data=stage_data,
            now_epoch=int(time.time()) if now_epoch is None else now_epoch,
        )
    _verify_staged_release(release_dir, stage_data)
    worker_unit = str(stage_data["worker_unit"])
    tunnel_unit = str(stage_data["tunnel_unit"])
    worker_port = int(stage_data["worker_port"])
    worker_dropin_path = (
        layout.systemd_root / f"{worker_unit}.d" / PROFILE_DROPIN_NAME
    )
    expected_worker_dropin = release_dir / "worker-secure-profile.conf"
    already_switched = (
        worker_dropin_path.is_symlink()
        and worker_dropin_path.resolve() == expected_worker_dropin.resolve()
    )
    worker_dropin, tunnel_dropin, tunnel_unit_path = _assert_active_profile_links(
        release_dir=release_dir,
        stage_data=stage_data,
        layout=layout,
        require_worker_profile=already_switched,
    )
    profile_mode = str(stage_data["profile_mode"])
    override_path = worker_dropin.parent / PUBLIC_OVERRIDE_NAME
    legacy_path = worker_dropin.parent / "10-secure-loopback.conf"
    if already_switched:
        transition_incomplete = (
            profile_mode == "migration"
            and (
                override_path.exists()
                or override_path.is_symlink()
                or legacy_path.exists()
                or legacy_path.is_symlink()
            )
        )
        override = _validate_profile_dropins(
            worker_dropin_dir=worker_dropin.parent,
            tunnel_dropin_dir=tunnel_dropin.parent,
            tunnel_unit_path=tunnel_unit_path,
            layout=layout,
            profile_mode="migration" if transition_incomplete else "first-install",
            allow_legacy_loopback=transition_incomplete,
            indexed_worker_units=tuple(_stage_indexed_units(stage_data)),
        )
    else:
        override = _validate_profile_dropins(
            worker_dropin_dir=worker_dropin.parent,
            tunnel_dropin_dir=tunnel_dropin.parent,
            tunnel_unit_path=tunnel_unit_path,
            layout=layout,
            profile_mode=profile_mode,
            allow_legacy_loopback=stage_data.get("legacy_loopback_dropin_sha256") is not None,
            indexed_worker_units=tuple(_stage_indexed_units(stage_data)),
        )
    if override is not None and _sha256_file(override) != stage_data.get(
        "public_override_sha256"
    ):
        raise WorkerReleaseError("public override changed after secure profile was staged")
    if profile_mode == "migration":
        _verified_public_override_backup(release_dir, stage_data)

    service_controller = controller or SystemdServiceController()
    if cutover_recovery:
        try:
            service_controller.assert_active(tunnel_unit)
        except Exception as inactive_tunnel_exc:
            _atomic_symlink(
                release_dir,
                layout.state_root / "active" / worker_unit,
            )
            stage_data["cutover_performed"] = True
            stage_data["rollback_state"] = "in_progress"
            stage_data["cutover_failure_at_epoch"] = (
                int(time.time()) if now_epoch is None else now_epoch
            )
            _write_text(
                release_dir / "stage-receipt.json",
                json.dumps(stage_data, sort_keys=True, indent=2) + "\n",
                mode=0o600,
            )
            try:
                _rollback_secure_profile_unlocked(
                    release_id=release_id,
                    layout=layout,
                    controller=service_controller,
                )
            except Exception as rollback_exc:
                raise WorkerReleaseError(
                    "interrupted cutover found an inactive tunnel and rollback requires retry"
                ) from rollback_exc
            raise WorkerReleaseError(
                "interrupted cutover found an inactive tunnel; safe state restored"
            ) from inactive_tunnel_exc
    else:
        service_controller.assert_active(tunnel_unit)
    service_controller.assert_unit_loaded(
        tunnel_unit,
        fragment_path=tunnel_unit_path,
        dropin_paths=[tunnel_dropin],
    )
    disabled_override = release_dir / "rollback" / f"{PUBLIC_OVERRIDE_NAME}.disabled"
    authorization = release_dir / "cutover-authorized"
    stage_data["cutover_state"] = "in_progress"
    stage_data.setdefault(
        "cutover_started_at_epoch",
        int(time.time()) if now_epoch is None else now_epoch,
    )
    _write_text(
        release_dir / "stage-receipt.json",
        json.dumps(stage_data, sort_keys=True, indent=2) + "\n",
        mode=0o600,
    )
    try:
        _disable_legacy_profile(
            release_dir=release_dir,
            stage_data=stage_data,
            worker_dropin_dir=worker_dropin.parent,
            layout=layout,
        )
        if not already_switched:
            _atomic_symlink(expected_worker_dropin, worker_dropin)
        # ADR-0009: indexed worker links install idempotently on every cutover
        # pass — a crash between link writes must not strand a rerun, and
        # _atomic_symlink is a safe overwrite toward the same target.
        indexed_units_by_dropin: dict[Path, str] = {}
        for _idx, _unit in enumerate(_stage_indexed_units(stage_data), start=1):
            _indexed_dropin = layout.systemd_root / f"{_unit}.d" / PROFILE_DROPIN_NAME
            _atomic_symlink(
                release_dir / f"worker-secure-profile-gpu{_idx}.conf", _indexed_dropin
            )
            indexed_units_by_dropin[_indexed_dropin] = _unit
        if not authorization.exists():
            _write_text(authorization, release_id + "\n", mode=0o600)
        elif (
            authorization.is_symlink()
            or not authorization.is_file()
            or stat.S_IMODE(authorization.stat().st_mode) != 0o600
            or authorization.read_text() != release_id + "\n"
        ):
            raise WorkerReleaseError("cutover authorization drifted")
        if override is not None:
            os.replace(override, disabled_override)
        _assert_active_profile_links(
            release_dir=release_dir,
            stage_data=stage_data,
            layout=layout,
        )
        _validate_profile_dropins(
            worker_dropin_dir=worker_dropin.parent,
            tunnel_dropin_dir=tunnel_dropin.parent,
            tunnel_unit_path=tunnel_unit_path,
            layout=layout,
            profile_mode="first-install",
            indexed_worker_units=tuple(_stage_indexed_units(stage_data)),
        )
        service_controller.daemon_reload()
        worker_allowed_dropins = [worker_dropin]
        if profile_mode == "first-install":
            worker_allowed_dropins.insert(
                0,
                worker_dropin.parent / STAGED_GUARD_DROPIN_NAME,
            )
        service_controller.assert_unit_loaded(
            worker_unit,
            fragment_path=layout.systemd_root / worker_unit,
            dropin_paths=worker_allowed_dropins,
        )
        service_controller.restart(worker_unit)
        # Public port closure is checked only after the loopback profile has
        # restarted, never before the verified receipt and override removal.
        service_controller.assert_loopback_only(worker_port)
        for _offset, (_indexed_dropin, _unit) in enumerate(
            indexed_units_by_dropin.items(), start=1
        ):
            _indexed_allowed = [_indexed_dropin]
            if profile_mode == "first-install":
                _indexed_allowed.insert(
                    0, _indexed_dropin.parent / STAGED_GUARD_DROPIN_NAME
                )
            service_controller.assert_unit_loaded(
                _unit,
                fragment_path=layout.systemd_root / _unit,
                dropin_paths=_indexed_allowed,
            )
            service_controller.restart(_unit)
            service_controller.assert_loopback_only(worker_port + _offset)
        probe_env = _strict_secret_env(release_dir / "backend-cutover-probe.env")
        service_controller.assert_authenticated_backend_route(
            probe_url=probe_env["FILMFORGE_BACKEND_CUTOVER_PROBE_URL"],
            probe_token=probe_env["FILMFORGE_BACKEND_CUTOVER_PROBE_TOKEN"],
            release_id=release_id,
            worker_code_release_id=str(stage_data["worker_code_release_id"]),
            worker_dependency_freeze_sha256=str(
                stage_data["worker_dependency_freeze_sha256"]
            ),
            worker_public_url=str(stage_data["worker_public_url"]),
        )
        if profile_mode == "first-install":
            # This durable marker is written only after the authenticated route
            # proof and before either unit is enabled. A power loss can therefore
            # never boot an enabled candidate lacking a recoverable proof.
            _write_text(boot_authorization, release_id + "\n", mode=0o600)
            # Provision-only deliberately leaves both services disabled.  Make
            # boot persistence part of the verified cutover transaction, after
            # the authenticated route proof and before publishing active CAS.
            service_controller.enable(tunnel_unit)
            service_controller.enable(worker_unit)
            service_controller.assert_enabled(tunnel_unit)
            service_controller.assert_enabled(worker_unit)
            for _unit in indexed_units_by_dropin.values():
                service_controller.enable(_unit)
                service_controller.assert_enabled(_unit)
    except Exception as exc:
        active_profile = layout.state_root / "active" / worker_unit
        _atomic_symlink(release_dir, active_profile)
        stage_data["cutover_performed"] = True
        stage_data["rollback_state"] = "in_progress"
        stage_data["cutover_failure_at_epoch"] = (
            int(time.time()) if now_epoch is None else now_epoch
        )
        _write_text(
            release_dir / "stage-receipt.json",
            json.dumps(stage_data, sort_keys=True, indent=2) + "\n",
            mode=0o600,
        )
        try:
            _rollback_secure_profile_unlocked(
                release_id=release_id,
                layout=layout,
                controller=service_controller,
            )
        except Exception as rollback_exc:
            raise WorkerReleaseError(
                "secure cutover failed and automatic safe-state rollback also failed"
            ) from rollback_exc
        raise WorkerReleaseError(
            "secure cutover failed; the prior safe state was restored"
        ) from exc

    # Publish the active CAS pointer before the receipt bit. If receipt writing
    # is interrupted, rerunning cutover recognizes the exact managed links,
    # repeats the authenticated proof, and completes this idempotent commit.
    _atomic_symlink(release_dir, layout.state_root / "active" / worker_unit)
    stage_data["cutover_performed"] = True
    stage_data["cutover_state"] = "complete"
    stage_data["cutover_at_epoch"] = int(time.time()) if now_epoch is None else now_epoch
    _write_text(
        release_dir / "stage-receipt.json",
        json.dumps(stage_data, sort_keys=True, indent=2) + "\n",
        mode=0o600,
    )


def _assert_managed_link_or_absent(path: Path, expected: Path, *, label: str) -> bool:
    if path.is_symlink():
        if path.resolve() != expected.resolve():
            raise WorkerReleaseError(
                f"secure-profile rollback release is not active: {label} points elsewhere"
            )
        return True
    if path.exists():
        raise WorkerReleaseError(
            f"secure-profile rollback release is not active: {label} is unmanaged"
        )
    return False


@contextmanager
def _code_release_transaction_lock(releases_root: Path):
    """Use the same kernel lock file as generated install/activate scripts."""

    releases_root.mkdir(parents=True, exist_ok=True)
    lock_path = releases_root / ".release.lock"
    if lock_path.is_symlink():
        raise WorkerReleaseError("worker code release lock file is unsafe")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise WorkerReleaseError("worker code release lock is unavailable") from exc
    try:
        os.fchmod(descriptor, 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise WorkerReleaseError("worker code release lock file is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _assert_rollback_code_release(path: Path, releases_root: Path) -> None:
    releases_dir = releases_root / "releases"
    if (
        path.parent.resolve() != releases_dir.resolve()
        or not path.is_dir()
        or not (path / ".ready").is_file()
        or not (path / ".dependency-freeze.txt").is_file()
        or not (path / ".dependency-freeze.sha256").is_file()
    ):
        raise WorkerReleaseError("previous worker code release is incomplete")
    expected_freeze = (path / ".dependency-freeze.sha256").read_text().strip()
    if expected_freeze != _sha256_file(path / ".dependency-freeze.txt"):
        raise WorkerReleaseError("previous worker dependency snapshot drifted")
    for candidate in (path, *path.rglob("*")):
        if not candidate.is_symlink() and candidate.stat().st_mode & 0o222:
            raise WorkerReleaseError("previous worker code release became writable")


def _rollback_code_pointer_for_profile(stage_data: dict[str, object]) -> None:
    """CAS ``current`` away from a finalized profile's exact candidate."""

    code_release = Path(str(stage_data["worker_module_dir"]))
    release_id = str(stage_data["worker_code_release_id"])
    if code_release.name != release_id or code_release.parent.name != "releases":
        raise WorkerReleaseError("secure profile has an invalid code release root")
    releases_root = code_release.parent.parent
    current = releases_root / "current"
    previous = releases_root / "previous"
    with _code_release_transaction_lock(releases_root):
        if current.is_symlink():
            current_target = current.resolve()
            if current_target != code_release.resolve():
                # Cutover can be rolled back before finalize; in that valid
                # state current still names the older release.
                return
            if previous.is_symlink():
                previous_target = previous.resolve()
                if previous_target == code_release.resolve():
                    raise WorkerReleaseError("previous code pointer names failed release")
                _assert_rollback_code_release(previous_target, releases_root)
                _atomic_symlink(previous_target, current)
            elif previous.exists():
                raise WorkerReleaseError("previous code pointer is not a symlink")
            else:
                current.unlink()
        elif current.exists():
            raise WorkerReleaseError("current code pointer is not a symlink")


def _rollback_secure_profile_unlocked(
    *,
    release_id: str,
    layout: SecureProfileLayout = SecureProfileLayout(),
    controller: ServiceController | None = None,
    prevalidate_only: bool = False,
) -> None:
    """Idempotently restore the verified pre-cutover state and code CAS."""

    if not _SAFE_ID.fullmatch(release_id):
        raise WorkerReleaseError("invalid secure-profile release id")
    release_dir = layout.releases_root / release_id
    stage_data = _stage_data(release_dir)
    worker_unit = str(stage_data["worker_unit"])
    tunnel_unit = str(stage_data["tunnel_unit"])
    worker_port = int(stage_data["worker_port"])
    profile_mode = str(stage_data["profile_mode"])
    worker_dropin = layout.systemd_root / f"{worker_unit}.d" / PROFILE_DROPIN_NAME
    tunnel_dropin = layout.systemd_root / f"{tunnel_unit}.d" / PROFILE_DROPIN_NAME
    tunnel_unit_path = layout.systemd_root / tunnel_unit
    active_profile = layout.state_root / "active" / worker_unit
    guard = worker_dropin.parent / STAGED_GUARD_DROPIN_NAME
    authorization = release_dir / "cutover-authorized"
    boot_authorization = release_dir / "boot-authorized"
    rollback_in_progress = stage_data.get("rollback_state") == "in_progress"
    rollback_complete = (
        stage_data.get("rollback_state") == "complete"
        and stage_data.get("cutover_performed") is False
    )
    if stage_data.get("rollback_state") not in {None, "in_progress", "complete"}:
        raise WorkerReleaseError("secure-profile rollback journal is invalid")

    link_state = {
        "worker profile": _assert_managed_link_or_absent(
            worker_dropin,
            release_dir / "worker-secure-profile.conf",
            label="worker profile",
        ),
        "tunnel profile": _assert_managed_link_or_absent(
            tunnel_dropin,
            release_dir / "tunnel-secure-profile.conf",
            label="tunnel profile",
        ),
        "tunnel unit": _assert_managed_link_or_absent(
            tunnel_unit_path,
            _tunnel_unit_artifact(release_dir, stage_data),
            label="tunnel unit",
        ),
        "active profile": _assert_managed_link_or_absent(
            active_profile,
            release_dir,
            label="active profile",
        ),
    }
    if not rollback_in_progress and not rollback_complete and not all(link_state.values()):
        raise WorkerReleaseError("secure-profile rollback release is not fully active")
    guard_present = False
    if profile_mode == "first-install":
        guard_present = _assert_managed_link_or_absent(
            guard,
            release_dir / "worker-staged-guard.conf",
            label="first-install guard",
        )
        if not rollback_in_progress and not rollback_complete and not guard_present:
            raise WorkerReleaseError("first-install rollback guard is missing")
        if boot_authorization.exists() or boot_authorization.is_symlink():
            if (
                boot_authorization.is_symlink()
                or not boot_authorization.is_file()
                or stat.S_IMODE(boot_authorization.stat().st_mode) != 0o600
                or boot_authorization.read_text() != release_id + "\n"
            ):
                raise WorkerReleaseError("first-install boot authorization drifted")
        elif (
            stage_data.get("cutover_performed") is True
            and not rollback_in_progress
            and not rollback_complete
        ):
            raise WorkerReleaseError("first-install boot authorization is missing")
    elif guard.exists() or guard.is_symlink():
        raise WorkerReleaseError("migration profile has an unexpected first-install guard")

    if authorization.exists() or authorization.is_symlink():
        if (
            authorization.is_symlink()
            or not authorization.is_file()
            or stat.S_IMODE(authorization.stat().st_mode) != 0o600
            or authorization.read_text() != release_id + "\n"
        ):
            raise WorkerReleaseError("cutover authorization drifted before rollback")
    elif not rollback_in_progress and not rollback_complete:
        raise WorkerReleaseError("cutover authorization is missing before rollback")

    backup: Path | None = None
    override = worker_dropin.parent / PUBLIC_OVERRIDE_NAME
    if profile_mode == "migration":
        backup = _verified_public_override_backup(release_dir, stage_data)
        _legacy_profile_paths(
            release_dir=release_dir,
            stage_data=stage_data,
            worker_dropin_dir=worker_dropin.parent,
            layout=layout,
        )
        if override.exists() or override.is_symlink():
            if (
                override.is_symlink()
                or not override.is_file()
                or _sha256_file(override) != stage_data.get("public_override_sha256")
            ):
                raise WorkerReleaseError("restored public override drifted")
    elif override.exists() or override.is_symlink():
        raise WorkerReleaseError("first-install rollback found an unexpected public override")

    if prevalidate_only:
        return

    safe_links_absent = not any(link_state.values()) and not guard_present
    safe_markers_absent = not (
        authorization.exists()
        or authorization.is_symlink()
        or boot_authorization.exists()
        or boot_authorization.is_symlink()
    )
    if rollback_complete:
        if not safe_links_absent or not safe_markers_absent:
            raise WorkerReleaseError(
                "completed secure-profile rollback still has active managed state"
            )
        # Repeated outer rollback is a no-op after the inner cutover rollback
        # has already stopped/disabled services and removed their managed unit.
        # Re-run the code CAS check and canonicalize old/incomplete journal
        # fields without asking systemd to operate on a now-missing unit.
        _rollback_code_pointer_for_profile(stage_data)
        changed = False
        if stage_data.get("cutover_state") != "rolled_back":
            stage_data["cutover_state"] = "rolled_back"
            changed = True
        if not isinstance(stage_data.get("rolled_back_at_epoch"), int):
            stage_data["rolled_back_at_epoch"] = int(time.time())
            changed = True
        if changed:
            _write_text(
                release_dir / "stage-receipt.json",
                json.dumps(stage_data, sort_keys=True, indent=2) + "\n",
                mode=0o600,
            )
        return
    if (
        rollback_in_progress
        and stage_data.get("cutover_performed") is False
        and isinstance(stage_data.get("rolled_back_at_epoch"), int)
        and safe_links_absent
        and safe_markers_absent
    ):
        # A process may be interrupted after writing every durable safe-state
        # marker but before normalizing the journal seen by an outer rollback.
        # The final timestamp plus the absence of every managed link/authority
        # is the fail-closed completion proof; do not re-stop a removed unit.
        _rollback_code_pointer_for_profile(stage_data)
        stage_data["cutover_state"] = "rolled_back"
        stage_data["rollback_state"] = "complete"
        _write_text(
            release_dir / "stage-receipt.json",
            json.dumps(stage_data, sort_keys=True, indent=2) + "\n",
            mode=0o600,
        )
        return

    # Journal precedes the first destructive operation. Every subsequent step
    # accepts its own completed state, so SSH loss/systemd failure is retryable.
    stage_data["rollback_state"] = "in_progress"
    stage_data.setdefault("rollback_started_at_epoch", int(time.time()))
    _write_text(
        release_dir / "stage-receipt.json",
        json.dumps(stage_data, sort_keys=True, indent=2) + "\n",
        mode=0o600,
    )
    service_controller = controller or SystemdServiceController()
    if backup is not None and not override.exists():
        _copy_secret(backup, override)
    if backup is not None and _sha256_file(override) != stage_data.get(
        "public_override_sha256"
    ):
        raise WorkerReleaseError("restored public override failed verification")
    authorization.unlink(missing_ok=True)
    boot_authorization.unlink(missing_ok=True)
    if worker_dropin.is_symlink():
        worker_dropin.unlink()
    for _unit in _stage_indexed_units(stage_data):
        _indexed_dropin = worker_dropin.parent.parent / f"{_unit}.d" / PROFILE_DROPIN_NAME
        if _indexed_dropin.is_symlink():
            _indexed_dropin.unlink()
    _restore_legacy_profile(
        release_dir=release_dir,
        stage_data=stage_data,
        worker_dropin_dir=worker_dropin.parent,
        layout=layout,
    )
    service_controller.daemon_reload()
    if profile_mode == "migration":
        rollback_dropins = [override]
        if stage_data.get("legacy_loopback_dropin_sha256") is not None:
            rollback_dropins.insert(0, worker_dropin.parent / "10-secure-loopback.conf")
        service_controller.assert_unit_loaded(
            worker_unit,
            fragment_path=layout.systemd_root / worker_unit,
            dropin_paths=rollback_dropins,
        )
        service_controller.restart(worker_unit)
        service_controller.assert_active(worker_unit)
        service_controller.assert_public_listener(worker_port)
    else:
        service_controller.stop(worker_unit)
        service_controller.disable(worker_unit)
        for _unit in _stage_indexed_units(stage_data):
            service_controller.stop(_unit)
            service_controller.disable(_unit)
    # The managed tunnel link is removed only after stop+disable below. If an
    # interrupted retry finds it absent, those operations already completed;
    # asking systemd to stop an absent linked unit would turn safe rollback into
    # a false failure.
    if link_state["tunnel unit"]:
        service_controller.stop(tunnel_unit)
        if profile_mode == "first-install":
            service_controller.disable(tunnel_unit)
    for managed in (tunnel_dropin, tunnel_unit_path):
        if managed.is_symlink():
            managed.unlink()
    if guard.is_symlink():
        guard.unlink()
    for _unit in _stage_indexed_units(stage_data):
        _indexed_guard = (
            worker_dropin.parent.parent / f"{_unit}.d" / STAGED_GUARD_DROPIN_NAME
        )
        if _indexed_guard.is_symlink():
            _indexed_guard.unlink()
    service_controller.daemon_reload()
    if active_profile.is_symlink():
        active_profile.unlink()
    _rollback_code_pointer_for_profile(stage_data)
    stage_data["cutover_performed"] = False
    stage_data["cutover_state"] = "rolled_back"
    stage_data["rollback_state"] = "complete"
    stage_data["rolled_back_at_epoch"] = int(time.time())
    _write_text(
        release_dir / "stage-receipt.json",
        json.dumps(stage_data, sort_keys=True, indent=2) + "\n",
        mode=0o600,
    )


@_locked_profile_operation
def rollback_secure_profile(
    *,
    release_id: str,
    layout: SecureProfileLayout = SecureProfileLayout(),
    controller: ServiceController | None = None,
) -> None:
    _rollback_secure_profile_unlocked(
        release_id=release_id,
        layout=layout,
        controller=controller,
    )


def _release_for_managed_target(
    path: Path,
    *,
    expected_name: str | None,
    layout: SecureProfileLayout,
) -> Path:
    """Resolve one managed symlink to its exact secure-profile release."""

    if not path.is_symlink():
        if path.exists():
            raise WorkerReleaseError(
                "rehydrated secure-profile repair found an unmanaged path"
            )
        raise WorkerReleaseError(
            "rehydrated secure-profile repair found a missing managed link"
        )
    target = path.resolve()
    release_dir = target if expected_name is None else target.parent
    if expected_name is not None and target.name != expected_name:
        raise WorkerReleaseError(
            "rehydrated secure-profile repair found an unexpected link target"
        )
    if (
        release_dir.parent.resolve() != layout.releases_root.resolve()
        or not _SAFE_ID.fullmatch(release_dir.name)
    ):
        raise WorkerReleaseError(
            "rehydrated secure-profile repair found a target outside managed releases"
        )
    return release_dir


@_locked_profile_operation
def retire_rehydrated_secure_profile(
    *,
    worker_unit: str,
    layout: SecureProfileLayout = SecureProfileLayout(),
    controller: ServiceController | None = None,
) -> str | None:
    """Retire the secure profile carried by a reused OS volume.

    A power loss during ``stage_secure_profile`` can leave the prior active
    profile intact while one shared tunnel link already names the new,
    never-cut-over release. Ordinary rollback correctly rejects that mixed
    compare-and-swap state. Rehydration is the one place where the old VM is
    known dead, so repair only this narrowly proven pre-cutover displacement,
    then run the ordinary journaled rollback.
    """

    if not _SAFE_UNIT.fullmatch(worker_unit):
        raise WorkerReleaseError("invalid rehydrated worker unit")
    active_pointer = layout.state_root / "active" / worker_unit
    if not active_pointer.exists() and not active_pointer.is_symlink():
        return None
    active_release = _release_for_managed_target(
        active_pointer,
        expected_name=None,
        layout=layout,
    )
    active_data = _stage_data(active_release)
    if active_data.get("worker_unit") != worker_unit:
        raise WorkerReleaseError(
            "rehydrated active pointer names a profile for another worker"
        )
    _verify_staged_release(active_release, active_data)

    tunnel_unit = str(active_data["tunnel_unit"])
    worker_dropin = (
        layout.systemd_root / f"{worker_unit}.d" / PROFILE_DROPIN_NAME
    )
    tunnel_dropin = (
        layout.systemd_root / f"{tunnel_unit}.d" / PROFILE_DROPIN_NAME
    )
    tunnel_unit_path = layout.systemd_root / tunnel_unit
    worker_owner = _release_for_managed_target(
        worker_dropin,
        expected_name="worker-secure-profile.conf",
        layout=layout,
    )
    if worker_owner != active_release:
        raise WorkerReleaseError(
            "rehydrated secure-profile repair refuses a displaced worker profile"
        )

    shared_links: list[tuple[str, Path, str]] = [
        ("tunnel profile", tunnel_dropin, "tunnel-secure-profile.conf"),
        (
            "tunnel unit",
            tunnel_unit_path,
            _tunnel_unit_artifact(active_release, active_data).name,
        ),
    ]
    if active_data.get("profile_mode") == "first-install":
        shared_links.append(
            (
                "first-install guard",
                worker_dropin.parent / STAGED_GUARD_DROPIN_NAME,
                "worker-staged-guard.conf",
            )
        )
        for unit in _stage_indexed_units(active_data):
            shared_links.append(
                (
                    f"indexed guard {unit}",
                    layout.systemd_root / f"{unit}.d" / STAGED_GUARD_DROPIN_NAME,
                    "worker-staged-guard.conf",
                )
            )

    foreign_releases: set[Path] = set()
    displaced: list[tuple[str, Path, str, Path]] = []
    for label, path, artifact in shared_links:
        owner = _release_for_managed_target(
            path,
            expected_name=artifact,
            layout=layout,
        )
        if owner != active_release:
            foreign_releases.add(owner)
            displaced.append((label, path, artifact, owner))

    if foreign_releases:
        if len(foreign_releases) != 1:
            raise WorkerReleaseError(
                "rehydrated secure-profile repair found multiple foreign releases"
            )
        foreign_release = next(iter(foreign_releases))
        foreign_data = _stage_data(foreign_release)
        _verify_staged_release(foreign_release, foreign_data)
        if (
            foreign_data.get("worker_unit") != worker_unit
            or foreign_data.get("tunnel_unit") != tunnel_unit
            or foreign_data.get("cutover_performed") is not False
            or foreign_data.get("cutover_state") is not None
            or foreign_data.get("rollback_state") is not None
            or (foreign_release / "cutover-authorized").exists()
            or (foreign_release / "boot-authorized").exists()
        ):
            raise WorkerReleaseError(
                "rehydrated secure-profile repair refuses a foreign release that reached cutover"
            )
        active_data["rehydrated_link_repair"] = {
            "foreign_release_id": foreign_release.name,
            "displaced_links": [label for label, *_rest in displaced],
            "state": "in_progress",
        }
        _write_text(
            active_release / "stage-receipt.json",
            json.dumps(active_data, sort_keys=True, indent=2) + "\n",
            mode=0o600,
        )
        for _label, path, artifact, _owner in displaced:
            _atomic_symlink(active_release / artifact, path)
        staged_pointer = layout.state_root / "staged" / worker_unit
        if staged_pointer.is_symlink():
            staged_owner = _release_for_managed_target(
                staged_pointer,
                expected_name=None,
                layout=layout,
            )
            if staged_owner == foreign_release:
                staged_pointer.unlink()
            elif staged_owner != active_release:
                raise WorkerReleaseError(
                    "rehydrated secure-profile repair found an unrelated staged pointer"
                )
        elif staged_pointer.exists():
            raise WorkerReleaseError(
                "rehydrated secure-profile repair found an unmanaged staged pointer"
            )
        active_data = _stage_data(active_release)
        repair = dict(active_data["rehydrated_link_repair"])
        repair["state"] = "complete"
        active_data["rehydrated_link_repair"] = repair
        _write_text(
            active_release / "stage-receipt.json",
            json.dumps(active_data, sort_keys=True, indent=2) + "\n",
            mode=0o600,
        )

    _rollback_secure_profile_unlocked(
        release_id=active_release.name,
        layout=layout,
        controller=controller,
    )
    return active_release.name


@_locked_profile_operation
def rollback_secure_profiles(
    *,
    release_ids: Sequence[str],
    layout: SecureProfileLayout = SecureProfileLayout(),
    controller: ServiceController | None = None,
) -> None:
    """Prevalidate and roll back a fleet under one profile→code lock order."""

    normalized = list(release_ids)
    if not normalized or len(set(normalized)) != len(normalized):
        raise WorkerReleaseError("secure-profile rollback batch is empty or duplicated")
    for release_id in normalized:
        _rollback_secure_profile_unlocked(
            release_id=release_id,
            layout=layout,
            controller=controller,
            prevalidate_only=True,
        )
    failures: list[str] = []
    for release_id in reversed(normalized):
        try:
            _rollback_secure_profile_unlocked(
                release_id=release_id,
                layout=layout,
                controller=controller,
            )
        except Exception as exc:
            failures.append(f"{release_id}: {type(exc).__name__}: {exc}")
    if failures:
        raise WorkerReleaseError(
            "one or more secure profiles require rollback retry: " + "; ".join(failures)
        )


def build_cutover_receipt_template(
    staged: StagedSecureProfile,
    *,
    issued_at_epoch: int | None = None,
) -> dict[str, object]:
    """Create a false-by-default template for the independent verifier."""

    stage_data = _stage_data(staged.release_dir)
    if stage_data.get("tunnel_prepared") is not True:
        raise WorkerReleaseError(
            "receipt template may be created only after prepare verifies the staged tunnel"
        )
    template = {
        "schema": RECEIPT_SCHEMA,
        "profile_mode": stage_data["profile_mode"],
        "release_id": stage_data["release_id"],
        "worker_code_release_id": stage_data["worker_code_release_id"],
        "worker_dependency_freeze_sha256": stage_data[
            "worker_dependency_freeze_sha256"
        ],
        "worker_unit": stage_data["worker_unit"],
        "tunnel_unit": stage_data["tunnel_unit"],
        "worker_port": stage_data["worker_port"],
        "worker_public_url": stage_data["worker_public_url"],
        "tunnel_local_url": stage_data["tunnel_local_url"],
        "worker_secret_sha256": stage_data["worker_secret_sha256"],
        "backend_probe_secret_sha256": stage_data["backend_probe_secret_sha256"],
        "worker_dropin_sha256": stage_data["worker_dropin_sha256"],
        "tunnel_dropin_sha256": stage_data["tunnel_dropin_sha256"],
        "tunnel_unit_sha256": stage_data["tunnel_unit_sha256"],
        "worker_guard_sha256": stage_data["worker_guard_sha256"],
        "contract_sha256": stage_data["contract_sha256"],
        "issued_at_epoch": int(time.time()) if issued_at_epoch is None else issued_at_epoch,
        "tunnel_ready": False,
        "backend_bearer_client_ready": False,
        "backend_registration_ready": False,
        "worker_secret_fingerprint_match": False,
    }
    if stage_data.get("edge_provider") == "caddy":
        template.update({
            "edge_provider": "caddy",
            "caddy_secret_sha256": stage_data["caddy_secret_sha256"],
            "caddy_config_sha256": stage_data["caddy_config_sha256"],
            "caddy_exec_sha256": stage_data["caddy_exec_sha256"],
            "caddy_binary_sha256": stage_data["caddy_binary_sha256"],
            "caddy_tls_hostname": stage_data["caddy_tls_hostname"],
            "caddy_https_ports": stage_data["caddy_https_ports"],
            "edge_tls_hostname_ready": False,
        })
    else:
        template.update({
            "tunnel_secret_sha256": stage_data["tunnel_secret_sha256"],
            "tunnel_config_sha256": stage_data["tunnel_config_sha256"],
            "tunnel_credential_sha256": stage_data["tunnel_credential_sha256"],
            "tunnel_exec_sha256": stage_data["tunnel_exec_sha256"],
            "tunnel_binary_sha256": stage_data["tunnel_binary_sha256"],
        })
    return template


__all__ = [
    "BACKEND_PROBE_SCHEMA",
    "MAX_RECEIPT_AGE_SECONDS",
    "PROFILE_DROPIN_NAME",
    "PUBLIC_OVERRIDE_NAME",
    "RECEIPT_SCHEMA",
    "STAGED_GUARD_DROPIN_NAME",
    "SecureProfileLayout",
    "SecureWorkerContract",
    "StagedSecureProfile",
    "SystemdServiceController",
    "WorkerReleaseBundle",
    "WorkerReleaseError",
    "build_cutover_receipt_template",
    "build_worker_release_bundle",
    "cutover_secure_profile",
    "prepare_secure_profile",
    "retire_rehydrated_secure_profile",
    "rollback_secure_profile",
    "rollback_secure_profiles",
    "stage_secure_profile",
    "validate_secure_worker_contract",
    "worker_release_install_script",
    "worker_release_activate_script",
    "worker_release_rollback_script",
]
