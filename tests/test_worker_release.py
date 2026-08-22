from __future__ import annotations

import json
import hashlib
import os
import stat
import subprocess
import sys
import tarfile
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from gpu_worker.worker_release import (
    PROFILE_DROPIN_NAME,
    PUBLIC_OVERRIDE_NAME,
    RECEIPT_SCHEMA,
    SecureProfileLayout,
    SecureWorkerContract,
    SystemdServiceController,
    WorkerReleaseError,
    build_cutover_receipt_template,
    build_worker_release_bundle,
    cutover_secure_profile,
    prepare_secure_profile,
    rollback_secure_profile,
    stage_secure_profile,
    worker_release_install_script,
    worker_release_activate_script,
    worker_release_rollback_script,
    _code_release_transaction_lock,
)


@pytest.mark.parametrize(
    ("state", "returncode"),
    [
        ("alias", 0),
        ("linked", 0),
        ("disabled", 1),
        ("not-found", 4),
    ],
)
def test_systemd_disabled_gate_accepts_non_boot_enabled_states(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    returncode: int,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=returncode, stdout=f"{state}\n", stderr=""
        ),
    )

    SystemdServiceController().assert_disabled("filmforge-worker-edge-gpu0.service")


@pytest.mark.parametrize("state", ["enabled", "enabled-runtime", "unexpected"])
def test_systemd_disabled_gate_rejects_enabled_or_unknown_states(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=f"{state}\n", stderr=""
        ),
    )

    with pytest.raises(WorkerReleaseError):
        SystemdServiceController().assert_disabled(
            "filmforge-worker-edge-gpu0.service"
        )


def test_caddy_launcher_uses_portable_secure_mktemp_templates() -> None:
    launcher = (
        Path(__file__).parents[1] / "deploy/bin/filmforge-worker-caddy"
    ).read_text()

    assert "mktemp -t" not in launcher
    assert "filmforge-caddy-watchdog.XXXXXX" in launcher
    assert "filmforge-caddy-health.XXXXXX" in launcher


def test_systemd_public_listener_waits_for_caddy_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = SystemdServiceController()
    outputs = iter(
        [
            "",
            "LISTEN 0 4096 *:80 *:*\n",
        ]
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        controller,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=next(outputs), stderr=""
        ),
    )
    monkeypatch.setattr("gpu_worker.worker_release.time.sleep", sleeps.append)

    controller.assert_public_listener(80)

    assert sleeps == [0.1]


def test_systemd_public_listener_fails_after_bounded_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = SystemdServiceController()
    calls = 0

    def no_listeners(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(controller, "_run", no_listeners)
    monkeypatch.setattr("gpu_worker.worker_release.time.sleep", lambda _: None)

    with pytest.raises(WorkerReleaseError, match="edge port 80"):
        controller.assert_public_listener(80)

    assert calls == 50


def test_systemd_loopback_listener_waits_for_uvicorn_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = SystemdServiceController()
    outputs = iter(["", "LISTEN 0 4096 127.0.0.1:9000 0.0.0.0:*\n"])
    sleeps: list[float] = []
    monkeypatch.setattr(
        controller,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=next(outputs), stderr=""
        ),
    )
    monkeypatch.setattr("gpu_worker.worker_release.time.sleep", sleeps.append)

    controller.assert_loopback_only(9000)

    assert sleeps == [0.1]


def test_systemd_loopback_listener_rejects_public_bind_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = SystemdServiceController()
    calls = 0

    def public_listener(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="LISTEN 0 4096 0.0.0.0:9000 0.0.0.0:*\n",
            stderr="",
        )

    monkeypatch.setattr(controller, "_run", public_listener)
    monkeypatch.setattr("gpu_worker.worker_release.time.sleep", lambda _: None)

    with pytest.raises(WorkerReleaseError, match="publicly bound"):
        controller.assert_loopback_only(9000)

    assert calls == 1


def _write_0600(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o600)
    return path


def _fixture(
    tmp_path: Path,
    *,
    worker_env: str | None = None,
    tunnel_env: str | None = None,
    with_override: bool = True,
    with_tunnel_exec: bool = True,
    profile_mode: str = "migration",
) -> tuple[SecureWorkerContract, SecureProfileLayout]:
    systemd_root = tmp_path / "systemd"
    state_root = tmp_path / "state"
    worker_unit = "filmforge-worker-gpu0.service"
    tunnel_unit = "filmforge-worker-tunnel-gpu0.service"
    public_url = "https://gpu0.worker.example"
    code_release_id = "sha256-" + "a" * 24
    code_release_root = tmp_path / "code-releases" / "releases" / code_release_id
    (code_release_root / ".venv/bin").mkdir(parents=True)
    worker_python = code_release_root / ".venv/bin/python"
    worker_python.write_text("#!/bin/sh\nexit 0\n")
    worker_python.chmod(0o555)
    ready = code_release_root / ".ready"
    ready.write_text("a" * 64 + "\n")
    ready.chmod(0o444)
    source_marker = code_release_root / ".source-sha256"
    source_marker.write_text("a" * 64 + "\n")
    source_marker.chmod(0o444)
    dependency_freeze = code_release_root / ".dependency-freeze.txt"
    dependency_freeze.write_text("")
    dependency_freeze.chmod(0o444)
    dependency_marker = code_release_root / ".dependency-freeze.sha256"
    dependency_marker.write_text(hashlib.sha256(b"").hexdigest() + "\n")
    dependency_marker.chmod(0o444)
    worker_dropins = systemd_root / f"{worker_unit}.d"
    worker_dropins.mkdir(parents=True)
    if with_override:
        (worker_dropins / PUBLIC_OVERRIDE_NAME).write_text(
            "[Service]\nEnvironment=WORKER_API_AUTH_MODE=development\n"
            "ExecStart=\nExecStart=/usr/bin/worker --host 0.0.0.0 --port 9000\n"
        )
    tunnel_credential = _write_0600(
        tmp_path / "secrets" / "cloudflared.json",
        '{"fixture": true}\n',
    )
    tunnel_config = _write_0600(
        tmp_path / "secrets" / "cloudflared.yml",
        "tunnel: fixture\n"
        f"credentials-file: {tunnel_credential}\n"
        "ingress:\n"
        "  - hostname: gpu0.worker.example\n"
        "    service: http://127.0.0.1:9000\n"
        "  - service: http_status:404\n",
    )
    tunnel_exec = tmp_path / "filmforge-worker-tunnel"
    if with_tunnel_exec:
        tunnel_exec.write_text(
            "#!/bin/sh\n"
            ': "${FILMFORGE_TUNNEL_LOCAL_URL:?}"\n'
            ': "${FILMFORGE_TUNNEL_PUBLIC_URL:?}"\n'
            ': "${FILMFORGE_TUNNEL_WORKER_SECRET_FILE:?}"\n'
            "curl --config /protected/watchdog.conf\n"
            "exec /usr/bin/cloudflared tunnel run\n"
        )
        tunnel_exec.chmod(0o755)
    tunnel_binary = tmp_path / "cloudflared"
    tunnel_binary.write_text("#!/bin/sh\nexit 0\n")
    tunnel_binary.chmod(0o755)
    worker_env_path = _write_0600(
        tmp_path / "secrets" / "worker.env",
        worker_env
        if worker_env is not None
        else (
            "GPU_WORKER_API_TOKEN=super-private-worker-token\n"
            "WORKER_REGISTRATION_TOKEN=super-private-registration-token\n"
            "FILMFORGE_BACKEND_URL=https://backend.example\n"
            "WORKER_API_AUTH_MODE=required\n"
            f"WORKER_PUBLIC_URL={public_url}\n"
        ),
    )
    tunnel_env_path = _write_0600(
        tmp_path / "secrets" / "tunnel.env",
        tunnel_env
        if tunnel_env is not None
        else (
            f"FILMFORGE_TUNNEL_CONFIG_FILE={tunnel_config}\n"
            f"FILMFORGE_TUNNEL_CREDENTIAL_FILE={tunnel_credential}\n"
            "FILMFORGE_TUNNEL_LOCAL_URL=http://127.0.0.1:9000\n"
            f"FILMFORGE_TUNNEL_PUBLIC_URL={public_url}\n"
            f"FILMFORGE_TUNNEL_UNIT={tunnel_unit}\n"
        ),
    )
    backend_probe_env_path = _write_0600(
        tmp_path / "secrets" / "backend-probe.env",
        "FILMFORGE_BACKEND_CUTOVER_PROBE_URL="
        "https://backend.example/api/internal/worker-cutover-probe\n"
        "FILMFORGE_BACKEND_CUTOVER_PROBE_TOKEN=fixture-probe-token-123456\n",
    )
    contract = SecureWorkerContract(
        release_id="release-2026-08-10-a",
        worker_code_release_id=code_release_id,
        worker_unit=worker_unit,
        tunnel_unit=tunnel_unit,
        worker_port=9000,
        worker_public_url=public_url,
        tunnel_local_url="http://127.0.0.1:9000",
        worker_exec=worker_python,
        worker_module_dir=code_release_root,
        worker_secret_source=worker_env_path,
        tunnel_secret_source=tunnel_env_path,
        backend_probe_secret_source=backend_probe_env_path,
        tunnel_exec_source=tunnel_exec,
        tunnel_binary_source=tunnel_binary,
        profile_mode=profile_mode,
    )
    return contract, SecureProfileLayout(
        systemd_root=systemd_root,
        state_root=state_root,
    )


def _ready_receipt(staged, path: Path, *, now: int = 1_800_000_000) -> Path:
    receipt = build_cutover_receipt_template(staged, issued_at_epoch=now)
    receipt.update(
        {
            "tunnel_ready": True,
            "backend_bearer_client_ready": True,
            "backend_registration_ready": True,
            "worker_secret_fingerprint_match": True,
        }
    )
    return _write_0600(path, json.dumps(receipt))


def _first_install_receipt(staged, path: Path, *, now: int = 1_800_000_000) -> Path:
    receipt = build_cutover_receipt_template(staged, issued_at_epoch=now)
    receipt.update(
        {
            "tunnel_ready": True,
            # A guarded first-install process has not started its registration
            # loop yet.  The cutover's post-start backend probe proves both.
            "backend_bearer_client_ready": False,
            "backend_registration_ready": False,
            "worker_secret_fingerprint_match": True,
        }
    )
    return _write_0600(path, json.dumps(receipt))


def test_caddy_edge_is_versioned_and_requires_tls_receipt(tmp_path: Path) -> None:
    contract, layout = _fixture(tmp_path)
    caddy_config = _write_0600(
        tmp_path / "secrets" / "Caddyfile",
        "gpu0.worker.example {\n    reverse_proxy http://127.0.0.1:9000\n}\n",
    )
    caddy_env = _write_0600(
        tmp_path / "secrets" / "caddy.env",
        f"FILMFORGE_CADDY_CONFIG_FILE={caddy_config}\n"
        "FILMFORGE_CADDY_LOCAL_URL=http://127.0.0.1:9000\n"
        "FILMFORGE_CADDY_PUBLIC_URL=https://gpu0.worker.example\n"
        f"FILMFORGE_CADDY_UNIT={contract.tunnel_unit}\n",
    )
    caddy_exec = tmp_path / "filmforge-worker-caddy"
    caddy_exec.write_text(
        ': "${FILMFORGE_CADDY_LOCAL_URL:?}"\n'
        ': "${FILMFORGE_CADDY_PUBLIC_URL:?}"\n'
        ': "${FILMFORGE_EDGE_WORKER_SECRET_FILE:?}"\n'
        "curl --config /protected/watchdog.conf\n"
    )
    caddy_exec.chmod(0o755)
    caddy_binary = tmp_path / "caddy"
    caddy_binary.write_text("#!/bin/sh\nexit 0\n")
    caddy_binary.chmod(0o755)
    contract = replace(
        contract,
        edge_provider="caddy",
        tunnel_secret_source=caddy_env,
        tunnel_exec_source=caddy_exec,
        tunnel_binary_source=caddy_binary,
    )
    staged = stage_secure_profile(contract, layout)
    data = json.loads(staged.stage_receipt.read_text())
    assert data["edge_provider"] == "caddy"
    assert data["caddy_https_ports"] == [80, 443]
    assert (staged.release_dir / "Caddyfile").is_file()
    assert (staged.release_dir / "caddy").stat().st_mode & 0o111
    controller = _Controller(layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME)
    prepare_secure_profile(release_id=contract.release_id, layout=layout, controller=controller)
    assert ("restart", contract.tunnel_unit, True) in controller.events
    receipt = build_cutover_receipt_template(staged, issued_at_epoch=1_800_000_000)
    receipt.update({
        "tunnel_ready": True,
        "backend_bearer_client_ready": True,
        "backend_registration_ready": True,
        "worker_secret_fingerprint_match": True,
    })
    path = _write_0600(tmp_path / "receipt.json", json.dumps(receipt))
    with pytest.raises(WorkerReleaseError, match="TLS hostname"):
        cutover_secure_profile(
            release_id=contract.release_id, receipt_path=path, layout=layout,
            controller=controller, now_epoch=1_800_000_000,
        )
    receipt["edge_tls_hostname_ready"] = True
    _write_0600(path, json.dumps(receipt))
    cutover_secure_profile(
        release_id=contract.release_id, receipt_path=path, layout=layout,
        controller=controller, now_epoch=1_800_000_000,
    )
    assert json.loads(staged.stage_receipt.read_text())["cutover_performed"] is True


class _Controller:
    def __init__(
        self,
        override: Path,
        *,
        fail_loopback: bool = False,
        fail_backend_probe: bool = False,
        fail_stop_once: bool = False,
        fail_active_once: bool = False,
    ) -> None:
        self.override = override
        self.fail_loopback = fail_loopback
        self.fail_backend_probe = fail_backend_probe
        self.fail_stop_once = fail_stop_once
        self.fail_active_once = fail_active_once
        self.events: list[tuple[str, object, bool]] = []
        self.enabled_units: set[str] = set()
        self.enablement_events: list[tuple[str, str]] = []

    def assert_active(self, unit: str) -> None:
        self.events.append(("tunnel-active", unit, self.override.exists()))
        if self.fail_active_once:
            self.fail_active_once = False
            raise RuntimeError("fixture tunnel inactive after reboot")

    def assert_inactive(self, unit: str) -> None:
        self.events.append(("worker-inactive", unit, self.override.exists()))

    def assert_disabled(self, unit: str) -> None:
        assert unit not in self.enabled_units

    def assert_enabled(self, unit: str) -> None:
        assert unit in self.enabled_units

    def daemon_reload(self) -> None:
        self.events.append(("daemon-reload", None, self.override.exists()))

    def restart(self, unit: str) -> None:
        self.events.append(("restart", unit, self.override.exists()))

    def enable(self, unit: str) -> None:
        self.enabled_units.add(unit)
        self.enablement_events.append(("enable", unit))

    def disable(self, unit: str) -> None:
        self.enabled_units.discard(unit)
        self.enablement_events.append(("disable", unit))

    def stop(self, unit: str) -> None:
        self.events.append(("stop", unit, self.override.exists()))
        if self.fail_stop_once:
            self.fail_stop_once = False
            raise RuntimeError("fixture interrupted rollback stop")

    def assert_loopback_only(self, port: int) -> None:
        self.events.append(("loopback-only", port, self.override.exists()))
        if self.fail_loopback:
            raise RuntimeError("fixture public listener remains")

    def assert_public_listener(self, port: int) -> None:
        self.events.append(("public-listener", port, self.override.exists()))

    def assert_unit_loaded(
        self,
        unit: str,
        *,
        fragment_path: Path,
        dropin_paths: list[Path],
    ) -> None:
        self.events.append(("unit-loaded", unit, self.override.exists()))

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
        assert probe_token == "fixture-probe-token-123456"
        assert worker_code_release_id == "sha256-" + "a" * 24
        assert worker_dependency_freeze_sha256 == hashlib.sha256(b"").hexdigest()
        self.events.append(("backend-route", worker_public_url, self.override.exists()))
        if self.fail_backend_probe:
            raise RuntimeError("backend could not authenticate worker route")


def _prepare(staged, contract, layout, controller: _Controller, *, now: int) -> None:
    prepare_secure_profile(
        release_id=contract.release_id,
        layout=layout,
        controller=controller,
        now_epoch=now,
    )
    controller.events.clear()


def test_bundle_is_content_addressed_and_excludes_git_venv_and_secrets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("APP = 1\n")
    (source / "requirements.txt").write_text("fastapi\n")
    (source / "requirements.lock").write_text("fastapi==0.139.0\n")
    (source / "module.py").write_text("VALUE = 2\n")
    (source / ".env").write_text("TOKEN=must-not-ship\n")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("credential=must-not-ship\n")
    (source / ".venv").mkdir()
    (source / ".venv" / "secret").write_text("must-not-ship\n")
    (source / "worker-secrets.env").write_text("TOKEN=must-not-ship\n")
    (source / "cloudflare-credential.json").write_text("must-not-ship\n")
    (source / "private.key").write_text("must-not-ship\n")

    with build_worker_release_bundle(source) as first:
        with build_worker_release_bundle(source) as second:
            assert first.release_id == second.release_id
            assert first.source_sha256 == second.source_sha256
        with tarfile.open(first.archive_path, "r:gz") as archive:
            names = archive.getnames()
            assert "gpu_worker/app.py" in names
            assert "gpu_worker/module.py" in names
            assert not any(".git" in name for name in names)
            assert not any(".venv" in name for name in names)
            assert not any(name.endswith(".env") for name in names)
            assert not any("credential" in name for name in names)
            assert not any(name.endswith(".key") for name in names)
            assert b"must-not-ship" not in first.archive_path.read_bytes()


def test_release_id_changes_when_executable_contract_changes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("APP = 1\n")
    (source / "requirements.txt").write_text("")
    (source / "requirements.lock").write_text("")
    launcher = source / "launcher"
    launcher.write_text("#!/bin/sh\nexit 0\n")
    launcher.chmod(0o644)
    with build_worker_release_bundle(source) as first:
        first_id = first.release_id
    launcher.chmod(0o755)
    with build_worker_release_bundle(source) as second:
        assert second.release_id != first_id


def test_production_bundle_requires_one_clean_tracked_commit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("APP = 1\n")
    (source / "requirements.txt").write_text("fastapi\n")
    (source / "requirements.lock").write_text("fastapi==0.139.0\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "fixture@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Fixture"],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-qm", "fixture"],
        check=True,
    )

    with build_worker_release_bundle(
        source,
        require_committed_source=True,
    ) as bundle:
        assert bundle.git_commit is not None
        assert bundle.tracked_manifest_sha256 is not None

    (source / "untracked.py").write_text("UNREVIEWED = True\n")
    with pytest.raises(WorkerReleaseError, match="dirty or untracked"):
        build_worker_release_bundle(source, require_committed_source=True)


def test_generated_installer_executes_candidate_without_promoting_current(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("APP = 1\n")
    (source / "requirements.txt").write_text("")
    (source / "requirements.lock").write_text("")
    executable = source / "launcher"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    releases_root = tmp_path / "worker-releases"
    venv_seed = Path(sys.executable).parent.parent

    with build_worker_release_bundle(source) as bundle:
        script = worker_release_install_script(
            archive_path=str(bundle.archive_path),
            archive_sha256=bundle.archive_sha256,
            source_sha256=bundle.source_sha256,
            release_id=bundle.release_id,
            releases_root=str(releases_root),
            venv_path=str(venv_seed),
        )
        result = subprocess.run(
            ["bash", "-s"],
            input=script,
            text=True,
            capture_output=True,
            check=True,
        )
        release_id = bundle.release_id

    candidate = releases_root / "releases" / release_id
    assert f"WORKER_RELEASE_CANDIDATE_ROOT={candidate / 'gpu_worker'}" in result.stdout
    assert not (releases_root / "current").exists()
    assert (candidate / ".ready").read_text().strip()
    assert stat.S_IMODE((candidate / ".ready").stat().st_mode) == 0o444
    writable_after_retry = [
        path
        for path in candidate.rglob("*")
        if not path.is_symlink() and path.stat().st_mode & 0o222
    ]
    assert not writable_after_retry, writable_after_retry

    # Simulate SIGKILL after .ready became 0444 but before the recursive
    # immutability pass completed. The retry must classify and safely replace
    # this unreferenced candidate instead of wedging the release id forever.
    interrupted_file = candidate / "gpu_worker/app.py"
    interrupted_file.chmod(0o644)
    with build_worker_release_bundle(source) as retry_bundle:
        retry = subprocess.run(
            ["bash", "-s"],
            input=worker_release_install_script(
                archive_path=str(retry_bundle.archive_path),
                archive_sha256=retry_bundle.archive_sha256,
                source_sha256=retry_bundle.source_sha256,
                release_id=retry_bundle.release_id,
                releases_root=str(releases_root),
                venv_path=str(venv_seed),
            ),
            text=True,
            capture_output=True,
            check=True,
        )
    assert f"WORKER_RELEASE_ID={release_id}" in retry.stdout
    writable_after_retry = [
        path
        for path in candidate.rglob("*")
        if not path.is_symlink() and path.stat().st_mode & 0o222
    ]
    assert not writable_after_retry, writable_after_retry

    import_env = os.environ.copy()
    import_env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [str(candidate / ".venv/bin/python"), "-c", "import gpu_worker.app"],
        cwd=candidate,
        env=import_env,
        text=True,
        check=True,
    )
    assert not (candidate / "gpu_worker/__pycache__").exists()
    (releases_root / "current").symlink_to(candidate)
    activated = subprocess.run(
        ["bash", "-s"],
        input=worker_release_activate_script(
            releases_root=str(releases_root),
            release_id=release_id,
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    assert f"WORKER_RELEASE_ACTIVE={release_id}" in activated.stdout


def test_release_installer_is_immutable_and_never_pulls_or_mutates_git() -> None:
    script = worker_release_install_script(
        archive_path="/tmp/worker-release.tar.gz",
        archive_sha256="a" * 64,
        source_sha256="b" * 64,
        release_id="sha256-" + "b" * 24,
        releases_root="/opt/filmforge-worker-releases",
        venv_path="/opt/filmforge-worker-runtime/.venv",
    )

    assert "git pull" not in script
    assert "git clone" not in script
    assert "git -C" not in script
    assert "/.git/" not in script
    assert "sha256sum \"$ARCHIVE\"" in script
    assert "ACTUAL_SOURCE_SHA256" in script
    assert "installed worker release content digest mismatch" in script
    assert "mv -Tf \"$CURRENT.next\"" not in script
    assert "WORKER_RELEASE_CANDIDATE_ROOT" in script
    assert '"$TARGET/.venv/bin/python" -m pip install' in script
    assert "gpu_worker/requirements.lock" in script
    assert script.index("digest mismatch") < script.index("WORKER_RELEASE_CANDIDATE_ROOT")
    assert 'TARGET="$RELEASES_ROOT/releases/$RELEASE_ID"' in script
    assert "os.chmod(path, 0o555 if mode & 0o111 else 0o444)" in script
    assert ".release.lock" in script
    assert ".release.lock.d" not in script
    assert "fcntl.flock(8" in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    activation = worker_release_activate_script(
        releases_root="/opt/filmforge-worker-releases",
        release_id="sha256-" + "b" * 24,
    )
    assert "worker candidate is not ready" in activation
    assert activation.index(".ready") < activation.index('mv -Tf "$CURRENT.next"')
    assert ".release.lock" in activation
    assert ".release.lock.d" not in activation
    assert "fcntl.flock(8" in activation
    subprocess.run(["bash", "-n"], input=activation, text=True, check=True)


def test_cutover_activation_refuses_root_bytecode_drift_from_provision(
    tmp_path: Path,
) -> None:
    """Regression for the fresh InfiniteTalk provision -> cutover failure.

    The paid box ran the readiness interpreter as root without ``-B``.  Root
    could create a writable ``__pycache__`` below the read-only candidate; a
    cutover retry must never promote that candidate even when its ready marker
    and dependency snapshot still look valid.
    """

    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("APP = 1\n")
    (source / "requirements.txt").write_text("")
    (source / "requirements.lock").write_text("")
    releases_root = tmp_path / "worker-releases"
    venv_seed = Path(sys.executable).parent.parent

    with build_worker_release_bundle(source) as bundle:
        subprocess.run(
            ["bash", "-s"],
            input=worker_release_install_script(
                archive_path=str(bundle.archive_path),
                archive_sha256=bundle.archive_sha256,
                source_sha256=bundle.source_sha256,
                release_id=bundle.release_id,
                releases_root=str(releases_root),
                venv_path=str(venv_seed),
            ),
            text=True,
            capture_output=True,
            check=True,
        )
        release_id = bundle.release_id

    candidate = releases_root / "releases" / release_id
    package = candidate / "gpu_worker"
    # Model root's ability to bypass a 0555 package directory: leave the
    # resulting bytecode artifact writable, exactly as on the paid VM.
    package.chmod(0o755)
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "app.cpython-312.pyc").write_bytes(b"runtime bytecode")
    cache.chmod(0o755)
    package.chmod(0o555)

    activation = subprocess.run(
        ["bash", "-s"],
        input=worker_release_activate_script(
            releases_root=str(releases_root),
            release_id=release_id,
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert activation.returncode != 0
    assert "worker candidate became writable" in activation.stderr
    assert not (releases_root / "current").exists()


def test_release_rollback_is_compare_and_swap_guarded() -> None:
    script = worker_release_rollback_script(
        releases_root="/opt/filmforge-worker-releases",
        failed_release_id="sha256-" + "c" * 24,
    )
    assert "FAILED_TARGET" in script
    assert ".release.lock" in script
    assert ".release.lock.d" not in script
    assert "fcntl.flock(8" in script
    assert "failed first-install candidate stopped" in script
    assert script.index("FAILED_RELEASE_ID") < script.index("CURRENT.rollback")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_code_release_kernel_lock_has_no_owner_publish_race(tmp_path: Path) -> None:
    releases_root = tmp_path / "releases-root"
    second_entered = threading.Event()

    def contender() -> None:
        with _code_release_transaction_lock(releases_root):
            second_entered.set()

    with _code_release_transaction_lock(releases_root):
        thread = threading.Thread(target=contender)
        thread.start()
        assert not second_entered.wait(0.2)

    assert second_entered.wait(2)
    thread.join(timeout=2)
    lock_file = releases_root / ".release.lock"
    assert lock_file.is_file() and not lock_file.is_symlink()
    assert stat.S_IMODE(lock_file.stat().st_mode) == 0o600


def test_stage_keeps_public_override_lexically_authoritative_and_noops_on_restart(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(tmp_path)
    staged = stage_secure_profile(contract, layout)
    dropin_dir = layout.systemd_root / f"{contract.worker_unit}.d"

    assert sorted(path.name for path in dropin_dir.glob("*.conf")) == [
        PUBLIC_OVERRIDE_NAME,
    ]
    assert PROFILE_DROPIN_NAME < PUBLIC_OVERRIDE_NAME
    assert (dropin_dir / PUBLIC_OVERRIDE_NAME).is_file()
    assert not staged.worker_dropin.exists()
    tunnel_unit = layout.systemd_root / contract.tunnel_unit
    assert tunnel_unit.is_symlink()
    assert tunnel_unit.resolve() == (
        staged.release_dir / "filmforge-worker-tunnel.service"
    ).resolve()
    assert (staged.release_dir / "filmforge-worker-tunnel").stat().st_mode & 0o111
    assert stat.S_IMODE((staged.release_dir / "worker-secrets.env").stat().st_mode) == 0o600
    assert stat.S_IMODE((staged.release_dir / "tunnel-secrets.env").stat().st_mode) == 0o600
    assert stat.S_IMODE((staged.release_dir / "cloudflared.yml").stat().st_mode) == 0o600
    assert stat.S_IMODE((staged.release_dir / "cloudflared-credential.json").stat().st_mode) == 0o600
    staged_tunnel_env = (staged.release_dir / "tunnel-secrets.env").read_text()
    assert f"FILMFORGE_TUNNEL_CONFIG_FILE={staged.release_dir / 'cloudflared.yml'}" in staged_tunnel_env
    assert f"CLOUDFLARED_BIN={staged.release_dir / 'cloudflared'}" in staged_tunnel_env
    assert (
        f"FILMFORGE_TUNNEL_WORKER_CODE_RELEASE_ID={contract.worker_code_release_id}"
        in staged_tunnel_env
    )
    assert str(contract.tunnel_secret_source) not in staged_tunnel_env

    worker_dropin = (staged.release_dir / "worker-secure-profile.conf").read_text()
    assert "EnvironmentFile=" in worker_dropin
    assert "Environment=PYTHONDONTWRITEBYTECODE=1" in worker_dropin
    assert f"Environment=WORKER_CODE_RELEASE_ID={contract.worker_code_release_id}" in worker_dropin
    assert f"BindsTo={contract.tunnel_unit}" in worker_dropin
    assert "--host 127.0.0.1" in worker_dropin
    assert "super-private-worker-token" not in worker_dropin
    assert "super-private-registration-token" not in worker_dropin
    assert "super-private-tunnel-token" not in worker_dropin

    controller = _Controller(dropin_dir / PUBLIC_OVERRIDE_NAME)
    prepare_secure_profile(
        release_id=contract.release_id,
        layout=layout,
        controller=controller,
        now_epoch=1_800_000_000,
    )
    assert not staged.worker_dropin.exists()
    assert all(
        event[1] != contract.worker_unit
        for event in controller.events
        if event[0] in {"restart", "stop"}
    )


def test_stage_adopts_known_legacy_loopback_profile_under_public_override(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(tmp_path)
    legacy_env = _write_0600(
        layout.state_root.parent / "worker-gpu0-secure.env",
        "WORKER_API_AUTH_MODE=required\n"
        "WORKER_PUBLIC_URL=http://127.0.0.1:19000\n",
    )
    dropin_dir = layout.systemd_root / f"{contract.worker_unit}.d"
    legacy_dropin = dropin_dir / "10-secure-loopback.conf"
    legacy_dropin.write_text(
        "[Service]\n"
        f"EnvironmentFile={legacy_env}\n"
        "ExecStart=\n"
        "ExecStart=/usr/bin/worker --host 127.0.0.1 --port 9000\n"
    )

    staged = stage_secure_profile(contract, layout)

    assert legacy_dropin.exists()
    assert legacy_env.exists()
    assert (
        staged.release_dir / "rollback" / "10-secure-loopback.conf.original"
    ).is_file()
    assert (
        staged.release_dir / "rollback" / "legacy-worker-secure.env.original"
    ).is_file()
    assert (dropin_dir / PUBLIC_OVERRIDE_NAME).is_file()

    now = 1_800_000_000
    controller = _Controller(dropin_dir / PUBLIC_OVERRIDE_NAME)
    _prepare(staged, contract, layout, controller, now=now)
    cutover_secure_profile(
        release_id=contract.release_id,
        receipt_path=_ready_receipt(staged, tmp_path / "receipt.json", now=now),
        layout=layout,
        controller=controller,
        now_epoch=now,
    )
    assert not legacy_dropin.exists()
    assert not legacy_env.exists()
    assert (
        staged.release_dir / "rollback" / "10-secure-loopback.conf.disabled"
    ).is_file()
    assert (
        staged.release_dir / "rollback" / "legacy-worker-secure.env.disabled"
    ).is_file()


def test_stage_never_moves_an_unrelated_legacy_environment_file(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(tmp_path)
    unrelated = _write_0600(
        tmp_path / "unrelated.env",
        "WORKER_API_AUTH_MODE=required\n"
        "WORKER_PUBLIC_URL=http://127.0.0.1:19000\n",
    )
    dropin_dir = layout.systemd_root / f"{contract.worker_unit}.d"
    (dropin_dir / "10-secure-loopback.conf").write_text(
        "[Service]\n"
        f"EnvironmentFile={unrelated}\n"
        "ExecStart=\n"
        "ExecStart=/usr/bin/worker --host 127.0.0.1 --port 9000\n"
    )

    with pytest.raises(WorkerReleaseError, match="unexpected env file"):
        stage_secure_profile(contract, layout)

    assert unrelated.is_file()


def test_partial_legacy_disable_recovers_and_failed_cutover_can_retry(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(tmp_path)
    legacy_env = _write_0600(
        layout.state_root.parent / "worker-gpu0-secure.env",
        "WORKER_API_AUTH_MODE=required\n"
        "WORKER_PUBLIC_URL=http://127.0.0.1:19000\n",
    )
    dropin_dir = layout.systemd_root / f"{contract.worker_unit}.d"
    legacy_dropin = dropin_dir / "10-secure-loopback.conf"
    legacy_dropin.write_text(
        "[Service]\n"
        f"EnvironmentFile={legacy_env}\n"
        "ExecStart=\n"
        "ExecStart=/usr/bin/worker --host 127.0.0.1 --port 9000\n"
    )
    staged = stage_secure_profile(contract, layout)
    override = dropin_dir / PUBLIC_OVERRIDE_NAME
    now = 1_800_000_000
    prepare_secure_profile(
        release_id=contract.release_id,
        layout=layout,
        controller=_Controller(override),
        now_epoch=now,
    )

    disabled_dropin = (
        staged.release_dir / "rollback" / "10-secure-loopback.conf.disabled"
    )
    os.replace(legacy_dropin, disabled_dropin)
    disabled_dropin.chmod(0o600)
    receipt_path = _ready_receipt(staged, tmp_path / "receipt.json", now=now)
    failing = _Controller(override, fail_backend_probe=True)
    with pytest.raises(WorkerReleaseError, match="safe state was restored"):
        cutover_secure_profile(
            release_id=contract.release_id,
            receipt_path=receipt_path,
            layout=layout,
            controller=failing,
            now_epoch=now,
        )

    assert legacy_dropin.exists()
    assert legacy_env.exists()
    assert not disabled_dropin.exists()
    staged = stage_secure_profile(contract, layout)
    _prepare(
        staged,
        contract,
        layout,
        _Controller(override),
        now=now,
    )
    cutover_secure_profile(
        release_id=contract.release_id,
        receipt_path=receipt_path,
        layout=layout,
        controller=_Controller(override),
        now_epoch=now,
    )
    assert not override.exists()


def test_stage_refuses_unknown_worker_or_tunnel_dropin(tmp_path: Path) -> None:
    contract, layout = _fixture(tmp_path)
    dropin_dir = layout.systemd_root / f"{contract.worker_unit}.d"
    (dropin_dir / "50-unmanaged.conf").write_text("[Service]\nEnvironment=X=1\n")
    with pytest.raises(WorkerReleaseError, match="unmanaged worker drop-ins"):
        stage_secure_profile(contract, layout)


def test_tunnel_ingress_must_be_one_exact_route_then_terminal_404(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(tmp_path)
    tunnel_values = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in contract.tunnel_secret_source.read_text().splitlines()
    }
    config_path = Path(tunnel_values["FILMFORGE_TUNNEL_CONFIG_FILE"])
    credential_path = tunnel_values["FILMFORGE_TUNNEL_CREDENTIAL_FILE"]
    config_path.write_text(
        "tunnel: fixture\n"
        f"credentials-file: {credential_path}\n"
        "ingress:\n"
        "  - hostname: gpu0.worker.example\n"
        "    service: http://127.0.0.1:9999\n"
        "  - hostname: gpu0.worker.example\n"
        "    service: http://127.0.0.1:9000\n"
        "  - service: http_status:404\n"
    )
    with pytest.raises(WorkerReleaseError, match="tunnel config does not match"):
        stage_secure_profile(contract, layout)


def test_stage_refuses_restart_landmine_when_public_override_is_absent(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(tmp_path, with_override=False)
    with pytest.raises(WorkerReleaseError, match="must remain installed"):
        stage_secure_profile(contract, layout)
    assert not (
        layout.systemd_root / f"{contract.worker_unit}.d" / PROFILE_DROPIN_NAME
    ).exists()


@pytest.mark.parametrize(
    ("worker_env", "tunnel_env", "with_tunnel_exec", "message"),
    [
        (
            "WORKER_REGISTRATION_TOKEN=r\nFILMFORGE_BACKEND_URL=https://b\n"
            "WORKER_API_AUTH_MODE=required\nWORKER_PUBLIC_URL=https://gpu0.worker.example\n",
            None,
            True,
            "GPU_WORKER_API_TOKEN",
        ),
        (
            "GPU_WORKER_API_TOKEN=t\nFILMFORGE_BACKEND_URL=https://b\n"
            "WORKER_API_AUTH_MODE=required\nWORKER_PUBLIC_URL=https://gpu0.worker.example\n",
            None,
            True,
            "WORKER_REGISTRATION_TOKEN",
        ),
        (
            "GPU_WORKER_API_TOKEN=t\nWORKER_REGISTRATION_TOKEN=r\n"
            "WORKER_API_AUTH_MODE=required\nWORKER_PUBLIC_URL=https://gpu0.worker.example\n",
            None,
            True,
            "FILMFORGE_BACKEND_URL",
        ),
        (
            "GPU_WORKER_API_TOKEN=t\nWORKER_REGISTRATION_TOKEN=r\n"
            "FILMFORGE_BACKEND_URL=https://b\nWORKER_API_AUTH_MODE=development\n"
            "WORKER_PUBLIC_URL=https://gpu0.worker.example\n",
            None,
            True,
            "AUTH_MODE=required",
        ),
        (
            None,
            "FILMFORGE_TUNNEL_LOCAL_URL=http://127.0.0.1:9000\n"
            "FILMFORGE_TUNNEL_PUBLIC_URL=https://gpu0.worker.example\n"
            "FILMFORGE_TUNNEL_UNIT=filmforge-worker-tunnel-gpu0.service\n",
            True,
            "FILMFORGE_TUNNEL_CONFIG_FILE",
        ),
        (None, None, False, "tunnel executable"),
    ],
)
def test_stage_refuses_missing_tunnel_client_auth_or_secrets(
    tmp_path: Path,
    worker_env: str | None,
    tunnel_env: str | None,
    with_tunnel_exec: bool,
    message: str,
) -> None:
    contract, layout = _fixture(
        tmp_path,
        worker_env=worker_env,
        tunnel_env=tunnel_env,
        with_tunnel_exec=with_tunnel_exec,
    )
    with pytest.raises(WorkerReleaseError, match=message):
        stage_secure_profile(contract, layout)


def test_stage_requires_secret_files_to_be_exactly_mode_0600(tmp_path: Path) -> None:
    contract, layout = _fixture(tmp_path)
    contract.worker_secret_source.chmod(0o640)
    with pytest.raises(WorkerReleaseError, match="mode 0600"):
        stage_secure_profile(contract, layout)


def test_cutover_refuses_stale_or_incomplete_receipt_before_service_change(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(tmp_path)
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    controller = _Controller(override)
    _prepare(staged, contract, layout, controller, now=100)
    receipt = build_cutover_receipt_template(staged, issued_at_epoch=100)
    receipt["tunnel_ready"] = True
    receipt_path = _write_0600(tmp_path / "receipt.json", json.dumps(receipt))

    with pytest.raises(WorkerReleaseError):
        cutover_secure_profile(
            release_id=contract.release_id,
            receipt_path=receipt_path,
            layout=layout,
            controller=controller,
            now_epoch=1_000_000,
        )

    assert override.exists()
    assert controller.events == []


def test_receipt_template_refuses_unprepared_tunnel(tmp_path: Path) -> None:
    contract, layout = _fixture(tmp_path)
    staged = stage_secure_profile(contract, layout)
    with pytest.raises(WorkerReleaseError, match="only after prepare"):
        build_cutover_receipt_template(staged)


def test_cutover_refuses_corrupt_public_override_rollback_copy(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(tmp_path)
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    controller = _Controller(override)
    now = 1_800_000_000
    _prepare(staged, contract, layout, controller, now=now)
    controller.events.clear()
    backup = staged.release_dir / "rollback" / PUBLIC_OVERRIDE_NAME
    backup.chmod(0o600)
    backup.write_text("[Service]\nExecStart=/tampered\n")
    backup.chmod(0o600)

    with pytest.raises(WorkerReleaseError, match="rollback copy content drifted"):
        cutover_secure_profile(
            release_id=contract.release_id,
            receipt_path=_ready_receipt(staged, tmp_path / "receipt.json", now=now),
            layout=layout,
            controller=controller,
            now_epoch=now,
        )

    assert override.exists()
    assert controller.events == []


def test_cutover_removes_public_override_only_after_receipt_and_tunnel_gate(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(tmp_path)
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    controller = _Controller(override)
    now = 1_800_000_000
    _prepare(staged, contract, layout, controller, now=now)
    receipt_path = _ready_receipt(staged, tmp_path / "receipt.json", now=now)

    cutover_secure_profile(
        release_id=contract.release_id,
        receipt_path=receipt_path,
        layout=layout,
        controller=controller,
        now_epoch=now,
    )

    assert not override.exists()
    assert controller.events == [
        ("tunnel-active", contract.tunnel_unit, True),
        ("unit-loaded", contract.tunnel_unit, True),
        ("daemon-reload", None, False),
        ("unit-loaded", contract.worker_unit, False),
        ("restart", contract.worker_unit, False),
        ("loopback-only", 9000, False),
        ("backend-route", contract.worker_public_url, False),
    ]
    stage_data = json.loads(staged.stage_receipt.read_text())
    assert stage_data["cutover_performed"] is True


def test_failed_public_port_closure_automatically_restores_override(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(tmp_path)
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    controller = _Controller(override, fail_loopback=True)
    now = 1_800_000_000
    _prepare(staged, contract, layout, controller, now=now)
    receipt_path = _ready_receipt(staged, tmp_path / "receipt.json", now=now)

    with pytest.raises(WorkerReleaseError, match="safe state was restored"):
        cutover_secure_profile(
            release_id=contract.release_id,
            receipt_path=receipt_path,
            layout=layout,
            controller=controller,
            now_epoch=now,
        )

    assert override.exists()
    assert ("unit-loaded", contract.worker_unit, True) in controller.events
    assert ("restart", contract.worker_unit, True) in controller.events
    assert ("tunnel-active", contract.worker_unit, True) in controller.events
    assert ("public-listener", 9000, True) in controller.events


def test_failed_post_restart_backend_probe_restores_override(tmp_path: Path) -> None:
    contract, layout = _fixture(tmp_path)
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    controller = _Controller(override, fail_backend_probe=True)
    now = 1_800_000_000
    _prepare(staged, contract, layout, controller, now=now)
    receipt_path = _ready_receipt(staged, tmp_path / "receipt.json", now=now)

    with pytest.raises(WorkerReleaseError, match="safe state was restored"):
        cutover_secure_profile(
            release_id=contract.release_id,
            receipt_path=receipt_path,
            layout=layout,
            controller=controller,
            now_epoch=now,
        )

    assert override.exists()
    assert ("backend-route", contract.worker_public_url, False) in controller.events


def test_cutover_resumes_after_worker_link_before_override_removal(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(tmp_path)
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    controller = _Controller(override)
    now = 1_800_000_000
    _prepare(staged, contract, layout, controller, now=now)
    staged.worker_dropin.symlink_to(
        staged.release_dir / "worker-secure-profile.conf"
    )

    cutover_secure_profile(
        release_id=contract.release_id,
        receipt_path=_ready_receipt(staged, tmp_path / "receipt.json", now=now),
        layout=layout,
        controller=controller,
        now_epoch=now,
    )

    assert not override.exists()
    assert staged.worker_dropin.is_symlink()
    assert (
        layout.state_root / "active" / contract.worker_unit
    ).resolve() == staged.release_dir.resolve()


def test_first_install_never_creates_public_override_and_fails_closed(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(
        tmp_path,
        with_override=False,
        profile_mode="first-install",
    )
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    guard = (
        layout.systemd_root
        / f"{contract.worker_unit}.d"
        / "00-filmforge-staged-guard.conf"
    )
    assert guard.is_symlink()
    assert not staged.worker_dropin.exists()
    controller = _Controller(override, fail_backend_probe=True)
    now = 1_800_000_000
    prepare_secure_profile(
        release_id=contract.release_id,
        layout=layout,
        controller=controller,
        now_epoch=now,
    )
    assert controller.events[0] == ("worker-inactive", contract.worker_unit, False)
    controller.events.clear()

    with pytest.raises(WorkerReleaseError, match="safe state was restored"):
        cutover_secure_profile(
            release_id=contract.release_id,
            receipt_path=_first_install_receipt(staged, tmp_path / "receipt.json", now=now),
            layout=layout,
            controller=controller,
            now_epoch=now,
        )
    assert not override.exists()
    assert not (staged.release_dir / "cutover-authorized").exists()
    assert not (staged.release_dir / "boot-authorized").exists()
    assert not staged.worker_dropin.exists()
    assert ("stop", contract.worker_unit, False) in controller.events
    assert ("stop", contract.tunnel_unit, False) in controller.events
    assert ("disable", contract.worker_unit) in controller.enablement_events
    assert ("disable", contract.tunnel_unit) in controller.enablement_events


def test_cutover_cleanup_failure_journal_resumes_without_tunnel_or_receipt(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(
        tmp_path,
        with_override=False,
        profile_mode="first-install",
    )
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    now = 1_800_000_000
    prepare_secure_profile(
        release_id=contract.release_id,
        layout=layout,
        controller=_Controller(override),
        now_epoch=now,
    )
    failing = _Controller(
        override,
        fail_backend_probe=True,
        fail_stop_once=True,
    )
    with pytest.raises(WorkerReleaseError, match="automatic safe-state rollback"):
        cutover_secure_profile(
            release_id=contract.release_id,
            receipt_path=_first_install_receipt(staged, tmp_path / "receipt.json", now=now),
            layout=layout,
            controller=failing,
            now_epoch=now,
        )

    interrupted = json.loads(staged.stage_receipt.read_text())
    assert interrupted["cutover_performed"] is True
    assert interrupted["cutover_state"] == "in_progress"
    assert interrupted["rollback_state"] == "in_progress"

    with pytest.raises(WorkerReleaseError, match="safe state restored"):
        cutover_secure_profile(
            release_id=contract.release_id,
            receipt_path=tmp_path / "missing-stale-receipt.json",
            layout=layout,
            controller=_Controller(override),
            now_epoch=now + 86_400,
        )

    completed = json.loads(staged.stage_receipt.read_text())
    assert completed["cutover_performed"] is False
    assert completed["cutover_state"] == "rolled_back"
    assert completed["rollback_state"] == "complete"


def test_reboot_during_cutover_with_inactive_tunnel_rolls_back_safely(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(
        tmp_path,
        with_override=False,
        profile_mode="first-install",
    )
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    now = 1_800_000_000
    prepare_secure_profile(
        release_id=contract.release_id,
        layout=layout,
        controller=_Controller(override),
        now_epoch=now,
    )
    stage_data = json.loads(staged.stage_receipt.read_text())
    stage_data["cutover_state"] = "in_progress"
    stage_data["cutover_performed"] = False
    _write_0600(staged.stage_receipt, json.dumps(stage_data))
    staged.worker_dropin.symlink_to(staged.release_dir / "worker-secure-profile.conf")
    _write_0600(
        staged.release_dir / "cutover-authorized",
        contract.release_id + "\n",
    )

    with pytest.raises(WorkerReleaseError, match="inactive tunnel; safe state restored"):
        cutover_secure_profile(
            release_id=contract.release_id,
            receipt_path=tmp_path / "stale-missing-receipt.json",
            layout=layout,
            controller=_Controller(override, fail_active_once=True),
            now_epoch=now + 86_400,
        )

    completed = json.loads(staged.stage_receipt.read_text())
    assert completed["cutover_performed"] is False
    assert completed["rollback_state"] == "complete"
    assert not staged.worker_dropin.exists()


def test_first_install_cutover_enables_boot_units_and_rollback_disables_them(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(
        tmp_path,
        with_override=False,
        profile_mode="first-install",
    )
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    controller = _Controller(override)
    now = 1_800_000_000
    _prepare(staged, contract, layout, controller, now=now)

    cutover_secure_profile(
        release_id=contract.release_id,
        receipt_path=_first_install_receipt(staged, tmp_path / "receipt.json", now=now),
        layout=layout,
        controller=controller,
        now_epoch=now,
    )

    assert controller.enabled_units == {contract.worker_unit, contract.tunnel_unit}
    assert (staged.release_dir / "boot-authorized").read_text() == (
        contract.release_id + "\n"
    )
    assert controller.enablement_events == [
        ("enable", contract.tunnel_unit),
        ("enable", contract.worker_unit),
    ]

    rollback_secure_profile(
        release_id=contract.release_id,
        layout=layout,
        controller=controller,
    )

    assert controller.enabled_units == set()
    assert controller.enablement_events[-2:] == [
        ("disable", contract.worker_unit),
        ("disable", contract.tunnel_unit),
    ]


def test_first_install_post_start_probe_failure_rolls_back_and_stops(tmp_path: Path) -> None:
    contract, layout = _fixture(tmp_path, with_override=False, profile_mode="first-install")
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    now = 1_800_000_000
    _prepare(staged, contract, layout, _Controller(override), now=now)
    failing = _Controller(override, fail_backend_probe=True)
    with pytest.raises(WorkerReleaseError, match="safe state was restored"):
        cutover_secure_profile(
            release_id=contract.release_id,
            receipt_path=_first_install_receipt(staged, tmp_path / "receipt.json", now=now),
            layout=layout,
            controller=failing,
            now_epoch=now,
        )
    data = json.loads(staged.stage_receipt.read_text())
    assert data["cutover_performed"] is False
    assert data["rollback_state"] == "complete"
    assert ("stop", contract.worker_unit, False) in failing.events
    assert ("stop", contract.tunnel_unit, False) in failing.events


def test_migration_refuses_false_pre_cutover_backend_readiness(tmp_path: Path) -> None:
    contract, layout = _fixture(tmp_path)
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    now = 1_800_000_000
    _prepare(staged, contract, layout, _Controller(override), now=now)
    receipt = build_cutover_receipt_template(staged, issued_at_epoch=now)
    receipt.update({
        "tunnel_ready": True,
        "backend_bearer_client_ready": False,
        "backend_registration_ready": False,
        "worker_secret_fingerprint_match": True,
    })
    with pytest.raises(WorkerReleaseError, match="backend_bearer_client_ready"):
        cutover_secure_profile(
            release_id=contract.release_id,
            receipt_path=_write_0600(tmp_path / "receipt.json", json.dumps(receipt)),
            layout=layout,
            controller=_Controller(override),
            now_epoch=now,
        )


def test_stage_refuses_secure_to_secure_mixed_profile_landmine(tmp_path: Path) -> None:
    contract, layout = _fixture(
        tmp_path,
        with_override=False,
        profile_mode="first-install",
    )
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    controller = _Controller(override)
    now = 1_800_000_000
    _prepare(staged, contract, layout, controller, now=now)
    cutover_secure_profile(
        release_id=contract.release_id,
        receipt_path=_first_install_receipt(staged, tmp_path / "receipt.json", now=now),
        layout=layout,
        controller=controller,
        now_epoch=now,
    )
    managed = [
        layout.systemd_root / f"{contract.worker_unit}.d" / PROFILE_DROPIN_NAME,
        layout.systemd_root / f"{contract.tunnel_unit}.d" / PROFILE_DROPIN_NAME,
        layout.systemd_root / contract.tunnel_unit,
        layout.systemd_root
        / f"{contract.worker_unit}.d"
        / "00-filmforge-staged-guard.conf",
        layout.state_root / "active" / contract.worker_unit,
    ]
    before = {path: path.resolve() for path in managed}

    with pytest.raises(WorkerReleaseError, match="secure-to-secure"):
        stage_secure_profile(
            replace(contract, release_id="release-2026-08-10-b"),
            layout,
        )

    assert {path: path.resolve() for path in managed} == before


def test_interrupted_secure_rollback_is_retryable_and_restores_code_cas(
    tmp_path: Path,
) -> None:
    contract, layout = _fixture(
        tmp_path,
        with_override=False,
        profile_mode="first-install",
    )
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    now = 1_800_000_000
    cutover_controller = _Controller(override)
    _prepare(staged, contract, layout, cutover_controller, now=now)
    cutover_secure_profile(
        release_id=contract.release_id,
        receipt_path=_first_install_receipt(staged, tmp_path / "receipt.json", now=now),
        layout=layout,
        controller=cutover_controller,
        now_epoch=now,
    )

    candidate = contract.worker_module_dir
    releases_root = candidate.parent.parent
    old_release = candidate.parent / ("sha256-" + "b" * 24)
    old_release.mkdir()
    (old_release / ".ready").write_text("b" * 64 + "\n")
    (old_release / ".dependency-freeze.txt").write_text("")
    (old_release / ".dependency-freeze.sha256").write_text(
        hashlib.sha256(b"").hexdigest() + "\n"
    )
    for path in old_release.iterdir():
        path.chmod(0o444)
    old_release.chmod(0o555)
    (releases_root / "current").symlink_to(candidate)
    (releases_root / "previous").symlink_to(old_release)

    with pytest.raises(RuntimeError, match="interrupted rollback"):
        rollback_secure_profile(
            release_id=contract.release_id,
            layout=layout,
            controller=_Controller(override, fail_stop_once=True),
        )

    interrupted = json.loads(staged.stage_receipt.read_text())
    assert interrupted["cutover_performed"] is True
    assert interrupted["rollback_state"] == "in_progress"
    assert (releases_root / "current").resolve() == candidate.resolve()

    rollback_secure_profile(
        release_id=contract.release_id,
        layout=layout,
        controller=_Controller(override),
    )

    completed = json.loads(staged.stage_receipt.read_text())
    assert completed["cutover_performed"] is False
    assert completed["rollback_state"] == "complete"
    assert (releases_root / "current").resolve() == old_release.resolve()


def test_explicit_rollback_restores_override_before_restart(tmp_path: Path) -> None:
    contract, layout = _fixture(tmp_path)
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    now = 1_800_000_000
    initial_controller = _Controller(override)
    _prepare(staged, contract, layout, initial_controller, now=now)
    cutover_secure_profile(
        release_id=contract.release_id,
        receipt_path=_ready_receipt(staged, tmp_path / "receipt.json", now=now),
        layout=layout,
        controller=_Controller(override),
        now_epoch=now,
    )
    controller = _Controller(override)

    rollback_secure_profile(
        release_id=contract.release_id,
        layout=layout,
        controller=controller,
    )

    assert override.exists()
    assert controller.events == [
        ("daemon-reload", None, True),
        ("unit-loaded", contract.worker_unit, True),
        ("restart", contract.worker_unit, True),
        ("tunnel-active", contract.worker_unit, True),
        ("public-listener", 9000, True),
        ("stop", contract.tunnel_unit, True),
        ("daemon-reload", None, True),
    ]
    assert json.loads(staged.stage_receipt.read_text())["cutover_performed"] is False


def test_profile_rollback_is_compare_and_swap_guarded(tmp_path: Path) -> None:
    first, layout = _fixture(tmp_path)
    stage_secure_profile(first, layout)
    second = replace(first, release_id="release-2026-08-10-b")
    stage_secure_profile(second, layout)

    override = layout.systemd_root / f"{first.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    with pytest.raises(WorkerReleaseError, match="not (?:the )?active"):
        rollback_secure_profile(
            release_id=first.release_id,
            layout=layout,
            controller=_Controller(override),
        )


def test_receipt_schema_is_explicit_and_false_by_default(tmp_path: Path) -> None:
    contract, layout = _fixture(tmp_path)
    staged = stage_secure_profile(contract, layout)
    override = layout.systemd_root / f"{contract.worker_unit}.d" / PUBLIC_OVERRIDE_NAME
    controller = _Controller(override)
    _prepare(staged, contract, layout, controller, now=1_800_000_000)
    template = build_cutover_receipt_template(staged)

    assert template["schema"] == RECEIPT_SCHEMA
    assert template["tunnel_ready"] is False
    assert template["backend_bearer_client_ready"] is False
    assert template["backend_registration_ready"] is False
    assert template["worker_secret_fingerprint_match"] is False


def test_repo_managed_tunnel_launcher_uses_protected_files_not_token_argv() -> None:
    launcher = Path(__file__).parents[1] / "deploy" / "bin" / "filmforge-worker-tunnel"
    source = launcher.read_text()

    subprocess.run(["bash", "-n", str(launcher)], check=True)
    assert "FILMFORGE_TUNNEL_LOCAL_URL" in source
    assert "FILMFORGE_TUNNEL_PUBLIC_URL" in source
    assert "FILMFORGE_TUNNEL_CREDENTIAL_FILE" in source
    assert "FILMFORGE_TUNNEL_CUTOVER_AUTHORIZATION_FILE" in source
    assert "FILMFORGE_TUNNEL_WORKER_SECRET_FILE" in source
    assert "FILMFORGE_TUNNEL_WORKER_CODE_RELEASE_ID" in source
    assert "curl --config" in source
    assert 'test "$WATCHDOG_STATUS" = "200"' in source
    assert "max-filesize = 65536" in source
    assert 'value.get("code_release_id") != expected_code_release_id' in source
    assert 'value.get("worker_ok") is not True' in source
    assert "WATCHDOG_FAILURES" in source
    assert "three consecutive" in source
    assert "must have mode 0600" in source
    assert "tunnel config does not match" in source
    assert "--token" not in source
    assert "FILMFORGE_TUNNEL_TOKEN" not in source


def test_tunnel_watchdog_rejects_redirect_and_unrelated_200_body(tmp_path: Path) -> None:
    launcher = Path(__file__).parents[1] / "deploy" / "bin" / "filmforge-worker-tunnel"
    source = launcher.read_text()
    validator = source.rsplit("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    body = tmp_path / "health.json"
    public_url = "https://gpu0.worker.example"
    code_release_id = "sha256-" + "a" * 24

    body.write_text(
        json.dumps(
            {
                "ok": True,
                "worker_ok": True,
                "public_url": public_url,
                "code_release_id": code_release_id,
            }
        )
    )
    good = subprocess.run(
        [sys.executable, "-", str(body), public_url, code_release_id],
        input=validator,
        text=True,
        capture_output=True,
    )
    assert good.returncode == 0

    body.write_text('{"ok": true, "message": "unrelated static page"}\n')
    unrelated = subprocess.run(
        [sys.executable, "-", str(body), public_url, code_release_id],
        input=validator,
        text=True,
        capture_output=True,
    )
    assert unrelated.returncode != 0
    # The launcher admits only an exact 200 before invoking the validator, so
    # Cloudflare Access 302 responses cannot reset the failure counter.
    assert 'test "$WATCHDOG_STATUS" = "200"' in source
