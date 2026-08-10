from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest

from gpu_worker import secure_one_click as one
from gpu_worker import deploy_gpu


class _FakeDeployApi:
    def __init__(self, events: list[str], *, fail_phase: str | None = None) -> None:
        self.events = events
        self.fail_phase = fail_phase
        self.bundle_calls = 0

    def _prepare_worker_release_bundle(self, args):
        self.bundle_calls += 1
        self.events.append("bundle")
        return object()

    def verda_deploy(self, args):
        phase = next(
            item.split("=", 1)[1]
            for item in args.env_vars
            if item.startswith("WORKER_DEPLOY_PHASE=")
        )
        self.events.append(phase)
        if phase == "stage-code":
            args._verda_active_instance_id = "a" * 32
            args._verda_active_ip = "203.0.113.10"
            args._verda_worker_release_id = "sha256-" + "b" * 24
            args._verda_worker_source_root = (
                "/opt/filmforge-worker-releases/releases/sha256-"
                + "b" * 24
                + "/gpu_worker"
            )
            args._verda_ssh_cmd = ["ssh", "root@example"]
            args._verda_scp_cmd = ["scp"]
            args._verda_destination = "root@example"
        if phase == self.fail_phase:
            return 7
        return 0

    verda_fresh_deploy = verda_deploy


class _FakeDns:
    instances: list["_FakeDns"] = []

    def __init__(self, **kwargs):
        del kwargs
        self.events: list[str] = []
        self.rolled_back = False
        self.__class__.instances.append(self)

    def preflight(self):
        self.events.append("dns-preflight")
        return None

    def point_to(self, ip, previous):
        assert ip == "203.0.113.10"
        assert previous is None
        self.events.append("dns-point")
        return one.DnsMutation(previous=None, changed=True)

    def rollback(self, mutation):
        assert mutation.changed
        self.rolled_back = True


class _FakeFly:
    instances: list["_FakeFly"] = []

    def __init__(self, **kwargs):
        del kwargs
        self.events: list[str] = []
        self.__class__.instances.append(self)

    def preflight(self):
        self.events.append("fly-preflight")

    def sync_fail_closed_secrets(self, values):
        assert values.worker_api_token == "w" * 40
        self.events.append("fly-disabled")

    def enable_worker_dispatch(self):
        self.events.append("fly-enabled")

    def disable_worker_dispatch(self):
        self.events.append("fly-disabled-rollback")


class _UnusedRunner(one.CommandRunner):
    def run(self, *args, **kwargs):  # pragma: no cover - a failed test calls this
        raise AssertionError((args, kwargs))


def _args(tmp_path: Path, *, workers: int = 1) -> Namespace:
    backend_env = tmp_path / ".env"
    backend_env.write_text("placeholder=1\n")
    os.chmod(backend_env, 0o600)
    return Namespace(
        verda=True,
        verda_fresh=False,
        verda_worker_count=workers,
        verda_worker_plan="",
        worker_edge_hostname="gpu-worker.anapana.ai",
        worker_edge_domain="anapana.ai",
        backend_env=backend_env,
        fly_app="filmforgepythonbackend",
        env_vars=[],
    )


@pytest.fixture
def automatic_fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events: list[str] = []
    _FakeDns.instances.clear()
    _FakeFly.instances.clear()
    caddy = tempfile.TemporaryDirectory()
    binary = Path(caddy.name) / "caddy"
    binary.write_bytes(b"caddy")
    os.chmod(binary, 0o755)
    monkeypatch.setattr(one, "VercelDns", _FakeDns)
    monkeypatch.setattr(one, "FlyBackend", _FakeFly)
    monkeypatch.setattr(one, "_download_caddy", lambda runner: caddy)
    monkeypatch.setattr(
        one,
        "_load_or_create_secrets",
        lambda path: one.DeploymentSecrets(
            worker_api_token="w" * 40,
            registration_token="r" * 32,
            cutover_probe_token="p" * 40,
            backend_url="https://filmforgepythonbackend.fly.dev",
        ),
    )
    monkeypatch.setattr(
        one,
        "_stage_profile_sources",
        lambda **kwargs: (
            kwargs["args"].env_vars[
                next(
                    i
                    for i, item in enumerate(kwargs["args"].env_vars)
                    if item.startswith("WORKER_SECURITY_STAGE_RECEIPTS=")
                )
            ].split("=", 1)[1],
            "/cutover.json",
        ),
    )
    monkeypatch.setattr(one, "_wait_for_tls_hostname", lambda *a, **k: events.append("tls"))
    monkeypatch.setattr(one, "_authorize_cutover_receipt", lambda **k: events.append("receipt"))
    monkeypatch.setattr(
        one,
        "_remote_profile_operation",
        lambda **kwargs: events.append(kwargs["operation"]),
    )
    monkeypatch.setattr(one, "_finalize_local_backend_env", lambda *a: events.append("local-env"))
    return events, caddy


def test_one_click_runs_exact_secure_phase_order(
    automatic_fakes,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events, _caddy = automatic_fakes
    api = _FakeDeployApi(events)

    result = one.run_secure_verda_first_install(
        _args(tmp_path),
        deploy_api=api,
        runner=_UnusedRunner(),
    )

    assert result == 0
    assert events == [
        "bundle",
        "stage-code",
        "tls",
        "provision-only",
        "tls",
        "receipt",
        "cutover",
        "activate",
        "local-env",
    ]
    assert _FakeFly.instances[-1].events == [
        "fly-preflight",
        "fly-disabled",
        "fly-enabled",
    ]
    output = capsys.readouterr().out
    assert "WORKER_URLS=https://gpu-worker.anapana.ai" in output
    assert "w" * 20 not in output
    assert "r" * 20 not in output


def test_one_click_rejects_multiple_workers_before_bundle_or_provider(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    api = _FakeDeployApi(events)

    with pytest.raises(one.OneClickDeploymentError, match="exactly one"):
        one.run_secure_verda_first_install(
            _args(tmp_path, workers=2),
            deploy_api=api,
            runner=_UnusedRunner(),
        )

    assert api.bundle_calls == 0
    assert events == []


def test_one_click_rolls_back_profile_and_dns_on_paid_phase_failure(
    automatic_fakes,
    tmp_path: Path,
) -> None:
    events, _caddy = automatic_fakes
    api = _FakeDeployApi(events, fail_phase="provision-only")

    with pytest.raises(one.OneClickDeploymentError, match="rc=7"):
        one.run_secure_verda_first_install(
            _args(tmp_path),
            deploy_api=api,
            runner=_UnusedRunner(),
        )

    assert "rollback" in events
    assert _FakeDns.instances[-1].rolled_back is True
    assert "activate" not in events
    assert _FakeFly.instances[-1].events[-1] == "fly-disabled-rollback"


class _DnsRunner(one.CommandRunner):
    def __init__(self):
        self.records: list[dict[str, str]] = []
        self.commands: list[list[str]] = []
        self.counter = 0

    def run(self, command, **kwargs):
        del kwargs
        command = list(command)
        self.commands.append(command)
        if "?limit=100" in command[2]:
            payload = {"records": list(self.records)}
        elif "DELETE" in command:
            record_id = command[2].rsplit("/", 1)[1]
            self.records = [row for row in self.records if row["id"] != record_id]
            payload = {"ok": True}
        else:
            value = next(item.split("=", 1)[1] for item in command if item.startswith("value="))
            self.counter += 1
            row = {
                "id": f"rec-{self.counter}",
                "name": "gpu-worker",
                "type": "A",
                "value": value,
            }
            self.records.append(row)
            payload = row
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")


def test_vercel_dns_replace_and_rollback_are_exact() -> None:
    runner = _DnsRunner()
    runner.records = [
        {"id": "old", "name": "gpu-worker", "type": "A", "value": "198.51.100.2"}
    ]
    dns = one.VercelDns(
        runner=runner,
        hostname="gpu-worker.anapana.ai",
        domain="anapana.ai",
        executable="vercel",
    )
    previous = dns.preflight()
    mutation = dns.point_to("203.0.113.10", previous)
    assert runner.records[0]["value"] == "203.0.113.10"

    dns.rollback(mutation)

    assert runner.records == [
        {"id": "rec-2", "name": "gpu-worker", "type": "A", "value": "198.51.100.2"}
    ]


def test_caddy_archive_requires_exact_digest_and_regular_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        data = b"verified-caddy"
        member = tarfile.TarInfo("caddy")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    raw = payload.getvalue()
    monkeypatch.setattr(one, "CADDY_ARCHIVE_SHA256", __import__("hashlib").sha256(raw).hexdigest())

    class Runner(one.CommandRunner):
        def run(self, command, **kwargs):
            del kwargs
            target = Path(command[command.index("-o") + 1])
            target.write_bytes(raw)
            return subprocess.CompletedProcess(command, 0, "", "")

    directory = one._download_caddy(Runner())
    try:
        binary = Path(directory.name) / "caddy"
        assert binary.read_bytes() == b"verified-caddy"
        assert binary.stat().st_mode & 0o111
    finally:
        directory.cleanup()


def test_exact_instance_resume_rejects_identity_drift_before_ssh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cli = tmp_path / "verda"
    cli.write_text("cli")
    args = Namespace(
        verda_cli=cli,
        verda_hostname="filmforge-verda-worker",
        verda_location="FIN-03",
        verda_instance_type="1A100.22V",
    )
    instance_id = "c" * 32
    monkeypatch.setattr(deploy_gpu, "_verda_check", lambda *a, **k: "ok")
    monkeypatch.setattr(
        deploy_gpu,
        "_verda_json",
        lambda *a, **k: [
            {
                "id": instance_id,
                "hostname": "different-worker",
                "status": "running",
                "ip": "203.0.113.12",
                "location": "FIN-03",
                "instance_type": "1A100.22V",
            }
        ],
    )

    with pytest.raises(RuntimeError, match="hostname drifted"):
        deploy_gpu._resume_verda_instance(args, instance_id)


def test_exact_instance_resume_returns_only_matching_running_vm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cli = tmp_path / "verda"
    cli.write_text("cli")
    args = Namespace(
        verda_cli=cli,
        verda_hostname="filmforge-verda-worker",
        verda_location="FIN-03",
        verda_instance_type="1A100.22V",
    )
    instance_id = "d" * 32
    monkeypatch.setattr(deploy_gpu, "_verda_check", lambda *a, **k: "ok")
    monkeypatch.setattr(
        deploy_gpu,
        "_verda_json",
        lambda *a, **k: [
            {
                "id": instance_id,
                "hostname": "filmforge-verda-worker",
                "status": "running",
                "ip": "203.0.113.13",
                "location": "FIN-03",
                "instance_type": "1A100.22V",
            }
        ],
    )

    assert deploy_gpu._resume_verda_instance(args, instance_id) == (
        "203.0.113.13",
        instance_id,
    )
