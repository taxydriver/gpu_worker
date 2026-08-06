from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from gpu_worker import deploy_gpu


def test_run_forwards_timeout_to_subprocess(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(deploy_gpu.subprocess, "run", fake_run)

    deploy_gpu.run(["true"], timeout=37)

    assert captured["timeout"] == 37


def test_ssh_liveness_policy_detects_recycled_vm() -> None:
    cmd = deploy_gpu.add_default_ssh_liveness_policy(
        ["ssh", "-i", "/tmp/key", "root@203.0.113.10"]
    )

    assert "ConnectTimeout=8" in cmd
    assert "BatchMode=yes" in cmd
    assert "ServerAliveInterval=15" in cmd
    assert "ServerAliveCountMax=3" in cmd


def test_ssh_liveness_policy_preserves_explicit_values() -> None:
    original = [
        "ssh",
        "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=5",
        "root@203.0.113.10",
    ]

    cmd = deploy_gpu.add_default_ssh_liveness_policy(original)

    assert cmd.count("ConnectTimeout=20") == 1
    assert "ConnectTimeout=8" not in cmd
    assert cmd.count("ServerAliveInterval=5") == 1
    assert "ServerAliveInterval=15" not in cmd


def test_verda_ssh_timeout_becomes_actionable_error(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def timeout_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(deploy_gpu, "run", timeout_run)

    with pytest.raises(RuntimeError, match="VM may have been interrupted"):
        deploy_gpu._run_verda_ssh_script(
            ["ssh", "root@203.0.113.10"],
            "echo ok",
            timeout_sec=123,
            capture_output=False,
            operation="installing test stack",
        )

    assert captured["timeout"] == 123
    assert captured["capture_output"] is False
    assert captured["cmd"][-2:] == ["bash", "-s"]


def test_verda_state_probe_parses_remote_result(monkeypatch) -> None:
    def completed(*args, **kwargs):
        return subprocess.CompletedProcess(
            ["ssh"],
            0,
            stdout=(
                "DATA_STATE=blank\nWORKER_READY=0\nCOMFY_READY=0\n"
                "BOOTSTRAP_READY=0\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(deploy_gpu, "_run_verda_ssh_script", completed)

    assert deploy_gpu._probe_verda_rehydrate_state(["ssh", "root@example"]) == {
        "DATA_STATE": "blank",
        "WORKER_READY": "0",
        "COMFY_READY": "0",
        "BOOTSTRAP_READY": "0",
    }


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"DATA_STATE": "blank", "WORKER_READY": "0", "COMFY_READY": "0", "BOOTSTRAP_READY": "0"}, True),
        ({"DATA_STATE": "filesystem:ext4", "WORKER_READY": "0", "COMFY_READY": "1", "BOOTSTRAP_READY": "1"}, True),
        ({"DATA_STATE": "filesystem:ext4", "WORKER_READY": "1", "COMFY_READY": "1", "BOOTSTRAP_READY": "0"}, True),
        ({"DATA_STATE": "filesystem:ext4", "WORKER_READY": "1", "COMFY_READY": "1", "BOOTSTRAP_READY": "1"}, False),
    ],
)
def test_verda_pair_bootstrap_decision(state, expected) -> None:
    assert deploy_gpu._verda_pair_needs_bootstrap(state) is expected


@pytest.mark.parametrize("data_state", ["unknown", "missing", "filesystem:xfs"])
def test_verda_pair_rejects_unsafe_data_volume(data_state) -> None:
    state = {
        "DATA_STATE": data_state,
        "WORKER_READY": "0",
        "COMFY_READY": "0",
        "BOOTSTRAP_READY": "0",
    }

    with pytest.raises(RuntimeError, match="refusing|unsupported"):
        deploy_gpu._verda_pair_needs_bootstrap(state)


def test_verda_spot_deploy_retries_remote_disconnect(monkeypatch) -> None:
    attempts = 0

    def deploy(args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise subprocess.CalledProcessError(255, ["ssh"])
        return 0

    monkeypatch.setattr(deploy_gpu, "verda_deploy", deploy)
    monkeypatch.setattr(deploy_gpu, "_wait_for_interrupted_verda_pair", lambda args: True)
    monkeypatch.setenv("VERDA_SPOT_DEPLOY_ATTEMPTS", "3")

    result = deploy_gpu.verda_deploy_with_spot_retries(
        SimpleNamespace(verda_contract="spot")
    )

    assert result == 0
    assert attempts == 2


def test_verda_spot_deploy_does_not_retry_regular_remote_error(monkeypatch) -> None:
    attempts = 0

    def deploy(args):
        nonlocal attempts
        attempts += 1
        raise subprocess.CalledProcessError(1, ["ssh"])

    monkeypatch.setattr(deploy_gpu, "verda_deploy", deploy)

    with pytest.raises(subprocess.CalledProcessError):
        deploy_gpu.verda_deploy_with_spot_retries(
            SimpleNamespace(verda_contract="spot")
        )

    assert attempts == 1
