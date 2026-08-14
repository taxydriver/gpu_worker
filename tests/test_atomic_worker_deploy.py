from __future__ import annotations

import inspect
import subprocess

import pytest

from gpu_worker import deploy_gpu


def _complete_contract(*, count: int = 1) -> list[str]:
    public_urls = ",".join(
        f"https://gpu{index}.worker.example" for index in range(count)
    )
    local_urls = ",".join(
        f"http://127.0.0.1:{9000 + index}" for index in range(count)
    )
    tunnel_units = ",".join(
        f"filmforge-worker-tunnel-gpu{index}.service" for index in range(count)
    )
    stage_receipts = ",".join(
        f"/etc/filmforge/worker-security/releases/profile-gpu{index}/stage-receipt.json"
        for index in range(count)
    )
    return [
        "GPU_WORKER_API_TOKEN=fixture-worker-secret",
        "WORKER_REGISTRATION_TOKEN=fixture-registration-secret",
        "FILMFORGE_BACKEND_URL=https://backend.example",
        "WORKER_API_AUTH_MODE=required",
        "FILMFORGE_BACKEND_CLIENT_AUTH_MODE=bearer",
        "WORKER_DEPLOY_PHASE=activate",
        f"WORKER_PUBLIC_URLS={public_urls}",
        f"WORKER_TUNNEL_LOCAL_URLS={local_urls}",
        f"WORKER_TUNNEL_UNITS={tunnel_units}",
        f"WORKER_SECURITY_STAGE_RECEIPTS={stage_receipts}",
    ]


def test_complete_contract_preflight_passes_before_any_gpu_setup() -> None:
    result = deploy_gpu._preflight_complete_worker_contract(
        _complete_contract(count=2),
        worker_port=9000,
        expected_worker_count=2,
    )
    assert result["backend_auth_mode"] == "bearer"
    assert result["public_urls"] == [
        "https://gpu0.worker.example",
        "https://gpu1.worker.example",
    ]

    source = inspect.getsource(deploy_gpu._do_deploy)
    assert source.index("_preflight_complete_worker_contract") < source.index(
        "_stage_worker_release_over_ssh"
    )
    assert "git pull failed" not in source
    assert "git clone" not in source

    runpod_source = inspect.getsource(deploy_gpu.runpod_deploy)
    assert runpod_source.index("_preflight_complete_worker_contract") < runpod_source.index(
        "rp.create_pod"
    )
    vast_source = inspect.getsource(deploy_gpu.vast_deploy)
    assert vast_source.index("_preflight_complete_worker_contract") < vast_source.index(
        "create = _vastai"
    )
    for deploy_function in (deploy_gpu.verda_deploy, deploy_gpu.verda_fresh_deploy):
        verda_source = inspect.getsource(deploy_function)
        assert verda_source.index("_preflight_complete_worker_contract") < verda_source.index(
            "_verda_check"
        )
        assert verda_source.index("_prepare_worker_release_bundle") < verda_source.index(
            "_verda_check"
        )
        assert 'bundle=getattr(args, "_prepared_worker_release_bundle", None)' in verda_source

    verda_source = inspect.getsource(deploy_gpu.verda_deploy)
    assert verda_source.index('if deploy_phase == "stage-code"') < verda_source.index(
        "_verda_pair_needs_bootstrap"
    )
    fresh_source = inspect.getsource(deploy_gpu.verda_fresh_deploy)
    assert fresh_source.index('if deploy_phase == "stage-code"') < fresh_source.index(
        "verda_fresh_install_script"
    )

    runpod_source = inspect.getsource(deploy_gpu.runpod_deploy)
    assert runpod_source.index("_prepare_worker_release_bundle") < runpod_source.index(
        "rp.create_pod"
    )
    vast_source = inspect.getsource(deploy_gpu.vast_deploy)
    assert vast_source.index("_prepare_worker_release_bundle") < vast_source.index(
        'create = _vastai'
    )


@pytest.mark.parametrize(
    "missing_key",
    [
        "GPU_WORKER_API_TOKEN",
        "WORKER_REGISTRATION_TOKEN",
        "FILMFORGE_BACKEND_URL",
        "WORKER_API_AUTH_MODE",
        "FILMFORGE_BACKEND_CLIENT_AUTH_MODE",
        "WORKER_PUBLIC_URLS",
        "WORKER_TUNNEL_LOCAL_URLS",
        "WORKER_TUNNEL_UNITS",
        "WORKER_SECURITY_STAGE_RECEIPTS",
        "WORKER_DEPLOY_PHASE",
    ],
)
def test_preflight_refuses_every_half_configured_contract(missing_key: str) -> None:
    env_vars = [
        item
        for item in _complete_contract()
        if not item.startswith(f"{missing_key}=")
    ]
    with pytest.raises(RuntimeError, match=missing_key):
        deploy_gpu._preflight_complete_worker_contract(
            env_vars,
            worker_port=9000,
            expected_worker_count=1,
        )


def test_preflight_refuses_missing_backend_bearer_client() -> None:
    env_vars = [
        "FILMFORGE_BACKEND_CLIENT_AUTH_MODE=none"
        if item.startswith("FILMFORGE_BACKEND_CLIENT_AUTH_MODE=")
        else item
        for item in _complete_contract()
    ]
    with pytest.raises(RuntimeError, match="bearer-sending client"):
        deploy_gpu._preflight_complete_worker_contract(
            env_vars,
            worker_port=9000,
            expected_worker_count=1,
        )


def test_preflight_refuses_tls_tunnel_cardinality_and_port_drift() -> None:
    with pytest.raises(RuntimeError, match="equal cardinality"):
        wrong_units = [
            "WORKER_TUNNEL_UNITS=filmforge-worker-tunnel-gpu0.service"
            if item.startswith("WORKER_TUNNEL_UNITS=")
            else item
            for item in _complete_contract(count=2)
        ]
        deploy_gpu._preflight_complete_worker_contract(
            wrong_units,
            worker_port=9000,
            expected_worker_count=2,
        )

    wrong_port = [
        "WORKER_TUNNEL_LOCAL_URLS=http://127.0.0.1:19000"
        if item.startswith("WORKER_TUNNEL_LOCAL_URLS=")
        else item
        for item in _complete_contract()
    ]
    with pytest.raises(RuntimeError, match="loopback worker ports"):
        deploy_gpu._preflight_complete_worker_contract(
            wrong_port,
            worker_port=9000,
            expected_worker_count=1,
        )


def test_h100_dirty_checkout_can_never_be_pulled_or_tolerated_again() -> None:
    """Regression for a526cb5 + local edits/untracked 915de05 on /opt."""

    rehydrate = deploy_gpu.verda_rehydrate_script(
        public_ip="203.0.113.10",
        worker_port=9000,
        comfy_port=18188,
        worker_count=1,
        remote_root="/opt/filmforge_gpu_worker",
        worker_source_root=(
            "/opt/filmforge-worker-releases/current/gpu_worker"
        ),
    )
    fresh = deploy_gpu.verda_fresh_install_script(
        worker_repo_url="https://example.invalid/gpu_worker.git",
        comfy_repo_url="https://example.invalid/ComfyUI.git",
        pytorch_index_url="https://example.invalid/cu130",
        remote_root="/opt/filmforge_gpu_worker",
        worker_source_root=(
            "/opt/filmforge-worker-releases/current/gpu_worker"
        ),
    )

    for script in (rehydrate, fresh):
        assert 'git -C "$WORKER_ROOT"' not in script
        assert 'test -d "$WORKER_ROOT/.git"' not in script
        assert "WORKER_REPO_URL" not in script
        assert "pull --ff-only </dev/null 2>&1 || true" not in script
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "immutable gpu_worker release is missing" in rehydrate
    assert "immutable gpu_worker release is missing" in fresh


def test_systemd_worker_credentials_are_mode_0600_environment_files() -> None:
    scripts = [
        deploy_gpu.vast_multi_gpu_script(
            remote_root="/workspace/runtime",
            worker_port=9000,
            comfy_port=18188,
            worker_count=1,
            worker_source_root="/workspace/releases/current/gpu_worker",
        ),
        deploy_gpu.verda_rehydrate_script(
            public_ip="203.0.113.10",
            worker_port=9000,
            comfy_port=18188,
            worker_count=1,
            remote_root="/opt/filmforge_gpu_worker",
            worker_source_root=(
                "/opt/filmforge-worker-releases/current/gpu_worker"
            ),
        ),
    ]
    credential_keys = (
        "GPU_WORKER_API_TOKEN",
        "WORKER_API_TOKEN",
        "WORKER_REGISTRATION_TOKEN",
        "RENDER_BROKER_WORKER_TOKEN",
    )
    for script in scripts:
        assert "install -m 0600" in script
        assert "EnvironmentFile=$worker_secret_env" in script
        assert "Environment=PYTHONDONTWRITEBYTECODE=1" in script
        for key in credential_keys:
            assert f"Environment={key}=" not in script
            assert f'echo "Environment={key}=' not in script
        assert script.index("install -m 0600") < script.index(
            'cat > "/etc/systemd/system/filmforge-worker-gpu'
        )
        assert 'nohup env "${worker_env[@]}"' not in script
        assert script.index("secure-profile stage receipt") < script.index("nvidia-smi")
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_remote_bootstrap_never_exports_worker_bearers_to_installers() -> None:
    exports = deploy_gpu.build_bootstrap_env_exports(
        [
            "GPU_WORKER_API_TOKEN=worker-secret",
            "WORKER_REGISTRATION_TOKEN=registration-secret",
            "RENDER_BROKER_WORKER_TOKEN=broker-secret",
            "FILMFORGE_BACKEND_URL=https://backend.example",
            "WORKER_API_AUTH_MODE=required",
        ]
    )
    assert "worker-secret" not in exports
    assert "registration-secret" not in exports
    assert "broker-secret" not in exports
    assert "FILMFORGE_BACKEND_URL=https://backend.example" in exports
    assert "WORKER_API_AUTH_MODE=required" in exports


def test_worker_urls_are_returned_only_from_verified_receipt_contract() -> None:
    pinned = ["https://gpu0.worker.example"]
    release_id = "sha256-" + "a" * 24

    assert deploy_gpu._verified_receipt_worker_urls(
        "WORKER_RELEASE_STAGED=1\nWORKER_URL=https://stale.example\n",
        release_id=release_id,
        public_urls=pinned,
    ) == []
    assert deploy_gpu._verified_receipt_worker_urls(
        f"WORKER_RELEASE_VERIFIED={release_id}\nWORKER_URL=https://evil.example\n",
        release_id=release_id,
        public_urls=pinned,
    ) == pinned

    source = inspect.getsource(deploy_gpu._do_deploy)
    assert "_verified_receipt_worker_urls" in source
    assert "_runpod_proxy_url" not in source
    assert "GPU_WORKER_BASE_URL" not in source
    assert "extract_worker_url(remote_result.stdout)" not in source


def test_activation_failure_rolls_back_profile_before_code_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        deploy_gpu,
        "_rollback_secure_profiles_over_ssh",
        lambda **_kwargs: events.append("profile"),
    )
    monkeypatch.setattr(
        deploy_gpu,
        "_rollback_worker_release_over_ssh",
        lambda **_kwargs: events.append("code"),
    )

    assert deploy_gpu._rollback_failed_worker_transaction_over_ssh(
        ssh_cmd=["ssh", "worker"],
        releases_root="/opt/releases",
        worker_source_root="/opt/releases/releases/sha256-a/gpu_worker",
        failed_release_id="sha256-a",
        stage_receipt_paths=["/etc/filmforge/stage.json"],
        deploy_phase="activate",
    )
    assert events == ["profile", "code"]


def test_profile_rollback_failure_never_claims_pointer_only_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def fail_profile(**_kwargs) -> None:
        events.append("profile")
        raise RuntimeError("profile rollback failed")

    monkeypatch.setattr(
        deploy_gpu,
        "_rollback_secure_profiles_over_ssh",
        fail_profile,
    )
    monkeypatch.setattr(
        deploy_gpu,
        "_rollback_worker_release_over_ssh",
        lambda **_kwargs: events.append("code"),
    )

    assert not deploy_gpu._rollback_failed_worker_transaction_over_ssh(
        ssh_cmd=["ssh", "worker"],
        releases_root="/opt/releases",
        worker_source_root="/opt/releases/releases/sha256-a/gpu_worker",
        failed_release_id="sha256-a",
        stage_receipt_paths=["/etc/filmforge/stage.json"],
        deploy_phase="activate",
    )
    assert events == ["profile"]


def test_finalize_holds_profile_lock_while_revalidating_and_promoting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_run(_cmd, **kwargs):
        captured["script"] = kwargs["input_text"]
        return subprocess.CompletedProcess(_cmd, 0, "", "")

    monkeypatch.setattr(deploy_gpu, "run", fake_run)
    release_id = "sha256-" + "a" * 24
    deploy_gpu._activate_worker_release_over_ssh(
        ssh_cmd=["ssh", "worker"],
        releases_root="/opt/releases",
        release_id=release_id,
        worker_source_root=f"/opt/releases/releases/{release_id}/gpu_worker",
        stage_receipt_paths=["/etc/filmforge/worker-security/releases/p0/stage-receipt.json"],
    )

    script = captured["script"]
    assert "/etc/filmforge/worker-security/.profile.lock" in script
    assert script.index("flock 9") < script.index(
        "secure-profile cutover is not complete"
    )
    assert script.index("secure-profile cutover is not complete") < script.index(
        'CURRENT="$RELEASES_ROOT/current"'
    )
    assert "first-install boot authorization drifted" in script
    assert '"systemctl", "is-active", "--quiet"' in script
    assert "worker health identity changed before code finalization" in script
    assert 'health.get("code_release_id") != failed_release_id' in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_remote_profile_rollback_prevalidates_and_uses_one_batch_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_run(_cmd, **kwargs):
        captured["script"] = kwargs["input_text"]
        return subprocess.CompletedProcess(_cmd, 0, "", "")

    monkeypatch.setattr(deploy_gpu, "run", fake_run)
    release_id = "sha256-" + "a" * 24
    deploy_gpu._rollback_secure_profiles_over_ssh(
        ssh_cmd=["ssh", "worker"],
        worker_source_root=f"/opt/releases/releases/{release_id}/gpu_worker",
        failed_release_id=release_id,
        stage_receipt_paths=[
            "/etc/filmforge/worker-security/releases/p0/stage-receipt.json",
            "/etc/filmforge/worker-security/releases/p1/stage-receipt.json",
        ],
    )

    script = captured["script"]
    assert script.index("profiles = []") < script.index(
        "rollback_secure_profiles(release_ids=profiles)"
    )
    assert script.index("sys.dont_write_bytecode = True") < script.index(
        "from gpu_worker.worker_release import rollback_secure_profiles"
    )
    assert "from gpu_worker.worker_release import rollback_secure_profiles" in script
    assert "rollback_state" in script
    compile(script, "<remote-profile-rollback>", "exec")
