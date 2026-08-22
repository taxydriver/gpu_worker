"""Per-GPU department plan for a multi-GPU Verda box.

One GPU = one department. A 4-GPU box runs two generation workers (FLUX/WAN
inside ComfyUI), one vision worker (resident vLLM/Qwen3-VL) and one audio worker
(resident Parler + SA3), all four sharing the one /mnt/data volume.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gpu_worker import deploy_gpu

PLAN = list(deploy_gpu.DEFAULT_4GPU_WORKER_PLAN)


def _script(plan: list[str] | None = None, **kwargs) -> str:
    return deploy_gpu.verda_rehydrate_script(
        public_ip="203.0.113.9",
        worker_port=9000,
        comfy_port=8188,
        worker_count=0,
        remote_root="/opt/filmforge_gpu_worker",
        worker_plan=plan,
        **kwargs,
    )


def _fresh_script() -> str:
    return deploy_gpu.verda_fresh_install_script(
        worker_repo_url="https://example.com/worker.git",
        comfy_repo_url="https://example.com/comfy.git",
        pytorch_index_url="https://example.com/torch",
        remote_root="/opt/filmforge_gpu_worker",
    )


# ── plan parsing ──────────────────────────────────────────────────────────────


def test_empty_spec_means_no_plan() -> None:
    assert deploy_gpu.parse_worker_plan("") == []
    assert deploy_gpu.parse_worker_plan("   ") == []


def test_parses_and_normalizes() -> None:
    assert deploy_gpu.parse_worker_plan(" Generation, generation ,VISION, audio ") == PLAN


def test_rejects_unknown_department() -> None:
    with pytest.raises(RuntimeError, match="Unknown worker-plan department"):
        deploy_gpu.parse_worker_plan("generation,rendering")


def test_rejects_two_of_a_resident_singleton() -> None:
    # Parler/SA3 and vLLM bind fixed ports and share one model cache — a second
    # copy on the same box would fight the first for both.
    with pytest.raises(RuntimeError, match="At most one vision and one audio"):
        deploy_gpu.parse_worker_plan("vision,vision,audio,generation")
    with pytest.raises(RuntimeError, match="At most one vision and one audio"):
        deploy_gpu.parse_worker_plan("audio,audio")


# ── generated deploy script ───────────────────────────────────────────────────


def test_script_is_valid_bash(tmp_path: Path) -> None:
    path = tmp_path / "deploy.sh"
    path.write_text(_script(PLAN))
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_fresh_install_script_is_valid_bash(tmp_path: Path) -> None:
    path = tmp_path / "fresh-install.sh"
    path.write_text(_fresh_script())
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_fresh_install_only_formats_a_safely_blank_data_volume() -> None:
    script = _fresh_script()

    assert "wipefs -n --noheadings --output TYPE /dev/vdb" in script
    assert script.count("dd if=/dev/vdb bs=4M") == 2
    assert "refusing to format it automatically" in script
    assert "mkfs.ext4 -F /dev/vdb" in script
    assert "touch /mnt/data/.filmforge-bootstrap-complete" in script


def test_fresh_install_repairs_inconsistent_verda_ubuntu_mirror() -> None:
    script = _fresh_script()

    assert "apt-get install --simulate" in script
    assert "https://archive.ubuntu.com/ubuntu/" in script
    assert "https://security.ubuntu.com/ubuntu/" in script
    assert "switching to canonical HTTPS mirrors" in script
    assert "/var/backups/filmforge-apt" in script
    assert ".filmforge-original" not in script


def test_rehydrate_requires_an_initialized_ext4_data_volume() -> None:
    script = _script(PLAN)

    assert 'if test "$VDB_FS" != "ext4"; then' in script
    assert "deploy preflight should bootstrap an incomplete volume pair" in script


def test_rehydrate_repairs_nvswitch_fabric_startup_race_once() -> None:
    script = _script(PLAN)

    assert "test -e /dev/nvidia-nvswitchctl" in script
    assert "timeout 180 systemctl restart nvidia-fabricmanager.service" in script
    assert "provider-side GPU fabric fault" in script


def test_plan_reaches_the_script() -> None:
    assert "WORKER_PLAN_SPEC=generation,generation,vision,audio" in _script(PLAN)


def test_no_plan_keeps_the_homogeneous_box() -> None:
    script = _script(None)
    assert "WORKER_PLAN_SPEC=''" in script
    # Every worker still falls back to the same capability broadcast.
    assert "${WORKER_CAPABILITIES:-flux2_stills,wan_i2v,ltx_i2v,character_loras}" in script


def test_worker_units_honor_deploy_concurrency_environment() -> None:
    script = _script(PLAN)
    assert "Environment=WORKER_MAX_CONCURRENT_JOBS=${WORKER_MAX_CONCURRENT_JOBS:-10}" in script
    assert "Environment=WORKER_MAX_CONCURRENT_JOBS=1" not in script


def test_capabilities_are_per_department() -> None:
    script = _script(PLAN)
    assert 'vision) echo "qwen_vision" ;;' in script
    assert 'audio)  echo "tts_dialogue,stable_audio3" ;;' in script


def test_comfyui_only_starts_on_generation_cards() -> None:
    script = _script(PLAN)
    # Unit creation and both enable/health loops are gated on the department —
    # a second ComfyUI on the vision card would hold VRAM the resident vLLM needs.
    assert script.count('test "$(dept_for_idx "$idx")" = "generation" || continue') == 2
    assert 'if test "$dept" = "generation"; then' in script


def test_comfyui_uses_a_per_gpu_database() -> None:
    script = _script(PLAN)
    assert "--database-url sqlite:///${comfy_user_dir}/comfyui.db" in script


def test_vision_card_advertises_the_vllm_url_for_discovery() -> None:
    # Verda boxes have a public IP, so no cloudflared hop: the backend's
    # GET /api/render-broker/vision-worker reads this straight off registration.
    assert (
        "Environment=WORKER_VISION_BASE_URL=http://${PUBLIC_IP}:${VLLM_PORT}/v1"
        in _script(PLAN)
    )
    assert "VLLM_PORT=8100" in _script(PLAN)
    assert "VLLM_PORT=8300" in _script(PLAN, vllm_port=8300)


def test_non_generation_workers_get_a_dead_comfy_url() -> None:
    # Unset would default to :8188 — gpu0's ComfyUI, i.e. another card.
    assert "Environment=COMFY_BASE_URL=http://127.0.0.1:1" in _script(PLAN)


def test_audio_servers_are_pinned_to_the_plan_card() -> None:
    script = _script(PLAN)
    assert 'export AUDIO_GPU_INDEX="$AUDIO_GPU_IDX"' in script
    # The plan already wrote the worker unit's capabilities; setup_audio_services
    # must not rewrite them (it would target gpu0).
    assert "export AUDIO_SKIP_WORKER_CAPS=1" in script


def test_plan_capabilities_are_re_asserted_after_the_provisioners() -> None:
    # provision_*.sh / setup_audio_services.sh are read from the BOX's git
    # checkout, not from the deploy we pipe in — an older checkout bolts audio
    # caps onto gpu0 and restarts it, recreating the department mix. Observed
    # live on 2026-07-26; the plan gets the last word.
    script = _script(PLAN)
    guard = script.index("Re-assert the plan's capabilities")
    audio = script.index("setting up the sound stage")
    assert guard > audio, "the drift guard must run AFTER the provisioners"
    assert "capabilities drifted from the plan" in script
    # Second half of the same drift: an old script leaves the resident audio
    # servers unpinned, so they load onto GPU 0 (seen live — Parler took 4.9GB
    # of a render card).
    assert "is not pinned to gpu${AUDIO_GPU_IDX} — re-pinning" in script


def test_drift_guard_is_a_no_op_without_a_plan() -> None:
    script = _script(None)
    assert 'if test "${#WORKER_PLAN[@]}" -gt 0; then' in script


def test_stale_resident_services_are_stopped_before_comfyui_starts() -> None:
    """filmforge-vllm/-parler/-sa3 live on the OS VOLUME, which is reattached on
    every deploy — so they boot even when this deploy has no vision/audio card.
    Seen twice on 2026-07-26: an 83GB vLLM on gpu2 while that card's worker
    advertised wan_i2v (guaranteed OOM). Must run before ComfyUI loads, so the
    reclaimed card is already free.
    """
    script = _script(PLAN)
    guard = script.index("_stop_stale_resident")
    comfy_start = script.index('systemctl enable --now "comfyui-gpu')
    assert guard < comfy_start, "stale residents must be stopped BEFORE ComfyUI starts"
    # Wanted departments are the union of the plan and the capability broadcast,
    # so a legacy caps-only audio box keeps its servers.
    assert 'case ",${WORKER_PLAN_SPEC}," in *,vision,*) _wants_vision=1 ;; esac' in script
    assert '*,tts_dialogue,*|*,stable_audio3,*) _wants_audio=1 ;;' in script


def test_stale_resident_guard_is_present_without_a_plan() -> None:
    # The no-plan case is the one that bit: 4 generation workers with a live vLLM.
    assert "_stop_stale_resident filmforge-vllm" in _script(None)


def test_audio_still_triggers_on_capabilities_without_a_plan() -> None:
    # The pre-plan single-department box is unchanged.
    assert "*,tts_dialogue,*|*,stable_audio3,*) _audio_wanted=1 ;;" in _script(None)


def _provision_only_exit(script: str) -> int:
    # Select the rehydrate body's provision-only gate, not the earlier security
    # stage gate which emits the same receipt token.
    return script.index("WORKER_RELEASE_STAGED_ONLY=", script.index("wait_comfy_healthy()"))


def test_infinitetalk_assets_and_readiness_precede_secure_cutover() -> None:
    for script in (_script(PLAN), _script(None)):
        provision = script.index("bash provision_infinitetalk.sh")
        restart = script.index('systemctl restart "comfyui-gpu${idx}.service"', provision)
        rewait = script.index("wait_comfy_healthy", restart)
        materialize = script.index('ensure_asset_group("infinitetalk_v1")', rewait)
        readiness = script.index("readiness = check_infinitetalk_readiness()", materialize)
        seal = script.index("immutable worker candidate failed pre-cutover seal", readiness)
        gate = _provision_only_exit(script)
        assert provision < restart < rewait < materialize < readiness < seal < gate
        assert "bash provision_infinitetalk.sh" not in script[gate:]
        assert 'ensure_asset_group("infinitetalk_v1")' not in script[gate:]
        assert '*,infinitetalk,*|*,infinitetalk_v1,*' in script
        assert 'COMFY_BASE_URL="http://127.0.0.1:${_infinitetalk_comfy_port}"' in script
        assert 'COMFY_DIR="$COMFY_ROOT"' in script
        assert 'PYTHONDONTWRITEBYTECODE=1' in script[materialize - 500:materialize]
        assert 'PYTHONPATH="$WORKER_MODULE_DIR"' in script
        assert '"$WORKER_ROOT/.venv/bin/python" -' in script
        assert "InfiniteTalk readiness failed before secure cutover" in script
        assert "worker release candidate is incomplete; refusing staged receipt" in script


def test_infinitetalk_provisioner_serializes_ensure_and_rehydrate_callers() -> None:
    provisioner = Path(deploy_gpu.__file__).with_name("provision_infinitetalk.sh")
    source = provisioner.read_text()
    subprocess.run(["bash", "-n", str(provisioner)], check=True)

    assert 'LOCK_FILE="$COMFY/.filmforge_infinitetalk.provision.lock"' in source
    assert 'flock -w "${INFINITETALK_PROVISION_LOCK_TIMEOUT_SEC:-1800}" 9' in source
    # The mutating clone path is strictly after the lock acquisition, so both
    # rehydrate and asset-manager ensure calls use the same filesystem lock.
    assert source.index("flock -w") < source.index("node ComfyUI-WanVideoWrapper")


def test_hf_token_is_injected_for_a_plan_with_audio(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    args = SimpleNamespace(
        env_vars=[],
        backend_env=tmp_path / "missing.env",
        verda_worker_plan=",".join(PLAN),
    )
    assert "HF_TOKEN=hf_test_token" in deploy_gpu._verda_env_vars(args)
