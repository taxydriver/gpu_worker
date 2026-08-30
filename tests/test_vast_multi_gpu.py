from __future__ import annotations

from types import SimpleNamespace

from gpu_worker import deploy_gpu


def _vast_args(*, worker_count: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        vast_min_vram_gb=24,
        vast_max_price=1.0,
        vast_disk_gb=200,
        vast_max_upload_cost=None,
        vast_max_download_cost=None,
        vast_limit=25,
        vast_gpu="RTX A4000",
        vast_allow_fallback_gpu=False,
        vast_worker_count=worker_count,
    )


def test_vast_offer_gpu_count_handles_common_payload_shapes() -> None:
    assert deploy_gpu._vast_offer_gpu_count({"num_gpus": "2"}) == 2
    assert deploy_gpu._vast_offer_gpu_count({"gpus": [{"id": 0}, {"id": 1}]}) == 2
    assert deploy_gpu._vast_offer_gpu_count({"gpu_name": "NVIDIA RTX A6000 x2"}) == 2
    assert deploy_gpu._vast_offer_gpu_count({"gpu_name": "2x RTX A4000"}) == 2
    assert deploy_gpu._vast_offer_gpu_count({"gpu_ram": 24, "gpu_total_ram": 48}) == 2
    # A model name alone is not authoritative proof of count; secure deploys
    # must refuse rather than guess one before provider mutation.
    assert deploy_gpu._vast_offer_gpu_count({"gpu_name": "RTX 4090"}) == 0


def test_select_vast_offer_prefers_requested_multi_gpu_offer(monkeypatch) -> None:
    offers = [
        {
            "id": "single",
            "gpu_name": "RTX A4000",
            "num_gpus": 1,
            "dph_total": 0.20,
            "inet_up_cost": 0.01,
            "inet_down_cost": 0.01,
            "reliability2": 0.99,
        },
        {
            "id": "dual",
            "gpu_name": "RTX A4000",
            "num_gpus": 2,
            "dph_total": 0.40,
            "inet_up_cost": 0.01,
            "inet_down_cost": 0.01,
            "reliability2": 0.90,
        },
    ]

    monkeypatch.setattr(deploy_gpu, "_vastai", lambda *args, **kwargs: offers)

    selected = deploy_gpu._select_vast_offer(_vast_args(worker_count=2))

    assert selected["id"] == "dual"


def test_select_vast_offer_falls_back_when_requested_multi_gpu_is_unavailable(monkeypatch) -> None:
    offers = [
        {
            "id": "single",
            "gpu_name": "RTX A4000",
            "num_gpus": 1,
            "dph_total": 0.20,
            "inet_up_cost": 0.01,
            "inet_down_cost": 0.01,
            "reliability2": 0.99,
        }
    ]

    monkeypatch.setattr(deploy_gpu, "_vastai", lambda *args, **kwargs: offers)

    selected = deploy_gpu._select_vast_offer(_vast_args(worker_count=2))

    assert selected["id"] == "single"


def test_multi_gpu_script_is_gpu_agnostic_and_starts_comfy_per_gpu() -> None:
    """The multi-GPU script must auto-detect physical GPUs and provision a
    comfyui + worker systemd unit per GPU — the bug was the SSH path using the
    legacy single-worker remote_script (no ComfyUI, one worker on 9000)."""
    script = deploy_gpu.vast_multi_gpu_script(
        remote_root="/workspace/filmforge_gpu_worker",
        worker_port=9000,
        comfy_port=18188,
        worker_count=0,  # 0 = auto-detect from PHYSICAL_GPU_COUNT
    )
    # GPU-agnostic: count comes from the box, not a hardcoded number.
    assert "PHYSICAL_GPU_COUNT=" in script
    assert 'nvidia-smi -L' in script
    # One comfyui + one worker unit per GPU, pinned via CUDA_VISIBLE_DEVICES.
    assert "comfyui-gpu${idx}.service" in script
    assert "filmforge-worker-gpu${idx}.service" in script
    assert "CUDA_VISIBLE_DEVICES=${idx}" in script
    # Distinct per-GPU worker id (not a single shared RENDER_BROKER_WORKER_ID).
    assert "WORKER_ID_FILE=/workspace/.filmforge_worker_gpu${idx}.id" in script
    # Honors WORKER_PUBLIC_URLS so every gpuN gets a routable public URL.
    assert "WORKER_PUBLIC_URLS" in script
    # The Infra deploy value is exported into this script; the template must not
    # overwrite it with the historical hardcoded single-job limit.
    assert 'WORKER_MAX_CONCURRENT_JOBS="${WORKER_MAX_CONCURRENT_JOBS:-10}"' in script
    assert "Environment=WORKER_MAX_CONCURRENT_JOBS=${WORKER_MAX_CONCURRENT_JOBS:-10}" in script
    assert "Environment=WORKER_MAX_CONCURRENT_JOBS=1" not in script


def test_legacy_single_worker_script_does_not_start_comfy() -> None:
    """Regression guard: the legacy remote_script never starts ComfyUI (it only
    waits for one) and pins a single worker — must NOT be used for multi-GPU."""
    script = deploy_gpu.remote_script("/workspace/filmforge_gpu_worker", 9000)
    assert 'cat > "/etc/systemd/system/filmforge-worker-gpu' not in script
    assert "PHYSICAL_GPU_COUNT" not in script  # not GPU-aware


def test_extract_worker_urls_deduplicates_aggregate_and_per_worker_lines() -> None:
    output = "\n".join(
        [
            "WORKER_URL=http://1.2.3.4:31000",
            "WORKER_URL=http://1.2.3.4:31001",
            "WORKER_URLS=http://1.2.3.4:31000,http://1.2.3.4:31001",
        ]
    )

    assert deploy_gpu.extract_worker_urls(output) == [
        "http://1.2.3.4:31000",
        "http://1.2.3.4:31001",
    ]
