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
    assert deploy_gpu._vast_offer_gpu_count({"gpu_name": "RTX 4090"}) == 1


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
