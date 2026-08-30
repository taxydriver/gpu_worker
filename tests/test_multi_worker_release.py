"""ADR-0009 regression: N workers behind one secure edge.

The contract stays singular — one hostname, one tunnel, one cutover proof —
and workers 1..N-1 ride as per-index staged drop-ins plus handle_path routes
in the one exact Caddyfile. worker_count=1 must remain byte-identical to the
pre-ADR profile in every artifact, or existing receipts would break.
"""
from pathlib import Path

import pytest

from gpu_worker.worker_release import (
    WorkerReleaseError,
    _indexed_local_url,
    _indexed_public_url,
    _indexed_worker_units,
    _validated_worker_count,
    expected_caddy_config,
)


EDGE = "https://gpu-worker.example.com"
LOCAL = "http://127.0.0.1:9000"


def test_single_worker_caddyfile_is_byte_identical_to_pre_adr_form():
    assert expected_caddy_config(public_url=EDGE, local_url=LOCAL) == (
        "gpu-worker.example.com {\n"
        "    reverse_proxy http://127.0.0.1:9000\n"
        "}\n"
    )


def test_multi_worker_caddyfile_routes_each_index_and_root_last():
    config = expected_caddy_config(public_url=EDGE, local_url=LOCAL, worker_count=3)
    assert config == (
        "gpu-worker.example.com {\n"
        "    handle_path /gpu1/* {\n"
        "        reverse_proxy http://127.0.0.1:9001\n"
        "    }\n"
        "    handle_path /gpu2/* {\n"
        "        reverse_proxy http://127.0.0.1:9002\n"
        "    }\n"
        "    reverse_proxy http://127.0.0.1:9000\n"
        "}\n"
    )


def test_indexed_urls_and_units():
    assert _indexed_public_url(EDGE, 2) == f"{EDGE}/gpu2"
    assert _indexed_local_url(LOCAL, 3) == "http://127.0.0.1:9003"
    assert _indexed_worker_units("filmforge-worker-gpu0.service", 3) == [
        "filmforge-worker-gpu1.service",
        "filmforge-worker-gpu2.service",
    ]
    assert _indexed_worker_units("filmforge-worker-gpu0.service", 1) == []


def test_worker_count_bounds_and_unit_shape():
    with pytest.raises(WorkerReleaseError, match="1..8"):
        _validated_worker_count(0, "filmforge-worker-gpu0.service")
    with pytest.raises(WorkerReleaseError, match="1..8"):
        _validated_worker_count(9, "filmforge-worker-gpu0.service")
    with pytest.raises(WorkerReleaseError, match="gpu0-indexed"):
        _validated_worker_count(2, "filmforge-worker.service")
    # A single worker never needs the gpu0 naming convention.
    assert _validated_worker_count(1, "filmforge-worker.service") == 1


def test_local_url_requires_explicit_port():
    with pytest.raises(WorkerReleaseError, match="explicit port"):
        _indexed_local_url("http://127.0.0.1", 1)


def test_render_validator_accepts_multi_form_only_with_count(tmp_path: Path):
    # Regression for the first live proof failure (2026-08-17): the contract-
    # validation re-render defaulted worker_count while the stage writer
    # threaded it, so a correct 2-worker Caddyfile was refused. Every caller
    # of _render_caddy_config must pass the contract's count.
    from gpu_worker.worker_release import _render_caddy_config

    config = tmp_path / "Caddyfile"
    config.write_text(
        expected_caddy_config(public_url=EDGE, local_url=LOCAL, worker_count=2)
    )
    rendered = _render_caddy_config(
        config, public_url=EDGE, local_url=LOCAL, worker_count=2
    )
    assert "handle_path /gpu1/*" in rendered
    with pytest.raises(WorkerReleaseError, match="exactly terminate"):
        _render_caddy_config(config, public_url=EDGE, local_url=LOCAL)
