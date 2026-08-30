from __future__ import annotations

import base64
import binascii
import hashlib
from pathlib import Path
import struct
import zlib

import pytest

from gpu_worker.comfy_client import (
    apply_comfy_input_files,
    observe_staged_input_receipts,
)
from gpu_worker.schemas import ComfyInputFile
from gpu_worker.schemas import RunRequest


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _rgb_png(red: int, green: int, blue: int) -> bytes:
    def _chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(bytes((0, red, green, blue))))
        + _chunk(b"IEND", b"")
    )


def test_apply_comfy_input_files_uses_subfolder_path_for_load_image(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "input"
    source = input_dir / "source" / "front.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(PNG_BYTES)

    monkeypatch.setattr("gpu_worker.comfy_client.comfy_input_dir", lambda: input_dir)
    monkeypatch.setattr(
        "gpu_worker.comfy_client.served_file_roots",
        lambda: {"input": input_dir, "output": tmp_path / "output"},
    )

    payload = {
        "40": {
            "class_type": "LoadImage",
            "inputs": {"image": "characters/front.png", "type": "input"},
        }
    }
    patched = apply_comfy_input_files(
        payload,
        [
            ComfyInputFile(
                node_id="40",
                filename="front.png",
                source_path=str(source),
                subfolder="characters",
            )
        ],
    )

    assert (input_dir / "characters" / "front.png").read_bytes() == PNG_BYTES
    assert patched["40"]["inputs"]["image"] == "characters/front.png"


def test_apply_comfy_input_files_replaces_invalid_cached_image(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "input"
    source = input_dir / "source" / "front.png"
    cached = input_dir / "characters" / "front.png"
    source.parent.mkdir(parents=True)
    cached.parent.mkdir(parents=True)
    source.write_bytes(PNG_BYTES)
    cached.write_bytes(b"not an image")

    monkeypatch.setattr("gpu_worker.comfy_client.comfy_input_dir", lambda: input_dir)
    monkeypatch.setattr(
        "gpu_worker.comfy_client.served_file_roots",
        lambda: {"input": input_dir, "output": tmp_path / "output"},
    )

    payload = {
        "40": {
            "class_type": "LoadImage",
            "inputs": {"image": "characters/front.png", "type": "input"},
        }
    }
    patched = apply_comfy_input_files(
        payload,
        [
            ComfyInputFile(
                node_id="40",
                filename="front.png",
                source_path=str(source),
                subfolder="characters",
            )
        ],
    )

    assert cached.read_bytes() == PNG_BYTES
    assert patched["40"]["inputs"]["image"] == "characters/front.png"


def test_protected_inputs_are_content_addressed_across_sequential_jobs(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr("gpu_worker.comfy_client.comfy_input_dir", lambda: input_dir)

    payload = {"40": {"class_type": "LoadImage", "inputs": {"image": "old.png"}}}
    image_a = _rgb_png(255, 0, 0)
    image_b = _rgb_png(0, 0, 255)
    digest_a = hashlib.sha256(image_a).hexdigest()
    digest_b = hashlib.sha256(image_b).hexdigest()

    patched_a = apply_comfy_input_files(
        payload,
        [
            ComfyInputFile(
                node_id="40",
                filename="candidate_reference_1.png",
                source_data=base64.b64encode(image_a).decode("ascii"),
                expected_sha256=digest_a,
            )
        ],
    )
    patched_b = apply_comfy_input_files(
        payload,
        [
            ComfyInputFile(
                node_id="40",
                filename="candidate_reference_1.png",
                source_data=base64.b64encode(image_b).decode("ascii"),
                expected_sha256=digest_b,
            )
        ],
    )

    staged_a = patched_a["40"]["inputs"]["image"]
    staged_b = patched_b["40"]["inputs"]["image"]
    assert staged_a == f"sha256_{digest_a}.png"
    assert staged_b == f"sha256_{digest_b}.png"
    assert staged_a != staged_b
    assert (input_dir / staged_a).read_bytes() == image_a
    assert (input_dir / staged_b).read_bytes() == image_b
    assert observe_staged_input_receipts(
        patched_b,
        [
            ComfyInputFile(
                node_id="40",
                filename="candidate_reference_1.png",
                expected_sha256=digest_b,
            )
        ],
    ) == [
        {"node_id": "40", "input_name": "image", "content_sha256": digest_b}
    ]


def test_protected_cached_input_is_replaced_then_rehashed_before_graph(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr("gpu_worker.comfy_client.comfy_input_dir", lambda: input_dir)
    payload = {"40": {"class_type": "LoadImage", "inputs": {"image": "old.png"}}}
    expected_bytes = _rgb_png(0, 255, 0)
    stale_bytes = _rgb_png(255, 0, 0)
    expected = hashlib.sha256(expected_bytes).hexdigest()
    destination = input_dir / f"sha256_{expected}.png"
    destination.write_bytes(stale_bytes)

    spec = ComfyInputFile(
        node_id="40",
        filename="candidate_reference_1.png",
        source_data=base64.b64encode(expected_bytes).decode("ascii"),
        expected_sha256=expected,
    )
    patched = apply_comfy_input_files(payload, [spec])
    assert destination.read_bytes() == expected_bytes

    # A mutation after staging is caught by the immediate pre-submit observer.
    destination.write_bytes(stale_bytes)
    with pytest.raises(RuntimeError, match="Observed staged input"):
        observe_staged_input_receipts(patched, [spec])


def test_infinitetalk_observes_approved_mpeg_before_job_scoped_wav_normalization(
    monkeypatch, tmp_path: Path
):
    from gpu_worker import app as worker_app

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr("gpu_worker.comfy_client.comfy_input_dir", lambda: input_dir)
    monkeypatch.setattr(
        worker_app,
        "get_settings",
        lambda: type("Settings", (), {"comfy_input_dir": str(input_dir)})(),
    )
    audio = b"ID3" + b"approved-mpeg" * 8
    digest = hashlib.sha256(audio).hexdigest()
    normalized = input_dir / "infinitetalk" / "g90-contract" / f"{digest}.wav"
    monkeypatch.setattr(
        worker_app,
        "normalized_audio_path_for_source",
        lambda source_digest, *, job_id: normalized,
    )

    def fake_normalize(source: Path, *, job_id: str) -> Path:
        assert source.read_bytes() == audio
        assert job_id == "g90-contract"
        normalized.parent.mkdir(parents=True)
        normalized.write_bytes(b"RIFF" + b"normalized-wav")
        return normalized

    monkeypatch.setattr(worker_app, "normalize_approved_mpeg_to_wav", fake_normalize)
    real_observe = worker_app.observe_staged_input_receipts
    observed = []

    def observe(payload, specs):
        receipts = real_observe(payload, specs)
        observed.append(receipts)
        return receipts

    monkeypatch.setattr(worker_app, "observe_staged_input_receipts", observe)
    request = RunRequest(
        job_id="g90-contract",
        asset_group="infinitetalk_v1",
        comfy_payload={"10": {"class_type": "LoadAudio", "inputs": {"audio": "old.mp3"}}},
        comfy_input_files=[
            ComfyInputFile(
                node_id="10",
                input_name="audio",
                filename="approved_take.mp3",
                source_data=base64.b64encode(audio).decode("ascii"),
                expected_sha256=digest,
                content_type="audio/mpeg",
            )
        ],
    )

    prepared, observer_specs = worker_app._prepare_comfy_inputs(request)

    assert observed[0] == [
        {"node_id": "10", "input_name": "audio", "content_sha256": digest}
    ]
    assert prepared["10"]["inputs"]["audio"] == str(normalized.relative_to(input_dir))
    assert observer_specs[0].content_type == "audio/wav"
    assert observer_specs[0].expected_sha256 == hashlib.sha256(
        normalized.read_bytes()
    ).hexdigest()
