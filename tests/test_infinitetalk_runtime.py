from __future__ import annotations

import base64
import binascii
import hashlib
from pathlib import Path
import struct
from types import SimpleNamespace
import wave
import zlib

import pytest
from pydantic import ValidationError

from gpu_worker import app
from gpu_worker import asset_manager
from gpu_worker import comfy_client
from gpu_worker import infinitetalk as runtime
from gpu_worker.schemas import ComfyInputFile, InfiniteTalkTwoPersonRouting, RunRequest


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _rgb_png(red: int, green: int, blue: int) -> bytes:
    """Build a CRC-valid one-pixel RGB PNG for container-preserving tamper tests."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes((0, red, green, blue))))
        + chunk(b"IEND", b"")
    )


def _gray_png(width: int, height: int, value: int = 127) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\0" + bytes([value]) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _decode_gray_png(data: bytes) -> tuple[int, int, bytes]:
    width, height = struct.unpack(">II", data[16:24])
    offset = 8
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        if kind == b"IDAT":
            compressed.extend(payload)
        offset += 12 + length
    rows = zlib.decompress(bytes(compressed))
    pixels = bytearray()
    stride = width + 1
    for row in range(height):
        assert rows[row * stride] == 0
        pixels.extend(rows[row * stride + 1:(row + 1) * stride])
    return width, height, bytes(pixels)


def test_readiness_requires_every_a1_node_and_wav2vec_dependency(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_required_files", lambda: ())
    python = tmp_path / "python"
    python.touch()
    monkeypatch.setattr(runtime, "_comfy_python", lambda: python)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(
        runtime.requests,
        "get",
        lambda *a, **k: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"MultiTalkModelLoader": {}},
        ),
    )

    result = runtime.check_infinitetalk_readiness()

    assert result.ready is False
    assert "MultiTalkWav2VecEmbeds" in result.missing_node_classes
    assert "DownloadAndLoadWav2VecModel" in result.missing_node_classes


def test_unready_infinitetalk_is_withheld_from_advertisement(monkeypatch):
    monkeypatch.setattr(
        app,
        "get_settings",
        lambda: SimpleNamespace(resolved_capabilities=lambda: ["wan_i2v", "infinitetalk"]),
    )
    monkeypatch.setattr(
        app,
        "check_infinitetalk_readiness",
        lambda **_kwargs: runtime.InfiniteTalkReadiness(ready=False, missing_files=("missing",)),
    )

    capabilities, readiness = app._advertised_capabilities()

    assert capabilities == ["wan_i2v_v1"]
    assert readiness == {
        "ready": False,
        "missing_files": ["missing"],
        "missing_node_classes": [],
        "wav2vec_dependency_error": None,
        "multitalk_contract_error": None,
        "multi_checkpoint_error": None,
        "comfy_error": None,
    }


def test_a2_capability_is_withheld_without_strict_readiness_but_a1_remains(monkeypatch):
    monkeypatch.setattr(
        app,
        "get_settings",
        lambda: SimpleNamespace(
            resolved_capabilities=lambda: [
                "infinitetalk_v1",
                "infinitetalk_two_person_v1",
            ]
        ),
    )
    checks = []

    def readiness(*, require_two_person=False):
        checks.append(require_two_person)
        return runtime.InfiniteTalkReadiness(
            ready=not require_two_person,
            multitalk_contract_error=("missing explicit-mask schema" if require_two_person else None),
        )

    monkeypatch.setattr(app, "check_infinitetalk_readiness", readiness)

    capabilities, details = app._advertised_capabilities()

    assert checks == [False, True]
    assert capabilities == ["infinitetalk_v1"]
    assert details["ready"] is False
    assert details["multitalk_contract_error"] == "missing explicit-mask schema"


def test_a2_object_info_requires_audio2_masks_and_para_schema():
    valid = {
        "MultiTalkWav2VecEmbeds": {
            "input": {
                "required": {"multi_audio_type": [["para", "add"], {}]},
                "optional": {"audio_2": ["AUDIO"], "ref_target_masks": ["MASK"]},
            }
        }
    }
    assert runtime._multitalk_object_contract_error(valid) is None

    no_masks = {
        "MultiTalkWav2VecEmbeds": {
            "input": {
                "required": {"multi_audio_type": [["para", "add"], {}]},
                "optional": {"audio_2": ["AUDIO"]},
            }
        }
    }
    assert "ref_target_masks" in runtime._multitalk_object_contract_error(no_masks)

    no_para = {
        "MultiTalkWav2VecEmbeds": {
            "input": {
                "required": {"multi_audio_type": [["add"], {}]},
                "optional": {"audio_2": ["AUDIO"], "ref_target_masks": ["MASK"]},
            }
        }
    }
    assert "para" in runtime._multitalk_object_contract_error(no_para)


def test_a2_checkpoint_readiness_requires_exact_ensure_digest_and_detects_replacement(monkeypatch, tmp_path):
    header = b'{"tensor":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}'
    checkpoint = tmp_path / "multi.safetensors"
    checkpoint.write_bytes(struct.pack("<Q", len(header)) + header + b"ABCD")
    approved = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(runtime, "_MULTITALK_MODEL_PATH", checkpoint)
    monkeypatch.setattr(runtime, "_MULTITALK_MODEL_BYTES", checkpoint.stat().st_size)
    monkeypatch.setattr(runtime, "_MULTITALK_MODEL_SHA256", approved)
    asset_manager._VERIFIED_CHECKSUM_FACTS.clear()

    assert "has not passed exact digest verification" in runtime._multi_checkpoint_error()
    asset_manager.verify_asset_checksum(checkpoint, approved)
    assert runtime._multi_checkpoint_error() is None

    checkpoint.write_bytes(struct.pack("<Q", len(header)) + header + b"WXYZ")
    assert "has not passed exact digest verification" in runtime._multi_checkpoint_error()


def test_warm_a2_ensure_hashes_once_then_enables_strict_readiness(monkeypatch, tmp_path):
    header = b'{"tensor":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}'
    checkpoint = tmp_path / "multi.safetensors"
    checkpoint.write_bytes(struct.pack("<Q", len(header)) + header + b"ABCD")
    approved = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(runtime, "_MULTITALK_MODEL_PATH", checkpoint)
    monkeypatch.setattr(runtime, "_MULTITALK_MODEL_BYTES", checkpoint.stat().st_size)
    monkeypatch.setattr(runtime, "_MULTITALK_MODEL_SHA256", approved)
    monkeypatch.setattr(
        asset_manager,
        "get_asset_group",
        lambda _group: [{
            "name": "infinitetalk_multi",
            "path": str(checkpoint),
            "url": "https://example.invalid/multi.safetensors",
            "sha256": approved,
        }],
    )
    monkeypatch.setattr(runtime, "_required_files", lambda: ())
    python = tmp_path / "python"
    python.touch()
    monkeypatch.setattr(runtime, "_comfy_python", lambda: python)
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    object_info = {name: {} for name in runtime.A2_REQUIRED_NODE_CLASSES}
    object_info["MultiTalkWav2VecEmbeds"] = {
        "input": {
            "required": {"multi_audio_type": [["para", "add"], {}]},
            "optional": {"audio_2": ["AUDIO"], "ref_target_masks": ["MASK"]},
        }
    }
    monkeypatch.setattr(
        runtime.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: object_info,
        ),
    )
    monkeypatch.setattr(runtime, "_wrapper_patch_error", lambda: None)
    asset_manager._VERIFIED_CHECKSUM_FACTS.clear()
    original_verify = asset_manager._verify_sha256
    verify_calls = []

    def verify(path, expected):
        verify_calls.append((path, expected))
        original_verify(path, expected)

    monkeypatch.setattr(asset_manager, "_verify_sha256", verify)

    first = asset_manager.ensure_asset_group("infinitetalk_two_person_v1")
    assert first.downloaded_assets == []
    assert len(verify_calls) == 1
    assert runtime.check_infinitetalk_readiness(require_two_person=True).ready is True

    second = asset_manager.ensure_asset_group("infinitetalk_two_person_v1")
    assert second.downloaded_assets == []
    assert len(verify_calls) == 1

    checkpoint.write_bytes(struct.pack("<Q", len(header)) + header + b"WXYZ")
    assert "has not passed exact digest verification" in runtime._multi_checkpoint_error()
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        asset_manager.ensure_asset_group("infinitetalk_two_person_v1")
    assert len(verify_calls) == 2
    assert not checkpoint.exists()


@pytest.mark.parametrize("matching", [True, False])
def test_concurrent_pinned_asset_completion_is_verified_inside_lock(monkeypatch, tmp_path, matching):
    target = tmp_path / "multi.safetensors"
    approved_bytes = b"approved-multi-checkpoint"
    expected_sha256 = hashlib.sha256(approved_bytes).hexdigest()
    completed_bytes = approved_bytes if matching else b"corrupt-multi-checkpoint"

    class CompletingLock:
        def __enter__(self):
            target.write_bytes(completed_bytes)
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(asset_manager.portalocker, "Lock", lambda *args, **kwargs: CompletingLock())
    monkeypatch.setattr(
        asset_manager,
        "_download_to_path",
        lambda *args, **kwargs: pytest.fail("concurrent completion must not redownload"),
    )
    asset_manager._VERIFIED_CHECKSUM_FACTS.clear()
    asset = {
        "name": "infinitetalk_multi",
        "path": str(target),
        "url": "https://example.invalid/multi.safetensors",
        "sha256": expected_sha256,
    }

    if matching:
        assert asset_manager._ensure_single_asset(asset) is None
        assert asset_manager.asset_checksum_is_verified(target, expected_sha256) is True
        assert target.read_bytes() == approved_bytes
    else:
        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            asset_manager._ensure_single_asset(asset)
        assert not target.exists()


def test_activated_worker_preload_rebuilds_a2_checksum_before_heartbeat(monkeypatch, tmp_path):
    header = b'{"tensor":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}'
    checkpoint = tmp_path / "multi.safetensors"
    checkpoint.write_bytes(struct.pack("<Q", len(header)) + header + b"ABCD")
    approved = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(runtime, "_MULTITALK_MODEL_PATH", checkpoint)
    monkeypatch.setattr(runtime, "_MULTITALK_MODEL_BYTES", checkpoint.stat().st_size)
    monkeypatch.setattr(runtime, "_MULTITALK_MODEL_SHA256", approved)
    monkeypatch.setattr(
        asset_manager,
        "get_asset_group",
        lambda _group: [{
            "name": "infinitetalk_multi",
            "path": str(checkpoint),
            "url": "https://example.invalid/multi.safetensors",
            "sha256": approved,
        }],
    )
    settings = SimpleNamespace(
        worker_vram_gb=80.0,
        worker_provider="verda",
        worker_instance_id="instance-a2",
        worker_gpu_name="H100 80GB",
        worker_vision_base_url=None,
        comfy_base_url="http://127.0.0.1:8188",
        resolved_capabilities=lambda: ["infinitetalk_two_person_v1"],
        resolved_worker_public_url=lambda: "https://worker.example",
        resolved_backend_url=lambda: "https://backend.example",
        resolved_worker_id=lambda: "00000000-0000-0000-0000-000000000001",
        resolved_worker_name=lambda: "a2-worker",
        resolved_max_concurrent_jobs=lambda: 1,
        resolved_input_url_allowed_hosts=lambda: {"storage.example"},
    )
    monkeypatch.setattr(app, "get_settings", lambda: settings)
    monkeypatch.setattr(app, "_worker_api_auth_ready", lambda: True)
    monkeypatch.setattr(app, "_broker_headers", lambda: {})
    monkeypatch.setattr(app, "_free_vram_mib", lambda: 80 * 1024)
    monkeypatch.setattr(app, "comfy_queue_depth", lambda: None)
    monkeypatch.setattr(app, "is_comfy_healthy", lambda: True)
    monkeypatch.setattr(app, "_performance_snapshot", lambda: {})
    monkeypatch.setattr(app, "_preload_asset_groups", lambda: ["infinitetalk_two_person_v1"])
    monkeypatch.setattr(app, "_ensure_runtime_provisioned", lambda _group: False)
    monkeypatch.setattr(app, "_WARMED_GROUPS", set())

    def readiness(*, require_two_person=False):
        assert require_two_person is True
        error = runtime._multi_checkpoint_error()
        return runtime.InfiniteTalkReadiness(
            ready=error is None,
            multi_checkpoint_error=error,
        )

    monkeypatch.setattr(app, "check_infinitetalk_readiness", readiness)
    heartbeats = []
    monkeypatch.setattr(
        app.requests,
        "post",
        lambda _url, *, json, **_kwargs: (
            heartbeats.append(json.copy())
            or SimpleNamespace(raise_for_status=lambda: None)
        ),
    )
    asset_manager._VERIFIED_CHECKSUM_FACTS.clear()

    app._send_heartbeat_now()
    assert heartbeats[-1]["capabilities"] == []
    assert heartbeats[-1]["metadata"]["infinitetalk_readiness"]["ready"] is False

    app._preflight_download_all()
    assert asset_manager.asset_checksum_is_verified(checkpoint, approved) is True

    app._send_heartbeat_now()
    assert heartbeats[-1]["capabilities"] == ["infinitetalk_two_person_v1"]
    assert heartbeats[-1]["metadata"]["infinitetalk_readiness"]["ready"] is True


def test_a2_wrapper_readiness_requires_pinned_commit_and_positive_patch(monkeypatch, tmp_path):
    wrapper = tmp_path / "custom_nodes" / "ComfyUI-WanVideoWrapper"
    wrapper.mkdir(parents=True)
    sampler = wrapper / "nodes_sampler.py"
    sampler.write_text(
        '"ref_target_masks": (multitalk_embeds or {}).get("ref_target_masks", ref_target_masks) '
        "if multitalk_audio_embeds is not None else None,\n"
    )
    monkeypatch.setenv("COMFY_DIR", str(tmp_path))
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=runtime._WAN_WRAPPER_COMMIT + "\n",
            stderr="",
        ),
    )
    assert runtime._wrapper_patch_error() is None

    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="0" * 40 + "\n", stderr=""),
    )
    assert "commit does not match" in runtime._wrapper_patch_error()

    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=runtime._WAN_WRAPPER_COMMIT + "\n",
            stderr="",
        ),
    )
    sampler.write_text(
        '"ref_target_masks": ref_target_masks if multitalk_audio_embeds is not None else None,\n'
    )
    assert "not positively verified" in runtime._wrapper_patch_error()


def _write_pcm_wav(path: Path, *, seconds: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * int(16_000 * seconds))


def _infinitetalk_request() -> tuple[RunRequest, bytes, bytes]:
    image = PNG_BYTES
    # Match the failed proof's non-empty source size. The conversion is faked
    # below because staging/receipt provenance, not ffmpeg, is under test.
    mpeg = b"M" * 49_781
    image_digest = hashlib.sha256(image).hexdigest()
    mpeg_digest = hashlib.sha256(mpeg).hexdigest()
    return (
        RunRequest(
            job_id="36c870a5-db87-5771-b2de-3464aa552d54",
            asset_group="infinitetalk_v1",
            comfy_payload={
                "6": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
                "10": {"class_type": "LoadAudio", "inputs": {"audio": "old.mp3"}},
            },
            comfy_input_files=[
                ComfyInputFile(
                    node_id="6",
                    input_name="image",
                    filename="approved.png",
                    source_data=base64.b64encode(image).decode("ascii"),
                    expected_sha256=image_digest,
                    content_type="image/png",
                ),
                ComfyInputFile(
                    node_id="10",
                    input_name="audio",
                    filename="approved.mp3",
                    source_data=base64.b64encode(mpeg).decode("ascii"),
                    expected_sha256=mpeg_digest,
                    content_type="audio/mpeg",
                ),
            ],
        ),
        image,
        mpeg,
    )


def _configure_input_root(monkeypatch, input_root: Path) -> None:
    settings = SimpleNamespace(comfy_input_dir=str(input_root))
    monkeypatch.setattr(comfy_client, "comfy_input_dir", lambda: input_root)
    monkeypatch.setattr(app, "get_settings", lambda: settings)
    monkeypatch.setattr(runtime, "get_settings", lambda: settings)


def _fake_normalizer(source: Path, *, job_id: str) -> Path:
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = runtime.normalized_audio_path_for_source(source_digest, job_id=job_id)
    # 48,640 mono s16 frames plus the 44-byte header reproduces the observed
    # 97,324-byte normalized WAV while remaining a valid 3.04 second input.
    _write_pcm_wav(destination, seconds=3.04)
    return destination


def _a2_graph(*, speaker_slot: int, num_frames: int = 53) -> dict:
    listener_slot = 1 if speaker_slot == 2 else 2
    audio_inputs = {
        f"audio_{speaker_slot}": ["10", 0],
        f"audio_{listener_slot}": ["16", 0],
    }
    return {
        "2": {"class_type": "MultiTalkModelLoader", "inputs": {"model": "Wan2_1-InfiniteTalk-Multi_fp16.safetensors"}},
        "6": {"class_type": "LoadImage", "inputs": {"image": "approved.png"}},
        "10": {"class_type": "LoadAudio", "inputs": {"audio": "approved.mp3"}},
        "11": {"class_type": "MultiTalkWav2VecEmbeds", "inputs": {
            **audio_inputs,
            "multi_audio_type": "para",
            "normalize_loudness": False,
            "num_frames": num_frames,
            "fps": 25.0,
            "ref_target_masks": ["93", 0],
        }},
        "12": {"class_type": "WanVideoImageToVideoMultiTalk", "inputs": {
            "width": 832,
            "height": 480,
            "mode": "multitalk",
            "start_image": ["6", 0],
        }},
        "15": {"class_type": "VHS_VideoCombine", "inputs": {
            "audio": ["10", 0],
            "frame_rate": 25.0,
        }},
        "16": {"class_type": "LoadAudio", "inputs": {"audio": "listener.wav"}},
        "90": {"class_type": "LoadImage", "inputs": {"image": "slot1.png"}},
        "91": {"class_type": "LoadImage", "inputs": {"image": "slot2.png"}},
        "92": {"class_type": "ImageBatch", "inputs": {"image1": ["90", 0], "image2": ["91", 0]}},
        "93": {"class_type": "ImageToMask", "inputs": {"image": ["95", 0], "channel": "red"}},
        "94": {"class_type": "LoadImage", "inputs": {"image": "background.png"}},
        "95": {"class_type": "ImageBatch", "inputs": {"image1": ["92", 0], "image2": ["94", 0]}},
    }


def _a2_request(*, speaker_slot: int, job_id: str = "scene23-b0") -> RunRequest:
    still = _gray_png(1024, 576)
    mpeg = b"approved-mpeg" * 100
    still_digest = hashlib.sha256(still).hexdigest()
    mpeg_digest = hashlib.sha256(mpeg).hexdigest()
    slot_regions = (
        (0.08, 0.18, 0.42, 0.92),
        (0.58, 0.16, 0.92, 0.94),
    )
    listener_slot = 1 if speaker_slot == 2 else 2
    routing = InfiniteTalkTwoPersonRouting(
        schema_version="infinitetalk_two_person_routing_v1",
        mode="two_person_parallel",
        multi_audio_type="para",
        speaker_slot=speaker_slot,
        listener_slot=listener_slot,
        slot_regions=slot_regions,
        speaker_region=slot_regions[speaker_slot - 1],
        listener_region=slot_regions[listener_slot - 1],
        coordinate_space="normalized_0_1",
        source_still_sha256=still_digest,
        source_dimensions={"width": 1024, "height": 576},
        spatial_authority_sha256=hashlib.sha256(f"authority-{job_id}".encode()).hexdigest(),
        expected_duration_sec=2.0,
        listener_audio_kind="silence_pcm",
    )
    return RunRequest(
        job_id=job_id,
        asset_group="infinitetalk_two_person_v1",
        comfy_payload=_a2_graph(speaker_slot=speaker_slot),
        comfy_input_files=[
            ComfyInputFile(
                node_id="6",
                input_name="image",
                filename="approved.png",
                source_data=base64.b64encode(still).decode("ascii"),
                expected_sha256=still_digest,
                content_type="image/png",
            ),
            ComfyInputFile(
                node_id="10",
                input_name="audio",
                filename="approved.mp3",
                source_data=base64.b64encode(mpeg).decode("ascii"),
                expected_sha256=mpeg_digest,
                content_type="audio/mpeg",
            ),
        ],
        infinitetalk_routing=routing,
    )


def _configure_a2_runtime(monkeypatch, input_root: Path) -> None:
    _configure_input_root(monkeypatch, input_root)

    def normalize(source: Path, *, job_id: str) -> Path:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = runtime.normalized_audio_path_for_source(digest, job_id=job_id)
        _write_pcm_wav(destination, seconds=2.0)
        return destination

    monkeypatch.setattr(app, "normalize_approved_mpeg_to_wav", normalize)
    monkeypatch.setattr(runtime, "_multi_checkpoint_error", lambda: None)
    monkeypatch.setattr(runtime, "_wrapper_patch_error", lambda: None)


def test_a2_materializes_slot_order_background_and_exact_silence(monkeypatch, tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    _configure_a2_runtime(monkeypatch, input_root)
    request = _a2_request(speaker_slot=2)

    payload, observer_specs = app._prepare_comfy_inputs(request)
    receipts = comfy_client.observe_staged_input_receipts(payload, observer_specs)
    receipt_by_node = {receipt["node_id"]: receipt for receipt in receipts}

    assert set(receipt_by_node) == {"6", "10", "16", "90", "91", "94"}
    assert payload["11"]["inputs"]["audio_1"] == ["16", 0]
    assert payload["11"]["inputs"]["audio_2"] == ["10", 0]
    assert payload["11"]["inputs"]["multi_audio_type"] == "para"
    assert payload["11"]["inputs"]["ref_target_masks"] == ["93", 0]
    assert payload["15"]["inputs"]["audio"] == ["10", 0]

    listener_path = input_root / payload["16"]["inputs"]["audio"]
    with wave.open(str(listener_path), "rb") as listener:
        assert (listener.getnchannels(), listener.getframerate(), listener.getsampwidth()) == (1, 16_000, 2)
        assert listener.getnframes() == 32_000
        assert listener.readframes(listener.getnframes()) == b"\0\0" * 32_000
    speaker_path = input_root / payload["10"]["inputs"]["audio"]
    with wave.open(str(speaker_path), "rb") as speaker:
        assert speaker.getnframes() == 32_000

    decoded = {}
    for node_id in ("90", "91", "94"):
        decoded[node_id] = _decode_gray_png(
            (input_root / payload[node_id]["inputs"]["image"]).read_bytes()
        )
    assert all((width, height) == (832, 480) for width, height, _ in decoded.values())
    slot_1, slot_2, background = (decoded[node][2] for node in ("90", "91", "94"))
    assert any(slot_1) and any(slot_2) and any(background)
    for left, right, neutral in zip(slot_1, slot_2, background):
        assert not (left and right)
        assert neutral == (0 if left or right else 255)

    routing_receipt = runtime.build_two_person_routing_receipt(
        request.infinitetalk_routing,
        receipts,
    )
    assert routing_receipt["speaker_slot"] == 2
    assert routing_receipt["listener_slot"] == 1
    assert routing_receipt["mask_sha256"] == {
        "slot_1": receipt_by_node["90"]["content_sha256"],
        "slot_2": receipt_by_node["91"]["content_sha256"],
        "background": receipt_by_node["94"]["content_sha256"],
    }
    generated_specs = {spec.node_id: spec for spec in observer_specs if spec.node_id in {"16", "90", "91", "94"}}
    assert all(
        spec.source_data is None and spec.source_path is None and spec.source_url is None
        for spec in generated_specs.values()
    )


def test_scene23_speaker_swap_changes_audio_slot_not_authored_mask_order(monkeypatch, tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    _configure_a2_runtime(monkeypatch, input_root)
    b0 = _a2_request(speaker_slot=2, job_id="scene23-b0")
    b1 = _a2_request(speaker_slot=1, job_id="scene23-b1")

    payload_b0, specs_b0 = app._prepare_comfy_inputs(b0)
    payload_b1, specs_b1 = app._prepare_comfy_inputs(b1)
    receipts_b0 = comfy_client.observe_staged_input_receipts(payload_b0, specs_b0)
    receipts_b1 = comfy_client.observe_staged_input_receipts(payload_b1, specs_b1)

    assert payload_b0["11"]["inputs"]["audio_2"] == ["10", 0]
    assert payload_b1["11"]["inputs"]["audio_1"] == ["10", 0]
    hashes_b0 = {receipt["node_id"]: receipt["content_sha256"] for receipt in receipts_b0}
    hashes_b1 = {receipt["node_id"]: receipt["content_sha256"] for receipt in receipts_b1}
    assert hashes_b0["90"] == hashes_b1["90"]
    assert hashes_b0["91"] == hashes_b1["91"]
    assert hashes_b0["94"] == hashes_b1["94"]


def test_a2_effective_mask_and_silence_tampering_fail_before_submit(monkeypatch, tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    _configure_a2_runtime(monkeypatch, input_root)
    request = _a2_request(speaker_slot=1)
    payload, specs = app._prepare_comfy_inputs(request)

    slot_1_path = input_root / payload["90"]["inputs"]["image"]
    slot_1_path.write_bytes(_gray_png(832, 480, value=255))
    with pytest.raises(RuntimeError, match="Observed staged input digest mismatch"):
        comfy_client.observe_staged_input_receipts(payload, specs)

    payload, specs = app._prepare_comfy_inputs(request)
    listener_path = input_root / payload["16"]["inputs"]["audio"]
    _write_pcm_wav(listener_path, seconds=1.0)
    with pytest.raises(RuntimeError, match="Observed staged input digest mismatch"):
        comfy_client.observe_staged_input_receipts(payload, specs)


def test_a2_retry_reobserves_all_effective_graph_inputs(monkeypatch):
    request = _a2_request(speaker_slot=2, job_id="retry-a2")
    sentinel_specs = [
        ComfyInputFile(
            node_id=node_id,
            input_name="audio" if node_id in {"10", "16"} else "image",
            filename=f"{node_id}.bin",
            expected_sha256="a" * 64,
        )
        for node_id in ("6", "10", "16", "90", "91", "94")
    ]
    observations = []
    monkeypatch.setattr(app, "_asset_group_allowed", lambda _group: True)
    monkeypatch.setattr(app, "_effective_vram_floor", lambda _group: None)
    monkeypatch.setattr(app, "_free_vram_on_group_switch", lambda _group: None)
    monkeypatch.setattr(
        app,
        "ensure_asset_group",
        lambda _group: SimpleNamespace(downloaded_assets=[], asset_check_sec=0.0, download_sec=0.0),
    )
    monkeypatch.setattr(app, "_ensure_runtime_provisioned", lambda _group: False)
    monkeypatch.setattr(app, "_prepare_comfy_inputs", lambda _request: (request.comfy_payload, sentinel_specs))

    def observe(_payload, specs):
        observations.append([spec.node_id for spec in specs])
        if len(observations) == 2:
            raise RuntimeError("Observed staged input digest mismatch")
        return []

    monkeypatch.setattr(app, "observe_staged_input_receipts", observe)
    monkeypatch.setattr(app, "submit_prompt", lambda _payload: (_ for _ in ()).throw(RuntimeError("Comfy stopped")))
    monkeypatch.setattr(app, "is_comfy_healthy", lambda: False)
    monkeypatch.setattr(app, "restart_comfy", lambda: 0.0)
    monkeypatch.setattr(app, "_send_heartbeat_now", lambda: None)
    monkeypatch.setattr(
        app,
        "get_settings",
        lambda: SimpleNamespace(comfy_start_cmd="restart", comfy_base_url="http://127.0.0.1:8188"),
    )

    result = app._execute_run(request)

    assert observations == [
        ["6", "10", "16", "90", "91", "94"],
        ["6", "10", "16", "90", "91", "94"],
    ]
    assert result.ok is False
    assert result.restart_performed is True
    assert result.error == "Observed staged input digest mismatch"


def test_success_response_exposes_top_level_a2_routing_receipt(monkeypatch):
    request = _a2_request(speaker_slot=2, job_id="receipt-a2")
    mask_hashes = {"90": "1" * 64, "91": "2" * 64, "94": "3" * 64}
    receipts = [
        {"node_id": "6", "input_name": "image", "content_sha256": "6" * 64},
        {"node_id": "10", "input_name": "audio", "content_sha256": "a" * 64},
        {"node_id": "16", "input_name": "audio", "content_sha256": "b" * 64},
        *[
            {"node_id": node_id, "input_name": "image", "content_sha256": digest}
            for node_id, digest in mask_hashes.items()
        ],
    ]
    monkeypatch.setattr(app, "_asset_group_allowed", lambda _group: True)
    monkeypatch.setattr(app, "_effective_vram_floor", lambda _group: None)
    monkeypatch.setattr(app, "_free_vram_on_group_switch", lambda _group: None)
    monkeypatch.setattr(
        app,
        "ensure_asset_group",
        lambda _group: SimpleNamespace(downloaded_assets=[], asset_check_sec=0.0, download_sec=0.0),
    )
    monkeypatch.setattr(app, "_ensure_runtime_provisioned", lambda _group: False)
    monkeypatch.setattr(app, "_prepare_comfy_inputs", lambda _request: (request.comfy_payload, []))
    monkeypatch.setattr(app, "observe_staged_input_receipts", lambda *_args: receipts)
    monkeypatch.setattr(app, "submit_prompt", lambda _payload: "prompt-a2")
    monkeypatch.setattr(app, "poll_for_completion", lambda **_kwargs: {"outputs": {}})
    monkeypatch.setattr(app, "collect_output_paths", lambda _history: [])
    monkeypatch.setattr(app, "build_output_files", lambda _outputs: [])
    monkeypatch.setattr(app, "_send_heartbeat_now", lambda: None)
    monkeypatch.setattr(
        app,
        "get_settings",
        lambda: SimpleNamespace(comfy_start_cmd="", comfy_base_url="http://127.0.0.1:8188"),
    )

    result = app._execute_run(request)

    assert result.ok is True
    assert result.infinitetalk_routing_receipt is not None
    assert result.infinitetalk_routing_receipt.model_dump() == {
        "schema_version": "infinitetalk_two_person_routing_receipt_v1",
        "spatial_authority_sha256": request.infinitetalk_routing.spatial_authority_sha256,
        "source_still_sha256": request.infinitetalk_routing.source_still_sha256,
        "speaker_slot": 2,
        "listener_slot": 1,
        "mode": "two_person_parallel",
        "multi_audio_type": "para",
        "mask_sha256": {
            "slot_1": "1" * 64,
            "slot_2": "2" * 64,
            "background": "3" * 64,
        },
    }


def test_a2_rejects_missing_overlapping_or_invalid_masks(monkeypatch, tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    _configure_a2_runtime(monkeypatch, input_root)

    missing = _a2_request(speaker_slot=1)
    del missing.comfy_payload["94"]
    with pytest.raises(ValueError, match="node 94"):
        app._prepare_comfy_inputs(missing)

    with pytest.raises(ValidationError, match="must not overlap"):
        InfiniteTalkTwoPersonRouting(
            schema_version="infinitetalk_two_person_routing_v1",
            mode="two_person_parallel",
            multi_audio_type="para",
            speaker_slot=1,
            listener_slot=2,
            slot_regions=((0.1, 0.1, 0.7, 0.8), (0.6, 0.1, 0.9, 0.8)),
            speaker_region=(0.1, 0.1, 0.7, 0.8),
            listener_region=(0.6, 0.1, 0.9, 0.8),
            coordinate_space="normalized_0_1",
            source_still_sha256="a" * 64,
            source_dimensions={"width": 1024, "height": 576},
            spatial_authority_sha256="b" * 64,
            expected_duration_sec=2.0,
            listener_audio_kind="silence_pcm",
        )

    no_background = _a2_request(speaker_slot=1, job_id="no-background")
    no_background.infinitetalk_routing.slot_regions = (
        (0.0, 0.0, 0.5, 1.0),
        (0.5, 0.0, 1.0, 1.0),
    )
    no_background.infinitetalk_routing.speaker_region = no_background.infinitetalk_routing.slot_regions[0]
    no_background.infinitetalk_routing.listener_region = no_background.infinitetalk_routing.slot_regions[1]
    with pytest.raises(ValueError, match="neutral background must each be nonempty"):
        app._prepare_comfy_inputs(no_background)


def test_a2_rejects_add_mode_wrong_duration_or_sum_frame_guard(monkeypatch, tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    _configure_a2_runtime(monkeypatch, input_root)

    add_mode = _a2_request(speaker_slot=1, job_id="add-mode")
    add_mode.comfy_payload["11"]["inputs"]["multi_audio_type"] = "add"
    with pytest.raises(ValueError, match="parallel audio routing"):
        app._prepare_comfy_inputs(add_mode)

    wrong_duration = _a2_request(speaker_slot=1, job_id="wrong-duration")
    wrong_duration.infinitetalk_routing.expected_duration_sec = 2.2
    with pytest.raises(ValueError, match="duration does not match"):
        app._prepare_comfy_inputs(wrong_duration)

    summed_guard = _a2_request(speaker_slot=1, job_id="summed-guard")
    summed_guard.comfy_payload["11"]["inputs"]["num_frames"] = 101
    with pytest.raises(ValueError, match="frame guard"):
        app._prepare_comfy_inputs(summed_guard)


def test_a2_requires_distinct_capability_and_frozen_routing(monkeypatch, tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    _configure_a2_runtime(monkeypatch, input_root)
    request = _a2_request(speaker_slot=1)

    request.asset_group = "infinitetalk_v1"
    with pytest.raises(ValueError, match="requires infinitetalk_two_person_v1"):
        app._prepare_comfy_inputs(request)

    request.asset_group = "infinitetalk_two_person_v1"
    request.infinitetalk_routing = None
    with pytest.raises(ValueError, match="requires frozen routing authority"):
        app._prepare_comfy_inputs(request)


def test_effective_receipts_attest_png_and_normalized_wav(monkeypatch, tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    _configure_input_root(monkeypatch, input_root)
    monkeypatch.setattr(app, "normalize_approved_mpeg_to_wav", _fake_normalizer)
    request, image, mpeg = _infinitetalk_request()

    payload, observer_specs = app._prepare_comfy_inputs(request)
    receipts = comfy_client.observe_staged_input_receipts(payload, observer_specs)

    wav_name = payload["10"]["inputs"]["audio"]
    wav_path = input_root / wav_name
    wav_digest = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    assert len(mpeg) == 49_781
    assert wav_path.stat().st_size == 97_324
    assert receipts == [
        {
            "node_id": "6",
            "input_name": "image",
            "content_sha256": hashlib.sha256(image).hexdigest(),
        },
        {"node_id": "10", "input_name": "audio", "content_sha256": wav_digest},
    ]
    assert wav_digest != hashlib.sha256(mpeg).hexdigest()
    wav_spec = observer_specs[1]
    assert wav_spec.content_type == "audio/wav"
    assert wav_spec.expected_sha256 == wav_digest
    assert wav_spec.source_data is None
    assert wav_spec.source_path is None
    assert wav_spec.source_url is None
    assert wav_spec.subfolder == ""


def test_effective_receipts_reject_valid_png_or_wav_tampering(monkeypatch, tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    _configure_input_root(monkeypatch, input_root)
    monkeypatch.setattr(app, "normalize_approved_mpeg_to_wav", _fake_normalizer)
    request, _, _ = _infinitetalk_request()
    payload, observer_specs = app._prepare_comfy_inputs(request)
    wav_path = input_root / payload["10"]["inputs"]["audio"]
    original_wav = wav_path.read_bytes()

    _write_pcm_wav(wav_path, seconds=2.0)
    with pytest.raises(RuntimeError, match="Observed staged input digest mismatch"):
        comfy_client.observe_staged_input_receipts(payload, observer_specs)

    wav_path.write_bytes(original_wav)
    image_path = input_root / payload["6"]["inputs"]["image"]
    image_path.write_bytes(_rgb_png(255, 0, 0))
    with pytest.raises(RuntimeError, match="Observed staged input digest mismatch"):
        comfy_client.observe_staged_input_receipts(payload, observer_specs)


def test_empty_audio_and_nonempty_digest_mismatch_have_distinct_errors(monkeypatch, tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    monkeypatch.setattr(comfy_client, "comfy_input_dir", lambda: input_root)
    expected = hashlib.sha256(b"approved-mp3").hexdigest()
    spec = ComfyInputFile(
        node_id="10",
        input_name="audio",
        filename="approved.mp3",
        expected_sha256=expected,
        content_type="audio/mpeg",
    )
    payload = {"10": {"inputs": {"audio": "observed.mp3"}}}
    observed = input_root / "observed.mp3"

    observed.write_bytes(b"non-empty-but-different")
    with pytest.raises(RuntimeError, match="Observed staged input digest mismatch"):
        comfy_client.observe_staged_input_receipts(payload, [spec])
    observed.write_bytes(b"")
    with pytest.raises(RuntimeError, match="Observed staged input file is empty"):
        comfy_client.observe_staged_input_receipts(payload, [spec])


def test_bad_mp3_is_rejected_before_normalization(monkeypatch, tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    _configure_input_root(monkeypatch, input_root)
    request, _, _ = _infinitetalk_request()
    real_apply = app.apply_comfy_input_files
    normalize_calls = []

    def apply_then_corrupt(payload, specs):
        staged = real_apply(payload, specs)
        audio_path = input_root / staged["10"]["inputs"]["audio"]
        audio_path.write_bytes(b"non-empty-mutated-mp3")
        return staged

    monkeypatch.setattr(app, "apply_comfy_input_files", apply_then_corrupt)
    monkeypatch.setattr(
        app,
        "normalize_approved_mpeg_to_wav",
        lambda *args, **kwargs: normalize_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="Observed staged input digest mismatch"):
        app._prepare_comfy_inputs(request)
    assert normalize_calls == []


def test_execute_run_reobserves_effective_inputs_on_retry(monkeypatch):
    request = RunRequest(job_id="retry-proof", asset_group="infinitetalk_v1", comfy_payload={})
    observations = []
    monkeypatch.setattr(app, "_asset_group_allowed", lambda _group: True)
    monkeypatch.setattr(app, "_effective_vram_floor", lambda _group: None)
    monkeypatch.setattr(app, "_free_vram_on_group_switch", lambda _group: None)
    monkeypatch.setattr(
        app,
        "ensure_asset_group",
        lambda _group: SimpleNamespace(downloaded_assets=[], asset_check_sec=0.0, download_sec=0.0),
    )
    monkeypatch.setattr(app, "_ensure_runtime_provisioned", lambda _group: False)
    monkeypatch.setattr(app, "_prepare_comfy_inputs", lambda _request: ({}, []))

    def observe(_payload, _specs):
        observations.append("observed")
        if len(observations) == 2:
            raise RuntimeError("Observed staged input digest mismatch")
        return []

    monkeypatch.setattr(app, "observe_staged_input_receipts", observe)
    monkeypatch.setattr(
        app,
        "submit_prompt",
        lambda _payload: (_ for _ in ()).throw(RuntimeError("Comfy stopped")),
    )
    monkeypatch.setattr(app, "is_comfy_healthy", lambda: False)
    monkeypatch.setattr(app, "restart_comfy", lambda: 0.0)
    monkeypatch.setattr(app, "_send_heartbeat_now", lambda: None)
    monkeypatch.setattr(
        app,
        "get_settings",
        lambda: SimpleNamespace(comfy_start_cmd="restart", comfy_base_url="http://127.0.0.1:8188"),
    )

    result = app._execute_run(request)

    assert observations == ["observed", "observed"]
    assert result.ok is False
    assert result.restart_performed is True
    assert result.error == "Observed staged input digest mismatch"


def test_normalizer_rejects_symlinked_job_dir_before_cache_reuse(monkeypatch, tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    _configure_input_root(monkeypatch, input_root)
    source = input_root / "approved.mp3"
    source.write_bytes(b"approved-dialogue")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    job_id = "symlink-cache-proof"
    job_scope = runtime._job_scope(job_id)
    attacker_dir = tmp_path / "attacker-controlled"
    attacker_dir.mkdir()
    wrong_wav = attacker_dir / f"{source_digest}.wav"
    _write_pcm_wav(wrong_wav, seconds=1.0)
    job_dir = input_root / "infinitetalk" / job_scope
    job_dir.parent.mkdir()
    job_dir.symlink_to(attacker_dir, target_is_directory=True)
    subprocess_calls = []
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="unsafe symlink"):
        runtime.normalize_approved_mpeg_to_wav(source, job_id=job_id)

    assert runtime._is_valid_normalized_wav(wrong_wav)
    assert subprocess_calls == []


def test_observer_rejects_ancestor_symlink_even_for_matching_wav(monkeypatch, tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    monkeypatch.setattr(comfy_client, "comfy_input_dir", lambda: input_root)
    job_scope = runtime._job_scope("observer-symlink-proof")
    attacker_dir = tmp_path / "attacker-controlled"
    matching_wav = attacker_dir / "matching.wav"
    _write_pcm_wav(matching_wav, seconds=1.0)
    digest = hashlib.sha256(matching_wav.read_bytes()).hexdigest()
    job_dir = input_root / "infinitetalk" / job_scope
    job_dir.parent.mkdir()
    job_dir.symlink_to(attacker_dir, target_is_directory=True)
    payload = {
        "10": {
            "inputs": {
                "audio": f"infinitetalk/{job_scope}/{matching_wav.name}",
            }
        }
    }
    spec = ComfyInputFile(
        node_id="10",
        input_name="audio",
        filename=matching_wav.name,
        expected_sha256=digest,
        content_type="audio/wav",
    )

    with pytest.raises(RuntimeError, match="unsafe symlink"):
        comfy_client.observe_staged_input_receipts(payload, [spec])


def test_mpeg_normalization_is_streamed_bounded_and_job_isolated(monkeypatch, tmp_path):
    source = tmp_path / "approved.mp3"
    source.write_bytes(b"approved-dialogue")
    input_root = tmp_path / "comfy-input"
    monkeypatch.setattr(runtime, "get_settings", lambda: SimpleNamespace(comfy_input_dir=str(input_root)))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout="2.0\n", stderr="")
        _write_pcm_wav(Path(command[-1]))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    # A full-file Path.read_bytes implementation would fail this test.  The
    # normalizer must hash the upload incrementally under its size limit.
    monkeypatch.setattr(Path, "read_bytes", lambda _path: (_ for _ in ()).throw(AssertionError("must stream")))
    first = runtime.normalize_approved_mpeg_to_wav(source, job_id="job/one")
    second = runtime.normalize_approved_mpeg_to_wav(source, job_id="job:one")
    cached = runtime.normalize_approved_mpeg_to_wav(source, job_id="job/one")

    assert first != second
    assert first.parent.name != second.parent.name
    assert cached == first
    assert first.name.endswith(".wav")
    assert runtime._is_valid_normalized_wav(first)
    with wave.open(str(first), "rb") as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) == (1, 16_000, 2, "NONE")
    ffmpeg_calls = [call for call in calls if call[0] == "ffmpeg"]
    assert len(ffmpeg_calls) == 2
    assert ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"] == ffmpeg_calls[0][ffmpeg_calls[0].index("-ac"):ffmpeg_calls[0].index("-ac") + 6]


def test_mpeg_normalization_discards_corrupt_cache_and_rejects_oversize_output(monkeypatch, tmp_path):
    source = tmp_path / "approved.mp3"
    source.write_bytes(b"approved-dialogue")
    monkeypatch.setattr(runtime, "get_settings", lambda: SimpleNamespace(comfy_input_dir=str(tmp_path / "comfy-input")))
    ffmpeg_runs = 0

    def fake_run(command, **kwargs):
        nonlocal ffmpeg_runs
        if command[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout="2.0\n", stderr="")
        ffmpeg_runs += 1
        _write_pcm_wav(Path(command[-1]))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    destination = runtime.normalize_approved_mpeg_to_wav(source, job_id="proof")
    destination.write_bytes(b"RIFF\x00\x00\x00\x00WAVE-not-a-real-wave")
    assert runtime.normalize_approved_mpeg_to_wav(source, job_id="proof") == destination
    assert ffmpeg_runs == 2
    assert runtime._is_valid_normalized_wav(destination)

    def oversized_wav(command, **kwargs):
        if command[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout="2.0\n", stderr="")
        _write_pcm_wav(Path(command[-1]), seconds=16.0)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", oversized_wav)
    with pytest.raises(RuntimeError, match="bounded 16 kHz mono PCM WAV"):
        runtime.normalize_approved_mpeg_to_wav(source, job_id="oversized-output")


def test_mpeg_normalization_rejects_large_upload_long_duration_and_long_job_id(monkeypatch, tmp_path):
    source = tmp_path / "approved.mp3"
    source.write_bytes(b"x" * (runtime.MAX_APPROVED_MPEG_BYTES + 1))
    monkeypatch.setattr(runtime, "get_settings", lambda: SimpleNamespace(comfy_input_dir=str(tmp_path / "comfy-input")))
    with pytest.raises(ValueError, match="byte compatibility-proof limit"):
        runtime.normalize_approved_mpeg_to_wav(source, job_id="proof")

    source.write_bytes(b"approved-dialogue")
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=0, stdout="15.1\n", stderr=""),
    )
    with pytest.raises(ValueError, match="single line"):
        runtime.normalize_approved_mpeg_to_wav(source, job_id="proof")

    max_accepted_scope = runtime._job_scope("x" * runtime.MAX_JOB_ID_UTF8_BYTES)
    assert len(max_accepted_scope.encode("ascii")) == runtime.MAX_JOB_SCOPE_COMPONENT_BYTES
    with pytest.raises(ValueError, match="job_id exceeds"):
        runtime.normalize_approved_mpeg_to_wav(source, job_id="x" * (runtime.MAX_JOB_ID_UTF8_BYTES + 1))


def test_infinitetalk_provisioner_runs_once_per_worker_process(monkeypatch):
    asset_manager._PROVISIONED_GROUPS.clear()
    calls = []
    monkeypatch.setattr(
        asset_manager.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )

    assert asset_manager.ensure_asset_group_provisioned("infinitetalk_v1") is True
    assert asset_manager.ensure_asset_group_provisioned("infinitetalk_v1") is False
    assert calls == [["bash", str(Path(asset_manager.__file__).resolve().parents[1] / "gpu_worker/provision_infinitetalk.sh")]]
