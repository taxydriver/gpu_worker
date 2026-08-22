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

from gpu_worker import app
from gpu_worker import asset_manager
from gpu_worker import comfy_client
from gpu_worker import infinitetalk as runtime
from gpu_worker.schemas import ComfyInputFile, RunRequest


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
        lambda: runtime.InfiniteTalkReadiness(ready=False, missing_files=("missing",)),
    )

    capabilities, readiness = app._advertised_capabilities()

    assert capabilities == ["wan_i2v_v1"]
    assert readiness == {
        "ready": False,
        "missing_files": ["missing"],
        "missing_node_classes": [],
        "wav2vec_dependency_error": None,
        "comfy_error": None,
    }


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
