from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from gpu_worker import app
from gpu_worker import asset_manager
from gpu_worker import infinitetalk as runtime


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

    capabilities, readiness, flux_ipadapter_readiness = app._advertised_capabilities()

    assert capabilities == ["wan_i2v_v1"]
    assert readiness == {
        "ready": False,
        "missing_files": ["missing"],
        "missing_node_classes": [],
        "wav2vec_dependency_error": None,
        "comfy_error": None,
    }
    assert flux_ipadapter_readiness is None  # not declared → not probed


def _write_pcm_wav(path: Path, *, seconds: float = 2.0) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * int(16_000 * seconds))


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
