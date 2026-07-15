#!/usr/bin/env python3
"""Standalone Stable Audio 3 inference — runs INSIDE the sa3_spike venv.

Provisioner-backed capability (see provision_sa3.sh + asset_registry stable_audio3_v1):
SA3 is a distilled 8-step music model loaded via the official `stable-audio-3` lib
(NOT a ComfyUI single-file checkpoint), so the worker invokes it as a subprocess in
its own venv rather than through the ComfyUI graph. This script takes a prompt +
duration and writes an mp3.

Invoked by gpu_worker/app.py::_run_sa3_audio as:
    <sa3_spike>/bin/python sa3_infer.py --prompt "..." --seconds 30 --out /path/out.mp3

⚠️ VERIFY-BEFORE-DEPLOY: the generate(...) call + return handling below follow the
documented `stable-audio-3` API (StableAudioModel.from_pretrained("medium") →
generate(prompt, duration, steps=8, cfg_scale≈1.0)). If the validated spike used a
different call shape or return type, correct _generate() / _to_pcm() to match — this
file is the single place that touches the SA3 library.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[sa3_infer] {msg}", file=sys.stderr, flush=True)


def _generate(prompt: str, seconds: float):
    """Run SA3. Returns (audio, sample_rate) where audio is a torch/np array.

    Uses the model's own default guidance for the distilled model (cfg_scale≈1.0,
    8 steps). Kept in one place so the exact call is easy to reconcile with the spike.
    """
    from stable_audio_3 import StableAudioModel  # type: ignore

    model = StableAudioModel.from_pretrained("medium")
    _log(f"model loaded; generating {seconds:.1f}s")
    audio = model.generate(
        prompt=prompt,
        duration=float(seconds),
        steps=8,
        cfg_scale=1.0,
    )
    # Sample rate: prefer the model's own, else Stable Audio's canonical 44.1kHz.
    sample_rate = int(
        getattr(model, "sample_rate", None)
        or getattr(getattr(model, "config", None), "sample_rate", None)
        or 44100
    )
    return audio, sample_rate


def _to_frames(audio):
    """Normalize the model output to a float32 numpy array shaped [frames, channels]
    (soundfile's convention). Handles torch tensors / numpy arrays shaped [samples],
    [channels, samples], or [batch, channels, samples]."""
    import numpy as np

    arr = audio
    # torch tensor → numpy
    if hasattr(arr, "detach"):
        arr = arr.detach().to("cpu").float().numpy()
    arr = np.asarray(arr, dtype="float32")

    # Collapse a leading batch dim.
    if arr.ndim == 3:
        arr = arr[0]
    # Shape to [frames, channels].
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim == 2:
        # [channels, samples] (channels small) → transpose to [frames, channels]
        if arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
            arr = arr.T
    else:
        raise ValueError(f"unexpected audio ndim={arr.ndim} shape={arr.shape}")

    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:  # guard against slight overshoot before clipping
        arr = arr / peak
    return np.clip(arr, -1.0, 1.0)


def _write_wav(path: Path, frames, sample_rate: int) -> None:
    """Write a WAV. Prefer soundfile (the validated-spike save path); fall back to the
    `wave` stdlib (int16) so the venv works even without soundfile installed."""
    try:
        import soundfile as sf  # matches the spike: sf.write(path, audio, sr)
        sf.write(str(path), frames, sample_rate)
        return
    except ImportError:
        _log("soundfile not available — falling back to wave stdlib (int16)")

    import numpy as np
    pcm16 = (np.asarray(frames, dtype="float32") * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(int(pcm16.shape[1]) if pcm16.ndim > 1 else 1)
        w.setsampwidth(2)  # int16
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())


def _transcode_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame",
         "-qscale:a", "2", str(mp3_path)],
        check=True, capture_output=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", required=True, help="output .mp3 path")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    audio, sample_rate = _generate(args.prompt, args.seconds)
    frames = _to_frames(audio)
    _log(f"frames ready: shape={frames.shape} @ {sample_rate}Hz")

    if out.suffix.lower() == ".mp3":
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "sa3.wav"
            _write_wav(wav, frames, sample_rate)
            _transcode_to_mp3(wav, out)
    else:
        _write_wav(out, frames, sample_rate)

    _log(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
