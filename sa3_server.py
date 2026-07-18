"""Resident Stable Audio 3 server — warm music cues, lazy VRAM.

Runs inside the sa3_spike venv (see provision_sa3.sh). Solves the ~50s
model-load-per-cue floor of the subprocess path (sa3_infer.py): the model loads
on the FIRST cue of a session (~50s), stays resident (~15GB) so follow-up cues
render in seconds, then unloads itself after SA3_IDLE_UNLOAD_SEC of quiet to
hand the VRAM back to renders.

POST /music  {"prompt": "...", "seconds": 30}  -> audio/mpeg (mp3)
GET  /health -> {"ok": true, "model_loaded": bool}

If loading fails (e.g. card full mid-render), returns 503 — the worker's
_run_sa3_audio falls back to the subprocess path, which knows how to evict.
Reuses sa3_infer's generation/normalization/writing helpers verbatim so there
is still exactly one place that touches the SA3 library call shape.

Deployed as systemd unit filmforge-sa3.service (see setup_audio_services.sh).
"""

import io
import json
import os
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sa3_infer  # same directory; helpers _to_frames/_write_wav + the API shape

PORT = int(os.environ.get("SA3_PORT", "9102"))
IDLE_UNLOAD_SEC = int(os.environ.get("SA3_IDLE_UNLOAD_SEC", "600"))

_state = {"model": None, "sample_rate": 44100, "last_used": 0.0}
_lock = threading.Lock()  # load, generate, and unload are mutually exclusive


def _ensure_model():
    """Load SA3 if not resident. Caller holds _lock. Raises on failure (e.g. OOM)."""
    if _state["model"] is None:
        from stable_audio_3 import StableAudioModel  # type: ignore

        print("[sa3_server] loading model …", flush=True)
        started = time.monotonic()
        model = StableAudioModel.from_pretrained("medium")
        _state["model"] = model
        _state["sample_rate"] = int(
            getattr(model, "sample_rate", None)
            or getattr(getattr(model, "config", None), "sample_rate", None)
            or 44100
        )
        print(f"[sa3_server] model resident in {time.monotonic() - started:.1f}s", flush=True)
    _state["last_used"] = time.monotonic()


def _idle_reaper():
    while True:
        time.sleep(60)
        with _lock:
            if _state["model"] is not None and time.monotonic() - _state["last_used"] > IDLE_UNLOAD_SEC:
                print("[sa3_server] idle — unloading model, returning VRAM", flush=True)
                _state["model"] = None
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass


def _generate_mp3(prompt: str, seconds: float) -> bytes:
    with _lock:
        _ensure_model()
        audio = _state["model"].generate(
            prompt=prompt, duration=float(seconds), steps=8, cfg_scale=1.0
        )
        _state["last_used"] = time.monotonic()
    frames = sa3_infer._to_frames(audio)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "cue.wav"
        mp3 = Path(td) / "cue.mp3"
        sa3_infer._write_wav(wav, frames, _state["sample_rate"])
        sa3_infer._transcode_to_mp3(wav, mp3)
        return mp3.read_bytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "model_loaded": _state["model"] is not None})
        else:
            self._json(404, {"ok": False})

    def do_POST(self):
        if self.path != "/music":
            self._json(404, {"ok": False})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            prompt = str(req.get("prompt") or "").strip()
            seconds = max(5.0, min(120.0, float(req.get("seconds") or 30.0)))
            if not prompt:
                self._json(400, {"ok": False, "error": "empty prompt"})
                return
            mp3 = _generate_mp3(prompt, seconds)
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(mp3)))
            self.end_headers()
            self.wfile.write(mp3)
        except Exception as exc:
            # OOM / load failure / anything → 503 so the worker falls back to
            # the subprocess path (which can evict comfy models).
            self._json(503, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    threading.Thread(target=_idle_reaper, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
