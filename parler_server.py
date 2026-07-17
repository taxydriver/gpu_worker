"""Resident Indic Parler-TTS server — Maya's voice box.

Runs inside the parler_spike venv (see provision_tts.sh) and holds the model in
VRAM (~4GB) so each line renders in seconds instead of paying the ~60s load per
utterance. Stdlib HTTP only — no extra deps in the venv.

POST /tts  {"text": "...", "description": "<voice description>"}  -> audio/wav
GET  /health -> {"ok": true, "model_loaded": true}

The voice IS the description string (Parler is prompt-conditioned, not a cloner).
Language comes from the text itself — Indic Parler reads Telugu/Tamil/Hindi/
Indian-English scripts natively.

Deployed as systemd unit filmforge-parler.service; the gpu_worker app proxies
tts_dialogue_v1 jobs to this server (see app.py _run_tts_dialogue).
"""

import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PARLER_PORT", "9101"))
MODEL_ID = "ai4bharat/indic-parler-tts"

DEFAULT_DESCRIPTION = (
    "A young adult female speaker with a clear, warm, expressive voice, "
    "speaking naturally at a moderate pace, recorded very close with no "
    "background noise."
)

_state = {"model": None, "tokenizer": None, "desc_tokenizer": None, "sr": 44100}
_lock = threading.Lock()  # one generation at a time — single GPU


def _load():
    import torch  # noqa: F401 — ensures CUDA context in this process
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    print(f"[parler_server] loading {MODEL_ID} …", flush=True)
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    _state.update(
        model=model,
        tokenizer=tokenizer,
        desc_tokenizer=desc_tokenizer,
        sr=model.config.sampling_rate,
    )
    print(f"[parler_server] ready on :{PORT} (sr={_state['sr']})", flush=True)


def _generate_wav(text: str, description: str) -> bytes:
    import soundfile as sf

    model, tok, dtok = _state["model"], _state["tokenizer"], _state["desc_tokenizer"]
    desc = dtok(description, return_tensors="pt").to("cuda")
    prompt = tok(text, return_tensors="pt").to("cuda")
    with _lock:
        generation = model.generate(
            input_ids=desc.input_ids,
            attention_mask=desc.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
        )
    audio = generation.cpu().to(dtype=__import__("torch").float32).numpy().squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, _state["sr"], format="WAV")
    return buf.getvalue()


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
        if self.path != "/tts":
            self._json(404, {"ok": False})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            text = str(req.get("text") or "").strip()
            if not text:
                self._json(400, {"ok": False, "error": "empty text"})
                return
            if _state["model"] is None:
                self._json(503, {"ok": False, "error": "model still loading"})
                return
            description = str(req.get("description") or DEFAULT_DESCRIPTION)
            wav = _generate_wav(text[:1200], description)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.end_headers()
            self.wfile.write(wav)
        except Exception as exc:
            self._json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    # 0.0.0.0: the rail service calls this directly (Maya's voice must not queue
    # behind render jobs). Same open-port posture as the worker on :9000.
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=_load, daemon=True).start()  # serve /health during load
    server.serve_forever()
