from __future__ import annotations

from pathlib import Path

from gpu_worker.comfy_client import apply_comfy_input_files
from gpu_worker.schemas import ComfyInputFile


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
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
