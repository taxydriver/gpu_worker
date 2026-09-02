from pathlib import Path
from types import SimpleNamespace
import base64

import pytest

from gpu_worker import app as app_mod


def _file(path: Path):
    return SimpleNamespace(path=str(path), filename=path.name)


def test_history_without_output_is_not_success(tmp_path):
    with pytest.raises(RuntimeError, match="no output paths"):
        app_mod._assert_material_outputs("flux_stills_v1", [], [])


def test_empty_output_file_is_not_success(tmp_path):
    empty = tmp_path / "empty.png"
    empty.touch()
    with pytest.raises(RuntimeError, match="absent or empty"):
        app_mod._assert_material_outputs(
            "flux_stills_v1", [str(empty)], [_file(empty)]
        )


def test_still_success_requires_a_readable_image(tmp_path):
    bad = tmp_path / "not-an-image.png"
    bad.write_bytes(b"material but not pixels")
    with pytest.raises(RuntimeError, match="no readable image"):
        app_mod._assert_material_outputs(
            "flux_stills_v1", [str(bad)], [_file(bad)]
        )

    good = tmp_path / "good.png"
    good.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    ))
    app_mod._assert_material_outputs(
        "flux_stills_v1", [str(good)], [_file(good)]
    )
