from __future__ import annotations

import stat
from types import SimpleNamespace

import pytest

from gpu_worker import stitch


class _Response:
    def __init__(self, *, status=200, headers=None, chunks=()):
        self.status = status
        self.headers = dict(headers or {})
        self._chunks = list(chunks)

    def stream(self, *, amt, decode_content):
        assert amt > 0
        assert decode_content is False
        yield from self._chunks

    def close(self):
        return None


class _Pool:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def request(self, method, target, **kwargs):
        self.calls.append((method, target, kwargs))
        return self.response

    def close(self):
        return None


def test_stitch_download_rejects_private_url_before_network(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        stitch,
        "_open_pinned_http_pool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network must not run")
        ),
    )
    with pytest.raises(stitch._RemoteInputRejected):
        stitch._download("http://127.0.0.1/clip.mp4", str(tmp_path / "clip.mp4"))


def test_stitch_download_is_bounded_private_and_redirect_free(monkeypatch, tmp_path) -> None:
    payload = b"bounded-video-bytes"
    response = _Response(
        headers={"Content-Type": "video/mp4", "Content-Length": str(len(payload))},
        chunks=[payload],
    )
    pool = _Pool(response)
    target = SimpleNamespace(
        scheme="https",
        request_target="/clip.mp4",
        host_header="cdn.example",
    )
    monkeypatch.setattr(stitch, "_validate_remote_input_url", lambda _url: target)
    monkeypatch.setattr(
        stitch, "_open_pinned_http_pool", lambda *_args, **_kwargs: pool
    )
    destination = tmp_path / "clip.mp4"
    stitch._download("https://cdn.example/clip.mp4", str(destination))

    assert destination.read_bytes() == payload
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert pool.calls[0][2]["redirect"] is False


def test_stitch_download_rejects_oversized_chunked_body(monkeypatch, tmp_path) -> None:
    response = _Response(
        headers={"Content-Type": "video/mp4"},
        chunks=[b"1234", b"5678", b"9"],
    )
    monkeypatch.setattr(stitch, "_MAX_STITCH_INPUT_BYTES", 8)
    target = SimpleNamespace(
        scheme="https",
        request_target="/clip.mp4",
        host_header="cdn.example",
    )
    monkeypatch.setattr(stitch, "_validate_remote_input_url", lambda _url: target)
    monkeypatch.setattr(
        stitch, "_open_pinned_http_pool", lambda *_args, **_kwargs: _Pool(response)
    )
    with pytest.raises(RuntimeError, match="stitch input fetch failed"):
        stitch._download("https://cdn.example/clip.mp4", str(tmp_path / "clip.mp4"))
    assert list(tmp_path.iterdir()) == []


def test_signed_upload_url_is_validated_before_put(monkeypatch, tmp_path) -> None:
    media = tmp_path / "cut.mp4"
    media.write_bytes(b"cut")
    monkeypatch.setattr(
        stitch,
        "_open_pinned_http_pool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("PUT must not run")
        ),
    )
    with pytest.raises(stitch._RemoteInputRejected):
        stitch.upload_via_signed_put(
            "http://169.254.169.254/upload",
            str(media),
        )


def test_signed_upload_requires_https_before_connect(monkeypatch, tmp_path) -> None:
    media = tmp_path / "cut.mp4"
    media.write_bytes(b"cut")
    target = SimpleNamespace(scheme="http")
    monkeypatch.setattr(stitch, "_validate_remote_input_url", lambda _url: target)
    monkeypatch.setattr(
        stitch,
        "_open_pinned_http_pool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("PUT must not run")
        ),
    )

    with pytest.raises(RuntimeError, match="requires HTTPS"):
        stitch.upload_via_signed_put(
            "http://uploads.example/cut.mp4?signature=redacted",
            str(media),
        )
