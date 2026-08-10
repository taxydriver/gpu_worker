from __future__ import annotations

import base64
import hashlib
import logging
import socket
import stat
import traceback
from types import SimpleNamespace

import pytest

from gpu_worker import comfy_client, deploy_gpu, worker_auth
from gpu_worker.schemas import ComfyInputFile


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x00\x00\x00\x00"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_worker_auth_rejects_missing_token_in_required_mode() -> None:
    with pytest.raises(worker_auth.WorkerAPIAuthError) as caught:
        worker_auth.verify_worker_api_authorization(
            expected_token=None,
            auth_mode="required",
            authorization=None,
        )
    assert caught.value.status_code == 503


def test_worker_auth_rejects_wrong_token_with_constant_time_compare(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def compare(provided: str, expected: str) -> bool:
        calls.append((provided, expected))
        return False

    monkeypatch.setattr(worker_auth.secrets, "compare_digest", compare)
    with pytest.raises(worker_auth.WorkerAPIAuthError) as caught:
        worker_auth.verify_worker_api_authorization(
            expected_token="fixture-worker-token",
            auth_mode="required",
            authorization="Bearer wrong-token",
        )
    assert caught.value.status_code == 401
    assert calls == [("wrong-token", "fixture-worker-token")]


def test_worker_auth_accepts_right_token() -> None:
    worker_auth.verify_worker_api_authorization(
        expected_token="fixture-worker-token",
        auth_mode="required",
        authorization="bearer fixture-worker-token",
    )


def test_worker_auth_allows_missing_token_only_in_explicit_test_mode() -> None:
    worker_auth.verify_worker_api_authorization(
        expected_token=None,
        auth_mode="test",
        authorization=None,
    )


def _settings(allowed_hosts: str, chunk_size: int = 4):
    return SimpleNamespace(
        worker_input_url_allowed_hosts=allowed_hosts,
        download_chunk_size=chunk_size,
    )


def _public_dns(_host, port, *, type):
    assert type == socket.SOCK_STREAM
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]


class _Response:
    def __init__(self, *, status=200, headers=None, chunks=()):
        self.status = status
        self.headers = dict(headers or {})
        self._chunks = chunks
        self.closed = False

    def stream(self, *, amt, decode_content):
        assert amt > 0
        assert decode_content is False
        for chunk in self._chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


class _Pool:
    def __init__(self, response: _Response):
        self.response = response
        self.calls = []
        self.closed = False

    def request(self, method, target, **kwargs):
        self.calls.append((method, target, kwargs))
        return self.response

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "url,allowed_host",
    [
        ("http://127.0.0.1/input.png", "127.0.0.1"),
        ("http://[::1]/input.png", "::1"),
        ("http://169.254.169.254/latest/meta-data", "169.254.169.254"),
        ("http://2130706433/input.png", "2130706433"),
    ],
)
def test_remote_input_rejects_ip_literals_and_decimal_variant(
    monkeypatch, url, allowed_host
) -> None:
    monkeypatch.setattr(comfy_client, "get_settings", lambda: _settings(allowed_host))
    with pytest.raises(comfy_client._RemoteInputRejected):
        comfy_client._validate_remote_input_url(url)


@pytest.mark.parametrize("host", ["0x7f000001", "0x7f.0x0.0x0.0x1"])
def test_remote_input_rejects_hex_ip_variants(monkeypatch, host) -> None:
    monkeypatch.setattr(comfy_client, "get_settings", lambda: _settings(host))
    monkeypatch.setattr(
        comfy_client.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ],
    )
    with pytest.raises(comfy_client._RemoteInputRejected):
        comfy_client._validate_remote_input_url(f"http://{host}/input.png")


def test_remote_input_rejects_dns_name_resolving_private(monkeypatch) -> None:
    monkeypatch.setattr(
        comfy_client, "get_settings", lambda: _settings("images.example.test")
    )
    monkeypatch.setattr(
        comfy_client.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", 443))
        ],
    )
    with pytest.raises(comfy_client._RemoteInputRejected, match="dns_non_public"):
        comfy_client._validate_remote_input_url(
            "https://images.example.test/input.png"
        )


def _install_fetch(monkeypatch, response: _Response) -> _Pool:
    pool = _Pool(response)
    monkeypatch.setattr(
        comfy_client,
        "_open_pinned_http_pool",
        lambda *_args, **_kwargs: pool,
    )
    monkeypatch.setattr(comfy_client.socket, "getaddrinfo", _public_dns)
    monkeypatch.setattr(
        comfy_client,
        "get_settings",
        lambda: _settings("images.example.test"),
    )
    return pool


def _image_spec() -> ComfyInputFile:
    return ComfyInputFile(
        node_id="40",
        filename="reference.png",
        source_url="https://images.example.test/reference.png",
    )


def test_remote_input_does_not_follow_redirect_to_private(monkeypatch, tmp_path) -> None:
    response = _Response(
        status=302,
        headers={"Location": "http://127.0.0.1/private"},
    )
    pool = _install_fetch(monkeypatch, response)
    with pytest.raises(comfy_client._RemoteInputRejected, match="redirect_forbidden"):
        comfy_client._download_input_source(
            "https://images.example.test/reference.png",
            tmp_path / "reference.png",
            _image_spec(),
        )
    assert len(pool.calls) == 1
    assert pool.calls[0][2]["redirect"] is False


def test_remote_input_rejects_oversized_declared_body(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(comfy_client, "_MAX_REMOTE_INPUT_BYTES", 8)
    _install_fetch(
        monkeypatch,
        _Response(
            headers={"Content-Type": "image/png", "Content-Length": "9"},
            chunks=[b"ignored"],
        ),
    )
    with pytest.raises(comfy_client._RemoteInputRejected, match="body_too_large"):
        comfy_client._download_input_source(
            "https://images.example.test/reference.png",
            tmp_path / "reference.png",
            _image_spec(),
        )
    assert not (tmp_path / "reference.png").exists()


def test_remote_input_rejects_oversized_chunked_body_and_cleans_temp(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(comfy_client, "_MAX_REMOTE_INPUT_BYTES", 8)
    _install_fetch(
        monkeypatch,
        _Response(
            headers={"Content-Type": "image/png"},
            chunks=[b"1234", b"5678", b"9"],
        ),
    )
    with pytest.raises(comfy_client._RemoteInputRejected, match="body_too_large"):
        comfy_client._download_input_source(
            "https://images.example.test/reference.png",
            tmp_path / "reference.png",
            _image_spec(),
        )
    assert list(tmp_path.iterdir()) == []


def test_remote_input_enforces_media_content_type(monkeypatch, tmp_path) -> None:
    _install_fetch(
        monkeypatch,
        _Response(
            headers={"Content-Type": "text/html"},
            chunks=[b"not an image"],
        ),
    )
    with pytest.raises(comfy_client._RemoteInputRejected, match="content_type_mismatch"):
        comfy_client._download_input_source(
            "https://images.example.test/reference.png",
            tmp_path / "reference.png",
            _image_spec(),
        )


def test_remote_input_errors_and_logs_redact_credentials_query_path_and_body(
    monkeypatch, tmp_path, caplog
) -> None:
    credential_url = (
        "https://user-secret:password-secret@images.example.test/"
        "private-path-secret.png?token=query-secret"
    )
    monkeypatch.setattr(
        comfy_client,
        "get_settings",
        lambda: _settings("images.example.test"),
    )
    with caplog.at_level(logging.WARNING, logger=comfy_client.LOGGER.name):
        with pytest.raises(comfy_client._RemoteInputRejected) as credential_error:
            comfy_client._validate_remote_input_url(credential_url)

    response = _Response(
        headers={"Content-Type": "image/png"},
        chunks=[RuntimeError("response-body-secret")],
    )
    _install_fetch(monkeypatch, response)
    query_url = (
        "https://images.example.test/private-path-secret.png?token=query-secret"
    )
    destination = tmp_path / "private-path-secret.png"
    with pytest.raises(comfy_client._RemoteInputRejected) as transport_error:
        comfy_client._download_input_source(
            query_url,
            destination,
            _image_spec(),
        )
    rendered = "".join(
        traceback.format_exception(
            type(transport_error.value),
            transport_error.value,
            transport_error.value.__traceback__,
        )
    )
    public_text = "\n".join(
        [str(credential_error.value), str(transport_error.value), caplog.text, rendered]
    )
    for secret in (
        "user-secret",
        "password-secret",
        "private-path-secret",
        "query-secret",
        "response-body-secret",
    ):
        assert secret not in public_text


def test_valid_allowlisted_remote_input_uses_private_atomic_file(monkeypatch, tmp_path) -> None:
    response = _Response(
        headers={
            "Content-Type": "image/png; charset=binary",
            "Content-Length": str(len(PNG_BYTES)),
        },
        chunks=[PNG_BYTES[:8], PNG_BYTES[8:]],
    )
    pool = _install_fetch(monkeypatch, response)
    destination = tmp_path / "reference.png"
    comfy_client._download_input_source(
        "https://images.example.test/reference.png?version=1",
        destination,
        _image_spec(),
    )
    assert destination.read_bytes() == PNG_BYTES
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    method, target, kwargs = pool.calls[0]
    assert method == "GET"
    assert target == "/reference.png?version=1"
    assert kwargs["headers"] == {"Host": "images.example.test"}
    assert kwargs["redirect"] is False


def test_remote_input_connects_to_vetted_ip_with_original_tls_identity(
    monkeypatch, tmp_path
) -> None:
    dns_calls = 0

    def resolve_once(_host, port, *, type):
        nonlocal dns_calls
        dns_calls += 1
        if dns_calls > 1:
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))
            ]
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", port),
            )
        ]

    response = _Response(
        headers={
            "Content-Type": "image/png",
            "Content-Length": str(len(PNG_BYTES)),
        },
        chunks=[PNG_BYTES],
    )
    constructor = {}

    class Pool(_Pool):
        def __init__(self, host, **kwargs):
            super().__init__(response)
            constructor.update(host=host, **kwargs)

    monkeypatch.setattr(comfy_client.socket, "getaddrinfo", resolve_once)
    monkeypatch.setattr(
        comfy_client,
        "get_settings",
        lambda: _settings("images.example.test"),
    )
    monkeypatch.setattr(comfy_client, "HTTPSConnectionPool", Pool)

    destination = tmp_path / "reference.png"
    comfy_client._download_input_source(
        "https://images.example.test/reference.png?token=opaque",
        destination,
        _image_spec(),
    )

    assert dns_calls == 1
    assert constructor["host"] == "93.184.216.34"
    assert constructor["port"] == 443
    assert constructor["assert_hostname"] == "images.example.test"
    assert constructor["server_hostname"] == "images.example.test"
    assert destination.read_bytes() == PNG_BYTES


def test_attested_candidate_rejects_source_url_without_network(monkeypatch, tmp_path) -> None:
    digest = hashlib.sha256(PNG_BYTES).hexdigest()
    monkeypatch.setattr(comfy_client, "comfy_input_dir", lambda: tmp_path)
    monkeypatch.setattr(
        comfy_client.requests,
        "Session",
        lambda: (_ for _ in ()).throw(AssertionError("network must not run")),
    )
    with pytest.raises(RuntimeError, match="inline source_data only"):
        comfy_client.stage_comfy_input_file(
            ComfyInputFile(
                node_id="40",
                filename="candidate_reference.png",
                source_url="https://images.example.test/reference.png",
                expected_sha256=digest,
            )
        )


def test_valid_attested_inline_candidate_stays_content_addressed(monkeypatch, tmp_path) -> None:
    digest = hashlib.sha256(PNG_BYTES).hexdigest()
    monkeypatch.setattr(comfy_client, "comfy_input_dir", lambda: tmp_path)
    staged = comfy_client.stage_comfy_input_file(
        ComfyInputFile(
            node_id="40",
            filename="candidate_reference.png",
            source_data=base64.b64encode(PNG_BYTES).decode("ascii"),
            expected_sha256=digest,
        )
    )
    assert staged == f"sha256_{digest}.png"
    assert (tmp_path / staged).read_bytes() == PNG_BYTES


def test_attested_candidate_verifies_new_inline_bytes_before_cache_reuse(
    monkeypatch, tmp_path
) -> None:
    digest = hashlib.sha256(PNG_BYTES).hexdigest()
    cached = tmp_path / f"sha256_{digest}.png"
    cached.write_bytes(PNG_BYTES)
    monkeypatch.setattr(comfy_client, "comfy_input_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="content digest mismatch"):
        comfy_client.stage_comfy_input_file(
            ComfyInputFile(
                node_id="40",
                filename="candidate_reference.png",
                source_data=base64.b64encode(b"different image bytes").decode("ascii"),
                expected_sha256=digest,
            )
        )

    assert cached.read_bytes() == PNG_BYTES


def test_deploy_scripts_forward_worker_ingress_security_environment() -> None:
    scripts = [
        deploy_gpu.remote_script("/workspace/worker", 9000),
        deploy_gpu.vast_multi_gpu_script(
            remote_root="/workspace/worker",
            worker_port=9000,
            comfy_port=18188,
            worker_count=1,
        ),
        deploy_gpu.verda_rehydrate_script(
            public_ip="203.0.113.10",
            worker_port=9000,
            comfy_port=18188,
            worker_count=1,
            remote_root="/workspace/worker",
        ),
    ]
    for script in scripts:
        assert "GPU_WORKER_API_TOKEN" in script
        assert "WORKER_API_AUTH_MODE" in script
        assert "WORKER_INPUT_URL_ALLOWED_HOSTS" in script


def test_verda_env_auto_injects_worker_ingress_policy(tmp_path) -> None:
    backend_env = tmp_path / "backend.env"
    backend_env.write_text(
        "WORKER_API_AUTH_MODE=required\n"
        "WORKER_INPUT_URL_ALLOWED_HOSTS=storage.example.test,cdn.example.test\n"
    )
    args = SimpleNamespace(
        env_vars=[],
        backend_env=backend_env,
        verda_worker_plan="generation",
    )

    resolved = deploy_gpu._verda_env_vars(
        args,
        instance_id="fixture-verda-instance-id",
    )

    assert "WORKER_INSTANCE_ID=fixture-verda-instance-id" in resolved
    assert "WORKER_API_AUTH_MODE=required" in resolved
    assert (
        "WORKER_INPUT_URL_ALLOWED_HOSTS=storage.example.test,cdn.example.test"
        in resolved
    )


def test_deploy_rejects_public_cleartext_worker_url() -> None:
    with pytest.raises(RuntimeError, match="requires HTTPS"):
        deploy_gpu._validate_worker_public_url_env(
            [
                "WORKER_API_AUTH_MODE=required",
                "WORKER_PUBLIC_URL=http://203.0.113.60:9000",
            ]
        )


def test_deploy_accepts_tls_or_loopback_worker_url() -> None:
    deploy_gpu._validate_worker_public_url_env(
        [
            "WORKER_API_AUTH_MODE=required",
            "WORKER_PUBLIC_URLS=https://worker-a.example,http://127.0.0.1:9001",
        ]
    )


@pytest.mark.parametrize(
    "worker_url",
    ["https://worker.example", "http://127.0.0.1:9000", "http://[::1]:9000"],
)
def test_worker_warmup_uses_proxyless_no_redirect_transport(
    monkeypatch, worker_url
) -> None:
    captured = {"requests": []}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, size):
            assert size == deploy_gpu._MAX_WORKER_WARMUP_RESPONSE_BYTES + 1
            return b'{"ok": true}'

    class Opener:
        def open(self, request, **kwargs):
            captured["requests"].append(request)
            captured["open_kwargs"] = kwargs
            return Response()

    def build(*handlers):
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(deploy_gpu, "build_opener", build)
    result = deploy_gpu.warm_remote_worker(
        worker_url,
        ["flux_stills_v1"],
        api_token="fixture-worker-token",
        timeout_sec=30,
    )

    assert result == {"ok": True}
    assert len(captured["requests"]) == 1
    request = captured["requests"][0]
    assert request.full_url == f"{worker_url}/assets/ensure"
    assert request.get_header("Authorization") == "Bearer fixture-worker-token"
    assert captured["open_kwargs"] == {"timeout": 30}
    proxy_handlers = [
        handler
        for handler in captured["handlers"]
        if isinstance(handler, deploy_gpu.ProxyHandler)
    ]
    assert len(proxy_handlers) == 1
    assert proxy_handlers[0].proxies == {}
    assert any(
        isinstance(handler, deploy_gpu._RejectWorkerRedirectHandler)
        for handler in captured["handlers"]
    )


def test_worker_warmup_rejects_public_cleartext_before_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        deploy_gpu,
        "build_opener",
        lambda *_args: (_ for _ in ()).throw(AssertionError("transport must not open")),
    )
    with pytest.raises(RuntimeError, match="requires HTTPS"):
        deploy_gpu.warm_remote_worker(
            "http://203.0.113.60:9000",
            ["flux_stills_v1"],
            api_token="fixture-worker-token",
        )


def test_worker_warmup_redirect_never_replays_bearer(
    monkeypatch, caplog
) -> None:
    calls = []
    attacker_url = "http://169.254.169.254/latest/meta-data/"
    token = "fixture-worker-token-must-not-leak"

    class Opener:
        def __init__(self, handlers):
            self.redirect_handler = next(
                handler
                for handler in handlers
                if isinstance(handler, deploy_gpu._RejectWorkerRedirectHandler)
            )

        def open(self, request, **_kwargs):
            calls.append(request)
            redirected = self.redirect_handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {"Location": attacker_url},
                attacker_url,
            )
            if redirected is not None:
                calls.append(redirected)
            raise deploy_gpu.HTTPError(
                request.full_url,
                302,
                "Found",
                {"Location": attacker_url},
                None,
            )

    monkeypatch.setattr(
        deploy_gpu,
        "build_opener",
        lambda *handlers: Opener(handlers),
    )

    with pytest.raises(RuntimeError, match="redirect rejected") as caught:
        deploy_gpu.warm_remote_worker(
            "https://worker.example",
            ["flux_stills_v1"],
            api_token=token,
        )

    assert len(calls) == 1
    assert calls[0].full_url == "https://worker.example/assets/ensure"
    assert calls[0].get_header("Authorization") == f"Bearer {token}"
    assert attacker_url not in str(caught.value)
    assert token not in str(caught.value)
    assert attacker_url not in caplog.text
    assert token not in caplog.text


def test_worker_warmup_response_is_hard_capped(monkeypatch) -> None:
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, size):
            return b"x" * size

    class Opener:
        def open(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(deploy_gpu, "build_opener", lambda *_handlers: Opener())
    with pytest.raises(RuntimeError, match="exceeded byte limit"):
        deploy_gpu.warm_remote_worker(
            "https://worker.example",
            ["flux_stills_v1"],
            api_token="fixture-worker-token",
        )


def test_vast_direct_public_http_is_never_advertised(monkeypatch) -> None:
    monkeypatch.setattr(
        deploy_gpu,
        "_vastai",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider lookup must not run")
        ),
    )
    assert deploy_gpu._vast_direct_worker_url("instance-1", 9000) is None


def test_verda_script_requires_explicit_tls_worker_urls() -> None:
    script = deploy_gpu.verda_rehydrate_script(
        public_ip="203.0.113.10",
        worker_port=9000,
        comfy_port=18188,
        worker_count=1,
        remote_root="/workspace/worker",
    )
    assert "TLS WORKER_PUBLIC_URLS entry is required" in script
    assert "Environment=WORKER_PUBLIC_URL=http://${PUBLIC_IP}" not in script
    assert 'Environment="WORKER_PUBLIC_URL=${worker_public_url}"' in script
    assert "Environment=WORKER_HOST=127.0.0.1" in script
    assert "--host 127.0.0.1" in script
    assert "uvicorn gpu_worker.app:app --host 0.0.0.0" not in script
