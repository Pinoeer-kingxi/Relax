# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio

import httpx
import pytest

from relax.utils import http_utils
from relax.utils.http_utils import _post, init_http_client, router_worker_base_url, router_worker_base_urls


class _StubClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def post(self, url, json=None, headers=None):
        self.calls += 1
        return self._responses.pop(0)


def _response(status_code: int, *, body: str | dict):
    request = httpx.Request("POST", "http://test/post")
    if isinstance(body, dict):
        return httpx.Response(status_code, request=request, json=body)
    return httpx.Response(status_code, request=request, text=body)


def test_post_does_not_retry_non_retryable_400():
    client = _StubClient(
        [
            _response(
                400,
                body={"error": {"message": "Requested token count exceeds the model's maximum context length."}},
            )
        ]
    )

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_post(client, "http://test/post", {}, max_retries=5))

    assert client.calls == 1


def test_post_retries_retryable_503_then_succeeds():
    client = _StubClient(
        [
            _response(503, body={"error": {"message": "No available workers"}}),
            _response(200, body={"ok": True}),
        ]
    )

    result = asyncio.run(_post(client, "http://test/post", {}, max_retries=5))

    assert result == {"ok": True}
    assert client.calls == 2


def test_internal_http_client_does_not_inherit_ambient_proxy(monkeypatch):
    captured: dict = {}

    class _Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    args = type(
        "Args",
        (),
        {
            "rollout_num_gpus": 2,
            "rollout_num_gpus_per_engine": 1,
            "sglang_server_concurrency": 4,
            "use_distributed_post": False,
        },
    )()
    monkeypatch.setattr(http_utils, "_http_client", None)
    monkeypatch.setattr(http_utils.httpx, "AsyncClient", _Client)

    init_http_client(args)

    assert captured["trust_env"] is False


@pytest.mark.parametrize(
    ("worker_url", "expected"),
    [
        ("http://worker:8000", "http://worker:8000"),
        ("http://worker:8000@0", "http://worker:8000"),
        ("http://[2001:db8::1]:8000@12", "http://[2001:db8::1]:8000"),
        ("http://user:password@worker:8000", "http://user:password@worker:8000"),
        ("http://user:p%40ss@worker:8000@2", "http://user:p%40ss@worker:8000"),
        ("http://user@123", "http://user@123"),
        ("http://worker:8000@rank0", "http://worker:8000@rank0"),
        ("http://worker:8000@-1", "http://worker:8000@-1"),
        ("http://worker:8000@²", "http://worker:8000@²"),
        ("http://worker:8000/path@0", "http://worker:8000/path@0"),
        ("http://worker:8000?rank=@0", "http://worker:8000?rank=@0"),
        ("", ""),
    ],
)
def test_router_worker_base_url(worker_url, expected):
    assert router_worker_base_url(worker_url) == expected


def test_router_worker_base_urls_stably_deduplicates_dp_ranks():
    urls = [
        "http://worker-a:8000@0",
        "http://worker-b:8000@0",
        "http://worker-a:8000@1",
        "http://worker-b:8000@1",
    ]

    assert router_worker_base_urls(urls) == [
        "http://worker-a:8000",
        "http://worker-b:8000",
    ]
