# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepeyes_v2_agentic"
sys.path.insert(0, str(EXAMPLE_DIR))

import app.search_utils as search_utils_module  # noqa: E402
from app.search_backends import (  # noqa: E402
    SearchBackendError,
    SearchConfigurationError,
    SearchSession,
    resolve_search_config,
    run_search,
)
from app.search_utils import search  # noqa: E402


def _client_factory(handler):
    def factory(**kwargs):
        return httpx.Client(transport=httpx.MockTransport(handler), **kwargs)

    return factory


def _external_config(endpoint: str = "https://search.test/v1") -> dict:
    return {
        "backend": "external",
        "top_k": 4,
        "timeout_s": 1,
        "max_retries": 1,
        "backoff_s": 0,
        "trust_env": False,
        "max_observation_chars": 4096,
        "external": {
            "endpoint": endpoint,
            "method": "POST",
            "request_fields": {"query": "q", "size": "limit"},
            "static_fields": {"mode": "basic"},
            "results_path": "payload.items",
            "result_fields": {
                "title": "name",
                "link": "url",
                "snippet": "description",
                "date": "published",
            },
            "auth": {"header": "X-Api-Key", "env": "TEST_SEARCH_TOKEN"},
        },
    }


def test_default_mock_is_deterministic_and_offline(monkeypatch):
    for name in (
        "DEEPEYES_V2_WEB_SEARCH_CONFIG",
        "DEEPEYES_V2_WEB_SEARCH_BACKEND",
        "DEEPEYES_V2_WEB_SEARCH_URL",
        "DEEPEYES_V2_WEB_SEARCH_TOP_K",
        "DEEPEYES_V2_WEB_SEARCH_TIMEOUT_S",
        "DEEPEYES_V2_WEB_SEARCH_MAX_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)

    def forbidden_client(**_kwargs):
        raise AssertionError("mock search attempted to construct an HTTP client")

    first = run_search("same query", client_factory=forbidden_client)
    second = run_search("same query", client_factory=forbidden_client)
    assert first == second
    assert first["elapsed_time"] == 0.0
    assert len(first["data"]) == 5
    assert set(first["data"][0]) == {"title", "link", "snippet", "date"}


def test_config_file_then_environment_override(tmp_path, monkeypatch):
    config_file = tmp_path / "search.yaml"
    config_file.write_text("backend: mock\ntop_k: 2\n", encoding="utf-8")
    monkeypatch.setenv("DEEPEYES_V2_WEB_SEARCH_CONFIG", str(config_file))
    monkeypatch.setenv("DEEPEYES_V2_WEB_SEARCH_TOP_K", "3")
    assert resolve_search_config()["top_k"] == 3


@pytest.mark.parametrize(
    "config",
    [
        {"backend": "unknown"},
        {"backend": "mock", "timeout_s": float("nan")},
        {"backend": "retriever", "retriever": {}},
        {
            "backend": "external",
            "external": {
                "endpoint": "http://not-loopback.test/search",
                "request_fields": {"query": "q", "size": "n"},
                "results_path": "results",
                "result_fields": {"title": "title", "link": "url", "snippet": "text"},
            },
        },
    ],
)
def test_invalid_config_fails_at_boundary(config):
    with pytest.raises(SearchConfigurationError):
        resolve_search_config(config)
    assert search("query", config=config) == "Error"


@pytest.mark.parametrize(
    ("query", "size"),
    [("", 1), (None, 1), ("x" * 8193, 1), ("query", True), ("query", 0), ("query", 51)],
)
def test_invalid_query_or_size_is_rejected(query, size):
    with pytest.raises(SearchConfigurationError):
        run_search(query, size=size, config={"backend": "mock"})


def test_retriever_request_and_normalization():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "result": [
                    [
                        {"document": {"contents": "Title from contents\nBody", "url": "https://a.test"}},
                        {"document": {"title": "No URL", "text": "Snippet"}},
                    ]
                ]
            },
        )

    config = {"backend": "retriever", "top_k": 2, "retriever": {"url": "http://retriever:8000/retrieve"}}
    result = run_search("capital of France", config=config, client_factory=_client_factory(handler))
    assert seen["payload"] == {"queries": ["capital of France"], "topk": 2, "return_scores": True}
    assert result["data"] == [
        {
            "title": "Title from contents",
            "link": "https://a.test",
            "snippet": "Title from contents\nBody",
            "date": None,
        },
        {"title": "No URL", "link": "", "snippet": "Snippet", "date": None},
    ]


def test_external_mapping_and_auth_are_config_driven(monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "not-a-real-secret")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Api-Key"] == "not-a-real-secret"
        assert json.loads(request.content) == {"mode": "basic", "q": "mapped query", "limit": 1}
        return httpx.Response(
            200,
            json={
                "payload": {
                    "items": [
                        {
                            "name": "Mapped",
                            "url": "https://result.test/item",
                            "description": "evidence",
                            "published": None,
                        }
                    ]
                }
            },
        )

    result = run_search(
        "mapped query",
        size=1,
        config=_external_config(),
        client_factory=_client_factory(handler),
    )
    assert result["data"] == [
        {
            "title": "Mapped",
            "link": "https://result.test/item",
            "snippet": "evidence",
            "date": None,
        }
    ]


def test_external_get_uses_query_parameters(monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    config = _external_config()
    config["external"]["method"] = "GET"

    def handler(request: httpx.Request) -> httpx.Response:
        assert dict(request.url.params) == {"mode": "basic", "q": "query", "limit": "1"}
        return httpx.Response(200, json={"payload": {"items": []}})

    assert run_search("query", size=1, config=config, client_factory=_client_factory(handler))["data"] == []


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_retryable_status_retries_then_recovers(monkeypatch, status):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status)
        return httpx.Response(200, json={"payload": {"items": []}})

    result = run_search(
        "retry",
        config=_external_config(),
        client_factory=_client_factory(handler),
        sleep=lambda _delay: None,
    )
    assert result["data"] == []
    assert calls == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_non_retryable_status_fails_once(monkeypatch, status):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status)

    with pytest.raises(SearchBackendError, match="non-retryable"):
        run_search("permanent", config=_external_config(), client_factory=_client_factory(handler))
    assert calls == 1


@pytest.mark.parametrize(
    "error_type",
    [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout],
)
def test_timeout_phases_have_bounded_retries(monkeypatch, error_type):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise error_type("slow", request=request)

    with pytest.raises(SearchBackendError, match="after retries"):
        run_search(
            "timeout",
            config=_external_config(),
            client_factory=_client_factory(handler),
            sleep=lambda _delay: None,
        )
    assert calls == 2


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(302, headers={"Location": "https://other.test"}), "redirect"),
        (httpx.Response(200, content=b"not-json"), "valid JSON"),
        (httpx.Response(200, content=b""), "valid JSON"),
    ],
)
def test_malformed_http_responses_fail(monkeypatch, response, expected):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    with pytest.raises(SearchBackendError, match=expected):
        run_search(
            "bad response",
            config=_external_config(),
            client_factory=_client_factory(lambda _request: response),
        )


def test_missing_auth_is_non_secret_error(monkeypatch):
    monkeypatch.delenv("TEST_SEARCH_TOKEN", raising=False)
    with pytest.raises(SearchConfigurationError, match="missing or invalid") as error:
        run_search("secret", config=_external_config())
    assert "TEST_SEARCH_TOKEN" not in str(error.value)


def test_public_search_boundary_returns_error(monkeypatch):
    def fail(*_args, **_kwargs):
        raise SearchBackendError("private upstream detail")

    monkeypatch.setattr(search_utils_module, "run_search", fail)
    assert search("query") == "Error"


def test_search_session_reuses_and_closes_client(monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    clients = []
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"payload": {"items": []}})

    def factory(**kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    with SearchSession(_external_config(), client_factory=factory) as session:
        assert session.search("first")["data"] == []
        assert session.search("second")["data"] == []
    assert calls == 2
    assert len(clients) == 1
    assert clients[0].is_closed
