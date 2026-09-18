# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import format_datetime
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
    _read_limited,
    _retry_after_seconds,
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
        "retry_budget_s": 5,
        "backoff_initial_s": 0,
        "backoff_max_s": 0,
        "trust_env": False,
        "max_response_bytes": 4096,
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
    first = run_search("same query")
    second = run_search("same query")
    assert first == second
    assert first["elapsed_time"] == 0.0
    assert len(first["data"]) == 5
    assert set(first["data"][0]) == {"title", "link", "snippet", "date"}


def test_default_mock_never_constructs_an_http_client():
    def forbidden_client(**_kwargs):
        raise AssertionError("mock search attempted to construct an HTTP client")

    assert run_search("offline", config={"backend": "mock"}, client_factory=forbidden_client)["data"]


def test_config_precedence_file_env_and_explicit_size(tmp_path, monkeypatch):
    config_file = tmp_path / "search config.yaml"
    config_file.write_text("backend: mock\ntop_k: 2\ntimeout_s: 3\n", encoding="utf-8")
    monkeypatch.setenv("DEEPEYES_V2_WEB_SEARCH_CONFIG", str(config_file))
    monkeypatch.setenv("DEEPEYES_V2_WEB_SEARCH_TOP_K", "3")
    resolved = resolve_search_config({"backend": "mock", "top_k": 1})
    assert resolved["top_k"] == 3
    assert len(run_search("query", size=4)["data"]) == 4


def test_url_override_reports_malformed_backend_section(monkeypatch):
    monkeypatch.setenv("DEEPEYES_V2_WEB_SEARCH_URL", "http://127.0.0.1:8000/retrieve")
    with pytest.raises(SearchConfigurationError, match="retriever config"):
        resolve_search_config({"backend": "retriever", "retriever": "not-a-mapping"})


def test_external_auth_sources_are_strictly_mutually_exclusive():
    config = _external_config()
    config["external"]["auth"] = {"header": "Authorization", "env": "", "file": "token.txt"}
    with pytest.raises(SearchConfigurationError, match="exactly one"):
        resolve_search_config(config)


def test_external_mapping_paths_are_validated_before_requests():
    config = _external_config()
    config["external"]["result_fields"]["date"] = 42
    with pytest.raises(SearchConfigurationError, match="result_fields.date"):
        resolve_search_config(config)


@pytest.mark.parametrize(
    ("query", "size"),
    [("", 1), (None, 1), ("x" * 8193, 1), ("query", True), ("query", 0), ("query", 51)],
)
def test_query_and_size_validation(query, size):
    with pytest.raises(SearchConfigurationError):
        run_search(query, size=size, config={"backend": "mock"})


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
def test_invalid_config_fails_closed(config):
    with pytest.raises(SearchConfigurationError):
        resolve_search_config(config)
    assert search("query", config=config) == "Error"


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
    assert result["data"][0]["title"] == "Title from contents"
    assert result["data"][1]["link"] == ""


def test_external_mapping_and_auth_are_config_driven(monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "not-a-real-secret")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers["X-Api-Key"]
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "payload": {
                    "items": [
                        {
                            "name": "Result",
                            "url": "https://result.test/page",
                            "description": "Useful text",
                            "published": "2026-09-17",
                        }
                    ]
                }
            },
        )

    result = run_search(
        "test mapping",
        size=1,
        config=_external_config(),
        client_factory=_client_factory(handler),
    )
    assert seen["authorization"] == "not-a-real-secret"
    assert seen["payload"] == {"mode": "basic", "q": "test mapping", "limit": 1}
    assert result["data"] == [
        {
            "title": "Result",
            "link": "https://result.test/page",
            "snippet": "Useful text",
            "date": "2026-09-17",
        }
    ]


def test_external_get_mapping_uses_query_parameters(monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    config = _external_config()
    config["external"]["method"] = "GET"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == "get mapping"
        assert request.url.params["limit"] == "1"
        assert request.url.params["mode"] == "basic"
        return httpx.Response(
            200,
            json={"payload": {"items": [{"name": "GET", "url": "https://result.test/get", "description": "ok"}]}},
        )

    result = run_search("get mapping", size=1, config=config, client_factory=_client_factory(handler))
    assert result["data"][0]["title"] == "GET"


def test_relative_auth_file_requires_private_permissions(tmp_path, monkeypatch):
    token = tmp_path / "token.txt"
    token.write_text("file-secret\n", encoding="utf-8")
    token.chmod(0o600)
    config = _external_config()
    config["external"]["auth"] = {"header": "Authorization", "prefix": "Bearer ", "file": "token.txt"}
    config_file = tmp_path / "search.yaml"
    import yaml

    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("DEEPEYES_V2_WEB_SEARCH_CONFIG", str(config_file))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer file-secret"
        return httpx.Response(200, json={"payload": {"items": []}})

    assert run_search("file auth", client_factory=_client_factory(handler))["data"] == []
    token.chmod(0o644)
    with pytest.raises(SearchConfigurationError, match="group or others"):
        run_search("file auth", client_factory=_client_factory(handler))


def test_retryable_status_recovers_without_sleeping(monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"Retry-After": "0"}, text="busy")
        return httpx.Response(
            200,
            json={"payload": {"items": [{"name": "Recovered", "url": "https://result.test", "description": "ok"}]}},
        )

    result = run_search(
        "retry",
        size=1,
        config=_external_config(),
        client_factory=_client_factory(handler),
        sleep=lambda _: None,
    )
    assert calls == 2
    assert result["data"][0]["title"] == "Recovered"


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_all_retryable_statuses_retry_once_then_recover(monkeypatch, status):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"payload": {"items": []}})

    assert (
        run_search(
            "retry status",
            config=_external_config(),
            client_factory=_client_factory(handler),
            sleep=lambda _: None,
        )["data"]
        == []
    )
    assert calls == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_non_retryable_statuses_fail_after_one_request(monkeypatch, status):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status)

    with pytest.raises(SearchBackendError, match="non-retryable"):
        run_search("permanent", config=_external_config(), client_factory=_client_factory(handler))
    assert calls == 1


def test_retry_after_http_date_and_budget_are_respected(monkeypatch):
    now = datetime(2026, 9, 17, tzinfo=timezone.utc)
    assert _retry_after_seconds(format_datetime(now), now.timestamp()) == 0
    assert _retry_after_seconds(format_datetime(now.replace(second=10)), now.timestamp()) == 10
    assert _retry_after_seconds("not-a-date", now.timestamp()) == 0

    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "30"})

    with pytest.raises(SearchBackendError, match="after retries"):
        run_search("budget", config=_external_config(), client_factory=_client_factory(handler))
    assert calls == 1


def test_retry_uses_one_credential_snapshot(monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "first-token")
    seen_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers["X-Api-Key"])
        if len(seen_headers) == 1:
            monkeypatch.setenv("TEST_SEARCH_TOKEN", "rotated-token")
            return httpx.Response(503)
        return httpx.Response(200, json={"payload": {"items": []}})

    run_search(
        "credential snapshot",
        config=_external_config(),
        client_factory=_client_factory(handler),
        sleep=lambda _: None,
    )
    assert seen_headers == ["first-token", "first-token"]


def test_new_logical_search_observes_rotated_credential(monkeypatch):
    seen_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers["X-Api-Key"])
        return httpx.Response(200, json={"payload": {"items": []}})

    factory = _client_factory(handler)
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "first-token")
    run_search("first", config=_external_config(), client_factory=factory)
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "rotated-token")
    run_search("second", config=_external_config(), client_factory=factory)
    assert seen_headers == ["first-token", "rotated-token"]


def test_timeout_retries_are_bounded(monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(SearchBackendError, match="after retries"):
        run_search(
            "timeout",
            config=_external_config(),
            client_factory=_client_factory(handler),
            sleep=lambda _: None,
        )
    assert calls == 2


@pytest.mark.parametrize(
    "error_type",
    [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout],
)
def test_all_httpx_timeout_phases_use_bounded_retries(monkeypatch, error_type):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise error_type("slow", request=request)

    with pytest.raises(SearchBackendError, match="after retries"):
        run_search(
            "timeout phase",
            config=_external_config(),
            client_factory=_client_factory(handler),
            sleep=lambda _: None,
        )
    assert calls == 2


def test_response_size_limit_is_enforced(monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    config = _external_config()
    config["max_response_bytes"] = 8
    with pytest.raises(SearchBackendError, match="size limit"):
        run_search(
            "large",
            config=config,
            client_factory=_client_factory(lambda _request: httpx.Response(200, json={"payload": {"items": []}})),
        )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(302, headers={"Location": "https://other.test"}), "redirect"),
        (httpx.Response(200, content=b"not-json"), "valid JSON"),
        (httpx.Response(200, content=b""), "empty"),
    ],
)
def test_bad_http_responses_raise_backend_error(monkeypatch, response, expected):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    with pytest.raises(SearchBackendError, match=expected):
        run_search(
            "bad response",
            config=_external_config(),
            client_factory=_client_factory(lambda _request: response),
        )


def test_missing_auth_is_error_without_exposing_variable_value(monkeypatch):
    monkeypatch.delenv("TEST_SEARCH_TOKEN", raising=False)
    with pytest.raises(SearchConfigurationError, match="missing or invalid") as error:
        run_search("secret", config=_external_config())
    assert "TEST_SEARCH_TOKEN" not in str(error.value)


def test_public_search_boundary_converts_backend_failures_to_error(monkeypatch):
    def fail(*_args, **_kwargs):
        raise SearchBackendError("private upstream detail")

    monkeypatch.setattr(search_utils_module, "run_search", fail)
    assert search("query") == "Error"


def test_search_session_reuses_client_and_caches_success(monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    calls = 0
    clients = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"payload": {"items": [{"name": "Cached", "url": "https://result.test", "description": "ok"}]}},
        )

    def factory(**kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    config = _external_config()
    config.update(cache_ttl_s=60, cache_max_entries=2)
    session = SearchSession(config, client_factory=factory)
    try:
        first = session.search("same", size=1)
        second = session.search("same", size=1)
        assert first["data"] == second["data"]
        assert second["elapsed_time"] == 0.0
        assert calls == 1
        assert len(clients) == 1
        assert session.snapshot_metrics()["cache_hit_count"] == 1
    finally:
        session.close()
    assert clients[0].is_closed


def test_search_session_circuit_breaker_opens_and_recovers(monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    now = [0.0]
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    config = _external_config()
    config.update(max_retries=0, circuit_breaker_failures=2, circuit_breaker_reset_s=10)
    session = SearchSession(
        config,
        client_factory=_client_factory(handler),
        monotonic=lambda: now[0],
        sleep=lambda _delay: None,
    )
    try:
        with pytest.raises(SearchBackendError, match="after retries"):
            session.search("first")
        with pytest.raises(SearchBackendError, match="after retries"):
            session.search("second")
        with pytest.raises(SearchBackendError, match="circuit breaker"):
            session.search("blocked")
        assert calls == 2
        now[0] = 11
        with pytest.raises(SearchBackendError, match="after retries"):
            session.search("half open")
        assert calls == 3
        assert session.snapshot_metrics()["circuit_open_count"] == 1
    finally:
        session.close()


def test_search_session_allows_only_one_half_open_recovery_probe(monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_TOKEN", "token")
    now = [0.0]
    recovery_entered = threading.Event()
    release_recovery = threading.Event()
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        recovery_entered.set()
        assert release_recovery.wait(timeout=2)
        return httpx.Response(200, json={"payload": {"items": []}})

    config = _external_config()
    config.update(max_retries=0, circuit_breaker_failures=1, circuit_breaker_reset_s=10)
    session = SearchSession(config, client_factory=_client_factory(handler), monotonic=lambda: now[0])
    try:
        with pytest.raises(SearchBackendError):
            session.search("open")
        now[0] = 11
        with ThreadPoolExecutor(max_workers=1) as executor:
            probe = executor.submit(session.search, "probe")
            assert recovery_entered.wait(timeout=2)
            with pytest.raises(SearchBackendError, match="awaiting its recovery probe"):
                session.search("blocked")
            release_recovery.set()
            assert probe.result(timeout=2)["data"] == []
        assert calls == 2
    finally:
        release_recovery.set()
        session.close()


def test_stream_reader_enforces_total_deadline_between_chunks():
    now = [0.0]

    class SlowResponse:
        def iter_bytes(self):
            yield b"{"
            now[0] = 2.0
            yield b"}"

    with pytest.raises(SearchBackendError, match="total request deadline"):
        _read_limited(SlowResponse(), 100, deadline=1.0, monotonic=lambda: now[0])
