# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pluggable text-search backends for the DeepEyes V2 example."""

from __future__ import annotations

import copy
import math
import os
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote, urlsplit

import httpx
import yaml


DEFAULT_SEARCH_CONFIG: dict[str, Any] = {
    "backend": "mock",
    "top_k": 5,
    "timeout_s": 5.0,
    "max_retries": 2,
    "backoff_s": 0.25,
    "trust_env": False,
    "max_observation_chars": 12000,
}
ENV_OVERRIDES = {
    "DEEPEYES_V2_WEB_SEARCH_BACKEND": ("backend", str),
    "DEEPEYES_V2_WEB_SEARCH_TOP_K": ("top_k", int),
    "DEEPEYES_V2_WEB_SEARCH_TIMEOUT_S": ("timeout_s", float),
    "DEEPEYES_V2_WEB_SEARCH_MAX_RETRIES": ("max_retries", int),
}
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class SearchConfigurationError(ValueError):
    """The selected search backend is not configured correctly."""


class SearchBackendError(RuntimeError):
    """A search request failed or returned an invalid response."""


class ResolvedSearchConfig(dict[str, Any]):
    """Validated snapshot that does not re-read configuration files."""


def resolve_search_config(config: Mapping[str, Any] | None = None) -> ResolvedSearchConfig:
    """Load, merge, and validate one search configuration."""
    source: Mapping[str, Any] = config or {}
    config_path = os.environ.get("DEEPEYES_V2_WEB_SEARCH_CONFIG")
    if config_path is not None:
        if not config_path.strip():
            raise SearchConfigurationError("DEEPEYES_V2_WEB_SEARCH_CONFIG must not be empty")
        path = Path(config_path).expanduser()
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise SearchConfigurationError(f"unable to read web-search config: {type(exc).__name__}") from exc
        if not isinstance(loaded, dict):
            raise SearchConfigurationError("web-search config root must be a mapping")
        source = loaded

    resolved = copy.deepcopy(DEFAULT_SEARCH_CONFIG)
    resolved.update(copy.deepcopy(dict(source)))
    for env_name, (key, converter) in ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        if not raw.strip():
            raise SearchConfigurationError(f"{env_name} must not be empty")
        try:
            resolved[key] = converter(raw)
        except ValueError as exc:
            raise SearchConfigurationError(f"{env_name} has an invalid value") from exc

    url_override = os.environ.get("DEEPEYES_V2_WEB_SEARCH_URL")
    if url_override is not None:
        if not url_override.strip():
            raise SearchConfigurationError("DEEPEYES_V2_WEB_SEARCH_URL must not be empty")
        section_name = "retriever" if resolved.get("backend") == "retriever" else "external"
        section = resolved.get(section_name)
        if not isinstance(section, dict):
            raise SearchConfigurationError(f"{section_name} config must be a mapping")
        section["url" if section_name == "retriever" else "endpoint"] = url_override

    _validate_config(resolved)
    return ResolvedSearchConfig(resolved)


def _validate_config(config: Mapping[str, Any]) -> None:
    backend = config.get("backend")
    if backend not in {"mock", "retriever", "external"}:
        raise SearchConfigurationError("backend must be one of mock, retriever, external")
    _validate_int(config.get("top_k"), "top_k", 1, 50)
    _validate_int(config.get("max_retries"), "max_retries", 0, 10)
    _validate_int(config.get("max_observation_chars"), "max_observation_chars", 1024, 65536)
    _validate_number(config.get("timeout_s"), "timeout_s", positive=True)
    _validate_number(config.get("backoff_s"), "backoff_s")
    if not isinstance(config.get("trust_env"), bool):
        raise SearchConfigurationError("trust_env must be boolean")

    if backend == "retriever":
        retriever = config.get("retriever")
        if not isinstance(retriever, dict):
            raise SearchConfigurationError("retriever config must be a mapping")
        _validate_url(retriever.get("url"), "retriever.url", allow_http=True)
    elif backend == "external":
        _validate_external(config.get("external"))


def _validate_external(external: Any) -> None:
    if not isinstance(external, dict):
        raise SearchConfigurationError("external config must be a mapping")
    _validate_url(external.get("endpoint"), "external.endpoint", allow_loopback_http=True)
    if external.get("method", "POST") not in {"GET", "POST"}:
        raise SearchConfigurationError("external.method must be GET or POST")
    request_fields = external.get("request_fields")
    result_fields = external.get("result_fields")
    if not isinstance(request_fields, dict):
        raise SearchConfigurationError("external.request_fields must be a mapping")
    if not isinstance(result_fields, dict):
        raise SearchConfigurationError("external.result_fields must be a mapping")
    for label, value in (
        ("external.request_fields.query", request_fields.get("query")),
        ("external.request_fields.size", request_fields.get("size")),
        ("external.results_path", external.get("results_path")),
        ("external.result_fields.title", result_fields.get("title")),
        ("external.result_fields.link", result_fields.get("link")),
        ("external.result_fields.snippet", result_fields.get("snippet")),
    ):
        _validate_path(value, label)
    if request_fields["query"] == request_fields["size"]:
        raise SearchConfigurationError("external query and size fields must be distinct")
    if "date" in result_fields:
        _validate_path(result_fields["date"], "external.result_fields.date")
    if not isinstance(external.get("static_fields", {}), dict):
        raise SearchConfigurationError("external.static_fields must be a mapping")
    auth = external.get("auth")
    if auth is not None:
        if not isinstance(auth, dict):
            raise SearchConfigurationError("external.auth must be a mapping")
        env_name = auth.get("env")
        header = auth.get("header")
        if not isinstance(env_name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name) is None:
            raise SearchConfigurationError("external.auth.env must be a valid environment variable name")
        if not isinstance(header, str) or not header.strip():
            raise SearchConfigurationError("external.auth.header must be a non-empty string")
        if not isinstance(auth.get("prefix", ""), str):
            raise SearchConfigurationError("external.auth.prefix must be a string")


def _validate_int(value: Any, label: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise SearchConfigurationError(f"{label} must be an integer in [{minimum}, {maximum}]")


def _validate_number(value: Any, label: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SearchConfigurationError(f"{label} must be a finite number")
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "non-negative"
        raise SearchConfigurationError(f"{label} must be {qualifier}")


def _validate_path(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value or any(not part for part in value.split(".")):
        raise SearchConfigurationError(f"{label} must be a non-empty dotted path")


def _validate_url(
    value: Any,
    label: str,
    *,
    allow_http: bool = False,
    allow_loopback_http: bool = False,
) -> None:
    if not isinstance(value, str) or any(character.isspace() for character in value):
        raise SearchConfigurationError(f"{label} must be an HTTP(S) URL")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SearchConfigurationError(f"{label} must be an HTTP(S) URL")
    is_loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme == "http" and not allow_http and not (allow_loopback_http and is_loopback):
        raise SearchConfigurationError(f"{label} must use HTTPS except for loopback testing")


def validate_query_and_size(query: Any, size: Any) -> tuple[str, int]:
    if not isinstance(query, str) or not query.strip():
        raise SearchConfigurationError("query must be a non-empty string")
    query = query.strip()
    if len(query) > 8192:
        raise SearchConfigurationError("query exceeds 8192 characters")
    _validate_int(size, "size", 1, 50)
    return query, size


class SearchSession:
    """Per-agent search runtime that reuses one HTTP connection pool."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config if isinstance(config, ResolvedSearchConfig) else resolve_search_config(config)
        self._monotonic = monotonic
        self._sleep = sleep
        self._closed = False
        self._client = None
        if self.config["backend"] != "mock":
            self._client = client_factory(
                timeout=httpx.Timeout(self.config["timeout_s"]),
                follow_redirects=False,
                trust_env=self.config["trust_env"],
            )

    def search(self, query: str, size: int | None = None) -> dict[str, Any]:
        if self._closed:
            raise SearchBackendError("search session is closed")
        return run_search(
            query,
            size=size,
            config=self.config,
            monotonic=self._monotonic,
            sleep=self._sleep,
            client=self._client,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            self._client.close()

    def __enter__(self) -> SearchSession:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def run_search(
    query: str,
    *,
    size: int | None = None,
    config: Mapping[str, Any] | None = None,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    resolved = config if isinstance(config, ResolvedSearchConfig) else resolve_search_config(config)
    query, size = validate_query_and_size(query, resolved["top_k"] if size is None else size)
    if resolved["backend"] == "mock":
        return _mock_search(query, size)

    started = monotonic()
    if resolved["backend"] == "retriever":
        payload = {"queries": [query], "topk": size, "return_scores": True}
        response = _request_json(
            url=resolved["retriever"]["url"],
            method="POST",
            payload=payload,
            headers={},
            config=resolved,
            client_factory=client_factory,
            sleep=sleep,
            client=client,
        )
        rows = _normalise_retriever(response, size)
    else:
        rows = _external_search(
            query,
            size,
            resolved,
            client_factory=client_factory,
            sleep=sleep,
            client=client,
        )
    return {"elapsed_time": max(0.0, monotonic() - started), "data": rows}


def _mock_search(query: str, size: int) -> dict[str, Any]:
    encoded = quote(query, safe="")
    return {
        "elapsed_time": 0.0,
        "data": [
            {
                "title": f"Offline mock result {index + 1}",
                "link": f"https://example.invalid/search/{index + 1}?q={encoded}",
                "snippet": f"Deterministic offline result for: {query}",
                "date": None,
            }
            for index in range(size)
        ],
    }


def _external_search(
    query: str,
    size: int,
    config: Mapping[str, Any],
    **request_kwargs: Any,
) -> list[dict[str, Any]]:
    external = config["external"]
    request_fields = external["request_fields"]
    payload = dict(external.get("static_fields", {}))
    payload[request_fields["query"]] = query
    payload[request_fields["size"]] = size
    response = _request_json(
        url=external["endpoint"],
        method=external.get("method", "POST"),
        payload=payload,
        headers=_external_headers(external),
        config=config,
        **request_kwargs,
    )
    raw_rows = _dotted_get(response, external["results_path"])
    if not isinstance(raw_rows, list):
        raise SearchBackendError("external results path does not contain a list")
    return [_normalise_external_row(row, external["result_fields"]) for row in raw_rows[:size]]


def _external_headers(external: Mapping[str, Any]) -> dict[str, str]:
    auth = external.get("auth")
    if auth is None:
        return {}
    secret = os.environ.get(auth["env"])
    if not secret or "\n" in secret or "\r" in secret:
        raise SearchConfigurationError("external authentication value is missing or invalid")
    return {auth["header"]: f"{auth.get('prefix', '')}{secret}"}


def _request_json(
    *,
    url: str,
    method: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    config: Mapping[str, Any],
    client_factory: Callable[..., httpx.Client],
    sleep: Callable[[float], None],
    client: httpx.Client | None = None,
) -> Any:
    client_context = (
        nullcontext(client)
        if client is not None
        else client_factory(
            timeout=httpx.Timeout(config["timeout_s"]),
            follow_redirects=False,
            trust_env=config["trust_env"],
        )
    )
    last_error = "unknown"
    with client_context as active_client:
        for attempt in range(config["max_retries"] + 1):
            try:
                request_kwargs: dict[str, Any] = {"headers": dict(headers)}
                request_kwargs["params" if method == "GET" else "json"] = dict(payload)
                response = active_client.request(method, url, **request_kwargs)
                if 300 <= response.status_code < 400:
                    raise SearchBackendError("redirect responses are not accepted")
                if response.status_code >= 400:
                    if response.status_code not in RETRYABLE_STATUS_CODES:
                        raise SearchBackendError(f"non-retryable HTTP status {response.status_code}")
                    last_error = f"http_{response.status_code}"
                else:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise SearchBackendError("search response is not valid JSON") from exc
            except SearchBackendError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = type(exc).__name__
            except httpx.HTTPError as exc:
                raise SearchBackendError(f"search HTTP failure: {type(exc).__name__}") from exc
            if attempt < config["max_retries"]:
                sleep(config["backoff_s"] * (2**attempt))
    raise SearchBackendError(f"search request failed after retries ({last_error})")


def _normalise_retriever(data: Any, size: int) -> list[dict[str, Any]]:
    if not isinstance(data, dict) or not isinstance(data.get("result"), list) or len(data["result"]) != 1:
        raise SearchBackendError("retriever response must contain one result batch")
    raw_rows = data["result"][0]
    if not isinstance(raw_rows, list):
        raise SearchBackendError("retriever result batch must be a list")
    rows: list[dict[str, Any]] = []
    for raw in raw_rows[:size]:
        if not isinstance(raw, dict):
            raise SearchBackendError("retriever result item must be a mapping")
        document = raw.get("document", raw)
        if not isinstance(document, dict):
            raise SearchBackendError("retriever document must be a mapping")
        snippet = _optional_string(document, "contents") or _optional_string(document, "text") or ""
        title = _optional_string(document, "title") or next(
            (line.strip() for line in snippet.splitlines() if line.strip()),
            None,
        )
        if not title:
            raise SearchBackendError("retriever document has no usable title or text")
        link = _optional_string(document, "url") or _optional_string(document, "link") or ""
        if link:
            _validate_result_link(link)
        rows.append(
            {
                "title": title,
                "link": link,
                "snippet": snippet,
                "date": _normalise_date(document.get("date")),
            }
        )
    return rows


def _normalise_external_row(raw: Any, fields: Mapping[str, str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SearchBackendError("external result item must be a mapping")
    title = _dotted_get(raw, fields["title"])
    link = _dotted_get(raw, fields["link"])
    snippet = _dotted_get(raw, fields["snippet"])
    if not isinstance(title, str) or not title.strip():
        raise SearchBackendError("external result title must be a non-empty string")
    if not isinstance(link, str) or not link.strip():
        raise SearchBackendError("external result link must be a non-empty string")
    if not isinstance(snippet, str):
        raise SearchBackendError("external result snippet must be a string")
    _validate_result_link(link)
    date_path = fields.get("date")
    date = _normalise_date(_dotted_get(raw, date_path, missing_ok=True)) if date_path else None
    return {"title": title.strip(), "link": link.strip(), "snippet": snippet, "date": date}


def _optional_string(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SearchBackendError(f"{key} must be a string when present")
    return value.strip() or None


def _normalise_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise SearchBackendError("result date must be a string or null")
    return value


def _validate_result_link(link: str) -> None:
    parsed = urlsplit(link)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SearchBackendError("result link must be an HTTP(S) URL")


def _dotted_get(mapping: Any, path: str, *, missing_ok: bool = False) -> Any:
    current = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            if missing_ok:
                return None
            raise SearchBackendError(f"response path is missing: {path}")
        current = current[part]
    return current
