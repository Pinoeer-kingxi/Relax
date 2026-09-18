# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pluggable text-search backends for the DeepEyes V2 example."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import stat
import threading
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
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
    "retry_budget_s": 20.0,
    "backoff_initial_s": 0.25,
    "backoff_max_s": 2.0,
    "trust_env": False,
    "max_response_bytes": 1024 * 1024,
    "max_observation_chars": 12000,
    "cache_ttl_s": 0.0,
    "cache_max_entries": 128,
    "circuit_breaker_failures": 5,
    "circuit_breaker_reset_s": 30.0,
}
ENV_OVERRIDES = {
    "DEEPEYES_V2_WEB_SEARCH_BACKEND": ("backend", str),
    "DEEPEYES_V2_WEB_SEARCH_TOP_K": ("top_k", int),
    "DEEPEYES_V2_WEB_SEARCH_TIMEOUT_S": ("timeout_s", float),
    "DEEPEYES_V2_WEB_SEARCH_MAX_RETRIES": ("max_retries", int),
}
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class SearchConfigurationError(ValueError):
    """The selected search backend is not configured safely or completely."""


class SearchBackendError(RuntimeError):
    """A search request failed or returned an invalid response."""


class ResolvedSearchConfig(dict[str, Any]):
    """Validated per-session snapshot that must not re-read config files."""


def resolve_search_config(config: Mapping[str, Any] | None = None) -> ResolvedSearchConfig:
    """Resolve and validate a per-session search configuration snapshot."""
    source: Mapping[str, Any] = config or {}
    config_dir: Path | None = None
    config_path = os.environ.get("DEEPEYES_V2_WEB_SEARCH_CONFIG")
    if config_path is not None:
        if not config_path.strip():
            raise SearchConfigurationError("DEEPEYES_V2_WEB_SEARCH_CONFIG must not be empty")
        path = Path(config_path).expanduser().resolve()
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise SearchConfigurationError(f"unable to read web-search config: {type(exc).__name__}") from exc
        if not isinstance(loaded, dict):
            raise SearchConfigurationError("web-search config root must be a mapping")
        source = loaded
        config_dir = path.parent

    resolved = copy.deepcopy(DEFAULT_SEARCH_CONFIG)
    resolved.update(copy.deepcopy(dict(source)))
    if config_dir is not None:
        resolved["_config_dir"] = str(config_dir)

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
        backend = resolved.get("backend")
        if backend == "retriever":
            section = resolved.setdefault("retriever", {})
            if not isinstance(section, dict):
                raise SearchConfigurationError("retriever config must be a mapping")
            section["url"] = url_override
        elif backend == "external":
            section = resolved.setdefault("external", {})
            if not isinstance(section, dict):
                raise SearchConfigurationError("external config must be a mapping")
            section["endpoint"] = url_override

    _validate_config(resolved)
    return ResolvedSearchConfig(resolved)


def _validate_config(config: Mapping[str, Any]) -> None:
    backend = config.get("backend")
    if backend not in {"mock", "retriever", "external"}:
        raise SearchConfigurationError("backend must be one of mock, retriever, external")
    _validate_int(config, "top_k", minimum=1, maximum=50)
    _validate_int(config, "max_retries", minimum=0, maximum=10)
    _validate_int(config, "max_response_bytes", minimum=1, maximum=64 * 1024 * 1024)
    _validate_int(config, "max_observation_chars", minimum=1024, maximum=65536)
    _validate_int(config, "cache_max_entries", minimum=1, maximum=10000)
    _validate_int(config, "circuit_breaker_failures", minimum=1, maximum=1000)
    for key in (
        "timeout_s",
        "retry_budget_s",
        "backoff_initial_s",
        "backoff_max_s",
        "cache_ttl_s",
        "circuit_breaker_reset_s",
    ):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise SearchConfigurationError(f"{key} must be a finite non-negative number")
    if config["timeout_s"] <= 0 or config["retry_budget_s"] <= 0 or config["circuit_breaker_reset_s"] <= 0:
        raise SearchConfigurationError("timeout_s, retry_budget_s, and circuit_breaker_reset_s must be positive")
    if not isinstance(config.get("trust_env"), bool):
        raise SearchConfigurationError("trust_env must be boolean")

    if backend == "retriever":
        retriever = config.get("retriever")
        if not isinstance(retriever, dict):
            raise SearchConfigurationError("retriever config must be a mapping")
        _validate_http_url(retriever.get("url"), "retriever.url", allow_http=True)
    elif backend == "external":
        _validate_external_config(config)


def _validate_int(config: Mapping[str, Any], key: str, *, minimum: int, maximum: int) -> None:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise SearchConfigurationError(f"{key} must be an integer in [{minimum}, {maximum}]")


def _validate_external_config(config: Mapping[str, Any]) -> None:
    external = config.get("external")
    if not isinstance(external, dict):
        raise SearchConfigurationError("external config must be a mapping")
    endpoint = external.get("endpoint")
    _validate_http_url(endpoint, "external.endpoint", allow_loopback_http=True)
    parsed = urlsplit(endpoint)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SearchConfigurationError(
            "external.endpoint must not contain credentials, query parameters, or fragments"
        )
    method = external.get("method", "POST")
    if method not in {"GET", "POST"}:
        raise SearchConfigurationError("external.method must be GET or POST")
    request_fields = external.get("request_fields")
    if not isinstance(request_fields, dict):
        raise SearchConfigurationError("external.request_fields must be a mapping")
    query_field = request_fields.get("query")
    size_field = request_fields.get("size")
    if (
        not all(isinstance(item, str) and item and not _has_control_chars(item) for item in (query_field, size_field))
        or query_field == size_field
    ):
        raise SearchConfigurationError("external query and size fields must be distinct non-empty strings")
    static_fields = external.get("static_fields", {})
    if not isinstance(static_fields, dict):
        raise SearchConfigurationError("external.static_fields must be a mapping")
    try:
        json.dumps(static_fields)
    except (TypeError, ValueError) as exc:
        raise SearchConfigurationError("external.static_fields must be JSON serializable") from exc
    if query_field in static_fields or size_field in static_fields:
        raise SearchConfigurationError("external.static_fields must not override query or size")
    _validate_dotted_path(external.get("results_path"), "external.results_path")
    result_fields = external.get("result_fields")
    if not isinstance(result_fields, dict):
        raise SearchConfigurationError("external.result_fields must be a mapping")
    for key in ("title", "link", "snippet"):
        _validate_dotted_path(result_fields.get(key), f"external.result_fields.{key}")
    if "date" in result_fields:
        _validate_dotted_path(result_fields["date"], "external.result_fields.date")
    auth = external.get("auth")
    if auth is not None:
        if not isinstance(auth, dict):
            raise SearchConfigurationError("external.auth must be a mapping")
        sources = [key for key in ("env", "file") if key in auth]
        if len(sources) != 1:
            raise SearchConfigurationError("external.auth must configure exactly one of env or file")
        source = sources[0]
        source_value = auth[source]
        if not isinstance(source_value, str) or not source_value or _has_control_chars(source_value):
            raise SearchConfigurationError("external.auth source must be a non-empty string")
        if source == "env" and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", source_value) is None:
            raise SearchConfigurationError("external.auth.env must be a valid environment variable name")
        header = auth.get("header")
        if not isinstance(header, str) or not header or _has_control_chars(header):
            raise SearchConfigurationError("external.auth.header is invalid")
        prefix = auth.get("prefix", "")
        if not isinstance(prefix, str) or _has_control_chars(prefix):
            raise SearchConfigurationError("external.auth.prefix is invalid")


def _has_control_chars(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _validate_dotted_path(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value or any(not part for part in value.split(".")):
        raise SearchConfigurationError(f"{label} must be a non-empty dotted path")
    if _has_control_chars(value):
        raise SearchConfigurationError(f"{label} contains invalid control characters")


def _validate_http_url(
    value: Any,
    label: str,
    *,
    allow_http: bool = False,
    allow_loopback_http: bool = False,
) -> None:
    if not isinstance(value, str) or not value:
        raise SearchConfigurationError(f"{label} must be a non-empty URL")
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value):
        raise SearchConfigurationError(f"{label} contains invalid whitespace or control characters")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SearchConfigurationError(f"{label} must be an HTTP(S) URL")
    try:
        parsed.port
    except ValueError as exc:
        raise SearchConfigurationError(f"{label} contains an invalid port") from exc
    is_loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme != "https" and not allow_http and not (allow_loopback_http and is_loopback):
        raise SearchConfigurationError(f"{label} must use HTTPS except for loopback testing")


def validate_query_and_size(query: Any, size: Any) -> tuple[str, int]:
    if not isinstance(query, str) or not query.strip():
        raise SearchConfigurationError("query must be a non-empty string")
    query = query.strip()
    if len(query) > 8192:
        raise SearchConfigurationError("query exceeds 8192 characters")
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 50:
        raise SearchConfigurationError("size must be an integer in [1, 50]")
    return query, size


@dataclass
class _CacheEntry:
    expires_at: float
    result: dict[str, Any]


class SearchSession:
    """Per-agent search runtime with connection reuse, bounded TTL cache, and
    circuit breaking."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        self.config = config if isinstance(config, ResolvedSearchConfig) else resolve_search_config(config)
        _validate_config(self.config)
        self._monotonic = monotonic
        self._sleep = sleep
        self._wall_time = wall_time
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, int], _CacheEntry] = OrderedDict()
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._circuit_probe_in_progress = False
        self._closed = False
        self.metrics: dict[str, int] = {
            "search_count": 0,
            "cache_hit_count": 0,
            "failure_count": 0,
            "circuit_open_count": 0,
            "http_attempt_count": 0,
            "retry_count": 0,
        }
        self._client = None
        if self.config["backend"] != "mock":
            self._client = client_factory(
                timeout=httpx.Timeout(self.config["timeout_s"]),
                follow_redirects=False,
                trust_env=self.config["trust_env"],
            )

    def search(self, query: str, size: int | None = None) -> dict[str, Any]:
        query, size = validate_query_and_size(query, self.config["top_k"] if size is None else size)
        key = (query, size)
        now = self._monotonic()
        with self._lock:
            if self._closed:
                raise SearchBackendError("search session is closed")
            self.metrics["search_count"] += 1
            cached = self._cache.get(key)
            if cached is not None:
                if cached.expires_at > now:
                    self._cache.move_to_end(key)
                    self.metrics["cache_hit_count"] += 1
                    result = copy.deepcopy(cached.result)
                    result["elapsed_time"] = 0.0
                    return result
                del self._cache[key]
            if now < self._circuit_open_until:
                self.metrics["circuit_open_count"] += 1
                raise SearchBackendError("search circuit breaker is open")
            if self._circuit_open_until > 0:
                if self._circuit_probe_in_progress:
                    self.metrics["circuit_open_count"] += 1
                    raise SearchBackendError("search circuit breaker is awaiting its recovery probe")
                self._circuit_probe_in_progress = True

        try:
            result = run_search(
                query,
                size=size,
                config=self.config,
                monotonic=self._monotonic,
                sleep=self._sleep,
                wall_time=self._wall_time,
                client=self._client,
                request_metric_increment=self._increment_metric,
            )
        except SearchBackendError:
            with self._lock:
                self.metrics["failure_count"] += 1
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.config["circuit_breaker_failures"]:
                    self._circuit_open_until = self._monotonic() + self.config["circuit_breaker_reset_s"]
                self._circuit_probe_in_progress = False
            raise
        except Exception:
            with self._lock:
                self._circuit_probe_in_progress = False
            raise

        with self._lock:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
            self._circuit_probe_in_progress = False
            if self.config["cache_ttl_s"] > 0:
                self._cache[key] = _CacheEntry(
                    expires_at=self._monotonic() + self.config["cache_ttl_s"],
                    result=copy.deepcopy(result),
                )
                self._cache.move_to_end(key)
                while len(self._cache) > self.config["cache_max_entries"]:
                    self._cache.popitem(last=False)
        return result

    def snapshot_metrics(self) -> dict[str, int]:
        with self._lock:
            return dict(self.metrics)

    def _increment_metric(self, name: str) -> None:
        with self._lock:
            self.metrics[name] += 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cache.clear()
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
    wall_time: Callable[[], float] = time.time,
    client: httpx.Client | None = None,
    request_metric_increment: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if isinstance(config, ResolvedSearchConfig):
        resolved = dict(config)
        _validate_config(resolved)
    else:
        resolved = resolve_search_config(config)
    query, size = validate_query_and_size(query, resolved["top_k"] if size is None else size)
    started = monotonic()
    backend = resolved["backend"]
    if backend == "mock":
        return _mock_search(query, size)
    if backend == "retriever":
        payload = {"queries": [query], "topk": size, "return_scores": True}
        data = _request_json(
            url=resolved["retriever"]["url"],
            method="POST",
            payload=payload,
            headers={},
            config=resolved,
            client_factory=client_factory,
            monotonic=monotonic,
            sleep=sleep,
            wall_time=wall_time,
            client=client,
            request_metric_increment=request_metric_increment,
        )
        rows = _normalise_retriever(data, size)
    else:
        rows = _external_search(
            query,
            size,
            resolved,
            client_factory=client_factory,
            monotonic=monotonic,
            sleep=sleep,
            wall_time=wall_time,
            client=client,
            request_metric_increment=request_metric_increment,
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
    headers = _external_headers(external, config)
    data = _request_json(
        url=external["endpoint"],
        method=external.get("method", "POST"),
        payload=payload,
        headers=headers,
        config=config,
        **request_kwargs,
    )
    raw_rows = _dotted_get(data, external["results_path"])
    if not isinstance(raw_rows, list):
        raise SearchBackendError("external results path does not contain a list")
    fields = external["result_fields"]
    return [_normalise_external_row(row, fields) for row in raw_rows[:size]]


def _external_headers(external: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, str]:
    auth = external.get("auth")
    if auth is None:
        return {}
    if "env" in auth:
        secret = os.environ.get(auth["env"], "")
    else:
        path = Path(auth["file"]).expanduser()
        if not path.is_absolute():
            path = Path(config.get("_config_dir", os.getcwd())) / path
        try:
            if stat.S_IMODE(path.stat().st_mode) & 0o077:
                raise SearchConfigurationError("external auth file must not be accessible by group or others")
            secret = path.read_text(encoding="utf-8").rstrip("\r\n")
        except OSError as exc:
            raise SearchConfigurationError(f"unable to read external auth file: {type(exc).__name__}") from exc
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
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    wall_time: Callable[[], float],
    client: httpx.Client | None = None,
    request_metric_increment: Callable[[str], None] | None = None,
) -> Any:
    deadline = monotonic() + config["retry_budget_s"]
    last_category = "unknown"
    client_context = (
        nullcontext(client)
        if client is not None
        else client_factory(
            timeout=httpx.Timeout(config["timeout_s"]),
            follow_redirects=False,
            trust_env=config["trust_env"],
        )
    )
    with client_context as active_client:
        for attempt in range(config["max_retries"] + 1):
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            if request_metric_increment is not None:
                request_metric_increment("http_attempt_count")
            try:
                request_kwargs: dict[str, Any] = {
                    "headers": dict(headers),
                    "timeout": httpx.Timeout(min(config["timeout_s"], remaining)),
                }
                request_kwargs["params" if method == "GET" else "json"] = dict(payload)
                with active_client.stream(method, url, **request_kwargs) as response:
                    status = response.status_code
                    if 300 <= status < 400:
                        raise SearchBackendError("redirect responses are not accepted")
                    if status >= 400:
                        if status not in RETRYABLE_STATUS_CODES:
                            raise SearchBackendError(f"non-retryable HTTP status {status}")
                        last_category = f"http_{status}"
                        retry_after = _retry_after_seconds(response.headers.get("Retry-After"), wall_time())
                    else:
                        body = _read_limited(
                            response,
                            config["max_response_bytes"],
                            deadline=deadline,
                            monotonic=monotonic,
                        )
                        try:
                            return json.loads(body)
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise SearchBackendError("search response is not valid JSON") from exc
            except SearchBackendError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_category = type(exc).__name__
                retry_after = 0.0
            except httpx.HTTPError as exc:
                raise SearchBackendError(f"search HTTP failure: {type(exc).__name__}") from exc

            if attempt >= config["max_retries"]:
                break
            backoff = min(config["backoff_initial_s"] * (2**attempt), config["backoff_max_s"])
            delay = max(backoff, retry_after)
            if delay > deadline - monotonic():
                break
            if request_metric_increment is not None:
                request_metric_increment("retry_count")
            sleep(delay)
    raise SearchBackendError(f"search request failed after retries ({last_category})")


def _read_limited(
    response: httpx.Response,
    limit: int,
    *,
    deadline: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> bytes:
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_bytes():
        if deadline is not None and monotonic() >= deadline:
            raise SearchBackendError("search response exceeded the total request deadline")
        size += len(chunk)
        if size > limit:
            raise SearchBackendError("search response exceeds configured size limit")
        chunks.append(chunk)
    if not chunks:
        raise SearchBackendError("search response body is empty")
    if deadline is not None and monotonic() >= deadline:
        raise SearchBackendError("search response exceeded the total request deadline")
    return b"".join(chunks)


def _retry_after_seconds(value: str | None, now: float) -> float:
    if value is None:
        return 0.0
    value = value.strip()
    if value.isascii() and value.isdigit():
        return float(value)
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, parsed.timestamp() - now)


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
        contents = _optional_string(document, "contents")
        text = _optional_string(document, "text")
        title = _optional_string(document, "title")
        snippet = contents if contents is not None else text
        if not title:
            title = next((line.strip() for line in (snippet or "").splitlines() if line.strip()), None)
        if not title:
            raise SearchBackendError("retriever document has no usable title or text")
        link = _optional_string(document, "url") or _optional_string(document, "link") or ""
        if link:
            _validate_result_link(link)
        date = _normalise_date(document.get("date"))
        rows.append({"title": title, "link": link, "snippet": snippet or "", "date": date})
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
    if key not in mapping or mapping[key] is None:
        return None
    value = mapping[key]
    if not isinstance(value, str):
        raise SearchBackendError(f"{key} must be a string when present")
    return value.strip() or None


def _normalise_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise SearchBackendError("result date must be a string or null")
    return value


def _validate_result_link(link: str) -> None:
    parsed = urlsplit(link)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in link)
    ):
        raise SearchBackendError("result link must be an HTTP(S) URL")


def _dotted_get(mapping: Any, path: str, *, missing_ok: bool = False) -> Any:
    current = mapping
    for part in path.split("."):
        if not part or not isinstance(current, dict) or part not in current:
            if missing_ok:
                return None
            raise SearchBackendError(f"response path is missing: {path}")
        current = current[part]
    return current
