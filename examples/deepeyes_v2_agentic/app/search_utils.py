# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Search tool helpers for the DeepEyesV2 env.

* :func:`search` dispatches to a deterministic mock, Search-R1 retriever, or a
  configured external HTTP API.
* :func:`image_search` serves cached results keyed by ``data_idx`` from JSON
  files listed in ``DEEPEYES_V2_SEARCH_CACHE_PATHS`` (colon/comma-separated).
  Missing / unparsable caches degrade to returning ``"Error"`` so the env
  surfaces a clean failure instead of crashing at import.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

from app.search_backends import SearchBackendError, SearchConfigurationError, SearchSession, run_search

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _load_image_search_cache() -> dict:
    """Load image-search caches from JSON files, controlled by env var.

    Set ``DEEPEYES_V2_SEARCH_CACHE_PATHS`` to a colon- or comma-separated list
    of JSON files. Missing or invalid files are skipped with a warning.
    """
    raw = os.environ.get("DEEPEYES_V2_SEARCH_CACHE_PATHS", "")
    if not raw.strip():
        return {}

    paths: list[str] = []
    for chunk in raw.replace(",", ":").split(":"):
        chunk = chunk.strip()
        if chunk:
            paths.append(chunk)

    merged: dict = {}
    for p in paths:
        if not os.path.isfile(p):
            logger.warning(f"[search_utils] image-search cache not found: {p} (skipping)")
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                merged.update(json.load(f))
        except Exception as exc:
            logger.warning(f"[search_utils] failed to load cache {p}: {exc}")
    return merged


# Lazily-initialised global so the module remains importable even when no cache
# is configured.
_IMAGE_SEARCH_CACHE: dict | None = None


def _get_image_search_cache() -> dict:
    global _IMAGE_SEARCH_CACHE
    if _IMAGE_SEARCH_CACHE is None:
        _IMAGE_SEARCH_CACHE = _load_image_search_cache()
    return _IMAGE_SEARCH_CACHE


def search(
    query: str,
    size: int | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    session: SearchSession | None = None,
):
    """Run text search and return the environment's stable result shape::

        {"elapsed_time": float, "data": [{"title", "link", "snippet", "date"?}, ...]}

    Search failures preserve the existing ``"Error"`` convention so a session
    can continue reasoning instead of crashing.
    """
    try:
        if session is not None:
            return session.search(query, size=size)
        return run_search(query, size=size, config=config)
    except (SearchBackendError, SearchConfigurationError) as exc:
        logger.warning(f"[search] backend failed: {type(exc).__name__}")
        return "Error"


def image_search(_query, data_idx: str | None = None):
    """Image-search via cached results, keyed by ``data_idx``.

    Only ``fvqa`` indexed entries are served. If the cache is empty (no
    ``DEEPEYES_V2_SEARCH_CACHE_PATHS``), returns ``"Error"`` so the env
    propagates a clean failure.
    """
    if data_idx is None or "fvqa" not in str(data_idx):
        logger.warning("image_search failed, no fvqa found in data index")
        return "Error"

    cache = _get_image_search_cache()
    cached_data = cache.get(data_idx, {})
    if not cached_data:
        logger.warning(f"image_search: data_idx={data_idx} not in cache (cache size={len(cache)})")
        return "Error"

    tool_returned_web_title = cached_data.get("tool_returned_web_title", [])
    cached_images_path = cached_data.get("cached_images_path", [])

    return_cached_images_path: list[str] = []
    return_tool_returned_web_title: list[str] = []
    for title, path in zip(tool_returned_web_title, cached_images_path):
        if path is not None and os.path.exists(path):
            return_cached_images_path.append(path)
            return_tool_returned_web_title.append(title)

    return {
        "tool_returned_web_title": return_tool_returned_web_title,
        "cached_images_path": return_cached_images_path,
    }
