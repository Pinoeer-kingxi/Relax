# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure request validation shared by the Search-R1 retrieval server tests."""

from __future__ import annotations


MAX_QUERY_CHARS = 8192


class RetrievalValidationError(ValueError):
    """The request cannot be executed safely by the configured index."""


def validate_retrieval_request(
    queries: list[str],
    topk: int | None,
    *,
    default_topk: int,
    max_topk: int,
    max_batch_queries: int,
    index_size: int,
) -> tuple[list[str], int]:
    if not queries:
        raise RetrievalValidationError("queries must contain at least one item")
    if len(queries) > max_batch_queries:
        raise RetrievalValidationError(f"queries must contain at most {max_batch_queries} items")
    normalized: list[str] = []
    for query in queries:
        if not isinstance(query, str) or not query.strip():
            raise RetrievalValidationError("every query must be a non-empty string")
        query = query.strip()
        if len(query) > MAX_QUERY_CHARS:
            raise RetrievalValidationError(f"query exceeds {MAX_QUERY_CHARS} characters")
        normalized.append(query)

    effective_topk = default_topk if topk is None else topk
    if isinstance(effective_topk, bool) or not isinstance(effective_topk, int):
        raise RetrievalValidationError("topk must be an integer")
    if effective_topk < 1 or effective_topk > max_topk:
        raise RetrievalValidationError(f"topk must be in [1, {max_topk}]")
    if index_size < 1:
        raise RetrievalValidationError("retrieval index is empty")
    if effective_topk > index_size:
        raise RetrievalValidationError(f"topk must not exceed index size ({index_size})")
    return normalized, effective_topk
