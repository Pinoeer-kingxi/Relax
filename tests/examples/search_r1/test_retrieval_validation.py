# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "search_r1"
sys.path.insert(0, str(EXAMPLE_DIR))

from retrieval_validation import RetrievalValidationError, validate_retrieval_request  # noqa: E402


def _validate(queries, topk=None, **overrides):
    options = {
        "default_topk": 3,
        "max_topk": 50,
        "max_batch_queries": 512,
        "index_size": 100,
    }
    options.update(overrides)
    return validate_retrieval_request(queries, topk, **options)


def test_request_validation_normalizes_queries_and_applies_default_topk():
    assert _validate(["  query  "]) == (["query"], 3)


@pytest.mark.parametrize(
    ("queries", "topk", "overrides", "message"),
    [
        ([], None, {}, "at least one"),
        ([""], None, {}, "non-empty"),
        (["x" * 8193], None, {}, "8192"),
        (["a", "b"], None, {"max_batch_queries": 1}, "at most 1"),
        (["q"], True, {}, "integer"),
        (["q"], 0, {}, r"\[1, 50\]"),
        (["q"], 51, {}, r"\[1, 50\]"),
        (["q"], 4, {"index_size": 3}, "index size"),
        (["q"], 1, {"index_size": 0}, "empty"),
    ],
)
def test_request_validation_rejects_unsafe_inputs(queries, topk, overrides, message):
    with pytest.raises(RetrievalValidationError, match=message):
        _validate(queries, topk, **overrides)
