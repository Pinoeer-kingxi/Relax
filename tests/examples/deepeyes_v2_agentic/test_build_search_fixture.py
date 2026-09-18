# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "examples" / "deepeyes_v2_agentic" / "scripts" / "build_search_fixture.py"
_SPEC = importlib.util.spec_from_file_location("task10_build_search_fixture", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_dataset_documents_are_clean_deterministic_and_bounded():
    rows = [
        {"text": "short"},
        {"text": "= heading =" + " ignored" * 20},
        {"text": "A   sufficiently long source paragraph " + "with useful words " * 8},
        {"text": "A   sufficiently long source paragraph " + "with useful words " * 8},
        {"text": "A second sufficiently long paragraph " + "with different useful words " * 8},
    ]

    documents = list(_MODULE._dataset_documents(rows, 2))

    assert len(documents) == 2
    assert documents[0]["id"].startswith("dataset-")
    assert documents[0]["title"] == "Wikitext excerpt 1"
    assert "  " not in documents[0]["text"]
    assert documents[0]["url"] == ""
    assert documents[0]["id"] != documents[1]["id"]


def test_golden_queries_reference_unique_curated_documents():
    document_ids = {document["id"] for document in _MODULE.GOLDEN_DOCUMENTS}
    expected_ids = [query["expected_id"] for query in _MODULE.GOLDEN_QUERIES]

    assert len(expected_ids) == len(set(expected_ids))
    assert set(expected_ids) == document_ids


def test_local_dataset_file_must_exist(tmp_path):
    args = Namespace(
        dataset_file=tmp_path / "missing.parquet",
        dataset_cache_dir=None,
        dataset_split="train",
        dataset_revision=None,
        dataset="unused",
        dataset_config="unused",
    )

    try:
        _MODULE._load_dataset(args)
    except _MODULE.FixtureBuildError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("missing local dataset was accepted")


def test_model_hashes_require_a_nonempty_directory(tmp_path):
    try:
        _MODULE._model_file_hashes(tmp_path / "missing")
    except _MODULE.FixtureBuildError as exc:
        assert "not a directory" in str(exc)
    else:
        raise AssertionError("missing model directory was accepted")

    empty = tmp_path / "empty"
    empty.mkdir()
    try:
        _MODULE._model_file_hashes(empty)
    except _MODULE.FixtureBuildError as exc:
        assert "contains no files" in str(exc)
    else:
        raise AssertionError("empty model directory was accepted")


def test_model_hashes_are_sorted_and_reproducible(tmp_path):
    (tmp_path / "z.json").write_text("z", encoding="utf-8")
    (tmp_path / "a.json").write_text("a", encoding="utf-8")

    hashes = _MODULE._model_file_hashes(tmp_path)

    assert list(hashes) == ["a.json", "z.json"]
    assert all(len(digest) == 64 for digest in hashes.values())
