# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "examples" / "deepeyes_v2_agentic" / "scripts" / "evaluate_search_quality.py"
SPEC = importlib.util.spec_from_file_location("task10_evaluate_search_quality", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _args(tmp_path: Path, input_path: Path, **overrides):
    defaults = {
        "input": input_path,
        "config": None,
        "query_field": "prompt",
        "answers_field": "extra_info.answers",
        "top_k": 3,
        "max_examples": None,
        "group_field": None,
        "max_examples_per_group": None,
        "baseline_report": None,
        "require_all_success": True,
        "min_answer_hit_rate": 1.0,
        "credentials_env_file": None,
        "credentials_key": "tvly_api_key",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_normalization_and_first_answer_rank_handle_accents_and_punctuation():
    rows = [
        {"title": "Wrong", "snippet": "No match", "link": "", "date": None},
        {"title": "Le dramaturge", "snippet": "Jean-Baptiste Poquelin, dit Molière.", "link": "", "date": None},
    ]
    assert MODULE._first_answer_rank(rows, ["Moliere"]) == 2


def test_short_answer_requires_token_boundaries():
    rows = [{"title": "Paul", "snippet": "A name", "link": "", "date": None}]
    assert MODULE._first_answer_rank(rows, ["Au"]) is None


def test_mock_evaluation_writes_reproducible_redacted_metrics(tmp_path, monkeypatch):
    input_path = tmp_path / "eval.jsonl"
    input_path.write_text(
        json.dumps({"prompt": "offline marker", "extra_info": {"answers": ["offline marker"]}}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPEYES_V2_WEB_SEARCH_CONFIG", raising=False)
    report, exit_code = MODULE.evaluate(_args(tmp_path, input_path))
    assert exit_code == 0
    assert report["status"] == "passed"
    assert report["backend"] == "mock"
    assert report["metrics"]["success_rate"] == 1.0
    assert report["metrics"]["answer_hit_at_k"] == 1.0
    assert report["examples"][0]["answer_rank"] == 1
    serialized = json.dumps(report)
    assert "offline marker" not in serialized
    assert len(report["input_sha256"]) == 64


def test_evaluation_threshold_failure_has_nonzero_exit(tmp_path, monkeypatch):
    input_path = tmp_path / "eval.json"
    input_path.write_text(
        json.dumps([{"prompt": "question", "extra_info": {"answers": "answer absent from mock"}}]),
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPEYES_V2_WEB_SEARCH_CONFIG", raising=False)
    report, exit_code = MODULE.evaluate(_args(tmp_path, input_path, min_answer_hit_rate=1.0))
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["checks"]["all_searches_succeeded"] is True
    assert report["checks"]["minimum_answer_hit_rate"] is False


def test_chat_message_query_and_missing_answers_validation(tmp_path):
    assert MODULE._query_text([{"role": "assistant", "content": "x"}, {"role": "user", "content": " q "}]) == "q"
    try:
        MODULE._answers([])
    except MODULE.EvaluationInputError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("empty answers were accepted")


def test_parquet_numpy_arrays_are_accepted():
    prompt = np.array([{"role": "user", "content": " official question "}], dtype=object)
    answers = np.array(["first", "second"], dtype=object)
    assert MODULE._query_text(prompt) == "official question"
    assert MODULE._answers(answers) == ["first", "second"]


def test_group_metrics_stratified_selection_and_baseline_delta(tmp_path, monkeypatch):
    input_path = tmp_path / "grouped.jsonl"
    rows = [
        {"prompt": f"marker-{index}", "extra_info": {"answers": [f"marker-{index}"], "source": source}}
        for index, source in enumerate(["nq", "nq", "hotpot", "hotpot"])
    ]
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.delenv("DEEPEYES_V2_WEB_SEARCH_CONFIG", raising=False)
    args = _args(
        tmp_path,
        input_path,
        group_field="extra_info.source",
        max_examples_per_group=1,
    )
    baseline, exit_code = MODULE.evaluate(args)
    assert exit_code == 0
    assert baseline["metrics"]["example_count"] == 2
    assert set(baseline["group_metrics"]) == {"hotpot", "nq"}
    assert all(group["example_count"] == 1 for group in baseline["group_metrics"].values())

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    compared, exit_code = MODULE.evaluate(
        _args(
            tmp_path,
            input_path,
            group_field="extra_info.source",
            max_examples_per_group=1,
            baseline_report=baseline_path,
        )
    )
    assert exit_code == 0
    assert compared["baseline_delta"]["answer_hit_at_k"] == 0.0
    assert compared["baseline_delta"]["mean_reciprocal_rank"] == 0.0
