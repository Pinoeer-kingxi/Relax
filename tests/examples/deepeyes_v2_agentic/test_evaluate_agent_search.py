# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "examples" / "deepeyes_v2_agentic" / "scripts" / "evaluate_agent_search.py"
SPEC = importlib.util.spec_from_file_location("task10_evaluate_agent_search", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _args(input_path: Path, condition: str = "search") -> argparse.Namespace:
    return argparse.Namespace(
        input=input_path,
        config=None,
        condition=condition,
        query_field="prompt",
        answers_field="extra_info.answers",
        group_field="extra_info.source",
        max_examples=None,
        max_examples_per_group=None,
        baseline_report=None,
        min_exact_match=1.0,
        require_all_success=True,
        credentials_env_file=None,
        credentials_key="tvly_api_key",
    )


def test_agent_quality_evaluator_scores_final_answers_and_groups(tmp_path, monkeypatch):
    input_path = tmp_path / "agent.jsonl"
    input_path.write_text(
        "".join(
            json.dumps({"prompt": query, "extra_info": {"answers": [answer], "source": source}}) + "\n"
            for query, answer, source in [("capital", "Paris", "nq"), ("play", "Shakespeare", "hotpot")]
        ),
        encoding="utf-8",
    )
    answers = iter(["Paris", "Shakespeare"])

    def fake_run_agent(**kwargs):
        assert kwargs["force_search"] is True
        return (
            {
                "metadata": {
                    "stop_reason": "env_done",
                    "final_answer": next(answers),
                    "web_search_attempt_count": 1,
                    "branch_counts": {"answer": 1, "tool_call": 1, "code": 0, "format_error": 0},
                }
            },
            0,
        )

    monkeypatch.setattr(MODULE, "_run_agent", fake_run_agent)
    monkeypatch.delenv("DEEPEYES_V2_WEB_SEARCH_CONFIG", raising=False)
    report, exit_code = MODULE.evaluate(_args(input_path))
    assert exit_code == 0
    assert report["metrics"]["exact_match"] == 1.0
    assert report["metrics"]["search_usage_rate"] == 1.0
    assert set(report["group_metrics"]) == {"hotpot", "nq"}
    assert all("prediction_hash" in example for example in report["examples"])
    assert "Paris" not in json.dumps(report)


def test_agent_quality_evaluator_enforces_no_search_policy(tmp_path, monkeypatch):
    input_path = tmp_path / "agent.json"
    input_path.write_text(
        json.dumps([{"prompt": "capital", "extra_info": {"answers": ["Paris"], "source": "nq"}}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        MODULE,
        "_run_agent",
        lambda **_kwargs: (
            {
                "metadata": {
                    "stop_reason": "env_done",
                    "final_answer": "Paris",
                    "web_search_attempt_count": 0,
                    "branch_counts": {"answer": 1, "tool_call": 0, "code": 0, "format_error": 0},
                }
            },
            0,
        ),
    )
    monkeypatch.delenv("DEEPEYES_V2_WEB_SEARCH_CONFIG", raising=False)
    report, exit_code = MODULE.evaluate(_args(input_path, condition="no-search"))
    assert exit_code == 0
    assert report["checks"]["search_policy_followed"] is True


def test_agent_quality_evaluator_compares_ordered_baseline(tmp_path, monkeypatch):
    input_path = tmp_path / "agent.json"
    input_path.write_text(
        json.dumps([{"prompt": "capital", "extra_info": {"answers": ["Paris"], "source": "nq"}}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        MODULE,
        "_run_agent",
        lambda **_kwargs: (
            {
                "metadata": {
                    "stop_reason": "env_done",
                    "final_answer": "Paris",
                    "web_search_attempt_count": 1,
                    "branch_counts": {"answer": 1, "tool_call": 1},
                }
            },
            0,
        ),
    )
    monkeypatch.delenv("DEEPEYES_V2_WEB_SEARCH_CONFIG", raising=False)
    baseline_report, _ = MODULE.evaluate(_args(input_path, condition="search"))
    baseline_report["condition"] = "no-search"
    baseline_report["metrics"]["exact_match"] = 0.0
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline_report), encoding="utf-8")

    args = _args(input_path)
    args.baseline_report = baseline_path
    report, exit_code = MODULE.evaluate(args)
    assert exit_code == 0
    assert report["baseline_delta"]["condition"] == "no-search"
    assert report["baseline_delta"]["metrics"]["exact_match"] == 1.0
