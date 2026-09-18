#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Evaluate final-answer quality of a live DeepEyes V2 agent with or without
search."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(EXAMPLE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from app.search_backends import SearchConfigurationError, resolve_search_config  # noqa: E402
from evaluate_search_quality import (  # noqa: E402
    EvaluationInputError,
    _answers,
    _get_dotted,
    _normalized_text,
    _query_text,
    _read_records,
    _select_records,
    _sha256_file,
)
from smoke_search import (  # noqa: E402
    _run_agent,
    check_search_prerequisites,
    code_state,
    config_fingerprint,
    install_credential,
)


def _exact_match(prediction: str | None, answers: list[str]) -> bool:
    if not prediction:
        return False
    normalized = _normalized_text(prediction)
    return any(normalized == _normalized_text(answer) for answer in answers)


def _baseline_delta(current: dict[str, Any], baseline_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    comparable_fields = ("schema_version", "input_sha256", "query_field", "answers_field", "group_field", "model")
    if not isinstance(baseline, dict) or any(baseline.get(key) != current.get(key) for key in comparable_fields):
        raise EvaluationInputError("baseline report is not comparable to the current agent evaluation")
    current_hashes = [example["query_hash"] for example in current["examples"]]
    baseline_examples = baseline.get("examples")
    if (
        not isinstance(baseline_examples, list)
        or [example.get("query_hash") for example in baseline_examples] != current_hashes
    ):
        raise EvaluationInputError("baseline report selected a different ordered example set")
    baseline_metrics = baseline.get("metrics")
    if not isinstance(baseline_metrics, dict):
        raise EvaluationInputError("baseline report has no metrics")
    deltas = {}
    for key in ("agent_success_rate", "exact_match", "final_answer_rate", "search_usage_rate", "mean_search_count"):
        try:
            deltas[key] = current["metrics"][key] - baseline_metrics[key]
        except (KeyError, TypeError) as exc:
            raise EvaluationInputError(f"baseline report is missing metric: {key}") from exc
    try:
        deltas["mean_latency_s"] = current["metrics"]["latency_s"]["mean"] - baseline_metrics["latency_s"]["mean"]
    except (KeyError, TypeError) as exc:
        raise EvaluationInputError("baseline report is missing mean latency") from exc
    return {"condition": baseline.get("condition"), "metrics": deltas}


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    install_credential(args.credentials_env_file, args.credentials_key)
    if args.config is not None:
        os.environ["DEEPEYES_V2_WEB_SEARCH_CONFIG"] = str(args.config.resolve())
    config = resolve_search_config()
    if args.condition == "search":
        check_search_prerequisites(config)
    records = _select_records(_read_records(args.input), args)
    if not records:
        raise EvaluationInputError("input contains no evaluation examples")

    source_state = code_state()
    report: dict[str, Any] = {
        "schema_version": 1,
        "code_sha": source_state["sha"],
        "code_dirty": source_state["dirty"],
        "code_worktree_fingerprint": source_state["worktree_fingerprint"],
        "backend": config["backend"],
        "condition": args.condition,
        "config_fingerprint": config_fingerprint(config),
        "input_sha256": _sha256_file(args.input),
        "query_field": args.query_field,
        "answers_field": args.answers_field,
        "group_field": args.group_field,
        "model": os.environ.get("OPENAI_MODEL", "model"),
        "examples": [],
    }
    for index, record in enumerate(records):
        query = _query_text(_get_dotted(record, args.query_field))
        answers = _answers(_get_dotted(record, args.answers_field))
        started = time.monotonic()
        example: dict[str, Any] = {
            "index": index,
            "query_hash": hashlib.sha256(query.encode()).hexdigest()[:16],
            "agent_success": False,
            "exact_match": False,
            "search_count": 0,
            "final_answer_present": False,
        }
        if args.group_field:
            example["group"] = str(_get_dotted(record, args.group_field))
        try:
            output, _model_requests = _run_agent(
                query=query,
                scripted=False,
                force_search=args.condition == "search",
            )
            metadata = output.get("metadata", {})
            prediction = metadata.get("final_answer")
            branch_counts = metadata.get("branch_counts", {})
            example.update(
                agent_success=metadata.get("stop_reason") == "env_done",
                exact_match=_exact_match(prediction, answers),
                search_count=int(metadata.get("web_search_attempt_count", 0)),
                final_answer_present=bool(prediction),
                model_request_count=sum(int(value) for value in branch_counts.values()),
                stop_reason=metadata.get("stop_reason"),
                prediction_hash=(
                    hashlib.sha256(str(prediction).encode()).hexdigest()[:16] if prediction is not None else None
                ),
            )
        except Exception as exc:  # noqa: BLE001
            example["error_category"] = type(exc).__name__
        example["latency_s"] = round(time.monotonic() - started, 6)
        report["examples"].append(example)

    def metrics(examples: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(examples)
        latencies = [float(example["latency_s"]) for example in examples]
        return {
            "example_count": total,
            "agent_success_rate": sum(bool(example["agent_success"]) for example in examples) / total,
            "exact_match": sum(bool(example["exact_match"]) for example in examples) / total,
            "final_answer_rate": sum(bool(example["final_answer_present"]) for example in examples) / total,
            "search_usage_rate": sum(int(example["search_count"]) > 0 for example in examples) / total,
            "mean_search_count": statistics.fmean(int(example["search_count"]) for example in examples),
            "latency_s": {
                "mean": statistics.fmean(latencies),
                "max": max(latencies),
            },
        }

    report["metrics"] = metrics(report["examples"])
    if args.group_field:
        report["group_metrics"] = {
            group: metrics([example for example in report["examples"] if example["group"] == group])
            for group in sorted({example["group"] for example in report["examples"]})
        }
    if getattr(args, "baseline_report", None) is not None:
        report["baseline_delta"] = _baseline_delta(report, args.baseline_report)
    expected_search = args.condition == "search"
    checks = {
        "minimum_exact_match": report["metrics"]["exact_match"] >= args.min_exact_match,
        "all_agents_succeeded": not args.require_all_success or report["metrics"]["agent_success_rate"] == 1.0,
        "search_policy_followed": (
            report["metrics"]["search_usage_rate"] == 1.0
            if expected_search
            else report["metrics"]["search_usage_rate"] == 0.0
        ),
    }
    report["checks"] = checks
    report["status"] = "passed" if all(checks.values()) else "failed"
    return report, 0 if report["status"] == "passed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--condition", choices=("search", "no-search"), required=True)
    parser.add_argument("--query-field", default="prompt")
    parser.add_argument("--answers-field", default="extra_info.answers")
    parser.add_argument("--group-field")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-examples-per-group", type=int)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--min-exact-match", type=float, default=0.0)
    parser.add_argument("--require-all-success", action="store_true")
    parser.add_argument("--credentials-env-file", type=Path)
    parser.add_argument("--credentials-key", default="tvly_api_key")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max-examples must be positive")
    if args.max_examples_per_group is not None and args.max_examples_per_group < 1:
        parser.error("--max-examples-per-group must be positive")
    if not 0.0 <= args.min_exact_match <= 1.0:
        parser.error("--min-exact-match must be in [0, 1]")
    return args


def main() -> int:
    args = parse_args()
    try:
        report, exit_code = evaluate(args)
    except (EvaluationInputError, SearchConfigurationError, FileNotFoundError, OSError, ValueError) as exc:
        report = {"schema_version": 1, "status": "failed", "error_category": type(exc).__name__}
        exit_code = 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
