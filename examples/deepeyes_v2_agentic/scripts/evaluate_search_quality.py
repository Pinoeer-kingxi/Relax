#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Evaluate a configured text-search backend on answer-bearing QA examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(EXAMPLE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from app.search_backends import SearchConfigurationError, resolve_search_config  # noqa: E402
from app.search_utils import search  # noqa: E402
from smoke_search import (  # noqa: E402
    check_search_prerequisites,
    code_state,
    config_fingerprint,
    install_credential,
    valid_result_schema,
)


class EvaluationInputError(ValueError):
    """The evaluation dataset does not satisfy the documented input
    contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvaluationInputError(f"input does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationInputError(f"invalid JSONL at line {line_number}") from exc
            if not isinstance(value, dict):
                raise EvaluationInputError(f"JSONL row {line_number} must be an object")
            records.append(value)
        return records
    if suffix == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvaluationInputError("invalid JSON input") from exc
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise EvaluationInputError("JSON input must be an array of objects")
        return value
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise EvaluationInputError("reading Parquet requires pandas") from exc
        return pd.read_parquet(path).to_dict(orient="records")
    raise EvaluationInputError("input extension must be .jsonl, .json, or .parquet")


def _get_dotted(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise EvaluationInputError(f"missing field: {path}")
        value = value[part]
    return value


def _query_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not isinstance(value, (str, bytes)) and hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        user_messages = [
            item.get("content")
            for item in value
            if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), str)
        ]
        if user_messages and user_messages[-1].strip():
            return user_messages[-1].strip()
    raise EvaluationInputError("query must be a non-empty string or a chat-message list with a user message")


def _answers(value: Any) -> list[str]:
    if not isinstance(value, (str, bytes)) and hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        candidates: Iterable[Any] = [value]
    elif isinstance(value, (list, tuple)):
        candidates = value
    else:
        raise EvaluationInputError("answers must be a string or list of strings")
    answers = [item.strip() for item in candidates if isinstance(item, str) and item.strip()]
    if not answers:
        raise EvaluationInputError("answers must contain at least one non-empty string")
    return answers


def _normalized_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join("".join(character.lower() if character.isalnum() else " " for character in without_marks).split())


def _first_answer_rank(rows: list[dict[str, Any]], answers: list[str]) -> int | None:
    normalized_answers = [_normalized_text(answer) for answer in answers]
    for rank, row in enumerate(rows, start=1):
        normalized_row = _normalized_text(f"{row['title']} {row['snippet']}")
        haystack = f" {normalized_row} "
        if any(answer and f" {answer} " in haystack for answer in normalized_answers):
            return rank
    return None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _metrics_from_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(examples)
    if total == 0:
        raise EvaluationInputError("cannot calculate metrics for an empty example set")
    latencies = [float(example["latency_s"]) for example in examples]
    successes = sum(bool(example["success"]) for example in examples)
    answer_hits = sum(example["answer_rank"] is not None for example in examples)
    reciprocal_rank = sum(1.0 / example["answer_rank"] for example in examples if example["answer_rank"] is not None)
    result_count = sum(int(example["result_count"]) for example in examples)
    return {
        "example_count": total,
        "success_count": successes,
        "success_rate": successes / total,
        "answer_hit_at_k": answer_hits / total,
        "mean_reciprocal_rank": reciprocal_rank / total,
        "mean_result_count": result_count / total,
        "latency_s": {
            "mean": statistics.fmean(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies),
        },
    }


def _select_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    per_group = getattr(args, "max_examples_per_group", None)
    group_field = getattr(args, "group_field", None)
    if per_group is not None:
        if not group_field:
            raise EvaluationInputError("--max-examples-per-group requires --group-field")
        counts: dict[str, int] = {}
        selected = []
        for record in records:
            group = str(_get_dotted(record, group_field))
            if counts.get(group, 0) >= per_group:
                continue
            counts[group] = counts.get(group, 0) + 1
            selected.append(record)
        records = selected
    if args.max_examples is not None:
        records = records[: args.max_examples]
    return records


def _baseline_delta(report: dict[str, Any], baseline_path: Path) -> dict[str, Any]:
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationInputError("unable to read baseline report") from exc
    if not isinstance(baseline, dict) or not isinstance(baseline.get("metrics"), dict):
        raise EvaluationInputError("baseline report does not contain metrics")
    for key in ("input_sha256", "query_field", "answers_field", "top_k"):
        if baseline.get(key) != report.get(key):
            raise EvaluationInputError(f"baseline report {key} does not match")
    current_metrics = report["metrics"]
    baseline_metrics = baseline["metrics"]
    return {
        "backend": baseline.get("backend"),
        "config_fingerprint": baseline.get("config_fingerprint"),
        "success_rate": current_metrics["success_rate"] - baseline_metrics["success_rate"],
        "answer_hit_at_k": current_metrics["answer_hit_at_k"] - baseline_metrics["answer_hit_at_k"],
        "mean_reciprocal_rank": current_metrics["mean_reciprocal_rank"] - baseline_metrics["mean_reciprocal_rank"],
        "mean_latency_s": current_metrics["latency_s"]["mean"] - baseline_metrics["latency_s"]["mean"],
    }


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    install_credential(args.credentials_env_file, args.credentials_key)
    if args.config is not None:
        import os

        os.environ["DEEPEYES_V2_WEB_SEARCH_CONFIG"] = str(args.config.resolve())
    config = resolve_search_config()
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
        "config_fingerprint": config_fingerprint(config),
        "input_sha256": _sha256_file(args.input),
        "query_field": args.query_field,
        "answers_field": args.answers_field,
        "group_field": getattr(args, "group_field", None),
        "top_k": args.top_k if args.top_k is not None else config["top_k"],
        "examples": [],
    }
    for index, record in enumerate(records):
        query = _query_text(_get_dotted(record, args.query_field))
        answers = _answers(_get_dotted(record, args.answers_field))
        started = time.monotonic()
        result = search(query, size=args.top_k, config=config)
        wall_time = time.monotonic() - started
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        example: dict[str, Any] = {
            "index": index,
            "query_hash": query_hash,
            "success": False,
            "result_count": 0,
            "answer_rank": None,
            "latency_s": round(wall_time, 6),
        }
        if report["group_field"]:
            example["group"] = str(_get_dotted(record, report["group_field"]))
        if result != "Error" and valid_result_schema(result):
            rows = result["data"]
            rank = _first_answer_rank(rows, answers)
            example.update(success=True, result_count=len(rows), answer_rank=rank)
        report["examples"].append(example)

    metrics = _metrics_from_examples(report["examples"])
    report["metrics"] = metrics
    if report["group_field"]:
        group_names = sorted({example["group"] for example in report["examples"]})
        report["group_metrics"] = {
            group: _metrics_from_examples([example for example in report["examples"] if example["group"] == group])
            for group in group_names
        }
    if getattr(args, "baseline_report", None) is not None:
        report["baseline_delta"] = _baseline_delta(report, args.baseline_report)
    checks = {
        "all_searches_succeeded": not args.require_all_success or metrics["success_count"] == metrics["example_count"],
        "minimum_answer_hit_rate": metrics["answer_hit_at_k"] >= args.min_answer_hit_rate,
    }
    report["checks"] = checks
    report["status"] = "passed" if all(checks.values()) else "failed"
    return report, 0 if report["status"] == "passed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--query-field", default="prompt")
    parser.add_argument("--answers-field", default="extra_info.answers")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--group-field")
    parser.add_argument("--max-examples-per-group", type=int)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--require-all-success", action="store_true")
    parser.add_argument("--min-answer-hit-rate", type=float, default=0.0)
    parser.add_argument("--credentials-env-file", type=Path)
    parser.add_argument("--credentials-key", default="tvly_api_key")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k is not None and not 1 <= args.top_k <= 50:
        parser.error("--top-k must be in [1, 50]")
    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max-examples must be positive")
    if args.max_examples_per_group is not None and args.max_examples_per_group < 1:
        parser.error("--max-examples-per-group must be positive")
    if not 0.0 <= args.min_answer_hit_rate <= 1.0:
        parser.error("--min-answer-hit-rate must be in [0, 1]")
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
