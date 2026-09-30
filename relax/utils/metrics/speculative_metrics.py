# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from typing import Any, Iterable

from relax.utils.speculative import SpeculativeCounts, SpeculativeGeneration


def _counter_metrics(
    counts: list[SpeculativeCounts],
    total: int,
    prefix: str,
    *,
    coverage_known: bool = True,
) -> dict[str, int | float]:
    """Keep all known counter totals; compute ratios over complete pairs."""
    metrics: dict[str, int | float] = {}
    for numerator, denominator, ratio_name, cohort in (
        ("accepted", "proposed", "accept_rate", "accept"),
        ("completion", "verify", "tokens_per_verify", "verify"),
    ):
        numerator_values = [getattr(item, numerator) for item in counts if getattr(item, numerator) is not None]
        denominator_values = [getattr(item, denominator) for item in counts if getattr(item, denominator) is not None]
        pairs = [
            (getattr(item, numerator), getattr(item, denominator))
            for item in counts
            if getattr(item, numerator) is not None and getattr(item, denominator) is not None
        ]
        numerator_total = sum(numerator_values)
        denominator_total = sum(denominator_values)
        metrics[f"{prefix}{numerator}_total"] = numerator_total
        metrics[f"{prefix}{denominator}_total"] = denominator_total
        metrics[f"{prefix}{cohort}_covered_count"] = len(pairs)
        metrics[f"{prefix}{cohort}_uncovered_count"] = max(0, total - len(pairs))
        if total and coverage_known:
            metrics[f"{prefix}{cohort}_count_coverage"] = len(pairs) / total
        paired_denominator_total = sum(bottom for _, bottom in pairs)
        if paired_denominator_total > 0:
            metrics[f"{prefix}{ratio_name}"] = sum(top for top, _ in pairs) / paired_denominator_total
    return metrics


def _sample_counts(sample: Any) -> SpeculativeCounts | None:
    records = getattr(sample, "spec_generations", None)
    if records is None:
        metadata = getattr(sample, "metadata", None) or {}
        if "agentic_trace" in metadata:
            return None
        info = getattr(sample, "spec_info", None)
        if info is None or not hasattr(info, "counts") or getattr(info, "legacy_counts", False):
            return None
        return info.counts
    if not isinstance(records, list) or not records:
        return None

    unique: dict[tuple[str, str], SpeculativeGeneration] = {}
    for raw_record in records:
        record = SpeculativeGeneration.from_dict(raw_record)
        session_id = getattr(sample, "session_id", None)
        if record is None or (session_id is not None and session_id != record.session_id):
            return None
        key = (record.session_id, record.generation_id)
        if key in unique and unique[key] != record:
            return None
        unique[key] = record
    total = SpeculativeCounts(0, 0, 0, 0)
    for record in unique.values():
        total = total.plus(record.counts)
    return total


def compute_speculative_metrics(samples: Iterable[Any]) -> dict[str, int | float]:
    """Aggregate one exported batch, deduplicating committed generation IDs."""
    generations: dict[tuple[str, str], SpeculativeGeneration] = {}
    conflicts: set[tuple[str, str]] = set()
    ordinary_counts: list[SpeculativeCounts] = []
    agentic_sample_count = legacy_sample_count = invalid_record_count = record_occurrence_count = 0

    for sample in samples:
        records = getattr(sample, "spec_generations", None)
        if records is None:
            metadata = getattr(sample, "metadata", None) or {}
            info = getattr(sample, "spec_info", None)
            counts = getattr(info, "counts", None)
            if counts is not None and not getattr(info, "legacy_counts", False) and "agentic_trace" not in metadata:
                ordinary_counts.append(counts)
            else:
                legacy_sample_count += 1
            continue

        agentic_sample_count += 1
        if not isinstance(records, list):
            invalid_record_count += 1
            continue
        for raw_record in records:
            record_occurrence_count += 1
            record = SpeculativeGeneration.from_dict(raw_record)
            sample_session = getattr(sample, "session_id", None)
            if record is None or (sample_session is not None and sample_session != record.session_id):
                invalid_record_count += 1
                continue
            key = (record.session_id, record.generation_id)
            existing = generations.get(key)
            if existing is not None and existing != record:
                conflicts.add(key)
            else:
                generations[key] = record

    metrics: dict[str, int | float] = {
        "spec/agentic_sample_count": agentic_sample_count,
        "spec/ordinary_sample_count": len(ordinary_counts),
        "spec/legacy_sample_count": legacy_sample_count,
        "spec/record_occurrence_count": record_occurrence_count,
        "spec/unique_generation_count": len(generations),
        "spec/conflicting_generation_count": len(conflicts),
        "spec/invalid_record_count": invalid_record_count,
    }
    metrics.update(
        _counter_metrics(
            [record.counts for key, record in generations.items() if key not in conflicts],
            len(generations),
            "spec/",
            coverage_known=invalid_record_count == 0,
        )
    )
    if ordinary_counts:
        metrics.update(_counter_metrics(ordinary_counts, len(ordinary_counts), "spec/sample/"))
    return metrics


def compute_speculative_log_metrics(samples: list[Any], *, enabled: bool = False) -> dict[str, int | float]:
    """Return new batch metrics plus complete legacy aliases when provable."""
    sample_counts = [_sample_counts(sample) for sample in samples]
    has_speculative_fields = any(
        counts is not None and any(value is not None for value in (counts.accepted, counts.proposed, counts.verify))
        for counts in sample_counts
    )
    legacy_counter_names = (
        "spec_accept_token_num",
        "spec_draft_token_num",
        "spec_verify_ct",
    )
    has_speculative_fields |= any(
        any((getattr(getattr(sample, "spec_info", None), key, 0) or 0) > 0 for key in legacy_counter_names)
        for sample in samples
    )
    if not enabled and not has_speculative_fields:
        return {}

    metrics = compute_speculative_metrics(samples)
    if not samples or metrics["spec/conflicting_generation_count"] or metrics["spec/invalid_record_count"]:
        return metrics

    source_counts = (
        metrics["spec/agentic_sample_count"],
        metrics["spec/ordinary_sample_count"],
        metrics["spec/legacy_sample_count"],
    )
    if sum(count > 0 for count in source_counts) != 1:
        return metrics

    for numerator, denominator, key, legacy_numerator, legacy_denominator in (
        ("accepted", "proposed", "spec_accept_rate", "spec_accept_token_num", "spec_draft_token_num"),
        ("completion", "verify", "spec_accept_length", "completion_token_num", "spec_verify_ct"),
    ):
        if source_counts[2] > 0:
            pairs = [
                (
                    getattr(getattr(sample, "spec_info", None), legacy_numerator, None),
                    getattr(getattr(sample, "spec_info", None), legacy_denominator, None),
                )
                for sample in samples
            ]
        else:
            pairs = [
                (getattr(counts, numerator), getattr(counts, denominator))
                for counts in sample_counts
                if counts is not None
            ]
        if len(pairs) == len(samples) and all(
            top is not None and bottom is not None and bottom > 0 for top, bottom in pairs
        ):
            metrics[key] = sum(top / bottom for top, bottom in pairs) / len(pairs)
    return metrics
