"""Small, source-traceable metrics used by the S2 ablation harness."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def first_relevant_rank(ranking: Sequence[str], relevant_ids: Iterable[str]) -> int | None:
    """Return the one-based rank of the first relevant item, or ``None``."""

    relevant = set(relevant_ids)
    return next(
        (rank for rank, candidate_id in enumerate(ranking, start=1) if candidate_id in relevant),
        None,
    )


def recall_at_k(
    rankings: Sequence[Sequence[str]],
    relevant_ids: Sequence[Iterable[str]],
    *,
    k: int,
) -> dict[str, int | float]:
    """Return macro query hit-rate at ``k`` with an explicit denominator.

    In this project each query is counted as one success when at least one
    trusted relevant item occurs in the first ``k`` positions. This is the
    retrieval-success form of Recall@k used by the S2 diagnostic harness.
    """

    _validate_parallel_inputs(rankings, relevant_ids)
    if k < 1:
        raise ValueError("k must be positive")
    numerator = sum(
        bool(set(ranking[:k]) & set(relevant))
        for ranking, relevant in zip(rankings, relevant_ids, strict=True)
    )
    denominator = len(rankings)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else 0.0,
    }


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[str]],
    relevant_ids: Sequence[Iterable[str]],
) -> dict[str, int | float]:
    """Return mean reciprocal rank; a query with no relevant item contributes zero."""

    _validate_parallel_inputs(rankings, relevant_ids)
    reciprocal_rank_sum = 0.0
    for ranking, relevant in zip(rankings, relevant_ids, strict=True):
        rank = first_relevant_rank(ranking, relevant)
        reciprocal_rank_sum += 0.0 if rank is None else 1.0 / rank
    denominator = len(rankings)
    return {
        "reciprocal_rank_sum": reciprocal_rank_sum,
        "denominator": denominator,
        "value": reciprocal_rank_sum / denominator if denominator else 0.0,
    }


def evidence_retention_rate(before: Sequence[str], after: Sequence[str]) -> dict[str, int | float]:
    """Return the fraction of unique input evidence IDs retained in output."""

    input_ids = set(before)
    retained = input_ids & set(after)
    denominator = len(input_ids)
    return {
        "numerator": len(retained),
        "denominator": denominator,
        "value": len(retained) / denominator if denominator else 1.0,
    }


def duplicate_slot_rate(candidate_ids: Sequence[str]) -> dict[str, int | float]:
    """Return duplicate output slots divided by all output slots."""

    duplicate_slots = len(candidate_ids) - len(set(candidate_ids))
    denominator = len(candidate_ids)
    return {
        "numerator": duplicate_slots,
        "denominator": denominator,
        "value": duplicate_slots / denominator if denominator else 0.0,
    }


def arithmetic_mean(values: Sequence[float]) -> dict[str, int | float]:
    """Return a transparent arithmetic mean for descriptive latency reporting."""

    total = float(sum(values))
    denominator = len(values)
    return {
        "sum": total,
        "denominator": denominator,
        "value": total / denominator if denominator else 0.0,
    }


def _validate_parallel_inputs(
    rankings: Sequence[Sequence[str]],
    relevant_ids: Sequence[Iterable[str]],
) -> None:
    if len(rankings) != len(relevant_ids):
        raise ValueError("rankings and relevant_ids must have equal length")


__all__ = [
    "arithmetic_mean",
    "duplicate_slot_rate",
    "evidence_retention_rate",
    "first_relevant_rank",
    "mean_reciprocal_rank",
    "recall_at_k",
]
