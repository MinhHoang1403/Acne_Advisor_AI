"""Retrieval metric helpers exported separately for the V3 evaluation contract."""

from __future__ import annotations

from typing import Any

from .deterministic import summarize_metrics


def summarize_retrieval_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the retrieval/grounding section without unsupported rank metrics."""

    return summarize_metrics(results)["retrieval_and_grounding"]


__all__ = ["summarize_retrieval_metrics"]
