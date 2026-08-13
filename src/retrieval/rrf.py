"""Small, score-scale-independent Reciprocal Rank Fusion primitive."""

from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    *,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse two ranked lists using point IDs and one-indexed ranks."""

    scores: dict[str, float] = {}
    documents: dict[str, dict[str, Any]] = {}

    for channel, weight, rank_key, score_key in (
        (dense_results, dense_weight, "dense_rank", "dense_score"),
        (sparse_results, sparse_weight, "sparse_rank", "sparse_score"),
    ):
        for rank, document in enumerate(channel, start=1):
            document_id = str(document.get("id", ""))
            if not document_id:
                continue
            scores[document_id] = scores.get(document_id, 0.0) + weight / (k + rank)
            documents.setdefault(document_id, dict(document))
            documents[document_id][rank_key] = rank
            documents[document_id][score_key] = document.get("score", 0.0)

    ranked_ids = sorted(scores, key=lambda document_id: scores[document_id], reverse=True)
    fused: list[dict[str, Any]] = []
    for document_id in ranked_ids:
        document = documents[document_id]
        document["rrf_score"] = round(scores[document_id], 6)
        document["score"] = document["rrf_score"]
        fused.append(document)
    return fused


__all__ = ["reciprocal_rank_fusion"]
