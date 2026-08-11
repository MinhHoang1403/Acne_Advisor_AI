"""Offline-testable metadata scoring for Phase 2A chunk retrieval."""

from __future__ import annotations

import hashlib
import os
from typing import Any

from src.knowledge.normalizer import normalize_text_key
from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate


FIELD_WEIGHTS: dict[str, float] = {
    "drug_product": 0.24,
    "active_ingredient": 0.22,
    "drug_class": 0.18,
    "condition": 0.12,
    "query_intent_hint": 0.12,
    "safety_context": 0.10,
    "concern": 0.10,
    "content_type": 0.08,
}
MAX_METADATA_BOOST = 0.50


def score_candidate_with_metadata(
    candidate: RetrievedCandidate,
    normalized_query: NormalizedQuery,
) -> RetrievedCandidate:
    """Return a candidate copy with metadata match debug and boosted score."""

    matched = match_payload_metadata(candidate.payload, normalized_query)
    boost = min(
        sum(FIELD_WEIGHTS[field] for field in matched if field in FIELD_WEIGHTS),
        MAX_METADATA_BOOST,
    )
    base_score = float(candidate.score or 0.0)
    fused_score = round(base_score + boost, 6)
    debug = dict(candidate.debug)
    debug["metadata_boost"] = round(boost, 6)
    debug["base_score"] = base_score
    return candidate.model_copy(
        update={
            "score": round(base_score, 6),
            "fused_score": fused_score,
            "matched_metadata": {**candidate.matched_metadata, **matched},
            "debug": debug,
        }
    )


def boost_chunk_results(
    chunks: list[dict[str, Any]],
    normalized_query: NormalizedQuery,
    collection_name: str | None = None,
) -> list[RetrievedCandidate]:
    """Convert Qdrant chunk result dicts into boosted chunk candidates."""

    collection = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge")
    candidates = [
        score_candidate_with_metadata(
            RetrievedCandidate(
                candidate_id=_chunk_candidate_id(chunk),
                source="chunk",
                collection=collection,
                text=str(chunk.get("text") or chunk.get("content") or ""),
                score=float(chunk.get("score", 0.0) or 0.0),
                fused_score=None,
                payload=dict(chunk),
                matched_metadata={},
                rank=index + 1,
                debug={
                    "dense_rank": chunk.get("dense_rank"),
                    "sparse_rank": chunk.get("sparse_rank"),
                    "rrf_score": chunk.get("rrf_score"),
                },
            ),
            normalized_query,
        )
        for index, chunk in enumerate(chunks)
    ]
    candidates.sort(key=lambda candidate: candidate.fused_score or candidate.score or 0.0, reverse=True)
    return [
        candidate.model_copy(update={"rank": index + 1})
        for index, candidate in enumerate(candidates)
    ]


def annotate_chunk_results(
    chunks: list[dict[str, Any]],
    normalized_query: NormalizedQuery,
    collection_name: str | None = None,
) -> list[RetrievedCandidate]:
    """Build V5 chunk candidates without letting metadata change RRF order.

    Metadata matches are preserved for downstream inspection and later reranker
    features. The RRF score remains the primary score, with deterministic
    upstream-rank and candidate-ID tie breaks.
    """

    collection = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge")
    candidates: list[RetrievedCandidate] = []
    for index, chunk in enumerate(chunks):
        base_score = float(chunk.get("rrf_score", chunk.get("score", 0.0)) or 0.0)
        matched = match_payload_metadata(chunk, normalized_query)
        candidate = RetrievedCandidate(
            candidate_id=_chunk_candidate_id(chunk),
            source="chunk",
            collection=collection,
            text=str(chunk.get("text") or chunk.get("content") or ""),
            score=round(base_score, 6),
            fused_score=round(base_score, 6),
            payload=dict(chunk),
            matched_metadata=matched,
            rank=index + 1,
            debug={
                "dense_rank": chunk.get("dense_rank"),
                "sparse_rank": chunk.get("sparse_rank"),
                "rrf_score": chunk.get("rrf_score", base_score),
                "base_score": base_score,
                "metadata_annotation_only": True,
            },
        )
        candidates.append(candidate)

    candidates.sort(key=_annotation_sort_key)
    return [
        candidate.model_copy(update={"rank": index + 1})
        for index, candidate in enumerate(candidates)
    ]


def match_payload_metadata(
    payload: dict[str, Any],
    normalized_query: NormalizedQuery,
) -> dict[str, list[str]]:
    matched: dict[str, list[str]] = {}
    expected = {
        "drug_product": normalized_query.drug_product,
        "active_ingredient": normalized_query.active_ingredient,
        "drug_class": normalized_query.drug_class,
        "condition": normalized_query.condition,
        "query_intent_hint": normalized_query.query_intent_hint,
        "safety_context": normalized_query.safety_context,
        "concern": _as_list(normalized_query.metadata.get("concern")),
        "content_type": _as_list(normalized_query.metadata.get("content_type")),
    }
    payload_aliases = {
        "active_ingredient": ["active_ingredient", "active_ingredients", "ingredient"],
        "safety_context": ["safety_context", "safety_contexts"],
    }
    for field, query_values in expected.items():
        if not query_values:
            continue
        payload_values: list[str] = []
        for payload_field in payload_aliases.get(field, [field]):
            payload_values.extend(str(value) for value in _as_list(payload.get(payload_field)) if value)
        hits = _overlap(query_values, payload_values)
        if hits:
            matched[field] = hits
    return matched


def _overlap(query_values: list[str], payload_values: list[str]) -> list[str]:
    query_keys = {normalize_text_key(value).replace(" ", "_") for value in query_values}
    hits: list[str] = []
    for value in payload_values:
        key = normalize_text_key(value).replace(" ", "_")
        if key in query_keys and value not in hits:
            hits.append(value)
    return hits


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)] if value else []


def _chunk_candidate_id(chunk: dict[str, Any]) -> str:
    for field in ("id", "chunk_id", "point_id"):
        if chunk.get(field):
            return str(chunk[field])
    text = str(chunk.get("text") or "")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"chunk:{digest}"


def _annotation_sort_key(candidate: RetrievedCandidate) -> tuple[float, int, int, str]:
    """Keep RRF primary ordering with deterministic upstream tie breaks."""

    return (
        -float(candidate.score or 0.0),
        _rank_or_max(candidate.debug.get("dense_rank")),
        _rank_or_max(candidate.debug.get("sparse_rank")),
        candidate.candidate_id,
    )


def _rank_or_max(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 10**9
    return parsed if parsed > 0 else 10**9


__all__ = [
    "FIELD_WEIGHTS",
    "annotate_chunk_results",
    "boost_chunk_results",
    "match_payload_metadata",
    "score_candidate_with_metadata",
]
