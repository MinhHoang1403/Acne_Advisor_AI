"""Passive V4-to-V5 compatibility adapters used by the R1 shadow path."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from src.retrieval.contracts import (
    ContextItem,
    NormalizedQuery,
    PackedContext,
    QueryExpansion,
    RerankTrace,
    RetrievedCandidate,
)
from src.retrieval.v5_contracts import (
    CandidateObservationV5,
    CandidateProvenanceV5,
    QueryContextV5,
    QueryObservationV5,
    RetrievalPipelineVersion,
    RetrievalStageEventV5,
    RetrievalStageV5,
    RetrievalTraceV5,
    ScoreNamespaceV5,
    ShadowComparisonV5,
    ShadowStageComparisonV5,
)


DEFAULT_RETRIEVAL_PIPELINE_VERSION = RetrievalPipelineVersion.V4
V5_CONFIG_FINGERPRINT = "retrieval_v5_contracts_v1"


@dataclass(frozen=True)
class RetrievalPipelineSelection:
    """Requested mode and the safe execution mode available at R1."""

    requested: RetrievalPipelineVersion
    execution: RetrievalPipelineVersion
    shadow_enabled: bool
    warning_code: str | None = None


def retrieval_pipeline_selection_from_env() -> RetrievalPipelineSelection:
    """Resolve the migration selector without silently promoting unfinished V5."""

    raw = os.getenv("RETRIEVAL_PIPELINE_VERSION", DEFAULT_RETRIEVAL_PIPELINE_VERSION.value)
    try:
        requested = RetrievalPipelineVersion(raw.strip().lower())
    except ValueError:
        return RetrievalPipelineSelection(
            requested=DEFAULT_RETRIEVAL_PIPELINE_VERSION,
            execution=RetrievalPipelineVersion.V4,
            shadow_enabled=False,
            warning_code="INVALID_RETRIEVAL_PIPELINE_VERSION",
        )

    if requested == RetrievalPipelineVersion.V5:
        return RetrievalPipelineSelection(
            requested=requested,
            execution=RetrievalPipelineVersion.V4,
            shadow_enabled=True,
            warning_code="V5_NOT_PROMOTED_USING_V4_SHADOW",
        )
    return RetrievalPipelineSelection(
        requested=requested,
        execution=RetrievalPipelineVersion.V4,
        shadow_enabled=requested == RetrievalPipelineVersion.V5_SHADOW,
    )


def build_v5_shadow_trace(
    *,
    original_query: str,
    retrieval_query: str,
    normalized_query: NormalizedQuery,
    expansion: QueryExpansion,
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    fused_results: list[dict[str, Any]],
    entity_candidates: list[RetrievedCandidate],
    chunk_candidates: list[RetrievedCandidate],
    merged_candidates: list[RetrievedCandidate],
    reranked_candidates: list[RetrievedCandidate],
    rerank_trace: RerankTrace,
    packed_context: PackedContext,
    timings_ms: Mapping[str, float],
    selection: RetrievalPipelineSelection,
) -> RetrievalTraceV5:
    """Create a passive V5 trace from already-finalized legacy stage outputs."""

    query_context = query_context_from_legacy(
        original_query=original_query,
        retrieval_query=retrieval_query,
        normalized_query=normalized_query,
    )
    trace = RetrievalTraceV5(
        trace_id=_sha256(f"{query_context.query_id}:{query_context.intent}"),
        pipeline_version=selection.requested,
        config_fingerprint=V5_CONFIG_FINGERPRINT,
        query_hash=_sha256(query_context.original_query),
        query_observation=QueryObservationV5(
            intent=query_context.intent,
            language=query_context.language,
            normalized_entity_ids=query_context.normalized_entity_ids,
            safety_flags=query_context.safety_flags,
            expansion_terms=tuple(_dedupe(expansion.expanded_terms)),
        ),
    )
    trace = trace.append_event(
        RetrievalStageEventV5(
            stage=RetrievalStageV5.QUERY_UNDERSTANDING,
            warning_codes=tuple(
                code
                for code in (selection.warning_code,)
                if code
            ),
            elapsed_ms=timings_ms.get("normalize_expand"),
        )
    )
    trace = trace.append_event(
        _mapping_event(
            RetrievalStageV5.DENSE,
            dense_results,
            elapsed_ms=timings_ms.get("dense"),
        )
    )
    trace = trace.append_event(
        _mapping_event(
            RetrievalStageV5.SPARSE,
            sparse_results,
            elapsed_ms=timings_ms.get("sparse"),
        )
    )
    trace = trace.append_event(
        RetrievalStageEventV5(
            stage=RetrievalStageV5.ENTITY_RESOLUTION,
            candidates=tuple(
                _candidate_observation(candidate, legacy_compat_only=True)
                for candidate in entity_candidates
            ),
            elapsed_ms=timings_ms.get("entity"),
        )
    )
    trace = trace.append_event(
        _mapping_event(
            RetrievalStageV5.KNOWLEDGE_FUSION,
            fused_results,
        )
    )
    trace = trace.append_event(
        RetrievalStageEventV5(
            stage=RetrievalStageV5.LEGACY_METADATA,
            candidates=tuple(_candidate_observation(candidate) for candidate in chunk_candidates),
            elapsed_ms=timings_ms.get("boost"),
        )
    )
    trace = trace.append_event(
        RetrievalStageEventV5(
            stage=RetrievalStageV5.LEGACY_CANDIDATE_MERGE,
            candidates=tuple(
                _candidate_observation(candidate, legacy_compat_only=True)
                for candidate in merged_candidates
            ),
        )
    )
    rerank_scores = {
        item.candidate.candidate_id: item.rerank_score
        for item in rerank_trace.ranked_candidates
    }
    trace = trace.append_event(
        RetrievalStageEventV5(
            stage=RetrievalStageV5.RERANK,
            candidates=tuple(
                _candidate_observation(
                    candidate,
                    reranker_final=rerank_scores.get(candidate.candidate_id),
                    legacy_compat_only=True,
                )
                for candidate in reranked_candidates
            ),
            warning_codes=("RERANK_FALLBACK",) if rerank_trace.fallback_used else (),
            elapsed_ms=timings_ms.get("rerank"),
        )
    )
    trace = trace.append_event(
        RetrievalStageEventV5(
            stage=RetrievalStageV5.PACKER,
            candidates=tuple(_context_observation(item) for item in packed_context.items),
            elapsed_ms=timings_ms.get("pack"),
            context_sha256=_sha256(packed_context.context_text),
        )
    )
    trace = trace.append_event(
        RetrievalStageEventV5(
            stage=RetrievalStageV5.GRAPH,
            elapsed_ms=timings_ms.get("neo4j"),
        )
    )
    return trace


def query_context_from_legacy(
    *,
    original_query: str,
    retrieval_query: str,
    normalized_query: NormalizedQuery,
) -> QueryContextV5:
    """Convert legacy query normalization into the immutable V5 contract."""

    normalized_entities = tuple(
        _dedupe(
            [
                *normalized_query.drug_product,
                *normalized_query.active_ingredient,
                *normalized_query.drug_class,
                *normalized_query.condition,
            ]
        )
    )
    return QueryContextV5(
        query_id=_sha256(f"{original_query}:{retrieval_query}"),
        original_query=original_query,
        retrieval_query=retrieval_query,
        intent=normalized_query.intent,
        language=_detect_language(normalized_query.original_query),
        normalized_entity_ids=normalized_entities,
        safety_flags=tuple(_dedupe(normalized_query.safety_context)),
    )


def compare_v4_to_v5_shadow(
    *,
    trace: RetrievalTraceV5,
    dense_results: Iterable[Mapping[str, Any]],
    sparse_results: Iterable[Mapping[str, Any]],
    fused_results: Iterable[Mapping[str, Any]],
    merged_candidates: Iterable[RetrievedCandidate],
    reranked_candidates: Iterable[RetrievedCandidate],
    packed_context: PackedContext,
) -> ShadowComparisonV5:
    """Compare ordered V4 stage identities against an immutable V5 shadow trace."""

    expected = (
        ("dense", _mapping_ids(dense_results), RetrievalStageV5.DENSE),
        ("sparse", _mapping_ids(sparse_results), RetrievalStageV5.SPARSE),
        ("rrf", _mapping_ids(fused_results), RetrievalStageV5.KNOWLEDGE_FUSION),
        ("merged", _candidate_ids(merged_candidates), RetrievalStageV5.LEGACY_CANDIDATE_MERGE),
        ("reranker_output", _candidate_ids(reranked_candidates), RetrievalStageV5.RERANK),
        ("packed", _context_ids(packed_context.items), RetrievalStageV5.PACKER),
    )
    stages = tuple(
        ShadowStageComparisonV5(
            stage=name,
            legacy_candidate_ids=legacy_ids,
            shadow_candidate_ids=trace.candidate_ids_for(stage),
            equivalent=legacy_ids == trace.candidate_ids_for(stage),
        )
        for name, legacy_ids, stage in expected
    )
    legacy_context_sha256 = _sha256(packed_context.context_text)
    shadow_context_sha256 = _context_hash_from_trace(trace)
    return ShadowComparisonV5(
        stages=stages,
        legacy_context_sha256=legacy_context_sha256,
        shadow_context_sha256=shadow_context_sha256,
        equivalent=all(item.equivalent for item in stages)
        and legacy_context_sha256 == shadow_context_sha256,
    )


def _mapping_event(
    stage: RetrievalStageV5,
    candidates: Iterable[Mapping[str, Any]],
    *,
    elapsed_ms: float | None = None,
) -> RetrievalStageEventV5:
    return RetrievalStageEventV5(
        stage=stage,
        candidates=tuple(
            _mapping_observation(candidate, stage=stage, rank=index)
            for index, candidate in enumerate(candidates, start=1)
        ),
        elapsed_ms=elapsed_ms,
    )


def _mapping_observation(
    candidate: Mapping[str, Any],
    *,
    stage: RetrievalStageV5,
    rank: int,
) -> CandidateObservationV5:
    candidate_id = _mapping_candidate_id(candidate)
    dense = _finite_or_none(candidate.get("dense_score"))
    sparse = _finite_or_none(candidate.get("sparse_score"))
    if stage == RetrievalStageV5.DENSE:
        dense = _finite_or_none(candidate.get("score"))
    if stage == RetrievalStageV5.SPARSE:
        sparse = _finite_or_none(candidate.get("score"))
    return CandidateObservationV5(
        candidate_id=candidate_id,
        source="chunk",
        collection=_string_or_none(candidate.get("collection")),
        rank=rank,
        scores=ScoreNamespaceV5(
            dense_similarity=dense,
            sparse_bm25_score=sparse,
            rrf=_finite_or_none(candidate.get("rrf_score")),
            legacy_compat_score=(
                _finite_or_none(candidate.get("score"))
                if stage not in {RetrievalStageV5.DENSE, RetrievalStageV5.SPARSE}
                else None
            ),
        ),
        provenance=_mapping_provenance(candidate, candidate_id),
        metadata_features=tuple(_metadata_features(candidate)),
    )


def _candidate_observation(
    candidate: RetrievedCandidate,
    *,
    reranker_final: float | None = None,
    legacy_compat_only: bool = False,
) -> CandidateObservationV5:
    payload = candidate.payload
    return CandidateObservationV5(
        candidate_id=candidate.candidate_id,
        source=candidate.source,
        collection=candidate.collection,
        rank=candidate.rank,
        scores=ScoreNamespaceV5(
            rrf=_finite_or_none(payload.get("rrf_score")),
            reranker_final=_finite_or_none(reranker_final),
            legacy_compat_score=_finite_or_none(candidate.fused_score or candidate.score),
        ),
        provenance=_mapping_provenance(payload, candidate.candidate_id),
        metadata_features=tuple(_metadata_features(candidate.matched_metadata, candidate.debug)),
        legacy_compat_only=legacy_compat_only,
    )


def _context_observation(item: ContextItem) -> CandidateObservationV5:
    return CandidateObservationV5(
        candidate_id=item.item_id,
        source=item.source,
        rank=item.rank,
        scores=ScoreNamespaceV5(
            rrf=_finite_or_none(item.payload.get("rrf_score")),
            legacy_compat_score=_finite_or_none(item.fused_score or item.score),
        ),
        provenance=_mapping_provenance(item.payload, item.item_id),
        metadata_features=tuple(_metadata_features(item.matched_metadata)),
        legacy_compat_only=True,
    )


def _mapping_provenance(candidate: Mapping[str, Any], candidate_id: str) -> CandidateProvenanceV5:
    return CandidateProvenanceV5(
        point_id=_string_or_none(candidate.get("id")),
        chunk_id=_string_or_none(candidate.get("chunk_id")) or candidate_id,
        document_id=_string_or_none(candidate.get("document_id")),
        source_path=_string_or_none(candidate.get("source_path") or candidate.get("source_file")),
    )


def _metadata_features(*values: Mapping[str, Any]) -> list[str]:
    features: list[str] = []
    for value in values:
        for key, item in value.items():
            if item not in (None, False, [], {}, ""):
                features.append(str(key))
    return _dedupe(features)


def _mapping_ids(candidates: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(_mapping_candidate_id(candidate) for candidate in candidates)


def _candidate_ids(candidates: Iterable[RetrievedCandidate]) -> tuple[str, ...]:
    return tuple(candidate.candidate_id for candidate in candidates)


def _context_ids(items: Iterable[ContextItem]) -> tuple[str, ...]:
    return tuple(item.item_id for item in items)


def _mapping_candidate_id(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("chunk_id") or candidate.get("id")
    return str(value or "unknown")


def _context_hash_from_trace(trace: RetrievalTraceV5) -> str:
    for event in trace.events:
        if event.stage == RetrievalStageV5.PACKER:
            return event.context_sha256 or ""
    return ""


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in (float("inf"), float("-inf")) else None


def _string_or_none(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _detect_language(query: str) -> str:
    return "vi" if any(ord(character) > 127 for character in query) else "und"


__all__ = [
    "DEFAULT_RETRIEVAL_PIPELINE_VERSION",
    "RetrievalPipelineSelection",
    "V5_CONFIG_FINGERPRINT",
    "build_v5_shadow_trace",
    "compare_v4_to_v5_shadow",
    "query_context_from_legacy",
    "retrieval_pipeline_selection_from_env",
]
