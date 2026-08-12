"""Compatibility adapters for staged Retrieval V5 execution and shadow traces."""

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
    CandidateDropV5,
    CandidateObservationV5,
    CandidateProvenanceV5,
    CandidateTransitionReasonV5,
    CandidateTransitionStatusV5,
    CandidateTransitionV5,
    DropReasonV5,
    EvidenceSelectionResultV5,
    EvidencePackingStatusV5,
    EvidenceSufficiencyV5,
    GraphSignalObservationV5,
    GraphSignalV5,
    PackedEvidenceV5,
    QueryContextV5,
    QueryObservationV5,
    RankedEvidenceV5,
    RetrievalPipelineVersion,
    RetrievalStageEventV5,
    RetrievalStageSummaryV5,
    RetrievalStageV5,
    RetrievalTraceV5,
    RETRIEVAL_V5_CONFIG_VERSION,
    ScoreNamespaceV5,
    SelectedEvidenceV5,
    ShadowComparisonV5,
    ShadowStageComparisonV5,
)
from src.retrieval.v5_signals import (
    build_entity_signals,
    build_graph_signals,
    entity_graph_seed_names,
)


DEFAULT_RETRIEVAL_PIPELINE_VERSION = RetrievalPipelineVersion.V5
V5_CONFIG_FINGERPRINT = RETRIEVAL_V5_CONFIG_VERSION


@dataclass(frozen=True)
class RetrievalPipelineSelection:
    """Requested mode and the staged execution mode available in V5."""

    requested: RetrievalPipelineVersion
    execution: RetrievalPipelineVersion
    shadow_enabled: bool
    warning_code: str | None = None


def retrieval_pipeline_selection_from_env() -> RetrievalPipelineSelection:
    """Resolve the release selector while retaining explicit V4 rollback."""

    raw = os.getenv("RETRIEVAL_PIPELINE_VERSION", DEFAULT_RETRIEVAL_PIPELINE_VERSION.value)
    try:
        requested = RetrievalPipelineVersion(raw.strip().lower())
    except ValueError:
        return RetrievalPipelineSelection(
            requested=DEFAULT_RETRIEVAL_PIPELINE_VERSION,
            execution=DEFAULT_RETRIEVAL_PIPELINE_VERSION,
            shadow_enabled=DEFAULT_RETRIEVAL_PIPELINE_VERSION == RetrievalPipelineVersion.V5,
            warning_code="INVALID_RETRIEVAL_PIPELINE_VERSION",
        )

    if requested == RetrievalPipelineVersion.V5:
        return RetrievalPipelineSelection(
            requested=requested,
            execution=RetrievalPipelineVersion.V5,
            shadow_enabled=True,
        )
    return RetrievalPipelineSelection(
        requested=requested,
        execution=RetrievalPipelineVersion.V4,
        shadow_enabled=requested == RetrievalPipelineVersion.V5_SHADOW,
    )


def retrieval_v5_config_version_from_env() -> str:
    """Resolve the V5 semantic contract version for trace and cache identity."""

    return (
        os.getenv("RETRIEVAL_V5_CONFIG_VERSION", V5_CONFIG_FINGERPRINT).strip()
        or V5_CONFIG_FINGERPRINT
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
    graph_facts: list[dict[str, Any]],
    timings_ms: Mapping[str, float],
    selection: RetrievalPipelineSelection,
    candidate_policy_candidates: list[RetrievedCandidate] | None = None,
    candidate_policy_drops: tuple[CandidateDropV5, ...] = (),
    ranked_evidence: tuple[RankedEvidenceV5, ...] = (),
    evidence_selection_result: EvidenceSelectionResultV5 | None = None,
    packed_evidence: PackedEvidenceV5 | None = None,
    graph_signals: tuple[GraphSignalV5, ...] | None = None,
    graph_lookup_attempted: bool = True,
    graph_warning_codes: tuple[str, ...] = (),
) -> RetrievalTraceV5:
    """Create a redacted V5 trace from finalized staged retrieval outputs."""

    query_context = query_context_from_legacy(
        original_query=original_query,
        retrieval_query=retrieval_query,
        normalized_query=normalized_query,
    )
    entity_signals = build_entity_signals(entity_candidates)
    graph_seed_names = entity_graph_seed_names(entity_signals)
    graph_signals = (
        graph_signals
        if graph_signals is not None
        else build_graph_signals(graph_facts, entity_signals)
    )
    trace = RetrievalTraceV5(
        trace_id=_sha256(f"{query_context.query_id}:{query_context.intent}"),
        pipeline_version=selection.requested,
        config_fingerprint=retrieval_v5_config_version_from_env(),
        query_hash=_sha256(query_context.original_query),
        query_observation=QueryObservationV5(
            intent=query_context.intent,
            language=query_context.language,
            normalized_entity_ids=query_context.normalized_entity_ids,
            safety_flags=query_context.safety_flags,
            expansion_terms=tuple(_dedupe(expansion.expanded_terms)),
        ),
        entity_signals=entity_signals,
        graph_seed_names=graph_seed_names,
        graph_signals=graph_signals,
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
    metadata_stage = (
        RetrievalStageV5.METADATA_ANNOTATION
        if selection.execution == RetrievalPipelineVersion.V5
        else RetrievalStageV5.LEGACY_METADATA
    )
    trace = trace.append_event(
        RetrievalStageEventV5(
            stage=metadata_stage,
            candidates=tuple(_candidate_observation(candidate) for candidate in chunk_candidates),
            elapsed_ms=timings_ms.get("boost"),
        )
    )
    trace = trace.append_event(
        RetrievalStageEventV5(
            stage=RetrievalStageV5.SOURCE_EVIDENCE_POOL,
            candidates=tuple(
                _candidate_observation(candidate)
                for candidate in chunk_candidates
                if candidate.source == "chunk"
            ),
        )
    )
    if selection.execution == RetrievalPipelineVersion.V5:
        trace = trace.append_event(
            _graph_event(
                graph_signals=graph_signals,
                graph_seed_names=graph_seed_names,
                graph_fact_count=len(graph_facts),
                lookup_attempted=graph_lookup_attempted,
                warning_codes=graph_warning_codes,
                elapsed_ms=timings_ms.get("neo4j"),
            )
        )
    if selection.execution == RetrievalPipelineVersion.V5:
        trace = trace.append_event(
            RetrievalStageEventV5(
                stage=RetrievalStageV5.CANDIDATE_POLICY,
                candidates=tuple(
                    _candidate_observation(candidate)
                    for candidate in candidate_policy_candidates or []
                ),
                drops=candidate_policy_drops,
            )
        )
    else:
        trace = trace.append_event(
            RetrievalStageEventV5(
                stage=RetrievalStageV5.LEGACY_CANDIDATE_MERGE,
                candidates=tuple(
                    _candidate_observation(candidate, legacy_compat_only=True)
                    for candidate in merged_candidates
                ),
            )
        )
    legacy_execution = selection.execution == RetrievalPipelineVersion.V4
    rerank_scores = {
        item.candidate.candidate_id: item.rerank_score
        for item in rerank_trace.ranked_candidates
    }
    rerank_observations = (
        tuple(_ranked_evidence_observation(item) for item in ranked_evidence)
        if selection.execution == RetrievalPipelineVersion.V5 and ranked_evidence
        else tuple(
            _candidate_observation(
                candidate,
                reranker_final=rerank_scores.get(candidate.candidate_id),
                legacy_compat_only=legacy_execution,
            )
            for candidate in reranked_candidates
        )
    )
    rerank_transitions = build_reranker_transitions_v5(
        policy_candidates=(
            candidate_policy_candidates
            if selection.execution == RetrievalPipelineVersion.V5
            and candidate_policy_candidates is not None
            else merged_candidates
        ),
        ranked_evidence=ranked_evidence,
        reranked_candidates=reranked_candidates,
        fallback_used=rerank_trace.fallback_used,
    )
    trace = trace.append_event(
        RetrievalStageEventV5(
            stage=RetrievalStageV5.RERANK,
            candidates=rerank_observations,
            drops=tuple(
                CandidateDropV5(
                    candidate_id=transition.candidate_id,
                    reason=DropReasonV5.RERANK_REMOVED,
                )
                for transition in rerank_transitions
                if transition.status == CandidateTransitionStatusV5.REMOVED
            ),
            transitions=rerank_transitions,
            warning_codes=("RERANK_FALLBACK",) if rerank_trace.fallback_used else (),
            elapsed_ms=timings_ms.get("rerank"),
            summary=RetrievalStageSummaryV5(
                input_count=rerank_trace.input_count,
                output_count=rerank_trace.output_count,
                status_code=("FALLBACK" if rerank_trace.fallback_used else "COMPLETED"),
            ),
        )
    )
    if selection.execution == RetrievalPipelineVersion.V5:
        selected_evidence = (
            evidence_selection_result.selected_evidence
            if evidence_selection_result is not None
            else ()
        )
        selector_transitions = build_selector_transitions_v5(
            ranked_evidence=ranked_evidence,
            result=evidence_selection_result,
        )
        trace = trace.append_event(
            RetrievalStageEventV5(
                stage=RetrievalStageV5.EVIDENCE_SELECTOR,
                candidates=tuple(
                    _selected_evidence_observation(item)
                    for item in selected_evidence
                ),
                drops=tuple(
                    CandidateDropV5(
                        candidate_id=transition.candidate_id,
                        reason=DropReasonV5.EVIDENCE_SELECTOR_REMOVED,
                    )
                    for transition in selector_transitions
                    if transition.status == CandidateTransitionStatusV5.NOT_SELECTED
                ),
                transitions=selector_transitions,
                warning_codes=(
                    ("EVIDENCE_INSUFFICIENT",)
                    if evidence_selection_result is not None
                    and evidence_selection_result.status
                    != EvidenceSufficiencyV5.SUFFICIENT
                    else ()
                ),
                elapsed_ms=timings_ms.get("evidence_selector"),
                summary=(
                    RetrievalStageSummaryV5(
                        input_count=len(ranked_evidence),
                        output_count=len(selected_evidence),
                        status_code=evidence_selection_result.status.value,
                        required_roles=evidence_selection_result.requirements.required_roles,
                        satisfied_roles=evidence_selection_result.satisfied_roles,
                        missing_roles=evidence_selection_result.missing_roles,
                    )
                    if evidence_selection_result is not None
                    else None
                ),
            )
        )
    trace = trace.append_event(
        RetrievalStageEventV5(
            stage=RetrievalStageV5.PACKER,
            candidates=tuple(
                _context_observation(
                    item,
                    legacy_compat_only=legacy_execution,
                    extra_metadata_features=(
                        ("critical",)
                        if packed_evidence is not None
                        and item.item_id in packed_evidence.critical_evidence_ids
                        else ()
                    ),
                )
                for item in packed_context.items
            ),
            drops=(packed_evidence.drops if packed_evidence is not None else ()),
            warning_codes=(
                ("PACKER_OVERFLOW",)
                if packed_evidence is not None
                and packed_evidence.status != EvidencePackingStatusV5.SUFFICIENT
                else ()
            ),
            elapsed_ms=timings_ms.get("pack"),
            context_sha256=_sha256(packed_context.context_text),
            summary=(
                RetrievalStageSummaryV5(
                    input_count=(
                        len(evidence_selection_result.selected_evidence)
                        if evidence_selection_result is not None
                        else len(packed_context.items)
                    ),
                    output_count=len(packed_context.items),
                    status_code=packed_evidence.status.value,
                    used_items=packed_evidence.used_items,
                    max_items=packed_evidence.max_items,
                    used_characters=packed_evidence.character_count,
                    max_characters=packed_evidence.max_characters,
                    estimated_tokens=packed_evidence.token_count,
                    max_tokens=packed_evidence.max_tokens,
                    token_count_mode=packed_evidence.token_count_mode,
                )
                if packed_evidence is not None
                else None
            ),
        )
    )
    if selection.execution != RetrievalPipelineVersion.V5:
        trace = trace.append_event(
            _graph_event(
                graph_signals=graph_signals,
                graph_seed_names=graph_seed_names,
                graph_fact_count=len(graph_facts),
                lookup_attempted=graph_lookup_attempted,
                warning_codes=graph_warning_codes,
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


def _ranked_evidence_observation(evidence: RankedEvidenceV5) -> CandidateObservationV5:
    candidate = evidence.candidate.candidate
    return CandidateObservationV5(
        candidate_id=candidate.candidate_id,
        source="chunk",
        rank=evidence.output_rank,
        scores=evidence.scores,
        provenance=candidate.provenance,
        metadata_features=candidate.metadata_features,
    )


def _selected_evidence_observation(selected_evidence: SelectedEvidenceV5) -> CandidateObservationV5:
    observation = _ranked_evidence_observation(selected_evidence.evidence)
    return observation.model_copy(
        update={"metadata_features": (*observation.metadata_features, *selected_evidence.roles)}
    )


def _graph_event(
    *,
    graph_signals: tuple[GraphSignalV5, ...],
    graph_seed_names: tuple[str, ...],
    graph_fact_count: int,
    lookup_attempted: bool,
    warning_codes: tuple[str, ...],
    elapsed_ms: float | None,
) -> RetrievalStageEventV5:
    degraded = any(code in {"GRAPH_LOOKUP_TIMEOUT", "GRAPH_LOOKUP_UNAVAILABLE"} for code in warning_codes)
    status_code = "DEGRADED" if degraded else "AVAILABLE" if graph_signals else "EMPTY"
    return RetrievalStageEventV5(
        stage=RetrievalStageV5.GRAPH,
        graph_signals=tuple(
            GraphSignalObservationV5(
                signal_id=signal.signal_id,
                source_entity_id=signal.source_entity_id,
                relation_path=signal.relation_path,
                target_entity_id=signal.target_entity_id,
                medical_claim_eligible=signal.medical_claim_eligible,
            )
            for signal in graph_signals
        ),
        warning_codes=warning_codes,
        elapsed_ms=elapsed_ms,
        summary=RetrievalStageSummaryV5(
            status_code=status_code,
            graph_lookup_attempted=lookup_attempted,
            graph_seed_count=len(graph_seed_names),
            graph_result_count=graph_fact_count,
            graph_signal_count=len(graph_signals),
        ),
    )


def build_reranker_transitions_v5(
    *,
    policy_candidates: Iterable[RetrievedCandidate],
    ranked_evidence: tuple[RankedEvidenceV5, ...],
    reranked_candidates: Iterable[RetrievedCandidate],
    fallback_used: bool,
) -> tuple[CandidateTransitionV5, ...]:
    policy = tuple(policy_candidates)
    ranked_by_id = {
        item.candidate.candidate.candidate_id: item
        for item in ranked_evidence
    }
    output_rank_by_id = {
        candidate.candidate_id: rank
        for rank, candidate in enumerate(reranked_candidates, start=1)
    }
    transitions: list[CandidateTransitionV5] = []
    for input_rank, candidate in enumerate(policy, start=1):
        evidence = ranked_by_id.get(candidate.candidate_id)
        output_rank = output_rank_by_id.get(candidate.candidate_id)
        if fallback_used:
            status = CandidateTransitionStatusV5.FALLBACK_RESTORED
            reason = CandidateTransitionReasonV5.RERANK_FALLBACK
        elif output_rank is None:
            status = CandidateTransitionStatusV5.REMOVED
            reason = CandidateTransitionReasonV5.RERANK_TOP_N_REMOVED
        else:
            status = CandidateTransitionStatusV5.RETAINED
            reason = CandidateTransitionReasonV5.RERANK_RETAINED
        observation = _candidate_observation(candidate)
        transitions.append(
            CandidateTransitionV5(
                candidate_id=candidate.candidate_id,
                source=candidate.source,
                provenance=observation.provenance,
                input_rank=(evidence.input_rank if evidence is not None else input_rank),
                output_rank=output_rank,
                scores=(evidence.scores if evidence is not None else observation.scores),
                status=status,
                reason=reason,
            )
        )
    return tuple(transitions)


def build_selector_transitions_v5(
    *,
    ranked_evidence: tuple[RankedEvidenceV5, ...],
    result: EvidenceSelectionResultV5 | None,
) -> tuple[CandidateTransitionV5, ...]:
    selected_ids = {
        item.evidence.candidate.candidate.candidate_id
        for item in (result.selected_evidence if result is not None else ())
    }
    return tuple(
        CandidateTransitionV5(
            candidate_id=evidence.candidate.candidate.candidate_id,
            source="chunk",
            provenance=evidence.candidate.candidate.provenance,
            input_rank=evidence.output_rank,
            output_rank=(evidence.output_rank if evidence.candidate.candidate.candidate_id in selected_ids else None),
            scores=evidence.scores,
            status=(
                CandidateTransitionStatusV5.SELECTED
                if evidence.candidate.candidate.candidate_id in selected_ids
                else CandidateTransitionStatusV5.NOT_SELECTED
            ),
            reason=(
                CandidateTransitionReasonV5.SELECTOR_SELECTED
                if evidence.candidate.candidate.candidate_id in selected_ids
                else CandidateTransitionReasonV5.SELECTOR_NOT_SELECTED
            ),
        )
        for evidence in ranked_evidence
    )


def _context_observation(
    item: ContextItem,
    *,
    legacy_compat_only: bool = False,
    extra_metadata_features: tuple[str, ...] = (),
) -> CandidateObservationV5:
    return CandidateObservationV5(
        candidate_id=item.item_id,
        source=item.source,
        rank=item.rank,
        scores=ScoreNamespaceV5(
            rrf=_finite_or_none(item.payload.get("rrf_score")),
            legacy_compat_score=_finite_or_none(item.fused_score or item.score),
        ),
        provenance=_mapping_provenance(item.payload, item.item_id),
        metadata_features=tuple(
            _dedupe([*_metadata_features(item.matched_metadata), *extra_metadata_features])
        ),
        legacy_compat_only=legacy_compat_only,
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
    "build_reranker_transitions_v5",
    "build_selector_transitions_v5",
    "compare_v4_to_v5_shadow",
    "query_context_from_legacy",
    "retrieval_pipeline_selection_from_env",
    "retrieval_v5_config_version_from_env",
]
