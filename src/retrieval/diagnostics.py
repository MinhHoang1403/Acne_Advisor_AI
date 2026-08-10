"""Evaluation-only retrieval trace contracts for System V4 diagnostics.

The helpers in this module observe candidates after each existing pipeline
stage. They do not participate in retrieval scores, ordering, or selection.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field

from src.retrieval.contracts import (
    ContextItem,
    NormalizedQuery,
    PackedContext,
    QueryExpansion,
    RerankTrace,
    RetrievedCandidate,
)


class KnowledgeLossStage(str, Enum):
    """The first observable stage where an expected concept is unavailable."""

    SOURCE_MISS = "SOURCE_MISS"
    PARSING_MISS = "PARSING_MISS"
    CHUNKING_MISS = "CHUNKING_MISS"
    INDEXING_MISS = "INDEXING_MISS"
    QUERY_UNDERSTANDING_MISS = "QUERY_UNDERSTANDING_MISS"
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    FUSION_MISS = "FUSION_MISS"
    RERANK_MISS = "RERANK_MISS"
    CONTEXT_PACKING_MISS = "CONTEXT_PACKING_MISS"
    GENERATION_MISS = "GENERATION_MISS"
    VERIFIER_MISS = "VERIFIER_MISS"
    EVALUATION_LABEL_ISSUE = "EVALUATION_LABEL_ISSUE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


class CandidateDiagnosticRecord(BaseModel):
    """Compact candidate metadata safe to retain in evaluation artifacts."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    source: str | None = None
    channel: str
    rank: int | None = None
    raw_score: float | None = None
    normalized_score: float | None = None
    chunk_id: str | None = None
    source_id: str | None = None
    source_name: str | None = None
    entity_ids: list[str] = Field(default_factory=list)
    matched_aliases: list[str] = Field(default_factory=list)
    parent_section: str | None = None


class StagePresence(BaseModel):
    """Whether concept evidence exists at one observable pipeline stage."""

    model_config = ConfigDict(extra="forbid")

    present: bool | None = None
    rank: int | None = None
    candidate_ids: list[str] = Field(default_factory=list)


class RetrievalPresence(BaseModel):
    """Channel-level raw-candidate evidence for one expected concept."""

    model_config = ConfigDict(extra="forbid")

    dense: StagePresence = Field(default_factory=StagePresence)
    sparse: StagePresence = Field(default_factory=StagePresence)
    entity: StagePresence = Field(default_factory=StagePresence)
    graph: StagePresence = Field(default_factory=StagePresence)
    candidate: StagePresence = Field(default_factory=StagePresence)


class ConceptKnowledgeTrace(BaseModel):
    """Stage-by-stage evidence for one evaluation expected concept."""

    model_config = ConfigDict(extra="forbid")

    concept: str
    critical: bool = False
    source_present: bool | None = None
    parsed_present: bool | None = None
    chunk_present: bool | None = None
    indexed_present: bool | None = None
    retrieval: RetrievalPresence = Field(default_factory=RetrievalPresence)
    fusion: StagePresence = Field(default_factory=StagePresence)
    rerank: StagePresence = Field(default_factory=StagePresence)
    packed_context: StagePresence = Field(default_factory=StagePresence)
    prompt_evidence: StagePresence = Field(default_factory=StagePresence)
    final_answer_present: bool | None = None
    verifier_detected: bool | None = None
    loss_stage: KnowledgeLossStage | None = None


class QueryDiagnosticTrace(BaseModel):
    """Actual query forms used by the existing retrieval implementation."""

    model_config = ConfigDict(extra="forbid")

    original_query: str
    retrieval_query: str
    normalized_query: str
    rewritten_query: str | None = None
    detected_entities: list[str] = Field(default_factory=list)
    expanded_aliases: list[str] = Field(default_factory=list)
    expanded_terms: list[str] = Field(default_factory=list)


class PromptEvidenceTrace(BaseModel):
    """Prompt-context IDs and per-concept presence without prompt text."""

    model_config = ConfigDict(extra="forbid")

    selected_candidate_ids: list[str] = Field(default_factory=list)
    concept_presence: dict[str, bool] = Field(default_factory=dict)
    item_count: int = 0


class RetrievalDiagnosticTrace(BaseModel):
    """Compact trace emitted only when evaluation diagnostics are enabled."""

    model_config = ConfigDict(extra="forbid")

    version: str = "retrieval_diagnostics_v1"
    enabled: bool = True
    query_trace: QueryDiagnosticTrace
    concept_traces: list[ConceptKnowledgeTrace] = Field(default_factory=list)
    raw_candidate_trace: list[CandidateDiagnosticRecord] = Field(default_factory=list)
    fusion_trace: dict[str, Any] = Field(default_factory=dict)
    rerank_trace: dict[str, Any] = Field(default_factory=dict)
    pack_trace: dict[str, Any] = Field(default_factory=dict)
    prompt_evidence_trace: PromptEvidenceTrace | None = None
    warnings: list[str] = Field(default_factory=list)


ConceptMatcher = Callable[[str, str], bool]


def build_retrieval_diagnostic_trace(
    *,
    expected_concepts: list[str],
    critical: bool,
    original_query: str,
    retrieval_query: str,
    normalized_query: NormalizedQuery,
    expansion: QueryExpansion,
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    entity_candidates: list[RetrievedCandidate],
    fused_results: list[dict[str, Any]],
    chunk_candidates: list[RetrievedCandidate],
    merged_candidates: list[RetrievedCandidate],
    reranked_candidates: list[RetrievedCandidate],
    packed_context: PackedContext,
    rerank_trace: RerankTrace,
    context_max_items: int,
    context_max_chars: int,
    warnings: list[str],
    concept_matcher: ConceptMatcher,
) -> RetrievalDiagnosticTrace:
    """Build trace-only evidence after the existing retrieval result is fixed."""

    concepts = _dedupe(expected_concepts)
    query_trace = QueryDiagnosticTrace(
        original_query=original_query,
        retrieval_query=retrieval_query,
        normalized_query=normalized_query.normalized_text,
        rewritten_query=retrieval_query if retrieval_query != original_query else None,
        detected_entities=_dedupe(
            [
                *normalized_query.drug_product,
                *normalized_query.active_ingredient,
                *normalized_query.drug_class,
                *normalized_query.condition,
            ]
        ),
        expanded_aliases=list(expansion.alias_terms),
        expanded_terms=list(expansion.expanded_terms),
    )
    raw_trace = [
        *(_compact_mapping_candidates(dense_results, "dense")),
        *(_compact_mapping_candidates(sparse_results, "sparse")),
        *(_compact_candidates(entity_candidates, "entity")),
    ]
    fusion_input_ids = [
        *_mapping_candidate_ids(dense_results),
        *_mapping_candidate_ids(sparse_results),
        *_candidate_ids(entity_candidates),
    ]
    fusion_trace = {
        "input_candidate_ids": fusion_input_ids,
        "input_count": len(fusion_input_ids),
        "rrf_candidates": [
            record.model_dump(mode="json")
            for record in _compact_mapping_candidates(fused_results, "rrf")
        ],
        "rrf_output_candidate_ids": _mapping_candidate_ids(fused_results),
        "rrf_output_count": len(fused_results),
        "metadata_boost_candidate_ids": _candidate_ids(chunk_candidates),
        "output_candidate_ids": _candidate_ids(merged_candidates),
        "output_count": len(merged_candidates),
        "dropped_candidate_ids": _dropped_ids(fusion_input_ids, _candidate_ids(merged_candidates)),
        "merged_candidates": [
            record.model_dump(mode="json")
            for record in _compact_candidates(merged_candidates, "merged")
        ],
    }
    rerank_input_ids = _candidate_ids(merged_candidates)
    rerank_output_ids = _candidate_ids(reranked_candidates)
    rerank_trace_data = {
        "enabled": rerank_trace.enabled,
        "provider": rerank_trace.provider,
        "input_candidate_ids": rerank_input_ids,
        "output_candidate_ids": rerank_output_ids,
        "dropped_candidate_ids": _dropped_ids(rerank_input_ids, rerank_output_ids),
        "input_count": rerank_trace.input_count,
        "output_count": rerank_trace.output_count,
        "top_n": rerank_trace.top_n,
        "fallback_used": rerank_trace.fallback_used,
        "ranked_candidates": [
            {
                "candidate_id": item.candidate.candidate_id,
                "input_rank": item.candidate.rank,
                "output_rank": item.rerank_rank,
                "rerank_score": item.rerank_score,
                "score_breakdown": item.score_breakdown.model_dump(mode="json"),
            }
            for item in rerank_trace.ranked_candidates
        ],
        "warnings": list(rerank_trace.warnings),
    }
    pack_trace = dict((packed_context.debug or {}).get("pack_trace") or {})
    pack_trace.setdefault("max_items", context_max_items)
    pack_trace.setdefault("max_chars", context_max_chars)
    pack_trace.setdefault("input_candidate_ids", rerank_output_ids)
    pack_trace.setdefault("selected_candidate_ids", _context_item_ids(packed_context.items))
    pack_trace.setdefault(
        "dropped_candidate_ids",
        _dropped_ids(rerank_output_ids, _context_item_ids(packed_context.items)),
    )
    pack_trace.setdefault("selected_item_count", len(packed_context.items))
    pack_trace.setdefault("actual_context_char_count", len(packed_context.context_text))
    pack_trace.setdefault("selected_sources", sorted({item.source for item in packed_context.items}))
    pack_trace.setdefault(
        "drop_reasons",
        {
            str(item.get("candidate_id")): str(item.get("reason") or "UNKNOWN_DROP_REASON")
            for item in pack_trace.get("dropped_candidates") or []
            if isinstance(item, dict) and item.get("candidate_id")
        },
    )
    concept_traces = [
        _build_concept_trace(
            concept=concept,
            critical=critical,
            dense_results=dense_results,
            sparse_results=sparse_results,
            entity_candidates=entity_candidates,
            merged_candidates=merged_candidates,
            reranked_candidates=reranked_candidates,
            packed_items=packed_context.items,
            concept_matcher=concept_matcher,
        )
        for concept in concepts
    ]
    return RetrievalDiagnosticTrace(
        query_trace=query_trace,
        concept_traces=concept_traces,
        raw_candidate_trace=raw_trace,
        fusion_trace=fusion_trace,
        rerank_trace=rerank_trace_data,
        pack_trace=pack_trace,
        warnings=list(warnings),
    )


def build_prompt_evidence_trace(
    *,
    expected_concepts: list[str],
    contexts: list[dict[str, Any]],
    concept_matcher: ConceptMatcher,
) -> PromptEvidenceTrace:
    """Describe the contexts actually forwarded to prompt construction."""

    return PromptEvidenceTrace(
        selected_candidate_ids=_context_mapping_ids(contexts),
        concept_presence={
            concept: any(concept_matcher(_mapping_text(context), concept) for context in contexts)
            for concept in _dedupe(expected_concepts)
        },
        item_count=len(contexts),
    )


def _build_concept_trace(
    *,
    concept: str,
    critical: bool,
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    entity_candidates: list[RetrievedCandidate],
    merged_candidates: list[RetrievedCandidate],
    reranked_candidates: list[RetrievedCandidate],
    packed_items: list[ContextItem],
    concept_matcher: ConceptMatcher,
) -> ConceptKnowledgeTrace:
    dense = _mapping_presence(dense_results, concept, concept_matcher)
    sparse = _mapping_presence(sparse_results, concept, concept_matcher)
    entity = _candidate_presence(entity_candidates, concept, concept_matcher)
    candidate = _combined_presence((dense, sparse, entity))
    fusion = _candidate_presence(merged_candidates, concept, concept_matcher)
    rerank = _candidate_presence(reranked_candidates, concept, concept_matcher)
    packed = _context_presence(packed_items, concept, concept_matcher)
    return ConceptKnowledgeTrace(
        concept=concept,
        critical=critical,
        # A retrieved hit proves neither corpus-wide chunk presence nor index
        # completeness. Those stages remain unknown until cheap provenance
        # reconstruction is available.
        retrieval=RetrievalPresence(dense=dense, sparse=sparse, entity=entity, candidate=candidate),
        fusion=fusion,
        rerank=rerank,
        packed_context=packed,
    )


def _mapping_presence(
    candidates: Iterable[dict[str, Any]],
    concept: str,
    matcher: ConceptMatcher,
) -> StagePresence:
    matches = [
        (rank, _mapping_candidate_id(candidate))
        for rank, candidate in enumerate(candidates, start=1)
        if matcher(_mapping_text(candidate), concept)
    ]
    return _presence_from_matches(matches)


def _candidate_presence(
    candidates: Iterable[RetrievedCandidate],
    concept: str,
    matcher: ConceptMatcher,
) -> StagePresence:
    matches = [
        (candidate.rank or rank, candidate.candidate_id)
        for rank, candidate in enumerate(candidates, start=1)
        if matcher(candidate.text, concept)
    ]
    return _presence_from_matches(matches)


def _context_presence(
    items: Iterable[ContextItem],
    concept: str,
    matcher: ConceptMatcher,
) -> StagePresence:
    matches = [
        (item.rank or rank, item.item_id)
        for rank, item in enumerate(items, start=1)
        if matcher(item.text, concept)
    ]
    return _presence_from_matches(matches)


def _combined_presence(values: Iterable[StagePresence]) -> StagePresence:
    matches = [(value.rank, candidate_id) for value in values for candidate_id in value.candidate_ids]
    return _presence_from_matches([(rank or 0, candidate_id) for rank, candidate_id in matches])


def _presence_from_matches(matches: list[tuple[int, str]]) -> StagePresence:
    if not matches:
        return StagePresence(present=False)
    ordered = sorted(matches, key=lambda item: item[0])
    return StagePresence(present=True, rank=ordered[0][0] or None, candidate_ids=_dedupe([item[1] for item in ordered])[:3])


def _compact_mapping_candidates(
    candidates: Iterable[dict[str, Any]],
    channel: str,
) -> list[CandidateDiagnosticRecord]:
    return [
        _compact_mapping_candidate(candidate, channel, rank)
        for rank, candidate in enumerate(candidates, start=1)
    ]


def _compact_mapping_candidate(
    candidate: dict[str, Any],
    channel: str,
    rank: int,
) -> CandidateDiagnosticRecord:
    return CandidateDiagnosticRecord(
        candidate_id=_mapping_candidate_id(candidate),
        source=str(candidate.get("retrieval_source") or "chunk"),
        channel=channel,
        rank=rank,
        raw_score=_first_float(candidate.get("dense_score"), candidate.get("sparse_score"), candidate.get("score")),
        normalized_score=_as_float(candidate.get("rrf_score")),
        chunk_id=_optional_string(candidate.get("chunk_id")),
        source_id=_optional_string(candidate.get("document_id") or candidate.get("source_id")),
        source_name=_optional_string(candidate.get("source_file") or candidate.get("source_path")),
        entity_ids=_strings(candidate.get("entity_ids") or candidate.get("graph_nodes")),
        matched_aliases=_strings(candidate.get("matched_aliases")),
        parent_section=_optional_string(candidate.get("header") or candidate.get("parent_header_path")),
    )


def _compact_candidates(
    candidates: Iterable[RetrievedCandidate],
    channel: str,
) -> list[CandidateDiagnosticRecord]:
    records: list[CandidateDiagnosticRecord] = []
    for rank, candidate in enumerate(candidates, start=1):
        payload = candidate.payload
        records.append(
            CandidateDiagnosticRecord(
                candidate_id=candidate.candidate_id,
                source=candidate.source,
                channel=channel,
                rank=candidate.rank or rank,
                raw_score=_as_float(candidate.score),
                normalized_score=_as_float(candidate.fused_score),
                chunk_id=_optional_string(payload.get("chunk_id")),
                source_id=_optional_string(payload.get("document_id") or payload.get("source_id") or payload.get("entity_id")),
                source_name=_optional_string(payload.get("source_file") or payload.get("source_path")),
                entity_ids=_strings(payload.get("entity_ids") or payload.get("graph_nodes") or payload.get("entity_id")),
                matched_aliases=_strings(candidate.matched_metadata.get("aliases") if candidate.matched_metadata else None),
                parent_section=_optional_string(payload.get("header") or payload.get("parent_header_path")),
            )
        )
    return records


def _mapping_candidate_id(candidate: dict[str, Any]) -> str:
    for key in ("id", "chunk_id", "point_id", "document_id"):
        if candidate.get(key):
            return str(candidate[key])
    return "unknown-candidate"


def _mapping_text(candidate: dict[str, Any]) -> str:
    return str(candidate.get("text") or candidate.get("content") or "")


def _candidate_ids(candidates: Iterable[RetrievedCandidate]) -> list[str]:
    return [candidate.candidate_id for candidate in candidates]


def _mapping_candidate_ids(candidates: Iterable[dict[str, Any]]) -> list[str]:
    return [_mapping_candidate_id(candidate) for candidate in candidates]


def _context_item_ids(items: Iterable[ContextItem]) -> list[str]:
    return [item.item_id for item in items]


def _context_mapping_ids(contexts: Iterable[dict[str, Any]]) -> list[str]:
    return [
        str(context.get("chunk_id") or context.get("entity_id") or context.get("point_id") or context.get("id") or "unknown-context")
        for context in contexts
    ]


def _dropped_ids(input_ids: list[str], output_ids: list[str]) -> list[str]:
    selected = set(output_ids)
    return [candidate_id for candidate_id in input_ids if candidate_id not in selected]


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _first_float(*values: Any) -> float | None:
    for value in values:
        converted = _as_float(value)
        if converted is not None:
            return converted
    return None


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None and value != "" else None


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return _dedupe([str(item) for item in value if item])
    return [str(value)] if value else []


def _dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item.casefold() in seen:
            continue
        seen.add(item.casefold())
        output.append(item)
    return output


__all__ = [
    "CandidateDiagnosticRecord",
    "ConceptKnowledgeTrace",
    "KnowledgeLossStage",
    "PromptEvidenceTrace",
    "QueryDiagnosticTrace",
    "RetrievalDiagnosticTrace",
    "build_prompt_evidence_trace",
    "build_retrieval_diagnostic_trace",
]
