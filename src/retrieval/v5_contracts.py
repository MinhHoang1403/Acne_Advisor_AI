"""Versioned, immutable retrieval contracts for the staged V5 migration.

R1 intentionally keeps these contracts outside the legacy execution path.  They
describe observations made by compatibility adapters and never mutate a V4
candidate, score, rank, or context.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


RETRIEVAL_V5_CONTRACT_VERSION = "retrieval_v5_contracts_v1"
TRACE_EVENT_LIMIT = 64
TRACE_CANDIDATE_LIMIT = 256


class RetrievalPipelineVersion(str, Enum):
    """Selectable retrieval execution modes during the V5 migration."""

    V4 = "v4"
    V5_SHADOW = "v5_shadow"
    V5 = "v5"


class RetrievalStageV5(str, Enum):
    """Stable stage names for bounded V5 observations."""

    QUERY_UNDERSTANDING = "QUERY_UNDERSTANDING"
    DENSE = "DENSE"
    SPARSE = "SPARSE"
    ENTITY_RESOLUTION = "ENTITY_RESOLUTION"
    KNOWLEDGE_FUSION = "KNOWLEDGE_FUSION"
    LEGACY_METADATA = "LEGACY_METADATA"
    LEGACY_CANDIDATE_MERGE = "LEGACY_CANDIDATE_MERGE"
    RERANK = "RERANK"
    PACKER = "PACKER"
    GRAPH = "GRAPH"


class DropReasonV5(str, Enum):
    """Stable reasons used by later V5 stages for candidate removal."""

    RAW_DENSE_MISS = "RAW_DENSE_MISS"
    RAW_SPARSE_MISS = "RAW_SPARSE_MISS"
    RAW_ALL_CHANNELS_MISS = "RAW_ALL_CHANNELS_MISS"
    FUSION_DROPPED = "FUSION_DROPPED"
    DEDUPE_REMOVED = "DEDUPE_REMOVED"
    SOURCE_DIVERSITY_REMOVED = "SOURCE_DIVERSITY_REMOVED"
    CANDIDATE_BUDGET_REMOVED = "CANDIDATE_BUDGET_REMOVED"
    RERANK_REMOVED = "RERANK_REMOVED"
    RERANK_FALLBACK = "RERANK_FALLBACK"
    EVIDENCE_SELECTOR_REMOVED = "EVIDENCE_SELECTOR_REMOVED"
    PACKER_BUDGET_REMOVED = "PACKER_BUDGET_REMOVED"
    PACKER_DUPLICATE_REMOVED = "PACKER_DUPLICATE_REMOVED"
    PACKER_CLIPPED = "PACKER_CLIPPED"
    ENTITY_ONLY_SIGNAL = "ENTITY_ONLY_SIGNAL"
    GRAPH_ONLY_SIGNAL = "GRAPH_ONLY_SIGNAL"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QueryContextV5(_FrozenModel):
    """Immutable query-understanding output consumed by V5 stages."""

    schema_version: str = RETRIEVAL_V5_CONTRACT_VERSION
    query_id: str
    original_query: str
    retrieval_query: str
    intent: str
    language: str = "unknown"
    normalized_entity_ids: tuple[str, ...] = ()
    safety_flags: tuple[str, ...] = ()


class QueryObservationV5(_FrozenModel):
    """Redacted query-understanding details safe to include in a trace."""

    intent: str = ""
    language: str = "unknown"
    normalized_entity_ids: tuple[str, ...] = ()
    safety_flags: tuple[str, ...] = ()
    expansion_terms: tuple[str, ...] = ()


class ScoreNamespaceV5(_FrozenModel):
    """Namespaced score observations; fields are never overwritten in V5."""

    dense_similarity: float | None = None
    sparse_bm25_score: float | None = None
    rrf: float | None = None
    reranker_semantic: float | None = None
    reranker_rule: float | None = None
    reranker_final: float | None = None
    legacy_compat_score: float | None = None

    @field_validator(
        "dense_similarity",
        "sparse_bm25_score",
        "rrf",
        "reranker_semantic",
        "reranker_rule",
        "reranker_final",
        "legacy_compat_score",
    )
    @classmethod
    def _scores_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("score observations must be finite")
        return value


class CandidateProvenanceV5(_FrozenModel):
    """Stable infrastructure and document provenance without candidate text."""

    point_id: str | None = None
    chunk_id: str | None = None
    document_id: str | None = None
    source_path: str | None = None


class CandidateObservationV5(_FrozenModel):
    """Redacted candidate observation for an append-only trace."""

    candidate_id: str
    source: str
    collection: str | None = None
    rank: int | None = None
    scores: ScoreNamespaceV5 = Field(default_factory=ScoreNamespaceV5)
    provenance: CandidateProvenanceV5 = Field(default_factory=CandidateProvenanceV5)
    metadata_features: tuple[str, ...] = ()
    legacy_compat_only: bool = False


class KnowledgeCandidateV5(_FrozenModel):
    """Source-backed chunk candidate, before knowledge-channel fusion."""

    candidate_id: str
    provenance: CandidateProvenanceV5
    scores: ScoreNamespaceV5
    metadata_features: tuple[str, ...] = ()


class EntitySignalV5(_FrozenModel):
    """Structural side-channel signal. R2 will populate it from entity lookup."""

    entity_id: str
    canonical_name: str
    entity_type: str
    matched_terms: tuple[str, ...] = ()
    match_confidence: float
    graph_seed_ids: tuple[str, ...] = ()
    safety_annotations: tuple[str, ...] = ()

    @field_validator("match_confidence")
    @classmethod
    def _confidence_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("entity match confidence must be finite")
        return value


class GraphSignalV5(_FrozenModel):
    """Relational side-channel signal. R2 will populate it from graph lookup."""

    signal_id: str
    source_entity_id: str
    relation_path: tuple[str, ...]
    target_entity_id: str | None = None
    path_confidence: float | None = None
    source_chunk_ids: tuple[str, ...] = ()

    @field_validator("path_confidence")
    @classmethod
    def _path_confidence_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("graph path confidence must be finite")
        return value


class FusedKnowledgeCandidateV5(_FrozenModel):
    """Knowledge-only RRF output retaining all upstream observations."""

    candidate: KnowledgeCandidateV5
    rrf_rank: int
    scores: ScoreNamespaceV5


class RankedEvidenceV5(_FrozenModel):
    """Reranker output retaining immutable source/fusion semantics."""

    candidate: FusedKnowledgeCandidateV5
    input_rank: int
    output_rank: int
    scores: ScoreNamespaceV5
    fallback_used: bool = False


class SelectedEvidenceV5(_FrozenModel):
    """Coverage-oriented evidence selection output for R6."""

    evidence: RankedEvidenceV5
    roles: tuple[str, ...]
    selection_reason: str
    critical: bool = False


class PackedEvidenceV5(_FrozenModel):
    """Budgeted prompt serialization output for R7."""

    selected_evidence_ids: tuple[str, ...]
    clipped_evidence_ids: tuple[str, ...] = ()
    omitted_evidence_ids: tuple[str, ...] = ()
    character_count: int = 0
    max_characters: int = 0
    source_paths: tuple[str, ...] = ()
    critical_evidence_preserved: bool = True


class CandidateDropV5(_FrozenModel):
    """One explicit candidate removal observation."""

    candidate_id: str
    reason: DropReasonV5


class RetrievalStageEventV5(_FrozenModel):
    """One immutable, redacted event in a bounded retrieval trace."""

    stage: RetrievalStageV5
    candidates: tuple[CandidateObservationV5, ...] = ()
    drops: tuple[CandidateDropV5, ...] = ()
    warning_codes: tuple[str, ...] = ()
    elapsed_ms: float | None = None
    context_sha256: str | None = None

    @field_validator("candidates")
    @classmethod
    def _candidate_count_is_bounded(
        cls,
        value: tuple[CandidateObservationV5, ...],
    ) -> tuple[CandidateObservationV5, ...]:
        if len(value) > TRACE_CANDIDATE_LIMIT:
            raise ValueError(f"trace candidates exceed limit {TRACE_CANDIDATE_LIMIT}")
        return value

    @field_validator("elapsed_ms")
    @classmethod
    def _elapsed_time_is_finite(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("trace elapsed time must be finite and non-negative")
        return value


class ShadowStageComparisonV5(_FrozenModel):
    """Exact ID/order comparison between legacy and passive V5 shadow output."""

    stage: str
    legacy_candidate_ids: tuple[str, ...]
    shadow_candidate_ids: tuple[str, ...]
    equivalent: bool


class ShadowComparisonV5(_FrozenModel):
    """V4/V5-shadow equivalence result for R1 structural validation."""

    stages: tuple[ShadowStageComparisonV5, ...]
    legacy_context_sha256: str
    shadow_context_sha256: str
    equivalent: bool


class RetrievalTraceV5(_FrozenModel):
    """Append-only, redaction-aware V5 retrieval trace."""

    schema_version: str = RETRIEVAL_V5_CONTRACT_VERSION
    trace_id: str
    pipeline_version: RetrievalPipelineVersion
    config_fingerprint: str
    query_hash: str
    query_observation: QueryObservationV5 = Field(default_factory=QueryObservationV5)
    events: tuple[RetrievalStageEventV5, ...] = ()
    shadow_comparison: ShadowComparisonV5 | None = None

    @field_validator("events")
    @classmethod
    def _event_count_is_bounded(
        cls,
        value: tuple[RetrievalStageEventV5, ...],
    ) -> tuple[RetrievalStageEventV5, ...]:
        if len(value) > TRACE_EVENT_LIMIT:
            raise ValueError(f"trace events exceed limit {TRACE_EVENT_LIMIT}")
        return value

    def append_event(self, event: RetrievalStageEventV5) -> "RetrievalTraceV5":
        """Return a new trace; an existing trace remains unchanged."""

        if len(self.events) >= TRACE_EVENT_LIMIT:
            raise ValueError(f"trace events exceed limit {TRACE_EVENT_LIMIT}")
        return self.model_copy(update={"events": (*self.events, event)})

    def candidate_ids_for(self, stage: RetrievalStageV5) -> tuple[str, ...]:
        """Return ordered candidate IDs observed for one stage."""

        for event in self.events:
            if event.stage == stage:
                return tuple(candidate.candidate_id for candidate in event.candidates)
        return ()


__all__ = [
    "CandidateDropV5",
    "CandidateObservationV5",
    "CandidateProvenanceV5",
    "DropReasonV5",
    "EntitySignalV5",
    "FusedKnowledgeCandidateV5",
    "GraphSignalV5",
    "KnowledgeCandidateV5",
    "PackedEvidenceV5",
    "QueryContextV5",
    "QueryObservationV5",
    "RETRIEVAL_V5_CONTRACT_VERSION",
    "RankedEvidenceV5",
    "RetrievalPipelineVersion",
    "RetrievalStageEventV5",
    "RetrievalStageV5",
    "RetrievalTraceV5",
    "ScoreNamespaceV5",
    "SelectedEvidenceV5",
    "ShadowComparisonV5",
    "ShadowStageComparisonV5",
]
