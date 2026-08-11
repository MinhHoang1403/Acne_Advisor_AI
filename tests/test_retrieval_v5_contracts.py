from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.retrieval.contracts import ContextItem, PackedContext, RerankTrace, RetrievedCandidate
from src.retrieval.query_expansion import expand_normalized_query
from src.retrieval.query_normalization import normalize_query
from src.retrieval.v5_compat import (
    RetrievalPipelineSelection,
    build_v5_shadow_trace,
    compare_v4_to_v5_shadow,
    query_context_from_legacy,
    retrieval_pipeline_selection_from_env,
)
from src.retrieval.v5_contracts import (
    CandidateObservationV5,
    RetrievalPipelineVersion,
    RetrievalStageEventV5,
    RetrievalStageV5,
    RetrievalTraceV5,
    ScoreNamespaceV5,
)


def _candidate(candidate_id: str, score: float = 0.2) -> RetrievedCandidate:
    return RetrievedCandidate(
        candidate_id=candidate_id,
        source="chunk",
        collection="acne_knowledge",
        text=f"Evidence text for {candidate_id}",
        score=score,
        fused_score=score,
        payload={
            "chunk_id": candidate_id,
            "document_id": "document-a",
            "source_file": "source-a.pdf",
            "rrf_score": 0.02,
        },
        rank=1,
        matched_metadata={"active_ingredient": ["adapalene"]},
    )


def _packed_context(candidate: RetrievedCandidate) -> PackedContext:
    return PackedContext(
        original_query="Adapalene la gi?",
        intent="drug_identity",
        items=[
            ContextItem(
                item_id=candidate.candidate_id,
                source="chunk",
                role="evidence",
                text=candidate.text,
                payload=candidate.payload,
                score=candidate.score,
                fused_score=candidate.fused_score,
                rank=candidate.rank,
                matched_metadata=candidate.matched_metadata,
                reason="fixture",
            )
        ],
        context_text=candidate.text,
        chunk_items_count=1,
    )


def test_score_namespace_is_finite_and_frozen() -> None:
    score = ScoreNamespaceV5(dense_similarity=0.82, sparse_bm25_score=1.4, rrf=0.02)

    assert score.dense_similarity == 0.82
    with pytest.raises(ValidationError):
        ScoreNamespaceV5(rrf=float("nan"))
    with pytest.raises(ValidationError):
        score.rrf = 0.03  # type: ignore[misc]


def test_trace_append_returns_new_redacted_trace_without_mutating_original() -> None:
    trace = RetrievalTraceV5(
        trace_id="trace-1",
        pipeline_version=RetrievalPipelineVersion.V5_SHADOW,
        config_fingerprint="config",
        query_hash="query-hash",
    )
    observation = CandidateObservationV5(candidate_id="chunk-1", source="chunk")
    updated = trace.append_event(
        RetrievalStageEventV5(
            stage=RetrievalStageV5.DENSE,
            candidates=(observation,),
        )
    )

    assert trace.events == ()
    assert updated.candidate_ids_for(RetrievalStageV5.DENSE) == ("chunk-1",)
    assert "Evidence text" not in json.dumps(updated.model_dump(mode="json"))


def test_query_context_conversion_is_stable_and_preserves_entities() -> None:
    normalized = normalize_query("Tazorac co dung khi mang thai khong?")

    first = query_context_from_legacy(
        original_query=normalized.original_query,
        retrieval_query=normalized.original_query,
        normalized_query=normalized,
    )
    second = query_context_from_legacy(
        original_query=normalized.original_query,
        retrieval_query=normalized.original_query,
        normalized_query=normalized,
    )

    assert first == second
    assert "Tazorac" in first.normalized_entity_ids
    assert first.language == "und"


def test_shadow_adapter_preserves_stage_identity_order_and_context_hash() -> None:
    normalized = normalize_query("Adapalene la gi?")
    expansion = expand_normalized_query(normalized)
    candidate = _candidate("chunk-a")
    packed = _packed_context(candidate)
    dense = [{"id": "chunk-a", "score": 0.82, "source_file": "source-a.pdf"}]
    sparse = [{"id": "chunk-a", "score": 1.2, "source_file": "source-a.pdf"}]
    fused = [
        {
            "id": "chunk-a",
            "score": 0.02,
            "dense_score": 0.82,
            "sparse_score": 1.2,
            "rrf_score": 0.02,
            "source_file": "source-a.pdf",
        }
    ]
    selection = RetrievalPipelineSelection(
        requested=RetrievalPipelineVersion.V5_SHADOW,
        execution=RetrievalPipelineVersion.V4,
        shadow_enabled=True,
    )
    trace = build_v5_shadow_trace(
        original_query=normalized.original_query,
        retrieval_query=normalized.original_query,
        normalized_query=normalized,
        expansion=expansion,
        dense_results=dense,
        sparse_results=sparse,
        fused_results=fused,
        entity_candidates=[],
        chunk_candidates=[candidate],
        merged_candidates=[candidate],
        reranked_candidates=[candidate],
        rerank_trace=RerankTrace(
            provider="fixture",
            enabled=False,
            input_count=1,
            output_count=1,
            top_n=1,
        ),
        packed_context=packed,
        graph_facts=[],
        timings_ms={"dense": 1.0, "sparse": 2.0, "pack": 3.0},
        selection=selection,
    )
    comparison = compare_v4_to_v5_shadow(
        trace=trace,
        dense_results=dense,
        sparse_results=sparse,
        fused_results=fused,
        merged_candidates=[candidate],
        reranked_candidates=[candidate],
        packed_context=packed,
    )

    dense_observation = trace.events[1].candidates[0]
    assert dense_observation.scores.dense_similarity == 0.82
    assert dense_observation.scores.sparse_bm25_score is None
    assert trace.candidate_ids_for(RetrievalStageV5.KNOWLEDGE_FUSION) == ("chunk-a",)
    assert trace.query_observation.intent == "drug_identity"
    assert trace.query_observation.expansion_terms
    assert comparison.equivalent is True
    assert all(stage.equivalent for stage in comparison.stages)
    serialized_trace = json.dumps(trace.model_dump(mode="json"))
    assert candidate.model_dump(mode="json")["text"] not in serialized_trace


def test_pipeline_selection_keeps_v4_default_and_enables_explicit_v5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRIEVAL_PIPELINE_VERSION", "v5")
    requested_v5 = retrieval_pipeline_selection_from_env()
    monkeypatch.setenv("RETRIEVAL_PIPELINE_VERSION", "invalid")
    invalid = retrieval_pipeline_selection_from_env()

    assert requested_v5.execution == RetrievalPipelineVersion.V5
    assert requested_v5.shadow_enabled is True
    assert requested_v5.warning_code is None
    assert invalid.execution == RetrievalPipelineVersion.V4
    assert invalid.warning_code == "INVALID_RETRIEVAL_PIPELINE_VERSION"
