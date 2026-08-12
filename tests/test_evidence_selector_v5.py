from __future__ import annotations

from src.retrieval.evidence_selector import select_evidence_v5
from src.retrieval.v5_contracts import (
    CandidateProvenanceV5,
    EntitySignalV5,
    EvidenceSufficiencyV5,
    FusedKnowledgeCandidateV5,
    GraphSignalV5,
    KnowledgeCandidateV5,
    QueryContextV5,
    RankedEvidenceV5,
    ScoreNamespaceV5,
)


def _context(*, safety_flags: tuple[str, ...] = ()) -> QueryContextV5:
    return QueryContextV5(
        query_id="query-id",
        original_query="Tazorac có dùng khi mang thai không?",
        retrieval_query="Tazorac có dùng khi mang thai không?",
        intent="safety" if safety_flags else "drug_identity",
        safety_flags=safety_flags,
    )


def _evidence(candidate_id: str, *, metadata_features: tuple[str, ...] = ()) -> RankedEvidenceV5:
    scores = ScoreNamespaceV5(
        dense_similarity=0.8,
        sparse_bm25_score=1.1,
        rrf=0.02,
        reranker_final=0.9,
    )
    candidate = KnowledgeCandidateV5(
        candidate_id=candidate_id,
        provenance=CandidateProvenanceV5(
            chunk_id=candidate_id,
            document_id="document-1",
            source_path="source.pdf",
        ),
        scores=scores,
        metadata_features=metadata_features,
    )
    return RankedEvidenceV5(
        candidate=FusedKnowledgeCandidateV5(
            candidate=candidate,
            rrf_rank=1,
            scores=scores,
        ),
        input_rank=1,
        output_rank=1,
        scores=scores,
    )


def test_selector_preserves_ranked_evidence_order_and_score_namespaces() -> None:
    first = _evidence("first")
    second = _evidence("second")

    result = select_evidence_v5(
        query_context=_context(),
        ranked_evidence=[first, second],
    )

    assert [item.evidence.candidate.candidate.candidate_id for item in result.selected_evidence] == [
        "first",
        "second",
    ]
    assert result.selected_evidence[0].evidence.scores == first.scores
    assert result.status == EvidenceSufficiencyV5.SUFFICIENT
    assert result.selected_evidence[0].roles == ("primary", "source_traceability")


def test_selector_marks_source_backed_pregnancy_evidence_critical_without_dropping_others() -> None:
    safety = _evidence("pregnancy", metadata_features=("safety_context",))
    primary = _evidence("tazarotene")

    result = select_evidence_v5(
        query_context=_context(safety_flags=("pregnancy",)),
        ranked_evidence=[primary, safety],
    )

    assert result.status == EvidenceSufficiencyV5.SUFFICIENT
    assert [item.evidence.candidate.candidate.candidate_id for item in result.selected_evidence] == [
        "tazarotene",
        "pregnancy",
    ]
    assert result.selected_evidence[1].critical is True
    assert {"safety", "critical"} <= set(result.selected_evidence[1].roles)


def test_structural_entity_or_graph_signals_cannot_satisfy_missing_critical_evidence() -> None:
    entity_signal = EntitySignalV5(
        entity_id="active_ingredient:tazarotene",
        canonical_name="tazarotene",
        entity_type="active_ingredient",
        match_confidence=1.0,
        safety_annotations=("pregnancy",),
    )
    graph_signal = GraphSignalV5(
        signal_id="graph:tazarotene:contraindicated",
        source_entity_id=entity_signal.entity_id,
        relation_path=("CONTRAINDICATED_IN", "pregnancy"),
        medical_claim_eligible=False,
    )

    result = select_evidence_v5(
        query_context=_context(safety_flags=("pregnancy",)),
        ranked_evidence=[_evidence("general")],
        entity_signals=[entity_signal],
        graph_signals=[graph_signal],
    )

    assert result.status == EvidenceSufficiencyV5.CRITICAL_EVIDENCE_MISSING
    assert result.entity_signal_count == 1
    assert result.graph_signal_count == 1
    assert result.selected_evidence[0].critical is False


def test_empty_selector_result_is_bounded_insufficient_state() -> None:
    result = select_evidence_v5(query_context=_context(), ranked_evidence=[])

    assert result.status == EvidenceSufficiencyV5.INSUFFICIENT
    assert result.missing_roles == ("primary", "source_traceability")
    assert result.selected_evidence == ()
