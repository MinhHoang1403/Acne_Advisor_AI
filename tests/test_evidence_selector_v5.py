from __future__ import annotations

from src.retrieval.evidence_selector import select_evidence_v5
from src.retrieval.v5_compat import build_selector_transitions_v5
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


def test_tazorac_graph_relation_adds_source_backed_drug_class_requirement() -> None:
    graph_signal = GraphSignalV5(
        signal_id="graph:tazarotene:class",
        source_entity_id="active_ingredient:tazarotene",
        relation_path=("BELONGS_TO_CLASS", "topical_retinoid"),
        target_entity_id="drug_class:topical_retinoid",
        medical_claim_eligible=False,
    )
    missing = select_evidence_v5(
        query_context=_context(),
        ranked_evidence=[_evidence("tazorac-source")],
        graph_signals=[graph_signal],
    )
    covered = select_evidence_v5(
        query_context=_context(),
        ranked_evidence=[_evidence("tazorac-class-source", metadata_features=("drug_class",))],
        graph_signals=[graph_signal],
    )

    assert missing.requirements.graph_required_roles == ("drug_class",)
    assert missing.status == EvidenceSufficiencyV5.INSUFFICIENT
    assert "drug_class" in missing.missing_roles
    assert graph_signal.medical_claim_eligible is False
    assert covered.status == EvidenceSufficiencyV5.SUFFICIENT
    assert "drug_class" in covered.satisfied_roles


def test_pregnancy_graph_requirement_still_needs_critical_source_evidence() -> None:
    graph_signal = GraphSignalV5(
        signal_id="graph:tazarotene:pregnancy",
        source_entity_id="active_ingredient:tazarotene",
        relation_path=("CONTRAINDICATED_IN", "pregnancy"),
        target_entity_id="safety_context:pregnancy",
        medical_claim_eligible=False,
    )
    no_source = select_evidence_v5(
        query_context=_context(safety_flags=("pregnancy",)),
        ranked_evidence=[_evidence("general")],
        graph_signals=[graph_signal],
    )
    source_backed = select_evidence_v5(
        query_context=_context(safety_flags=("pregnancy",)),
        ranked_evidence=[_evidence("pregnancy-source", metadata_features=("contraindications",))],
        graph_signals=[graph_signal],
    )

    assert no_source.status == EvidenceSufficiencyV5.CRITICAL_EVIDENCE_MISSING
    assert no_source.graph_signal_count == 1
    assert source_backed.status == EvidenceSufficiencyV5.SUFFICIENT
    assert source_backed.selected_evidence[0].critical is True


def test_graph_unavailable_preserves_source_backed_selector_behavior() -> None:
    result = select_evidence_v5(
        query_context=_context(safety_flags=("pregnancy",)),
        ranked_evidence=[_evidence("pregnancy-source", metadata_features=("safety_context",))],
        graph_signals=[],
    )

    assert result.status == EvidenceSufficiencyV5.SUFFICIENT
    assert result.graph_signal_count == 0


def test_selector_trace_marks_selected_and_neutral_not_selected_transitions() -> None:
    first = _evidence("first")
    second = _evidence("second")
    complete = select_evidence_v5(query_context=_context(), ranked_evidence=[first, second])
    partial = complete.model_copy(update={"selected_evidence": complete.selected_evidence[:1]})
    transitions = build_selector_transitions_v5(
        ranked_evidence=(first, second),
        result=partial,
    )

    assert transitions[0].status.value == "SELECTED"
    assert transitions[1].status.value == "NOT_SELECTED"
    assert transitions[1].reason.value == "SELECTOR_NOT_SELECTED"
