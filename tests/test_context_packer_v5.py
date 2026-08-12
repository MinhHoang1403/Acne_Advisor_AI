from __future__ import annotations

from src.retrieval.context_packer_v5 import (
    pack_selected_evidence_v5,
    packed_evidence_to_legacy_context_v5,
)
from src.retrieval.contracts import RetrievedCandidate
from src.retrieval.query_normalization import normalize_query
from src.retrieval.v5_contracts import (
    CandidateProvenanceV5,
    DropReasonV5,
    EvidencePackingStatusV5,
    FusedKnowledgeCandidateV5,
    KnowledgeCandidateV5,
    RankedEvidenceV5,
    ScoreNamespaceV5,
    SelectedEvidenceV5,
)


def _selected(
    candidate_id: str,
    text: str,
    *,
    critical: bool = False,
    roles: tuple[str, ...] = ("primary", "source_traceability"),
) -> SelectedEvidenceV5:
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
            source_path=f"{candidate_id}.md",
        ),
        text=text,
        scores=scores,
    )
    ranked = RankedEvidenceV5(
        candidate=FusedKnowledgeCandidateV5(
            candidate=candidate,
            rrf_rank=1,
            scores=scores,
        ),
        input_rank=1,
        output_rank=1,
        scores=scores,
    )
    return SelectedEvidenceV5(
        evidence=ranked,
        roles=roles,
        selection_reason="test",
        critical=critical,
    )


def test_packer_preserves_selector_order_and_provenance_without_score_mutation() -> None:
    first = _selected("first", "First source-backed evidence.")
    second = _selected("second", "Second source-backed evidence.")

    packed = pack_selected_evidence_v5(
        selected_evidence=[first, second],
        max_items=3,
        max_characters=1000,
        max_tokens=250,
    )

    assert packed.status == EvidencePackingStatusV5.SUFFICIENT
    assert packed.selected_evidence_ids == ("first", "second")
    assert packed.clipped_evidence_ids == ()
    assert packed.omitted_evidence_ids == ()
    assert packed.source_paths == ("first.md", "second.md")
    assert packed.character_count <= packed.max_characters
    assert packed.token_count <= packed.max_tokens
    assert packed.rendered_blocks[0].endswith("First source-backed evidence.")
    assert packed.rendered_blocks[1].endswith("Second source-backed evidence.")


def test_packer_reserves_critical_evidence_before_noncritical_redundancy() -> None:
    noncritical = _selected("redundant", "x" * 280)
    pregnancy = _selected(
        "pregnancy",
        "Tazorac/tazarotene requires pregnancy safety review.",
        critical=True,
        roles=("safety", "critical", "source_traceability"),
    )

    packed = pack_selected_evidence_v5(
        selected_evidence=[noncritical, pregnancy],
        max_items=3,
        max_characters=220,
        max_tokens=55,
    )

    assert packed.critical_evidence_preserved is True
    assert packed.selected_evidence_ids == ("pregnancy",)
    assert packed.critical_evidence_ids == ("pregnancy",)
    assert packed.omitted_evidence_ids == ("redundant",)
    assert packed.status == EvidencePackingStatusV5.OVERFLOW
    assert packed.drops[-1].reason == DropReasonV5.PACKER_BUDGET_REMOVED


def test_packer_returns_explicit_critical_overflow_without_silent_drop() -> None:
    critical = _selected(
        "tazorac-pregnancy",
        "critical " * 80,
        critical=True,
        roles=("safety", "critical", "source_traceability"),
    )

    packed = pack_selected_evidence_v5(
        selected_evidence=[critical],
        max_items=2,
        max_characters=160,
        max_tokens=40,
    )

    assert packed.status == EvidencePackingStatusV5.CRITICAL_EVIDENCE_OVERFLOW
    assert packed.critical_evidence_preserved is False
    assert packed.selected_evidence_ids == ()
    assert packed.omitted_evidence_ids == ("tazorac-pregnancy",)
    assert packed.drops[0].reason == DropReasonV5.PACKER_BUDGET_REMOVED


def test_packer_clips_noncritical_evidence_with_explicit_trace() -> None:
    evidence = _selected("long", "long evidence " * 30)

    packed = pack_selected_evidence_v5(
        selected_evidence=[evidence],
        max_items=2,
        max_characters=180,
        max_tokens=45,
    )

    assert packed.status == EvidencePackingStatusV5.OVERFLOW
    assert packed.selected_evidence_ids == ("long",)
    assert packed.clipped_evidence_ids == ("long",)
    assert packed.omitted_evidence_ids == ()
    assert packed.context_text.endswith("...[truncated]")
    assert packed.drops[0].reason == DropReasonV5.PACKER_CLIPPED


def test_empty_selection_is_bounded_insufficient() -> None:
    packed = pack_selected_evidence_v5(
        selected_evidence=[],
        max_items=2,
        max_characters=200,
        max_tokens=50,
    )

    assert packed.status == EvidencePackingStatusV5.INSUFFICIENT
    assert packed.context_text == ""
    assert packed.selected_evidence_ids == ()


def test_legacy_adapter_keeps_v5_rendered_order_without_legacy_reselection() -> None:
    first = _selected("first", "First evidence.")
    second = _selected("second", "Second evidence.")
    packed = pack_selected_evidence_v5(
        selected_evidence=[first, second],
        max_items=3,
        max_characters=1000,
        max_tokens=250,
    )
    candidates = [
        RetrievedCandidate(
            candidate_id="first",
            source="chunk",
            collection="acne_knowledge",
            text="raw first",
            score=0.7,
            payload={"chunk_id": "first", "source_file": "first.md"},
        ),
        RetrievedCandidate(
            candidate_id="second",
            source="chunk",
            collection="acne_knowledge",
            text="raw second",
            score=0.6,
            payload={"chunk_id": "second", "source_file": "second.md"},
        ),
    ]

    context = packed_evidence_to_legacy_context_v5(
        normalized_query=normalize_query("mụn đầu đen là gì?"),
        selected_evidence=[first, second],
        packed_evidence=packed,
        candidates=candidates,
    )

    assert [item.item_id for item in context.items] == ["first", "second"]
    assert context.context_text == packed.context_text
    assert context.items[0].text == packed.rendered_blocks[0]


def test_historical_r7_sentinels_remain_serialized_when_budget_allows() -> None:
    sentinels = ("đánh giá", "pregnancy", "Tazorac", "bã nhờn", "mồ hôi", "nguồn", "blackhead", "oxidation")
    evidence = [_selected(f"sentinel-{index}", value) for index, value in enumerate(sentinels)]

    packed = pack_selected_evidence_v5(
        selected_evidence=evidence,
        max_items=10,
        max_characters=5000,
        max_tokens=1250,
    )

    assert packed.status == EvidencePackingStatusV5.SUFFICIENT
    for sentinel in sentinels:
        assert sentinel in packed.context_text
