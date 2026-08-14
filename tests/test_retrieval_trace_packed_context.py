from src.observability.trace_exporter import build_observability_event
from src.retrieval.context_packer import pack_context
from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate


def test_retrieval_trace_and_packed_context_have_one_obvious_contract() -> None:
    query = NormalizedQuery(
        original_query="Mụn đầu đen là gì?",
        normalized_text="Mụn đầu đen là gì?",
        intent="medical_question",
    )
    candidate = RetrievedCandidate(
        candidate_id="chunk-1",
        source="chunk",
        collection="acne_knowledge",
        text="Source-backed acne evidence.",
        fused_score=0.03,
        rank=1,
        payload={"chunk_id": "chunk-1", "source_id": "guideline"},
    )
    packed = pack_context(query, [candidate])
    trace = {
        "architecture": "dense_bm25_rrf",
        "channels": {"dense": {"count": 1}, "bm25": {"count": 1}},
        "selected_ids": ["chunk-1"],
        "warnings": [],
        "elapsed_ms": 3.0,
    }

    event = build_observability_event(
        query=query.original_query,
        state={
            "retrieval_trace": trace,
            "packed_context": packed.model_dump(mode="json"),
            "retrieval_attempt": 1,
            "evidence_assessment": {"usable": True},
        },
    )

    assert event.summary.retrieval_candidates_count == 2
    assert event.summary.packed_context_items_count == 1
    assert event.summary.evidence_usable is True
    assert "rerank" not in event.model_dump_json().lower()
