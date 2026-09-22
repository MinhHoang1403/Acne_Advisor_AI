from src.observability.trace_exporter import build_observability_event
from src.retrieval.context_packer import pack_context
from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate
from src.retrieval.service import _fused_candidate_trace


def test_retrieval_trace_and_packed_context_have_one_obvious_contract() -> None:
    query = NormalizedQuery(
        original_query="Mụn đầu đen là gì?",
        normalized_text="Mụn đầu đen là gì?",
    )
    candidate = RetrievedCandidate(
        candidate_id="chunk-1",
        collection="acne_knowledge",
        text="Source-backed acne evidence.",
        fused_score=0.03,
        rank=1,
        payload={"chunk_id": "chunk-1", "source_id": "guideline"},
    )
    packed = pack_context(query, [candidate])
    trace = {
        "architecture": "dense_bm25_rrf_local_reranker",
        "channels": {"dense": {"count": 1}, "bm25": {"count": 1}},
        "reranker": {
            "enabled": True,
            "status": "succeeded",
            "model": "fake-reranker",
            "fallback_used": False,
        },
        "selected_ids": ["chunk-1"],
        "warnings": [],
        "elapsed_ms": 3.0,
    }

    event = build_observability_event(
        query=query.original_query,
        request_id="8d88ca48-18e3-45b9-a08d-b034d85ef71d",
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
    assert event.summary.metadata["reranker"] == trace["reranker"]


def test_fused_trace_distinguishes_dropped_from_not_examined_candidates() -> None:
    query = NormalizedQuery(original_query="query", normalized_text="query")
    candidate_texts = ["X" * 500, "selected evidence", "unexamined evidence"]
    candidates = [
        RetrievedCandidate(
            candidate_id=f"chunk-{index}",
            collection="acne_knowledge",
            text=candidate_texts[index - 1],
            rank=index,
            payload={"chunk_id": f"chunk-{index}", "source_id": "guideline"},
        )
        for index in range(1, 4)
    ]

    packed = pack_context(query, candidates, max_items=1, max_chars=256)
    fused = _fused_candidate_trace(candidates, packed)

    assert [item["packing_status"] for item in fused] == [
        "dropped",
        "selected",
        "not_examined",
    ]
    assert fused[0]["drop_reason"] == "character_limit"
    assert fused[0]["not_examined_reason"] is None
    assert fused[2]["drop_reason"] is None
    assert fused[2]["not_examined_reason"] == "item_limit_reached"
    assert packed.debug["selected_ids"] == ["chunk-2"]
    assert packed.context_text.endswith("\nselected evidence")
