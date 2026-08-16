from src.retrieval.context_packer import pack_context, packed_context_to_response_contexts
from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate


def _query() -> NormalizedQuery:
    return NormalizedQuery(
        original_query="Adapalene và benzoyl peroxide khác nhau thế nào?",
        normalized_text="Adapalene và benzoyl peroxide khác nhau thế nào?",
    )


def _candidate(
    candidate_id: str, text: str, rank: int, source: str = "source-a"
) -> RetrievedCandidate:
    return RetrievedCandidate(
        candidate_id=candidate_id,
        collection="acne_knowledge",
        text=text,
        score=1.0 / rank,
        fused_score=1.0 / rank,
        rank=rank,
        payload={
            "chunk_id": candidate_id,
            "document_id": source,
            "source_id": source,
            "source_path": f"sample_data/{source}.pdf",
            "header": "Retinoid tại chỗ",
            "active_ingredient": ["adapalene"],
            "drug_class": ["topical retinoid"],
        },
    )


def test_packer_preserves_fused_order_and_provenance() -> None:
    packed = pack_context(
        _query(),
        [_candidate("chunk-1", "Evidence one", 1), _candidate("chunk-2", "Evidence two", 2)],
        max_items=2,
        max_chars=1000,
    )

    assert [item.item_id for item in packed.items] == ["chunk-1", "chunk-2"]
    assert all(item.reason == "rrf_rank" for item in packed.items)
    assert packed.chunk_items_count == 2
    assert "source=source-a" in packed.context_text
    assert "chunk=chunk-1" in packed.context_text
    assert "section=Retinoid tại chỗ" in packed.context_text
    assert "active_ingredient=adapalene" in packed.context_text
    assert "drug_class=topical retinoid" in packed.context_text


def test_packer_exposes_only_bounded_local_scope_without_reordering_text() -> None:
    candidate = _candidate("chunk-1", "Exact evidence text.", 1)
    candidate.payload["arbitrary_internal_metadata"] = "must not enter prompt"
    packed = pack_context(_query(), [candidate], max_chars=1000)

    assert packed.context_text.endswith("\nExact evidence text.")
    assert "arbitrary_internal_metadata" not in packed.context_text
    assert [item.item_id for item in packed.items] == ["chunk-1"]


def test_packer_enforces_explicit_item_and_character_limits() -> None:
    packed = pack_context(
        _query(),
        [
            _candidate("chunk-1", "A" * 500, 1),
            _candidate("chunk-2", "B" * 500, 2),
            _candidate("chunk-3", "C" * 500, 3),
        ],
        max_items=2,
        max_chars=300,
    )

    assert len(packed.items) == 1
    assert len(packed.context_text) <= 300
    assert packed.debug["limits"] == {"max_items": 2, "max_chars": 300}
    assert any(item["reason"] == "character_limit" for item in packed.debug["dropped"])


def test_packer_deduplicates_only_by_stable_candidate_id() -> None:
    first = _candidate("same", "First version", 1)
    duplicate = _candidate("same", "Second version", 2)
    packed = pack_context(_query(), [first, duplicate])

    assert [item.item_id for item in packed.items] == ["same"]
    assert packed.debug["dropped"] == [{"candidate_id": "same", "reason": "duplicate_id"}]


def test_response_context_adapter_retains_internal_source_identifiers() -> None:
    packed = pack_context(_query(), [_candidate("chunk-1", "Evidence", 1)])
    context = packed_context_to_response_contexts(packed)[0]

    assert context["chunk_id"] == "chunk-1"
    assert context["source_id"] == "source-a"
    assert context["source_path"] == "sample_data/source-a.pdf"
    assert context["context_role"] == "medical_evidence"
    assert context["context_pack_reason"] == "rrf_rank"


def test_empty_or_missing_source_text_is_not_prompt_evidence() -> None:
    empty = _candidate("empty", "", 1)
    packed = pack_context(_query(), [empty])

    assert packed.items == []
    assert packed.warnings == ["No usable source evidence was available for the prompt."]
