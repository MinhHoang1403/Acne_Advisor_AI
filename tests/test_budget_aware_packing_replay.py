from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from research.budget_aware_packing.baseline_replay import (
    KB_BUILD_ID,
    ReplayInputError,
    _compare,
    _validate_attempt_traces,
    canonical_json_sha256,
    load_frozen_inputs,
    reconstruct_candidate,
    sha256_bytes,
    sha256_text,
    validate_frozen_identity,
)
from src.retrieval.context_packer import _render_block, pack_context
from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate
from src.retrieval.service import merge_retrieval_candidates


def _query() -> NormalizedQuery:
    return NormalizedQuery(original_query="query", normalized_text="query")


def _payload(candidate_id: str, *, text: str = "evidence", source_id: str = "source") -> dict:
    return {
        "build_id": KB_BUILD_ID,
        "chunk_id": candidate_id,
        "source_id": source_id,
        "section_path": ["section"],
        "text": text,
    }


def _trace(candidate_id: str, **updates: object) -> dict:
    value = {
        "candidate_id": candidate_id,
        "source_id": "source",
        "section": "section",
        "rank": 1,
        "fused_score": 1.0,
    }
    value.update(updates)
    return value


def _candidate(
    candidate_id: str,
    text: str = "evidence",
    *,
    source_id: str = "source",
    rerank_rank: int | None = None,
    rank: int | None = None,
) -> RetrievedCandidate:
    return RetrievedCandidate(
        candidate_id=candidate_id,
        collection="acne_knowledge",
        text=text,
        fused_score=1.0,
        rerank_rank=rerank_rank,
        rank=rank,
        payload=_payload(candidate_id, text=text, source_id=source_id),
    )


@pytest.fixture(scope="module")
def frozen_inputs():
    return load_frozen_inputs(Path.cwd())


def test_correct_frozen_identity(frozen_inputs) -> None:
    assert all(validate_frozen_identity(frozen_inputs).values())


def test_checksum_mismatch_fails_closed(frozen_inputs) -> None:
    altered = replace(frozen_inputs, raw_bytes=b"{}\n")
    with pytest.raises(ReplayInputError, match="raw_bytes_match"):
        validate_frozen_identity(altered)


def test_missing_candidate_text_fails_reconstruction() -> None:
    with pytest.raises(ReplayInputError, match="missing text"):
        reconstruct_candidate(_trace("chunk"), _payload("chunk", text=""))


def test_missing_provenance_fails_reconstruction() -> None:
    payload = _payload("chunk")
    payload.pop("source_id")
    with pytest.raises(ReplayInputError, match="missing provenance"):
        reconstruct_candidate(_trace("chunk", source_id=None), payload)


def test_duplicate_candidate_id_fails_attempt_validation() -> None:
    with pytest.raises(ReplayInputError, match="duplicate candidate ID"):
        _validate_attempt_traces([_trace("same"), _trace("same")])


def test_different_candidate_text_hash_fails_reconstruction() -> None:
    with pytest.raises(ReplayInputError, match="text checksum mismatch"):
        reconstruct_candidate(_trace("chunk", text_sha256="0" * 64), _payload("chunk"))


def test_wrong_selected_order_is_recorded_as_mismatch() -> None:
    mismatches: list[dict] = []
    _compare(
        mismatches,
        case_id="case",
        attempt_index=1,
        field="selected_ids",
        expected=["a", "b"],
        actual=["b", "a"],
    )
    assert mismatches == [
        {
            "case_id": "case",
            "attempt_index": 1,
            "field": "selected_ids",
            "expected": ["a", "b"],
            "actual": ["b", "a"],
        }
    ]


def test_rerank_and_fallback_order_are_replayed_as_supplied() -> None:
    reranked = [_candidate("b", rerank_rank=2), _candidate("a", rerank_rank=1)]
    packed = pack_context(_query(), reranked)
    assert [item.item_id for item in packed.items] == ["b", "a"]
    assert {item.reason for item in packed.items} == {"rerank_rank"}

    fallback = [_candidate("d", rank=2), _candidate("c", rank=1)]
    packed_fallback = pack_context(_query(), fallback)
    assert [item.item_id for item in packed_fallback.items] == ["d", "c"]
    assert {item.reason for item in packed_fallback.items} == {"rrf_rank"}


def test_rendered_header_mismatch_changes_rendered_identity() -> None:
    first = _render_block(_candidate("chunk", source_id="source-a"), 1)
    second = _render_block(_candidate("chunk", source_id="source-b"), 1)
    assert first != second
    assert sha256_text(first) != sha256_text(second)


def test_separator_counting_is_exact() -> None:
    candidates = [_candidate("a", "one"), _candidate("b", "two")]
    packed = pack_context(_query(), candidates)
    assert (
        len(packed.context_text)
        == sum(
            len(_render_block(candidate, index)) for index, candidate in enumerate(candidates, 1)
        )
        + 2
    )


def test_exact_7000_character_boundary() -> None:
    seed = _candidate("boundary", "x")
    overhead = len(_render_block(seed, 1)) - 1
    exact = _candidate("boundary", "x" * (7000 - overhead))
    assert len(pack_context(_query(), [exact], max_chars=7000).context_text) == 7000

    too_large = _candidate("boundary", "x" * (7001 - overhead))
    rejected = pack_context(_query(), [too_large], max_chars=7000)
    assert not rejected.items
    assert rejected.debug["dropped"][0]["reason"] == "character_limit"


def test_nine_item_boundary() -> None:
    packed = pack_context(_query(), [_candidate(str(index)) for index in range(10)])
    assert len(packed.items) == 9
    assert packed.debug["not_examined"] == [{"candidate_id": "9", "reason": "item_limit_reached"}]


def test_retry_union_retains_unique_candidates_and_attempt_metadata() -> None:
    retained = [_candidate("a", rank=1), _candidate("shared", rank=2)]
    acquired = [_candidate("shared", rank=1), _candidate("b", rank=2)]
    merged, duplicates = merge_retrieval_candidates(retained, acquired, retrieval_attempt=2)
    by_id = {candidate.candidate_id: candidate for candidate in merged}
    assert set(by_id) == {"a", "shared", "b"}
    assert duplicates == ["shared"]
    assert by_id["shared"].debug["seen_in_attempts"] == [1, 2]


def test_character_limit_drop_reason() -> None:
    packed = pack_context(_query(), [_candidate("large", "x" * 400)], max_chars=256)
    assert packed.debug["dropped"] == [{"candidate_id": "large", "reason": "character_limit"}]


def test_item_limit_not_examined_reason() -> None:
    packed = pack_context(
        _query(),
        [_candidate(str(index)) for index in range(4)],
        max_items=2,
    )
    assert packed.debug["not_examined"] == [
        {"candidate_id": "2", "reason": "item_limit_reached"},
        {"candidate_id": "3", "reason": "item_limit_reached"},
    ]


def test_context_hash_equality_for_identical_replay() -> None:
    candidates = [_candidate("a", "one"), _candidate("b", "two")]
    first = pack_context(_query(), candidates).context_text
    second = pack_context(_query(), candidates).context_text
    assert sha256_text(first) == sha256_text(second)


def test_raw_lf_crlf_handling_distinguishes_bytes_not_canonical_json() -> None:
    lf = b'{"a": 1}\n'
    crlf = b'{"a": 1}\r\n'
    assert sha256_bytes(lf) != sha256_bytes(crlf)
    assert canonical_json_sha256(json.loads(lf)) == canonical_json_sha256(json.loads(crlf))


def test_reconstructed_source_mismatch_fails_closed() -> None:
    with pytest.raises(ReplayInputError, match="source provenance mismatch"):
        reconstruct_candidate(_trace("chunk", source_id="wrong"), _payload("chunk"))
