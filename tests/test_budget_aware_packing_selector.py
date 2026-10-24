from __future__ import annotations

import inspect
import itertools
import sqlite3
import urllib.request
from pathlib import Path

import pytest

from research.budget_aware_packing.budget_aware_selector import (
    SELECTOR_NAME,
    SelectionInputError,
    _Solution,
    _is_better,
    rank_utility,
    select_budget_aware,
)
from src.retrieval.context_packer import _render_block, pack_context
from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate


def _query() -> NormalizedQuery:
    return NormalizedQuery(original_query="query", normalized_text="query")


def _candidate(
    candidate_id: str,
    text: str = "evidence",
    *,
    source_id: str = "source",
    rerank_score: float | None = None,
    rerank_rank: int | None = None,
    rank: int | None = None,
) -> RetrievedCandidate:
    return RetrievedCandidate(
        candidate_id=candidate_id,
        collection="acne_knowledge",
        text=text,
        fused_score=0.1,
        rerank_score=rerank_score,
        rerank_rank=rerank_rank,
        rank=rank,
        payload={
            "chunk_id": candidate_id,
            "source_id": source_id,
            "section_path": ["section"],
            "text": text,
        },
    )


def test_empty_candidates() -> None:
    result = select_budget_aware(_query(), [])
    assert result.selector_name == SELECTOR_NAME
    assert result.selected_candidate_ids == ()
    assert result.total_utility == 0


def test_one_candidate_fits() -> None:
    result = select_budget_aware(_query(), [_candidate("a")])
    assert result.selected_candidate_ids == ("a",)
    assert result.total_utility == rank_utility(1)


def test_one_candidate_too_large() -> None:
    result = select_budget_aware(_query(), [_candidate("a", "x" * 8000)])
    assert result.selected_candidate_ids == ()


def test_all_candidates_fit() -> None:
    candidates = [_candidate(str(index)) for index in range(5)]
    result = select_budget_aware(_query(), candidates)
    assert result.selected_candidate_ids == tuple(str(index) for index in range(5))


def test_more_than_nine_candidates_selects_at_most_nine() -> None:
    result = select_budget_aware(_query(), [_candidate(str(index)) for index in range(12)])
    assert result.selected_candidate_ids == tuple(str(index) for index in range(9))


def test_mixed_long_and_short_chunks() -> None:
    candidates = [
        _candidate("long", "x" * 500),
        _candidate("short-a", "a"),
        _candidate("short-b", "b"),
    ]
    result = select_budget_aware(_query(), candidates, max_chars=300)
    assert result.selected_candidate_ids == ("short-a", "short-b")


def test_duplicate_ids_fail_closed() -> None:
    with pytest.raises(SelectionInputError, match="Duplicate candidate ID"):
        select_budget_aware(_query(), [_candidate("same"), _candidate("same")])


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(" "),
        _candidate("a", " "),
    ],
)
def test_missing_id_or_text_fails_closed(candidate: RetrievedCandidate) -> None:
    with pytest.raises(SelectionInputError, match="missing a stable ID or text"):
        select_budget_aware(_query(), [candidate])


def test_negative_and_zero_rerank_scores_do_not_change_input_order() -> None:
    candidates = [
        _candidate("negative", rerank_score=-3.0),
        _candidate("zero", rerank_score=0.0),
    ]
    result = select_budget_aware(_query(), candidates)
    assert result.selected_candidate_ids == ("negative", "zero")


def test_reranker_fallback_order_is_the_supplied_order() -> None:
    candidates = [_candidate("rank-two", rank=2), _candidate("rank-one", rank=1)]
    result = select_budget_aware(_query(), candidates)
    assert result.selected_candidate_ids == ("rank-two", "rank-one")
    assert {item.reason for item in result.packed_context.items} == {"rrf_rank"}


def test_stable_tie_break_contract() -> None:
    earlier = _Solution(utility=10, positions=(1, 4), candidate_ids=("z", "z"), rendered_chars=8)
    later = _Solution(utility=10, positions=(2, 3), candidate_ids=("a", "a"), rendered_chars=1)
    assert _is_better(earlier, later)

    smaller = _Solution(utility=10, positions=(1, 4), candidate_ids=("z", "z"), rendered_chars=7)
    assert _is_better(smaller, earlier)

    lexical = _Solution(utility=10, positions=(1, 4), candidate_ids=("a", "z"), rendered_chars=7)
    assert _is_better(lexical, smaller)


def test_exact_rendered_cost() -> None:
    candidates = [_candidate("a", "one"), _candidate("b", "two")]
    result = select_budget_aware(_query(), candidates)
    expected = (
        sum(len(_render_block(candidate, index)) for index, candidate in enumerate(candidates, 1))
        + 2
    )
    assert len(result.packed_context.context_text) == expected


def test_whole_chunk_preservation() -> None:
    text = "complete qualifier and negation remain together"
    result = select_budget_aware(_query(), [_candidate("a", text)])
    assert result.packed_context.items[0].text == text
    assert result.packed_context.context_text.endswith(text)


def test_source_and_provenance_preservation() -> None:
    candidate = _candidate("a", source_id="source-a")
    result = select_budget_aware(_query(), [candidate])
    item = result.packed_context.items[0]
    assert item.item_id == "a"
    assert item.payload["chunk_id"] == "a"
    assert item.payload["source_id"] == "source-a"


def test_multi_attempt_retained_candidates_are_independent_and_not_mutated() -> None:
    retained = _candidate("retained", rank=1)
    acquired = _candidate("acquired", rank=2)
    before = retained.model_dump()
    first = select_budget_aware(_query(), [retained])
    second = select_budget_aware(_query(), [retained, acquired])
    assert first.selected_candidate_ids == ("retained",)
    assert second.selected_candidate_ids == ("retained", "acquired")
    assert retained.model_dump() == before


def test_deterministic_repeated_output() -> None:
    candidates = [_candidate(str(index), "x" * (index + 1)) for index in range(8)]
    first = select_budget_aware(_query(), candidates)
    second = select_budget_aware(_query(), candidates)
    assert first == second


def test_no_gold_fields_are_accepted_or_referenced() -> None:
    signature = inspect.signature(select_budget_aware)
    source = inspect.getsource(select_budget_aware)
    assert "gold" not in signature.parameters
    assert "gold" not in source.lower()


def test_no_inference_or_provider_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("external inference/network call attempted")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    result = select_budget_aware(_query(), [_candidate("a")])
    assert result.selected_candidate_ids == ("a",)


def test_no_database_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("database or filesystem write attempted")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    result = select_budget_aware(_query(), [_candidate("a")])
    assert result.selected_candidate_ids == ("a",)


def test_budget_is_never_exceeded() -> None:
    candidates = [_candidate(str(index), "x" * (100 + index * 37)) for index in range(15)]
    result = select_budget_aware(_query(), candidates, max_items=4, max_chars=900)
    assert len(result.packed_context.items) <= 4
    assert len(result.packed_context.context_text) <= 900


def test_dp_matches_brute_force_reference() -> None:
    candidates = [
        _candidate("a", "x" * 80),
        _candidate("b", "x" * 170),
        _candidate("c", "x" * 40),
        _candidate("d", "x" * 120),
        _candidate("e", "x" * 30),
        _candidate("f", "x" * 90),
    ]
    max_items = 3
    max_chars = 430
    feasible: list[tuple[int, tuple[int, ...], int, tuple[str, ...]]] = []
    for count in range(max_items + 1):
        for positions in itertools.combinations(range(len(candidates)), count):
            subset = [candidates[position] for position in positions]
            packed = pack_context(_query(), subset, max_items=max_items, max_chars=max_chars)
            if len(packed.items) != len(subset):
                continue
            feasible.append(
                (
                    sum(rank_utility(position + 1) for position in positions),
                    tuple(position + 1 for position in positions),
                    len(packed.context_text),
                    tuple(candidate.candidate_id for candidate in subset),
                )
            )
    optimum = min(feasible, key=lambda row: (-row[0], row[1], row[2], row[3]))
    result = select_budget_aware(_query(), candidates, max_items=max_items, max_chars=max_chars)
    assert (
        result.total_utility,
        result.selected_original_positions,
        len(result.packed_context.context_text),
        result.selected_candidate_ids,
    ) == optimum
