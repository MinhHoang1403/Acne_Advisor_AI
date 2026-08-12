from __future__ import annotations

from src.retrieval.candidate_policy import apply_candidate_policy, candidate_policy_budget
from src.retrieval.contracts import RetrievedCandidate
from src.retrieval.query_normalization import normalize_query
from src.retrieval.v5_contracts import DropReasonV5


def _candidate(
    candidate_id: str,
    score: float,
    *,
    source_file: str = "source-a.pdf",
    **payload: object,
) -> RetrievedCandidate:
    return RetrievedCandidate(
        candidate_id=candidate_id,
        source="chunk",
        collection="acne_knowledge",
        text=f"Evidence for {candidate_id}",
        score=score,
        fused_score=score,
        payload={"source_file": source_file, **payload},
    )


def test_candidate_policy_budget_preserves_legacy_cap_baseline() -> None:
    assert candidate_policy_budget(1) == 8
    assert candidate_policy_budget(5) == 10


def test_candidate_policy_keeps_exact_duplicates_without_modifying_scores_or_order() -> None:
    first = _candidate("same", 0.03)
    duplicate = _candidate("same", 0.01, source_file="source-b.pdf")
    next_candidate = _candidate("next", 0.02)

    result = apply_candidate_policy(
        [first, duplicate, next_candidate],
        normalize_query("Mụn đầu đen là gì?"),
        budget=4,
    )

    assert [candidate.candidate_id for candidate in result.candidates] == ["same", "same", "next"]
    assert result.candidates[0].score == first.score
    assert result.candidates[0].fused_score == first.fused_score
    assert result.drops == ()
    debug = result.debug_summary()
    assert debug["mode"] == "budget_only"
    assert debug["exact_dedupe_enabled"] is False
    assert debug["duplicate_slot_ratio"] == 0.333333


def test_candidate_policy_is_an_exact_pass_through_when_budget_covers_candidates() -> None:
    candidates = [
        _candidate("first", 0.03, source_file="first.pdf"),
        _candidate("second", 0.02, source_file="second.pdf"),
        _candidate("third", 0.01, source_file="third.pdf"),
    ]

    result = apply_candidate_policy(
        candidates,
        normalize_query("Mụn trứng cá là gì?"),
        budget=candidate_policy_budget(5),
    )

    assert list(result.candidates) == candidates
    assert result.drops == ()
    assert result.debug_summary()["candidate_policy_retention"] == 1.0


def test_candidate_policy_budgets_deterministically_without_source_diversity() -> None:
    result = apply_candidate_policy(
        [
            _candidate("first", 0.03, source_file="same.pdf"),
            _candidate("second", 0.02, source_file="same.pdf"),
            _candidate("third", 0.01, source_file="other.pdf"),
        ],
        normalize_query("Mụn trứng cá là gì?"),
        budget=2,
    )

    assert [candidate.candidate_id for candidate in result.candidates] == ["first", "second"]
    assert [(drop.candidate_id, drop.reason) for drop in result.drops] == [
        ("third", DropReasonV5.CANDIDATE_BUDGET_REMOVED),
    ]
    debug = result.debug_summary()
    assert debug["source_diversity_enabled"] is False
    assert debug["document_diversity_enabled"] is False
    assert debug["canonical_dedupe_enabled"] is False
    assert debug["unique_source_count"] == 1


def test_candidate_policy_does_not_apply_unproven_canonical_dedupe() -> None:
    result = apply_candidate_policy(
        [
            _candidate("one", 0.03, canonical_name="adapalene"),
            _candidate("two", 0.02, canonical_name="adapalene"),
        ],
        normalize_query("Adapalene la gi?"),
        budget=2,
    )

    assert [candidate.candidate_id for candidate in result.candidates] == ["one", "two"]
    assert result.drops == ()


def test_candidate_policy_records_safety_evidence_without_reordering_it() -> None:
    pregnancy = _candidate(
        "pregnancy",
        0.01,
        safety_context=["pregnancy"],
    )
    result = apply_candidate_policy(
        [
            _candidate("first", 0.03),
            pregnancy,
            _candidate("second", 0.02),
        ],
        normalize_query("Tôi đang mang thai, dùng retinoid được không?"),
        budget=2,
    )

    assert [candidate.candidate_id for candidate in result.candidates] == ["first", "pregnancy"]
    assert result.protected_candidate_ids == ("pregnancy",)
    assert result.debug_summary()["protected_evidence_preserved"] is True
    assert [(drop.candidate_id, drop.reason) for drop in result.drops] == [
        ("second", DropReasonV5.CANDIDATE_BUDGET_REMOVED),
    ]


def test_candidate_policy_does_not_reserve_safety_evidence_outside_inherited_budget() -> None:
    result = apply_candidate_policy(
        [
            _candidate("first", 0.03),
            _candidate("second", 0.02),
            _candidate("pregnancy", 0.01, safety_context=["pregnancy"]),
        ],
        normalize_query("Tôi đang mang thai, dùng retinoid được không?"),
        budget=2,
    )

    assert [candidate.candidate_id for candidate in result.candidates] == ["first", "second"]
    assert result.protected_candidate_ids == ("pregnancy",)
    assert result.debug_summary()["protected_evidence_preserved"] is False
