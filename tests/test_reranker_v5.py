from __future__ import annotations

import math

import pytest

from src.retrieval.contracts import RerankScoreBreakdown, RerankTrace, RerankedCandidate, RetrievedCandidate
from src.retrieval.query_normalization import normalize_query
from src.retrieval.reranker_v5 import policy_order_fallback_v5, rerank_policy_evidence_v5
from src.retrieval.v5_compat import query_context_from_legacy


def _candidate(candidate_id: str, score: float, **payload: object) -> RetrievedCandidate:
    return RetrievedCandidate(
        candidate_id=candidate_id,
        source="chunk",
        collection="acne_knowledge",
        text=f"Evidence for {candidate_id}",
        score=score,
        fused_score=score,
        rank=1,
        payload={
            "chunk_id": candidate_id,
            "document_id": f"document-{candidate_id}",
            "source_path": f"{candidate_id}.pdf",
            "dense_score": score + 0.1,
            "sparse_score": score + 0.2,
            "rrf_score": score + 0.3,
            **payload,
        },
    )


def _query_context():
    normalized = normalize_query("Adapalene có dùng khi mang thai không?")
    return normalized, query_context_from_legacy(
        original_query=normalized.original_query,
        retrieval_query=normalized.original_query,
        normalized_query=normalized,
    )


def _trace_for(
    candidates: list[RetrievedCandidate],
    *,
    input_count: int,
    top_n: int,
    fallback_used: bool = False,
    score: float = 0.9,
) -> RerankTrace:
    ranked = [
        RerankedCandidate(
            candidate=candidate.model_copy(
                update={
                    "fused_score": score - (index * 0.1),
                    "rank": index,
                    "debug": {"rerank_score": score - (index * 0.1)},
                }
            ),
            rerank_score=score - (index * 0.1),
            rerank_rank=index,
            score_breakdown=RerankScoreBreakdown(
                semantic_score=score - (index * 0.1),
                rule_score=0.5,
                retrieval_score=candidate.score,
                final_score=score - (index * 0.1),
            ),
        )
        for index, candidate in enumerate(candidates, start=1)
    ]
    return RerankTrace(
        provider="local_rules",
        enabled=not fallback_used,
        input_count=input_count,
        output_count=len(candidates),
        top_n=top_n,
        ranked_candidates=ranked,
        warnings=["provider fallback"] if fallback_used else [],
        timings_ms={"total": 1.0},
        fallback_used=fallback_used,
    )


def _runner(output: list[RetrievedCandidate], trace: RerankTrace):
    def run(**_kwargs):
        return output, trace

    return run


def test_v5_reranker_namespaces_scores_without_mutating_policy_input() -> None:
    normalized, context = _query_context()
    policy = [_candidate("pregnancy", 0.4), _candidate("general", 0.3)]
    before = [candidate.model_dump(mode="json") for candidate in policy]
    output = [policy[1], policy[0]]
    trace = _trace_for(output, input_count=2, top_n=2)

    result = rerank_policy_evidence_v5(
        query_context=context,
        normalized_query=normalized,
        candidates=policy,
        expansion=None,
        top_n=2,
        provider="local_rules",
        runner=_runner(output, trace),
    )

    assert [candidate.candidate_id for candidate in result.candidates] == ["general", "pregnancy"]
    assert [candidate.model_dump(mode="json") for candidate in policy] == before
    evidence = result.ranked_evidence[0]
    assert evidence.candidate.candidate.scores.dense_similarity == pytest.approx(0.4)
    assert evidence.candidate.candidate.scores.sparse_bm25_score == pytest.approx(0.5)
    assert evidence.candidate.scores.rrf == pytest.approx(0.6)
    assert evidence.scores.reranker_semantic == pytest.approx(0.8)
    assert evidence.scores.reranker_rule == pytest.approx(0.5)
    assert evidence.scores.reranker_final == pytest.approx(0.8)
    assert result.trace.fallback_used is False


@pytest.mark.parametrize("failure", ["partial", "missing", "duplicate", "nan", "infinite", "malformed"])
def test_v5_reranker_invalid_output_falls_back_to_complete_policy_order(failure: str) -> None:
    normalized, context = _query_context()
    policy = [_candidate("first", 0.4), _candidate("second", 0.3)]
    output = list(policy)
    trace = _trace_for(output, input_count=2, top_n=2)

    if failure == "partial":
        output = [policy[0]]
        trace = _trace_for(output, input_count=2, top_n=2)
    elif failure == "missing":
        output = [_candidate("unknown", 0.2), policy[1]]
        trace = _trace_for(output, input_count=2, top_n=2)
    elif failure == "duplicate":
        output = [policy[0], policy[0]]
        trace = _trace_for(output, input_count=2, top_n=2)
    elif failure in {"nan", "infinite", "malformed"}:
        invalid_score: float | str
        invalid_score = (
            math.nan
            if failure == "nan"
            else math.inf
            if failure == "infinite"
            else "not-a-score"
        )
        trace = _trace_for(output, input_count=2, top_n=2)
        original = trace.ranked_candidates[0]
        invalid_breakdown = RerankScoreBreakdown.model_construct(
            **{**original.score_breakdown.model_dump(), "final_score": invalid_score}
        )
        invalid = RerankedCandidate.model_construct(
            candidate=original.candidate,
            rerank_score=invalid_score,
            rerank_rank=original.rerank_rank,
            score_breakdown=invalid_breakdown,
        )
        trace = trace.model_copy(update={"ranked_candidates": [invalid, trace.ranked_candidates[1]]})

    result = rerank_policy_evidence_v5(
        query_context=context,
        normalized_query=normalized,
        candidates=policy,
        expansion=None,
        top_n=2,
        provider="local_rules",
        runner=_runner(output, trace),
    )

    assert [candidate.candidate_id for candidate in result.candidates] == ["first", "second"]
    assert result.trace.fallback_used is True
    assert any(warning.startswith("RERANK_FALLBACK_INVALID_OUTPUT") for warning in result.trace.warnings)
    assert all(item.fallback_used for item in result.ranked_evidence)
    assert all(item.scores.reranker_final is None for item in result.ranked_evidence)


def test_v5_reranker_provider_fallback_keeps_candidate_policy_order() -> None:
    normalized, context = _query_context()
    policy = [_candidate("first", 0.4), _candidate("pregnancy", 0.3)]
    provider_output = [policy[1], policy[0]]
    provider_trace = _trace_for(
        provider_output,
        input_count=2,
        top_n=2,
        fallback_used=True,
    )

    result = rerank_policy_evidence_v5(
        query_context=context,
        normalized_query=normalized,
        candidates=policy,
        expansion=None,
        top_n=2,
        provider="hybrid",
        runner=_runner(provider_output, provider_trace),
    )

    assert [candidate.candidate_id for candidate in result.candidates] == ["first", "pregnancy"]
    assert result.trace.fallback_used is True
    assert "RERANK_FALLBACK_PROVIDER" in result.trace.warnings


def test_v5_timeout_fallback_has_explicit_trace_code_and_preserves_all_policy_evidence() -> None:
    normalized, context = _query_context()
    policy = [_candidate("first", 0.4), _candidate("pregnancy", 0.3)]

    result = policy_order_fallback_v5(
        query_context=context,
        candidates=policy,
        provider="hybrid",
        top_n=1,
        warning="RERANK_FALLBACK_TIMEOUT",
    )

    assert [candidate.candidate_id for candidate in result.candidates] == ["first", "pregnancy"]
    assert result.trace.fallback_used is True
    assert "RERANK_FALLBACK_TIMEOUT" in result.trace.warnings
    assert [item.candidate.candidate.candidate_id for item in result.ranked_evidence] == [
        "first",
        "pregnancy",
    ]
