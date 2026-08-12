"""Validated V5 reranker boundary with deterministic Candidate Policy fallback."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.retrieval.contracts import (
    NormalizedQuery,
    QueryExpansion,
    RerankTrace,
    RerankedCandidate,
    RetrievedCandidate,
)
from src.retrieval.reranker import rerank_candidates
from src.retrieval.v5_contracts import (
    CandidateProvenanceV5,
    FusedKnowledgeCandidateV5,
    KnowledgeCandidateV5,
    QueryContextV5,
    RankedEvidenceV5,
    ScoreNamespaceV5,
)


RerankRunner = Callable[..., tuple[list[RetrievedCandidate], RerankTrace]]


class RerankerContractError(ValueError):
    """Raised when a provider result cannot safely enter the V5 pipeline."""


@dataclass(frozen=True)
class RerankerV5Result:
    """Validated reranker output plus immutable V5 evidence observations."""

    candidates: tuple[RetrievedCandidate, ...]
    ranked_evidence: tuple[RankedEvidenceV5, ...]
    trace: RerankTrace


def rerank_policy_evidence_v5(
    *,
    query_context: QueryContextV5,
    normalized_query: NormalizedQuery,
    candidates: list[RetrievedCandidate],
    expansion: QueryExpansion | None,
    top_n: int,
    provider: str,
    annotations: Mapping[str, Any] | None = None,
    runner: RerankRunner = rerank_candidates,
    timeout_seconds: float | None = None,
) -> RerankerV5Result:
    """Run the legacy scorer behind the V5 contract without changing its tuning.

    A normal result is accepted only when it is a complete, finite permutation
    of policy-approved evidence. Provider failures and invalid output revert to
    the complete Candidate Policy order rather than returning sparse evidence.
    """

    del annotations  # Reserved structured input; it must not affect R5 ranking.
    policy_candidates = tuple(candidates)
    try:
        reranked_candidates, trace = runner(
            normalized_query=normalized_query,
            candidates=list(policy_candidates),
            expansion=expansion,
            top_n=top_n,
            provider=provider,
            timeout_seconds=timeout_seconds,
        )
        if trace.fallback_used:
            return policy_order_fallback_v5(
                query_context=query_context,
                candidates=policy_candidates,
                provider=provider,
                top_n=top_n,
                warning="RERANK_FALLBACK_PROVIDER",
                prior_trace=trace,
            )
        validate_reranker_output_v5(
            candidates=policy_candidates,
            reranked_candidates=reranked_candidates,
            trace=trace,
            top_n=top_n,
        )
    except Exception as exc:
        return policy_order_fallback_v5(
            query_context=query_context,
            candidates=policy_candidates,
            provider=provider,
            top_n=top_n,
            warning=f"RERANK_FALLBACK_INVALID_OUTPUT:{type(exc).__name__}",
        )

    return RerankerV5Result(
        candidates=tuple(reranked_candidates),
        ranked_evidence=ranked_evidence_from_rerank_v5(
            policy_candidates=policy_candidates,
            reranked_candidates=reranked_candidates,
            trace=trace,
            fallback_used=False,
        ),
        trace=trace,
    )


def policy_order_fallback_v5(
    *,
    query_context: QueryContextV5,
    candidates: tuple[RetrievedCandidate, ...] | list[RetrievedCandidate],
    provider: str,
    top_n: int,
    warning: str,
    prior_trace: RerankTrace | None = None,
) -> RerankerV5Result:
    """Return every policy-approved candidate in its original deterministic order."""

    del query_context  # Keeps the fallback contract explicit and query-safe.
    policy_candidates = tuple(candidates)
    warnings = [*(prior_trace.warnings if prior_trace else []), warning]
    trace = RerankTrace(
        provider=prior_trace.provider if prior_trace else provider,
        enabled=False,
        input_count=len(policy_candidates),
        output_count=len(policy_candidates),
        top_n=max(0, int(top_n or 0)),
        ranked_candidates=[],
        warnings=list(dict.fromkeys(warnings)),
        timings_ms=dict(prior_trace.timings_ms) if prior_trace else {},
        requested_provider=(prior_trace.requested_provider if prior_trace else provider),
        fallback_used=True,
        semantic_model_available=(prior_trace.semantic_model_available if prior_trace else False),
    )
    return RerankerV5Result(
        candidates=policy_candidates,
        ranked_evidence=ranked_evidence_from_rerank_v5(
            policy_candidates=policy_candidates,
            reranked_candidates=policy_candidates,
            trace=trace,
            fallback_used=True,
        ),
        trace=trace,
    )


def ranked_evidence_from_rerank_v5(
    *,
    policy_candidates: tuple[RetrievedCandidate, ...] | list[RetrievedCandidate],
    reranked_candidates: tuple[RetrievedCandidate, ...] | list[RetrievedCandidate],
    trace: RerankTrace,
    fallback_used: bool,
) -> tuple[RankedEvidenceV5, ...]:
    """Build immutable score namespaces from policy input and reranker output."""

    policy_by_id: dict[str, list[tuple[int, RetrievedCandidate]]] = defaultdict(list)
    for input_rank, candidate in enumerate(policy_candidates, start=1):
        policy_by_id[candidate.candidate_id].append((input_rank, candidate))

    trace_by_id: dict[str, list[RerankedCandidate]] = defaultdict(list)
    for item in trace.ranked_candidates:
        trace_by_id[item.candidate.candidate_id].append(item)

    evidence: list[RankedEvidenceV5] = []
    for output_rank, candidate in enumerate(reranked_candidates, start=1):
        source_items = policy_by_id.get(candidate.candidate_id, [])
        if not source_items:
            raise RerankerContractError(f"reranker emitted unknown candidate {candidate.candidate_id!r}")
        input_rank, source_candidate = source_items.pop(0)
        reranked_items = trace_by_id.get(candidate.candidate_id, [])
        reranked_item = reranked_items.pop(0) if reranked_items else None
        upstream_scores = _upstream_scores(source_candidate)
        reranker_scores = _reranker_scores(reranked_item, upstream_scores)
        fused_candidate = FusedKnowledgeCandidateV5(
            candidate=KnowledgeCandidateV5(
                candidate_id=source_candidate.candidate_id,
                provenance=_provenance(source_candidate),
                scores=upstream_scores,
                metadata_features=_metadata_features(source_candidate),
            ),
            rrf_rank=input_rank,
            scores=upstream_scores,
        )
        evidence.append(
            RankedEvidenceV5(
                candidate=fused_candidate,
                input_rank=input_rank,
                output_rank=output_rank,
                scores=reranker_scores,
                fallback_used=fallback_used,
            )
        )
    return tuple(evidence)


def validate_reranker_output_v5(
    *,
    candidates: tuple[RetrievedCandidate, ...] | list[RetrievedCandidate],
    reranked_candidates: list[RetrievedCandidate],
    trace: RerankTrace,
    top_n: int,
) -> None:
    """Reject incomplete, malformed, or non-finite provider output explicitly."""

    policy_candidates = tuple(candidates)
    safe_top_n = max(0, int(top_n or 0))
    expected_count = min(len(policy_candidates), safe_top_n)
    if trace.input_count != len(policy_candidates):
        raise RerankerContractError("reranker input cardinality does not match Candidate Policy")
    if trace.output_count != len(reranked_candidates):
        raise RerankerContractError("reranker trace output cardinality does not match output")
    if len(reranked_candidates) != expected_count:
        raise RerankerContractError("reranker output is partial or has invalid cardinality")
    if len(trace.ranked_candidates) != expected_count:
        raise RerankerContractError("reranker trace is partial or has invalid cardinality")

    allowed_counts = Counter(candidate.candidate_id for candidate in policy_candidates)
    output_ids = [candidate.candidate_id for candidate in reranked_candidates]
    output_counts = Counter(output_ids)
    unknown_ids = set(output_counts) - set(allowed_counts)
    if unknown_ids:
        raise RerankerContractError(f"reranker emitted missing candidate IDs: {sorted(unknown_ids)!r}")
    if any(count > allowed_counts[candidate_id] for candidate_id, count in output_counts.items()):
        raise RerankerContractError("reranker emitted unexpected duplicate candidate IDs")

    trace_ids = [item.candidate.candidate_id for item in trace.ranked_candidates]
    if trace_ids != output_ids:
        raise RerankerContractError("reranker trace candidate IDs do not match output")
    for output_rank, item in enumerate(trace.ranked_candidates, start=1):
        if item.rerank_rank != output_rank:
            raise RerankerContractError("reranker ranks are not contiguous")
        _require_finite(item.rerank_score, "rerank score")
        _require_finite(item.score_breakdown.final_score, "rerank final score")
        for field in (
            "base_score",
            "semantic_score",
            "rule_score",
            "retrieval_score",
            "lexical_score",
            "entity_match_score",
            "metadata_match_score",
            "intent_alignment_score",
            "safety_alignment_score",
            "source_priority_score",
            "penalty",
        ):
            value = getattr(item.score_breakdown, field)
            if value is not None:
                _require_finite(value, field)


def _upstream_scores(candidate: RetrievedCandidate) -> ScoreNamespaceV5:
    payload = candidate.payload
    debug = candidate.debug
    return ScoreNamespaceV5(
        dense_similarity=_finite_or_none(payload.get("dense_score", debug.get("dense_score"))),
        sparse_bm25_score=_finite_or_none(payload.get("sparse_score", debug.get("sparse_score"))),
        rrf=_finite_or_none(payload.get("rrf_score")),
        legacy_compat_score=_finite_or_none(candidate.score),
    )


def _reranker_scores(
    item: RerankedCandidate | None,
    upstream_scores: ScoreNamespaceV5,
) -> ScoreNamespaceV5:
    if item is None:
        return upstream_scores
    return upstream_scores.model_copy(
        update={
            "reranker_semantic": _finite_or_none(item.score_breakdown.semantic_score),
            "reranker_rule": _finite_or_none(item.score_breakdown.rule_score),
            "reranker_final": _finite_or_none(item.rerank_score),
        }
    )


def _provenance(candidate: RetrievedCandidate) -> CandidateProvenanceV5:
    payload = candidate.payload
    return CandidateProvenanceV5(
        point_id=_string_or_none(payload.get("id") or payload.get("point_id")),
        chunk_id=_string_or_none(payload.get("chunk_id")) or candidate.candidate_id,
        document_id=_string_or_none(payload.get("document_id")),
        source_path=_string_or_none(payload.get("source_path") or payload.get("source_file")),
    )


def _metadata_features(candidate: RetrievedCandidate) -> tuple[str, ...]:
    return tuple(
        key
        for key, value in {**candidate.matched_metadata, **candidate.debug}.items()
        if value not in (None, False, "", [], {})
    )


def _require_finite(value: Any, field: str) -> None:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RerankerContractError(f"reranker {field} is malformed") from exc
    if not math.isfinite(parsed):
        raise RerankerContractError(f"reranker {field} is non-finite")


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _string_or_none(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


__all__ = [
    "RerankerContractError",
    "RerankerV5Result",
    "policy_order_fallback_v5",
    "ranked_evidence_from_rerank_v5",
    "rerank_policy_evidence_v5",
    "validate_reranker_output_v5",
]
