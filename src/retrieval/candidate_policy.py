"""Deterministic, score-preserving candidate policy for Retrieval V5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate
from src.retrieval.v5_contracts import CandidateDropV5, DropReasonV5


CANDIDATE_POLICY_VERSION = "candidate_policy_v1"
CANDIDATE_POLICY_MIN_BUDGET = 8


@dataclass(frozen=True)
class CandidatePolicyResult:
    """Policy-approved candidates plus explicit drop and retention evidence."""

    candidates: tuple[RetrievedCandidate, ...]
    drops: tuple[CandidateDropV5, ...]
    input_count: int
    ordered_candidate_count: int
    budget: int
    protected_candidate_ids: tuple[str, ...]

    def debug_summary(self) -> dict[str, Any]:
        """Return compact policy metadata suitable for retrieval diagnostics."""

        selected_sources = {
            source
            for candidate in self.candidates
            if (source := _source_identity(candidate))
        }
        selected_documents = {_document_identity(candidate) for candidate in self.candidates}
        selected_ids = {candidate.candidate_id for candidate in self.candidates}
        return {
            "version": CANDIDATE_POLICY_VERSION,
            "mode": "budget_only",
            "input_count": self.input_count,
            "ordered_candidate_count": self.ordered_candidate_count,
            "distinct_candidate_count": len(selected_ids),
            "approved_count": len(self.candidates),
            "budget": self.budget,
            "candidate_policy_retention": _ratio(len(self.candidates), self.input_count),
            "duplicate_slot_ratio": _duplicate_slot_ratio(self.candidates),
            "same_document_slot_ratio": _same_document_slot_ratio(self.candidates),
            "exact_dedupe_enabled": False,
            "canonical_dedupe_enabled": False,
            "exact_dedupe_removed_count": sum(
                drop.reason == DropReasonV5.DEDUPE_REMOVED for drop in self.drops
            ),
            "budget_removed_count": sum(
                drop.reason == DropReasonV5.CANDIDATE_BUDGET_REMOVED for drop in self.drops
            ),
            "protected_candidate_ids": list(self.protected_candidate_ids),
            "protected_evidence_preserved": all(
                candidate_id in selected_ids for candidate_id in self.protected_candidate_ids
            ),
            "source_diversity_enabled": False,
            "document_diversity_enabled": False,
            "unique_source_count": len(selected_sources),
            "unique_document_count": len(selected_documents),
        }


def apply_candidate_policy(
    candidates: list[RetrievedCandidate],
    normalized_query: NormalizedQuery,
    *,
    budget: int,
) -> CandidatePolicyResult:
    """Apply the inherited bounded candidate budget in upstream order.

    The input order is authoritative relevance order. Exact dedupe, canonical
    dedupe, and source/document diversity are deliberately disabled for this
    initial R4 pass. This policy never writes relevance scores or reorders a
    safety-context match; it records all budget drops explicitly in the trace.
    """

    if budget < 1:
        raise ValueError("candidate policy budget must be at least one")

    ordered = list(candidates)
    drops: list[CandidateDropV5] = []
    protected_ids = tuple(
        candidate.candidate_id
        for candidate in ordered
        if _matches_requested_safety_context(candidate, normalized_query)
    )
    selected = ordered[:budget]

    for candidate in ordered[budget:]:
        drops.append(
            CandidateDropV5(
                candidate_id=candidate.candidate_id,
                reason=DropReasonV5.CANDIDATE_BUDGET_REMOVED,
            )
        )

    return CandidatePolicyResult(
        candidates=tuple(selected),
        drops=tuple(drops),
        input_count=len(candidates),
        ordered_candidate_count=len(ordered),
        budget=budget,
        protected_candidate_ids=protected_ids,
    )


def candidate_policy_budget(top_k: int) -> int:
    """Preserve the legacy candidate-cap baseline for staged V5 rollout."""

    return max(top_k * 2, CANDIDATE_POLICY_MIN_BUDGET)


def _matches_requested_safety_context(
    candidate: RetrievedCandidate,
    normalized_query: NormalizedQuery,
) -> bool:
    requested = {_normalized_key(value) for value in normalized_query.safety_context if value}
    if not requested:
        return False
    values = [
        *_as_strings(candidate.matched_metadata.get("safety_context")),
        *_as_strings(candidate.payload.get("safety_context")),
        *_as_strings(candidate.payload.get("safety_contexts")),
        *_as_strings(candidate.payload.get("contraindications")),
    ]
    return bool(requested & {_normalized_key(value) for value in values if value})


def _as_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, str):
        return [value]
    return []


def _normalized_key(value: str) -> str:
    return value.strip().casefold()


def _source_identity(candidate: RetrievedCandidate) -> str:
    return str(candidate.payload.get("source_path") or candidate.payload.get("source_file") or "")


def _document_identity(candidate: RetrievedCandidate) -> str:
    return str(candidate.payload.get("document_id") or _source_identity(candidate) or candidate.candidate_id)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 1.0


def _duplicate_slot_ratio(candidates: tuple[RetrievedCandidate, ...]) -> float:
    seen: set[str] = set()
    duplicate_count = 0
    for candidate in candidates:
        if candidate.candidate_id in seen:
            duplicate_count += 1
        seen.add(candidate.candidate_id)
    return _ratio(duplicate_count, len(candidates))


def _same_document_slot_ratio(candidates: tuple[RetrievedCandidate, ...]) -> float:
    documents = [_document_identity(candidate) for candidate in candidates]
    repeated = sum(documents.count(document_id) > 1 for document_id in documents)
    return _ratio(repeated, len(documents))


__all__ = [
    "CANDIDATE_POLICY_MIN_BUDGET",
    "CANDIDATE_POLICY_VERSION",
    "CandidatePolicyResult",
    "apply_candidate_policy",
    "candidate_policy_budget",
]
