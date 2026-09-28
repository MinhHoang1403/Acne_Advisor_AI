"""Deterministic execution policy for bounded ActionDecision proposals.

The model owns semantic proposals such as the search query, evidence gap, scope
outcome, and direct-support IDs. This module owns only facts Python can enforce:
legal action/reason pairs, field shape, retrieval budget, retry legality, and
membership of proposed evidence IDs in the bounded evidence view.

The functions here are pure: they do not call providers, inspect environment
configuration, mutate state, or infer medical meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


MAX_RETRIEVAL_ATTEMPTS = 2

# This is an engineering limit on retrieval executions, not a confidence or
# medical-evidence sufficiency threshold.
DecisionAction = Literal["retrieve", "retry", "generate", "abstain"]
DecisionReason = Literal[
    "needs_evidence",
    "evidence_sufficient",
    "evidence_gap",
    "out_of_scope",
    "cannot_safely_proceed",
]

LEGAL_REASONS_BY_ACTION: dict[DecisionAction, frozenset[DecisionReason]] = {
    "retrieve": frozenset({"needs_evidence"}),
    "retry": frozenset({"evidence_gap"}),
    "generate": frozenset({"evidence_sufficient"}),
    "abstain": frozenset({"evidence_gap", "out_of_scope", "cannot_safely_proceed"}),
}

PolicyReasonCode = Literal[
    "accepted",
    "normalized_abstention",
    "illegal_action_reason",
    "invalid_generate_shape",
    "invalid_direct_evidence_ids",
    "invalid_initial_retrieval",
    "invalid_retry_transition",
    "retry_budget_exhausted",
    "retry_query_not_revised",
]


@dataclass(frozen=True)
class ActionPolicyContext:
    """Bounded runtime facts available to deterministic validation."""

    attempt_index: int
    evidence_available: bool
    retrieval_status: str | None = None
    visible_evidence_ids: frozenset[str] = frozenset()
    previous_queries: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionProposal:
    """Model-proposed fields after schema parsing, before execution policy."""

    action: DecisionAction
    retrieval_query: str | None
    missing_evidence: str | None
    reason_code: DecisionReason
    direct_supporting_evidence_ids: tuple[str, ...] | None


@dataclass(frozen=True)
class ActionPolicyResult:
    """Validated execution decision plus a bounded, text-free policy trace."""

    decision: ActionProposal
    changed_by_policy: bool
    policy_reason_code: PolicyReasonCode


def validate_action_proposal(
    proposal: ActionProposal,
    context: ActionPolicyContext,
) -> ActionPolicyResult:
    """Validate one semantic proposal without replacing it with heuristics."""

    query = _normalize_optional_text(proposal.retrieval_query)
    missing_evidence = _normalize_optional_text(proposal.missing_evidence)
    direct_support_ids = _normalize_evidence_ids(proposal.direct_supporting_evidence_ids)

    if proposal.reason_code not in LEGAL_REASONS_BY_ACTION[proposal.action]:
        return _rejected(proposal, "illegal_action_reason")

    if proposal.action == "abstain":
        validated = ActionProposal(
            action="abstain",
            retrieval_query=None,
            missing_evidence=missing_evidence,
            reason_code=proposal.reason_code,
            direct_supporting_evidence_ids=None,
        )
        return _accepted(proposal, validated, "normalized_abstention")

    if proposal.action == "generate":
        if not context.evidence_available or missing_evidence is not None or not direct_support_ids:
            return _rejected(proposal, "invalid_generate_shape")
        if not set(direct_support_ids).issubset(context.visible_evidence_ids):
            return _rejected(proposal, "invalid_direct_evidence_ids")
        validated = ActionProposal(
            action="generate",
            retrieval_query=None,
            missing_evidence=None,
            reason_code="evidence_sufficient",
            direct_supporting_evidence_ids=direct_support_ids,
        )
        return _accepted(proposal, validated)

    if proposal.action == "retrieve":
        if (
            context.attempt_index != 0
            or context.evidence_available
            or query is None
            or missing_evidence is not None
        ):
            return _rejected(proposal, "invalid_initial_retrieval")
        validated = ActionProposal(
            action="retrieve",
            retrieval_query=query,
            missing_evidence=None,
            reason_code="needs_evidence",
            direct_supporting_evidence_ids=None,
        )
        return _accepted(proposal, validated)

    if context.attempt_index >= MAX_RETRIEVAL_ATTEMPTS:
        return _rejected(proposal, "retry_budget_exhausted")
    if (
        context.attempt_index <= 0
        or (not context.evidence_available and context.retrieval_status != "no_evidence")
        or query is None
        or missing_evidence is None
    ):
        return _rejected(proposal, "invalid_retry_transition")

    previous = {comparison_key(item) for item in context.previous_queries}
    current_key = comparison_key(query)
    if not current_key or current_key in previous:
        return _rejected(proposal, "retry_query_not_revised")
    validated = ActionProposal(
        action="retry",
        retrieval_query=query,
        missing_evidence=missing_evidence,
        reason_code="evidence_gap",
        direct_supporting_evidence_ids=None,
    )
    return _accepted(proposal, validated)


def invalid_action_proposal() -> ActionProposal:
    """Return the existing fail-closed decision for invalid model output."""

    return ActionProposal(
        action="abstain",
        retrieval_query=None,
        missing_evidence=None,
        reason_code="evidence_gap",
        direct_supporting_evidence_ids=None,
    )


def comparison_key(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def _normalize_optional_text(value: str | None) -> str | None:
    return " ".join(str(value or "").split()) or None


def _normalize_evidence_ids(values: tuple[str, ...] | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item).strip() for item in values or () if str(item).strip()))


def _accepted(
    original: ActionProposal,
    validated: ActionProposal,
    reason: PolicyReasonCode = "accepted",
) -> ActionPolicyResult:
    return ActionPolicyResult(
        decision=validated,
        changed_by_policy=validated != original,
        policy_reason_code=reason if validated != original else "accepted",
    )


def _rejected(
    original: ActionProposal,
    reason: PolicyReasonCode,
) -> ActionPolicyResult:
    validated = invalid_action_proposal()
    return ActionPolicyResult(
        decision=validated,
        changed_by_policy=validated != original,
        policy_reason_code=reason,
    )


__all__ = [
    "LEGAL_REASONS_BY_ACTION",
    "MAX_RETRIEVAL_ATTEMPTS",
    "ActionPolicyContext",
    "ActionPolicyResult",
    "ActionProposal",
    "DecisionAction",
    "DecisionReason",
    "PolicyReasonCode",
    "comparison_key",
    "invalid_action_proposal",
    "validate_action_proposal",
]
