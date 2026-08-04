"""LangGraph nodes for deterministic safe fallback flow."""

from __future__ import annotations

from typing import Any

from src.agent.answer_formatting import normalize_answer_markdown, repair_terminal_punctuation
from src.agent.state import ClinicalState
from src.quality.safe_fallback import (
    build_safe_fallback_answer,
    decide_generation_fallback,
    decide_retrieval_fallback,
)
from src.quality.severity_guard import apply_severity_aware_answer_guard


def _guarded_fallback_answer(
    fallback_type: str,
    *,
    query: str | None,
    reason: str | None,
    fallback_answer: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Keep deterministic fallbacks aligned with the query's medical severity."""

    guarded = apply_severity_aware_answer_guard(
        query=query or "",
        answer=fallback_answer
        or build_safe_fallback_answer(fallback_type, query=query, reason=reason),
    )
    return guarded.answer, {
        "medical_severity": guarded.classification.severity,
        "severity_guard": guarded.classification.model_dump(mode="json"),
        "severity_guard_modified": guarded.modified,
        "severity_guard_cache_eligible": guarded.cache_eligible,
    }


async def fallback_decision_node(state: ClinicalState) -> dict[str, Any]:
    """Decide whether retrieval evidence is usable before answer generation."""

    decision = decide_retrieval_fallback(dict(state))
    if not decision.fallback_applied:
        return {
            "fallback_applied": False,
            "fallback_type": "none",
            "fallback_reason": None,
            "fallback_answer": None,
            "fallback_cache_eligible": True,
        }
    answer, severity_metadata = _guarded_fallback_answer(
        decision.fallback_type,
        query=state.get("standalone_question") or state.get("user_question"),
        reason=decision.fallback_reason,
    )
    return {
        "fallback_applied": True,
        "fallback_type": decision.fallback_type,
        "fallback_reason": decision.fallback_reason,
        "fallback_answer": answer,
        "fallback_cache_eligible": decision.fallback_cache_eligible,
        **severity_metadata,
    }


async def generation_fallback_decision_node(state: ClinicalState) -> dict[str, Any]:
    """Validate generated draft answer before finalize_response_node."""

    draft_answer = state.get("draft_answer")
    if isinstance(draft_answer, str):
        # Reuse the deterministic presentation cleanup before treating a
        # dangling Markdown heading as an invalid medical generation.
        draft_answer = normalize_answer_markdown(draft_answer)
        draft_answer = repair_terminal_punctuation(draft_answer)
    decision = decide_generation_fallback(draft_answer)
    if not decision.fallback_applied:
        return {
            "draft_answer": draft_answer,
            "fallback_applied": False,
            "fallback_type": "none",
            "fallback_reason": None,
            "fallback_answer": None,
            "fallback_cache_eligible": True,
        }
    answer, severity_metadata = _guarded_fallback_answer(
        decision.fallback_type,
        query=state.get("standalone_question") or state.get("user_question"),
        reason=decision.fallback_reason,
    )
    return {
        "fallback_applied": True,
        "fallback_type": decision.fallback_type,
        "fallback_reason": decision.fallback_reason,
        "fallback_answer": answer,
        "fallback_cache_eligible": decision.fallback_cache_eligible,
        **severity_metadata,
    }


async def safe_fallback_node(state: ClinicalState) -> dict[str, Any]:
    """Convert a safe fallback decision into the draft answer used by finalize."""

    fallback_type = state.get("fallback_type") or "no_retrieval_evidence"
    fallback_reason = state.get("fallback_reason")
    fallback_answer, severity_metadata = _guarded_fallback_answer(
        fallback_type,
        query=state.get("standalone_question") or state.get("user_question"),
        reason=fallback_reason,
        fallback_answer=state.get("fallback_answer"),
    )
    return {
        "draft_answer": fallback_answer,
        "sources": [],
        "actual_provider": "system",
        "actual_model": None,
        "llm_fallback_used": False,
        "fallback_provider": None,
        "fallback_model": None,
        "fallback_applied": True,
        "fallback_type": fallback_type,
        "fallback_reason": fallback_reason,
        "fallback_answer": fallback_answer,
        "fallback_cache_eligible": False,
        **severity_metadata,
    }


__all__ = [
    "fallback_decision_node",
    "generation_fallback_decision_node",
    "safe_fallback_node",
]
