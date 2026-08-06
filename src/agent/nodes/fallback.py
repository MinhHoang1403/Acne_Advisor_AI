"""LangGraph nodes for deterministic safe fallback flow."""

from __future__ import annotations

from typing import Any

from src.agent.answer_formatting import (
    grounded_entity_relation_answer,
    normalize_answer_markdown,
    repair_terminal_punctuation,
)
from src.agent.state import ClinicalState
from src.quality.safe_fallback import (
    build_safe_fallback_answer,
    assess_answerability,
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

    state_data = dict(state)
    answerability = assess_answerability(state_data)
    decision = decide_retrieval_fallback(state_data)
    query = state.get("standalone_question") or state.get("user_question") or ""
    recovery_answer = grounded_entity_relation_answer(str(query))
    if recovery_answer and decision.fallback_applied:
        return {
            "fallback_applied": True,
            "fallback_type": "grounded_direct_recovery",
            "fallback_reason": "Retrieval had no usable evidence; returned a verified taxonomy relation.",
            "fallback_answer": recovery_answer,
            "fallback_cache_eligible": False,
            "answerability": answerability.model_dump(mode="json"),
        }
    if not decision.fallback_applied:
        return {
            "fallback_applied": False,
            "fallback_type": "none",
            "fallback_reason": None,
            "fallback_answer": None,
            "fallback_cache_eligible": True,
            "answerability": answerability.model_dump(mode="json"),
        }
    clarification_options = (state.get("conversation_context") or {}).get("clarification_options") or []
    clarification_answer = None
    if decision.fallback_type == "ambiguous_reference" and clarification_options:
        clarification_answer = (
            "Mình cần làm rõ bạn đang hỏi về "
            + " hay ".join(str(option) for option in clarification_options)
            + ". Bạn có thể cho biết tên thuốc hoặc hoạt chất cụ thể không?"
        )
    answer, severity_metadata = _guarded_fallback_answer(
        decision.fallback_type,
        query=state.get("standalone_question") or state.get("user_question"),
        reason=decision.fallback_reason,
        fallback_answer=clarification_answer,
    )
    return {
        "fallback_applied": True,
        "fallback_type": decision.fallback_type,
        "fallback_reason": decision.fallback_reason,
        "fallback_answer": answer,
        "fallback_cache_eligible": decision.fallback_cache_eligible,
        "answerability": answerability.model_dump(mode="json"),
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
    query = state.get("standalone_question") or state.get("user_question") or ""
    recovery_answer = grounded_entity_relation_answer(str(query))
    if recovery_answer and decision.fallback_type in {"empty_generation", "invalid_generation"}:
        return {
            "draft_answer": recovery_answer,
            "fallback_applied": True,
            "fallback_type": "grounded_direct_recovery",
            "fallback_reason": "Model generation was unusable; returned a verified taxonomy relation.",
            "fallback_answer": recovery_answer,
            "fallback_cache_eligible": False,
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
