"""LangGraph nodes chuyển generation lỗi sang deterministic safe fallback."""

from __future__ import annotations

from typing import Any

from src.agent.answer_formatting import (
    normalize_answer_markdown,
    repair_terminal_punctuation,
)
from src.agent.state import ClinicalState
from src.quality.safe_fallback import (
    build_safe_fallback_answer,
    decide_generation_fallback,
)


async def generation_fallback_decision_node(state: ClinicalState) -> dict[str, Any]:
    """Kiểm draft answer trước khi ``finalize_response_node`` xử lý."""

    draft_answer = state.get("draft_answer")
    if isinstance(draft_answer, str):
        # Dùng presentation cleanup deterministic trước khi coi Markdown heading
        # bị treo là generation không hợp lệ.
        draft_answer = normalize_answer_markdown(draft_answer)
        draft_answer = repair_terminal_punctuation(draft_answer)
    decision = decide_generation_fallback(draft_answer)
    if not decision.fallback_applied:
        return {
            "draft_answer": draft_answer,
            "fallback_applied": False,
            "fallback_type": "none",
            "fallback_reason": None,
            "fallback_reason_code": None,
            "fallback_answer": None,
            "fallback_cache_eligible": True,
        }
    answer = build_safe_fallback_answer(
        decision.fallback_type,
        reason_code="generation_unavailable",
    )
    return {
        "fallback_applied": True,
        "fallback_type": decision.fallback_type,
        "fallback_reason": decision.fallback_reason,
        "fallback_reason_code": "generation_unavailable",
        "fallback_answer": answer,
        "fallback_cache_eligible": decision.fallback_cache_eligible,
    }


async def safe_fallback_node(state: ClinicalState) -> dict[str, Any]:
    """Chuyển safe-fallback decision thành draft answer cho bước finalize."""

    fallback_type = state.get("fallback_type") or "no_retrieval_evidence"
    fallback_reason = state.get("fallback_reason")
    fallback_reason_code = state.get("fallback_reason_code")
    fallback_answer = state.get("fallback_answer") or build_safe_fallback_answer(
        fallback_type,
        reason_code=fallback_reason_code,
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
        "fallback_reason_code": fallback_reason_code,
        "fallback_answer": fallback_answer,
        "fallback_cache_eligible": False,
    }


__all__ = [
    "generation_fallback_decision_node",
    "safe_fallback_node",
]
