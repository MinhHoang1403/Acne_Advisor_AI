"""
src/agent/nodes/respond.py
==========================
LangGraph node for final answer presentation.
"""

from __future__ import annotations

import logging
import time

from src.agent.answer_formatting import (
    CANONICAL_DISCLAIMER,
    finalize_answer_presentation,
    infer_response_profile,
)
from src.agent.state import ClinicalState
from src.agent.source_presentation import (
    build_grounded_source_answer,
    build_source_allowlist,
    is_source_request,
    validate_answer_source_mentions,
)
from src.agent.text_encoding import repair_mojibake

logger = logging.getLogger(__name__)


def _question_for_presentation(state: ClinicalState) -> str:
    return state.get("standalone_question") or state.get("user_question", "")


def _guardrail_draft(state: ClinicalState) -> str:
    guardrail = state.get("guardrail")
    refusal = repair_mojibake(
        state.get("refusal_message") or "Câu hỏi này nằm ngoài phạm vi hỗ trợ về mụn trứng cá."
    )
    refusal = refusal.replace(CANONICAL_DISCLAIMER, "").strip()

    if guardrail == "unsafe_prescription_request":
        return (
            "Tôi không thể kê đơn, chọn liều hoặc bỏ qua hướng dẫn an toàn cho thuốc trị mụn nguy cơ cao. "
            "Các thuốc như isotretinoin, retinoid hoặc kháng sinh cần bác sĩ da liễu đánh giá, kê đơn và theo dõi."
        )
    if guardrail == "medical_emergency_out_of_scope":
        return (
            "Tôi không thể đánh giá hoặc chẩn đoán triệu chứng này vì nó nằm ngoài phạm vi hỗ trợ về mụn. "
            "Nếu triệu chứng đang xảy ra, dữ dội hoặc nặng lên, hãy tìm trợ giúp y tế khẩn cấp/cấp cứu. "
            "Tôi vẫn có thể hỗ trợ các câu hỏi về mụn trứng cá và chăm sóc da mụn."
        )
    if guardrail in {"medical_emergency_allergy", "urgent_skin_eye_infection"}:
        return refusal
    if guardrail in {"out_of_domain", "out_of_domain_fallback", "unsafe_out_of_domain"}:
        return refusal
    return (
        f"{refusal}\n\n"
        "Bạn có thể hỏi về mụn trứng cá, chăm sóc da mụn, hoạt chất trị mụn, tác dụng phụ hoặc khi nào nên gặp bác sĩ da liễu."
    )


async def finalize_response_node(state: ClinicalState) -> dict:
    """Finalize every answer path through the unified presentation policy."""

    query = _question_for_presentation(state)
    guardrail = state.get("guardrail")
    fallback_type = state.get("fallback_type")
    severity = state.get("medical_severity")
    profile = infer_response_profile(
        query,
        severity=severity,
        guardrail=guardrail,
        fallback_type=fallback_type if state.get("fallback_applied") else None,
    )
    allowlist = state.get("source_allowlist") or build_source_allowlist(
        state.get("sources", []),
        state.get("vector_contexts", []),
    )

    def present_and_validate(draft: str, *, add_disclaimer: bool | None = None) -> tuple[str, dict]:
        started = time.perf_counter()
        source_request_answer = (
            build_grounded_source_answer(query, allowlist)
            if is_source_request(query)
            else draft
        )
        presented = finalize_answer_presentation(
            source_request_answer,
            user_question=query,
            response_profile=profile,
            severity=severity,
            guardrail=guardrail,
            fallback_type=fallback_type if state.get("fallback_applied") else None,
            add_disclaimer=add_disclaimer,
        )
        validation = validate_answer_source_mentions(presented, allowlist)
        diagnostics = {
            "version": "source_validation_v1",
            "allowlist_source_ids": list(validation.allowlist_source_ids),
            "removed_invalid_source_mentions": list(validation.removed_mentions),
            "invalid_source_name_count": len(validation.removed_mentions),
            "validation_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        return validation.answer, diagnostics

    if state.get("is_in_domain") is False:
        logger.info("Finalizing guardrail response with profile=%s.", profile)
        final_answer, source_validation = present_and_validate(
            _guardrail_draft(state),
            add_disclaimer=profile != "out_of_domain_emergency",
        )
        return {
            "final_answer": final_answer,
            "vector_contexts": [],
            "graph_facts": [],
            "symptoms": [],
            "sources": [],
            "actual_provider": "system",
            "actual_model": None,
            "response_profile": profile,
            "source_validation": source_validation,
            "performance_timings": {
                **(state.get("performance_timings") or {}),
                "source_validation": source_validation["validation_ms"],
            },
        }

    if state.get("cache_hit"):
        logger.debug("Finalizing cached response with profile=%s.", profile)
        cached_answer = state.get("final_answer") or state.get("cached_answer") or ""
        final_answer, source_validation = present_and_validate(repair_mojibake(cached_answer))
        return {
            "final_answer": final_answer,
            "cached_answer": final_answer,
            "response_profile": profile,
            "source_validation": source_validation,
            "performance_timings": {
                **(state.get("performance_timings") or {}),
                "source_validation": source_validation["validation_ms"],
            },
        }

    draft = repair_mojibake(state.get("draft_answer", ""))
    logger.debug("Finalizing generated/fallback response with profile=%s.", profile)
    final_answer, source_validation = present_and_validate(draft)
    return {
        "final_answer": repair_mojibake(final_answer),
        "response_profile": profile,
        "source_validation": source_validation,
        "performance_timings": {
            **(state.get("performance_timings") or {}),
            "source_validation": source_validation["validation_ms"],
        },
    }


__all__ = ["finalize_response_node"]
