"""Chuẩn hóa presentation và source mention cho mọi đường trả lời.

Generated, cached và fallback drafts đều đi qua cùng presentation policy. Source
allowlist giới hạn tên nguồn có thể hiển thị; validation này kiểm identity, không
đánh giá nội dung nguồn có đủ để chứng minh từng claim hay không.
"""

from __future__ import annotations

import logging
import time

from src.agent.answer_formatting import finalize_answer_presentation, infer_response_profile
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
    return state.get("normalized_question") or state.get("user_question", "")


async def finalize_response_node(state: ClinicalState) -> dict:
    """Đưa mọi answer path qua cùng presentation và source policy."""

    query = _question_for_presentation(state)
    fallback_type = state.get("fallback_type")
    severity = state.get("safety_severity")
    profile = infer_response_profile(
        query,
        severity=severity,
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

    if state.get("cache_hit"):
        logger.debug("Finalizing cached response with profile=%s.", profile)
        cached_answer = state.get("final_answer") or state.get("cached_answer") or ""
        final_answer, source_validation = present_and_validate(
            repair_mojibake(cached_answer),
        )
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
