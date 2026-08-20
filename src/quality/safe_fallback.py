"""Helper safe fallback deterministic cho Agentic RAG chat flow."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from src.agent.answer_formatting import assess_structural_quality


SAFE_FALLBACK_FLOW_VERSION = "safe_fallback_flow_v4"

GENERIC_NON_SAFETY_FALLBACK_ANSWER = (
    "Mình chưa thể đưa ra câu trả lời đủ tin cậy cho câu hỏi này lúc này. "
    "Bạn có thể thử diễn đạt cụ thể hơn hoặc thử lại sau."
)

_FALLBACK_REASON_LABELS: dict[str, str] = {
    "provider_unavailable": "Dịch vụ mô hình tạm thời không khả dụng",
    "out_of_scope": "Ngoài phạm vi hỗ trợ",
    "insufficient_evidence": "Chưa đủ bằng chứng",
    "cannot_safely_proceed": "Không thể tiếp tục an toàn",
    "retrieval_unavailable": "Dịch vụ truy hồi tạm thời không khả dụng",
    "generation_unavailable": "Không thể tạo câu trả lời đáng tin cậy",
}

_FALLBACK_REASON_BY_TYPE: dict[str, str] = {
    "no_retrieval_evidence": "insufficient_evidence",
    "insufficient_context": "insufficient_evidence",
    "provider_error": "provider_unavailable",
    "retrieval_error": "retrieval_unavailable",
    "empty_generation": "generation_unavailable",
    "invalid_generation": "generation_unavailable",
}

FallbackType = Literal[
    "none",
    "empty_query",
    "no_retrieval_evidence",
    "insufficient_context",
    "provider_error",
    "retrieval_error",
    "empty_generation",
    "invalid_generation",
    "out_of_scope",
    "cannot_safely_proceed",
]

FallbackReasonCode = Literal[
    "provider_unavailable",
    "out_of_scope",
    "insufficient_evidence",
    "cannot_safely_proceed",
    "retrieval_unavailable",
    "generation_unavailable",
]


class SafeFallbackDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fallback_applied: bool
    fallback_type: FallbackType = "none"
    fallback_reason: str | None = None
    fallback_cache_eligible: bool = True


def sanitize_fallback_reason(value: Any, *, max_chars: int = 160) -> str:
    """Trả reason ngắn và không làm lộ secret."""

    text = str(value or "").strip()
    if not text:
        return "Không có chi tiết lỗi."
    text = re.sub(
        r"(?i)(api[_-]?key|token|password|secret|authorization|bearer)\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)(\b[a-z][a-z0-9+.-]*://[^\s/:]+:)[^\s/@]+(@)",
        r"\1[REDACTED]\2",
        text,
    )
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def is_usable_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    placeholders = {
        "none",
        "null",
        "n/a",
        "na",
        "...",
        "[empty]",
        "<empty>",
    }
    return text.lower() not in placeholders


def decide_generation_fallback(value: Any) -> SafeFallbackDecision:
    if isinstance(value, str):
        if is_usable_text(value):
            structural_issues = assess_structural_quality(value)
            blocking_codes = {
                "incomplete_terminal_sentence",
                "truncated_generation",
                "empty_heading",
                "malformed_sentence_join",
            }
            for issue in structural_issues:
                if issue.get("code") in blocking_codes:
                    return SafeFallbackDecision(
                        fallback_applied=True,
                        fallback_type="invalid_generation",
                        fallback_reason=f"Structural generation issue: {issue.get('code')}",
                        fallback_cache_eligible=False,
                    )
            return SafeFallbackDecision(fallback_applied=False, fallback_type="none", fallback_cache_eligible=True)
        return SafeFallbackDecision(
            fallback_applied=True,
            fallback_type="empty_generation",
            fallback_reason="Model trả về câu trả lời rỗng.",
            fallback_cache_eligible=False,
        )
    return SafeFallbackDecision(
        fallback_applied=True,
        fallback_type="invalid_generation",
        fallback_reason=f"Generation output type không hợp lệ: {type(value).__name__}.",
        fallback_cache_eligible=False,
    )


def build_safe_fallback_answer(
    fallback_type: str,
    query: str | None = None,
    reason: str | None = None,
    reason_code: str | None = None,
) -> str:
    """Tạo infrastructure fallback mà không nhúng medical fact rules."""

    del query, reason

    effective_reason = reason_code or _FALLBACK_REASON_BY_TYPE.get(fallback_type)
    if effective_reason in {
        "provider_unavailable",
        "insufficient_evidence",
        "retrieval_unavailable",
        "generation_unavailable",
    }:
        return GENERIC_NON_SAFETY_FALLBACK_ANSWER

    if fallback_type == "empty_query":
        return (
            "Mình chưa nhận được câu hỏi đủ rõ. "
            "Bạn hãy nêu cụ thể điều muốn biết về mụn hoặc chăm sóc da."
        )
    if fallback_type == "out_of_scope":
        return (
            "Mình chỉ có thể hỗ trợ các câu hỏi về mụn và chăm sóc da liên quan. "
            "Bạn có thể hỏi lại trong phạm vi này."
        )
    if fallback_type == "cannot_safely_proceed":
        return (
            "Mình chưa thể xử lý câu hỏi này một cách an toàn ở thời điểm hiện tại. "
            "Bạn có thể viết lại câu hỏi rõ hơn hoặc thử lại sau."
        )
    return (
        "Mình chưa có đủ thông tin đáng tin cậy để trả lời chính xác câu hỏi này. "
        "Bạn có thể nêu cụ thể hơn điều muốn biết."
    )


def fallback_reason_label(reason_code: str | None) -> str | None:
    """Trả nhãn UX ổn định mà không thay đổi reason code nội bộ."""

    return _FALLBACK_REASON_LABELS.get(str(reason_code or ""))


def fallback_reason_code_from_agent_reason(reason_code: str | None) -> FallbackReasonCode:
    """Ánh xạ reason của bounded Agent sang vocabulary fallback ổn định."""

    return {
        "out_of_scope": "out_of_scope",
        "evidence_gap": "insufficient_evidence",
        "cannot_safely_proceed": "cannot_safely_proceed",
    }.get(str(reason_code or ""), "insufficient_evidence")  # type: ignore[return-value]


def fallback_type_for_reason(reason_code: FallbackReasonCode) -> FallbackType:
    return {
        "provider_unavailable": "provider_error",
        "out_of_scope": "out_of_scope",
        "insufficient_evidence": "no_retrieval_evidence",
        "cannot_safely_proceed": "cannot_safely_proceed",
        "retrieval_unavailable": "retrieval_error",
        "generation_unavailable": "invalid_generation",
    }[reason_code]  # type: ignore[return-value]


__all__ = [
    "SAFE_FALLBACK_FLOW_VERSION",
    "GENERIC_NON_SAFETY_FALLBACK_ANSWER",
    "FallbackType",
    "FallbackReasonCode",
    "SafeFallbackDecision",
    "build_safe_fallback_answer",
    "decide_generation_fallback",
    "fallback_reason_code_from_agent_reason",
    "fallback_reason_label",
    "fallback_type_for_reason",
    "is_usable_text",
    "sanitize_fallback_reason",
]
