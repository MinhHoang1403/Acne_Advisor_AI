"""Helper safe fallback deterministic cho Agentic RAG chat flow."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from src.agent.answer_formatting import assess_structural_quality


SAFE_FALLBACK_FLOW_VERSION = "safe_fallback_flow_v1"

FallbackType = Literal[
    "none",
    "empty_query",
    "no_retrieval_evidence",
    "insufficient_context",
    "retrieval_error",
    "empty_generation",
    "invalid_generation",
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


def build_safe_fallback_answer(fallback_type: str, query: str | None = None, reason: str | None = None) -> str:
    """Tạo infrastructure fallback mà không nhúng medical fact rules."""

    del query, reason

    if fallback_type == "empty_query":
        return (
            "Mình chưa nhận được câu hỏi đủ rõ. "
            "Bạn hãy nêu cụ thể điều muốn biết về mụn hoặc chăm sóc da."
        )
    if fallback_type == "retrieval_error":
        return (
            "Mình chưa thể lấy đủ thông tin đáng tin cậy để trả lời lúc này. "
            "Bạn vui lòng thử lại sau ít phút hoặc viết câu hỏi cụ thể hơn."
        )
    if fallback_type == "insufficient_context":
        return (
            "Mình chưa có đủ thông tin đáng tin cậy để trả lời chắc chắn câu hỏi này. "
            "Bạn có thể nêu cụ thể hơn điều muốn biết."
        )
    if fallback_type in {"empty_generation", "invalid_generation"}:
        return (
            "Mình chưa thể đưa ra câu trả lời đáng tin cậy từ thông tin hiện có. "
            "Bạn vui lòng thử lại hoặc viết câu hỏi cụ thể hơn."
        )
    return (
        "Mình chưa có đủ thông tin đáng tin cậy để trả lời chính xác câu hỏi này. "
        "Bạn có thể nêu cụ thể hơn điều muốn biết."
    )


__all__ = [
    "SAFE_FALLBACK_FLOW_VERSION",
    "FallbackType",
    "SafeFallbackDecision",
    "build_safe_fallback_answer",
    "decide_generation_fallback",
    "is_usable_text",
    "sanitize_fallback_reason",
]
