"""Deterministic safe fallback helpers for Phase 2 chat flow."""

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
    "grounded_direct_recovery",
]


class SafeFallbackDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fallback_applied: bool
    fallback_type: FallbackType = "none"
    fallback_reason: str | None = None
    fallback_cache_eligible: bool = True


def sanitize_fallback_reason(value: Any, *, max_chars: int = 160) -> str:
    """Return a short, secret-safe reason string."""

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
    """Build a short Vietnamese fallback answer without unsupported medical claims."""

    if fallback_type == "empty_query":
        return (
            "**Tóm tắt ngắn**\n"
            "Mình chưa nhận được câu hỏi đủ rõ để tư vấn an toàn.\n\n"
            "**Bạn có thể bổ sung**\n"
            "Hãy mô tả vấn đề da hoặc loại mụn, vị trí, thời gian xuất hiện, triệu chứng như đau/ngứa/sưng/mủ, "
            "và sản phẩm hoặc thuốc đang dùng.\n\n"
            "**Lưu ý**\n"
            "Thông tin này chỉ nhằm định hướng câu hỏi tiếp theo và không thay thế tư vấn y khoa chuyên nghiệp."
        )
    if fallback_type == "retrieval_error":
        return (
            "**Tóm tắt ngắn**\n"
            "Hệ thống tạm thời không truy xuất được nguồn kiến thức cần thiết để trả lời đáng tin cậy.\n\n"
            "**Bạn có thể làm gì tiếp theo**\n"
            "Vui lòng thử lại sau ít phút, hoặc viết rõ tên thuốc/hoạt chất, triệu chứng, thời gian dùng và bối cảnh như mang thai/cho con bú nếu có.\n\n"
            "**Lưu ý an toàn**\n"
            "Nếu có khó thở, sưng môi/mặt, sốt cao, đau dữ dội, mủ lan nhanh hoặc triệu chứng gần mắt, hãy đi khám hoặc cấp cứu ngay."
        )
    if fallback_type == "insufficient_context":
        return (
            "**Tóm tắt ngắn**\n"
            "Tài liệu hiện có chưa đủ bằng chứng được chọn để trả lời chính xác câu hỏi này.\n\n"
            "**Bạn có thể bổ sung**\n"
            "Hãy ghi đúng tên thuốc/hoạt chất, mô tả triệu chứng cụ thể hơn, thời gian sử dụng và các tình huống an toàn quan trọng nếu có.\n\n"
            "**Lưu ý**\n"
            "Mình sẽ không suy đoán hoặc bịa nguồn khi context chưa đủ."
        )
    if fallback_type in {"empty_generation", "invalid_generation"}:
        return (
            "**Tóm tắt ngắn**\n"
            "Mình chưa thể tạo câu trả lời đáng tin cậy từ thông tin hiện có.\n\n"
            "**Bạn có thể làm gì tiếp theo**\n"
            "Vui lòng thử lại hoặc viết câu hỏi cụ thể hơn về loại mụn, hoạt chất, thuốc đang dùng, triệu chứng và thời gian xuất hiện.\n\n"
            "**Lưu ý an toàn**\n"
            "Không tự dùng thuốc kê đơn như isotretinoin, kháng sinh uống hoặc phối hợp nhiều hoạt chất mạnh khi chưa có bác sĩ hướng dẫn."
        )
    return (
        "**Tóm tắt ngắn**\n"
        "Hệ thống chưa có đủ thông tin đáng tin cậy để trả lời chính xác câu hỏi này.\n\n"
        "**Bạn có thể bổ sung**\n"
        "Hãy ghi đúng tên thuốc/hoạt chất, mô tả triệu chứng, thời gian sử dụng và bối cảnh an toàn như mang thai hoặc cho con bú nếu liên quan.\n\n"
        "**Khi nào nên gặp bác sĩ**\n"
        "Nên đi khám nếu triệu chứng nặng, tiến triển nhanh, đau nhiều, chảy mủ, gần mắt hoặc ảnh hưởng rõ đến sinh hoạt."
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
