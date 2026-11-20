"""Helper safe fallback deterministic cho Agentic RAG chat flow."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from src.agent.answer_formatting import assess_structural_quality


SAFE_FALLBACK_FLOW_VERSION = "safe_fallback_flow_v4"

# Câu từ chối chung theo văn phong tư vấn lịch sự, thấu cảm để người dùng không bị hụt hẫng.
GENERIC_NON_SAFETY_FALLBACK_ANSWER = (
    "Cảm ơn bạn đã đặt câu hỏi. Với vấn đề này, mình chưa có đủ cơ sở y khoa rõ ràng "
    "để tư vấn chính xác nên không muốn trả lời phỏng đoán dễ gây hiểu nhầm cho bạn. "
    "Bạn có thể chia sẻ cụ thể hơn tình trạng da hoặc khía cạnh bạn đang quan tâm "
    "(như loại mụn, biểu hiện trên da hay hoạt chất đang dùng), hoặc trao đổi trực tiếp "
    "với bác sĩ da liễu để được kiểm tra kỹ hơn nhé."
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
            "Chào bạn, mình chưa thấy rõ nội dung câu hỏi. "
            "Bạn có thể chia sẻ cụ thể hơn về tình trạng mụn, biểu hiện trên da hoặc "
            "sản phẩm chăm sóc da bạn đang quan tâm nhé."
        )
    if fallback_type == "out_of_scope":
        return (
            "Cảm ơn bạn đã hỏi, nhưng câu hỏi này nằm ngoài phạm vi hỗ trợ về mụn và "
            "chăm sóc da của mình. Nếu bạn có băn khoăn nào liên quan đến mụn trứng cá, "
            "hoạt chất trị mụn hay cách chăm sóc da mụn, mình rất sẵn lòng giải đáp nhé."
        )
    if fallback_type == "cannot_safely_proceed":
        return (
            "Với tình huống này, mình chưa thể tư vấn trực tiếp một cách an toàn vì cần "
            "được đánh giá kỹ hơn trên thực tế. Bạn nên theo dõi sát biểu hiện trên da và "
            "trao đổi sớm với bác sĩ da liễu hoặc cơ sở y tế để được hướng dẫn phù hợp, "
            "hoặc nêu rõ hơn câu hỏi nếu bạn cần thêm thông tin tham khảo nhé."
        )
    return GENERIC_NON_SAFETY_FALLBACK_ANSWER


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
