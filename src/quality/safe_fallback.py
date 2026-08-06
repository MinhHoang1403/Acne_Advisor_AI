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
    "ambiguous_reference",
]

RetrievalStatus = Literal[
    "not_started",
    "success",
    "empty_query",
    "no_evidence",
    "insufficient_context",
    "recoverable_error",
]


class SafeFallbackDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fallback_applied: bool
    fallback_type: FallbackType = "none"
    fallback_reason: str | None = None
    fallback_cache_eligible: bool = True


class AnswerabilityResult(BaseModel):
    """Explicit answerability decision used before the safe fallback route."""

    is_in_domain: bool
    has_relevant_context: bool
    has_minimum_evidence: bool
    can_answer_directly: bool
    requires_clarification: bool
    requires_cautious_language: bool
    requires_fallback: bool
    reason_code: str


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


def has_usable_packed_context(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        return False
    items = value.get("items")
    if isinstance(items, list) and any(_item_has_text(item) for item in items):
        return True
    return is_usable_text(value.get("context_text"))


def has_usable_evidence(state: dict[str, Any]) -> bool:
    vector_contexts = state.get("vector_contexts") or []
    graph_facts = state.get("graph_facts") or []
    if isinstance(vector_contexts, list) and any(_item_has_text(item) for item in vector_contexts):
        return True
    if isinstance(graph_facts, list) and any(_graph_fact_has_evidence(item) for item in graph_facts):
        return True
    return has_usable_packed_context(state.get("packed_context"))


def assess_answerability(state: dict[str, Any]) -> AnswerabilityResult:
    """Classify answerability without turning recoverable misses into refusals."""

    in_domain = state.get("is_in_domain") is not False
    query = state.get("standalone_question") or state.get("normalized_question") or state.get("user_question")
    has_query = is_usable_text(query)
    has_evidence = has_usable_evidence(state)
    status = str(state.get("retrieval_status") or "not_started")
    conversation_context = state.get("conversation_context") or {}
    if not has_query:
        return AnswerabilityResult(
            is_in_domain=in_domain,
            has_relevant_context=False,
            has_minimum_evidence=False,
            can_answer_directly=False,
            requires_clarification=True,
            requires_cautious_language=True,
            requires_fallback=True,
            reason_code="EMPTY_QUERY",
        )
    if not in_domain:
        return AnswerabilityResult(
            is_in_domain=False,
            has_relevant_context=False,
            has_minimum_evidence=False,
            can_answer_directly=False,
            requires_clarification=False,
            requires_cautious_language=True,
            requires_fallback=False,
            reason_code="OUT_OF_DOMAIN",
        )
    if isinstance(conversation_context, dict) and conversation_context.get("unresolved_user_reference"):
        return AnswerabilityResult(
            is_in_domain=True,
            has_relevant_context=has_evidence,
            has_minimum_evidence=has_evidence,
            can_answer_directly=False,
            requires_clarification=True,
            requires_cautious_language=True,
            requires_fallback=True,
            reason_code="AMBIGUOUS_USER_REFERENCE",
        )
    if has_evidence:
        return AnswerabilityResult(
            is_in_domain=True,
            has_relevant_context=True,
            has_minimum_evidence=True,
            can_answer_directly=True,
            requires_clarification=False,
            requires_cautious_language=status in {"recoverable_error", "no_evidence"},
            requires_fallback=False,
            reason_code="RECOVERABLE_WITH_EVIDENCE" if status == "recoverable_error" else "RETRIEVED_EVIDENCE",
        )
    return AnswerabilityResult(
        is_in_domain=True,
        has_relevant_context=False,
        has_minimum_evidence=False,
        can_answer_directly=False,
        requires_clarification=False,
        requires_cautious_language=True,
        requires_fallback=True,
        reason_code="RETRIEVAL_ERROR" if status == "recoverable_error" else "NO_RELEVANT_CONTEXT",
    )


def decide_retrieval_fallback(state: dict[str, Any]) -> SafeFallbackDecision:
    answerability = assess_answerability(state)
    status = str(state.get("retrieval_status") or "not_started")
    if answerability.reason_code == "EMPTY_QUERY":
        return SafeFallbackDecision(
            fallback_applied=True,
            fallback_type="empty_query",
            fallback_reason="Câu hỏi rỗng hoặc chưa đủ nội dung để xử lý.",
            fallback_cache_eligible=False,
        )
    if answerability.reason_code == "AMBIGUOUS_USER_REFERENCE":
        return SafeFallbackDecision(
            fallback_applied=True,
            fallback_type="ambiguous_reference",
            fallback_reason="Cần làm rõ entity mà đại từ trong câu hỏi đang đề cập.",
            fallback_cache_eligible=False,
        )
    if answerability.can_answer_directly:
        return SafeFallbackDecision(fallback_applied=False, fallback_type="none", fallback_cache_eligible=True)
    if status == "recoverable_error":
        return SafeFallbackDecision(
            fallback_applied=True,
            fallback_type="retrieval_error",
            fallback_reason=sanitize_fallback_reason(state.get("retrieval_error")),
            fallback_cache_eligible=False,
        )
    if status == "insufficient_context":
        return SafeFallbackDecision(
            fallback_applied=True,
            fallback_type="insufficient_context",
            fallback_reason="Context packer không chọn được bằng chứng đủ dùng.",
            fallback_cache_eligible=False,
        )
    if not answerability.has_minimum_evidence:
        return SafeFallbackDecision(
            fallback_applied=True,
            fallback_type="no_retrieval_evidence",
            fallback_reason="Không có vector context, graph fact hoặc packed context usable.",
            fallback_cache_eligible=False,
        )
    return SafeFallbackDecision(fallback_applied=False, fallback_type="none", fallback_cache_eligible=True)


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
    if fallback_type == "ambiguous_reference":
        return (
            "Mình cần làm rõ đối tượng bạn đang nói tới trước khi trả lời chính xác. "
            "Bạn có thể cho biết tên thuốc, sản phẩm hoặc hoạt chất đang hỏi không?"
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


def _item_has_text(item: Any) -> bool:
    if hasattr(item, "model_dump"):
        item = item.model_dump(mode="json")
    if not isinstance(item, dict):
        return False
    return is_usable_text(item.get("text") or item.get("content") or item.get("page_content"))


def _graph_fact_has_evidence(item: Any) -> bool:
    if hasattr(item, "model_dump"):
        item = item.model_dump(mode="json")
    if not isinstance(item, dict):
        return False
    return any(
        is_usable_text(item.get(key))
        for key in ("evidence", "description", "related_description", "entity", "related_entity")
    )


__all__ = [
    "SAFE_FALLBACK_FLOW_VERSION",
    "FallbackType",
    "RetrievalStatus",
    "SafeFallbackDecision",
    "build_safe_fallback_answer",
    "assess_answerability",
    "decide_generation_fallback",
    "decide_retrieval_fallback",
    "has_usable_evidence",
    "has_usable_packed_context",
    "is_usable_text",
    "sanitize_fallback_reason",
]
