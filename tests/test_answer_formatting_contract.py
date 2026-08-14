from __future__ import annotations

import pytest

from src.agent.answer_formatting import (
    ANSWER_FORMATTING_CONTRACT_VERSION,
    answer_format_instruction_for_question,
    finalize_answer_presentation,
    normalize_answer_markdown,
)
from src.agent.nodes.respond import finalize_response_node


def test_formatting_contract_version_and_comparison_shape_are_current() -> None:
    instruction = answer_format_instruction_for_question("A và B khác nhau thế nào?")
    assert ANSWER_FORMATTING_CONTRACT_VERSION == "answer_formatting_contract_v12"
    assert "đối chiếu" in instruction
    assert "evidence" in instruction


def test_exact_count_truncates_existing_bullets_without_inventing_content() -> None:
    answer = finalize_answer_presentation(
        "- Ý A\n- Ý B\n- Ý C\n- Ý D",
        user_question="Liệt kê đúng 3 ý.",
    )
    assert [line for line in answer.splitlines() if line.startswith("- ")] == [
        "- Ý A",
        "- Ý B",
        "- Ý C",
    ]


def test_markdown_cleanup_repairs_surface_form_only() -> None:
    cleaned = normalize_answer_markdown("  **Mục**  \n\n\nNội dung kiểm soát.  ")
    assert cleaned == "**Mục**\n\nNội dung kiểm soát."


@pytest.mark.asyncio
async def test_finalize_preserves_short_valid_model_answer() -> None:
    draft = "Model trả lời ngắn từ evidence."
    result = await finalize_response_node(
        {"user_question": "Câu hỏi?", "draft_answer": draft, "is_in_domain": True}
    )
    assert result["final_answer"] == draft


def test_question_echo_is_removed_without_rewriting_medical_meaning() -> None:
    answer = finalize_answer_presentation(
        "Câu hỏi: Da mụn là gì?\nNội dung từ model.",
        user_question="Da mụn là gì?",
    )
    assert answer == "Nội dung từ model."


def test_duplicate_heading_and_disclaimer_cleanup_is_idempotent() -> None:
    draft = (
        "**Lưu ý**\nA\n\n**Lưu ý**\nB\n\n"
        "Thông tin mang tính tham khảo và không thay thế chẩn đoán của bác sĩ.\n\n"
        "Thông tin mang tính tham khảo và không thay thế chẩn đoán của bác sĩ."
    )
    once = finalize_answer_presentation(draft, user_question="Câu hỏi?")
    twice = finalize_answer_presentation(once, user_question="Câu hỏi?")
    assert once == twice
    assert once.count("Thông tin mang tính tham khảo") <= 1
