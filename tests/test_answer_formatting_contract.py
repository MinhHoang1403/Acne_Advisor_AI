from __future__ import annotations

import pytest

from src.agent.answer_formatting import (
    ANSWER_FORMATTING_CONTRACT_VERSION,
    CANONICAL_DISCLAIMER,
    answer_format_instruction_for_question,
    finalize_answer_presentation,
    normalize_answer_markdown,
    should_include_medical_disclaimer,
)
from src.agent.nodes.respond import finalize_response_node
from src.agent.prompts.medical_answer import MEDICAL_RAG_SYSTEM_PROMPT


def test_formatting_contract_version_and_comparison_shape_are_current() -> None:
    instruction = answer_format_instruction_for_question("A và B khác nhau thế nào?")
    assert ANSWER_FORMATTING_CONTRACT_VERSION == "answer_formatting_contract_v14"
    assert "đối chiếu" in instruction
    assert "evidence" in instruction


def test_direct_definition_instruction_is_short_without_forcing_a_list() -> None:
    instruction = answer_format_instruction_for_question("Mụn đầu đen là gì?")

    assert "tối đa hai đoạn ngắn" in instruction
    assert "không tự tạo danh sách" in instruction


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
    assert result["final_answer"].startswith(draft)
    assert CANONICAL_DISCLAIMER not in result["final_answer"]


def test_medical_prompt_requires_natural_proportional_answers_without_source_narration() -> None:
    prompt = MEDICAL_RAG_SYSTEM_PROMPT

    assert "tiếng Việt tự nhiên" in prompt
    assert "đủ ý để câu trả lời không dừng ở một định nghĩa cụt" in prompt
    assert "chỉ bổ sung chi tiết được EVIDENCE hỗ trợ" in prompt
    assert "một hoặc hai đoạn ngắn" in prompt
    assert 'không mở đầu hoặc lặp lại "Theo tài liệu..."' in prompt
    assert "không tự thêm nồng độ, liều, lịch dùng, thời gian, tổng liều" in prompt
    assert "không chẩn đoán, chọn điều trị cho người dùng, kê đơn" in prompt


@pytest.mark.asyncio
async def test_normal_and_cached_informational_answers_receive_no_generic_disclaimer() -> None:
    generated = await finalize_response_node(
        {"user_question": "Mụn là gì?", "draft_answer": "Câu trả lời ngắn."}
    )
    cached = await finalize_response_node(
        {
            "user_question": "Mụn là gì?",
            "cache_hit": True,
            "cached_answer": f"Câu trả lời đã lưu.\n\n{CANONICAL_DISCLAIMER}",
        }
    )

    assert CANONICAL_DISCLAIMER not in generated["final_answer"]
    assert CANONICAL_DISCLAIMER not in cached["final_answer"]


@pytest.mark.asyncio
async def test_emergency_receives_disclaimer_after_action_and_safe_fallback_does_not() -> None:
    emergency = await finalize_response_node(
        {
            "user_question": "Tôi khó thở.",
            "draft_answer": "Hãy gọi cấp cứu ngay.",
            "safety_severity": "emergency",
        }
    )
    fallback = await finalize_response_node(
        {
            "user_question": "Mụn là gì?",
            "draft_answer": "Mình chưa có đủ thông tin đáng tin cậy.",
            "fallback_applied": True,
            "fallback_type": "insufficient_context",
        }
    )

    assert emergency["final_answer"].startswith("Hãy gọi cấp cứu ngay.")
    assert emergency["final_answer"].count(CANONICAL_DISCLAIMER) == 1
    assert CANONICAL_DISCLAIMER not in fallback["final_answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "Thuốc nào có thể trị mụn trứng cá?",
        "Tôi nên dùng thuốc trị mụn thế nào?",
        "Có nên tăng liều thuốc trị mụn không?",
    ],
)
async def test_explicit_medication_advice_receives_one_canonical_disclaimer(
    question: str,
) -> None:
    result = await finalize_response_node(
        {"user_question": question, "draft_answer": "Thông tin điều trị khái quát."}
    )

    assert result["final_answer"].count(CANONICAL_DISCLAIMER) == 1


@pytest.mark.parametrize(
    "question",
    [
        "Mụn đầu đen là gì?",
        "Mụn viêm là gì?",
        "Da dầu có gây mụn không?",
        "Adapalene là gì?",
        "Adapalene dùng để làm gì?",
    ],
)
def test_ordinary_information_does_not_require_generic_disclaimer(question: str) -> None:
    assert should_include_medical_disclaimer(question) is False


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Tài liệu hiện có chưa đủ để trả lời câu hỏi này.", False),
        ("Tôi nên dùng thuốc trị mụn thế nào?", True),
    ],
)
def test_fallback_disclaimer_depends_on_underlying_question(
    question: str,
    expected: bool,
) -> None:
    assert (
        should_include_medical_disclaimer(question, fallback_type="safe_fallback")
        is expected
    )


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
