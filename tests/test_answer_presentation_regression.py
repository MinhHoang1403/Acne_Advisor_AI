from __future__ import annotations

from src.agent.answer_formatting import finalize_answer_presentation


def test_pregnancy_isotretinoin_answer_is_direct_and_non_contradictory() -> None:
    answer = finalize_answer_presentation(
        "Không. Thai kỳ có làm thay đổi cách đánh giá isotretinoin.",
        user_question="Thai kỳ có làm thay đổi cách đánh giá isotretinoin không?",
    )

    assert answer.startswith("Có. Thai kỳ làm thay đổi cách đánh giá isotretinoin")
    assert "Không. Thai kỳ có làm thay đổi" not in answer


def test_style_contract_removes_repeated_disclaimer() -> None:
    answer = finalize_answer_presentation(
        "Trả lời ngắn.\n\nThông tin mang tính tham khảo và không thay thế chẩn đoán của bác sĩ.\n\nThông tin mang tính tham khảo và không thay thế chẩn đoán của bác sĩ.",
        user_question="Benzoyl peroxide là gì?",
    )

    assert answer.count("Thông tin mang tính tham khảo") <= 1
