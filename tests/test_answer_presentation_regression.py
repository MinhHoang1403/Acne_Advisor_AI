from __future__ import annotations

from src.agent.answer_formatting import finalize_answer_presentation


def test_presentation_does_not_rewrite_model_polarity() -> None:
    answer = finalize_answer_presentation(
        "Không. Thai kỳ có làm thay đổi cách đánh giá isotretinoin.",
        user_question="Thai kỳ có làm thay đổi cách đánh giá isotretinoin không?",
    )

    assert answer.startswith("Không. Thai kỳ có làm thay đổi cách đánh giá isotretinoin")


def test_style_contract_removes_repeated_disclaimer() -> None:
    answer = finalize_answer_presentation(
        "Trả lời ngắn.\n\nThông tin mang tính tham khảo và không thay thế chẩn đoán của bác sĩ.\n\nThông tin mang tính tham khảo và không thay thế chẩn đoán của bác sĩ.",
        user_question="Benzoyl peroxide là gì?",
    )

    assert answer.count("Thông tin mang tính tham khảo") <= 1
