from __future__ import annotations

import pytest

from src.agent.semantic_signals import (
    contains_bounded_sequence,
    has_active_symptom,
    has_unnegated_concept,
    is_comparison_intent,
    is_medication_management_intent,
    is_prescription_execution_request,
    normalize_text,
)


def test_normalization_is_accent_case_and_punctuation_stable() -> None:
    assert normalize_text("  KÊ ĐƠN, cho tôi! ") == "ke don cho toi"


def test_bounded_sequence_allows_narrow_modifiers_but_not_unbounded_distance() -> None:
    assert contains_bounded_sequence(
        "toi uong thuoc tri mun moi duoc hai muoi phut thi kho tho",
        ("uong", "thuoc", "kho tho"),
        max_gap=8,
    )
    assert not contains_bounded_sequence(
        "uong thuoc " + "mot " * 20 + "kho tho",
        ("uong", "thuoc", "kho tho"),
        max_gap=8,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Tôi đang rất khó thở.", True),
        ("Tôi không khó thở.", False),
        ("Tôi đã hết khó thở.", False),
    ],
)
def test_unnegated_concept_respects_local_negation(text: str, expected: bool) -> None:
    assert has_unnegated_concept(text, ("kho tho",)) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Buồn nôn đã hết, nhưng tôi hiện đang khó thở.", True),
        ("Hôm qua tôi đau đầu. Hôm nay tôi đang khó thở.", True),
        ("Hôm qua tôi từng khó thở và sau đó đã ổn.", False),
        ("Hôm qua tôi từng khó thở, nhưng hôm nay tôi lại đang khó thở.", True),
        ("Tôi đang khó thở. Thuốc này có thể gây dị ứng không?", True),
        ("Nếu thuốc có thể gây dị ứng thì sao? Hiện tôi đang khó thở.", True),
        ("Tôi đang khó thở. Hôm qua tôi chỉ bị đau đầu.", True),
        ("Tôi từng khó thở nhưng giờ tôi lại khó thở.", True),
        ("Nếu hai giờ nữa tôi khó thở thì cần làm gì?", False),
        ("Nếu trong giờ tới tôi khó thở thì cần làm gì?", False),
        ("Tôi hiện không khó thở.", False),
        ("HIỆN TÔI ĐANG KHÓ THỞ!", True),
    ],
)
def test_active_symptom_state_is_local_to_each_concept_occurrence(
    text: str,
    expected: bool,
) -> None:
    assert has_active_symptom(text, ("kho tho",)) is expected


@pytest.mark.parametrize(
    "question",
    [
        "Kê đơn cho tôi.",
        "Kê cho tôi một đơn thuốc trị mụn.",
        "Bạn chọn giúp tôi liều phù hợp.",
        "Cho tôi liều dùng phù hợp.",
    ],
)
def test_prescription_execution_recognizes_request_semantics(question: str) -> None:
    assert is_prescription_execution_request(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "Đừng kê đơn cho tôi.",
        "Bác sĩ đã kê đơn cho tôi.",
        "Khi nào bác sĩ kê đơn?",
        "Tài liệu nói gì về việc kê đơn?",
    ],
)
def test_prescription_execution_rejects_negated_historical_or_reference_forms(
    question: str,
) -> None:
    assert is_prescription_execution_request(question) is False


@pytest.mark.parametrize(
    "question",
    [
        "Tôi có nên tăng số lần bôi adapalene không?",
        "Dùng benzoyl peroxide bao lâu?",
        "Tôi nên uống doxycycline với tần suất thế nào?",
        "Tôi có thể bôi tretinoin cùng benzoyl peroxide không?",
        "Kê cho tôi một đơn thuốc trị mụn.",
    ],
)
def test_medication_management_uses_shared_request_signals(question: str) -> None:
    assert is_medication_management_intent(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "Tôi có nên bôi tretinoin không?",
        "Tôi nên dùng tazarotene thế nào?",
        "Tôi có nên giảm tần suất dùng azelaic acid không?",
        "Tôi nên dùng salicylic acid bao lâu?",
        "Tôi có thể bôi clascoterone không?",
        "Tôi có nên uống minocycline không?",
        "Tôi nên dùng sarecycline thế nào?",
        "Tôi có nên giảm liều spironolactone không?",
        "Tôi có thể bôi trifarotene mỗi tối không?",
    ],
)
def test_medication_management_uses_canonical_supported_identity(question: str) -> None:
    assert is_medication_management_intent(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "Tretinoin là gì?",
        "Tazarotene dùng để làm gì?",
        "Azelaic acid có tác dụng gì?",
        "Trifarotene dùng để trị gì?",
    ],
)
def test_supported_medication_factual_questions_are_not_management(question: str) -> None:
    assert is_medication_management_intent(question) is False


@pytest.mark.parametrize(
    "question",
    [
        "Adapalene là gì?",
        "Benzoyl peroxide dùng để làm gì?",
        "Có nên dùng sữa rửa mặt này không?",
        "Chăm sóc mụn viêm thế nào?",
    ],
)
def test_medication_management_rejects_facts_and_ordinary_skincare(question: str) -> None:
    assert is_medication_management_intent(question) is False


def test_frequency_language_without_medication_does_not_become_medication_management() -> None:
    assert is_medication_management_intent("Tôi có nên tăng số lần rửa mặt không?") is False


@pytest.mark.parametrize(
    "question",
    [
        "Tôi có nên bôi kem dưỡng không?",
        "Tôi nên dùng sữa rửa mặt thế nào?",
        "Tôi có nên tăng tần suất dùng kem chống nắng không?",
    ],
)
def test_ordinary_skincare_remains_outside_medication_management(question: str) -> None:
    assert is_medication_management_intent(question) is False


@pytest.mark.parametrize(
    "question",
    [
        "Adapalene và benzoyl peroxide khác nhau thế nào?",
        "Adapalene có tốt hơn benzoyl peroxide không?",
        "Thuốc nào tác dụng nhanh hơn?",
        "Loại nào hiệu quả hơn và ít kích ứng hơn?",
    ],
)
def test_comparison_intent_covers_explicit_and_comparative_forms(question: str) -> None:
    assert is_comparison_intent(question) is True
