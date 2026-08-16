from __future__ import annotations

import pytest

from src.agent.semantic_signals import (
    contains_bounded_sequence,
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
        "Adapalene và benzoyl peroxide khác nhau thế nào?",
        "Adapalene có tốt hơn benzoyl peroxide không?",
        "Thuốc nào tác dụng nhanh hơn?",
        "Loại nào hiệu quả hơn và ít kích ứng hơn?",
    ],
)
def test_comparison_intent_covers_explicit_and_comparative_forms(question: str) -> None:
    assert is_comparison_intent(question) is True
