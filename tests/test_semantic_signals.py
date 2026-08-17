from __future__ import annotations

import pytest

from src.agent.semantic_signals import (
    BREATHING_DIFFICULTY_CONCEPTS,
    contains_bounded_sequence,
    has_active_symptom,
    has_medication_related_active_symptom,
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
    ("text", "expected"),
    [
        ("Hôm qua sốt. Tôi khó thở.", True),
        ("Nếu sốt thì nghỉ; tôi khó thở.", True),
        ("Tôi không đau đầu. Tôi khó thở.", True),
        ("Buồn nôn đã hết. Tôi khó thở.", True),
        ("Tôi khó thở. Buồn nôn đã hết.", True),
        ("Hôm qua tôi khó thở.", False),
        ("Hôm qua tôi khó thở; hiện tôi lại khó thở.", True),
        ("Tôi không khó thở.", False),
        ("Tôi khó thở nhưng đã hết.", False),
        ("HOM QUA SOT; TOI KHO THO!", True),
        ("Hôm qua sốt:\nHiện tôi KHÓ THỞ?", True),
    ],
)
def test_active_symptom_state_is_bounded_to_the_local_clause(
    text: str,
    expected: bool,
) -> None:
    bounded_text = normalize_text(text, preserve_boundaries=True)

    assert has_active_symptom(bounded_text, ("kho tho",)) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Tôi đang khó thở.", True),
        ("Tôi đang hụt hơi.", True),
        ("Tôi cảm thấy thở không đủ hơi.", True),
        ("Hiện tôi đang thở gấp!", True),
        ("TÔI ĐANG THỞ KHÔNG ĐỦ HƠI?", True),
        ("Tôi không thở gấp.", False),
        ("Tôi từng thở không đủ hơi nhưng đã hết.", False),
        ("Nếu sau này thở gấp thì sao?", False),
    ],
)
def test_shared_breathing_concepts_reuse_occurrence_local_state(
    text: str,
    expected: bool,
) -> None:
    bounded_text = normalize_text(text, preserve_boundaries=True)

    assert has_active_symptom(bounded_text, BREATHING_DIFFICULTY_CONCEPTS) is expected


@pytest.mark.parametrize(
    "text",
    [
        (
            "Khoảng 15 phút sau khi tôi thoa thuốc trị mụn,\n"
            "bây giờ tôi cảm thấy thở không đủ hơi."
        ),
        "Khoảng vài phút sau khi tôi bôi thuốc trị mụn, giờ tôi thấy khó thở.",
        "Tầm mười phút sau lúc tôi thoa thuốc, hiện tôi đang hụt hơi.",
        (
            "Sau khi tôi vừa dùng thuốc trị mụn, một lúc sau "
            "tôi thấy thở không đủ hơi."
        ),
        "Hai mươi phút sau khi tôi uống thuốc; bây giờ tôi thở gấp.",
        "KHOANG VAI PHUT SAU KHI TOI BOI THUOC TRI MUN, HIEN TOI DANG KHO THO!",
    ],
)
def test_medication_symptom_relation_accepts_owned_temporal_prefix(text: str) -> None:
    bounded_text = normalize_text(text, preserve_boundaries=True)

    assert has_medication_related_active_symptom(
        bounded_text,
        BREATHING_DIFFICULTY_CONCEPTS,
    ) is True


@pytest.mark.parametrize(
    "text",
    [
        "Vài phút sau khi mẹ tôi bôi thuốc trị mụn, giờ tôi thấy khó thở.",
        "Vài phút sau khi tôi bôi thuốc trị mụn, giờ mẹ tôi thấy khó thở.",
        "Nếu sau khi tôi bôi thuốc mà khó thở thì sao?",
        "Sau khi tôi bôi thuốc hôm qua có hụt hơi nhưng giờ đã hết.",
        (
            "Sau khi tôi bôi thuốc, tôi chạy bộ và giờ thở gấp vì chạy bộ, "
            "không liên quan thuốc."
        ),
        "Hôm qua sau khi tôi bôi thuốc. Hôm nay tôi khó thở vì chạy bộ.",
        "Sau khi tôi bôi thuốc, hiện tôi đang thở bình thường.",
        "Sau khi tôi bôi thuốc, hiện tôi không khó thở.",
        "Sau khi tôi chạy bộ, tôi bôi thuốc trị mụn; bây giờ tôi thấy thở gấp.",
    ],
)
def test_medication_symptom_relation_rejects_unowned_or_unrelated_context(
    text: str,
) -> None:
    bounded_text = normalize_text(text, preserve_boundaries=True)

    assert has_medication_related_active_symptom(
        bounded_text,
        BREATHING_DIFFICULTY_CONCEPTS,
    ) is False


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
        "Bạn quyết định giúp tôi nên dùng isotretinoin bao nhiêu mg mỗi ngày.",
        "Bạn chọn giúp tôi uống doxycycline mấy viên mỗi ngày.",
        "Hãy xác định giúp tôi bôi tretinoin nồng độ bao nhiêu phần trăm.",
        "Bạn tính giúp tôi cần dùng spironolactone bao nhiêu mg một ngày.",
    ],
)
def test_prescription_execution_recognizes_personalized_quantitative_selection(
    question: str,
) -> None:
    assert is_prescription_execution_request(question) is True


@pytest.mark.parametrize(
    "question",
    [
        (
            "Trước đây bác sĩ đã kê isotretinoin 20 mg. "
            "Bây giờ bạn quyết định giúp tôi nên dùng isotretinoin bao nhiêu mg mỗi ngày."
        ),
        (
            "Tài liệu nói về liều isotretinoin trước đây; "
            "bây giờ bạn xác định giúp tôi dùng isotretinoin bao nhiêu mg mỗi ngày."
        ),
        (
            'Tôi từng thấy "20 mg" trên hộp. '
            "Bây giờ bạn quyết định giúp tôi dùng isotretinoin bao nhiêu mg mỗi ngày."
        ),
        (
            "Nghiên cứu ghi nhận nhiều hàm lượng.\n"
            "Bạn chọn giúp tôi uống doxycycline mấy viên mỗi ngày."
        ),
        (
            "Bác sĩ từng trao đổi về thuốc: "
            "hiện bạn tính giúp tôi dùng spironolactone bao nhiêu mg một ngày."
        ),
        (
            "Trước đây tôi đã dùng isotretinoin 20 mg, nhưng bây giờ "
            "bạn quyết định giúp tôi dùng isotretinoin bao nhiêu mg mỗi ngày."
        ),
        "BAN XAC DINH GIUP TOI DUNG ISOTRETINOIN BAO NHIEU MG MOI NGAY?",
    ],
)
def test_prescription_execution_composes_current_request_within_local_span(
    question: str,
) -> None:
    assert is_prescription_execution_request(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "Isotretinoin 20 mg là dạng thuốc gì?",
        "Isotretinoin có những hàm lượng nào?",
        "Tài liệu nói gì về liều isotretinoin?",
        "Bác sĩ đã kê cho tôi isotretinoin 20 mg mỗi ngày.",
        "Nghiên cứu này sử dụng isotretinoin bao nhiêu mg?",
        "Bạn tôi nên dùng isotretinoin bao nhiêu mg mỗi ngày?",
        "Trước đây tôi đã dùng isotretinoin 20 mg mỗi ngày.",
        "Đừng quyết định giúp tôi dùng isotretinoin bao nhiêu mg.",
        'Bác sĩ hỏi: "Bạn quyết định giúp tôi dùng isotretinoin bao nhiêu mg?"',
    ],
)
def test_prescription_execution_rejects_non_executing_quantitative_mentions(
    question: str,
) -> None:
    assert is_prescription_execution_request(question) is False


@pytest.mark.parametrize(
    "question",
    [
        (
            "Bạn quyết định giúp tôi lịch tái khám. "
            "Isotretinoin có hàm lượng nào?"
        ),
        (
            "Tôi không nhờ bạn quyết định cho tôi; "
            "isotretinoin có hàm lượng bao nhiêu mg?"
        ),
        (
            "Tôi đang tìm hiểu cho mẹ. "
            "Bạn quyết định giúp mẹ tôi dùng isotretinoin bao nhiêu mg mỗi ngày."
        ),
        "Bạn quyết định giúp mẹ tôi dùng isotretinoin bao nhiêu mg mỗi ngày.",
        (
            "Tôi từng nghe bác sĩ nói.\n"
            'Bác sĩ hỏi: "Bạn quyết định giúp tôi dùng isotretinoin bao nhiêu mg?"'
        ),
        (
            "Bác sĩ đã quyết định cho tôi dùng isotretinoin 20 mg mỗi ngày; "
            "hiện tôi chỉ muốn biết thuốc có những hàm lượng nào."
        ),
        (
            "Bạn quyết định giúp tôi lịch tái khám vào tháng tới, còn "
            "isotretinoin có hàm lượng bao nhiêu mg?"
        ),
        (
            "Bạn quyết định giúp tôi dùng isotretinoin trong đợt tới, còn "
            "doxycycline có hàm lượng bao nhiêu mg?"
        ),
        "Tôi không nhờ bạn quyết định giúp tôi dùng isotretinoin bao nhiêu mg mỗi ngày.",
    ],
)
def test_prescription_execution_does_not_mix_unrelated_local_spans(
    question: str,
) -> None:
    assert is_prescription_execution_request(question) is False


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "Trước đây bác sĩ đã kê isotretinoin 20 mg. "
            "Bây giờ bạn quyết định giúp tôi dùng isotretinoin bao nhiêu mg mỗi ngày.",
            True,
        ),
        (
            "Bạn quyết định giúp tôi lịch tái khám. "
            "Isotretinoin có hàm lượng bao nhiêu mg?",
            False,
        ),
    ],
)
def test_medication_management_preserves_local_request_boundaries(
    question: str,
    expected: bool,
) -> None:
    assert is_medication_management_intent(question) is expected


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
