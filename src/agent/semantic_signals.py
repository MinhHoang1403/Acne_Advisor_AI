"""Tín hiệu ngôn ngữ deterministic nhỏ dùng chung cho các runtime contract.

Các hàm trong module này chỉ chuẩn hóa và nhận diện những khái niệm đã có owner
deterministic. Chúng không phân loại y khoa tổng quát và không thay thế LLM.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence


NEGATION_TOKENS = frozenset({"khong", "chua", "not", "never"})
RESOLUTION_PREFIXES = (
    "da het",
    "khong con",
    "gio het",
    "hien khong",
    "tung",
    "tung bi",
    "truoc day",
    "truoc day bi",
    "hom qua",
)
RESOLUTION_MARKERS = (
    "da het",
    "khong con",
    "gio het",
    "hien tai binh thuong",
    "hien gio binh thuong",
    "dang binh thuong",
    "tro lai binh thuong",
)
HYPOTHETICAL_MARKERS = ("neu", "gia su", "co gay", "co the gay", "trieu chung nao")
FIRST_PERSON = frozenset({"toi", "minh", "em", "tui"})
MEDICATION_ACTIONS = ("dung", "uong", "boi", "thoa", "tai dung")
MEDICATION_TERMS = (
    "thuoc",
    "adapalene",
    "benzoyl peroxide",
    "isotretinoin",
    "clindamycin",
    "erythromycin",
    "doxycycline",
    "retinoid",
)
ORDINARY_SKINCARE_TERMS = (
    "sua rua mat",
    "kem duong",
    "kem chong nang",
    "my pham",
    "san pham nay",
    "cham soc",
    "routine",
)


def normalize_text(value: str) -> str:
    """Case/accent/punctuation fold ổn định cho matching có giới hạn."""

    decomposed = unicodedata.normalize("NFD", str(value or "").casefold().replace("đ", "d"))
    accentless = "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", accentless).split())


def contains_bounded_sequence(
    text: str,
    concepts: Sequence[str],
    *,
    max_gap: int = 3,
) -> bool:
    """Khớp các phrase theo thứ tự với tối đa ``max_gap`` token xen giữa."""

    tokens = normalize_text(text).split()
    phrases = [normalize_text(concept).split() for concept in concepts]
    if not tokens or not phrases or any(not phrase for phrase in phrases):
        return False

    ends = _phrase_ends(tokens, phrases[0], start=0)
    for phrase in phrases[1:]:
        next_ends: list[int] = []
        for previous_end in ends:
            latest_start = min(len(tokens), previous_end + max_gap + 1)
            next_ends.extend(
                _phrase_ends(tokens, phrase, start=previous_end, stop=latest_start)
            )
        ends = next_ends
        if not ends:
            return False
    return bool(ends)


def has_unnegated_concept(
    text: str,
    concepts: Iterable[str],
    *,
    negation_window: int = 3,
) -> bool:
    """Nhận diện concept không bị phủ định hoặc đánh dấu đã hết ngay trước nó."""

    normalized = normalize_text(text)
    tokens = normalized.split()
    for concept in concepts:
        phrase = normalize_text(concept).split()
        for start, _ in _phrase_spans(tokens, phrase):
            prefix = tokens[max(0, start - negation_window) : start]
            prefix_text = " ".join(prefix)
            if NEGATION_TOKENS.intersection(prefix):
                continue
            if any(prefix_text.endswith(marker) for marker in RESOLUTION_PREFIXES):
                continue
            return True
    return False


def has_active_symptom(text: str, concepts: Iterable[str]) -> bool:
    """Trả true cho symptom hiện tại, loại negation, hypothetical và trạng thái đã hết."""

    normalized = normalize_text(text)
    if any(marker in normalized for marker in HYPOTHETICAL_MARKERS):
        return False
    if not has_unnegated_concept(normalized, concepts):
        return False
    if any(marker in normalized for marker in (*RESOLUTION_PREFIXES, *RESOLUTION_MARKERS)):
        current_markers = ("hien van", "bay gio dang", "hien dang", "van dang", "luc nay")
        return any(marker in normalized for marker in current_markers) and has_unnegated_concept(
            normalized, concepts
        )
    return True


def has_past_or_resolved_symptom(text: str, concepts: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    has_concept = any(normalize_text(concept) in normalized for concept in concepts)
    return has_concept and any(
        marker in normalized for marker in (*RESOLUTION_PREFIXES, *RESOLUTION_MARKERS)
    )


def has_first_person_reference(text: str) -> bool:
    normalized = normalize_text(text)
    third_person = (
        "ban toi",
        "me toi",
        "bo toi",
        "anh toi",
        "chi toi",
        "em toi",
        "anh ay",
        "co ay",
        "nguoi do",
    )
    return not any(marker in normalized for marker in third_person) and bool(
        FIRST_PERSON.intersection(normalized.split())
    )


def is_medication_use_event(text: str) -> bool:
    """Nhận diện người dùng mô tả hành vi dùng thuốc, không phải nhắc thuốc chung."""

    normalized = normalize_text(text)
    if not has_first_person_reference(normalized):
        return False
    return any(
        contains_bounded_sequence(normalized, (action, medication), max_gap=4)
        or contains_bounded_sequence(normalized, (medication, action), max_gap=4)
        for action in MEDICATION_ACTIONS
        for medication in MEDICATION_TERMS
    )


def is_prescription_execution_request(text: str) -> bool:
    """Nhận diện yêu cầu kê/chọn đơn hoặc liều cho chính người dùng."""

    normalized = normalize_text(text)
    tokens = normalized.split()
    if not has_first_person_reference(normalized):
        return False
    if any(marker in normalized for marker in ("bac si da", "khi nao bac si", "tai lieu", "noi gi ve")):
        return False

    action_positions = [
        start
        for phrase in ("ke", "chon", "cho")
        for start, _ in _phrase_spans(tokens, [phrase])
    ]
    if any(
        ({*NEGATION_TOKENS, "dung"}).intersection(tokens[max(0, position - 2) : position])
        for position in action_positions
    ):
        return False

    patterns = (
        ("ke", "toi", "don"),
        ("ke", "toi", "thuoc"),
        ("ke", "don", "toi"),
        ("ke", "thuoc", "toi"),
        ("chon", "toi", "lieu"),
        ("chon", "lieu", "toi"),
        ("cho", "toi", "lieu"),
        ("cho", "toi", "toa"),
        ("cho", "toi", "don"),
    )
    return any(contains_bounded_sequence(normalized, pattern, max_gap=3) for pattern in patterns)


def is_medication_management_intent(text: str) -> bool:
    """Phân biệt quản lý/cá nhân hóa thuốc với câu hỏi fact hoặc skincare chung."""

    normalized = normalize_text(text)
    if any(marker in normalized for marker in ("la gi", "dung de lam gi", "dung de tri gi", "co tac dung gi")):
        return False
    if any(marker in normalized for marker in ORDINARY_SKINCARE_TERMS):
        return False
    if is_prescription_execution_request(normalized):
        return True
    medication_present = any(term in normalized for term in MEDICATION_TERMS)
    if "thuoc nao" in normalized or any(
        marker in normalized for marker in ("tang lieu", "giam lieu", "chon lieu")
    ):
        return True
    if medication_present and any(
        marker in normalized for marker in ("tang so lan", "giam so lan", "tan suat")
    ):
        return True

    action_present = any(re.search(rf"(?:^| ){re.escape(action)}(?: |$)", normalized) for action in MEDICATION_ACTIONS)
    management_marker = any(
        marker in normalized
        for marker in (
            "co nen",
            "co the",
            "nen tu",
            "khong nen",
            "the nao",
            "ra sao",
            "bao lau",
            "may lan",
        )
    )
    return medication_present and action_present and management_marker


def is_comparison_intent(text: str) -> bool:
    """Nhận diện yêu cầu đối chiếu, không suy ra bên nào tốt hơn về y khoa."""

    normalized = normalize_text(text)
    explicit = ("khac nhau", "khac gi", "khac the nao", "so sanh", "doi chieu", "versus")
    comparative = ("tot hon", "nhanh hon", "hieu qua hon", "it kich ung hon")
    return "vs" in normalized.split() or any(
        marker in normalized for marker in (*explicit, *comparative)
    )


def _phrase_spans(tokens: list[str], phrase: list[str]) -> list[tuple[int, int]]:
    if not phrase:
        return []
    width = len(phrase)
    return [
        (index, index + width)
        for index in range(0, len(tokens) - width + 1)
        if tokens[index : index + width] == phrase
    ]


def _phrase_ends(
    tokens: list[str],
    phrase: list[str],
    *,
    start: int,
    stop: int | None = None,
) -> list[int]:
    latest_start = len(tokens) - len(phrase) + 1
    upper = latest_start if stop is None else min(stop, latest_start)
    return [
        index + len(phrase)
        for index in range(start, max(start, upper))
        if tokens[index : index + len(phrase)] == phrase
    ]


__all__ = [
    "contains_bounded_sequence",
    "has_active_symptom",
    "has_first_person_reference",
    "has_past_or_resolved_symptom",
    "has_unnegated_concept",
    "is_comparison_intent",
    "is_medication_management_intent",
    "is_medication_use_event",
    "is_prescription_execution_request",
    "normalize_text",
]
