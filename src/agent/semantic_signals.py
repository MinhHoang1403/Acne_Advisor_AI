"""Tín hiệu ngôn ngữ deterministic nhỏ dùng chung cho các runtime contract.

Các hàm trong module này chỉ chuẩn hóa và nhận diện những khái niệm đã có owner
deterministic. Chúng không phân loại y khoa tổng quát và không thay thế LLM.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from functools import lru_cache

from src.knowledge.normalizer import DrugEntityNormalizer


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
HYPOTHETICAL_MARKERS = (
    "neu",
    "if",
    "gia su",
    "co gay",
    "co the gay",
    "trieu chung nao",
)
FIRST_PERSON = frozenset({"toi", "minh", "em", "tui", "i"})
MEDICATION_ACTIONS = ("dung", "uong", "boi", "thoa", "tai dung")
PLANNED_EVENT_MARKERS = ("dinh", "du dinh", "se", "co nen", "co the")
CURRENT_STATE_MARKERS = (
    "hom nay",
    "hien tai",
    "hien gio",
    "bay gio",
    "luc nay",
    "den gio van",
    "hien",
    "nhung gio",
    "gio minh",
    "gio em",
    "gio tui",
    "gio day",
    "dang",
    "van",
)
HISTORICAL_STATE_MARKERS = (
    "hom qua",
    "truoc day",
    "da tung",
    "tung bi",
    "tung",
    "used to",
)
RESOLVED_AFTER_MARKERS = (
    "da het",
    "gio da het",
    "khong con nua",
    "hien tai binh thuong",
    "hien gio binh thuong",
    "tro lai binh thuong",
)
HISTORICAL_AFTER_MARKERS = ("truoc day", "hom qua", "toi qua")
THIRD_PERSON_SUBJECTS = frozenset(
    {
        "nguoi",
        "ban",
        "me",
        "bo",
        "anh",
        "chi",
        "be",
        "benh nhan",
        "person",
        "friend",
        "mother",
        "father",
        "patient",
        "someone",
    }
)
REPORTING_VERBS = frozenset({"said", "says", "told", "noi"})
CLAUSE_BOUNDARY = "clauseboundary"
SOFT_CLAUSE_BOUNDARY = "softclauseboundary"
QUOTE_BOUNDARY = "quoteboundary"
ORDINARY_SKINCARE_TERMS = (
    "sua rua mat",
    "kem duong",
    "kem chong nang",
    "my pham",
    "san pham nay",
    "cham soc",
    "routine",
)


def normalize_text(value: str, *, preserve_boundaries: bool = False) -> str:
    """Case/accent/punctuation fold ổn định cho matching có giới hạn."""

    raw = str(value or "").casefold()
    if preserve_boundaries:
        raw = re.sub(r'["“”]', f" {QUOTE_BOUNDARY} ", raw)
        raw = re.sub(r"[;:]+", f" {SOFT_CLAUSE_BOUNDARY} ", raw)
        raw = re.sub(r"[.!?\r\n]+", f" {CLAUSE_BOUNDARY} ", raw)
    decomposed = unicodedata.normalize("NFD", raw.replace("đ", "d"))
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
    """Trả true khi ít nhất một occurrence của symptom đang hoạt động."""

    tokens = normalize_text(text).split()
    return any(
        _occurrence_is_active(tokens, start, end)
        for concept in concepts
        for start, end in _phrase_spans(tokens, normalize_text(concept).split())
    )


def has_past_or_resolved_symptom(text: str, concepts: Iterable[str]) -> bool:
    tokens = normalize_text(text).split()
    return any(
        not _occurrence_is_active(tokens, start, end)
        for concept in concepts
        for start, end in _phrase_spans(tokens, normalize_text(concept).split())
    )


def has_first_person_reference(
    text: str,
    related_concepts: Iterable[str] | None = None,
) -> bool:
    """Nhận diện ngôi thứ nhất, có thể ràng buộc ownership với concept gần nó."""

    tokens = normalize_text(text).split()
    if related_concepts is None:
        return any(
            token in FIRST_PERSON and not _is_possessive_first_person(tokens, index)
            for index, token in enumerate(tokens)
        )
    return any(
        _span_has_first_person_owner(tokens, start)
        for concept in related_concepts
        for start, _ in _phrase_spans(tokens, normalize_text(concept).split())
    )


def is_medication_use_event(text: str) -> bool:
    """Nhận diện người dùng mô tả hành vi dùng thuốc, không phải nhắc thuốc chung."""

    normalized = normalize_text(text)
    if not has_first_person_reference(normalized):
        return False
    return bool(_medication_event_spans(normalized.split()))


def has_medication_related_active_symptom(
    text: str,
    concepts: Iterable[str],
    *,
    max_distance: int = 16,
) -> bool:
    """Ghép event dùng thuốc với symptom hiện tại bằng quan hệ thời gian hẹp."""

    normalized = normalize_text(text)
    if "khong lien quan thuoc" in normalized:
        return False
    tokens = normalized.split()
    active_symptoms = [
        (start, end)
        for concept in concepts
        for start, end in _phrase_spans(tokens, normalize_text(concept).split())
        if _occurrence_is_active(tokens, start, end)
    ]
    for event_start, event_end in _medication_event_spans(tokens):
        for symptom_start, symptom_end in active_symptoms:
            symptom_owner = _nearest_subject_owner(tokens, symptom_start)
            if symptom_owner is False:
                continue
            if symptom_owner is None and _nearest_subject_owner(tokens, event_start) is not True:
                continue
            distance = max(0, max(event_start, symptom_start) - min(event_end, symptom_end))
            if distance > max_distance:
                continue
            event_before_symptom = event_start <= symptom_start
            between = (
                tokens[event_end:symptom_start]
                if event_before_symptom
                else tokens[symptom_end:event_start]
            )
            context_start = max(0, min(event_start, symptom_start) - 3)
            context_end = min(len(tokens), max(event_end, symptom_end) + 3)
            event_prefix = tokens[max(0, event_start - 3) : event_start]
            if _has_medication_symptom_relation(
                between,
                context_tokens=tokens[context_start:context_end],
                event_prefix_tokens=event_prefix,
                event_before_symptom=event_before_symptom,
            ):
                return True
    return False


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
    medication_present = _has_medication_identity(normalized)
    if "thuoc nao" in normalized:
        return True
    if not medication_present:
        return False
    if any(marker in normalized for marker in ("tang lieu", "giam lieu", "chon lieu")):
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


def _occurrence_is_active(tokens: list[str], start: int, end: int) -> bool:
    """Đánh giá state trong cửa sổ của đúng concept occurrence."""

    prefix_start = max(0, start - 10)
    direct_prefix = tokens[max(0, start - 5) : start]
    direct_prefix_text = " ".join(direct_prefix)
    if NEGATION_TOKENS.intersection(direct_prefix) or any(
        contraction in direct_prefix_text for contraction in ("don t", "doesn t")
    ):
        return False

    current_end = _latest_marker_end(
        tokens,
        CURRENT_STATE_MARKERS,
        start=prefix_start,
        stop=start,
    )
    hypothetical_end = _latest_marker_end(
        tokens,
        HYPOTHETICAL_MARKERS,
        start=prefix_start,
        stop=start,
    )
    if hypothetical_end > current_end:
        return False

    historical_end = _latest_marker_end(
        tokens,
        (*HISTORICAL_STATE_MARKERS, *RESOLUTION_PREFIXES, *RESOLUTION_MARKERS),
        start=prefix_start,
        stop=start,
    )
    if historical_end > current_end:
        return False

    suffix_stop = min(len(tokens), end + 8)
    if _first_marker_start(
        tokens,
        RESOLVED_AFTER_MARKERS,
        start=end,
        stop=suffix_stop,
    ) >= 0:
        return False
    if current_end < 0 and _first_marker_start(
        tokens,
        HISTORICAL_AFTER_MARKERS,
        start=end,
        stop=suffix_stop,
    ) >= 0:
        return False
    return True


def _latest_marker_end(
    tokens: list[str],
    markers: Iterable[str],
    *,
    start: int,
    stop: int,
) -> int:
    ends = [
        span_end
        for marker in markers
        for _, span_end in _phrase_spans(tokens, normalize_text(marker).split())
        if start < span_end <= stop
    ]
    return max(ends, default=-1)


def _first_marker_start(
    tokens: list[str],
    markers: Iterable[str],
    *,
    start: int,
    stop: int,
) -> int:
    starts = [
        span_start
        for marker in markers
        for span_start, span_end in _phrase_spans(tokens, normalize_text(marker).split())
        if start <= span_start and span_end <= stop
    ]
    return min(starts, default=-1)


def _is_possessive_first_person(tokens: list[str], index: int) -> bool:
    if index > 0 and tokens[index - 1] in THIRD_PERSON_SUBJECTS:
        return True
    return (
        index > 1
        and tokens[index - 1] == "cua"
        and tokens[index - 2] in THIRD_PERSON_SUBJECTS
    )


def _span_has_first_person_owner(tokens: list[str], concept_start: int) -> bool:
    return _nearest_subject_owner(tokens, concept_start) is True


def _nearest_subject_owner(tokens: list[str], concept_start: int) -> bool | None:
    window_start = max(0, concept_start - 18)
    clause_boundaries = [
        index
        for index in range(window_start, concept_start)
        if tokens[index] in {CLAUSE_BOUNDARY, SOFT_CLAUSE_BOUNDARY}
    ]
    if clause_boundaries:
        window_start = max(clause_boundaries) + 1
    owners: list[tuple[int, bool]] = []
    for index in range(window_start, concept_start):
        token = tokens[index]
        if (
            token in FIRST_PERSON
            and not _is_possessive_first_person(tokens, index)
            and not _is_reported_first_person(tokens, index, window_start=window_start)
        ):
            owners.append((index, True))
        elif token in THIRD_PERSON_SUBJECTS:
            owners.append((index, False))
    return max(owners, key=lambda owner: owner[0])[1] if owners else None


def _is_reported_first_person(
    tokens: list[str],
    index: int,
    *,
    window_start: int,
) -> bool:
    report_positions = [
        position
        for position in range(max(window_start, index - 5), index)
        if tokens[position] in REPORTING_VERBS
    ]
    if not report_positions:
        return False
    report_position = max(report_positions)
    if tokens[report_position] == "noi" and QUOTE_BOUNDARY not in tokens[report_position:index]:
        return False
    return any(
        tokens[position] in THIRD_PERSON_SUBJECTS
        for position in range(max(window_start, report_position - 4), report_position)
    )


@lru_cache(maxsize=1)
def _canonical_medication_phrases() -> tuple[tuple[str, ...], ...]:
    """Lấy identity thuốc từ taxonomy owner hiện có, không duy trì allowlist thứ hai."""

    normalizer = DrugEntityNormalizer()
    supported_types = {"drug_product", "active_ingredient", "drug_class"}
    phrases = {
        tuple(normalize_text(alias).split())
        for alias, cards in normalizer.alias_index.items()
        if alias
        and any(card.entity_type in supported_types for card in cards)
        and normalize_text(alias)
    }
    phrases.add(("thuoc",))
    return tuple(sorted(phrases, key=lambda phrase: (-len(phrase), phrase)))


def _medication_spans(tokens: list[str]) -> list[tuple[int, int]]:
    return sorted(
        {
            span
            for phrase in _canonical_medication_phrases()
            for span in _phrase_spans(tokens, list(phrase))
        }
    )


def _medication_event_spans(tokens: list[str]) -> list[tuple[int, int]]:
    action_spans = [
        (normalize_text(action), *span)
        for action in MEDICATION_ACTIONS
        for span in _phrase_spans(tokens, normalize_text(action).split())
    ]
    events: set[tuple[int, int]] = set()
    for action, action_start, action_end in action_spans:
        if _medication_action_is_nonactual(tokens, action_start):
            continue
        if _nearest_subject_owner(tokens, action_start) is False:
            continue
        for medication_start, medication_end in _medication_spans(tokens):
            gap = max(action_start, medication_start) - min(action_end, medication_end)
            bridge = (
                tokens[action_end:medication_start]
                if action_end <= medication_start
                else tokens[medication_end:action_start]
            )
            if action == "dung" and any(
                token in {"canh", "gan", "ben", "truoc", "doi"} for token in bridge
            ):
                continue
            max_gap = 4
            if max(0, gap) <= max_gap:
                events.add(
                    (min(action_start, medication_start), max(action_end, medication_end))
                )
    return sorted(events)


def _medication_action_is_nonactual(tokens: list[str], action_start: int) -> bool:
    prefix_start = max(0, action_start - 5)
    prefix = tokens[prefix_start:action_start]
    if NEGATION_TOKENS.intersection(prefix):
        return True
    return _latest_marker_end(
        tokens,
        (*HYPOTHETICAL_MARKERS, *PLANNED_EVENT_MARKERS),
        start=prefix_start,
        stop=action_start,
    ) >= 0


def _has_medication_identity(text: str) -> bool:
    return bool(_medication_spans(normalize_text(text).split()))


def _has_medication_symptom_relation(
    between_tokens: list[str],
    *,
    context_tokens: list[str],
    event_prefix_tokens: list[str],
    event_before_symptom: bool,
) -> bool:
    between = " ".join(between_tokens)
    context = " ".join(context_tokens)
    event_prefix = " ".join(event_prefix_tokens)
    event_prefix_relation = any(
        event_prefix.endswith(marker) for marker in ("sau khi", "sau luc", "tu sau")
    )
    if event_before_symptom:
        has_ordered_relation = event_prefix_relation or any(
            marker in between
            for marker in ("xong", "thi", "roi", "sau do", "bat dau", "den gio van")
        )
    else:
        has_ordered_relation = any(
            marker in between
            for marker in ("sau khi", "sau luc", "tu sau", "bat dau")
        )
    has_historical_event = any(
        marker in context for marker in ("hom qua", "toi qua", "truoc day", "tung")
    )
    if has_historical_event and not has_ordered_relation:
        return False
    if CLAUSE_BOUNDARY in between_tokens and not has_ordered_relation:
        return False
    if event_before_symptom and "hom nay" in between and not has_ordered_relation:
        return False

    immediate_or_current = any(
        marker in context
        for marker in ("vua", "moi")
    )
    if not event_before_symptom:
        return has_ordered_relation
    return has_ordered_relation or immediate_or_current


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
    "has_medication_related_active_symptom",
    "has_past_or_resolved_symptom",
    "has_unnegated_concept",
    "is_comparison_intent",
    "is_medication_management_intent",
    "is_medication_use_event",
    "is_prescription_execution_request",
    "normalize_text",
]
