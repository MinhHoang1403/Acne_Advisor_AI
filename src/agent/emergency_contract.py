"""Shared emergency wording contract for deterministic answer paths."""

from __future__ import annotations

import re
import unicodedata


IMMEDIATE_EMERGENCY_ACTION_TERMS = (
    "gọi cấp cứu",
    "đến cơ sở cấp cứu",
    "đi cấp cứu",
    "cấp cứu ngay",
    "liên hệ cấp cứu ngay",
)

WEAK_EMERGENCY_ACTION_TERMS = (
    "có thể cần",
    "nên cân nhắc",
    "theo dõi thêm",
    "24-48 giờ",
    "24 48 giờ",
)


def build_generic_emergency_answer() -> str:
    """Return the generic emergency answer with immediate action first."""

    return (
        "**Tóm tắt ngắn**\n"
        "Bạn cần gọi cấp cứu hoặc đến cơ sở cấp cứu ngay; đây có thể là tình huống khẩn cấp nếu các triệu chứng này đang xảy ra hoặc tăng nhanh. "
        "Các dấu hiệu cần chú ý gồm khó thở, sưng môi/mặt/họng/quanh mắt, phát ban lan nhanh, "
        "sốt cao, đau dữ dội, phồng rộp hoặc bong tróc da.\n\n"
        "**Việc nên làm ngay**\n"
        "Ngưng thuốc/sản phẩm nghi gây phản ứng nếu vừa dùng, không bôi hoặc thử thêm hoạt chất mới, "
        "và mang theo thuốc/sản phẩm đã dùng khi đi khám.\n\n"
        "**Lưu ý**\n"
        "Tôi không thể chẩn đoán chắc chắn qua chat. Thông tin này chỉ nhằm định hướng an toàn và không thay thế cấp cứu hoặc khám trực tiếp tại cơ sở y tế."
    )


def build_anaphylaxis_like_emergency_answer() -> str:
    """Return an emergency answer for breathing difficulty plus swelling/rash."""

    return (
        "**Tóm tắt ngắn**\n"
        "Bạn cần gọi cấp cứu hoặc đến cơ sở cấp cứu ngay; đây có thể là tình huống khẩn cấp vì khó thở kèm sưng quanh mắt/môi/mặt/họng sau khi dùng thuốc hoặc sản phẩm trị mụn có thể là phản ứng dị ứng nghiêm trọng.\n\n"
        "**Việc nên làm ngay**\n"
        "- Ngưng thuốc/sản phẩm nghi gây phản ứng nếu vừa dùng.\n"
        "- Không bôi thêm hoạt chất trị mụn, không thử lại sản phẩm và không chờ theo dõi tại nhà nếu khó thở còn tiếp diễn.\n"
        "- Mang theo tên thuốc/sản phẩm, thời điểm dùng và thời điểm bắt đầu sưng/khó thở khi đi cấp cứu.\n\n"
        "**Lưu ý**\n"
        "Tôi không thể xác định nguyên nhân qua chat, nhưng khó thở kèm sưng vùng mắt/môi/mặt/họng là nhóm dấu hiệu cần xử trí cấp cứu tại cơ sở y tế trước khi bàn tiếp về routine trị mụn."
    )


def is_anaphylaxis_like_emergency_query(query: str) -> bool:
    """Detect reusable anaphylaxis-like emergency intent without exact-query matching."""

    text = _accentless(query)
    return _has_breathing_difficulty(text) and (_has_swelling_near_airway_or_eye(text) or _has_generalized_rash(text))


def first_sentence_has_immediate_emergency_action(answer: str) -> bool:
    """Return True when the first answer sentence includes direct emergency action."""

    first = _first_sentence(_accentless(answer))
    return any(_accentless(term) in first for term in IMMEDIATE_EMERGENCY_ACTION_TERMS)


def first_sentence_has_weak_emergency_action(answer: str) -> bool:
    """Return True when the first answer sentence uses weak emergency wording."""

    first = _first_sentence(_accentless(answer))
    return any(_accentless(term) in first for term in WEAK_EMERGENCY_ACTION_TERMS)


def _has_breathing_difficulty(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "kho tho",
            "bat dau kho tho",
            "tho gap",
            "kho khe",
            "nghen tho",
            "hut hoi",
            "kho hit tho",
            "kho hit vao",
            "kho tho ra",
            "tho kho khe",
            "wheezing",
            "shortness of breath",
            "difficulty breathing",
        )
    )


def _has_generalized_rash(text: str) -> bool:
    return any(marker in text for marker in ("me day", "phat ban", "noi man", "ban do lan nhanh"))


def _has_swelling_near_airway_or_eye(text: str) -> bool:
    if any(marker in text for marker in ("nghen hong", "nghet hong", "that hong", "co that hong", "kho nuot")):
        return True

    tokens = re.findall(r"[a-z0-9]+", text)
    swelling_indices = [
        index
        for index, token in enumerate(tokens)
        if token in {"sung", "phu"} or token.startswith("sung") or token.startswith("phu")
    ]
    regions = {"mat", "mi", "moi", "mieng", "luoi", "hong", "co"}
    for swelling_index in swelling_indices:
        start = max(0, swelling_index - 4)
        end = min(len(tokens), swelling_index + 5)
        if any(token in regions for token in tokens[start:end]):
            return True
    return False


def _first_sentence(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return ""
    match = re.search(r"[.!?]", normalized)
    return normalized[: match.start() + 1] if match else normalized


def _accentless(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    stripped = unicodedata.normalize("NFC", stripped)
    stripped = re.sub(r"[*_`()[\]{}\\/]+", " ", stripped.lower())
    return re.sub(r"\s+", " ", stripped).strip()


__all__ = [
    "IMMEDIATE_EMERGENCY_ACTION_TERMS",
    "WEAK_EMERGENCY_ACTION_TERMS",
    "build_anaphylaxis_like_emergency_answer",
    "build_generic_emergency_answer",
    "first_sentence_has_immediate_emergency_action",
    "first_sentence_has_weak_emergency_action",
    "is_anaphylaxis_like_emergency_query",
]
