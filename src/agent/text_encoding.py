"""Phát hiện và sửa một tập mojibake UTF-8 phổ biến.

Heuristic đếm số marker trước/sau mỗi phép latin1/cp1252 -> UTF-8 conversion và
chỉ nhận bản sửa khi marker count giảm. Count này chỉ chọn chuỗi ít dấu hiệu lỗi
hơn, không phải quality/confidence score. Nếu round-trip không được, module dùng
một mapping nhỏ đã biết thay vì ``errors=ignore`` làm mất ký tự âm thầm.
"""

from __future__ import annotations

MOJIBAKE_MARKERS = ("Ã", "Ä", "á»", "áº", "Æ", "Â", "Ð")

LOSSY_MOJIBAKE_REPLACEMENTS = {
    "Äiá»u trá»": "Điều trị",
    "Äiá»u": "Điều",
    "ÄÆ¡n": "đơn",
    "sÄ©": "sĩ",
    "bÃ¡c": "bác",
    "khÃ´ng": "không",
    "kÃª": "kê",
    "liá»…u": "liễu",
}


def looks_like_mojibake(value: str) -> bool:
    """Báo chuỗi có marker UTF-8 bị đọc nhầm latin1/cp1252 hay không."""
    return isinstance(value, str) and any(marker in value for marker in MOJIBAKE_MARKERS)


def repair_mojibake(value: str) -> str:
    """Sửa mojibake khi conversion làm số marker giảm; nếu không giữ nguyên."""
    if not isinstance(value, str) or not value or not looks_like_mojibake(value):
        return value

    original_score = sum(marker in value for marker in MOJIBAKE_MARKERS)
    for encoding in ("latin1", "cp1252"):
        try:
            repaired = value.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        if repaired and sum(marker in repaired for marker in MOJIBAKE_MARKERS) < original_score:
            return repaired

    repaired = value
    for broken, fixed in LOSSY_MOJIBAKE_REPLACEMENTS.items():
        repaired = repaired.replace(broken, fixed)
    if repaired != value:
        return repaired

    return value
