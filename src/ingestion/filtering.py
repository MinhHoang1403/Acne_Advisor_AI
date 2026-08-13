"""Current deterministic noisy-chunk filters.

These rules are preserved as-is for S3B structural cleanup. They are
``UNSOURCED_HEURISTIC`` behavior and require methodological review in S4A.
"""

from __future__ import annotations

import re


_DOTS_RE = re.compile(r"\.{3,}")
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")
_COPYRIGHT_RE = re.compile(
    r"(?:©|notice of rights|all rights reserved|subject to)",
    re.IGNORECASE,
)
_MEDICAL_RESCUE_RE = re.compile(
    r"(?:"
    r"benzoyl\s*peroxide|retinoid|tretinoin|isotretinoin|adapalene|tazarotene"
    r"|salicylic\s*acid|azelaic\s*acid|clindamycin|erythromycin|doxycycline"
    r"|minocycline|spironolactone|comedogenic|comedone|papule|pustule"
    r"|nodule|cyst|formulation|dosage|mg|topical|oral|cream|gel|lotion"
    r"|mụn|da|viêm|trị|thuốc|kem|bôi|uống"
    r")",
    re.IGNORECASE,
)
_SHORT_SAFETY_ACTION_RE = re.compile(
    r"(?:do\s+not\s+use|avoid(?:\s+use|\s+using)?|contraindicat(?:ed|ion)?|"
    r"stop\s+(?:use|using)|seek\s+(?:medical\s+)?(?:care|help)|"
    r"không\s+(?:dùng|sử\s+dụng)|tránh\s+(?:dùng|sử\s+dụng)|"
    r"chống\s+chỉ\s+định|ngừng\s+(?:dùng|sử\s+dụng)|đi\s+khám)",
    re.IGNORECASE,
)
_SHORT_SAFETY_CONTEXT_RE = re.compile(
    r"(?:pregnan(?:cy|t)?|breastfeed(?:ing)?|lactat(?:ion|ing)?|"
    r"allerg(?:y|ic)|swelling|angioedema|anaphyla|severe\s+(?:reaction|irritation)|"
    r"adverse\s+(?:reaction|effect)|antibiotic|mang\s+thai|thai\s+kỳ|"
    r"cho\s+con\s+bú|dị\s+ứng|sưng|phản\s+ứng\s+nặng|kháng\s+sinh)",
    re.IGNORECASE,
)


def is_short_medical_safety_statement(text: str, header: str | None = None) -> bool:
    """Recognise general short safety evidence without fixed examples."""

    combined = "\n".join(part for part in ((header or "").strip(), text.strip()) if part)
    return bool(
        _SHORT_SAFETY_ACTION_RE.search(combined)
        and _SHORT_SAFETY_CONTEXT_RE.search(combined)
    )


def is_noisy_chunk(text: str, header: str | None = None) -> tuple[bool, str]:
    """Return the existing noisy-chunk decision and diagnostic reason."""

    stripped = text.strip()
    text_len = len(stripped)
    hdr = (header or "").strip()

    dots_chars = sum(len(match.group()) for match in _DOTS_RE.finditer(stripped))
    if text_len > 0 and dots_chars / text_len > 0.40:
        return True, f"mostly_dots ({dots_chars}/{text_len} chars are dots)"

    if hdr.lower() in {"contents", "table of contents", "mục lục"} and dots_chars > 10:
        return True, f"toc_header '{hdr}' with dot-leaders"

    if _COPYRIGHT_RE.search(stripped) and text_len < 300:
        return True, f"copyright_notice (len={text_len})"

    non_empty_lines = [line for line in stripped.split("\n") if line.strip()]
    if non_empty_lines:
        page_num_lines = sum(1 for line in non_empty_lines if _PAGE_NUM_RE.match(line))
        if page_num_lines / len(non_empty_lines) > 0.5:
            return True, f"page_numbers ({page_num_lines}/{len(non_empty_lines)} lines)"

    if text_len < 80:
        if (
            _MEDICAL_RESCUE_RE.search(stripped)
            or _MEDICAL_RESCUE_RE.search(hdr)
            or is_short_medical_safety_statement(stripped, hdr)
        ):
            return False, ""
        return True, f"too_short (len={text_len})"

    return False, ""


__all__ = ["is_noisy_chunk", "is_short_medical_safety_statement"]
