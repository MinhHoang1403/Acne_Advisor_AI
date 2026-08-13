"""Conservative text normalization that never rewrites medical meaning."""

from __future__ import annotations

import re
import unicodedata


NORMALIZATION_CONTRACT_ID = "unicode_nfc_lf_exact_artifacts"

_EXACT_FOOTER_PATTERNS = (
    re.compile(r"^Acne vulgaris: management \(NG198\)$", re.IGNORECASE),
    re.compile(r"^© NICE 20\d{2}\. All rights reserved\..*$", re.IGNORECASE),
    re.compile(r"^Page \d{1,4} of \d{1,4}$", re.IGNORECASE),
)


def normalize_parsed_text(text: str) -> str:
    """Normalize Unicode/newlines/spacing and remove exact repeated PDF footers."""

    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = raw_line.rstrip(" \t")
        if any(pattern.fullmatch(line.strip()) for pattern in _EXACT_FOOTER_PATTERNS):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


__all__ = ["NORMALIZATION_CONTRACT_ID", "normalize_parsed_text"]
