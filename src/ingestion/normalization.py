"""Conservative text normalization that never rewrites medical meaning."""

from __future__ import annotations

import re
import unicodedata


NORMALIZATION_CONTRACT_ID = "unicode_nfc_lf_confirmed_nice_artifacts"

_EXACT_FOOTER_PATTERNS = (
    re.compile(r"^Acne vulgaris: management \(NG198\)$", re.IGNORECASE),
    re.compile(r"^© NICE 20\d{2}\. All rights reserved\..*$", re.IGNORECASE),
)
_PAGE_MARKER_RE = re.compile(
    r"^Page\s+\d{1,4}\s+of\s+\d{1,4}(?P<continuation>\s+.*)?$",
    re.IGNORECASE,
)
_PDF_LIGATURES = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "ft",
        "\ufb06": "st",
        "\u00a0": " ",
        "\u2007": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\u00ad": "",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
    }
)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_parsed_text(text: str) -> str:
    """Chuẩn hóa Unicode/newline/spacing và bỏ ký tự ẩn hoặc PDF footer lặp chính xác."""

    # Tách ligature PDF (ﬁ, ﬂ...), thay khoảng trắng đặc biệt thành space chuẩn,
    # bỏ BOM/zero-width/soft-hyphen và ký tự điều khiển C0 trước khi gom NFC.
    cleaned = (text or "").translate(_PDF_LIGATURES)
    cleaned = _CONTROL_CHARS_RE.sub("", cleaned)
    normalized = unicodedata.normalize("NFC", cleaned)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = raw_line.rstrip(" \t")
        stripped = line.strip()
        quote_prefix = ""
        artifact_text = stripped
        if artifact_text.startswith(">"):
            quote_prefix = "> "
            artifact_text = artifact_text[1:].lstrip()
        if any(pattern.fullmatch(artifact_text) for pattern in _EXACT_FOOTER_PATTERNS):
            continue
        page_marker = _PAGE_MARKER_RE.fullmatch(artifact_text)
        if page_marker:
            continuation = str(page_marker.group("continuation") or "").strip()
            if continuation:
                lines.append(f"{quote_prefix}{continuation}".rstrip())
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


__all__ = ["NORMALIZATION_CONTRACT_ID", "normalize_parsed_text"]
