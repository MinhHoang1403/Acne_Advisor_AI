"""Loại artifact chỉ khi pattern deterministic chứng minh được loại nội dung.

Filter bỏ text rỗng, page-number-only, một số standalone artifact chính xác và
dot leaders trong mục lục. Text ngắn không bị loại chỉ vì ngắn; module không dùng
LLM và không đánh giá medical quality. Exact deduplication giữ lần xuất hiện đầu.
Muốn đổi quy tắc lọc bắt đầu tại ``is_noisy_chunk()`` và các pattern trong file.
"""

from __future__ import annotations

import re


FILTER_CONTRACT_ID = "proof_only_artifact_filter"

_PAGE_NUMBER_ONLY_RE = re.compile(r"^(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?$", re.IGNORECASE)
_DOT_LEADER_RE = re.compile(r"^.{1,180}?\.{3,}\s*\d{1,4}\s*$")
_DOTS_ONLY_RE = re.compile(r"^\.{3,}$")
_TOC_HEADERS = frozenset({"contents", "table of contents", "mục lục"})
_EXACT_STANDALONE_ARTIFACTS = frozenset(
    {"references", "back to top", "© all rights reserved"}
)


def is_noisy_chunk(text: str, header: str | None = None) -> tuple[bool, str]:
    """Chỉ reject khi text khớp chính xác một artifact rule đã khai báo."""

    stripped = text.strip()
    if not stripped:
        return True, "empty"
    if _PAGE_NUMBER_ONLY_RE.fullmatch(stripped):
        return True, "page_number_only"
    if stripped.casefold() in _EXACT_STANDALONE_ARTIFACTS:
        return True, "exact_standalone_artifact"
    if (header or "").strip().casefold() in _TOC_HEADERS and all(
        _DOT_LEADER_RE.fullmatch(line.strip()) or _DOTS_ONLY_RE.fullmatch(line.strip())
        for line in stripped.splitlines() if line.strip()
    ):
        return True, "toc_dot_leaders"
    return False, ""


def is_short_medical_safety_statement(text: str, header: str | None = None) -> bool:
    """Helper tương thích: mọi text ngắn không rỗng đều được giữ lại."""

    del header
    return bool(text.strip())


def deduplicate_chunks(texts: list[str]) -> tuple[list[str], list[int]]:
    """Giữ exact normalized chunk đầu tiên và trả index của bản trùng bị bỏ."""

    seen: set[str] = set()
    kept: list[str] = []
    removed: list[int] = []
    for index, text in enumerate(texts):
        key = text.strip()
        if key in seen:
            removed.append(index)
            continue
        seen.add(key)
        kept.append(text)
    return kept, removed


__all__ = [
    "FILTER_CONTRACT_ID",
    "deduplicate_chunks",
    "is_noisy_chunk",
    "is_short_medical_safety_statement",
]
