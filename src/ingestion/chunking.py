"""Chia normalized Markdown thành chunk theo cấu trúc tài liệu.

Thứ tự ưu tiên boundary là heading, paragraph, sentence rồi word boundary khi
một đơn vị vẫn quá dài. ``section_path`` được giữ để provenance biết chunk thuộc
phần nào. Giới hạn 2400 ký tự và overlap 0 là engineering contract của build,
không phải kích thước được tuyên bố tối ưu về mặt khoa học.

Module không embed, không lọc theo semantic quality và không gọi provider.
Muốn đổi boundary hoặc resource limit bắt đầu tại ``structural_chunks()`` và các
hằng số ``CHUNK_*``; thay đổi này làm đổi build contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


CHUNK_CONTRACT_ID = "structure_first_chars_2400_no_overlap"
CHUNK_MAX_CHARS = 2400
CHUNK_OVERLAP_CHARS = 0

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[^\s])")


@dataclass(frozen=True)
class StructuralChunk:
    """Text chunk cùng đường dẫn heading dùng cho provenance."""

    text: str
    section_path: tuple[str, ...]


def structural_chunks(text: str, *, max_chars: int = CHUNK_MAX_CHARS) -> list[StructuralChunk]:
    """Chia Markdown theo heading, paragraph và sentence mà không tạo overlap."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    output: list[StructuralChunk] = []
    heading_stack: list[str] = []
    section_parts: list[str] = []

    def flush_section() -> None:
        nonlocal section_parts
        section_text = "\n\n".join(part for part in section_parts if part.strip()).strip()
        if section_text:
            for part in _split_with_boundaries(section_text, max_chars=max_chars):
                output.append(StructuralChunk(part, tuple(heading_stack)))
        section_parts = []

    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        heading_match = _HEADING_RE.match(lines[0].strip())
        if heading_match:
            flush_section()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(title)
            remainder = "\n".join(lines[1:]).strip()
            if remainder:
                section_parts.append(remainder)
        else:
            section_parts.append(block)
    flush_section()
    return output


def _split_with_boundaries(text: str, *, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        sentences = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(paragraph) if part.strip()]
        for sentence in sentences:
            if len(sentence) <= max_chars:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, max_chars=max_chars))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _hard_split(text: str, *, max_chars: int) -> list[str]:
    """Fallback theo word boundary cho sentence hoặc table row quá dài."""

    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        boundary = remaining.rfind(" ", 0, max_chars + 1)
        if boundary <= 0:
            boundary = max_chars
        parts.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def naive_split(text: str, size: int, overlap: int) -> list[str]:
    """Helper tương thích cho test; build thật dùng ``structural_chunks``."""

    parts: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        parts.append(text[start:start + size])
        start += step
    return parts


__all__ = [
    "CHUNK_CONTRACT_ID",
    "CHUNK_MAX_CHARS",
    "CHUNK_OVERLAP_CHARS",
    "StructuralChunk",
    "naive_split",
    "structural_chunks",
]
