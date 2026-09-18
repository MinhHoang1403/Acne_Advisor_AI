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


CHUNK_CONTRACT_ID = "structure_preserving_blocks_and_lead_ins_chars_2400_no_overlap"
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
    lead_ins: dict[int, str] = {}

    def flush_section(
        *,
        defer_as_lead_in: bool = False,
        retain_for_sibling_children: bool = False,
    ) -> str | None:
        nonlocal section_parts
        if (
            defer_as_lead_in
            and len(section_parts) == 1
            and section_parts[0].rstrip().endswith(":")
        ):
            lead_in = section_parts[0].strip()
            if retain_for_sibling_children:
                lead_ins[len(heading_stack)] = lead_in
            section_parts = []
            return lead_in
        if section_parts:
            for part in _split_structured_blocks(section_parts, max_chars=max_chars):
                output.append(StructuralChunk(part, tuple(heading_stack)))
        section_parts = []
        return None

    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        heading_match = _HEADING_RE.match(lines[0].strip())
        if heading_match:
            level = len(heading_match.group(1))
            deferred_lead_in = flush_section(
                defer_as_lead_in=True,
                retain_for_sibling_children=level > len(heading_stack),
            )
            for depth in tuple(lead_ins):
                if depth >= level:
                    lead_ins.pop(depth, None)
            title = heading_match.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(title)
            parent_lead_in = deferred_lead_in or lead_ins.get(level - 1)
            if parent_lead_in:
                section_parts.append(parent_lead_in)
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


def _split_structured_blocks(blocks: list[str], *, max_chars: int) -> list[str]:
    """Pack complete Markdown blocks and keep explicit continuations together."""

    units: list[str] = []
    for raw_block in blocks:
        block = raw_block.strip()
        if not block:
            continue
        if units and _continues_previous(units[-1], block):
            units[-1] = f"{units[-1]}\n\n{block}"
        else:
            units.append(block)

    chunks: list[str] = []
    current = ""
    for unit in units:
        split_units = (
            [unit]
            if len(unit) <= max_chars
            else _split_oversized_structured_unit(unit, max_chars=max_chars)
        )
        for split_unit in split_units:
            candidate = f"{current}\n\n{split_unit}".strip() if current else split_unit
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = split_unit
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def _continues_previous(previous: str, current: str) -> bool:
    previous_text = previous.rstrip()
    current_text = current.lstrip()
    if not previous_text or not current_text:
        return False
    previous_is_list = _is_list_block(previous_text)
    current_is_list = _is_list_block(current_text)
    if previous_is_list and current_is_list:
        return False
    if current_is_list and previous_text.endswith((":", ",")):
        return True
    return previous_text[-1] not in ".!?" and not previous_is_list


def _is_list_block(text: str) -> bool:
    first_line = text.splitlines()[0].lstrip("> ")
    return bool(re.match(r"^(?:[-*•－]|\d+[.)])\s+", first_line))


def _split_oversized_structured_unit(text: str, *, max_chars: int) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines()]
    if len(lines) > 1 and all(
        not line.strip() or line.lstrip().startswith("|") for line in lines
    ):
        return _split_table_rows(lines, max_chars=max_chars)
    return _split_with_boundaries(text, max_chars=max_chars)


def _split_table_rows(lines: list[str], *, max_chars: int) -> list[str]:
    """Split oversized Markdown tables between rows, never through a row."""

    rows = [line.strip() for line in lines if line.strip()]
    chunks: list[str] = []
    current = ""
    for row in rows:
        row_parts = [row] if len(row) <= max_chars else _hard_split(row, max_chars=max_chars)
        for row_part in row_parts:
            candidate = f"{current}\n{row_part}" if current else row_part
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = row_part
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
