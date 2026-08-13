"""Pure text-splitting primitives for the current Phase 1 pipeline.

Chunk-size and overlap methodology remain unchanged and require S4A review.
"""

from __future__ import annotations


def naive_split(text: str, size: int, overlap: int) -> list[str]:
    """Split text with the existing fixed-width overlap behavior."""

    parts: list[str] = []
    start = 0
    step = max(1, size - overlap)

    while start < len(text):
        parts.append(text[start:start + size])
        start += step

    return parts


__all__ = ["naive_split"]
