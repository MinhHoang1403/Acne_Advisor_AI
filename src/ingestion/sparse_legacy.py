"""Compatibility sparse vectors used by the currently indexed knowledge base.

This is a custom MD5-indexed normalized log-term-frequency representation.
It is not BM25: there is no corpus IDF, document-length normalization, ``k1``,
or ``b``. Qdrant's historical sparse vector key remains ``bm25`` until the
explicit S4A datastore migration.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter


SPARSE_VECTOR_SCHEMA_VERSION = "hashed_bm25_v1"
LEGACY_SPARSE_STORAGE_NAME = "bm25"


def tokenize_for_sparse(text: str) -> list[str]:
    """Return the exact token sequence used by the existing sparse index."""

    return re.findall(
        r"[a-zA-ZÀ-ỹ0-9][a-zA-ZÀ-ỹ0-9_\-/.%]*",
        text.lower(),
    )


def token_to_sparse_index(token: str) -> int:
    """Map one token to the existing stable 31-bit sparse index."""

    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def compute_sparse_vector(text: str) -> dict[str, list]:
    """Compute one legacy sparse vector without changing indexed semantics."""

    tokens = tokenize_for_sparse(text)
    if not tokens:
        return {"indices": [], "values": []}

    counts = Counter(tokens)
    max_tf = max(counts.values()) if counts else 1
    index_to_value: dict[int, float] = {}

    for token, count in counts.items():
        index = token_to_sparse_index(token)
        term_frequency = 1.0 + math.log(float(count))
        value = term_frequency / (1.0 + math.log(float(max_tf)))
        index_to_value[index] = index_to_value.get(index, 0.0) + float(value)

    sorted_items = sorted(index_to_value.items())
    return {
        "indices": [index for index, _ in sorted_items],
        "values": [value for _, value in sorted_items],
    }


def compute_sparse_vectors(texts: list[str]) -> list[dict[str, list]]:
    """Compute a batch while preserving input order and cardinality."""

    return [compute_sparse_vector(text) for text in texts]


__all__ = [
    "LEGACY_SPARSE_STORAGE_NAME",
    "SPARSE_VECTOR_SCHEMA_VERSION",
    "compute_sparse_vector",
    "compute_sparse_vectors",
    "token_to_sparse_index",
    "tokenize_for_sparse",
]
