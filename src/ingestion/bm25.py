"""Qdrant-native true BM25 contract shared by indexing and querying."""

from __future__ import annotations

import math
from collections import Counter

from qdrant_client import models


BM25_CONTRACT_ID = "qdrant_native_bm25_word_language_none"
BM25_MODEL = "qdrant/bm25"
BM25_VECTOR_NAME = "bm25"
BM25_K1 = 1.2
BM25_B = 0.75
BM25_AVG_LEN = 256.0
BM25_TOKENIZER = models.TokenizerType.WORD
BM25_LANGUAGE = "none"


def bm25_config() -> models.Bm25Config:
    """Return the one immutable document/query preprocessing configuration."""

    return models.Bm25Config(
        k=BM25_K1,
        b=BM25_B,
        avg_len=BM25_AVG_LEN,
        tokenizer=BM25_TOKENIZER,
        language=BM25_LANGUAGE,
        lowercase=True,
        ascii_folding=False,
    )


def bm25_document(text: str) -> models.Document:
    """Create a provider-side BM25 inference document for indexing or search."""

    return models.Document(text=text, model=BM25_MODEL, options=bm25_config())


def bm25_sparse_vector_config() -> models.SparseVectorParams:
    """Require collection-aware IDF at query time."""

    return models.SparseVectorParams(modifier=models.Modifier.IDF)


def reference_bm25_score(
    *,
    query_terms: list[str],
    document_terms: list[str],
    document_frequencies: dict[str, int],
    document_count: int,
    average_document_length: float,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    """Transparent BM25 reference formula used only for hand-calculated tests."""

    if document_count <= 0 or average_document_length <= 0:
        raise ValueError("Corpus size and average document length must be positive")
    frequencies = Counter(document_terms)
    length_ratio = len(document_terms) / average_document_length
    score = 0.0
    for term in query_terms:
        tf = frequencies[term]
        if tf <= 0:
            continue
        df = document_frequencies.get(term, 0)
        idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
        numerator = tf * (k1 + 1.0)
        denominator = tf + k1 * (1.0 - b + b * length_ratio)
        score += idf * numerator / denominator
    return score


__all__ = [
    "BM25_AVG_LEN",
    "BM25_B",
    "BM25_CONTRACT_ID",
    "BM25_K1",
    "BM25_LANGUAGE",
    "BM25_MODEL",
    "BM25_TOKENIZER",
    "BM25_VECTOR_NAME",
    "bm25_config",
    "bm25_document",
    "bm25_sparse_vector_config",
    "reference_bm25_score",
]
