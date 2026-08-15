from __future__ import annotations

import math

from qdrant_client import models

from src.ingestion.bm25 import (
    BM25_AVG_LEN,
    BM25_B,
    BM25_K1,
    BM25_LANGUAGE,
    BM25_MODEL,
    bm25_document,
    bm25_sparse_vector_config,
    reference_bm25_score,
)


def test_native_bm25_document_has_explicit_shared_options() -> None:
    document = bm25_document("Benzoyl peroxide trị mụn")

    assert document.model == BM25_MODEL == "qdrant/bm25"
    assert document.options.k == BM25_K1 == 1.2
    assert document.options.b == BM25_B == 0.75
    assert document.options.avg_len == BM25_AVG_LEN == 256.0
    assert document.options.tokenizer == models.TokenizerType.WORD
    assert document.options.language == BM25_LANGUAGE == "none"
    assert document.options.lowercase is True
    assert document.options.ascii_folding is False
    assert bm25_document("query").options == document.options


def test_native_bm25_collection_config_requires_idf() -> None:
    assert bm25_sparse_vector_config().modifier == models.Modifier.IDF


def test_reference_bm25_matches_hand_calculation() -> None:
    score = reference_bm25_score(
        query_terms=["acne"],
        document_terms=["acne", "acne", "care"],
        document_frequencies={"acne": 2},
        document_count=10,
        average_document_length=4.0,
    )
    idf = math.log(1 + (10 - 2 + 0.5) / (2 + 0.5))
    expected = idf * (2 * (1.2 + 1)) / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / 4))
    assert score == expected


def test_reference_bm25_has_tf_saturation_and_length_normalization() -> None:
    one = reference_bm25_score(
        query_terms=["acne"], document_terms=["acne"],
        document_frequencies={"acne": 1}, document_count=10,
        average_document_length=2,
    )
    repeated = reference_bm25_score(
        query_terms=["acne"], document_terms=["acne", "acne"],
        document_frequencies={"acne": 1}, document_count=10,
        average_document_length=2,
    )
    very_long = reference_bm25_score(
        query_terms=["acne"], document_terms=["acne", *("other" for _ in range(20))],
        document_frequencies={"acne": 1}, document_count=10,
        average_document_length=2,
    )
    assert one < repeated < one * 2
    assert very_long < one
