from __future__ import annotations

import pytest

from scripts.ingest_knowledge import compute_hashed_sparse_vectors
from src.database.vector_store import compute_sparse_vector as runtime_sparse_vector
from src.ingestion.sparse_legacy import (
    LEGACY_SPARSE_STORAGE_NAME,
    SPARSE_VECTOR_SCHEMA_VERSION,
    compute_sparse_vector,
    compute_sparse_vectors,
    token_to_sparse_index,
    tokenize_for_sparse,
)


@pytest.mark.parametrize(
    "text",
    [
        "Benzoyl peroxide BP bp",
        "mụn viêm mụn đầu đen",
        "",
        "A/B 10% A/B",
    ],
)
def test_ingestion_and_runtime_sparse_wrappers_preserve_exact_values(text: str) -> None:
    expected = compute_sparse_vector(text)

    assert compute_hashed_sparse_vectors([text])[0] == expected
    assert runtime_sparse_vector(text) == expected


def test_current_sparse_golden_contract_is_unchanged() -> None:
    assert SPARSE_VECTOR_SCHEMA_VERSION == "hashed_bm25_v1"
    assert LEGACY_SPARSE_STORAGE_NAME == "bm25"
    assert tokenize_for_sparse("Benzoyl peroxide BP bp") == [
        "benzoyl",
        "peroxide",
        "bp",
        "bp",
    ]
    assert token_to_sparse_index("bp") == 1560131687
    assert compute_sparse_vector("Benzoyl peroxide BP bp") == {
        "indices": [789117524, 1246384257, 1560131687],
        "values": [0.5906161091496412, 0.5906161091496412, 1.0],
    }


def test_sparse_batch_preserves_order_and_cardinality() -> None:
    texts = ["benzoyl peroxide", "", "adapalene adapalene"]

    assert compute_sparse_vectors(texts) == [compute_sparse_vector(text) for text in texts]
