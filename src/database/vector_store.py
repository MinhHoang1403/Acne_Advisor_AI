"""
src/database/vector_store.py – Vector Store Abstraction
=======================================================
Provides a unified interface for Qdrant and pgvector backends.
The active backend is selected via the VECTOR_DB_PROVIDER env var.

Pha 2 updates
-------------
- Fixed named vector support: Qdrant collection uses "dense" + "bm25"
- Added embed_query() for Gemini query embedding (task_type=retrieval_query)
- Added compute_sparse_vector() replicating Pha 1 BM25 hashing exactly
- Added search_sparse() for BM25 lexical search
- Added close() method for cleanup
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import logging
import math
import os
import re
import ssl
from collections import Counter
from pathlib import Path
from typing import Any

from src.resilience.exceptions import (
    PermanentProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

try:
    from dotenv import load_dotenv

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (from .env)
# ---------------------------------------------------------------------------
VECTOR_DB_PROVIDER = os.getenv("VECTOR_DB_PROVIDER", "qdrant").lower()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")


def qdrant_client_kwargs() -> dict[str, Any]:
    """Return AsyncQdrantClient kwargs without breaking unauthenticated local Qdrant."""
    kwargs: dict[str, Any] = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    return kwargs


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _embed_sync(text: str) -> list[float]:
    """Synchronous Gemini embedding call for a single query string.

    Uses task_type="retrieval_query" (vs. "retrieval_document" at ingestion).
    """
    from src.integrations.google_genai import embed_texts_sync

    key = GOOGLE_API_KEY.strip()
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to .env for embedding."
        )

    vectors = embed_texts_sync(
        [text],
        model_name=EMBEDDING_MODEL,
        task_type="retrieval_query",
        expected_dimensions=EMBEDDING_DIMENSIONS,
        api_key=key,
    )
    return vectors[0]


async def embed_query(text: str) -> list[float]:
    """Embed a query string asynchronously using Gemini.

    Transient transport failures are retried locally because an unavailable
    query embedding would otherwise force a safe fallback despite an intact
    knowledge base. Permanent provider/configuration errors are never retried.

    Returns a dense vector of EMBEDDING_DIMENSIONS floats.
    """
    max_attempts = _positive_int_env("EMBEDDING_QUERY_MAX_ATTEMPTS", 3)
    retry_base_delay = _positive_float_env("EMBEDDING_QUERY_RETRY_BASE_DELAY", 1.0)
    for attempt in range(1, max_attempts + 1):
        try:
            embedding = await asyncio.to_thread(_embed_sync, text)
            if len(embedding) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Query embedding dimension mismatch: got {len(embedding)}, "
                    f"expected {EMBEDDING_DIMENSIONS}. Check EMBEDDING_MODEL and "
                    "EMBEDDING_DIMENSIONS."
                )
            return embedding
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if attempt >= max_attempts or not _is_retryable_query_embedding_error(exc):
                raise
            delay_seconds = retry_base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Retrying query embedding after retryable transport failure: attempt=%d/%d error_type=%s delay_seconds=%.2f",
                attempt,
                max_attempts,
                exc.__class__.__name__,
                delay_seconds,
            )
            await asyncio.sleep(delay_seconds)
    raise RuntimeError("Query embedding retry loop exited unexpectedly.")


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _positive_float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _is_retryable_query_embedding_error(exc: Exception) -> bool:
    """Return whether retrying one live query embedding is safe and useful."""

    if isinstance(exc, PermanentProviderError):
        return False
    if isinstance(exc, (ProviderTimeoutError, ProviderUnavailableError, ssl.SSLError, OSError)):
        return True
    try:
        import httpx
    except ImportError:
        return False
    return isinstance(exc, httpx.TransportError)


# ---------------------------------------------------------------------------
# Sparse BM25 vector helpers
# ---------------------------------------------------------------------------
# These functions MUST produce identical output to the functions in
# scripts/ingest_knowledge.py so that query sparse vectors are compatible
# with the document sparse vectors stored in Qdrant.
# ---------------------------------------------------------------------------

def tokenize_for_sparse(text: str) -> list[str]:
    """Tokenize text for sparse vector – matches ingest_knowledge.py."""
    return re.findall(
        r"[a-zA-ZÀ-ỹ0-9][a-zA-ZÀ-ỹ0-9_\-/.%]*",
        text.lower(),
    )


def token_to_sparse_index(token: str) -> int:
    """Hash token to sparse index – matches ingest_knowledge.py."""
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def compute_sparse_vector(text: str) -> dict[str, list]:
    """Compute hashed sparse BM25 vector for a single text.

    Algorithm is identical to ingest_knowledge.py:compute_hashed_sparse_vectors()
    to ensure query vectors are compatible with indexed document vectors.

    Steps:
    1. Tokenize (Unicode-aware, lowercased)
    2. Count term frequencies
    3. Hash each token to a stable sparse index via MD5
    4. Log-scaled TF normalisation
    """
    tokens = tokenize_for_sparse(text)

    if not tokens:
        return {"indices": [], "values": []}

    counts = Counter(tokens)
    max_tf = max(counts.values()) if counts else 1

    index_to_value: dict[int, float] = {}

    for token, count in counts.items():
        idx = token_to_sparse_index(token)
        tf = 1.0 + math.log(float(count))
        value = tf / (1.0 + math.log(float(max_tf)))
        index_to_value[idx] = index_to_value.get(idx, 0.0) + float(value)

    sorted_items = sorted(index_to_value.items())

    return {
        "indices": [idx for idx, _ in sorted_items],
        "values": [val for _, val in sorted_items],
    }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VectorStore(abc.ABC):
    """Abstract vector store interface."""

    @abc.abstractmethod
    async def upsert(self, id: str, vector: list[float], payload: dict) -> None: ...

    @abc.abstractmethod
    async def search(
        self, query_vector: list[float], top_k: int = 5, filter: dict | None = None
    ) -> list[dict[str, Any]]: ...

    @abc.abstractmethod
    async def delete(self, id: str) -> None: ...


# ---------------------------------------------------------------------------
# Qdrant implementation
# ---------------------------------------------------------------------------

class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store with named vector support.

    The Pha 1 collection uses:
    - Named dense vector: ``"dense"`` (3072-dim, COSINE)
    - Named sparse vector: ``"bm25"`` (hashed sparse BM25)
    """

    def __init__(self) -> None:
        from qdrant_client import AsyncQdrantClient  # type: ignore[import]

        self._client = AsyncQdrantClient(**qdrant_client_kwargs())
        self._collection = QDRANT_COLLECTION_NAME

    async def upsert(self, id: str, vector: list[float], payload: dict) -> None:
        """Upsert a point with named dense vector and sparse BM25 when text exists."""
        from qdrant_client.models import PointStruct, SparseVector  # type: ignore[import]

        text = str(
            payload.get("text")
            or payload.get("content")
            or payload.get("page_content")
            or ""
        )
        vectors: dict[str, Any] = {"dense": vector}
        sparse = compute_sparse_vector(text)
        if sparse["indices"]:
            vectors["bm25"] = SparseVector(
                indices=sparse["indices"],
                values=sparse["values"],
            )
        else:
            logger.warning(
                "Upserting Qdrant point %s without bm25 sparse vector because payload text is empty.",
                id,
            )

        await self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(
                id=id,
                vector=vectors,
                payload=payload,
            )],
        )

    async def search(
        self, query_vector: list[float], top_k: int = 5, filter: dict | None = None
    ) -> list[dict[str, Any]]:
        """Semantic search using named dense vector ``"dense"``."""
        query_filter = None
        if filter is not None:
            from qdrant_client import models  # type: ignore[import]

            query_filter = (
                filter
                if isinstance(filter, models.Filter)
                else (
                    models.Filter.model_validate(filter)
                    if hasattr(models.Filter, "model_validate")
                    else models.Filter.parse_obj(filter)
                )
            )
        response = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            using="dense",
            limit=top_k,
            query_filter=query_filter,
        )
        return [{"id": r.id, "score": r.score, **(r.payload or {})} for r in response.points]

    async def search_sparse(
        self, text: str, top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Lexical search using named sparse BM25 vector ``"bm25"``.

        Computes a hashed sparse vector from *text* using the same algorithm
        as Pha 1 ingestion, then queries Qdrant's sparse index.
        """
        from qdrant_client import models  # type: ignore[import]

        sparse = compute_sparse_vector(text)

        if not sparse["indices"]:
            logger.warning("Empty sparse vector for query, returning empty results.")
            return []

        response = await self._client.query_points(
            collection_name=self._collection,
            query=models.SparseVector(
                indices=sparse["indices"],
                values=sparse["values"],
            ),
            using="bm25",
            limit=top_k,
        )
        return [{"id": r.id, "score": r.score, **(r.payload or {})} for r in response.points]

    async def delete(self, id: str) -> None:
        from qdrant_client.models import PointIdsList  # type: ignore[import]

        await self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=[id]),
        )

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        await self._client.close()


# ---------------------------------------------------------------------------
# pgvector implementation (placeholder)
# ---------------------------------------------------------------------------

class PgVectorStore(VectorStore):
    """pgvector-backed vector store (placeholder)."""

    async def upsert(self, id: str, vector: list[float], payload: dict) -> None:
        raise NotImplementedError("pgvector store not yet implemented.")

    async def search(
        self, query_vector: list[float], top_k: int = 5, filter: dict | None = None
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("pgvector store not yet implemented.")

    async def delete(self, id: str) -> None:
        raise NotImplementedError("pgvector store not yet implemented.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_vector_store() -> VectorStore:
    """Factory – returns the configured vector store instance."""
    if VECTOR_DB_PROVIDER == "qdrant":
        return QdrantVectorStore()
    return PgVectorStore()
