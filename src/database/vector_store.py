"""
src/database/vector_store.py – Vector Store Abstraction
=======================================================
Provides a unified interface for Qdrant and pgvector backends.
The active backend is selected via the VECTOR_DB_PROVIDER env var.

Pha 2 updates
-------------
- Fixed named vector support: Qdrant collection uses "dense" + "bm25"
- Added embed_query() under the Gemini Embedding 2 no-task-type contract
- Added search_sparse() for Qdrant-native true BM25
- Added close() method for cleanup
"""

from __future__ import annotations

import abc
import asyncio
import logging
import os
import ssl
from pathlib import Path
from typing import Any

from src.resilience.exceptions import (
    PermanentProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from src.ingestion.bm25 import BM25_VECTOR_NAME, bm25_document

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

    Gemini Embedding 2 does not accept a task type. Documents and queries share
    the same frozen model/configuration contract.
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
        task_type=None,
        expected_dimensions=EMBEDDING_DIMENSIONS,
        output_dimensions=EMBEDDING_DIMENSIONS,
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
    - Named sparse vector: ``"bm25"`` (Qdrant-native BM25 with collection IDF)
    """

    _shared_client: Any | None = None

    def __init__(self) -> None:
        from qdrant_client import AsyncQdrantClient  # type: ignore[import]

        if self.__class__._shared_client is None:
            self.__class__._shared_client = AsyncQdrantClient(**qdrant_client_kwargs())
            logger.debug("Created process-level Qdrant client for read operations.")
        self._client = self.__class__._shared_client
        self._collection = QDRANT_COLLECTION_NAME

    async def upsert(self, id: str, vector: list[float], payload: dict) -> None:
        """Upsert dense and native BM25 inference inputs when text exists."""
        from qdrant_client.models import PointStruct  # type: ignore[import]

        text = str(
            payload.get("text")
            or payload.get("content")
            or payload.get("page_content")
            or ""
        )
        vectors: dict[str, Any] = {"dense": vector}
        if text.strip():
            vectors[BM25_VECTOR_NAME] = bm25_document(text)
        else:
            logger.warning(
                "Upserting Qdrant point %s without BM25 input because payload text is empty.",
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
        """Search the Qdrant-native true BM25 channel."""

        if not text.strip():
            logger.warning("Empty BM25 query, returning empty results.")
            return []

        response = await self._client.query_points(
            collection_name=self._collection,
            query=bm25_document(text),
            using=BM25_VECTOR_NAME,
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
        """Keep the shared read client alive for the process lifetime."""

        return None

    @classmethod
    async def close_shared_client(cls) -> None:
        """Close the shared client explicitly during controlled process shutdown."""

        if cls._shared_client is not None:
            await cls._shared_client.close()
            cls._shared_client = None


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
