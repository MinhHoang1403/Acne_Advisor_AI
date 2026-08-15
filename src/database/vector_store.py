"""Adapter nối runtime retrieval với Gemini embedding và Qdrant.

Project gửi query text cho Gemini và nhận Dense vector 3072 chiều; project không
tự chạy neural embedding model. Qdrant dùng vector đó để thực thi cosine search.
Với BM25, project gửi ``Document`` cùng contract từ ``src/ingestion/bm25.py``;
Qdrant tạo sparse representation và thực thi search theo collection IDF.

Adapter này chỉ đọc collection kiến thức trong request path. EntityCards và
Neo4j không được truy vấn tại đây. Muốn đổi provider/model embedding hãy đọc
``_embed_sync``/``embed_query``; muốn đổi search call hãy đọc hai method của
``QdrantVectorStore``.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from src.resilience.budget import DeadlineBudget
from src.resilience.contracts import runtime_resilience_settings_from_env
from src.resilience.provider import call_provider_with_resilience
from src.resilience.retry import RetryPolicy
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
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")


def qdrant_client_kwargs() -> dict[str, Any]:
    """Tạo kwargs cho Qdrant, chỉ gửi API key khi cấu hình thực sự có giá trị."""
    kwargs: dict[str, Any] = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    return kwargs


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _embed_sync(text: str) -> list[float]:
    """Gửi một query tới Gemini embedding provider theo lời gọi đồng bộ.

    Gemini Embedding 2 does not accept a task type. Documents and queries share
    the same versioned model and index contract.
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
    """Nhận Dense query vector từ Gemini mà không chặn event loop.

    Transient transport failures are retried locally because an unavailable
    query embedding would otherwise force a safe fallback despite an intact
    knowledge base. Permanent provider/configuration errors are never retried.

    Returns a dense vector of EMBEDDING_DIMENSIONS floats.
    """
    settings = runtime_resilience_settings_from_env()
    max_attempts = _positive_int_env("EMBEDDING_QUERY_MAX_ATTEMPTS", 3)
    retry_base = _positive_float_env("EMBEDDING_QUERY_RETRY_BASE_DELAY", 1.0)

    async def operation(_: float) -> list[float]:
        embedding = await asyncio.to_thread(_embed_sync, text)
        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Query embedding dimension mismatch: got {len(embedding)}, "
                f"expected {EMBEDDING_DIMENSIONS}. Check EMBEDDING_MODEL and EMBEDDING_DIMENSIONS."
            )
        return embedding

    embedding, _ = await call_provider_with_resilience(
        provider_name=f"embedding:{EMBEDDING_MODEL}",
        operation=operation,
        budget=DeadlineBudget.from_timeout(settings.retrieval_timeout_seconds),
        timeout_seconds=settings.retrieval_timeout_seconds,
        retry_policy=RetryPolicy(
            max_retries=max_attempts - 1,
            base_delay_seconds=retry_base,
            max_delay_seconds=max(retry_base, retry_base * 4),
        ),
        sleep=asyncio.sleep,
    )
    return embedding


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


class QdrantVectorStore:
    """Adapter Qdrant read-only dùng chung trong process runtime.

    Collection kiến thức dùng:
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

    async def search(
        self, query_vector: list[float], top_k: int = 5, filter: dict | None = None
    ) -> list[dict[str, Any]]:
        """Yêu cầu Qdrant tìm candidate bằng named Dense vector ``"dense"``.

        Collection khai báo cosine distance; Qdrant, không phải method Python
        này, thực hiện vector search và tính score.
        """
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
        """Yêu cầu Qdrant thực thi BM25 search trên named vector ``"bm25"``."""

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

    async def close(self) -> None:
        """Giữ shared read client sống hết process; method này không đóng nó."""

        return None

    @classmethod
    async def close_shared_client(cls) -> None:
        """Đóng shared client khi application shutdown có kiểm soát."""

        if cls._shared_client is not None:
            await cls._shared_client.close()
            cls._shared_client = None
