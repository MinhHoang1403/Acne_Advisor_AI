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

from collections import OrderedDict
import asyncio
import inspect
import logging
import os
from pathlib import Path
import threading
from typing import Any

from src.resilience.budget import DeadlineBudget
from src.resilience.contracts import runtime_resilience_settings_from_env
from src.resilience.provider import call_provider_with_resilience
from src.resilience.retry import RetryPolicy
from src.ingestion.bm25 import BM25_VECTOR_NAME, active_bm25_document
from src.ingestion.embedding import active_query_embedding_text

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

def _embed_sync(text: str, *, request_timeout: float | None = None) -> list[float]:
    """Gửi một query tới Gemini embedding provider theo lời gọi đồng bộ.

    Gemini Embedding 2 không nhận ``task_type`` qua API. Document và query hiện
    dùng chung model/index contract, không thêm retrieval-specific instruction
    vào text; đây là cấu hình hiện tại chứ không phải lựa chọn đã chứng minh tối
    ưu cho corpus.
    """
    from src.integrations.google_genai import embed_texts_sync

    # Cho phép đọc động từ biến môi trường nếu module được import trước khi key được nạp.
    key = GOOGLE_API_KEY.strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to .env for embedding."
        )

    vectors = embed_texts_sync(
        [active_query_embedding_text(text)],
        model_name=EMBEDDING_MODEL,
        task_type=None,
        expected_dimensions=EMBEDDING_DIMENSIONS,
        output_dimensions=EMBEDDING_DIMENSIONS,
        api_key=key,
        request_timeout=request_timeout,
    )
    return vectors[0]


_ORIGINAL_EMBED_SYNC = _embed_sync
_QUERY_EMBEDDING_CACHE: OrderedDict[tuple[str, int, str], list[float]] = OrderedDict()
_QUERY_EMBEDDING_CACHE_LOCK = threading.Lock()
_QUERY_EMBEDDING_CACHE_MAX = 256


def clear_query_embedding_cache() -> None:
    """Xóa bộ nhớ đệm embedding câu hỏi trong tiến trình (dùng khi test hoặc reset)."""

    with _QUERY_EMBEDDING_CACHE_LOCK:
        _QUERY_EMBEDDING_CACHE.clear()


async def embed_query(text: str) -> list[float]:
    """Nhận Dense query vector từ Gemini mà không chặn event loop.

    Áp dụng bộ nhớ đệm LRU giới hạn kích thước (256 mục) cho các truy vấn lặp lại
    hoặc chuẩn hóa trùng khớp, tự động bỏ qua cache khi unit test monkeypatch ``_embed_sync``.
    """
    clean_text = " ".join((text or "").split())
    use_cache = (
        _embed_sync is _ORIGINAL_EMBED_SYNC
        and os.getenv("EMBEDDING_QUERY_CACHE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    )
    active_model = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
    active_dims = _positive_int_env("EMBEDDING_DIMENSIONS", EMBEDDING_DIMENSIONS)
    cache_key = (
        active_model,
        active_dims,
        active_query_embedding_text(clean_text) if clean_text else "",
    )
    if use_cache and clean_text:
        with _QUERY_EMBEDDING_CACHE_LOCK:
            cached_vec = _QUERY_EMBEDDING_CACHE.get(cache_key)
            if cached_vec is not None:
                _QUERY_EMBEDDING_CACHE.move_to_end(cache_key)
                return list(cached_vec)

    settings = runtime_resilience_settings_from_env()
    max_attempts = _positive_int_env("EMBEDDING_QUERY_MAX_ATTEMPTS", 3)
    retry_base = _positive_float_env("EMBEDDING_QUERY_RETRY_BASE_DELAY", 1.0)
    target_text = clean_text or text

    async def operation(effective_timeout: float) -> list[float]:
        # Chỉ truyền request_timeout nếu callable hỗ trợ (tránh vỡ test stub 1 tham số).
        params = inspect.signature(_embed_sync).parameters
        if "request_timeout" in params or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        ):
            embedding = await asyncio.to_thread(
                _embed_sync, target_text, request_timeout=effective_timeout
            )
        else:
            embedding = await asyncio.to_thread(_embed_sync, target_text)
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
    if use_cache and clean_text:
        with _QUERY_EMBEDDING_CACHE_LOCK:
            _QUERY_EMBEDDING_CACHE[cache_key] = list(embedding)
            _QUERY_EMBEDDING_CACHE.move_to_end(cache_key)
            while len(_QUERY_EMBEDDING_CACHE) > _QUERY_EMBEDDING_CACHE_MAX:
                _QUERY_EMBEDDING_CACHE.popitem(last=False)
    return list(embedding)


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
            query=active_bm25_document(text),
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
