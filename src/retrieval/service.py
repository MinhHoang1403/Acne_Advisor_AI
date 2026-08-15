"""Điều phối retrieval evidence bằng Dense, Qdrant-native BM25 và RRF.

Data flow của một lần retrieval:
``query -> Dense và BM25 song song -> RRF -> context packer -> evidence + trace``.
Project gửi query tới Gemini để nhận Dense vector; Qdrant thực thi cosine search
và BM25 search; Python hợp nhất rank và áp resource budget cho context.

Hai channel có timeout độc lập. Lỗi của một channel không làm mất evidence đã
nhận từ channel còn lại. Service không generate answer, không xác minh clinical
truth và không dùng EntityCards hay Neo4j để grounding câu trả lời thông thường.

Điểm thường cần chỉnh sửa:
- RRF defaults: các hằng số ``RRF_*`` trong module này.
- Candidate/context/timeout budgets: các biến môi trường đọc trong ``retrieve``.
- Packing/truncation: ``src/retrieval/context_packer.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.database.vector_store import QdrantVectorStore, embed_query
from src.quality.safe_fallback import sanitize_fallback_reason
from src.retrieval.context_packer import pack_context, packed_context_to_response_contexts
from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate
from src.retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

RRF_K = 60
RRF_DENSE_WEIGHT = 1.0
RRF_BM25_WEIGHT = 1.0
# Đây là engineering parameters cho rank fusion, không phải confidence threshold.


@dataclass
class RetrievalResult:
    """Source-backed retrieval output consumed by the agent and `/retrieve`."""

    vector_contexts: list[dict[str, Any]]
    sources: list[str]
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrieveEvidenceInput(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=8, ge=1, le=20)


class EvidenceRetriever:
    """Owner của request-time Dense + BM25 + RRF evidence retrieval.

    Instance dùng adapter Qdrant để chạy hai search channel, sau đó trả
    ``RetrievalResult`` có packed evidence và trace. Class không generate answer,
    không đánh giá medical truth và không có post-RRF reranker.
    """

    def __init__(self, vector_store: QdrantVectorStore | None = None) -> None:
        self._vector_store = vector_store or QdrantVectorStore()

    async def retrieve(
        self,
        query: str,
        top_k: int = 8,
        dense_weight: float = RRF_DENSE_WEIGHT,
        sparse_weight: float = RRF_BM25_WEIGHT,
        rrf_k: int = RRF_K,
        **_: Any,
    ) -> RetrievalResult:
        """Chạy hai channel độc lập, hợp nhất rank và đóng gói source evidence.

        ``top_k`` giới hạn số item context cuối cùng; nó không phải relevance
        threshold. Empty list là kết quả channel hợp lệ, còn exception/timeout
        được ghi nhận là channel failure để có thể trả degraded evidence.
        """

        started = time.perf_counter()
        clean_query = " ".join(query.split())
        if not clean_query:
            return RetrievalResult([], [], query, {"status": "empty_query"})

        candidate_limit = _bounded_env("RETRIEVAL_CANDIDATE_LIMIT", 16, 1, 50)
        context_items = min(top_k, _bounded_env("RETRIEVAL_CONTEXT_MAX_ITEMS", 8, 1, 20))
        context_chars = _bounded_env("RETRIEVAL_CONTEXT_MAX_CHARS", 6000, 512, 20000)

        timeout_seconds = _bounded_float_env("RETRIEVAL_TIMEOUT_SECONDS", 20.0, 0.1, 120.0)
        # Dense và BM25 độc lập nên chạy đồng thời. Mỗi coroutine được bọc bằng
        # timeout riêng; ``return_exceptions=True`` giữ outcome riêng của từng
        # channel thay vì hủy toàn bộ retrieval khi một phía thất bại.
        dense_task = asyncio.create_task(
            _run_channel_with_timeout(
                self._dense_search(clean_query, candidate_limit),
                timeout_seconds,
            )
        )
        bm25_task = asyncio.create_task(
            _run_channel_with_timeout(
                self._vector_store.search_sparse(clean_query, top_k=candidate_limit),
                timeout_seconds,
            )
        )
        dense_result, bm25_result = await asyncio.gather(
            dense_task,
            bm25_task,
            return_exceptions=True,
        )

        warnings: list[str] = []
        dense_results = _channel_or_warning("dense", dense_result, warnings)
        bm25_results = _channel_or_warning("bm25", bm25_result, warnings)
        if not dense_results and not bm25_results and (isinstance(dense_result, Exception) or isinstance(bm25_result, Exception)):
            errors = "; ".join(warnings) or "No retrieval channel returned evidence."
            raise RuntimeError(errors)

        fused = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            k=rrf_k,
        )
        candidates = [_to_candidate(item, rank) for rank, item in enumerate(fused, 1)]
        normalized = NormalizedQuery(
            original_query=query,
            normalized_text=clean_query,
        )
        packed = pack_context(
            normalized,
            candidates,
            max_items=context_items,
            max_chars=context_chars,
        )
        contexts = packed_context_to_response_contexts(packed)
        sources = list(dict.fromkeys(_source_id(context) for context in contexts if _source_id(context)))
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        dense_failed = isinstance(dense_result, BaseException)
        bm25_failed = isinstance(bm25_result, BaseException)
        # Status phân biệt evidence đầy đủ, degraded theo channel, không có
        # candidate hợp lệ, và lỗi có thể retry khi không channel nào sống sót.
        if contexts and dense_failed:
            retrieval_status = "degraded_dense"
        elif contexts and bm25_failed:
            retrieval_status = "degraded_bm25"
        elif contexts:
            retrieval_status = "ok"
        elif dense_failed or bm25_failed:
            retrieval_status = "recoverable_error"
        else:
            retrieval_status = "no_evidence"

        trace = {
            "architecture": "dense_bm25_rrf",
            "status": retrieval_status,
            "query": clean_query,
            "channels": {
                "dense": {"count": len(dense_results), "error": _error_name(dense_result)},
                "bm25": {"count": len(bm25_results), "error": _error_name(bm25_result)},
            },
            "rrf": {
                "k": rrf_k,
                "dense_weight": dense_weight,
                "bm25_weight": sparse_weight,
                "formula": "sum(weight / (k + one_indexed_rank))",
            },
            "fused_candidate_count": len(fused),
            "selected_ids": [item.item_id for item in packed.items],
            "warnings": [*warnings, *packed.warnings],
            "elapsed_ms": elapsed_ms,
        }
        return RetrievalResult(
            vector_contexts=contexts,
            sources=sources,
            query=query,
            metadata={
                "retrieval_trace": trace,
                "packed_context": packed.model_dump(mode="json"),
                "retrieval_status": retrieval_status,
                "timings": {"retrieval_total_ms": elapsed_ms},
            },
        )

    async def _dense_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Nhận vector từ Gemini rồi yêu cầu Qdrant thực thi cosine search."""

        vector = await embed_query(query)
        return await self._vector_store.search(vector, top_k=limit)

    async def close(self) -> None:
        try:
            await self._vector_store.close()
        except Exception as exc:
            logger.warning("Failed to close retrieval vector store: %s", sanitize_fallback_reason(exc))


# Đây là evidence tool duy nhất mà Agent gọi. LangChain dùng docstring bên dưới
# làm tool description, vì vậy nội dung tiếng Anh của docstring được giữ nguyên.
@tool(args_schema=RetrieveEvidenceInput)
async def retrieve_evidence(query: str, top_k: int = 8) -> dict[str, Any]:
    """Retrieve bounded medical source evidence with Dense + BM25 + RRF."""

    retriever = EvidenceRetriever()
    try:
        result = await retriever.retrieve(query, top_k=top_k)
        return {
            "vector_contexts": result.vector_contexts,
            "sources": result.sources,
            "metadata": result.metadata,
        }
    finally:
        await retriever.close()


def _to_candidate(item: dict[str, Any], rank: int) -> RetrievedCandidate:
    candidate_id = str(item.get("id") or item.get("chunk_id") or f"rrf-{rank}")
    text = str(item.get("text") or item.get("content") or item.get("page_content") or "")
    payload = {key: value for key, value in item.items() if key not in {"score", "rrf_score"}}
    return RetrievedCandidate(
        candidate_id=candidate_id,
        collection=os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge"),
        text=text,
        score=float(item.get("rrf_score") or item.get("score") or 0.0),
        fused_score=float(item.get("rrf_score") or item.get("score") or 0.0),
        payload=payload,
        rank=rank,
        debug={
            "dense_rank": item.get("dense_rank"),
            "bm25_rank": item.get("sparse_rank"),
            "dense_score": item.get("dense_score"),
            "bm25_score": item.get("sparse_score"),
        },
    )


def _channel_or_warning(name: str, value: Any, warnings: list[str]) -> list[dict[str, Any]]:
    if isinstance(value, BaseException):
        warnings.append(f"{name} channel unavailable: {sanitize_fallback_reason(value)}")
        logger.warning("%s retrieval channel unavailable: %s", name, sanitize_fallback_reason(value))
        return []
    return list(value or [])


async def _run_channel_with_timeout(channel: Any, timeout_seconds: float) -> Any:
    """Give one retrieval channel an independent finite outcome."""

    return await asyncio.wait_for(channel, timeout=timeout_seconds)


def _error_name(value: Any) -> str | None:
    return value.__class__.__name__ if isinstance(value, BaseException) else None


def _source_id(payload: dict[str, Any]) -> str:
    return str(
        payload.get("source_id")
        or payload.get("source_path")
        or payload.get("source_file")
        or payload.get("document_id")
        or ""
    ).strip()


def _bounded_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(maximum, max(minimum, value))


def _bounded_float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(maximum, max(minimum, value))


__all__ = ["EvidenceRetriever", "RetrievalResult", "retrieve_evidence"]
