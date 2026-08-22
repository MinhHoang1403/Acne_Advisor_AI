"""Điều phối Dense/BM25, RRF, local reranking và whole-chunk packing.

Data flow của một lần retrieval:
``query -> Dense và BM25 song song -> RRF -> reranker -> packer -> evidence``.
Project gửi query tới Gemini để nhận Dense vector; Qdrant thực thi cosine search
và BM25 search; Python hợp nhất rank và áp resource budget cho context.

Hai channel có timeout độc lập. Lỗi của một channel không làm mất evidence đã
nhận từ channel còn lại. Service không generate answer, không xác minh clinical
truth và không dùng EntityCards hay Neo4j để grounding câu trả lời thông thường.

Điểm thường cần chỉnh sửa:
- RRF defaults: các hằng số ``RRF_*`` trong module này.
- Candidate/context/timeout budgets: các biến môi trường đọc trong ``retrieve``.
- Whole-chunk packing: ``src/retrieval/context_packer.py``.
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
from src.retrieval.reranker import (
    CandidateReranker,
    CandidateScorer,
    RerankerSettings,
    rerank_candidates,
)
from src.retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

RRF_K = 60
RRF_DENSE_WEIGHT = 1.0
RRF_BM25_WEIGHT = 1.0
# Đây là engineering parameters cho rank fusion, không phải confidence threshold.


@dataclass
class RetrievalResult:
    """Retrieval output có source identity cho Agent và endpoint ``/retrieve``."""

    vector_contexts: list[dict[str, Any]]
    sources: list[str]
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrieveEvidenceInput(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=8, ge=1, le=20)


class EvidenceRetriever:
    """Owner của request-time hybrid retrieval và bounded context packing.

    Instance dùng adapter Qdrant để chạy hai search channel, sau đó trả
    ``RetrievalResult`` có packed evidence và trace. Class không generate answer
    hoặc đánh giá medical truth.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
        *,
        reranker: CandidateScorer | None = None,
        reranker_enabled: bool | None = None,
    ) -> None:
        self._vector_store = vector_store or QdrantVectorStore()
        settings = RerankerSettings.from_env()
        enabled = settings.enabled if reranker_enabled is None else reranker_enabled
        if reranker is not None and reranker_enabled is None:
            enabled = True
        self._reranker_settings = RerankerSettings(
            enabled=enabled,
            model_name=settings.model_name,
            device=settings.device,
            batch_size=settings.batch_size,
            timeout_seconds=settings.timeout_seconds,
        )
        self._reranker = reranker

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
        scorer = self._reranker
        if self._reranker_settings.enabled and scorer is None:
            scorer = CandidateReranker(self._reranker_settings)
            self._reranker = scorer
        rerank_outcome = await rerank_candidates(
            clean_query,
            candidates,
            scorer=scorer,
            enabled=self._reranker_settings.enabled,
            configured_model=self._reranker_settings.model_name,
        )
        if rerank_outcome.fallback_used:
            warnings.append(
                "Local reranker unavailable; exact RRF order was preserved "
                f"({rerank_outcome.fallback_reason})."
            )
            logger.warning(
                "Local reranker fallback preserved RRF order: %s",
                rerank_outcome.fallback_reason,
            )
        normalized = NormalizedQuery(
            original_query=query,
            normalized_text=clean_query,
        )
        packed = pack_context(
            normalized,
            rerank_outcome.candidates,
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
            "architecture": (
                "dense_bm25_rrf_local_reranker"
                if self._reranker_settings.enabled
                else "dense_bm25_rrf"
            ),
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
            "reranker": {
                "enabled": rerank_outcome.enabled,
                "status": rerank_outcome.status,
                "model": rerank_outcome.model_name,
                "fallback_used": rerank_outcome.fallback_used,
                "fallback_reason": rerank_outcome.fallback_reason,
                "elapsed_ms": rerank_outcome.elapsed_ms,
                "batch_size": self._reranker_settings.batch_size,
                "timeout_seconds": self._reranker_settings.timeout_seconds,
            },
            "selected_ids": [item.item_id for item in packed.items],
            "candidate_trace": {
                "dense": _channel_candidate_trace(dense_results),
                "bm25": _channel_candidate_trace(bm25_results),
                "fused": _fused_candidate_trace(rerank_outcome.candidates, packed),
            },
            "packer": {
                "limits": dict(packed.debug.get("limits") or {}),
                "context_chars": len(packed.context_text),
                "selected_ids": list(packed.debug.get("selected_ids") or []),
                "dropped": list(packed.debug.get("dropped") or []),
            },
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
                "timings": {
                    "retrieval_total_ms": elapsed_ms,
                    "reranker_ms": rerank_outcome.elapsed_ms,
                },
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


def _channel_candidate_trace(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose bounded channel rank metadata without duplicating source text."""

    return [
        {
            "candidate_id": str(item.get("id") or item.get("chunk_id") or f"candidate-{rank}"),
            "rank": rank,
            "native_score": item.get("score"),
        }
        for rank, item in enumerate(items, 1)
    ]


def _fused_candidate_trace(
    candidates: list[RetrievedCandidate],
    packed: Any,
) -> list[dict[str, Any]]:
    """Expose RRF and packer outcomes for diagnostics without changing selection."""

    selected_ids = set(packed.debug.get("selected_ids") or [])
    dropped_by_id = {
        str(entry.get("candidate_id") or ""): str(entry.get("reason") or "")
        for entry in packed.debug.get("dropped") or []
        if isinstance(entry, dict)
    }
    trace: list[dict[str, Any]] = []
    for candidate in candidates:
        payload = candidate.payload
        section_path = payload.get("section_path")
        section = payload.get("header") or (
            section_path[-1] if isinstance(section_path, list) and section_path else None
        )
        trace.append(
            {
                "candidate_id": candidate.candidate_id,
                "rank": candidate.rank,
                "rrf_rank": candidate.rank,
                "fused_score": candidate.fused_score,
                "rerank_rank": candidate.rerank_rank,
                "rerank_score": candidate.rerank_score,
                "dense_rank": candidate.debug.get("dense_rank"),
                "bm25_rank": candidate.debug.get("bm25_rank"),
                "source_id": _source_id(payload),
                "section": section,
                "packed": candidate.candidate_id in selected_ids,
                "drop_reason": dropped_by_id.get(candidate.candidate_id),
            }
        )
    return trace


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
