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
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.database.vector_store import QdrantVectorStore, embed_query
from src.quality.safe_fallback import sanitize_fallback_reason
from src.retrieval.context_packer import pack_context, packed_context_to_response_contexts
from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate
from src.retrieval.graph_retriever import (
    GRAPH_RAG_ENABLED_DEFAULT,
    GRAPH_RAG_WEIGHT_DEFAULT,
    GraphSearchResult,
    Neo4jGraphStore,
)
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
RRF_GRAPH_WEIGHT = GRAPH_RAG_WEIGHT_DEFAULT
# Đây là engineering parameters cho rank fusion đa kênh, không phải confidence threshold.

_process_reranker: CandidateScorer | None = None
_process_reranker_settings: RerankerSettings | None = None
_process_reranker_lock = threading.Lock()


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
    retained_retrieval_candidates: list[dict[str, Any]] = Field(default_factory=list)
    rerank_query: str | None = Field(default=None, max_length=4000)
    retrieval_attempt: int = Field(default=1, ge=1, le=2)


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
        graph_store: Neo4jGraphStore | None = None,
        graph_enabled: bool | None = None,
        reranker: CandidateScorer | None = None,
        reranker_enabled: bool | None = None,
    ) -> None:
        self._vector_store = vector_store or QdrantVectorStore()
        if graph_enabled is not None:
            resolved_graph_enabled = bool(graph_enabled)
        elif graph_store is not None:
            resolved_graph_enabled = True
        elif vector_store is not None and not isinstance(vector_store, QdrantVectorStore):
            # Giữ unit test offline truyền fake vector_store không bị gọi ra Neo4j thật.
            resolved_graph_enabled = False
        else:
            resolved_graph_enabled = _bool_env("GRAPH_RAG_ENABLED", GRAPH_RAG_ENABLED_DEFAULT)
        self._graph_enabled = resolved_graph_enabled
        self._graph_store = graph_store or (Neo4jGraphStore() if self._graph_enabled else None)

        settings = RerankerSettings.from_env()
        enabled = settings.enabled if reranker_enabled is None else reranker_enabled
        if reranker is not None and reranker_enabled is None:
            enabled = True
        self._reranker_settings = RerankerSettings(
            enabled=enabled,
            model_name=settings.model_name,
            device=settings.device,
            precision=settings.precision,
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
        graph_weight: float | None = None,
        rrf_k: int = RRF_K,
        retained_retrieval_candidates: Sequence[RetrievedCandidate | dict[str, Any]] | None = None,
        rerank_query: str | None = None,
        retrieval_attempt: int = 1,
        **_: Any,
    ) -> RetrievalResult:
        """Chạy ba kênh độc lập (Dense, BM25, Neo4j Graph), hợp nhất RRF và đóng gói chứng cứ.

        ``top_k`` giới hạn số item context cuối cùng; nó không phải relevance
        threshold. Empty list là kết quả channel hợp lệ, còn exception/timeout
        được ghi nhận là channel failure để có thể trả degraded evidence (fail-open).
        """

        started = time.perf_counter()
        clean_query = " ".join(query.split())
        if not clean_query:
            return RetrievalResult([], [], query, {"status": "empty_query"})

        candidate_limit = _bounded_env("RETRIEVAL_CANDIDATE_LIMIT", 16, 1, 50)
        context_items = min(top_k, _bounded_env("RETRIEVAL_CONTEXT_MAX_ITEMS", 9, 1, 20))
        # Cho phép đóng gói 6-8 đoạn chứng cứ trọn vẹn nhằm giảm nguy cơ thiếu chi tiết và phải retry.
        context_chars = _bounded_env("RETRIEVAL_CONTEXT_MAX_CHARS", 10000, 512, 20000)
        effective_graph_weight = (
            float(graph_weight)
            if graph_weight is not None
            else _bounded_float_env("GRAPH_RAG_WEIGHT", RRF_GRAPH_WEIGHT, 0.0, 3.0)
        )
        retained_candidates = _load_retained_candidates(
            retained_retrieval_candidates,
            limit=candidate_limit * 2,
        )

        timeout_seconds = _bounded_float_env("RETRIEVAL_TIMEOUT_SECONDS", 20.0, 0.1, 120.0)
        graph_timeout_seconds = _bounded_float_env("GRAPH_RAG_TIMEOUT_SECONDS", 8.0, 0.2, 30.0)
        # Dense, BM25 và Neo4j Graph hoàn toàn độc lập nên chạy đồng thời. Mỗi coroutine
        # được bọc bằng timeout riêng; ``return_exceptions=True`` đảm bảo lỗi một kênh
        # không hủy bỏ chứng cứ hợp lệ từ các kênh còn lại.
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
        if self._graph_enabled and self._graph_store is not None:
            from src.knowledge.versioning import resolve_active_knowledge_build_id

            active_kb_id = resolve_active_knowledge_build_id()
            graph_task = asyncio.create_task(
                _run_channel_with_timeout(
                    self._graph_store.search_graph(
                        clean_query,
                        top_k=candidate_limit,
                        kb_version=active_kb_id,
                    ),
                    graph_timeout_seconds,
                )
            )
            dense_result, bm25_result, graph_raw_result = await asyncio.gather(
                dense_task,
                bm25_task,
                graph_task,
                return_exceptions=True,
            )
        else:
            dense_result, bm25_result = await asyncio.gather(
                dense_task,
                bm25_task,
                return_exceptions=True,
            )
            graph_raw_result = GraphSearchResult(status="disabled")

        warnings: list[str] = []
        dense_results = _channel_or_warning("dense", dense_result, warnings)
        bm25_results = _channel_or_warning("bm25", bm25_result, warnings)

        graph_results: list[dict[str, Any]] = []
        graph_status = "disabled"
        graph_elapsed_ms: float | None = None
        graph_anchors: dict[str, Any] = {}
        graph_paths: list[str] = []
        if isinstance(graph_raw_result, BaseException):
            warnings.append(f"graph channel unavailable: {sanitize_fallback_reason(graph_raw_result)}")
            logger.warning(
                "graph retrieval channel unavailable: %s",
                sanitize_fallback_reason(graph_raw_result),
            )
            graph_status = "degraded"
        elif isinstance(graph_raw_result, GraphSearchResult):
            graph_results = list(graph_raw_result.candidates)
            graph_status = graph_raw_result.status
            graph_elapsed_ms = graph_raw_result.elapsed_ms
            graph_anchors = dict(graph_raw_result.anchors)
            graph_paths = list(graph_raw_result.paths)

        acquisition_failed = (
            not dense_results
            and not bm25_results
            and not graph_results
            and (isinstance(dense_result, BaseException) or isinstance(bm25_result, BaseException))
        )
        retained_recovery = acquisition_failed and bool(retained_candidates)
        if acquisition_failed and not retained_candidates:
            errors = "; ".join(warnings) or "No retrieval channel returned evidence."
            raise RuntimeError(errors)
        if retained_recovery:
            warnings.append(
                "New acquisition was unavailable; retained candidates were reranked and repacked."
            )

        fused = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            graph_results=graph_results if self._graph_enabled else None,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            graph_weight=effective_graph_weight,
            k=rrf_k,
        )
        acquired_candidates = [_to_candidate(item, rank) for rank, item in enumerate(fused, 1)]
        candidates, duplicate_ids = merge_retrieval_candidates(
            retained_candidates,
            acquired_candidates,
            retrieval_attempt=retrieval_attempt,
        )
        base_rerank_query = " ".join((rerank_query or clean_query).split()) or clean_query
        effective_rerank_query = _augment_rerank_query_with_brand_ingredients(
            base_rerank_query,
            graph_paths=graph_paths,
        )
        scorer = self._reranker
        if self._reranker_settings.enabled and scorer is None:
            scorer = _get_process_reranker(self._reranker_settings)
            self._reranker = scorer
        rerank_outcome = await rerank_candidates(
            effective_rerank_query,
            candidates,
            scorer=scorer,
            enabled=self._reranker_settings.enabled,
            configured_model=self._reranker_settings.model_name,
        )
        if rerank_outcome.fallback_used:
            preserved_order = (
                "exact RRF order"
                if not retained_candidates
                else "deterministic within-attempt rank order"
            )
            warnings.append(
                f"Local reranker unavailable; {preserved_order} was preserved "
                f"({rerank_outcome.fallback_reason})."
            )
            logger.warning(
                "Local reranker fallback preserved deterministic candidate order: %s",
                rerank_outcome.fallback_reason,
            )

        # Mở rộng có kiểm soát qua cạnh NEXT_CHUNK cho các đoạn bảng biểu đạt điểm cao
        # để tránh làm đứt gãy phần tiếp nối của bảng phác đồ / liều dùng lâm sàng.
        reranked_candidates = list(rerank_outcome.candidates)
        expanded_neighbor_ids: list[str] = []
        if (
            self._graph_enabled
            and self._graph_store is not None
            and not isinstance(graph_raw_result, BaseException)
            and self._reranker_settings.enabled
            and not rerank_outcome.fallback_used
            and reranked_candidates
            and _bool_env("GRAPH_RAG_NEIGHBOR_EXPANSION_ENABLED", True)
        ):
            reranked_candidates, expanded_neighbor_ids = await self._expand_adjacent_table_candidates(
                query=effective_rerank_query,
                reranked_candidates=reranked_candidates,
                scorer=scorer,
                retrieval_attempt=retrieval_attempt,
            )

        reordered_candidates = reranked_candidates
        if (
            self._graph_enabled
            and graph_results
            and rerank_outcome.status in {"succeeded", "ok"}
            and not rerank_outcome.fallback_used
        ):
            reordered_candidates = _reorder_for_graph_anchor_coverage(
                reranked_candidates,
                top_k=context_items,
            )

        packed_input_candidates = _prune_low_relevance_reranked_candidates(
            reordered_candidates,
            rerank_status=rerank_outcome.status,
            min_score=_bounded_float_env("RETRIEVAL_RERANK_MIN_SCORE", 0.18, 0.0, 1.0),
            relative_ratio=_bounded_float_env("RETRIEVAL_RERANK_RELATIVE_RATIO", 0.22, 0.0, 1.0),
        )
        kept_after_prune_ids = {item.candidate_id for item in packed_input_candidates}
        pruned_low_score_ids = {
            item.candidate_id
            for item in reordered_candidates
            if item.candidate_id not in kept_after_prune_ids
        }
        normalized = NormalizedQuery(
            original_query=rerank_query or query,
            normalized_text=effective_rerank_query,
        )
        packed = pack_context(
            normalized,
            packed_input_candidates,
            max_items=context_items,
            max_chars=context_chars,
        )
        contexts = packed_context_to_response_contexts(packed)
        sources = list(
            dict.fromkeys(_source_id(context) for context in contexts if _source_id(context))
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        dense_failed = isinstance(dense_result, BaseException)
        bm25_failed = isinstance(bm25_result, BaseException)
        # Status phân biệt evidence đầy đủ, degraded theo channel chính, không có
        # candidate hợp lệ, và lỗi có thể retry khi không channel nào sống sót.
        if contexts and retained_recovery:
            retrieval_status = "degraded_retained"
        elif contexts and dense_failed and bm25_failed:
            retrieval_status = "degraded_qdrant"
        elif contexts and dense_failed:
            retrieval_status = "degraded_dense"
        elif contexts and bm25_failed:
            retrieval_status = "degraded_bm25"
        elif contexts:
            retrieval_status = "ok"
        elif dense_failed or bm25_failed:
            retrieval_status = "recoverable_error"
        else:
            retrieval_status = "no_evidence"

        channels_trace: dict[str, Any] = {
            "dense": {"count": len(dense_results), "error": _error_name(dense_result)},
            "bm25": {"count": len(bm25_results), "error": _error_name(bm25_result)},
        }
        candidate_trace: dict[str, Any] = {
            "dense": _channel_candidate_trace(dense_results),
            "bm25": _channel_candidate_trace(bm25_results),
        }
        rrf_trace: dict[str, Any] = {
            "k": rrf_k,
            "dense_weight": dense_weight,
            "bm25_weight": sparse_weight,
            "formula": "sum(weight / (k + one_indexed_rank))",
        }
        if self._graph_enabled:
            channels_trace["graph"] = {
                "count": len(graph_results),
                "status": graph_status,
                "elapsed_ms": graph_elapsed_ms,
                "error": _error_name(graph_raw_result),
                "anchors": graph_anchors,
                "paths": graph_paths,
                "expanded_neighbors": expanded_neighbor_ids,
            }
            candidate_trace["graph"] = _channel_candidate_trace(graph_results)
            rrf_trace["graph_weight"] = effective_graph_weight

        candidate_trace["fused"] = _fused_candidate_trace(
            reranked_candidates,
            packed,
            pruned_ids=pruned_low_score_ids,
        )

        if self._graph_enabled:
            architecture_name = (
                "tri_channel_hybrid_graphrag_local_reranker"
                if self._reranker_settings.enabled
                else "tri_channel_hybrid_graphrag"
            )
        else:
            architecture_name = (
                "dense_bm25_rrf_local_reranker"
                if self._reranker_settings.enabled
                else "dense_bm25_rrf"
            )

        trace = {
            "architecture": architecture_name,
            "status": retrieval_status,
            "query": clean_query,
            "rerank_query": effective_rerank_query,
            "query_identity": {
                "acquisition_query": clean_query,
                "overall_rerank_query": effective_rerank_query,
            },
            "channels": channels_trace,
            "rrf": rrf_trace,
            "fused_candidate_count": len(fused),
            "eligible_candidate_count": len(candidates) + len(expanded_neighbor_ids),
            "reranker": {
                "enabled": rerank_outcome.enabled,
                "status": rerank_outcome.status,
                "model": rerank_outcome.model_name,
                "requested_device": self._reranker_settings.device,
                "device": getattr(scorer, "device", None),
                "device_fallback_reason": getattr(scorer, "device_fallback_reason", None),
                "requested_precision": getattr(
                    scorer,
                    "requested_precision",
                    self._reranker_settings.precision,
                ),
                "precision": getattr(scorer, "precision", None),
                "precision_fallback_reason": getattr(
                    scorer,
                    "precision_fallback_reason",
                    None,
                ),
                "model_load_count": getattr(scorer, "model_load_count", None),
                "fallback_used": rerank_outcome.fallback_used,
                "fallback_reason": rerank_outcome.fallback_reason,
                "elapsed_ms": rerank_outcome.elapsed_ms,
                "batch_size": self._reranker_settings.batch_size,
                "timeout_seconds": self._reranker_settings.timeout_seconds,
            },
            "selected_ids": [item.item_id for item in packed.items],
            "candidate_trace": candidate_trace,
            "retry_evidence": {
                "retrieval_attempt": retrieval_attempt,
                "retained_candidate_ids": [
                    candidate.candidate_id for candidate in retained_candidates
                ],
                "acquired_candidate_ids": [
                    candidate.candidate_id for candidate in acquired_candidates
                ],
                "duplicate_candidate_ids": duplicate_ids,
                "eligible_candidate_ids": [candidate.candidate_id for candidate in candidates],
                "reranked_candidate_ids": [
                    candidate.candidate_id for candidate in reranked_candidates
                ],
                "packed_candidate_ids": [item.item_id for item in packed.items],
                "fallback_ordering": (
                    "within_attempt_rank_then_first_seen_attempt_then_candidate_id"
                ),
            },
            "packer": {
                "limits": dict(packed.debug.get("limits") or {}),
                "context_chars": len(packed.context_text),
                "selected_ids": list(packed.debug.get("selected_ids") or []),
                "dropped": list(packed.debug.get("dropped") or []),
                "not_examined": list(packed.debug.get("not_examined") or []),
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
                "retained_retrieval_candidates": [
                    candidate.model_dump(mode="json") for candidate in reranked_candidates
                ],
                "timings": {
                    "retrieval_total_ms": elapsed_ms,
                    "reranker_ms": rerank_outcome.elapsed_ms,
                    **({"graph_ms": graph_elapsed_ms} if graph_elapsed_ms is not None else {}),
                },
            },
        )

    async def _expand_adjacent_table_candidates(
        self,
        *,
        query: str,
        reranked_candidates: list[RetrievedCandidate],
        scorer: CandidateScorer | None,
        retrieval_attempt: int,
    ) -> tuple[list[RetrievedCandidate], list[str]]:
        """Mở rộng và chấm điểm chéo các đoạn bảng biểu kế cận qua cạnh NEXT_CHUNK."""

        if self._graph_store is None or not reranked_candidates:
            return reranked_candidates, []

        raw_top = reranked_candidates[0].rerank_score
        if not isinstance(raw_top, (int, float)) or not (0.0 <= float(raw_top) <= 1.0):
            return reranked_candidates, []
        top_score = float(raw_top)
        if top_score < 0.35:
            return reranked_candidates, []

        table_seeds = [
            item.candidate_id
            for item in reranked_candidates[:3]
            if isinstance(item.rerank_score, (int, float))
            and 0.35 <= float(item.rerank_score) <= 1.0
            and str(item.payload.get("content_kind") or "") == "table"
        ]
        if not table_seeds:
            return reranked_candidates, []

        existing_ids = {item.candidate_id for item in reranked_candidates}
        try:
            from src.knowledge.versioning import resolve_active_knowledge_build_id

            neighbor_dicts = await _run_channel_with_timeout(
                self._graph_store.fetch_adjacent_table_chunks(
                    table_seeds,
                    exclude_ids=existing_ids,
                    kb_version=resolve_active_knowledge_build_id(),
                    limit=3,
                ),
                2.5,
            )
        except Exception as exc:
            logger.debug("Graph neighbor table expansion skipped safely: %s", exc.__class__.__name__)
            return reranked_candidates, []

        if not neighbor_dicts:
            return reranked_candidates, []

        base_rank = len(reranked_candidates)
        raw_neighbors = [
            _candidate_with_attempt_metadata(
                _to_candidate(item, base_rank + idx).model_copy(
                    update={"debug": {"graph_neighbor_expanded": True}}
                ),
                retrieval_attempt,
            )
            for idx, item in enumerate(neighbor_dicts, 1)
        ]
        neighbor_rerank = await rerank_candidates(
            query,
            raw_neighbors,
            scorer=scorer,
            enabled=self._reranker_settings.enabled,
            configured_model=self._reranker_settings.model_name,
        )
        if neighbor_rerank.fallback_used:
            return reranked_candidates, []

        keep_threshold = max(0.28, top_score * 0.40)
        accepted_neighbors: list[RetrievedCandidate] = []
        for cand in neighbor_rerank.candidates:
            cid = cand.candidate_id.strip()
            if (
                cid
                and cid not in existing_ids
                and isinstance(cand.rerank_score, (int, float))
                and float(cand.rerank_score) >= keep_threshold
            ):
                existing_ids.add(cid)
                accepted_neighbors.append(cand)
        if not accepted_neighbors:
            return reranked_candidates, []

        combined = sorted(
            [*reranked_candidates, *accepted_neighbors],
            key=lambda item: (
                -float(item.rerank_score if item.rerank_score is not None else -1.0),
                item.rank or 1_000_000,
                item.candidate_id,
            ),
        )
        reindexed = [
            item.model_copy(update={"rerank_rank": idx})
            for idx, item in enumerate(combined, 1)
        ]
        return reindexed, [item.candidate_id for item in accepted_neighbors]

    async def _dense_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Nhận vector từ Gemini rồi yêu cầu Qdrant thực thi cosine search."""

        vector = await embed_query(query)
        return await self._vector_store.search(vector, top_k=limit)

    async def close(self) -> None:
        try:
            await self._vector_store.close()
        except Exception as exc:
            logger.warning(
                "Failed to close retrieval vector store: %s", sanitize_fallback_reason(exc)
            )
        graph_store = getattr(self, "_graph_store", None)
        if graph_store is not None:
            try:
                await graph_store.close()
            except Exception as exc:
                logger.warning(
                    "Failed to close retrieval graph store: %s", sanitize_fallback_reason(exc)
                )


# Đây là evidence tool duy nhất mà Agent gọi. LangChain dùng docstring bên dưới
# làm tool description, vì vậy nội dung tiếng Anh của docstring được giữ nguyên.
@tool(args_schema=RetrieveEvidenceInput)
async def retrieve_evidence(
    query: str,
    top_k: int = 8,
    retained_retrieval_candidates: list[dict[str, Any]] | None = None,
    rerank_query: str | None = None,
    retrieval_attempt: int = 1,
) -> dict[str, Any]:
    """Retrieve bounded medical source evidence with Dense + BM25 + RRF."""

    retriever = EvidenceRetriever()
    try:
        result = await retriever.retrieve(
            query,
            top_k=top_k,
            retained_retrieval_candidates=retained_retrieval_candidates,
            rerank_query=rerank_query,
            retrieval_attempt=retrieval_attempt,
        )
        return {
            "vector_contexts": result.vector_contexts,
            "sources": result.sources,
            "metadata": result.metadata,
        }
    finally:
        await retriever.close()


def _get_process_reranker(settings: RerankerSettings) -> CandidateScorer:
    """Return one lazy local reranker for the active process configuration."""

    global _process_reranker, _process_reranker_settings
    with _process_reranker_lock:
        if _process_reranker is None or _process_reranker_settings != settings:
            _process_reranker = CandidateReranker(settings)
            _process_reranker_settings = settings
        return _process_reranker


async def warm_process_reranker() -> None:
    """Preload the enabled process reranker so model loading is not request latency."""

    settings = RerankerSettings.from_env()
    if not settings.enabled:
        return
    try:
        scorer = _get_process_reranker(settings)
        prepare = getattr(scorer, "prepare", None)
        if prepare is None:
            return
        await prepare()
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Local reranker startup preload failed; request-time fallback remains active: %s",
            exc.__class__.__name__,
        )


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
            "graph_rank": item.get("graph_rank"),
            "dense_score": item.get("dense_score"),
            "bm25_score": item.get("sparse_score"),
            "graph_score": item.get("graph_score"),
        },
    )


def merge_retrieval_candidates(
    retained_candidates: Sequence[RetrievedCandidate],
    acquired_candidates: Sequence[RetrievedCandidate],
    *,
    retrieval_attempt: int,
) -> tuple[list[RetrievedCandidate], list[str]]:
    """Deduplicate candidates by stable ID and preserve bounded attempt metadata.

    RRF ranks from different queries are not treated as calibrated scores. The
    pre-reranker order uses only each candidate's within-attempt rank, first-seen
    attempt, and stable ID so reranker fallback remains deterministic.
    """

    previous_attempt = max(1, retrieval_attempt - 1)
    prepared_retained = [
        _candidate_with_attempt_metadata(candidate, previous_attempt)
        for candidate in retained_candidates
    ]
    prepared_acquired = [
        _candidate_with_attempt_metadata(candidate, retrieval_attempt)
        for candidate in acquired_candidates
    ]
    merged_by_id: dict[str, RetrievedCandidate] = {}
    duplicate_ids: list[str] = []
    for candidate in [*prepared_retained, *prepared_acquired]:
        existing = merged_by_id.get(candidate.candidate_id)
        if existing is None:
            merged_by_id[candidate.candidate_id] = candidate
            continue
        merged_by_id[candidate.candidate_id] = existing.model_copy(
            update={
                "payload": _merge_candidate_payload(existing.payload, candidate.payload),
                "debug": _merge_candidate_debug(existing.debug, candidate.debug),
            }
        )
        if candidate.candidate_id not in duplicate_ids:
            duplicate_ids.append(candidate.candidate_id)

    merged = sorted(merged_by_id.values(), key=_cross_attempt_fallback_key)
    return merged, duplicate_ids


def _load_retained_candidates(
    values: Sequence[RetrievedCandidate | dict[str, Any]] | None,
    *,
    limit: int,
) -> list[RetrievedCandidate]:
    return [
        value if isinstance(value, RetrievedCandidate) else RetrievedCandidate.model_validate(value)
        for value in list(values or [])[:limit]
    ]


def _candidate_with_attempt_metadata(
    candidate: RetrievedCandidate,
    retrieval_attempt: int,
) -> RetrievedCandidate:
    debug = dict(candidate.debug)
    seen = {int(value) for value in debug.get("seen_in_attempts", []) if str(value).isdigit()}
    seen.add(retrieval_attempt)
    ranks = {
        str(key): int(value)
        for key, value in dict(debug.get("rank_by_attempt") or {}).items()
        if str(value).isdigit()
    }
    if candidate.rank is not None:
        ranks.setdefault(str(retrieval_attempt), candidate.rank)
    debug.update(
        {
            "seen_in_attempts": sorted(seen),
            "rank_by_attempt": ranks,
            "first_seen_attempt": min(seen),
        }
    )
    return candidate.model_copy(update={"debug": debug, "rerank_score": None, "rerank_rank": None})


def _merge_candidate_payload(first: dict[str, Any], duplicate: dict[str, Any]) -> dict[str, Any]:
    """Hợp nhất bằng chứng đồ thị Neo4j giữa các lượt retry mà không ghi đè nội dung gốc."""

    merged = dict(first)
    ev1 = first.get("graph_evidence")
    ev2 = duplicate.get("graph_evidence")
    if isinstance(ev1, dict) and isinstance(ev2, dict):
        merged["graph_evidence"] = {
            key: list(dict.fromkeys([*(ev1.get(key) or ()), *(ev2.get(key) or ())]))
            for key in ("matched_entities", "matched_facets", "expanded_paths")
        }
    elif isinstance(ev2, dict) and not isinstance(ev1, dict):
        merged["graph_evidence"] = dict(ev2)
    return merged


def _merge_candidate_debug(first: dict[str, Any], duplicate: dict[str, Any]) -> dict[str, Any]:
    debug = dict(first)
    for key in ("dense_rank", "bm25_rank", "graph_rank", "dense_score", "bm25_score"):
        if debug.get(key) is None and duplicate.get(key) is not None:
            debug[key] = duplicate[key]
    if duplicate.get("graph_score") is not None:
        debug["graph_score"] = max(
            float(debug.get("graph_score") or 0.0),
            float(duplicate.get("graph_score") or 0.0),
        )
    seen = {
        int(value)
        for value in [
            *first.get("seen_in_attempts", []),
            *duplicate.get("seen_in_attempts", []),
        ]
        if str(value).isdigit()
    }
    ranks = {
        **dict(first.get("rank_by_attempt") or {}),
        **dict(duplicate.get("rank_by_attempt") or {}),
    }
    debug.update(
        {
            "seen_in_attempts": sorted(seen),
            "rank_by_attempt": ranks,
            "first_seen_attempt": min(seen),
        }
    )
    return debug


def _cross_attempt_fallback_key(candidate: RetrievedCandidate) -> tuple[int, int, str]:
    ranks = [
        int(value)
        for value in dict(candidate.debug.get("rank_by_attempt") or {}).values()
        if str(value).isdigit()
    ]
    seen = [
        int(value) for value in candidate.debug.get("seen_in_attempts", []) if str(value).isdigit()
    ]
    return (
        min(ranks, default=candidate.rank or 1_000_000),
        min(seen, default=1_000_000),
        candidate.candidate_id,
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
    *,
    pruned_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Expose RRF and packer outcomes for diagnostics without changing selection."""

    selected_ids = set(packed.debug.get("selected_ids") or [])
    dropped_by_id = {
        str(entry.get("candidate_id") or ""): str(entry.get("reason") or "")
        for entry in packed.debug.get("dropped") or []
        if isinstance(entry, dict)
    }
    for cid in pruned_ids or set():
        dropped_by_id.setdefault(cid, "low_rerank_score")
    not_examined_by_id = {
        str(entry.get("candidate_id") or ""): str(entry.get("reason") or "")
        for entry in packed.debug.get("not_examined") or []
        if isinstance(entry, dict)
    }
    trace: list[dict[str, Any]] = []
    for candidate in candidates:
        payload = candidate.payload
        section_path = payload.get("section_path")
        section = payload.get("header") or (
            section_path[-1] if isinstance(section_path, list) and section_path else None
        )
        is_selected = candidate.candidate_id in selected_ids
        drop_reason = dropped_by_id.get(candidate.candidate_id)
        not_examined_reason = not_examined_by_id.get(candidate.candidate_id)
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
                "graph_rank": candidate.debug.get("graph_rank"),
                "graph_neighbor_expanded": bool(candidate.debug.get("graph_neighbor_expanded")),
                "seen_in_attempts": list(candidate.debug.get("seen_in_attempts") or []),
                "rank_by_attempt": dict(candidate.debug.get("rank_by_attempt") or {}),
                "source_id": _source_id(payload),
                "section": section,
                "packed": is_selected,
                "packing_status": (
                    "selected"
                    if is_selected
                    else "dropped"
                    if drop_reason is not None
                    else "not_examined"
                    if not_examined_reason is not None
                    else "unknown"
                ),
                "drop_reason": drop_reason,
                "not_examined_reason": not_examined_reason,
            }
        )
    return trace


def _channel_or_warning(name: str, value: Any, warnings: list[str]) -> list[dict[str, Any]]:
    if isinstance(value, BaseException):
        warnings.append(f"{name} channel unavailable: {sanitize_fallback_reason(value)}")
        logger.warning(
            "%s retrieval channel unavailable: %s", name, sanitize_fallback_reason(value)
        )
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


def _prune_low_relevance_reranked_candidates(
    candidates: list[RetrievedCandidate],
    *,
    rerank_status: str,
    min_score: float = 0.18,
    relative_ratio: float = 0.22,
    min_keep: int = 2,
) -> list[RetrievedCandidate]:
    """Loại bỏ các đoạn nhiễu ở đuôi danh sách sau khi cross-encoder chấm điểm thấp.

    ``rerank_candidates`` trả về ``status="succeeded"`` khi chấm thành công. Chỉ lọc
    khi toàn bộ điểm nằm trong thang xác suất sigmoid ``[0.0, 1.0]`` với ứng viên đầu
    bảng đủ tin cậy (``top_score >= min_score``), luôn giữ tối thiểu ``min_keep`` đoạn.
    """

    if rerank_status not in {"succeeded", "ok"} or len(candidates) <= min_keep or min_score <= 0.0:
        return candidates
    scores = [item.rerank_score for item in candidates]
    if not all(isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0 for score in scores):
        return candidates
    top_score = float(scores[0])  # type: ignore[arg-type]
    if top_score < min_score:
        return candidates

    threshold = max(min_score, top_score * relative_ratio)
    kept: list[RetrievedCandidate] = []
    for idx, item in enumerate(candidates):
        score = float(item.rerank_score)  # type: ignore[arg-type]
        if idx < min_keep or score >= threshold:
            kept.append(item)
    return kept


def _augment_rerank_query_with_brand_ingredients(
    query: str,
    *,
    graph_paths: Sequence[str],
) -> str:
    """Bổ sung hoạt chất chuẩn cho biệt dược (vd: Epiduo -> adapalene, benzoyl peroxide) khi chấm Cross-Encoder."""

    if not graph_paths:
        return query
    brand_ingredients: list[str] = []
    query_lower = query.lower()
    for path in graph_paths:
        if "-[HAS_ACTIVE_INGREDIENT]->" not in path:
            continue
        _, target = path.split("-[HAS_ACTIVE_INGREDIENT]->", 1)
        clean_ing = target.replace("_", " ").strip()
        if clean_ing and clean_ing.lower() not in query_lower and clean_ing not in brand_ingredients:
            brand_ingredients.append(clean_ing)
    if not brand_ingredients:
        return query
    return f"{query} ({', '.join(brand_ingredients)})"


def _candidate_graph_tokens(candidate: RetrievedCandidate) -> set[str]:
    evidence = candidate.payload.get("graph_evidence")
    if not isinstance(evidence, dict):
        return set()
    tokens: set[str] = set()
    for key in ("matched_entities", "matched_facets", "expanded_paths"):
        for item in evidence.get(key) or ():
            if item:
                tokens.add(str(item))
    return tokens


def _reorder_for_graph_anchor_coverage(
    candidates: list[RetrievedCandidate],
    *,
    top_k: int,
) -> list[RetrievedCandidate]:
    """Tối ưu độ phủ đa điểm neo (Submodular Anchor Coverage) cho tập ứng viên đã qua BGE Reranker.

    Giữ nguyên ứng viên hạng #1 của Cross-Encoder để bảo toàn độ chính xác đỉnh, sau đó
    ưu tiên vừa phải các ứng viên đạt chuẩn liên quan của BGE nhưng bao phủ thêm thực thể,
    tương tác hoặc chống chỉ định từ đồ thị Neo4j chưa xuất hiện ở các đoạn phía trên.
    """

    if len(candidates) <= 2 or top_k <= 1:
        return candidates
    scores = [item.rerank_score for item in candidates]
    if not all(isinstance(s, (int, float)) and 0.0 <= float(s) <= 1.0 for s in scores):
        return candidates

    top_score = float(scores[0])  # type: ignore[arg-type]
    if top_score < 0.25:
        return candidates

    eligibility_floor = max(0.18, top_score * 0.22)
    all_universe_tokens: set[str] = set()
    tokens_by_id: dict[str, set[str]] = {}
    for cand in candidates:
        toks = _candidate_graph_tokens(cand)
        tokens_by_id[cand.candidate_id] = toks
        all_universe_tokens.update(toks)

    if len(all_universe_tokens) <= 1:
        return candidates

    selected: list[RetrievedCandidate] = [candidates[0]]
    covered_tokens: set[str] = set(tokens_by_id.get(candidates[0].candidate_id, set()))
    remaining: list[RetrievedCandidate] = list(candidates[1:])

    selection_window = min(len(candidates), max(top_k + 2, 6))
    while remaining and len(selected) < selection_window:
        best_idx = 0
        best_utility = -1e9
        for idx, cand in enumerate(remaining):
            base_rerank = float(cand.rerank_score or 0.0)
            if base_rerank < eligibility_floor:
                utility = base_rerank
            else:
                cand_tokens = tokens_by_id.get(cand.candidate_id, set())
                uncovered = cand_tokens - covered_tokens
                coverage_gain = len(uncovered) / max(1, len(all_universe_tokens))
                graph_raw = float(cand.debug.get("graph_score") or 0.0)
                graph_bonus = 0.05 * min(1.0, graph_raw / 3.0) if graph_raw > 0 else 0.0
                utility = base_rerank + (0.22 * coverage_gain) + graph_bonus
            if utility > best_utility:
                best_utility = utility
                best_idx = idx
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        covered_tokens.update(tokens_by_id.get(chosen.candidate_id, set()))

    return [*selected, *remaining]


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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


__all__ = [
    "EvidenceRetriever",
    "RetrievalResult",
    "merge_retrieval_candidates",
    "retrieve_evidence",
    "warm_process_reranker",
]
