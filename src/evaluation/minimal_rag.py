"""MINIMAL_RAG_V1: an evaluation-only Dense + Sparse + RRF control pipeline."""

from __future__ import annotations

import asyncio
import os
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from src.agent.emergency_contract import (
    build_anaphylaxis_like_emergency_answer,
    is_anaphylaxis_like_emergency_query,
)
from src.agent.llm.provider import generate_llm_response
from src.database.vector_store import QdrantVectorStore, embed_query
from src.retrieval.rrf import reciprocal_rank_fusion

MINIMAL_RAG_SYSTEM_ID = "MINIMAL_RAG_V1"
CURRENT_SYSTEM_ID = "CURRENT_SYSTEM"
DEFAULT_CANDIDATE_DEPTH = 15
DEFAULT_EVIDENCE_DEPTH = 5
DEFAULT_CONTEXT_MAX_CHARS = 4200
DEFAULT_RRF_K = 60

Embedder = Callable[[str], Awaitable[list[float]]]
Generator = Callable[..., Awaitable[dict[str, Any]]]


class ReadOnlyVectorStore(Protocol):
    async def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]: ...

    async def search_sparse(self, text: str, top_k: int = 5) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class MinimalEvidence:
    evidence_id: str
    source_id: str
    source_path: str
    document_id: str
    text: str
    rank: int
    rrf_score: float


@dataclass(frozen=True)
class MinimalRagResult:
    system_id: str
    query: str
    normalized_query: str
    evidence: tuple[MinimalEvidence, ...]
    context: str
    answer: str
    citations: tuple[str, ...]
    latency_ms: dict[str, float]
    call_counts: dict[str, int]
    status: str
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def minimal_normalize_query(question: str) -> str:
    """Apply only Unicode NFC, trimming, and whitespace collapse."""

    return " ".join(unicodedata.normalize("NFC", question or "").split())


class MinimalRagService:
    """Linear experimental baseline; it is intentionally not an API route."""

    def __init__(
        self,
        *,
        vector_store: ReadOnlyVectorStore | None = None,
        embedder: Embedder = embed_query,
        generator: Generator = generate_llm_response,
        candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
        evidence_depth: int = DEFAULT_EVIDENCE_DEPTH,
        context_max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
        rrf_k: int = DEFAULT_RRF_K,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self._vector_store = vector_store or QdrantVectorStore()
        self._embedder = embedder
        self._generator = generator
        self._candidate_depth = max(1, candidate_depth)
        self._evidence_depth = max(1, evidence_depth)
        self._context_max_chars = max(256, context_max_chars)
        self._rrf_k = max(1, rrf_k)
        self._provider = provider or os.getenv("LLM_PROVIDER", "gemini")
        self._model = model

    async def run(self, question: str, *, generate_answer: bool = True) -> MinimalRagResult:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        calls = _empty_call_counts()
        normalized = minimal_normalize_query(question)
        if not normalized:
            return self._result(
                question,
                normalized,
                timings,
                calls,
                started,
                status="invalid_input",
                error="Question is empty after normalization.",
            )

        if is_anaphylaxis_like_emergency_query(normalized):
            return self._result(
                question,
                normalized,
                timings,
                calls,
                started,
                status="hard_safety_response",
                answer=build_anaphylaxis_like_emergency_answer(),
                diagnostics={"hard_guard": "anaphylaxis_like_emergency_v1"},
            )

        try:
            stage_started = time.perf_counter()
            calls["embedding"] += 1
            query_vector = await self._embedder(normalized)
            timings["embedding"] = _elapsed_ms(stage_started)

            calls["qdrant"] += 2
            dense_task = asyncio.create_task(
                _timed(self._vector_store.search(query_vector, top_k=self._candidate_depth))
            )
            sparse_task = asyncio.create_task(
                _timed(self._vector_store.search_sparse(normalized, top_k=self._candidate_depth))
            )
            (dense_results, dense_ms), (sparse_results, sparse_ms) = await asyncio.gather(
                dense_task,
                sparse_task,
            )
            timings["dense"] = dense_ms
            timings["sparse"] = sparse_ms

            stage_started = time.perf_counter()
            fused = reciprocal_rank_fusion(dense_results, sparse_results, k=self._rrf_k)
            timings["rrf"] = _elapsed_ms(stage_started)

            stage_started = time.perf_counter()
            evidence, context, excluded = _render_context(
                fused,
                max_items=self._evidence_depth,
                max_chars=self._context_max_chars,
            )
            timings["context"] = _elapsed_ms(stage_started)
            diagnostics = {
                "candidate_depth": self._candidate_depth,
                "evidence_depth": self._evidence_depth,
                "context_max_chars": self._context_max_chars,
                "rrf_k": self._rrf_k,
                "retrieval_attempts": 1,
                "dense_count": len(dense_results),
                "sparse_count": len(sparse_results),
                "fused_count": len(fused),
                "excluded_missing_provenance": excluded,
            }
            citations = tuple(dict.fromkeys(item.source_id for item in evidence))
            if not evidence:
                return self._result(
                    question,
                    normalized,
                    timings,
                    calls,
                    started,
                    evidence=evidence,
                    context=context,
                    citations=citations,
                    status="insufficient_evidence",
                    answer="Tài liệu hiện có chưa đủ thông tin có nguồn để trả lời câu hỏi này.",
                    diagnostics=diagnostics,
                )

            if not generate_answer:
                return self._result(
                    question,
                    normalized,
                    timings,
                    calls,
                    started,
                    evidence=evidence,
                    context=context,
                    citations=citations,
                    status="retrieved",
                    diagnostics=diagnostics,
                )

            stage_started = time.perf_counter()
            calls["generation"] += 1
            generated = await self._generator(
                prompt=_minimal_prompt(normalized, context),
                provider=self._provider,
                model=self._model,
                temperature=0.2,
                allow_fallback=False,
            )
            timings["generation"] = _elapsed_ms(stage_started)
            diagnostics["actual_provider"] = generated.get("provider")
            diagnostics["actual_model"] = generated.get("model")
            return self._result(
                question,
                normalized,
                timings,
                calls,
                started,
                evidence=evidence,
                context=context,
                answer=str(generated.get("text") or "").strip(),
                citations=citations,
                status="completed",
                diagnostics=diagnostics,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._result(
                question,
                normalized,
                timings,
                calls,
                started,
                status="provider_or_retrieval_error",
                answer="Hệ thống chưa thể truy hồi nguồn tin cậy lúc này. Vui lòng thử lại sau.",
                error=exc.__class__.__name__,
                diagnostics={"retrieval_attempts": 1},
            )

    @staticmethod
    def _result(
        query: str,
        normalized_query: str,
        timings: dict[str, float],
        calls: dict[str, int],
        started: float,
        *,
        evidence: tuple[MinimalEvidence, ...] = (),
        context: str = "",
        answer: str = "",
        citations: tuple[str, ...] = (),
        status: str,
        error: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> MinimalRagResult:
        complete_timings = dict(timings)
        complete_timings["total"] = _elapsed_ms(started)
        return MinimalRagResult(
            system_id=MINIMAL_RAG_SYSTEM_ID,
            query=query,
            normalized_query=normalized_query,
            evidence=evidence,
            context=context,
            answer=answer,
            citations=citations,
            latency_ms=complete_timings,
            call_counts=dict(calls),
            status=status,
            error=error,
            diagnostics=dict(diagnostics or {}),
        )


async def _timed(awaitable: Awaitable[Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    return await awaitable, _elapsed_ms(started)


def _render_context(
    fused: list[dict[str, Any]],
    *,
    max_items: int,
    max_chars: int,
) -> tuple[tuple[MinimalEvidence, ...], str, int]:
    selected: list[MinimalEvidence] = []
    rendered: list[str] = []
    used_chars = 0
    excluded = 0

    for candidate in fused:
        if len(selected) >= max_items:
            break
        evidence = _evidence_from_candidate(candidate, rank=len(selected) + 1)
        if evidence is None:
            excluded += 1
            continue
        separator_chars = 2 if rendered else 0
        marker = f"[SOURCE {evidence.source_id} | CHUNK {evidence.evidence_id}]\n"
        remaining = max_chars - used_chars - separator_chars - len(marker)
        if remaining <= 0:
            break
        text = evidence.text[:remaining].rstrip()
        if not text:
            continue
        block = marker + text
        rendered.append(block)
        used_chars += separator_chars + len(block)
        selected.append(evidence)
    return tuple(selected), "\n\n".join(rendered), excluded


def _evidence_from_candidate(candidate: dict[str, Any], *, rank: int) -> MinimalEvidence | None:
    evidence_id = str(candidate.get("chunk_id") or candidate.get("id") or "").strip()
    source_path = str(candidate.get("source_path") or candidate.get("source_file") or "").strip()
    document_id = str(candidate.get("document_id") or "").strip()
    text = str(candidate.get("text") or candidate.get("content") or "").strip()
    if not evidence_id or not text or not (source_path or document_id):
        return None
    source_id = str(candidate.get("source_identity") or source_path or document_id).strip()
    return MinimalEvidence(
        evidence_id=evidence_id,
        source_id=source_id,
        source_path=source_path,
        document_id=document_id,
        text=text,
        rank=rank,
        rrf_score=float(candidate.get("rrf_score") or candidate.get("score") or 0.0),
    )


def _minimal_prompt(question: str, context: str) -> str:
    return (
        "Bạn là trợ lý cung cấp thông tin tham khảo về mụn. Trả lời câu hỏi bằng tiếng Việt, "
        "chỉ dùng bằng chứng được cung cấp cho các sự kiện y khoa. Trích dẫn source ID trong "
        "ngoặc vuông. Nếu bằng chứng không đủ, nói rõ sự không chắc chắn. Không kê đơn, không "
        "bịa thông tin và không đưa khuyến nghị y khoa nghiêm trọng không được nguồn hỗ trợ.\n\n"
        f"CÂU HỎI:\n{question}\n\nBẰNG CHỨNG:\n{context}\n"
    )


def _empty_call_counts() -> dict[str, int]:
    return {
        "embedding": 0,
        "qdrant": 0,
        "neo4j": 0,
        "redis": 0,
        "reranker": 0,
        "generation": 0,
    }


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


__all__ = [
    "CURRENT_SYSTEM_ID",
    "DEFAULT_CANDIDATE_DEPTH",
    "DEFAULT_CONTEXT_MAX_CHARS",
    "DEFAULT_EVIDENCE_DEPTH",
    "DEFAULT_RRF_K",
    "MINIMAL_RAG_SYSTEM_ID",
    "MinimalEvidence",
    "MinimalRagResult",
    "MinimalRagService",
    "minimal_normalize_query",
]
