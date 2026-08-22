import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from src.database.vector_store import QdrantVectorStore
from src.retrieval import service as retrieval_service
from src.retrieval.service import EvidenceRetriever


Channel = Callable[[], Awaitable[list[dict[str, Any]]]]


def _evidence(candidate_id: str, source_id: str) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "score": 1.0,
        "text": f"Source evidence from {source_id}.",
        "chunk_id": candidate_id,
        "source_id": source_id,
    }


def _retriever(dense_channel: Channel, bm25_channel: Channel) -> EvidenceRetriever:
    class FakeStore:
        async def search_sparse(self, _query: str, top_k: int) -> list[dict[str, Any]]:
            return await bm25_channel()

        async def close(self) -> None:
            return None

    class ChannelRetriever(EvidenceRetriever):
        async def _dense_search(self, query: str, limit: int) -> list[dict[str, Any]]:
            return await dense_channel()

    return ChannelRetriever(FakeStore())


@pytest.mark.asyncio
async def test_default_test_environment_disables_process_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def dense() -> list[dict[str, Any]]:
        return [_evidence("dense-1", "source-1")]

    async def bm25() -> list[dict[str, Any]]:
        return []

    def unexpected_reranker(_settings: object) -> object:
        raise AssertionError("normal tests must not construct the process reranker")

    assert os.environ["RERANKER_ENABLED"] == "false"
    monkeypatch.setattr(retrieval_service, "_get_process_reranker", unexpected_reranker)

    result = await _retriever(dense, bm25).retrieve("mụn viêm")

    assert result.metadata["retrieval_trace"]["reranker"]["enabled"] is False


@pytest.mark.asyncio
async def test_vector_store_reuses_process_client(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[object] = []

    class FakeClient:
        def __init__(self, **kwargs):
            created.append(self)

    monkeypatch.setattr("qdrant_client.AsyncQdrantClient", FakeClient)
    QdrantVectorStore._shared_client = None
    first = QdrantVectorStore()
    second = QdrantVectorStore()

    assert first._client is second._client
    assert len(created) == 1
    QdrantVectorStore._shared_client = None


@pytest.mark.asyncio
async def test_dense_failure_degrades_to_bm25(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStore:
        async def search(self, *_args, **_kwargs):
            raise AssertionError("dense search is not reached when embedding fails")

        async def search_sparse(self, _query, top_k):
            return [
                {
                    "id": "bm25-1",
                    "score": 5.0,
                    "text": "Benzoyl peroxide is an antimicrobial acne treatment.",
                    "chunk_id": "bm25-1",
                    "source_id": "guideline",
                }
            ]

        async def close(self):
            return None

    async def fail_embedding(_query: str):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr("src.retrieval.service.embed_query", fail_embedding)
    result = await EvidenceRetriever(FakeStore()).retrieve("benzoyl peroxide", top_k=4)

    assert [item["id"] for item in result.vector_contexts] == ["bm25-1"]
    assert result.metadata["retrieval_trace"]["channels"]["dense"]["error"] == "RuntimeError"
    assert result.metadata["retrieval_status"] == "degraded_dense"


@pytest.mark.asyncio
async def test_both_channels_succeed_and_are_fused() -> None:
    async def dense() -> list[dict[str, Any]]:
        return [_evidence("dense-1", "dense-source")]

    async def bm25() -> list[dict[str, Any]]:
        return [_evidence("bm25-1", "bm25-source")]

    result = await _retriever(dense, bm25).retrieve("mụn viêm")
    trace = result.metadata["retrieval_trace"]

    assert result.metadata["retrieval_status"] == "ok"
    assert trace["channels"]["dense"] == {"count": 1, "error": None}
    assert trace["channels"]["bm25"] == {"count": 1, "error": None}
    assert {item["id"] for item in result.vector_contexts} == {"dense-1", "bm25-1"}


@pytest.mark.asyncio
async def test_dense_timeout_preserves_bm25_and_cancels_dense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense_cancelled = asyncio.Event()

    async def dense() -> list[dict[str, Any]]:
        try:
            await asyncio.sleep(10)
        finally:
            dense_cancelled.set()
        return []

    async def bm25() -> list[dict[str, Any]]:
        return [_evidence("bm25-1", "bm25-source")]

    monkeypatch.setenv("RETRIEVAL_TIMEOUT_SECONDS", "0.1")
    started = time.perf_counter()
    result = await _retriever(dense, bm25).retrieve("mụn viêm")
    elapsed = time.perf_counter() - started
    trace = result.metadata["retrieval_trace"]

    assert result.metadata["retrieval_status"] == "degraded_dense"
    assert [item["id"] for item in result.vector_contexts] == ["bm25-1"]
    assert result.sources == ["bm25-source"]
    assert trace["channels"]["dense"] == {"count": 0, "error": "TimeoutError"}
    assert trace["channels"]["bm25"]["count"] == 1
    assert dense_cancelled.is_set()
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_bm25_timeout_preserves_dense_and_cancels_bm25(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bm25_cancelled = asyncio.Event()

    async def dense() -> list[dict[str, Any]]:
        return [_evidence("dense-1", "dense-source")]

    async def bm25() -> list[dict[str, Any]]:
        try:
            await asyncio.sleep(10)
        finally:
            bm25_cancelled.set()
        return []

    monkeypatch.setenv("RETRIEVAL_TIMEOUT_SECONDS", "0.1")
    result = await _retriever(dense, bm25).retrieve("mụn viêm")
    trace = result.metadata["retrieval_trace"]

    assert result.metadata["retrieval_status"] == "degraded_bm25"
    assert [item["id"] for item in result.vector_contexts] == ["dense-1"]
    assert result.sources == ["dense-source"]
    assert trace["channels"]["bm25"] == {"count": 0, "error": "TimeoutError"}
    assert bm25_cancelled.is_set()


@pytest.mark.asyncio
async def test_bm25_failure_degrades_to_dense() -> None:
    async def dense() -> list[dict[str, Any]]:
        return [_evidence("dense-1", "dense-source")]

    async def bm25() -> list[dict[str, Any]]:
        raise RuntimeError("sparse unavailable")

    result = await _retriever(dense, bm25).retrieve("mụn viêm")

    assert [item["id"] for item in result.vector_contexts] == ["dense-1"]
    assert result.metadata["retrieval_status"] == "degraded_bm25"
    assert result.metadata["retrieval_trace"]["channels"]["bm25"]["error"] == "RuntimeError"


@pytest.mark.asyncio
async def test_both_channel_timeouts_fail_and_cancel_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense_cancelled = asyncio.Event()
    bm25_cancelled = asyncio.Event()

    async def slow(cancelled: asyncio.Event) -> list[dict[str, Any]]:
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()
        return []

    async def dense() -> list[dict[str, Any]]:
        return await slow(dense_cancelled)

    async def bm25() -> list[dict[str, Any]]:
        return await slow(bm25_cancelled)

    monkeypatch.setenv("RETRIEVAL_TIMEOUT_SECONDS", "0.1")

    with pytest.raises(RuntimeError):
        await _retriever(dense, bm25).retrieve("mụn viêm")

    assert dense_cancelled.is_set()
    assert bm25_cancelled.is_set()


@pytest.mark.asyncio
async def test_both_channel_exceptions_use_existing_failure_contract() -> None:
    async def dense() -> list[dict[str, Any]]:
        raise RuntimeError("dense unavailable")

    async def bm25() -> list[dict[str, Any]]:
        raise ValueError("sparse unavailable")

    with pytest.raises(RuntimeError, match="channel unavailable"):
        await _retriever(dense, bm25).retrieve("mụn viêm")


@pytest.mark.asyncio
async def test_empty_channel_is_not_marked_degraded() -> None:
    async def dense() -> list[dict[str, Any]]:
        return []

    async def bm25() -> list[dict[str, Any]]:
        return [_evidence("bm25-1", "bm25-source")]

    result = await _retriever(dense, bm25).retrieve("mụn viêm")
    trace = result.metadata["retrieval_trace"]

    assert result.metadata["retrieval_status"] == "ok"
    assert trace["channels"]["dense"] == {"count": 0, "error": None}
    assert [item["id"] for item in result.vector_contexts] == ["bm25-1"]


@pytest.mark.asyncio
async def test_both_empty_channels_keep_no_evidence_semantics() -> None:
    async def empty() -> list[dict[str, Any]]:
        return []

    result = await _retriever(empty, empty).retrieve("mụn viêm")

    assert result.metadata["retrieval_status"] == "no_evidence"
    assert result.vector_contexts == []
    assert result.sources == []
