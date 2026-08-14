import asyncio

import pytest

from src.database.vector_store import QdrantVectorStore
from src.retrieval.service import EvidenceRetriever


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
    assert result.metadata["retrieval_status"] == "ok"


@pytest.mark.asyncio
async def test_retrieval_timeout_is_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    class SlowStore:
        async def search(self, *_args, **_kwargs):
            await asyncio.sleep(1)

        async def search_sparse(self, *_args, **_kwargs):
            await asyncio.sleep(1)

        async def close(self):
            return None

    async def slow_embedding(_query: str):
        await asyncio.sleep(1)

    monkeypatch.setenv("RETRIEVAL_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setattr("src.retrieval.service.embed_query", slow_embedding)

    with pytest.raises(TimeoutError):
        await EvidenceRetriever(SlowStore()).retrieve("mụn viêm")
