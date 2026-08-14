from __future__ import annotations

import ssl

import pytest

from src.database import vector_store
from src.resilience.exceptions import PermanentProviderError


def test_runtime_vector_store_exposes_read_only_operations_only():
    assert hasattr(vector_store.QdrantVectorStore, "search")
    assert hasattr(vector_store.QdrantVectorStore, "search_sparse")
    assert not hasattr(vector_store.QdrantVectorStore, "upsert")
    assert not hasattr(vector_store.QdrantVectorStore, "delete")


@pytest.mark.asyncio
async def test_embed_query_retries_transient_transport_errors(monkeypatch):
    attempts: list[str] = []
    delays: list[float] = []

    def flaky_embedding(_: str) -> list[float]:
        attempts.append("call")
        if len(attempts) < 3:
            raise ssl.SSLEOFError("temporary EOF")
        return [0.1, 0.2, 0.3]

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(vector_store, "_embed_sync", flaky_embedding)
    monkeypatch.setattr(vector_store, "EMBEDDING_DIMENSIONS", 3)
    monkeypatch.setattr(vector_store.asyncio, "sleep", record_sleep)
    monkeypatch.setenv("EMBEDDING_QUERY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("EMBEDDING_QUERY_RETRY_BASE_DELAY", "0.25")

    assert await vector_store.embed_query("mụn viêm") == [0.1, 0.2, 0.3]
    assert attempts == ["call", "call", "call"]
    assert delays == [0.25, 0.5]


@pytest.mark.asyncio
async def test_embed_query_does_not_retry_permanent_provider_errors(monkeypatch):
    attempts: list[str] = []

    def permanent_failure(_: str) -> list[float]:
        attempts.append("call")
        raise PermanentProviderError("invalid credentials")

    monkeypatch.setattr(vector_store, "_embed_sync", permanent_failure)
    monkeypatch.setenv("EMBEDDING_QUERY_MAX_ATTEMPTS", "3")

    with pytest.raises(PermanentProviderError):
        await vector_store.embed_query("mụn viêm")
    assert attempts == ["call"]
