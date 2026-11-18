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
    assert len(delays) == 2
    assert 0.25 <= delays[0] <= 0.275
    assert 0.5 <= delays[1] <= 0.55


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


@pytest.mark.asyncio
async def test_embed_query_forwards_request_timeout_and_resolves_env_key(monkeypatch):
    seen_timeouts: list[float | None] = []
    seen_api_keys: list[str | None] = []

    def fake_embed_texts_sync(texts, **kwargs):
        seen_api_keys.append(kwargs.get("api_key"))
        seen_timeouts.append(kwargs.get("request_timeout"))
        return [[0.25, 0.5, 0.75]]

    monkeypatch.setattr(vector_store, "GOOGLE_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "dynamic-env-key")
    monkeypatch.setattr(vector_store, "EMBEDDING_DIMENSIONS", 3)
    monkeypatch.setattr(
        "src.integrations.google_genai.embed_texts_sync",
        fake_embed_texts_sync,
    )

    result = await vector_store.embed_query("adapalene")
    assert result == [0.25, 0.5, 0.75]
    assert seen_api_keys == ["dynamic-env-key"]
    assert len(seen_timeouts) == 1
    assert seen_timeouts[0] is not None and seen_timeouts[0] > 0
