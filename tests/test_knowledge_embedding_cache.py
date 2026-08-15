from __future__ import annotations

import pytest

from src.ingestion.embedding import EMBEDDING_DIMENSIONS, EmbeddingCache
from src.ingestion import index as ingestion_index
from src.resilience.exceptions import ProviderUnavailableError


@pytest.mark.asyncio
async def test_embedding_resolution_retries_finitely_and_checkpoints_success(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_embed(texts: list[str], *, api_key: str):
        nonlocal calls
        calls += 1
        assert api_key == "redacted-test-key"
        if calls == 1:
            raise ProviderUnavailableError("rate limited")
        return [[float(index)] * EMBEDDING_DIMENSIONS for index, _ in enumerate(texts)]

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(ingestion_index, "embed_documents", fake_embed)
    monkeypatch.setattr(ingestion_index.asyncio, "sleep", fake_sleep)
    cache = EmbeddingCache(tmp_path)

    vectors, stats = await ingestion_index.resolve_embeddings(
        ["one", "two"],
        cache=cache,
        api_key="redacted-test-key",
        batch_size=2,
        batch_delay_seconds=0,
        max_retries=2,
    )

    assert len(vectors) == 2
    assert stats == {
        "cache_hits": 0,
        "cache_misses": 2,
        "provider_calls": 1,
        "retry_count": 1,
    }
    assert sleeps == [30.0]
    assert cache.get("one") == vectors[0]
    assert cache.get("two") == vectors[1]


@pytest.mark.asyncio
async def test_embedding_resolution_stops_after_retry_limit(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_rate_limited(texts: list[str], *, api_key: str):
        raise ProviderUnavailableError("rate limited")

    async def no_wait(seconds: float) -> None:
        return None

    monkeypatch.setattr(ingestion_index, "embed_documents", always_rate_limited)
    monkeypatch.setattr(ingestion_index.asyncio, "sleep", no_wait)

    with pytest.raises(ProviderUnavailableError):
        await ingestion_index.resolve_embeddings(
            ["one"],
            cache=EmbeddingCache(tmp_path),
            api_key="redacted-test-key",
            batch_size=1,
            batch_delay_seconds=0,
            max_retries=2,
        )
