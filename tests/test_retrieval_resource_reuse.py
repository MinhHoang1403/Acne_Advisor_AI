from __future__ import annotations

import pytest

from src.database.graph_store import Neo4jGraphStore
from src.database.retriever import HybridRetriever
from src.database.vector_store import QdrantVectorStore
from src.retrieval.entity_retriever import EntityRetriever


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_qdrant_client_reused() -> None:
    original = QdrantVectorStore._shared_client
    shared = _Closable()
    QdrantVectorStore._shared_client = shared
    try:
        first = QdrantVectorStore()
        second = QdrantVectorStore()

        assert first._client is shared
        assert second._client is shared
    finally:
        QdrantVectorStore._shared_client = original


def test_neo4j_driver_reused() -> None:
    original = Neo4jGraphStore._shared_driver
    shared = _Closable()
    Neo4jGraphStore._shared_driver = shared
    try:
        first = Neo4jGraphStore()
        second = Neo4jGraphStore()

        assert first._driver is shared
        assert second._driver is shared
    finally:
        Neo4jGraphStore._shared_driver = original


@pytest.mark.asyncio
async def test_embedding_failure_cancels_overlapping_entity_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    cancelled = False

    class _EntityRetriever:
        async def retrieve(self, **kwargs):
            nonlocal cancelled
            try:
                await __import__("asyncio").sleep(1)
            except __import__("asyncio").CancelledError:
                cancelled = True
                raise

    async def _failing_embedding(query: str) -> list[float]:
        raise RuntimeError("embedding unavailable")

    retriever = object.__new__(HybridRetriever)
    retriever._entity_retriever = _EntityRetriever()
    monkeypatch.setattr("src.database.retriever.embed_query", _failing_embedding)

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        await retriever.retrieve("benzoyl peroxide là gì?")

    assert cancelled is True


@pytest.mark.asyncio
async def test_shared_entity_payload_cache_avoids_a_second_scroll() -> None:
    original_client = EntityRetriever._shared_client
    original_cache = EntityRetriever._payload_cache
    EntityRetriever._shared_client = _Closable()
    EntityRetriever._payload_cache = {"acne_entities": [{"canonical_name": "adapalene"}]}
    try:
        retriever = EntityRetriever("acne_entities")
        payloads = await retriever._scroll_entity_payloads()

        assert payloads == [{"canonical_name": "adapalene"}]
    finally:
        EntityRetriever._shared_client = original_client
        EntityRetriever._payload_cache = original_cache
