from __future__ import annotations

import logging
import json
import unicodedata
from types import SimpleNamespace

import pytest

from src.cache import exact_cache, redis_cache
from src.database.vector_store import QdrantVectorStore
from src.retrieval.service import EvidenceRetriever


def test_cache_question_unicode_forms_preserve_negation():
    question = "Tôi không dùng thuốc này"
    assert exact_cache.normalize_question(question) == exact_cache.normalize_question(unicodedata.normalize("NFD", question))
    assert exact_cache.normalize_question(question) != exact_cache.normalize_question("Tôi dùng thuốc này")


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"answer": "Text", "metadata": ["invalid"], "sources": ["source"]},
    {"answer": 4, "metadata": {}, "sources": ["source"]},
    {"answer": "Text", "metadata": {}, "sources": "source"},
])
async def test_malformed_exact_cache_payload_is_a_miss(monkeypatch, payload):
    async def get(_key):
        return json.dumps(payload)

    async def client():
        return SimpleNamespace(get=get)

    monkeypatch.setattr(exact_cache, "get_redis", client)
    assert await exact_cache.get_exact_cache("question", provider="gemini", model="test", pipeline_fingerprint="test") is None


@pytest.mark.asyncio
async def test_redis_connection_error_log_is_sanitized(monkeypatch, caplog):
    class FakeRedisModule:
        @staticmethod
        def from_url(url, decode_responses=True):
            del url, decode_responses

            class Client:
                async def ping(self):
                    raise RuntimeError("redis failed token=secret-value")

            return Client()

    monkeypatch.setattr(redis_cache, "REDIS_AVAILABLE", True)
    monkeypatch.setattr(redis_cache, "redis", FakeRedisModule, raising=False)
    monkeypatch.setattr(redis_cache, "_redis_client", None)

    with caplog.at_level(logging.WARNING):
        assert await redis_cache.get_redis() is None

    text = caplog.text
    assert "secret-value" not in text
    assert "token=[REDACTED]" in text


@pytest.mark.asyncio
async def test_exact_cache_store_error_log_is_sanitized(monkeypatch, caplog):
    class FakeRedis:
        async def setex(self, key, ttl, value):
            del key, ttl, value
            raise RuntimeError("cache write failed password=hidden-value")

    async def fake_get_redis():
        return FakeRedis()

    monkeypatch.setattr(exact_cache, "get_redis", fake_get_redis)

    with caplog.at_level(logging.WARNING):
        await exact_cache.set_answer_cache(
            normalized_question="mụn đầu đen là gì",
            answer="Mụn đầu đen là dạng nhân mụn mở.",
            sources=[],
            metadata={},
            provider="gemini",
            model="gemini-test",
            pipeline_fingerprint="fp",
        )

    text = caplog.text
    assert "hidden-value" not in text
    assert "password=[REDACTED]" in text


@pytest.mark.asyncio
async def test_qdrant_vector_store_search_handles_missing_payload():
    class FakeClient:
        async def query_points(self, **kwargs):
            del kwargs
            return SimpleNamespace(points=[SimpleNamespace(id="point-1", score=0.42, payload=None)])

    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._client = FakeClient()
    store._collection = "acne_knowledge"

    result = await store.search([0.0, 0.1], top_k=1)

    assert result == [{"id": "point-1", "score": 0.42}]


@pytest.mark.asyncio
async def test_qdrant_vector_store_search_passes_filter_to_client():
    captured = {}

    class FakeClient:
        async def query_points(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(points=[])

    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._client = FakeClient()
    store._collection = "acne_knowledge"

    await store.search(
        [0.0, 0.1],
        top_k=1,
        filter={"must": [{"key": "document_id", "match": {"value": "doc-1"}}]},
    )

    assert captured["query_filter"].must[0].key == "document_id"
    assert captured["query_filter"].must[0].match.value == "doc-1"


@pytest.mark.asyncio
async def test_evidence_retriever_close_logs_sanitized_errors(caplog):
    class BrokenComponent:
        async def close(self):
            raise RuntimeError("close failed api_key=secret-value")

    retriever = EvidenceRetriever.__new__(EvidenceRetriever)
    retriever._vector_store = BrokenComponent()
    with caplog.at_level(logging.WARNING):
        await retriever.close()

    text = caplog.text
    assert "secret-value" not in text
    assert "api_key=[REDACTED]" in text
