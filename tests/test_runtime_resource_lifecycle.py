from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.agent.nodes.preparation import prepare_request_node
from src.cache import redis_cache
from src.integrations import google_genai


def test_null_pool_environment_imports_without_queue_pool_options():
    result = subprocess.run(
        [sys.executable, "-c", "from src.database.connection import engine; from sqlalchemy.pool import NullPool; assert isinstance(engine.pool, NullPool)"],
        env={**os.environ, "DB_USE_NULL_POOL": "true"},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_zero_history_budget_really_disables_history(monkeypatch):
    monkeypatch.setenv("MAX_CONVERSATION_HISTORY_MESSAGES", "0")
    result = await prepare_request_node({"user_question": "Question", "conversation_history": [
        {"role": "user", "content": "Previous private context"},
    ]})
    assert result["conversation_context"] == {"messages": [], "message_count": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ConnectionError, asyncio.CancelledError])
async def test_redis_failed_initialization_closes_unpublished_client(monkeypatch, failure):
    closed = []

    async def ping():
        raise failure()

    async def close():
        closed.append(True)

    client = SimpleNamespace(ping=ping, aclose=close, close=close)
    monkeypatch.setenv("CACHE_ENABLED", "true")
    monkeypatch.setattr(redis_cache, "_redis_client", None)
    monkeypatch.setattr(redis_cache.redis, "from_url", lambda *a, **kw: client)
    if failure is asyncio.CancelledError:
        with pytest.raises(asyncio.CancelledError):
            await redis_cache.get_redis()
    else:
        assert await redis_cache.get_redis() is None
    assert closed == [True]
    assert redis_cache._redis_client is None


@pytest.mark.asyncio
async def test_google_owned_async_client_closed_on_cancellation(monkeypatch):
    closed = []

    async def generate(**kwargs):
        raise asyncio.CancelledError()

    async def close():
        closed.append(True)

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate), aclose=close))
    monkeypatch.setattr(google_genai, "build_google_genai_client", lambda **kw: client)
    with pytest.raises(asyncio.CancelledError):
        await google_genai.generate_text_async(prompt="x", system_prompt=None, model_name="test", temperature=0)
    assert closed == [True]


@pytest.mark.asyncio
async def test_concurrent_redis_initialization_publishes_only_a_ready_singleton(monkeypatch):
    ready, release = asyncio.Event(), asyncio.Event()
    clients = []

    async def ping():
        ready.set()
        await release.wait()
        return True

    def create(*args, **kwargs):
        client = SimpleNamespace(ping=ping)
        clients.append(client)
        return client

    monkeypatch.setenv("CACHE_ENABLED", "true")
    monkeypatch.setattr(redis_cache, "_redis_client", None)
    monkeypatch.setattr(redis_cache, "_redis_lock", asyncio.Lock())
    monkeypatch.setattr(redis_cache.redis, "from_url", create)
    first = asyncio.create_task(redis_cache.get_redis())
    await ready.wait()
    second = asyncio.create_task(redis_cache.get_redis())
    await asyncio.sleep(0)
    assert not second.done()
    release.set()
    assert await first is await second
    assert len(clients) == 1


@pytest.mark.asyncio
async def test_api_shutdown_closes_all_shared_connections_even_if_one_fails(monkeypatch):
    from src.api.app import shutdown_runtime_connections
    from src.database.connection import engine
    from src.database.vector_store import QdrantVectorStore
    closed = []

    async def broken_redis():
        closed.append("redis")
        raise RuntimeError("close failure")

    async def qdrant():
        closed.append("qdrant")

    async def postgres():
        closed.append("postgres")

    monkeypatch.setattr(redis_cache, "close_redis", broken_redis)
    monkeypatch.setattr(QdrantVectorStore, "close_shared_client", qdrant)
    monkeypatch.setattr(engine, "dispose", postgres)
    with pytest.raises(RuntimeError):
        await shutdown_runtime_connections()
    assert closed == ["redis", "qdrant", "postgres"]


@pytest.mark.parametrize("operation", ["generation", "embedding"])
def test_google_owned_sync_client_closed_on_failure(monkeypatch, operation):
    closed = []

    def fail(**kwargs):
        raise ValueError("Invalid response")

    client = SimpleNamespace(models=SimpleNamespace(generate_content=fail, embed_content=fail), close=lambda: closed.append(True))
    monkeypatch.setattr(google_genai, "build_google_genai_client", lambda **kw: client)
    with pytest.raises(ValueError):
        if operation == "generation":
            google_genai.generate_text_sync(prompt="x", system_prompt=None, model_name="test", temperature=0)
        else:
            google_genai.embed_texts_sync(["x"], model_name="test", task_type=None, expected_dimensions=3)
    assert closed == [True]
