from __future__ import annotations

import importlib
import asyncio
import uuid
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

api = importlib.import_module("src.api.app")


@pytest.fixture
async def persisted_api(monkeypatch, tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'chat.sqlite'}")

    @event.listens_for(engine.sync_engine, "connect")
    def setup(connection, _record):
        connection.create_function("now", 0, lambda: datetime.now(timezone.utc).isoformat())

    async with engine.begin() as connection:
        await connection.execute(text("CREATE TABLE chat_sessions (id TEXT PRIMARY KEY, user_id TEXT, title TEXT, hidden BOOLEAN, metadata TEXT, created_at TEXT, updated_at TEXT)"))
        await connection.execute(text("CREATE TABLE chat_messages (id TEXT PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, sources TEXT, metadata TEXT, created_at TEXT)"))
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def get_session():
        return sessions()

    calls = []

    async def agent(**kwargs):
        calls.append(kwargs)
        return {"answer": f"Answer {len(calls)}", "actual_provider": "system", "sources": [], "is_in_domain": False}

    monkeypatch.setattr(api, "_get_db_session", get_session)
    monkeypatch.setattr(api, "run_clinical_agent", agent)
    monkeypatch.setenv("RELEASE_READINESS_TEST_MODE", "")
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "false")
    try:
        async with AsyncClient(transport=ASGITransport(app=api.app), base_url="http://test") as client:
            yield client, calls, sessions
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_abstention_does_not_present_related_retrieval_sources(persisted_api, monkeypatch):
    client, _, sessions = persisted_api

    async def abstain(**kwargs):
        return {
            "answer": "Insufficient evidence.", "actual_provider": "system",
            "sources": [], "is_in_domain": True, "fallback_applied": True,
            "fallback_reason_code": "insufficient_evidence", "generation_invoked": False,
            "vector_contexts": [{"source_id": "synthetic-related", "text": "Related only."}],
        }

    monkeypatch.setattr(api, "run_clinical_agent", abstain)
    response = await client.post("/chat", json={"session_id": "gap", "message": "A question"})
    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert response.json()["source_metadata"] == []
    async with sessions() as session:
        stored = (await session.execute(text(
            "SELECT sources FROM chat_messages WHERE session_id='gap' AND role='assistant'"
        ))).scalar_one()
    assert stored is None


@pytest.mark.asyncio
async def test_completed_request_replays_the_original_turn_without_generation(persisted_api):
    client, calls, sessions = persisted_api
    payload = {"request_id": str(uuid.uuid4()), "session_id": "one", "message": "A question"}
    first = await client.post("/chat", json=payload)
    second = await client.post("/chat", json=payload)
    assert first.status_code == second.status_code == 200
    assert second.json()["answer"] == first.json()["answer"]
    assert len(calls) == 1
    async with sessions() as session:
        assert (await session.execute(text("SELECT count(*) FROM chat_messages"))).scalar_one() == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [{"message": "A different question"}, {"session_id": "another"}])
async def test_request_identity_cannot_be_reused_for_another_input(persisted_api, changed):
    client, calls, _ = persisted_api
    payload = {"request_id": str(uuid.uuid4()), "session_id": "one", "message": "A question"}
    assert (await client.post("/chat", json=payload)).status_code == 200
    conflict = await client.post("/chat", json={**payload, **changed})
    assert conflict.status_code == 409
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_inflight_request_id_is_locked_across_sessions(persisted_api, monkeypatch):
    client, _, _ = persisted_api
    started, release = asyncio.Event(), asyncio.Event()

    async def blocked_agent(**kwargs):
        started.set()
        await release.wait()
        return {"answer": "One answer", "actual_provider": "system", "is_in_domain": False}

    monkeypatch.setattr(api, "run_clinical_agent", blocked_agent)
    payload = {"request_id": str(uuid.uuid4()), "session_id": "one", "message": "A question"}
    task = asyncio.create_task(client.post("/chat", json=payload))
    await asyncio.wait_for(started.wait(), 5)
    try:
        conflict = await client.post("/chat", json={**payload, "session_id": "two"})
        assert conflict.status_code == 409
    finally:
        release.set()
        assert (await task).status_code == 200
    assert payload["request_id"] not in api.active_request_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["history", "persistence"])
async def test_stalled_optional_database_work_cannot_stall_chat(persisted_api, monkeypatch, stage):
    client, calls, sessions = persisted_api
    from src.database.repositories import chat_history as repo

    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setenv("CHAT_DB_TIMEOUT_SECONDS", "0.02")
    monkeypatch.setattr(repo, "get_recent_messages" if stage == "history" else "save_message", stalled)
    response = await asyncio.wait_for(client.post("/chat", json={"session_id": "one", "message": "A question"}), 0.5)
    assert response.status_code == 200
    assert len(calls) == 1
    if stage == "persistence":
        async with sessions() as session:
            assert (await session.execute(text("SELECT count(*) FROM chat_messages"))).scalar_one() == 0
