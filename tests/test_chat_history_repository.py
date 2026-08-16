from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from src.database.repositories.chat_history import get_messages, get_recent_messages


async def _session_with_messages(count: int) -> tuple[AsyncSession, object]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.execute(text("""
            CREATE TABLE chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                sources TEXT,
                metadata TEXT,
                created_at TIMESTAMP
            )
        """))
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for index in range(count):
            await connection.execute(
                text("""
                    INSERT INTO chat_messages
                        (id, session_id, role, content, sources, metadata, created_at)
                    VALUES (:id, 'session-a', :role, :content, NULL, NULL, :created_at)
                """),
                {
                    "id": f"message-{index:02d}",
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": f"content-{index}",
                    "created_at": start + timedelta(seconds=index),
                },
            )
    return AsyncSession(engine), engine


@pytest.mark.asyncio
@pytest.mark.parametrize(("count", "expected"), [(3, [0, 1, 2]), (5, [0, 1, 2, 3, 4]), (7, [2, 3, 4, 5, 6])])
async def test_get_recent_messages_selects_latest_n_in_chronological_order(
    count: int,
    expected: list[int],
) -> None:
    session, engine = await _session_with_messages(count)
    try:
        rows = await get_recent_messages(session, "session-a", limit=5)
    finally:
        await session.close()
        await engine.dispose()

    assert [row["id"] for row in rows] == [f"message-{index:02d}" for index in expected]
    assert [row["content"] for row in rows] == [f"content-{index}" for index in expected]
    assert [row["role"] for row in rows] == [
        "user" if index % 2 == 0 else "assistant" for index in expected
    ]


@pytest.mark.asyncio
async def test_recent_message_ties_use_id_for_stable_selection_and_order() -> None:
    session, engine = await _session_with_messages(0)
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    try:
        for message_id in ("a", "b", "c"):
            await session.execute(
                text("""
                    INSERT INTO chat_messages
                        (id, session_id, role, content, sources, metadata, created_at)
                    VALUES (:id, 'session-a', 'user', :id, NULL, NULL, :created_at)
                """),
                {"id": message_id, "created_at": timestamp},
            )
        await session.commit()
        rows = await get_recent_messages(session, "session-a", limit=2)
    finally:
        await session.close()
        await engine.dispose()

    assert [row["id"] for row in rows] == ["b", "c"]


@pytest.mark.asyncio
async def test_complete_history_reader_keeps_earliest_first_semantics() -> None:
    session, engine = await _session_with_messages(7)
    try:
        rows = await get_messages(session, "session-a", limit=5)
    finally:
        await session.close()
        await engine.dispose()

    assert [row["id"] for row in rows] == [f"message-{index:02d}" for index in range(5)]
