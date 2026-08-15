"""
src/database/repositories/chat_history.py
==========================================
Repository for chat session and message persistence in PostgreSQL.

All functions accept an AsyncSession (dependency injection) and use raw SQL
via sqlalchemy.text() for maximum clarity and control.

Safety:
- hide_session only sets hidden=true, never DELETEs.
- save_message uses INSERT ... ON CONFLICT DO NOTHING to prevent duplicates.
- create_or_update_session uses UPSERT (ON CONFLICT DO UPDATE).
- No API keys, raw exceptions, or sensitive data is stored.
"""

from __future__ import annotations

import logging
import uuid
import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def create_or_update_session(
    session: AsyncSession,
    session_id: str,
    title: str,
    user_id: Optional[str] = None,
    hidden: bool = False,
    metadata: Optional[dict] = None,
) -> dict:
    """
    UPSERT a chat session.
    If session_id already exists, update title only if the DB title is the default.
    Always updates updated_at.
    """
    result = await session.execute(
        text("""
            INSERT INTO chat_sessions (id, user_id, title, hidden, metadata, created_at, updated_at)
            VALUES (:id, :user_id, :title, :hidden, :metadata, now(), now())
            ON CONFLICT (id) DO UPDATE SET
                updated_at = now(),
                user_id = COALESCE(chat_sessions.user_id, EXCLUDED.user_id)
            RETURNING id, user_id, title, created_at, updated_at, hidden
        """),
        {
            "id": session_id,
            "user_id": user_id,
            "title": title,
            "hidden": hidden,
            "metadata": json.dumps(metadata, ensure_ascii=False) if metadata else None,
        },
    )
    row = result.mappings().fetchone()
    return dict(row) if row else {}


async def save_message(
    session: AsyncSession,
    session_id: str,
    role: str,
    content: str,
    message_id: Optional[str] = None,
    sources: Optional[list] = None,
    metadata: Optional[dict] = None,
    created_at: Optional[datetime] = None,
) -> dict:
    """
    Insert a chat message. Uses ON CONFLICT DO NOTHING on the primary key
    to prevent duplicates when the same message_id is sent again (retry/refresh).
    """
    if message_id is None:
        message_id = str(uuid.uuid4())

    # Sanitize metadata — strip any sensitive fields
    safe_metadata = _sanitize_metadata(metadata) if metadata else None

    ts = created_at or datetime.now(timezone.utc)

    result = await session.execute(
        text("""
            INSERT INTO chat_messages
                (id, session_id, role, content, sources, metadata, created_at)
            VALUES
                (:id, :session_id, :role, :content, :sources, :metadata, :created_at)
            ON CONFLICT (id) DO NOTHING
            RETURNING id, session_id, role, content, created_at
        """),
        {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "sources": _json_or_none(sources),
            "metadata": _json_or_none(safe_metadata),
            "created_at": ts,
        },
    )
    row = result.mappings().fetchone()
    return dict(row) if row else {"id": message_id, "duplicate": True}


async def get_sessions(
    session: AsyncSession,
    user_id: Optional[str] = None,
    include_hidden: bool = False,
) -> list[dict]:
    """
    Get chat sessions sorted by updated_at DESC.
    By default, only returns non-hidden sessions.
    """
    conditions = []
    params: dict[str, Any] = {}

    if not include_hidden:
        conditions.append("hidden = false")

    if user_id is not None:
        conditions.append("user_id = :user_id")
        params["user_id"] = user_id

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    result = await session.execute(
        text(f"""
            SELECT id, user_id, title, created_at, updated_at, hidden
            FROM chat_sessions
            {where_clause}
            ORDER BY updated_at DESC
        """),
        params,
    )
    return [dict(row) for row in result.mappings().fetchall()]


async def get_messages(
    session: AsyncSession,
    session_id: str,
    limit: int = 50,
) -> list[dict]:
    """
    Get messages for a session, sorted by created_at ASC.
    """
    result = await session.execute(
        text("""
            SELECT id, session_id, role, content, sources, metadata, created_at
            FROM chat_messages
            WHERE session_id = :session_id
            ORDER BY created_at ASC
            LIMIT :limit
        """),
        {"session_id": session_id, "limit": limit},
    )
    return [dict(row) for row in result.mappings().fetchall()]


async def rename_session(
    session: AsyncSession,
    session_id: str,
    title: str,
) -> bool:
    """Rename a session. Returns True if updated, False if not found."""
    result = await session.execute(
        text("""
            UPDATE chat_sessions
            SET title = :title, updated_at = now()
            WHERE id = :id
            RETURNING id
        """),
        {"id": session_id, "title": title},
    )
    return result.fetchone() is not None


async def hide_session(
    session: AsyncSession,
    session_id: str,
) -> bool:
    """
    Hide a session by setting hidden=true. Does NOT delete any data.
    Returns True if updated, False if not found.
    """
    result = await session.execute(
        text("""
            UPDATE chat_sessions
            SET hidden = true, updated_at = now()
            WHERE id = :id
            RETURNING id
        """),
        {"id": session_id},
    )
    return result.fetchone() is not None


async def touch_session(
    session: AsyncSession,
    session_id: str,
) -> bool:
    """Update the updated_at timestamp of a session."""
    result = await session.execute(
        text("""
            UPDATE chat_sessions
            SET updated_at = now()
            WHERE id = :id
            RETURNING id
        """),
        {"id": session_id},
    )
    return result.fetchone() is not None


async def session_exists(
    session: AsyncSession,
    session_id: str,
) -> bool:
    """Check if a session exists in the database."""
    result = await session.execute(
        text("SELECT 1 FROM chat_sessions WHERE id = :id"),
        {"id": session_id},
    )
    return result.fetchone() is not None


async def get_message_ids_for_session(
    session: AsyncSession,
    session_id: str,
) -> set[str]:
    """Get all message IDs for a session (used for dedup during sync)."""
    result = await session.execute(
        text("SELECT id FROM chat_messages WHERE session_id = :session_id"),
        {"session_id": session_id},
    )
    return {row[0] for row in result.fetchall()}


async def delete_all_chat_history(session: AsyncSession) -> dict[str, int]:
    """
    Delete persisted chat history only.

    This removes rows from chat_messages and chat_sessions, but does not touch
    any schema objects or indexed knowledge stores.
    """
    messages_result = await session.execute(
        text("DELETE FROM chat_messages")
    )
    sessions_result = await session.execute(
        text("DELETE FROM chat_sessions")
    )
    return {
        "deleted_messages": int(messages_result.rowcount or 0),
        "deleted_sessions": int(sessions_result.rowcount or 0),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_or_none(value: Any) -> Any:
    """Return the value as a JSON string if truthy, else None (for JSONB columns)."""
    if value is None:
        return None
    if isinstance(value, (list, dict)) and len(value) == 0:
        return None
    # For asyncpg + sqlalchemy.text(), we must serialize dicts/lists to JSON strings
    return json.dumps(value, ensure_ascii=False)


# Keys that must NEVER be stored in metadata
_SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "api_secret", "secret", "token",
    "password", "credential", "authorization", "auth",
    "google_api_key", "llama_cloud_api_key",
    "exception", "traceback", "stack_trace", "raw_error",
})


def _sanitize_metadata(meta: dict) -> dict:
    """Remove sensitive fields from metadata before storing."""
    if not isinstance(meta, dict):
        return {}
    return {
        k: v for k, v in meta.items()
        if k.lower() not in _SENSITIVE_KEYS
    }
