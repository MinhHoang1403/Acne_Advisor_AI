"""Repository lưu chat session/message trong PostgreSQL.

Caller cấp ``AsyncSession`` và sở hữu transaction boundary. Raw SQL được bind
parameter; message ID làm idempotency key qua ``ON CONFLICT DO NOTHING``. Hide
chỉ đặt cờ, còn delete toàn bộ là operation tách biệt. Metadata đi qua allow-by-
exclusion sanitizer để tránh lưu credential/raw error phổ biến.
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
    """UPSERT chat session và luôn cập nhật ``updated_at``.

    Khi ``session_id`` đã tồn tại, câu lệnh chỉ bổ sung ``user_id`` còn thiếu;
    title hiện có không bị ghi đè tại repository boundary này.
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
    """Ghi chat message theo idempotency key là primary key ``message_id``.

    ``ON CONFLICT DO NOTHING`` ngăn tạo bản sao khi client gửi lại cùng message
    do retry hoặc refresh.
    """
    if message_id is None:
        message_id = str(uuid.uuid4())

    # Metadata được lọc trước khi JSON serialization; message content giữ nguyên.
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
    """Đọc chat sessions theo ``updated_at`` giảm dần, mặc định bỏ session ẩn."""
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
    """Đọc messages của một session theo ``created_at`` tăng dần."""
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


async def get_recent_messages(
    session: AsyncSession,
    session_id: str,
    limit: int = 10,
) -> list[dict]:
    """Chọn latest N messages rồi trả lại theo thứ tự hội thoại tăng dần."""
    result = await session.execute(
        text("""
            SELECT id, session_id, role, content, sources, metadata, created_at
            FROM (
                SELECT id, session_id, role, content, sources, metadata, created_at
                FROM chat_messages
                WHERE session_id = :session_id
                ORDER BY created_at DESC, id DESC
                LIMIT :limit
            ) AS recent_messages
            ORDER BY created_at ASC, id ASC
        """),
        {"session_id": session_id, "limit": limit},
    )
    return [dict(row) for row in result.mappings().fetchall()]


async def rename_session(
    session: AsyncSession,
    session_id: str,
    title: str,
) -> bool:
    """Đổi tên session; trả ``False`` khi không tìm thấy ID."""
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
    """Đặt ``hidden=true`` mà không xóa dữ liệu; trả ``False`` nếu thiếu ID."""
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
    """Cập nhật timestamp ``updated_at`` của session."""
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
    """Kiểm tra session ID có tồn tại trong PostgreSQL hay không."""
    result = await session.execute(
        text("SELECT 1 FROM chat_sessions WHERE id = :id"),
        {"id": session_id},
    )
    return result.fetchone() is not None


async def get_message_ids_for_session(
    session: AsyncSession,
    session_id: str,
) -> set[str]:
    """Đọc toàn bộ message ID để loại trùng khi đồng bộ một session."""
    result = await session.execute(
        text("SELECT id FROM chat_messages WHERE session_id = :session_id"),
        {"session_id": session_id},
    )
    return {row[0] for row in result.fetchall()}


async def delete_all_chat_history(session: AsyncSession) -> dict[str, int]:
    """Chỉ xóa lịch sử chat đã persist.

    Hàm xóa rows trong ``chat_messages`` và ``chat_sessions`` nhưng không chạm
    schema object hay knowledge store đã index.
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
# Helpers tại SQLAlchemy/asyncpg JSONB boundary.
# ---------------------------------------------------------------------------

def _json_or_none(value: Any) -> Any:
    """Đổi dict/list khác rỗng thành JSON string phù hợp JSONB, còn lại giữ nguyên."""
    if value is None:
        return None
    if isinstance(value, (list, dict)) and len(value) == 0:
        return None
    # sqlalchemy.text + asyncpg cần JSON string rõ ràng cho dict/list parameters.
    return json.dumps(value, ensure_ascii=False)


# Các key nhạy cảm bị loại trước khi metadata được ghi.
_SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "api_secret", "secret", "token",
    "password", "credential", "authorization", "auth",
    "google_api_key", "llama_cloud_api_key",
    "exception", "traceback", "stack_trace", "raw_error",
})


def _sanitize_metadata(meta: dict) -> dict:
    """Loại field nhạy cảm khỏi metadata trước khi lưu."""
    if not isinstance(meta, dict):
        return {}
    return {
        k: v for k, v in meta.items()
        if k.lower() not in _SENSITIVE_KEYS
    }
