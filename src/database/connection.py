"""Khởi tạo SQLAlchemy engine và factory cho PostgreSQL sessions.

Connection URL đến từ ``DATABASE_URL``. Long-running API dùng connection pool;
test/script có thể bật ``DB_USE_NULL_POOL``. ``get_session`` sở hữu transaction:
commit khi dependency hoàn tất, rollback khi exception và đóng session khi thoát.
Module không chứa query nghiệp vụ; query chat nằm trong repositories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncGenerator

try:
    from dotenv import load_dotenv

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://user:password@localhost:5433/acne_agent_db",
)

# ---------------------------------------------------------------------------
# NullPool dành cho process ngắn/test; service dài hạn tái sử dụng connection pool.
# ---------------------------------------------------------------------------
_use_null_pool = os.getenv("DB_USE_NULL_POOL", "false").lower() == "true"

engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG",
    pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
    pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
    poolclass=NullPool if _use_null_pool else None,
)

# ---------------------------------------------------------------------------
# Session không expire object sau commit để API vẫn đọc được dữ liệu đã ghi.
# ---------------------------------------------------------------------------
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield một session và sở hữu commit/rollback cho FastAPI dependency."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
