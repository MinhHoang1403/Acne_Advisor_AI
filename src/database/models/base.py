"""SQLAlchemy declarative base và column mixins dùng chung."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base dùng chung cho mọi ORM model."""
    pass


# Export MetaData để schema initializer dùng cùng model registry.
metadata = Base.metadata


# ---------------------------------------------------------------------------
# Column mixins có thể tái sử dụng.
# ---------------------------------------------------------------------------

class UUIDPrimaryKeyMixin:
    """Thêm UUID primary key, mặc định do server sinh."""
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
        default=uuid.uuid4,
    )


class TimestampMixin:
    """Thêm hai timestamp columns ``created_at`` và ``updated_at``."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
