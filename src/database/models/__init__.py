"""Export SQLAlchemy metadata dùng chung cho ``init_schema.py``."""

from src.database.models.base import Base, metadata

__all__ = ["Base", "metadata"]
