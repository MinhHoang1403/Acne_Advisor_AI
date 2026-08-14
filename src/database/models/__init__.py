"""
src/database/models/__init__.py
================================
Export the shared SQLAlchemy metadata used by init_schema.py.
"""

from src.database.models.base import Base, metadata

__all__ = ["Base", "metadata"]
