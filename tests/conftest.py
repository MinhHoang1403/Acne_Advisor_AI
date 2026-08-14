"""
tests/conftest.py – Shared pytest fixtures
==========================================
"""

from __future__ import annotations

import os

import pytest

# Force test environment
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5433/acne_agent_db")


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
