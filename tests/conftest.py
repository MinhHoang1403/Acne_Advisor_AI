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
os.environ["RERANKER_ENABLED"] = "false"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
