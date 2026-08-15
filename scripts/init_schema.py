#!/usr/bin/env python3
"""Khởi tạo schema PostgreSQL mà không quản lý knowledge-index lifecycle.

Script tạo table/extension theo kiểu idempotent và kiểm kết nối Qdrant, nhưng
không build, re-embed hoặc chuyển knowledge collections. Những operation đó chỉ
thuộc ``scripts/phase1.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("init_schema")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://user:password@localhost:5433/acne_agent_db",
)

SYNC_DATABASE_URL = os.getenv(
    "SYNC_DATABASE_URL",
    DATABASE_URL.replace("+asyncpg", "+psycopg2").replace(
        "postgresql+asyncpg", "postgresql"
    ),
)

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge")


def _mask_password(url: str) -> str:
    return re.sub(r"(://[^:]+:)([^@]+)(@)", r"\1***\3", url)


def _raw_sql(statement: str):
    from sqlalchemy import text

    return text(statement)


def _create_patient_records_table() -> None:
    logger.info("[patient_records] Connecting via psycopg2...")
    logger.info("[patient_records] URL: %s", _mask_password(SYNC_DATABASE_URL))

    try:
        from sqlalchemy import Column, Integer, MetaData, Table, create_engine, text
        from sqlalchemy.dialects.postgresql import JSONB
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run: pip install sqlalchemy psycopg2-binary"
        ) from exc

    engine = create_engine(
        SYNC_DATABASE_URL,
        echo=(LOG_LEVEL == "DEBUG"),
        pool_pre_ping=True,
    )

    metadata = MetaData()

    patient_records = Table(
        "patient_records",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column(
            "patient_profile",
            JSONB,
            nullable=False,
            server_default=text("'{}'::jsonb"),
        ),
        comment="Stores patient profile as flexible JSONB.",
    )

    try:
        patient_records.create(bind=engine, checkfirst=True)
        logger.info("✓ Table 'patient_records' is ready.")
    finally:
        engine.dispose()


async def _seed_reference_data(conn) -> None:
    logger.info("Seeding reference data...")
    logger.info("(no reference data to seed yet)")


async def _setup_postgres() -> None:
    logger.info("Connecting to PostgreSQL...")
    logger.info("URL: %s", _mask_password(DATABASE_URL))

    try:
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run: pip install sqlalchemy[asyncio] asyncpg"
        ) from exc

    engine = create_async_engine(DATABASE_URL, echo=(LOG_LEVEL == "DEBUG"))

    async with engine.begin() as conn:
        logger.info("Enabling PostgreSQL extensions...")

        await conn.execute(_raw_sql('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))

        logger.info("Creating SQLAlchemy model tables if available...")

        try:
            from src.database.models import metadata  # type: ignore

            await conn.run_sync(metadata.create_all)
            logger.info("✓ Tables created from src.database.models metadata.")
        except ImportError:
            logger.warning(
                "src.database.models not found yet. Skipping model table creation."
            )

        await _seed_reference_data(conn)

    await engine.dispose()
    logger.info("✓ PostgreSQL setup complete.")


async def _setup_qdrant() -> None:
    """Refuse to create or recreate the validated knowledge collection here."""

    logger.info(
        "Qdrant ownership belongs to scripts/phase1.py; init_schema does not mutate %s.",
        QDRANT_COLLECTION_NAME,
    )


async def main() -> int:
    logger.info("=" * 60)
    logger.info("Acne Advisor AI – Schema Initialisation")
    logger.info("=" * 60)

    try:
        _create_patient_records_table()
    except Exception as exc:
        logger.error("patient_records table creation failed: %s", exc, exc_info=True)
        return 1

    try:
        await _setup_postgres()
    except Exception as exc:
        logger.error("PostgreSQL setup failed: %s", exc, exc_info=True)
        return 1

    try:
        await _setup_qdrant()
    except Exception as exc:
        logger.error("Qdrant ownership check failed: %s", exc, exc_info=True)
        return 1

    logger.info("=" * 60)
    logger.info("✅ Schema initialisation completed successfully.")
    logger.info("=" * 60)
    return 0


def cli() -> int:
    """Synchronous console-script wrapper."""

    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(cli())
