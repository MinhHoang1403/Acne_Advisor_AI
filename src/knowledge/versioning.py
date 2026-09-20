"""Helper quản lý knowledge-base version và embedding metadata."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_EMBEDDING_PROVIDER = "google"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSIONS = 3072

DEFAULT_TAXONOMY_VERSION = "acne_taxonomy_2026_08"
DEFAULT_ENTITY_SCHEMA_VERSION = "source_backed_entity_card"
DEFAULT_CHUNK_SCHEMA_VERSION = (
    "structure_preserving_blocks_and_lead_ins_chars_2400_no_overlap"
)
DEFAULT_INGESTION_PIPELINE_VERSION = "frozen_phase1_build"
DEFAULT_ACTIVE_KNOWLEDGE_MANIFEST = (
    Path(__file__).resolve().parents[2] / "data" / "knowledge_build_manifest.json"
)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_embedding_metadata() -> dict[str, Any]:
    """Trả embedding config có thể serialize vào KB payload."""

    return {
        "embedding_provider": _env_str("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER),
        "embedding_model": _env_str("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "embedding_dimensions": _env_int("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS),
    }


def resolve_active_knowledge_build_id(manifest_path: Path | None = None) -> str:
    """Return the activated manifest build ID or fail closed.

    The persisted manifest is the runtime authority. ``KB_VERSION`` remains a
    legacy deployment setting, but it cannot override the activated build and
    therefore cannot silently reuse a stale cache namespace.
    """

    # Import lazily to avoid loading the ingestion graph while this package is
    # still initializing.
    from src.ingestion.manifest import load_build_manifest, validate_build_id

    path = manifest_path or DEFAULT_ACTIVE_KNOWLEDGE_MANIFEST
    manifest = load_build_manifest(path)
    if manifest.get("status") != "activated":
        raise ValueError(f"Knowledge-build manifest at {path} is not activated")

    build_id = validate_build_id(
        manifest.get("build_id"),
        setting_name="knowledge manifest build_id",
    )
    collections = manifest.get("collections") or {}
    expected = {
        "knowledge_logical": "acne_knowledge",
        "knowledge_physical": f"acne_knowledge__{build_id}",
        "entity_logical": "acne_entities",
        "entity_physical": f"acne_entities__{build_id}",
    }
    if any(collections.get(name) != value for name, value in expected.items()):
        raise ValueError(
            "Activated knowledge manifest collection identity does not match build_id"
        )
    return build_id


def get_knowledge_versions(
    *,
    manifest_path: Path | None = None,
    build_id: str | None = None,
) -> dict[str, str]:
    """Trả version tags dùng để kiểm tính tương thích của KB build."""

    return {
        "kb_version": build_id or resolve_active_knowledge_build_id(manifest_path),
        "taxonomy_version": _env_str("TAXONOMY_VERSION", DEFAULT_TAXONOMY_VERSION),
        "entity_schema_version": _env_str(
            "ENTITY_SCHEMA_VERSION",
            DEFAULT_ENTITY_SCHEMA_VERSION,
        ),
        "chunk_schema_version": _env_str("CHUNK_SCHEMA_VERSION", DEFAULT_CHUNK_SCHEMA_VERSION),
        "ingestion_pipeline_version": _env_str(
            "INGESTION_PIPELINE_VERSION",
            DEFAULT_INGESTION_PIPELINE_VERSION,
        ),
    }


__all__ = [
    "DEFAULT_CHUNK_SCHEMA_VERSION",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_PROVIDER",
    "DEFAULT_ENTITY_SCHEMA_VERSION",
    "DEFAULT_INGESTION_PIPELINE_VERSION",
    "DEFAULT_TAXONOMY_VERSION",
    "get_embedding_metadata",
    "get_knowledge_versions",
    "resolve_active_knowledge_build_id",
]
