"""Knowledge-base versioning and embedding metadata helpers."""

from __future__ import annotations

import os
from typing import Any


DEFAULT_EMBEDDING_PROVIDER = "google"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSIONS = 3072

DEFAULT_KB_VERSION = "frozen_phase1_build"
DEFAULT_TAXONOMY_VERSION = "acne_taxonomy_2026_08"
DEFAULT_ENTITY_SCHEMA_VERSION = "source_backed_entity_card"
DEFAULT_CHUNK_SCHEMA_VERSION = "structure_first_chars_2400_no_overlap"
DEFAULT_INGESTION_PIPELINE_VERSION = "frozen_phase1_build"


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
    """Return serializable embedding config metadata for KB payloads."""

    return {
        "embedding_provider": _env_str("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER),
        "embedding_model": _env_str("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "embedding_dimensions": _env_int("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS),
    }


def get_knowledge_versions() -> dict[str, str]:
    """Return version tags used to validate clean KB rebuilds."""

    return {
        "kb_version": _env_str("KB_VERSION", DEFAULT_KB_VERSION),
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


def expected_kb_payload_metadata() -> dict[str, Any]:
    """Return the combined metadata expected on chunk/entity payloads."""

    return {
        **get_embedding_metadata(),
        **get_knowledge_versions(),
    }


def validate_embedding_config_compatibility(
    chunk_config: dict[str, Any],
    entity_config: dict[str, Any],
) -> list[str]:
    """Return mismatch messages for chunk/entity collection compatibility checks."""

    issues: list[str] = []
    for field in (
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "kb_version",
    ):
        chunk_value = chunk_config.get(field)
        entity_value = entity_config.get(field)
        if _normalize_value(chunk_value) != _normalize_value(entity_value):
            issues.append(
                f"{field} mismatch: chunk={chunk_value!r}, entity={entity_value!r}"
            )
    return issues


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return value


__all__ = [
    "DEFAULT_CHUNK_SCHEMA_VERSION",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_PROVIDER",
    "DEFAULT_ENTITY_SCHEMA_VERSION",
    "DEFAULT_INGESTION_PIPELINE_VERSION",
    "DEFAULT_KB_VERSION",
    "DEFAULT_TAXONOMY_VERSION",
    "expected_kb_payload_metadata",
    "get_embedding_metadata",
    "get_knowledge_versions",
    "validate_embedding_config_compatibility",
]
