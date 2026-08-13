"""Deterministic path-independent Phase 1 provenance identities."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

from src.ingestion.source_manifest import CanonicalSource


PROVENANCE_CONTRACT_ID = "portable_content_bound_provenance"
_CHUNK_NAMESPACE = uuid.UUID("c957660c-5c3f-5bca-8baf-67d9bb4273c8")

REQUIRED_PROVENANCE_FIELDS = frozenset(
    {
        "source_id", "source_title", "source_authority", "source_type",
        "source_url", "source_version_date", "source_content_hash", "document_id",
        "record_id", "page_start", "page_end", "section_path", "chunk_index",
        "chunk_id", "chunk_content_hash", "parser_contract_id",
        "normalization_contract_id", "chunk_contract_id", "filter_contract_id",
        "embedding_contract_id", "bm25_contract_id", "build_id", "ingested_at",
    }
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_id(source_id: str, source_content_hash: str) -> str:
    return f"doc_{sha256_text(source_id + chr(0) + source_content_hash)[:32]}"


def record_id(document_identity: str, portable_locator: str) -> str:
    return f"rec_{sha256_text(document_identity + chr(0) + portable_locator)[:32]}"


def chunk_id(
    document_identity: str,
    record_identity: str,
    section_path: tuple[str, ...],
    chunk_index: int,
    content_hash: str,
) -> str:
    raw = "\0".join(
        [document_identity, record_identity, "/".join(section_path), str(chunk_index), content_hash]
    )
    return str(uuid.uuid5(_CHUNK_NAMESPACE, raw))


def base_source_provenance(source: CanonicalSource) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "source_title": source.title,
        "source_authority": source.authority,
        "source_type": source.source_type,
        "source_url": source.original_url,
        "source_version_date": source.version_date,
        "source_content_hash": source.sha256,
        "document_id": document_id(source.source_id, source.sha256),
    }


def validate_provenance(payload: dict[str, Any]) -> list[str]:
    missing = sorted(field for field in REQUIRED_PROVENANCE_FIELDS if field not in payload)
    empty = sorted(
        field for field in REQUIRED_PROVENANCE_FIELDS
        if field in payload and payload[field] is None
    )
    return [*(f"missing:{field}" for field in missing), *(f"null:{field}" for field in empty)]


__all__ = [
    "PROVENANCE_CONTRACT_ID",
    "REQUIRED_PROVENANCE_FIELDS",
    "base_source_provenance",
    "chunk_id",
    "document_id",
    "record_id",
    "sha256_text",
    "validate_provenance",
]
