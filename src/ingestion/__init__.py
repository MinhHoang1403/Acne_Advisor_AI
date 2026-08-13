"""Phase 1 ingestion primitives and dermatology metadata helpers.

Public metadata names are loaded lazily so importing a low-level primitive
does not initialize knowledge or database adapters.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_TAXONOMY_EXPORTS = {
    "BODY_AREAS",
    "CONCERNS",
    "CONTENT_TYPES",
    "DOMAIN_TOPICS",
    "INGREDIENTS",
    "SAFETY_CONTEXTS",
    "SKIN_TYPES",
    "VIETNAMESE_MAPPINGS",
}
_METADATA_EXPORTS = {
    "DermatologyChunkMetadata",
    "enrich_domain_metadata",
    "extract_dermatology_metadata",
}


def __getattr__(name: str) -> Any:
    """Preserve package exports without eager cross-responsibility imports."""

    if name in _TAXONOMY_EXPORTS:
        return getattr(import_module("src.ingestion.dermatology_taxonomy"), name)
    if name in _METADATA_EXPORTS:
        return getattr(import_module("src.ingestion.domain_metadata"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Taxonomy
    "DOMAIN_TOPICS",
    "CONTENT_TYPES",
    "INGREDIENTS",
    "SKIN_TYPES",
    "CONCERNS",
    "BODY_AREAS",
    "SAFETY_CONTEXTS",
    "VIETNAMESE_MAPPINGS",
    # Metadata extraction
    "DermatologyChunkMetadata",
    "enrich_domain_metadata",
    "extract_dermatology_metadata",
]
