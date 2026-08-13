"""
src.ingestion – Phase 1.5: Dermatology-Aware Chunking
=====================================================

This package provides domain-specific taxonomy and metadata extraction
for dermatology / acne content chunks.

Modules
-------
dermatology_taxonomy
    Constants for domain topics, content types, ingredients, skin types,
    concerns, body areas, safety contexts, and Vietnamese ↔ English mappings.

domain_metadata
    Rule-based metadata extractor that annotates text chunks with
    structured dermatology metadata.

chunking and filtering
    Pure current splitting and noisy-chunk heuristic primitives.

sparse_legacy
    Compatibility custom hashed normalized log-TF representation. This is
    not canonical BM25; migration is reserved for S4A.
"""

from src.ingestion.dermatology_taxonomy import (
    BODY_AREAS,
    CONCERNS,
    CONTENT_TYPES,
    DOMAIN_TOPICS,
    INGREDIENTS,
    SAFETY_CONTEXTS,
    SKIN_TYPES,
    VIETNAMESE_MAPPINGS,
)
from src.ingestion.domain_metadata import (
    DermatologyChunkMetadata,
    enrich_domain_metadata,
    extract_dermatology_metadata,
)

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
