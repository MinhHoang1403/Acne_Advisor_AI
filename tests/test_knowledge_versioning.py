from __future__ import annotations

import argparse

from scripts.ingest_knowledge import (
    SemanticChunk,
    enrich_chunks_with_ingestion_metadata,
)
from scripts.validate_kb_collections import (
    parse_args,
    validate_sample_payloads,
)
from src.knowledge.entity_cards import build_entity_cards_from_taxonomy
from src.knowledge.entity_index import build_entity_point_payload
from src.knowledge.versioning import (
    expected_kb_payload_metadata,
    get_embedding_metadata,
    get_knowledge_versions,
    validate_embedding_config_compatibility,
)


def test_get_embedding_metadata_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "google")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")

    assert get_embedding_metadata() == {
        "embedding_provider": "google",
        "embedding_model": "models/gemini-embedding-2",
        "embedding_dimensions": 3072,
    }


def test_get_knowledge_versions_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("KB_VERSION", "acne_kb_v1")
    monkeypatch.setenv("TAXONOMY_VERSION", "drug_taxonomy_v1")
    monkeypatch.setenv("ENTITY_SCHEMA_VERSION", "entity_schema_v1")
    monkeypatch.setenv("CHUNK_SCHEMA_VERSION", "chunk_schema_v2")
    monkeypatch.setenv("INGESTION_PIPELINE_VERSION", "ingestion_pipeline_v2")

    versions = get_knowledge_versions()

    assert versions["kb_version"] == "acne_kb_v1"
    assert versions["taxonomy_version"] == "drug_taxonomy_v1"
    assert versions["entity_schema_version"] == "entity_schema_v1"
    assert versions["chunk_schema_version"] == "chunk_schema_v2"
    assert versions["ingestion_pipeline_version"] == "ingestion_pipeline_v2"


def test_chunk_ingestion_metadata_preserves_old_fields_and_adds_versions(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "google")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")
    monkeypatch.setenv("KB_VERSION", "acne_kb_v1")
    chunk = SemanticChunk(
        source_file="x.pdf",
        chunk_index=0,
        text="Benzoyl peroxide is not an antibiotic.",
        metadata={"source_file": "x.pdf", "custom_field": "keep-me"},
    )

    enrich_chunks_with_ingestion_metadata(
        chunks=[chunk],
        file_info={
            "document_id": "doc-1",
            "source_path": "sample_data/x.pdf",
            "content_hash": "hash",
            "file_size": 10,
            "modified_time": "2026-07-03T00:00:00+00:00",
        },
        ingestion_run_id="run-1",
        ingested_at="2026-07-03T01:00:00+00:00",
    )

    metadata = chunk.metadata
    assert metadata["custom_field"] == "keep-me"
    for key, value in expected_kb_payload_metadata().items():
        assert metadata[key] == value
    assert metadata["chunk_id"] == chunk.chunk_id
    assert metadata["document_id"] == "doc-1"


def test_entity_payload_has_embedding_and_version_metadata(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "google")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")
    monkeypatch.setenv("KB_VERSION", "acne_kb_v1")
    card = next(
        card
        for card in build_entity_cards_from_taxonomy()
        if card.entity_type == "drug_product" and card.canonical_name == "Epiduo"
    )

    payload = build_entity_point_payload(card)

    assert payload["embedding_provider"] == "google"
    assert payload["embedding_model"] == "models/gemini-embedding-2"
    assert payload["embedding_dimensions"] == 3072
    assert payload["kb_version"] == "acne_kb_v1"
    assert payload["taxonomy_version"] == "acne_taxonomy_2026_08"
    assert payload["entity_schema_version"] == "entity_schema_v1"


def test_compatibility_guard_detects_embedding_model_mismatch() -> None:
    issues = validate_embedding_config_compatibility(
        {
            "embedding_provider": "google",
            "embedding_model": "models/gemini-embedding-001",
            "embedding_dimensions": 3072,
            "kb_version": "acne_kb_v1",
        },
        {
            "embedding_provider": "google",
            "embedding_model": "models/gemini-embedding-2",
            "embedding_dimensions": 3072,
            "kb_version": "acne_kb_v1",
        },
    )

    assert any("embedding_model mismatch" in issue for issue in issues)


def test_compatibility_guard_passes_for_same_config() -> None:
    config = {
        "embedding_provider": "google",
        "embedding_model": "models/gemini-embedding-2",
        "embedding_dimensions": 3072,
        "kb_version": "acne_kb_v1",
    }

    assert validate_embedding_config_compatibility(config, dict(config)) == []


def test_validate_script_parse_args_without_qdrant(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "acne_knowledge")
    monkeypatch.setenv("CHUNK_QDRANT_COLLECTION_NAME", "acne_chunks_v1")
    monkeypatch.setenv("ENTITY_QDRANT_COLLECTION_NAME", "acne_entities_v1")

    args = parse_args([])

    assert isinstance(args, argparse.Namespace)
    assert args.chunk_collection == "acne_knowledge"
    assert args.entity_collection == "acne_entities_v1"
    assert args.sample_size == 5
    assert args.strict == "false"


def test_validate_payload_helper_reports_missing_fields() -> None:
    warnings, errors = validate_sample_payloads(
        [{"embedding_provider": "google"}],
        role="entity",
    )

    assert warnings == []
    assert errors
    assert "embedding_model" in errors[0]
    assert "canonical_name" in errors[0]
