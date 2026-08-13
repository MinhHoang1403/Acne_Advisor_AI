from __future__ import annotations

from src.knowledge.entity_cards import build_entity_cards_from_taxonomy
from src.knowledge.entity_index import build_entity_point_payload
from src.knowledge.versioning import (
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
    monkeypatch.setenv("KB_VERSION", "build-contract")
    monkeypatch.setenv("TAXONOMY_VERSION", "taxonomy-contract")
    monkeypatch.setenv("ENTITY_SCHEMA_VERSION", "entity-contract")
    monkeypatch.setenv("CHUNK_SCHEMA_VERSION", "chunk-contract")
    monkeypatch.setenv("INGESTION_PIPELINE_VERSION", "pipeline-contract")
    assert get_knowledge_versions() == {
        "kb_version": "build-contract",
        "taxonomy_version": "taxonomy-contract",
        "entity_schema_version": "entity-contract",
        "chunk_schema_version": "chunk-contract",
        "ingestion_pipeline_version": "pipeline-contract",
    }


def test_entity_payload_has_embedding_and_taxonomy_metadata(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "google")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")
    card = next(
        card for card in build_entity_cards_from_taxonomy()
        if card.entity_type == "drug_product" and card.canonical_name == "Epiduo"
    )
    payload = build_entity_point_payload(card, kb_version="content-build")
    assert payload["embedding_model"] == "models/gemini-embedding-2"
    assert payload["embedding_dimensions"] == 3072
    assert payload["kb_version"] == "content-build"
    assert payload["taxonomy_version"] == "acne_taxonomy_2026_08"


def test_compatibility_guard_detects_embedding_model_mismatch() -> None:
    common = {"embedding_provider": "google", "embedding_dimensions": 3072, "kb_version": "build"}
    issues = validate_embedding_config_compatibility(
        {**common, "embedding_model": "models/gemini-embedding-001"},
        {**common, "embedding_model": "models/gemini-embedding-2"},
    )
    assert any("embedding_model mismatch" in issue for issue in issues)
