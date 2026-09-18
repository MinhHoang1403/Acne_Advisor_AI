from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingestion.build import BUILD_MANIFEST_SCHEMA
from src.ingestion.bm25 import active_bm25_document
from src.ingestion.embedding import (
    EMBEDDING_CONTRACT_ID,
    LEGACY_RAW_EMBEDDING_CONTRACT_ID,
    active_query_embedding_text,
    format_embedding_document,
    format_embedding_query,
)
from src.knowledge.versioning import (
    get_embedding_metadata,
    get_knowledge_versions,
    resolve_active_knowledge_build_id,
    validate_embedding_config_compatibility,
)


BUILD_ID = "a" * 20


def _write_active_manifest(path: Path, *, build_id: str = BUILD_ID) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": BUILD_MANIFEST_SCHEMA,
                "status": "activated",
                "build_id": build_id,
                "collections": {
                    "knowledge_logical": "acne_knowledge",
                    "knowledge_physical": f"acne_knowledge__{build_id}",
                    "entity_logical": "acne_entities",
                    "entity_physical": f"acne_entities__{build_id}",
                },
            }
        ),
        encoding="utf-8",
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


def test_get_knowledge_versions_uses_manifest_build_and_other_contract_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_active_manifest(manifest_path)
    monkeypatch.setenv("KB_VERSION", "build-contract")
    monkeypatch.setenv("TAXONOMY_VERSION", "taxonomy-contract")
    monkeypatch.setenv("ENTITY_SCHEMA_VERSION", "entity-contract")
    monkeypatch.setenv("CHUNK_SCHEMA_VERSION", "chunk-contract")
    monkeypatch.setenv("INGESTION_PIPELINE_VERSION", "pipeline-contract")
    assert get_knowledge_versions(manifest_path=manifest_path) == {
        "kb_version": BUILD_ID,
        "taxonomy_version": "taxonomy-contract",
        "entity_schema_version": "entity-contract",
        "chunk_schema_version": "chunk-contract",
        "ingestion_pipeline_version": "pipeline-contract",
    }


def test_get_knowledge_versions_accepts_candidate_identity_without_active_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "candidate.json"
    _write_active_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "completed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("TAXONOMY_VERSION", "taxonomy-contract")

    versions = get_knowledge_versions(
        manifest_path=manifest_path,
        build_id="b" * 20,
    )

    assert versions["kb_version"] == "b" * 20
    assert versions["taxonomy_version"] == "taxonomy-contract"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"status": "completed"}, "not activated"),
        ({"build_id": "invalid"}, "20-character"),
    ],
)
def test_active_knowledge_build_fails_closed_for_invalid_manifest(
    tmp_path: Path,
    overrides: dict[str, str],
    message: str,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_active_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(overrides)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        resolve_active_knowledge_build_id(manifest_path)


def test_active_knowledge_build_rejects_collection_identity_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_active_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["collections"]["knowledge_physical"] = "acne_knowledge__" + "b" * 20
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="collection identity"):
        resolve_active_knowledge_build_id(manifest_path)


def test_compatibility_guard_detects_embedding_model_mismatch() -> None:
    common = {"embedding_provider": "google", "embedding_dimensions": 3072, "kb_version": "build"}
    issues = validate_embedding_config_compatibility(
        {**common, "embedding_model": "models/gemini-embedding-001"},
        {**common, "embedding_model": "models/gemini-embedding-2"},
    )
    assert any("embedding_model mismatch" in issue for issue in issues)


def test_embedding_instruction_format_is_exact() -> None:
    assert format_embedding_query("  mụn viêm  ") == (
        "task: question answering | query: mụn viêm"
    )
    assert format_embedding_document("  Nội dung  ", title="  Hướng dẫn  ") == (
        "title: Hướng dẫn | text: Nội dung"
    )


@pytest.mark.parametrize(
    ("contract_id", "expected"),
    [
        (EMBEDDING_CONTRACT_ID, "task: question answering | query: mụn"),
        (LEGACY_RAW_EMBEDDING_CONTRACT_ID, "mụn"),
    ],
)
def test_query_representation_follows_active_manifest(
    tmp_path: Path, contract_id: str, expected: str,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_active_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contracts"] = {"embedding": {"id": contract_id}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert active_query_embedding_text("mụn", manifest_path=manifest_path) == expected


def test_bm25_query_options_follow_active_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_active_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contracts"] = {
        "bm25": {
            "configuration": {
                "k": 1.2,
                "b": 0.75,
                "avg_len": 256.0,
                "tokenizer": "word",
                "language": "none",
                "lowercase": True,
                "ascii_folding": False,
            }
        }
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert active_bm25_document("mun", manifest_path=manifest_path).options.ascii_folding is False
