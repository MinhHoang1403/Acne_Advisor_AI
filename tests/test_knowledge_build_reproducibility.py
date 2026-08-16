from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ingestion.build import compile_knowledge, compute_build_identity
from src.ingestion.filtering import load_claim_exclusions
from src.ingestion.parser import load_or_parse_source
from src.ingestion.provenance import validate_provenance
from src.ingestion.source_manifest import load_source_manifest, load_web_record_catalog


def test_manifest_mismatch_is_only_non_blocking_for_offline_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.ingestion import pipeline

    async def fake_prepare_knowledge():
        return {
            "identity": SimpleNamespace(build_id="prepared-build"),
            "offline_validation": {
                "layers": [{"layer": "compiled", "passed": True, "errors": []}]
            },
            "compiled": SimpleNamespace(records=[]),
            "cards": [],
        }

    manifest = {
        "build_id": "active-build",
        "collections": {
            "knowledge_physical": "knowledge-active",
            "entity_physical": "entities-active",
        },
    }

    class FakeQdrantClient:
        async def close(self) -> None:
            return None

    async def fake_validate_collection(*args, **kwargs):
        return {"layer": "qdrant", "passed": True, "errors": []}

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pipeline, "prepare_knowledge", fake_prepare_knowledge)
    monkeypatch.setattr(pipeline, "load_build_manifest", lambda path: manifest)
    monkeypatch.setattr(pipeline, "AsyncQdrantClient", lambda **kwargs: FakeQdrantClient())
    monkeypatch.setattr(pipeline, "validate_qdrant_collection", fake_validate_collection)

    offline = asyncio.run(pipeline.validate_knowledge(manifest_path=manifest_path, live=False))
    live = asyncio.run(pipeline.validate_knowledge(manifest_path=manifest_path, live=True))

    offline_manifest = next(layer for layer in offline["layers"] if layer["layer"] == "manifest")
    assert offline["passed"] is True
    assert offline_manifest["warnings"] == ["prepared build is not activated"]
    assert live["passed"] is False
    assert "build ID mismatch" in live["errors"]


def test_build_identity_is_stable_across_taxonomy_line_endings(tmp_path: Path) -> None:
    source_manifest = Path(__file__).resolve().parents[1] / "data" / "sources" / "manifest.yaml"
    taxonomy_lf = tmp_path / "taxonomy-lf.yaml"
    taxonomy_crlf = tmp_path / "taxonomy-crlf.yaml"
    taxonomy_lf.write_bytes(b"version: v1\nentries:\n  - adapalene\n")
    taxonomy_crlf.write_bytes(b"version: v1\r\nentries:\r\n  - adapalene\r\n")

    lf_identity = compute_build_identity(source_manifest, taxonomy_lf)
    crlf_identity = compute_build_identity(source_manifest, taxonomy_crlf)

    assert lf_identity.taxonomy_hash == crlf_identity.taxonomy_hash
    assert lf_identity.build_id == crlf_identity.build_id


def test_frozen_actual_corpus_compiles_reproducibly_with_complete_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = load_source_manifest(root / "data" / "sources" / "manifest.yaml")
    missing = [
        source.local_filename
        for source in sources
        if not (root / "sample_data" / source.local_filename).is_file()
    ]
    if missing:
        pytest.skip("Canonical licensed corpus is not present in this checkout")

    async def load_artifacts():
        return {
            source.source_id: (
                await load_or_parse_source(
                    source,
                    source_dir=root / "sample_data",
                    cache_root=root / "data" / "cache" / "phase1" / "parsed",
                )
            )[0]
            for source in sources
        }

    artifacts = asyncio.run(load_artifacts())
    identity = compute_build_identity(
        root / "data" / "sources" / "manifest.yaml",
        root / "data" / "taxonomy" / "drug_aliases.yaml",
    )
    manifest_path = root / "data" / "sources" / "manifest.yaml"
    web_catalogs = {
        source.source_id: load_web_record_catalog(
            source,
            manifest_path=manifest_path,
            source_dir=root / "sample_data",
        )
        for source in sources
        if source.record_catalog
    }
    exclusions = tuple(
        action
        for source in sources
        for action in load_claim_exclusions(source, manifest_path=manifest_path)
    )
    first = compile_knowledge(
        sources,
        artifacts,
        identity,
        web_record_catalogs=web_catalogs,
        claim_exclusions=exclusions,
        ingested_at="A",
    )
    second = compile_knowledge(
        sources,
        artifacts,
        identity,
        web_record_catalogs=web_catalogs,
        claim_exclusions=exclusions,
        ingested_at="B",
    )

    assert first.identity == second.identity
    assert first.structural_hash == second.structural_hash
    assert [record["chunk_id"] for record in first.records] == [
        record["chunk_id"] for record in second.records
    ]
    assert len(first.records) == 512
    assert all(not validate_provenance(record) for record in first.records)
    represented_parents = {
        record.get("parent_source_id") or record["source_id"] for record in first.records
    }
    assert represented_parents == {source.source_id for source in sources}
    assert len([record for record in first.records if record.get("curation_action_ids")]) == 5
