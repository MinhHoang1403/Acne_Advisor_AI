from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest

from src.ingestion import pipeline as ingestion_pipeline
from src.ingestion.pipeline import inspect_embedding_cache_reuse, prepare_knowledge
from src.ingestion.source_manifest import (
    load_source_manifest,
    load_web_record_catalog,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "sources" / "manifest.yaml"


def _web_source():
    return next(
        source
        for source in load_source_manifest(MANIFEST)
        if source.local_filename == "web_raw_dataset.json"
    )


def _require_canonical_corpus() -> None:
    missing = [
        source.local_filename
        for source in load_source_manifest(MANIFEST)
        if not (ROOT / "sample_data" / source.local_filename).is_file()
    ]
    if missing:
        pytest.skip("Canonical licensed corpus is not present in this checkout")


def test_web_catalog_covers_all_raw_records_with_correct_publishers() -> None:
    _require_canonical_corpus()
    catalog = load_web_record_catalog(
        _web_source(),
        manifest_path=MANIFEST,
        source_dir=ROOT / "sample_data",
    )

    assert len(catalog) == 81
    assert Counter(item.authority for item in catalog.values()) == {
        "American Academy of Dermatology": 44,
        "National Health Service": 2,
        "WebMD": 21,
        "DermNet": 12,
        "Healthline": 1,
        "Da Lieu Viet Nam": 1,
    }
    aad = catalog["https://www.aad.org/public/diseases/acne"]
    dermnet = catalog["https://dermnetnz.org/topics/acne"]
    assert aad.source_id == "web_aad_eb67ac58641ae1c2"
    assert aad.authority == "American Academy of Dermatology"
    assert dermnet.source_id == "web_dermnetnz_35860b4af4910f27"
    assert dermnet.authority == "DermNet"


def test_web_catalog_fails_closed_when_raw_record_text_changes(tmp_path: Path) -> None:
    _require_canonical_corpus()
    source = _web_source()
    source_dir = tmp_path / "sample_data"
    manifest_dir = tmp_path / "sources"
    source_dir.mkdir()
    manifest_dir.mkdir()
    raw_path = ROOT / "sample_data" / source.local_filename
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw[0]["raw_text"] += " changed"
    (source_dir / source.local_filename).write_text(
        json.dumps(raw, ensure_ascii=False),
        encoding="utf-8",
    )
    (manifest_dir / source.record_catalog).write_bytes(
        (ROOT / "data" / "sources" / source.record_catalog).read_bytes()
    )

    with pytest.raises(ValueError, match="Raw text hash mismatch"):
        load_web_record_catalog(
            source,
            manifest_path=manifest_dir / "manifest.yaml",
            source_dir=source_dir,
        )


def test_compiled_web_provenance_and_claim_curation_are_exact() -> None:
    _require_canonical_corpus()
    prepared = asyncio.run(prepare_knowledge())
    records = prepared["compiled"].records
    web_records = [record for record in records if record.get("parent_source_id")]
    curated = [record for record in records if record.get("curation_action_ids")]

    assert len(records) == 512
    assert len(curated) == 5
    assert all(record["pre_curation_content_hash"] for record in curated)
    assert all(record["parent_source_id"] == "aad_public_acne_2026_07" for record in web_records)
    assert all(record["source_id"].startswith("web_") for record in web_records)
    assert all(record["source_file"] == "web_raw_dataset.json" for record in web_records)
    assert not any("phân loại thai kỳ" in record["text"].casefold() for record in records)
    assert sha256_file(ROOT / "sample_data" / "web_raw_dataset.json") == _web_source().sha256


def test_approved_build_cache_reuse_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_canonical_corpus()

    def forbidden_call(*args, **kwargs):
        raise AssertionError("inspect-cache must not invoke providers or datastores")

    async def forbidden_async_call(*args, **kwargs):
        raise AssertionError("inspect-cache must not parse or resolve embeddings")

    monkeypatch.setattr(ingestion_pipeline, "AsyncQdrantClient", forbidden_call)
    monkeypatch.setattr(ingestion_pipeline, "get_neo4j_driver", forbidden_call)
    monkeypatch.setattr(ingestion_pipeline, "load_or_parse_source", forbidden_async_call)
    monkeypatch.setattr(ingestion_pipeline, "resolve_embeddings", forbidden_async_call)
    monkeypatch.setattr(ingestion_pipeline.EmbeddingCache, "put", forbidden_call)

    cache_roots = (
        ROOT / "data" / "cache" / "phase1" / "parsed",
        ROOT / "data" / "cache" / "phase1" / "embeddings",
    )

    def snapshot() -> dict[str, tuple[int, int]]:
        return {
            str(path.relative_to(ROOT)): (path.stat().st_size, path.stat().st_mtime_ns)
            for root in cache_roots
            for path in root.rglob("*")
            if path.is_file()
        }

    cache_before = snapshot()

    result = asyncio.run(inspect_embedding_cache_reuse())

    assert snapshot() == cache_before

    assert result["passed"] is True
    assert result["build_id"] == "94d613bc9b33628de3ef"
    assert result["parsed"] == {
        "hits": 4,
        "misses": 0,
        "total": 4,
        "missing_or_invalid_source_ids": [],
    }
    knowledge_embeddings = result["knowledge_embeddings"]
    assert knowledge_embeddings["inspected"] is True
    assert knowledge_embeddings["total"] == 512
    assert knowledge_embeddings["hits"] + knowledge_embeddings["misses"] == 512
    assert result["entity_embeddings"] == {
        "inspected": True,
        "hits": 32,
        "misses": 0,
        "total": 32,
    }
    assert result["provider_calls"] == 0
