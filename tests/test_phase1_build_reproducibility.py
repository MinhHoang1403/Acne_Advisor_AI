from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.ingestion.build import compile_knowledge, compute_build_identity
from src.ingestion.parser import load_or_parse_source
from src.ingestion.provenance import validate_provenance
from src.ingestion.source_manifest import load_source_manifest


def test_frozen_actual_corpus_compiles_reproducibly_with_complete_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = load_source_manifest(root / "data" / "sources" / "manifest.yaml")
    missing = [source.local_filename for source in sources if not (root / "sample_data" / source.local_filename).is_file()]
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
    first = compile_knowledge(sources, artifacts, identity, ingested_at="A")
    second = compile_knowledge(sources, artifacts, identity, ingested_at="B")

    assert first.identity == second.identity
    assert first.structural_hash == second.structural_hash
    assert [record["chunk_id"] for record in first.records] == [
        record["chunk_id"] for record in second.records
    ]
    assert len(first.records) == 512
    assert all(not validate_provenance(record) for record in first.records)
    assert {record["source_id"] for record in first.records} == {
        source.source_id for source in sources
    }
