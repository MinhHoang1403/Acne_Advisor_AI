from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.ingestion.build import compile_knowledge, compute_build_identity
from src.ingestion.parser import load_or_parse_source
from src.ingestion.provenance import validate_provenance
from src.ingestion.source_manifest import load_source_manifest


def test_build_identity_is_stable_across_taxonomy_line_endings(tmp_path: Path) -> None:
    source_manifest = (
        Path(__file__).resolve().parents[1] / "data" / "sources" / "manifest.yaml"
    )
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
