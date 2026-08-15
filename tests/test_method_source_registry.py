import json
from pathlib import Path

from src.ingestion.build import compute_build_identity


def test_segmentation_reference_uses_verified_acl_metadata() -> None:
    registry = json.loads(
        Path("data/phase1_method_sources.json").read_text(encoding="utf-8")
    )
    records = {record["source_id"]: record for record in registry["sources"]}

    record = records["wang_segmentation_2025"]
    assert record["authors_or_organization"] == "Zhitong Wang et al."
    assert record["doi"] == "10.18653/v1/2025.findings-acl.422"
    assert record["role"] == "related_literature"
    assert "does not implement PIC" in record["limitations"]
    assert "qu_segmentation_2025" not in records


def test_nice_source_provenance_is_disclosed_in_registry_and_docs() -> None:
    registry = json.loads(
        Path("data/phase1_method_sources.json").read_text(encoding="utf-8")
    )
    records = {record["source_id"]: record for record in registry["sources"]}
    nice = records["nice_ng198_2026_08"]
    data_pipeline = Path("docs/DATA_PIPELINE.md").read_text(encoding="utf-8")
    references = Path("docs/REFERENCES.md").read_text(encoding="utf-8")

    assert nice["role"] == "frozen_project_corpus_snapshot"
    assert "official metadata last updated 2026-04-30" in nice["publication_date"]
    assert "current official-version provenance remains unresolved" in nice["limitations"]
    assert not nice["claim_supported"].startswith("Current ")
    assert "NICE Source Provenance" in data_pipeline
    assert "does not claim that it is a fully" in data_pipeline
    assert "NICE Source Provenance" in references


def test_method_registry_correction_does_not_change_validated_build_identity() -> None:
    identity = compute_build_identity(
        Path("data/sources/manifest.yaml"),
        Path("data/taxonomy/drug_aliases.yaml"),
    )

    assert identity.build_id == "ec0a6de32d58ac181af6"
