import json
from pathlib import Path


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


def test_nice_provenance_limitation_is_disclosed_without_registry_rewrite() -> None:
    data_pipeline = Path("docs/DATA_PIPELINE.md").read_text(encoding="utf-8")
    references = Path("docs/REFERENCES.md").read_text(encoding="utf-8")

    assert "Accepted NICE Provenance Limitation" in data_pipeline
    assert "does not claim that the snapshot is fully or currently verified" in data_pipeline
    assert "Accepted NICE Corpus Limitation" in references
