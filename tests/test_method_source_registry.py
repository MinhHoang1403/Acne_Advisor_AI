import json
from pathlib import Path

import yaml

from src.ingestion.build import compute_build_identity


def test_segmentation_reference_uses_verified_acl_metadata() -> None:
    registry = json.loads(
        Path("data/method_sources.json").read_text(encoding="utf-8")
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
        Path("data/method_sources.json").read_text(encoding="utf-8")
    )
    records = {record["source_id"]: record for record in registry["sources"]}
    nice = records["nice_ng198_2026_08"]
    data_pipeline = Path("docs/DATA_PIPELINE.md").read_text(encoding="utf-8")
    references = Path("docs/REFERENCES.md").read_text(encoding="utf-8")
    source_manifest = yaml.safe_load(
        Path("data/sources/manifest.yaml").read_text(encoding="utf-8")
    )
    nice_manifest = next(
        source for source in source_manifest["sources"] if source["source_id"] == "nice_ng198_2026_08"
    )

    assert nice["role"] == "frozen_project_corpus_snapshot"
    assert "official metadata last updated 2026-04-30" in nice["publication_date"]
    assert "current official-version provenance remains unresolved" in nice["limitations"]
    assert not nice["claim_supported"].startswith("Current ")
    assert "NICE Source Provenance" in data_pipeline
    assert "does not claim that it is a fully" in data_pipeline
    assert "NICE Source Provenance" in references
    assert nice_manifest["current_as_of_cutoff"] is None
    assert "remain unresolved" in nice_manifest["limitation"]
    assert nice_manifest["sha256"] == "8a5b8e104c1394a48bf20aaca5724c33c526a0be7466682b57204c8092a94869"


def test_new_safety_cross_checks_are_attributed_without_entering_retrieval_corpus() -> None:
    registry = json.loads(
        Path("data/method_sources.json").read_text(encoding="utf-8")
    )
    records = {record["source_id"]: record for record in registry["sources"]}

    breathing = records["nhs_anaphylaxis_shortness_of_breath_2026"]
    bleeding = records["st_john_ambulance_severe_bleeding_2025"]
    assert breathing["role"] == "safety_cross_check_not_retrieval_corpus"
    assert bleeding["role"] == "safety_cross_check_not_retrieval_corpus"
    assert bleeding["authors_or_organization"] == "St John Ambulance"
    assert "NHS_FIRST_AID_HEAVY_BLEEDING" in bleeding["limitations"]


def test_corrected_source_and_curation_contracts_have_expected_build_identity() -> None:
    identity = compute_build_identity(
        Path("data/sources/manifest.yaml"),
        Path("data/taxonomy/drug_aliases.yaml"),
    )

    assert identity.build_id == "45bc4bb563acb89725b5"


def test_method_traceability_covers_methods_and_technical_standards() -> None:
    registry = json.loads(
        Path("data/method_sources.json").read_text(encoding="utf-8")
    )
    records = {record["source_id"]: record for record in registry["sources"]}
    traceability = Path("docs/METHOD_TRACEABILITY.md").read_text(encoding="utf-8")
    required = {
        "qdrant_cosine_search_2026",
        "cormack_clarke_buettcher_rrf_2009",
        "yao_react_2023",
        "jiang_active_rag_2023",
        "jeong_adaptive_rag_2024",
        "aws_timeouts_retries_backoff_jitter_2019",
        "nist_fips_180_4_sha256",
        "ietf_rfc9562_uuidv5",
    }

    assert registry["verified_through"] == "2026-08-16"
    assert required <= records.keys()
    for source_id in required:
        assert source_id in traceability
        assert records[source_id]["limitations"]


def test_gemini_retrieval_instruction_is_an_evaluation_question_only() -> None:
    registry = json.loads(
        Path("data/method_sources.json").read_text(encoding="utf-8")
    )
    records = {record["source_id"]: record for record in registry["sources"]}
    methods = Path("docs/METHODS_AND_FORMULAS.md").read_text(encoding="utf-8")

    google = records["google_gemini_embedding2_2026"]
    assert "does not accept task_type" in google["claim_supported"]
    assert "unprefixed" in google["limitations"]
    assert "unmeasured evaluation question" in methods
