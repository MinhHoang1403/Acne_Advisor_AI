from __future__ import annotations

from src.ingestion.domain_metadata import enrich_domain_metadata


def test_enrich_dalacin_t_metadata() -> None:
    metadata = enrich_domain_metadata(
        "Dalacin T contains clindamycin and is a topical antibiotic used for acne."
    )

    assert "Dalacin T" in metadata["drug_product"]
    assert "clindamycin" in metadata["active_ingredient"]
    assert "topical_antibiotic" in metadata["drug_class"]
    assert "acne_vulgaris" in metadata["condition"]


def test_enrich_epiduo_metadata() -> None:
    metadata = enrich_domain_metadata("Epiduo gel contains adapalene and benzoyl peroxide.")

    assert "Epiduo" in metadata["drug_product"]
    assert "adapalene" in metadata["active_ingredient"]
    assert "benzoyl_peroxide" in metadata["active_ingredient"]
    assert "topical_retinoid" in metadata["drug_class"]
    assert "benzoyl_peroxide" not in metadata["drug_class"]


def test_enrich_differin_metadata() -> None:
    metadata = enrich_domain_metadata("Differin is adapalene, a topical retinoid.")

    assert "Differin" in metadata["drug_product"]
    assert "adapalene" in metadata["active_ingredient"]
    assert "topical_retinoid" in metadata["drug_class"]


def test_bp_not_antibiotic_metadata() -> None:
    metadata = enrich_domain_metadata("Benzoyl peroxide is not an antibiotic.")

    assert "benzoyl_peroxide" in metadata["active_ingredient"]
    assert "topical_antibiotic" not in metadata["drug_class"]
    assert "oral_antibiotic" not in metadata["drug_class"]


def test_intent_side_effect() -> None:
    metadata = enrich_domain_metadata(
        "Benzoyl peroxide can cause dryness, redness, peeling, irritation and bleaching of fabrics."
    )

    assert "side_effect" in metadata["query_intent_hint"]


def test_intent_pregnancy_safety() -> None:
    metadata = enrich_domain_metadata("Topical adapalene is not for use during pregnancy.")

    assert "pregnancy" in metadata["safety_context"]
    assert "pregnancy_safety" in metadata["query_intent_hint"]
    assert "contraindication" in metadata["query_intent_hint"]


def test_existing_metadata_preserved() -> None:
    existing_metadata = {"source_file": "x.pdf", "chunk_id": "abc"}

    metadata = enrich_domain_metadata("Epiduo contains adapalene.", existing_metadata)

    assert metadata["source_file"] == "x.pdf"
    assert metadata["chunk_id"] == "abc"
    assert "Epiduo" in metadata["drug_product"]


def test_domain_metadata_no_crash_on_empty() -> None:
    metadata = enrich_domain_metadata("")

    assert metadata["drug_product"] == []
    assert metadata["active_ingredient"] == []
    assert metadata["drug_class"] == []
    assert metadata["condition"] == []
    assert metadata["safety_context"] == []
    assert metadata["query_intent_hint"] == []
    assert metadata["taxonomy_version"] == "drug_taxonomy_v1"
    assert metadata["entity_schema_version"] == "entity_schema_v1"
