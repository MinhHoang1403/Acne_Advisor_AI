from __future__ import annotations

from pathlib import Path

from scripts.eval_phase1_readiness import (
    DEFAULT_GOLDEN_PATH,
    load_golden_cases,
    relationship_key,
    relationship_set,
    run_phase1_readiness_eval,
)
from scripts.ingest_knowledge import (
    _file_manifest_info,
    get_incremental_file_plan,
    ingestion_config_fingerprint,
)
from src.ingestion.cleanup import is_safe_chunk_collection_for_cleanup
from src.ingestion.domain_metadata import enrich_domain_metadata
from src.knowledge import DrugEntityNormalizer
from src.knowledge.entity_cards import build_entity_cards_from_taxonomy, entity_card_to_text
from src.knowledge.entity_index import build_entity_point_payload
from src.knowledge.graph_schema import build_entity_graph_records
from src.knowledge.versioning import (
    expected_kb_payload_metadata,
    get_embedding_metadata,
    get_knowledge_versions,
)


def _cases() -> list[dict]:
    return load_golden_cases(DEFAULT_GOLDEN_PATH)


def _card_index() -> dict[tuple[str, str], object]:
    return {
        (card.entity_type, card.canonical_name): card
        for card in build_entity_cards_from_taxonomy()
    }


def test_golden_set_has_required_phase1_cases() -> None:
    cases = _cases()
    ids = {case["id"] for case in cases}

    assert len(cases) == 6
    assert {
        "dalacin_t_identity",
        "epiduo_contains_bpo",
        "differin_class",
        "benzoyl_peroxide_not_antibiotic",
        "clindamycin_not_retinoid",
        "adapalene_not_antibiotic",
    } == ids


def test_golden_normalizer_eval() -> None:
    normalizer = DrugEntityNormalizer()

    for case in _cases():
        expected = case["expected"]
        result = normalizer.expand_query(case["query"])
        entity_pairs = {
            (entity["entity_type"], entity["canonical_name"])
            for entity in result["normalized_entities"]
        }

        for product in expected.get("drug_product", []):
            assert ("drug_product", product) in entity_pairs, case["id"]
        for ingredient in expected.get("active_ingredient", []):
            assert ingredient in result["active_ingredients"], case["id"]
        for class_name in expected.get("drug_class", []):
            assert class_name in result["drug_class"], case["id"]

        negative = expected.get("negative_expectations", {})
        for class_name in negative.get("drug_class_absent", []):
            assert class_name not in result["drug_class"], case["id"]


def test_golden_domain_metadata_eval() -> None:
    for case in _cases():
        metadata = enrich_domain_metadata(
            case["query"],
            existing_metadata={"source_file": "golden_eval.txt", "chunk_id": case["id"]},
        )
        expected = case["expected"]

        for field in (
            "drug_product",
            "active_ingredient",
            "drug_class",
            "condition",
            "safety_context",
            "query_intent_hint",
            "taxonomy_version",
            "entity_schema_version",
        ):
            assert field in metadata, case["id"]

        assert metadata["source_file"] == "golden_eval.txt"
        assert metadata["chunk_id"] == case["id"]
        for field in ("drug_product", "active_ingredient", "drug_class"):
            for value in expected.get(field, []):
                assert value in metadata[field], case["id"]

        negative = expected.get("negative_expectations", {})
        for class_name in negative.get("drug_class_absent", []):
            assert class_name not in metadata["drug_class"], case["id"]


def test_golden_entity_cards_eval() -> None:
    cards = _card_index()

    for case in _cases():
        expected = case["expected"]
        for entity_type, field in (
            ("drug_product", "drug_product"),
            ("active_ingredient", "active_ingredient"),
        ):
            for canonical_name in expected.get(field, []):
                card = cards[(entity_type, canonical_name)]
                payload = card.to_payload()
                for payload_field in (
                    "entity_type",
                    "canonical_name",
                    "aliases",
                    "active_ingredients",
                    "drug_class",
                    "taxonomy_version",
                    "entity_schema_version",
                    "metadata",
                ):
                    assert payload_field in payload, case["id"]

    benzoyl_peroxide = cards[("active_ingredient", "benzoyl_peroxide")]
    assert benzoyl_peroxide.metadata["not_antibiotic"] is True
    assert "not an antibiotic" in entity_card_to_text(benzoyl_peroxide).lower()
    assert "topical_antibiotic" not in benzoyl_peroxide.drug_class
    assert "oral_antibiotic" not in benzoyl_peroxide.drug_class


def test_golden_entity_qdrant_payload_eval() -> None:
    cards = _card_index()
    required_fields = {
        "text",
        "entity_type",
        "canonical_name",
        "aliases",
        "active_ingredients",
        "drug_class",
        "entity_id",
        "point_id",
        "kb_version",
        "taxonomy_version",
        "entity_schema_version",
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "chunk_schema_version",
        "ingestion_pipeline_version",
    }

    for case in _cases():
        expected = case["expected"]
        for entity_type, field in (
            ("drug_product", "drug_product"),
            ("active_ingredient", "active_ingredient"),
        ):
            for canonical_name in expected.get(field, []):
                payload = build_entity_point_payload(
                    cards[(entity_type, canonical_name)],
                    kb_version="acne_kb_v1",
                )
                assert required_fields.issubset(payload), case["id"]


def test_golden_deterministic_graph_eval() -> None:
    records = build_entity_graph_records(build_entity_cards_from_taxonomy())
    relationships = relationship_set(records)

    for case in _cases():
        expected = case["expected"]
        for edge in expected.get("required_graph_edges", []):
            assert relationship_key(edge) in relationships, case["id"]
        for edge in expected.get("negative_expectations", {}).get("forbidden_graph_edges", []):
            assert relationship_key(edge) not in relationships, case["id"]

    for edge in (
        {
            "source_label": "ActiveIngredient",
            "source_name": "adapalene",
            "relationship": "BELONGS_TO_CLASS",
            "target_label": "DrugClass",
            "target_name": "topical_antibiotic",
        },
        {
            "source_label": "ActiveIngredient",
            "source_name": "clindamycin",
            "relationship": "BELONGS_TO_CLASS",
            "target_label": "DrugClass",
            "target_name": "topical_retinoid",
        },
    ):
        assert relationship_key(edge) not in relationships


def test_phase1_version_metadata_eval() -> None:
    embedding = get_embedding_metadata()
    versions = get_knowledge_versions()
    payload = expected_kb_payload_metadata()

    assert isinstance(embedding["embedding_dimensions"], int)
    for field in (
        "kb_version",
        "taxonomy_version",
        "entity_schema_version",
        "chunk_schema_version",
        "ingestion_pipeline_version",
    ):
        assert versions[field]
        assert payload[field] == versions[field]


def test_phase1_cleanup_safety_eval(tmp_path: Path) -> None:
    assert is_safe_chunk_collection_for_cleanup("acne_entities_v1") is False
    assert is_safe_chunk_collection_for_cleanup("acne_knowledge") is True

    source = tmp_path / "doc.pdf"
    source.write_bytes(b"same")
    file_info = _file_manifest_info(source)
    manifest = {
        "documents": {
            file_info["source_path"]: {
                **{key: value for key, value in file_info.items() if key != "path"},
                "status": "completed",
                "graph_validated": True,
                "ingestion_config_fingerprint": ingestion_config_fingerprint(),
            }
        }
    }

    skip_plan = get_incremental_file_plan([source], manifest)
    assert skip_plan["skipped"][0]["cleanup_required"] is False

    manifest["documents"][file_info["source_path"]]["content_hash"] = "old-hash"
    changed_plan = get_incremental_file_plan([source], manifest)
    assert changed_plan["to_ingest"][0]["cleanup_required"] is True

    manifest["documents"][file_info["source_path"]]["content_hash"] = file_info["content_hash"]
    manifest["documents"][file_info["source_path"]]["status"] = "partial"
    partial_plan = get_incremental_file_plan([source], manifest)
    assert partial_plan["to_ingest"][0]["cleanup_required"] is True


def test_phase1_readiness_eval_script_summary_passes() -> None:
    summary = run_phase1_readiness_eval(DEFAULT_GOLDEN_PATH)

    assert summary["readiness"] == "PASS"
    assert summary["total_cases"] == 6
    assert summary["failures"] == []
