"""Offline Phase 1 readiness eval for taxonomy, metadata, payloads, and cleanup."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "phase1_ingest_eval_cases.json"


def load_golden_cases(path: Path = DEFAULT_GOLDEN_PATH) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Golden file must contain a list: {path}")
    for case in data:
        if not isinstance(case, dict) or "id" not in case or "query" not in case:
            raise ValueError(f"Invalid golden case: {case!r}")
    return data


def relationship_key(edge: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(edge["source_label"]),
        str(edge["source_name"]),
        str(edge["relationship"]),
        str(edge["target_label"]),
        str(edge["target_name"]),
    )


def relationship_set(records: dict[str, list[dict[str, Any]]]) -> set[tuple[str, str, str, str, str]]:
    return {relationship_key(edge) for edge in records["relationships"]}


def run_phase1_readiness_eval(
    golden_path: Path = DEFAULT_GOLDEN_PATH,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    cases = load_golden_cases(golden_path)
    normalizer = DrugEntityNormalizer()
    cards = build_entity_cards_from_taxonomy(normalizer)
    card_index = {(card.entity_type, card.canonical_name): card for card in cards}
    graph_records = build_entity_graph_records(cards)
    graph_edges = relationship_set(graph_records)

    failures: list[str] = []
    mapping_checks = 0

    for case in cases:
        case_id = str(case["id"])
        query = str(case["query"])
        expected = case.get("expected", {})
        if not isinstance(expected, dict):
            failures.append(f"{case_id}: expected must be an object")
            continue

        result = normalizer.expand_query(query)
        entity_pairs = {
            (entity["entity_type"], entity["canonical_name"])
            for entity in result["normalized_entities"]
        }
        metadata = enrich_domain_metadata(
            query,
            existing_metadata={"source_file": "golden_eval.txt", "chunk_id": case_id},
        )

        for field in ("drug_product", "active_ingredient", "drug_class"):
            expected_values = expected.get(field, [])
            for value in expected_values:
                mapping_checks += 1
                if field == "drug_product":
                    if ("drug_product", value) not in entity_pairs:
                        failures.append(f"{case_id}: normalizer missing {field}={value}")
                else:
                    normalizer_field = "active_ingredients" if field == "active_ingredient" else field
                    if value not in result.get(normalizer_field, []):
                        failures.append(f"{case_id}: normalizer missing {field}={value}")
                if value not in metadata.get(field, []):
                    failures.append(f"{case_id}: domain metadata missing {field}={value}")

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
            if field not in metadata:
                failures.append(f"{case_id}: metadata missing field {field}")

        if metadata.get("source_file") != "golden_eval.txt" or metadata.get("chunk_id") != case_id:
            failures.append(f"{case_id}: metadata did not preserve source_file/chunk_id")

        negative = expected.get("negative_expectations", {})
        absent_classes = negative.get("drug_class_absent", []) if isinstance(negative, dict) else []
        for class_name in absent_classes:
            if class_name in result.get("drug_class", []):
                failures.append(f"{case_id}: normalizer should not include drug_class={class_name}")
            if class_name in metadata.get("drug_class", []):
                failures.append(f"{case_id}: domain metadata should not include drug_class={class_name}")

        for edge in expected.get("required_graph_edges", []):
            edge_key = relationship_key(edge)
            if edge_key not in graph_edges:
                failures.append(f"{case_id}: graph missing required edge {edge_key}")

        forbidden_edges = negative.get("forbidden_graph_edges", []) if isinstance(negative, dict) else []
        for edge in forbidden_edges:
            edge_key = relationship_key(edge)
            if edge_key in graph_edges:
                failures.append(f"{case_id}: graph has forbidden edge {edge_key}")

        if verbose:
            print(f"[CASE] {case_id}: {query}")
            print(f"  normalizer active_ingredients={result.get('active_ingredients', [])}")
            print(f"  normalizer drug_class={result.get('drug_class', [])}")
            print(f"  metadata active_ingredient={metadata.get('active_ingredient', [])}")
            print(f"  metadata drug_class={metadata.get('drug_class', [])}")

    _check_entity_cards_and_payloads(cases, card_index, failures)
    _check_versions(failures)
    _check_cleanup_safety(failures)

    return {
        "total_cases": len(cases),
        "passed": not failures,
        "passed_mappings": mapping_checks - len([f for f in failures if "missing" in f]),
        "failed_mappings": len(failures),
        "failures": failures,
        "card_count": len(cards),
        "graph_nodes_count": len(graph_records["nodes"]),
        "graph_relationships_count": len(graph_records["relationships"]),
        "readiness": "PASS" if not failures else "FAIL",
    }


def _check_entity_cards_and_payloads(
    cases: list[dict[str, Any]],
    card_index: dict[tuple[str, str], Any],
    failures: list[str],
) -> None:
    required_payload_fields = {
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

    for case in cases:
        case_id = str(case["id"])
        expected = case.get("expected", {})
        for entity_type, field in (
            ("drug_product", "drug_product"),
            ("active_ingredient", "active_ingredient"),
        ):
            for canonical_name in expected.get(field, []):
                card = card_index.get((entity_type, canonical_name))
                if card is None:
                    failures.append(f"{case_id}: missing entity card {entity_type}:{canonical_name}")
                    continue

                payload = card.to_payload()
                for field_name in (
                    "entity_type",
                    "canonical_name",
                    "aliases",
                    "active_ingredients",
                    "drug_class",
                    "taxonomy_version",
                    "entity_schema_version",
                    "metadata",
                ):
                    if field_name not in payload:
                        failures.append(
                            f"{case_id}: card {entity_type}:{canonical_name} missing {field_name}"
                        )

                point_payload = build_entity_point_payload(card, kb_version="acne_kb_v1")
                for field_name in required_payload_fields:
                    if field_name not in point_payload:
                        failures.append(
                            f"{case_id}: Qdrant payload {entity_type}:{canonical_name} "
                            f"missing {field_name}"
                        )

    benzoyl_peroxide = card_index.get(("active_ingredient", "benzoyl_peroxide"))
    if benzoyl_peroxide is None:
        failures.append("benzoyl_peroxide: missing active ingredient card")
        return

    text = entity_card_to_text(benzoyl_peroxide).lower()
    if benzoyl_peroxide.metadata.get("not_antibiotic") is not True:
        failures.append("benzoyl_peroxide: metadata.not_antibiotic is not true")
    if "not an antibiotic" not in text:
        failures.append("benzoyl_peroxide: text does not contain 'not an antibiotic'")
    for class_name in ("topical_antibiotic", "oral_antibiotic"):
        if class_name in benzoyl_peroxide.drug_class:
            failures.append(f"benzoyl_peroxide: should not belong to {class_name}")


def _check_versions(failures: list[str]) -> None:
    embedding = get_embedding_metadata()
    versions = get_knowledge_versions()
    expected = expected_kb_payload_metadata()

    for key in ("embedding_provider", "embedding_model", "embedding_dimensions"):
        if expected.get(key) in (None, ""):
            failures.append(f"version metadata missing {key}")
    if not isinstance(embedding.get("embedding_dimensions"), int):
        failures.append("embedding_dimensions must be an int")
    for key in (
        "kb_version",
        "taxonomy_version",
        "entity_schema_version",
        "chunk_schema_version",
        "ingestion_pipeline_version",
    ):
        if not versions.get(key):
            failures.append(f"knowledge version missing {key}")
        if expected.get(key) != versions.get(key):
            failures.append(f"expected payload metadata mismatch for {key}")


def _check_cleanup_safety(failures: list[str]) -> None:
    if is_safe_chunk_collection_for_cleanup("acne_entities_v1"):
        failures.append("cleanup safety: acne_entities_v1 should be blocked")
    if not is_safe_chunk_collection_for_cleanup("acne_knowledge"):
        failures.append("cleanup safety: acne_knowledge should be allowed")

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "doc.pdf"
        source.write_bytes(b"same")
        info = _file_manifest_info(source)
        manifest = {
            "documents": {
                info["source_path"]: {
                    **{key: value for key, value in info.items() if key != "path"},
                    "status": "completed",
                    "graph_validated": True,
                    "ingestion_config_fingerprint": ingestion_config_fingerprint(),
                }
            }
        }
        skip_plan = get_incremental_file_plan([source], manifest)
        if skip_plan["skipped"][0]["cleanup_required"] is not False:
            failures.append("cleanup safety: unchanged completed file should not cleanup")

        manifest["documents"][info["source_path"]]["content_hash"] = "old-hash"
        changed_plan = get_incremental_file_plan([source], manifest)
        if changed_plan["to_ingest"][0]["cleanup_required"] is not True:
            failures.append("cleanup safety: changed file should require cleanup")

        manifest["documents"][info["source_path"]]["content_hash"] = info["content_hash"]
        manifest["documents"][info["source_path"]]["status"] = "partial"
        partial_plan = get_incremental_file_plan([source], manifest)
        if partial_plan["to_ingest"][0]["cleanup_required"] is not True:
            failures.append("cleanup safety: partial retry should require cleanup")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline Phase 1 readiness eval.")
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Path to golden eval cases JSON.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-case details.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_phase1_readiness_eval(args.golden, verbose=args.verbose)

    print(f"Phase 1 readiness: {summary['readiness']}")
    print(f"Total cases: {summary['total_cases']}")
    print(f"Passed mappings: {summary['passed_mappings']}")
    print(f"Failed mappings/checks: {summary['failed_mappings']}")
    print(f"Entity cards: {summary['card_count']}")
    print(f"Graph nodes: {summary['graph_nodes_count']}")
    print(f"Graph relationships: {summary['graph_relationships_count']}")

    if summary["failures"]:
        print("Failures:")
        for failure in summary["failures"]:
            print(f"- {failure}")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
