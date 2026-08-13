#!/usr/bin/env python3
"""Offline Phase 2B context packing readiness eval."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_phase2_retrieval import DEFAULT_GOLDEN_PATH, load_cases  # noqa: E402
from src.knowledge.entity_cards import build_entity_cards_from_taxonomy  # noqa: E402
from src.knowledge.entity_index import build_entity_point_payload  # noqa: E402
from src.retrieval.candidate_merge import merge_candidates  # noqa: E402
from src.retrieval.context_packer import pack_context  # noqa: E402
from src.retrieval.entity_retriever import retrieve_entity_candidates_from_payloads  # noqa: E402
from src.retrieval.metadata_boost import boost_chunk_results  # noqa: E402
from src.retrieval.query_expansion import expand_normalized_query  # noqa: E402
from src.retrieval.query_normalization import normalize_query  # noqa: E402


def run_phase2_context_packing_eval(path: Path = DEFAULT_GOLDEN_PATH) -> dict[str, Any]:
    cases = [
        case for case in load_cases(path)
        if case["id"] in {
            "dalacin_t_identity",
            "epiduo_contains_bpo",
            "benzoyl_peroxide_not_antibiotic",
            "differin_class",
            "blackheads_acne_type",
            "inflammatory_acne_treatment",
        }
    ]
    entity_payloads = [
        build_entity_point_payload(card, kb_version="acne_kb_v1")
        for card in build_entity_cards_from_taxonomy()
    ]
    failures: list[str] = []
    reports: list[dict[str, Any]] = []

    for case in cases:
        case_id = str(case["id"])
        expected = case["expected"]
        normalized = normalize_query(str(case["query"]))
        expansion = expand_normalized_query(normalized)
        entity_candidates = retrieve_entity_candidates_from_payloads(
            normalized_query=normalized,
            expansion=expansion,
            payloads=entity_payloads,
            collection_name="acne_entities",
            limit=8,
        )
        chunk_candidates = boost_chunk_results(
            _synthetic_context_chunks(case_id, expected),
            normalized,
            collection_name="acne_knowledge",
        )
        merged = merge_candidates(entity_candidates, chunk_candidates, normalized, limit=8)
        packed = pack_context(normalized, merged, max_items=5, max_chars=4000)
        case_failures = _evaluate_packed_case(case_id, expected, normalized, packed)
        failures.extend(case_failures)
        reports.append(
            {
                "id": case_id,
                "passed": not case_failures,
                "intent": normalized.intent,
                "entity_items_count": packed.entity_items_count,
                "chunk_items_count": packed.chunk_items_count,
                "selected_entity_ids": packed.debug.get("pack_trace", {}).get("selected_entity_ids", []),
                "selected_chunk_ids": packed.debug.get("pack_trace", {}).get("selected_chunk_ids", []),
                "warnings": packed.warnings,
                "failures": case_failures,
            }
        )

    passed = sum(1 for report in reports if report["passed"])
    return {
        "readiness": "PASS" if not failures else "FAIL",
        "total_cases": len(reports),
        "passed": passed,
        "failed": len(reports) - passed,
        "failures": failures,
        "cases": reports,
    }


def _evaluate_packed_case(case_id: str, expected: dict[str, Any], normalized: Any, packed: Any) -> list[str]:
    failures: list[str] = []
    text = packed.context_text.casefold()

    expected_intents = expected.get("intent_any") or [expected.get("intent")]
    if normalized.intent not in expected_intents:
        failures.append(f"{case_id}: intent {normalized.intent!r} not in {expected_intents!r}")

    for value in expected.get("drug_product", []):
        if value.casefold() not in text:
            failures.append(f"{case_id}: packed context missing product {value}")
    for value in expected.get("active_ingredient", []):
        if value.casefold() not in text:
            failures.append(f"{case_id}: packed context missing active ingredient {value}")
    for value in expected.get("drug_class", []):
        if value.casefold() not in text:
            failures.append(f"{case_id}: packed context missing drug class {value}")

    if expected.get("not_antibiotic") and "not_antibiotic=true" not in text:
        failures.append(f"{case_id}: packed context missing not_antibiotic metadata")
    for value in expected.get("negative_drug_class", []):
        if value in normalized.drug_class:
            failures.append(f"{case_id}: normalized query has forbidden class {value}")

    if normalized.intent in {"drug_identity", "ingredient_question", "class_check"}:
        if packed.entity_items_count == 0:
            failures.append(f"{case_id}: drug intent packed no entity card")
        if packed.chunk_items_count == 0:
            failures.append(f"{case_id}: drug intent packed no chunk evidence")

    if normalized.intent == "acne_type":
        if packed.chunk_items_count == 0:
            failures.append(f"{case_id}: acne type packed no chunk context")
        if packed.entity_items_count > packed.chunk_items_count:
            failures.append(f"{case_id}: acne type overpacked entity cards")
        metadata_any = expected.get("metadata_any", {})
        for values in metadata_any.values():
            if not any(value.casefold() in text for value in values):
                failures.append(f"{case_id}: acne type packed context lacks {values}")

    return failures


def _synthetic_context_chunks(case_id: str, expected: dict[str, Any]) -> list[dict[str, Any]]:
    metadata_any = expected.get("metadata_any", {})
    text_parts = [
        f"Evidence for {case_id}.",
        " ".join(expected.get("drug_product", [])),
        " ".join(expected.get("active_ingredient", [])),
        " ".join(expected.get("drug_class", [])),
        " ".join(expected.get("condition", [])),
    ]
    for values in metadata_any.values():
        text_parts.append(" ".join(values))
    payload: dict[str, Any] = {
        "id": f"{case_id}:chunk",
        "chunk_id": f"{case_id}:chunk",
        "text": " ".join(part for part in text_parts if part).strip(),
        "score": 0.08,
        "source_file": "phase2_context_packing_eval.json",
        "header": "Phase 2B Eval",
        "drug_product": expected.get("drug_product", []),
        "active_ingredient": expected.get("active_ingredient", []),
        "drug_class": expected.get("drug_class", []),
        "condition": expected.get("condition", []),
        "query_intent_hint": [expected.get("intent")] if expected.get("intent") else expected.get("intent_any", []),
    }
    for field, values in metadata_any.items():
        payload[field] = values
    return [payload]


def main() -> int:
    summary = run_phase2_context_packing_eval()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["readiness"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
