#!/usr/bin/env python3
"""Offline Phase 2A retrieval readiness eval."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.entity_cards import build_entity_cards_from_taxonomy  # noqa: E402
from src.knowledge.entity_index import build_entity_point_payload  # noqa: E402
from src.retrieval.candidate_merge import merge_candidates  # noqa: E402
from src.retrieval.contracts import RetrievedCandidate  # noqa: E402
from src.retrieval.entity_retriever import retrieve_entity_candidates_from_payloads  # noqa: E402
from src.retrieval.metadata_boost import boost_chunk_results  # noqa: E402
from src.retrieval.query_expansion import expand_normalized_query  # noqa: E402
from src.retrieval.query_normalization import normalize_query  # noqa: E402

DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "phase2_retrieval_eval_cases.json"


def load_cases(path: Path = DEFAULT_GOLDEN_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase2_retrieval_eval(path: Path = DEFAULT_GOLDEN_PATH) -> dict[str, Any]:
    cases = load_cases(path)
    entity_payloads = [
        build_entity_point_payload(card, kb_version="acne_kb_v1")
        for card in build_entity_cards_from_taxonomy()
    ]
    failures: list[str] = []
    case_reports: list[dict[str, Any]] = []

    for case in cases:
        case_id = str(case["id"])
        query = str(case["query"])
        expected = case["expected"]
        normalized = normalize_query(query)
        expansion = expand_normalized_query(normalized)
        entity_candidates = retrieve_entity_candidates_from_payloads(
            normalized_query=normalized,
            expansion=expansion,
            payloads=entity_payloads,
            collection_name="acne_entities",
            limit=8,
        )
        chunk_candidates = boost_chunk_results(
            _synthetic_chunk_results(case_id, expected),
            normalized,
            collection_name="acne_knowledge",
        )
        merged = merge_candidates(entity_candidates, chunk_candidates, normalized, limit=8)

        case_failures = _evaluate_case(
            case_id=case_id,
            expected=expected,
            normalized=normalized,
            expansion_terms=expansion.expanded_terms,
            entity_candidates=entity_candidates,
            chunk_candidates=chunk_candidates,
            merged=merged,
        )
        failures.extend(case_failures)
        case_reports.append(
            {
                "id": case_id,
                "passed": not case_failures,
                "intent": normalized.intent,
                "drug_product": normalized.drug_product,
                "active_ingredient": normalized.active_ingredient,
                "drug_class": normalized.drug_class,
                "condition": normalized.condition,
                "expanded_terms": expansion.expanded_terms,
                "top_entity": _candidate_label(entity_candidates[0]) if entity_candidates else None,
                "top_merged": _candidate_label(merged[0]) if merged else None,
                "failures": case_failures,
            }
        )

    passed_count = sum(1 for report in case_reports if report["passed"])
    return {
        "readiness": "PASS" if not failures else "FAIL",
        "total_cases": len(cases),
        "passed": passed_count,
        "failed": len(cases) - passed_count,
        "failures": failures,
        "cases": case_reports,
    }


def _evaluate_case(
    case_id: str,
    expected: dict[str, Any],
    normalized: Any,
    expansion_terms: list[str],
    entity_candidates: list[RetrievedCandidate],
    chunk_candidates: list[RetrievedCandidate],
    merged: list[RetrievedCandidate],
) -> list[str]:
    failures: list[str] = []
    expected_intents = expected.get("intent_any") or [expected.get("intent")]
    if normalized.intent not in expected_intents:
        failures.append(f"{case_id}: intent {normalized.intent!r} not in {expected_intents!r}")

    for field in ("drug_product", "active_ingredient", "drug_class", "condition"):
        actual = getattr(normalized, field)
        for value in expected.get(field, []):
            if value not in actual:
                failures.append(f"{case_id}: missing normalized {field}={value}")

    for value in expected.get("negative_drug_class", []):
        if value in normalized.drug_class:
            failures.append(f"{case_id}: unexpected drug_class={value}")

    for field, values in expected.get("metadata_any", {}).items():
        actual_values = normalized.metadata.get(field, [])
        if not any(value in actual_values for value in values):
            failures.append(f"{case_id}: metadata {field} lacks any of {values}")

    for value in [
        *expected.get("drug_product", []),
        *expected.get("active_ingredient", []),
        *expected.get("drug_class", []),
    ]:
        if not _contains_term(expansion_terms, value):
            failures.append(f"{case_id}: expansion missing {value}")

    if expected.get("not_antibiotic"):
        bp_candidate = _find_candidate(entity_candidates, "benzoyl_peroxide")
        if not bp_candidate or bp_candidate.payload.get("metadata", {}).get("not_antibiotic") is not True:
            failures.append(f"{case_id}: benzoyl peroxide entity lacks not_antibiotic metadata")

    if expected.get("drug_product") or expected.get("active_ingredient"):
        expected_entities = [*expected.get("drug_product", []), *expected.get("active_ingredient", [])]
        if not any(_candidate_matches(candidate, expected_entities) for candidate in entity_candidates):
            failures.append(f"{case_id}: entity retrieval missed {expected_entities}")

    if expected.get("drug_class") or expected.get("metadata_any"):
        if not any(candidate.matched_metadata for candidate in chunk_candidates):
            failures.append(f"{case_id}: chunk metadata boost had no matches")

    if normalized.intent in {"drug_identity", "ingredient_question", "class_check"}:
        if not merged or merged[0].source != "entity":
            failures.append(f"{case_id}: top merged candidate should be entity")
    if normalized.intent == "acne_type":
        if any(candidate.payload.get("entity_type") == "drug_product" for candidate in merged[:2]):
            failures.append(f"{case_id}: acne type query over-prioritized drug product")

    return failures


def _synthetic_chunk_results(case_id: str, expected: dict[str, Any]) -> list[dict[str, Any]]:
    payload = {
        "id": f"{case_id}:chunk",
        "chunk_id": f"{case_id}:chunk",
        "text": f"Golden retrieval chunk for {case_id}",
        "score": 0.05,
        "drug_product": expected.get("drug_product", []),
        "active_ingredient": expected.get("active_ingredient", []),
        "drug_class": expected.get("drug_class", []),
        "condition": expected.get("condition", []),
        "query_intent_hint": [expected.get("intent")] if expected.get("intent") else expected.get("intent_any", []),
        "source_file": "phase2_retrieval_eval.json",
    }
    for field, values in expected.get("metadata_any", {}).items():
        payload[field] = values
    return [payload]


def _contains_term(terms: list[str], value: str) -> bool:
    value_key = value.casefold()
    return any(term.casefold() == value_key for term in terms)


def _candidate_matches(candidate: RetrievedCandidate, expected_entities: list[str]) -> bool:
    expected_keys = {entity.casefold() for entity in expected_entities}
    values = [
        str(candidate.payload.get("canonical_name") or ""),
        *[str(alias) for alias in candidate.payload.get("aliases", []) or []],
    ]
    return any(value.casefold() in expected_keys for value in values)


def _find_candidate(candidates: list[RetrievedCandidate], canonical_name: str) -> RetrievedCandidate | None:
    for candidate in candidates:
        if str(candidate.payload.get("canonical_name") or "").casefold() == canonical_name.casefold():
            return candidate
    return None


def _candidate_label(candidate: RetrievedCandidate) -> dict[str, Any]:
    return {
        "source": candidate.source,
        "canonical_name": candidate.payload.get("canonical_name"),
        "entity_type": candidate.payload.get("entity_type"),
        "score": candidate.score,
        "fused_score": candidate.fused_score,
        "matched_metadata": candidate.matched_metadata,
    }


def main() -> int:
    summary = run_phase2_retrieval_eval()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["readiness"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
