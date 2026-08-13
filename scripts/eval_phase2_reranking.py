#!/usr/bin/env python3
"""Offline Phase 2C reranking readiness eval."""

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
from src.retrieval.contracts import RetrievedCandidate  # noqa: E402
from src.retrieval.entity_retriever import retrieve_entity_candidates_from_payloads  # noqa: E402
from src.retrieval.metadata_boost import boost_chunk_results  # noqa: E402
from src.retrieval.query_expansion import expand_normalized_query  # noqa: E402
from src.retrieval.query_normalization import normalize_query  # noqa: E402
from src.retrieval.reranker import rerank_candidates  # noqa: E402


def run_phase2_reranking_eval(path: Path = DEFAULT_GOLDEN_PATH) -> dict[str, Any]:
    cases = load_cases(path)
    entity_payloads = [
        build_entity_point_payload(card, kb_version="acne_kb_v1")
        for card in build_entity_cards_from_taxonomy()
    ]
    failures: list[str] = []
    reports: list[dict[str, Any]] = []

    metric_counts = {
        "entity_recall_at_5": 0,
        "relevant_candidate_at_5": 0,
        "top3_correctness": 0,
        "wrong_entity_penalty_check": 0,
        "acne_type_not_drug_dominated": 0,
    }

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
        entity_candidates.extend(_distractor_entities(entity_payloads, case_id))
        chunk_candidates = boost_chunk_results(
            _synthetic_rerank_chunks(case_id, expected),
            normalized,
            collection_name="acne_knowledge",
        )
        merged = merge_candidates(entity_candidates, chunk_candidates, normalized, limit=12)
        reranked, trace = rerank_candidates(
            normalized_query=normalized,
            candidates=merged,
            expansion=expansion,
            top_n=8,
            provider="local_rules",
        )
        case_failures, metric_passes = _evaluate_case(case_id, expected, normalized, reranked)
        failures.extend(case_failures)
        for metric, passed in metric_passes.items():
            if passed:
                metric_counts[metric] += 1
        reports.append(
            {
                "id": case_id,
                "passed": not case_failures,
                "intent": normalized.intent,
                "top5": [_candidate_label(candidate) for candidate in reranked[:5]],
                "rerank_provider": trace.provider,
                "failures": case_failures,
            }
        )

    total_cases = len(cases)
    metrics = {
        name: {
            "passed": count,
            "total": total_cases,
            "ratio": round(count / total_cases, 4) if total_cases else 0.0,
        }
        for name, count in metric_counts.items()
    }
    passed = sum(1 for report in reports if report["passed"])
    return {
        "readiness": "PASS" if not failures else "FAIL",
        "total_cases": total_cases,
        "passed": passed,
        "failed": total_cases - passed,
        "metrics": metrics,
        "failures": failures,
        "cases": reports,
    }


def _evaluate_case(
    case_id: str,
    expected: dict[str, Any],
    normalized: Any,
    reranked: list[RetrievedCandidate],
) -> tuple[list[str], dict[str, bool]]:
    failures: list[str] = []
    top5 = reranked[:5]
    top3 = reranked[:3]
    labels_top5 = [_candidate_terms(candidate) for candidate in top5]
    labels_top3 = [_candidate_terms(candidate) for candidate in top3]
    expected_terms = [
        *expected.get("drug_product", []),
        *expected.get("active_ingredient", []),
        *expected.get("drug_class", []),
        *expected.get("condition", []),
    ]
    metadata_terms = [
        value
        for values in expected.get("metadata_any", {}).values()
        for value in values
    ]

    relevant_at_5 = _any_terms(labels_top5, [*expected_terms, *metadata_terms])
    top3_correct = _any_terms(labels_top3, expected_terms or metadata_terms)
    entity_recall_at_5 = True
    wrong_entity_penalty = True
    acne_not_drug_dominated = True

    if expected_terms and normalized.intent != "acne_type":
        entity_recall_at_5 = _any_terms(labels_top5, expected_terms)
        if not entity_recall_at_5:
            failures.append(f"{case_id}: expected entity terms not found in top5: {expected_terms}")
    if not relevant_at_5:
        failures.append(f"{case_id}: no relevant candidate in top5")
    if not top3_correct:
        failures.append(f"{case_id}: no expected term in top3")

    if expected.get("not_antibiotic"):
        bp_top3 = [
            candidate for candidate in top3
            if candidate.payload.get("canonical_name") == "benzoyl_peroxide"
        ]
        if not bp_top3:
            failures.append(f"{case_id}: benzoyl_peroxide not in top3")
        elif not any(
            candidate.payload.get("metadata", {}).get("not_antibiotic") is True
            for candidate in bp_top3
        ):
            failures.append(f"{case_id}: top BP candidate lacks not_antibiotic metadata")

    for wrong_class in expected.get("negative_drug_class", []):
        if reranked and str(reranked[0].payload.get("canonical_name")) == wrong_class:
            wrong_entity_penalty = False
            failures.append(f"{case_id}: wrong class {wrong_class} ranked top1")

    if normalized.intent == "acne_type":
        if not any(candidate.source == "chunk" and _candidate_has_any(candidate, metadata_terms) for candidate in top5):
            failures.append(f"{case_id}: acne/blackhead/inflammatory chunk not found in top5")
        if reranked and reranked[0].payload.get("entity_type") in {"drug_product", "active_ingredient", "drug_class"}:
            acne_not_drug_dominated = False
            failures.append(f"{case_id}: drug entity dominated acne_type top1")

    return failures, {
        "entity_recall_at_5": entity_recall_at_5,
        "relevant_candidate_at_5": relevant_at_5,
        "top3_correctness": top3_correct,
        "wrong_entity_penalty_check": wrong_entity_penalty,
        "acne_type_not_drug_dominated": acne_not_drug_dominated,
    }


def _synthetic_rerank_chunks(case_id: str, expected: dict[str, Any]) -> list[dict[str, Any]]:
    relevant_text = " ".join(
        [
            f"Relevant evidence for {case_id}.",
            " ".join(expected.get("drug_product", [])),
            " ".join(expected.get("active_ingredient", [])),
            " ".join(expected.get("drug_class", [])),
            " ".join(expected.get("condition", [])),
            " ".join(value for values in expected.get("metadata_any", {}).values() for value in values),
        ]
    ).strip()
    relevant: dict[str, Any] = {
        "id": f"{case_id}:relevant_chunk",
        "chunk_id": f"{case_id}:relevant_chunk",
        "text": relevant_text or f"Relevant acne evidence for {case_id}",
        "score": 0.08,
        "source_file": "phase2_reranking_eval.json",
        "drug_product": expected.get("drug_product", []),
        "active_ingredient": expected.get("active_ingredient", []),
        "drug_class": expected.get("drug_class", []),
        "condition": expected.get("condition", []),
        "query_intent_hint": [expected.get("intent")] if expected.get("intent") else expected.get("intent_any", []),
    }
    for field, values in expected.get("metadata_any", {}).items():
        relevant[field] = values
    distractor = {
        "id": f"{case_id}:distractor_chunk",
        "chunk_id": f"{case_id}:distractor_chunk",
        "text": "General unrelated drug chunk about oral antibiotics and skincare.",
        "score": 0.5,
        "source_file": "phase2_reranking_eval.json",
        "active_ingredient": ["doxycycline"],
        "drug_class": ["oral_antibiotic"],
        "query_intent_hint": ["general_acne_question"],
    }
    return [distractor, relevant]


def _distractor_entities(payloads: list[dict[str, Any]], case_id: str) -> list[RetrievedCandidate]:
    names = ["topical_retinoid", "topical_antibiotic", "Dalacin T"]
    output: list[RetrievedCandidate] = []
    for payload in payloads:
        if payload.get("canonical_name") not in names:
            continue
        copied = dict(payload)
        copied["entity_id"] = f"distractor:{case_id}:{payload.get('canonical_name')}"
        output.append(
            RetrievedCandidate(
                candidate_id=str(copied["entity_id"]),
                source="entity",
                collection="acne_entities",
                text=str(copied.get("text") or copied.get("canonical_name") or ""),
                score=0.95,
                fused_score=1.1,
                payload=copied,
                matched_metadata={},
                rank=None,
                debug={"distractor": True},
            )
        )
    return output


def _candidate_label(candidate: RetrievedCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "canonical_name": candidate.payload.get("canonical_name"),
        "chunk_id": candidate.payload.get("chunk_id"),
        "rerank_score": candidate.debug.get("rerank_score"),
        "reasons": candidate.debug.get("rerank_reasons", []),
    }


def _candidate_terms(candidate: RetrievedCandidate) -> list[str]:
    payload = candidate.payload
    values = [
        candidate.text,
        payload.get("canonical_name"),
        *(_as_list(payload.get("aliases"))),
        *(_as_list(payload.get("active_ingredients"))),
        *(_as_list(payload.get("active_ingredient"))),
        *(_as_list(payload.get("drug_class"))),
        *(_as_list(payload.get("condition"))),
        *(_as_list(payload.get("concern"))),
        *(_as_list(payload.get("content_type"))),
    ]
    return [str(value) for value in values if value]


def _any_terms(candidate_terms: list[list[str]], expected_terms: list[str]) -> bool:
    return any(
        _contains_any(terms, expected_terms)
        for terms in candidate_terms
    )


def _candidate_has_any(candidate: RetrievedCandidate, expected_terms: list[str]) -> bool:
    return _contains_any(_candidate_terms(candidate), expected_terms)


def _contains_any(values: list[str], expected_terms: list[str]) -> bool:
    haystack = " ".join(values).casefold()
    return any(str(term).casefold() in haystack for term in expected_terms if term)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)] if value else []


def main() -> int:
    summary = run_phase2_reranking_eval()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["readiness"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
