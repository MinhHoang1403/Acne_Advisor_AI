"""Provider-bounded S2 component diagnostics without production mutations."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from scripts.eval_retrieval_v5_release import _run_case as run_locked_v5_case
from src.evaluation.ablation_metrics import (
    arithmetic_mean,
    duplicate_slot_rate,
    evidence_retention_rate,
    mean_reciprocal_rank,
    recall_at_k,
)
from src.retrieval.candidate_policy import apply_candidate_policy, candidate_policy_budget
from src.retrieval.contracts import RetrievedCandidate
from src.retrieval.query_expansion import expand_normalized_query
from src.retrieval.query_normalization import normalize_query
from src.retrieval.reranker import rerank_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEMANTIC_CASES = PROJECT_ROOT / "tests" / "golden" / "semantic_reranker_cases.json"
R8_CASES = PROJECT_ROOT / "tests" / "golden" / "retrieval_v5_release_cases.json"
P3_CASES = PROJECT_ROOT / "tests" / "golden" / "p3_evidence_sufficiency_cases.json"
PHASE2_CASES = PROJECT_ROOT / "tests" / "golden" / "phase2_retrieval_eval_cases.json"

COMPONENT_STATUSES = {
    "KEEP_EVIDENCE_SUPPORTED",
    "SIMPLIFY_EVIDENCE_SUPPORTED",
    "OPTIONAL_TOOL_EVIDENCE_SUPPORTED",
    "REMOVE_CANDIDATE_EVIDENCE_SUPPORTED",
    "INSUFFICIENT_TRUSTED_EVIDENCE",
    "NOT_CLEANLY_ISOLATABLE",
}


def run_s2_ablation(*, reranker_provider: str = "local_rules") -> dict[str, Any]:
    """Run fixed, provider-free diagnostics and one optional local reranker variant."""

    reranker = run_reranker_ablation(provider=reranker_provider)
    policy = run_candidate_policy_ablation()
    sufficiency = run_sufficiency_retry_ablation()
    entity = run_entity_ablation()
    locked = run_locked_component_diagnostics()
    decisions = {
        "reranker": _decision("INSUFFICIENT_TRUSTED_EVIDENCE", reranker),
        "candidate_policy": _decision("INSUFFICIENT_TRUSTED_EVIDENCE", policy),
        "evidence_sufficiency": _decision("INSUFFICIENT_TRUSTED_EVIDENCE", sufficiency),
        "retry": _decision("INSUFFICIENT_TRUSTED_EVIDENCE", sufficiency["retry"]),
        "entity": _decision("INSUFFICIENT_TRUSTED_EVIDENCE", entity),
        "graph": _decision("NOT_CLEANLY_ISOLATABLE", locked["graph"]),
        "selector": _decision("INSUFFICIENT_TRUSTED_EVIDENCE", locked["selector"]),
        "packer": _decision("KEEP_EVIDENCE_SUPPORTED", locked["packer"]),
    }
    return {
        "schema_version": "s2_controlled_ablation_v1",
        "overall_status": "S2_PARTIAL_EVIDENCE_READY_FOR_S3",
        "decision_policy": (
            "No component-removal decision is made from fixtures without canonical source-label "
            "provenance. Packer KEEP applies only to finite-budget engineering behavior."
        ),
        "a0": {
            "dense_depth": 15,
            "sparse_depth": 15,
            "rrf_k": 60,
            "rrf_weights": "equal",
            "top_k": 5,
            "frozen_from_s1": True,
        },
        "experiments": {
            "reranker": reranker,
            "candidate_policy": policy,
            "sufficiency_retry": sufficiency,
            "entity": entity,
            "graph": locked["graph"],
            "selector": locked["selector"],
            "packer": locked["packer"],
        },
        "component_decisions": decisions,
        "call_counts": _call_counts(reranker, locked),
    }


def run_reranker_ablation(*, provider: str) -> dict[str, Any]:
    """Compare retrieval order with one configured local reranking pass."""

    cases = _read_json(SEMANTIC_CASES)
    baseline_rankings: list[list[str]] = []
    variant_rankings: list[list[str]] = []
    relevant: list[list[str]] = []
    case_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    actual_providers: set[str] = set()
    fallback_count = 0
    for case in cases:
        candidates = _semantic_candidates(case)
        baseline = [item.candidate_id for item in candidates]
        normalized = normalize_query(str(case["query"]))
        expansion = expand_normalized_query(normalized)
        started = time.perf_counter()
        ranked, trace = rerank_candidates(
            normalized,
            candidates,
            expansion,
            top_n=len(candidates),
            provider=provider,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        variant = [item.candidate_id for item in ranked]
        positives = [str(value) for value in case.get("expected_top_ids", [])]
        baseline_rankings.append(baseline)
        variant_rankings.append(variant)
        relevant.append(positives)
        latencies.append(elapsed_ms)
        actual_providers.add(trace.provider)
        fallback_count += int(trace.fallback_used)
        case_rows.append(
            {
                "case_id": case["id"],
                "baseline_ids": baseline,
                "variant_ids": variant,
                "expected_ids": positives,
                "changed": baseline != variant,
                "actual_provider": trace.provider,
                "fallback_used": trace.fallback_used,
                "latency_ms": round(elapsed_ms, 3),
            }
        )
    return {
        "dataset": "semantic_reranker_cases.json",
        "case_count": len(cases),
        "gold_grade": "NOT_DECISION_GRADE_GOLD",
        "reason": "Hand-written relevance labels have no canonical corpus evidence IDs or review record.",
        "requested_provider": provider,
        "actual_providers": sorted(actual_providers),
        "fallback_count": fallback_count,
        "baseline": _ranking_metrics(baseline_rankings, relevant),
        "variant": _ranking_metrics(variant_rankings, relevant),
        "mean_variant_latency_ms": arithmetic_mean(latencies),
        "changed_case_count": sum(row["changed"] for row in case_rows),
        "cases": case_rows,
    }


def run_candidate_policy_ablation() -> dict[str, Any]:
    """Measure the current budget-only policy on unchanged fixture order."""

    rows: list[dict[str, Any]] = []
    for case in _read_json(SEMANTIC_CASES):
        candidates = _semantic_candidates(case)
        input_ids = [item.candidate_id for item in candidates]
        result = apply_candidate_policy(
            candidates,
            normalize_query(str(case["query"])),
            budget=candidate_policy_budget(5),
        )
        output_ids = [item.candidate_id for item in result.candidates]
        rows.append(
            {
                "case_id": case["id"],
                "input_ids": input_ids,
                "output_ids": output_ids,
                "budget": result.budget,
                "retention": evidence_retention_rate(input_ids, output_ids),
                "duplicate_slots": duplicate_slot_rate(output_ids),
                "drops": [drop.model_dump(mode="json") for drop in result.drops],
            }
        )
    return {
        "dataset": "semantic_reranker_cases.json",
        "case_count": len(rows),
        "gold_grade": "NOT_DECISION_GRADE_GOLD",
        "policy_mode": "budget_only",
        "cases_exercising_budget": sum(bool(row["drops"]) for row in rows),
        "mean_retention": arithmetic_mean([float(row["retention"]["value"]) for row in rows]),
        "mean_duplicate_slot_rate": arithmetic_mean(
            [float(row["duplicate_slots"]["value"]) for row in rows]
        ),
        "cases": rows,
    }


def run_sufficiency_retry_ablation() -> dict[str, Any]:
    """Compare a transparent role gate with the frozen P3 contract fixture."""

    payload = _read_json(P3_CASES)
    rows: list[dict[str, Any]] = []
    retry_triggered = retry_recovered = retry_still_insufficient = retry_unnecessary = 0
    for case in payload["cases"]:
        required = set(case.get("required_roles", []))
        initial = set(case.get("initial_roles", []))
        initial_sufficient = _roles_sufficient(case, required, initial)
        retry_expected = bool(case.get("expected_retry", False))
        retry_roles = set(case.get("retry_roles", initial))
        after_retry_sufficient = _roles_sufficient(case, required, retry_roles)
        retry_triggered += int(retry_expected)
        retry_recovered += int(retry_expected and not initial_sufficient and after_retry_sufficient)
        retry_still_insufficient += int(retry_expected and not after_retry_sufficient)
        retry_unnecessary += int(retry_expected and initial_sufficient)
        rows.append(
            {
                "case_id": case["id"],
                "category": case["category"],
                "required_roles": sorted(required),
                "initial_roles": sorted(initial),
                "initial_sufficient": initial_sufficient,
                "retry_expected_by_fixture": retry_expected,
                "retry_roles": sorted(retry_roles),
                "after_retry_sufficient": after_retry_sufficient,
                "expected_final": case["expected_final"],
            }
        )
    return {
        "dataset": "p3_evidence_sufficiency_cases.json",
        "case_count": len(rows),
        "gold_grade": "NOT_DECISION_GRADE_GOLD",
        "reason": "Contract fixtures encode implementation expectations, not observed clinical outcomes.",
        "lightweight_gate": "required roles subset of observed roles; critical safety additionally needs safety",
        "retry": {
            "triggered": retry_triggered,
            "recovered": retry_recovered,
            "still_insufficient": retry_still_insufficient,
            "unnecessary": retry_unnecessary,
            "external_calls": 0,
        },
        "cases": rows,
    }


def run_entity_ablation() -> dict[str, Any]:
    """Compare literal query mentions with taxonomy-backed normalized fields."""

    rows: list[dict[str, Any]] = []
    for case in _read_json(PHASE2_CASES):
        expected = {
            key: [str(value) for value in values]
            for key, values in case.get("expected", {}).items()
            if isinstance(values, list) and key in {"drug_product", "active_ingredient", "drug_class"}
        }
        expected_values = {value.casefold() for values in expected.values() for value in values}
        literal_text = str(case["query"]).casefold().replace("_", " ")
        literal_hits = {
            value for value in expected_values if value.replace("_", " ") in literal_text
        }
        normalized = normalize_query(str(case["query"]))
        normalized_values = {
            str(value).casefold()
            for field in (normalized.drug_product, normalized.active_ingredient, normalized.drug_class)
            for value in field
        }
        rows.append(
            {
                "case_id": case["id"],
                "expected": sorted(expected_values),
                "literal_hits": sorted(literal_hits),
                "normalized_hits": sorted(expected_values & normalized_values),
            }
        )
    expected_count = sum(len(row["expected"]) for row in rows)
    literal_count = sum(len(row["literal_hits"]) for row in rows)
    normalized_count = sum(len(row["normalized_hits"]) for row in rows)
    return {
        "dataset": "phase2_retrieval_eval_cases.json",
        "case_count": len(rows),
        "gold_grade": "NOT_DECISION_GRADE_GOLD",
        "reason": "Expected taxonomy fields lack canonical passage provenance and human review record.",
        "expected_label_count": expected_count,
        "literal_hit_count": literal_count,
        "normalized_hit_count": normalized_count,
        "diagnostic_gain": normalized_count - literal_count,
        "cases": rows,
    }


def run_locked_component_diagnostics() -> dict[str, Any]:
    """Inspect graph, selector, and packer behavior on locked static R8 records."""

    records: list[dict[str, Any]] = []
    timings: list[dict[str, float]] = []
    for case in _read_json(R8_CASES):
        record, elapsed = run_locked_v5_case(case)
        records.append(record)
        timings.append(elapsed)
    primary_retained = sum(
        record["expected"]["primary_source_id"] in record["packed_evidence"]["ids"]
        for record in records
    )
    critical_records = [record for record in records if record["expected"]["critical_source_id"]]
    critical_retained = sum(
        record["expected"]["critical_source_id"] in record["packed_evidence"]["ids"]
        for record in critical_records
    )
    selector_latency = [float(item.get("selector", 0.0)) for item in timings]
    packer_latency = [float(item.get("packer", 0.0)) for item in timings]
    return {
        "graph": {
            "dataset": "retrieval_v5_release_cases.json",
            "case_count": len(records),
            "gold_grade": "TRUSTED_REGRESSION_SET_WITH_CEILING",
            "graph_signal_count": sum(len(record["graph_signals"]) for record in records),
            "medical_claim_eligible_count": sum(
                bool(signal["medical_claim_eligible"])
                for record in records
                for signal in record["graph_signals"]
            ),
            "isolated_quality_delta": "N/A",
            "reason": "Static graph signals are synthetic and do not alter mapped evidence roles.",
        },
        "selector": {
            "dataset": "retrieval_v5_release_cases.json",
            "case_count": len(records),
            "gold_grade": "TRUSTED_REGRESSION_SET_WITH_CEILING",
            "primary_retained": primary_retained,
            "primary_denominator": len(records),
            "mean_latency_ms": arithmetic_mean(selector_latency),
            "semantic_value": "INSUFFICIENT_TRUSTED_EVIDENCE",
        },
        "packer": {
            "dataset": "retrieval_v5_release_cases.json",
            "case_count": len(records),
            "gold_grade": "ENGINEERING_CONTRACT_ONLY",
            "primary_retained": primary_retained,
            "primary_denominator": len(records),
            "critical_retained": critical_retained,
            "critical_denominator": len(critical_records),
            "all_within_item_budget": all(len(record["packed_evidence"]["ids"]) <= 5 for record in records),
            "all_source_backed": all(record["packed_evidence"]["source_paths"] for record in records),
            "mean_latency_ms": arithmetic_mean(packer_latency),
            "claim_scope": "Finite budget and provenance preservation only; not medical quality.",
        },
        "records": records,
    }


def _ranking_metrics(
    rankings: list[list[str]],
    relevant: list[list[str]],
) -> dict[str, Any]:
    return {
        "recall@1": recall_at_k(rankings, relevant, k=1),
        "recall@3": recall_at_k(rankings, relevant, k=3),
        "recall@5": recall_at_k(rankings, relevant, k=5),
        "mrr": mean_reciprocal_rank(rankings, relevant),
    }


def _roles_sufficient(case: dict[str, Any], required: set[str], observed: set[str]) -> bool:
    if case.get("in_domain") is False or case.get("invalid_provenance"):
        return False
    if case.get("retrieval_status") == "recoverable_error":
        return False
    if case.get("packer_status") == "CRITICAL_EVIDENCE_OVERFLOW":
        return False
    if not required.issubset(observed):
        return False
    return not case.get("critical_flags") or "safety" in observed


def _semantic_candidates(case: dict[str, Any]) -> list[RetrievedCandidate]:
    candidates: list[RetrievedCandidate] = []
    for rank, item in enumerate(case.get("candidates", []), start=1):
        candidate_id = str(item["candidate_id"])
        text = str(item.get("text") or "")
        score = float(item.get("retrieval_score", 0.0))
        candidates.append(
            RetrievedCandidate(
                candidate_id=candidate_id,
                source="chunk",
                collection="semantic_reranker_fixture",
                text=text,
                score=score,
                fused_score=score,
                payload={
                    "chunk_id": candidate_id,
                    "document_id": f"fixture:{case['id']}",
                    "source_path": "tests/golden/semantic_reranker_cases.json",
                    "text": text,
                },
                rank=rank,
            )
        )
    return candidates


def _decision(status: str, evidence: dict[str, Any]) -> dict[str, Any]:
    if status not in COMPONENT_STATUSES:
        raise ValueError(f"invalid S2 component status: {status}")
    return {
        "status": status,
        "decision_grade": status == "KEEP_EVIDENCE_SUPPORTED",
        "evidence_scope": evidence.get("claim_scope") or evidence.get("reason") or "diagnostic only",
    }


def _call_counts(reranker: dict[str, Any], locked: dict[str, Any]) -> dict[str, int]:
    return {
        "embedding_api": 0,
        "llm_api": 0,
        "qdrant_mutations": 0,
        "neo4j_mutations": 0,
        "reranker_invocations": int(reranker["case_count"]),
        "graph_signal_stages": int(locked["graph"]["case_count"]),
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "COMPONENT_STATUSES",
    "run_candidate_policy_ablation",
    "run_entity_ablation",
    "run_locked_component_diagnostics",
    "run_reranker_ablation",
    "run_s2_ablation",
    "run_sufficiency_retry_ablation",
]
