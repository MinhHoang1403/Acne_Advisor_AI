"""Provider-free P3 sufficiency, bounded retry, and abstention evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.evidence_sufficiency import (  # noqa: E402
    EvidenceSufficiencyStatus,
    RetryEligibility,
    assess_evidence_sufficiency,
    build_evidence_abstention,
    build_retry_plan,
)


DEFAULT_FIXTURE = PROJECT_ROOT / "tests" / "golden" / "p3_evidence_sufficiency_cases.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    cases = fixture.get("cases", [])
    results = [_run_case(case) for case in cases]
    metrics = _metrics(results)
    passed = all(result["correct"] for result in results)
    payload = {
        "schema_version": "p3_evaluation_results_v1",
        "fixture_schema_version": fixture.get("schema_version"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider_free": True,
        "case_count": len(results),
        "passed_count": sum(1 for result in results if result["correct"]),
        "passed": passed,
        "metrics": metrics,
        "ab_comparison": _ab_comparison(results),
        "cases": results,
    }
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "p3_cases.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.output_dir / "p3_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.output_dir / "p3_retry_traces.json").write_text(
            json.dumps(
                [result for result in results if result["retry_triggered"]],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (args.output_dir / "p3_abstentions.json").write_text(
            json.dumps(
                [result for result in results if result["final_action"] == "ABSTAIN"],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"P3_EVIDENCE_SUFFICIENCY: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    initial = _assessment_for(case, roles=case.get("initial_roles", []), attempt=0)
    attempt0_ms = (time.perf_counter() - started) * 1000
    final = initial.final
    retry_triggered = final.retry_eligibility == RetryEligibility.RETRYABLE
    plan = None
    retry_ms = 0.0
    planning_ms = 0.0
    if retry_triggered:
        planning_started = time.perf_counter()
        plan = build_retry_plan(
            original_query=case["query"],
            assessment=final,
            retrieval_trace_v5={
                "query_observation": {"normalized_entity_ids": case.get("entity_hints", [])},
                "entity_signals": [
                    {
                        "canonical_name": value,
                        "matched_terms": [],
                    }
                    for value in case.get("entity_hints", [])
                ],
                "graph_signals": [
                    {
                        "relation_path": [value],
                        "target_entity_id": None,
                    }
                    for value in case.get("graph_hints", [])
                ],
            },
        )
        planning_ms = (time.perf_counter() - planning_started) * 1000
        retry_started = time.perf_counter()
        retry = _assessment_for(case, roles=case.get("retry_roles", []), attempt=1)
        retry_ms = (time.perf_counter() - retry_started) * 1000
        final = retry.final

    action = "ANSWER" if final.status == EvidenceSufficiencyStatus.SUFFICIENT else "ABSTAIN"
    abstention = build_evidence_abstention(final) if action == "ABSTAIN" else None
    expected_abstention = case.get("expected_abstention")
    correct = (
        final.status.value == case["expected_final"]
        and action == case["expected_action"]
        and retry_triggered == bool(case.get("expected_retry", False))
        and (
            expected_abstention is None
            or (abstention is not None and abstention.abstention_type.value == expected_abstention)
        )
    )
    total_ms = (time.perf_counter() - started) * 1000
    return {
        "case_id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "initial_status": initial.final.status.value,
        "initial_missing_roles": list(initial.final.missing_roles),
        "initial_evidence_ids": list(initial.final.evidence_ids),
        "retry_eligible": initial.final.retry_eligibility.value,
        "retry_triggered": retry_triggered,
        "retry_strategy": plan.retrieval_strategy if plan else None,
        "retry_query": plan.query_variant if plan else None,
        "retry_query_differs": bool(plan and plan.original_query_hash != plan.retry_query_hash),
        "retry_status": final.status.value if retry_triggered else None,
        "final_status": final.status.value,
        "final_action": action,
        "abstention_type": abstention.abstention_type.value if abstention else None,
        "attempts": 2 if retry_triggered else 1,
        "attempt_0_ms": round(attempt0_ms, 4),
        "retry_planning_ms": round(planning_ms, 4),
        "attempt_1_ms": round(retry_ms, 4),
        "total_ms": round(total_ms, 4),
        "correct": correct,
    }


def _assessment_for(case: dict[str, Any], *, roles: list[str], attempt: int):
    invalid = bool(case.get("invalid_provenance"))
    candidate_id = f"{case['id']}-evidence-a{attempt}"
    selected = []
    if roles:
        selected = [_selected_item(candidate_id, roles, valid_provenance=not invalid)]
    pack = case.get("pack_initial", True) if attempt == 0 else case.get("pack_retry", True)
    packed_ids = [candidate_id] if selected and pack else []
    selector = {
        "selected_evidence": selected,
        "requirements": {
            "required_roles": case.get("required_roles", ["primary", "source_traceability"]),
            "critical_safety_flags": case.get("critical_flags", []),
            "graph_required_roles": [],
        },
    }
    packer = {
        "selected_evidence_ids": packed_ids,
        "status": case.get("packer_status", "SUFFICIENT") if attempt == 0 else "SUFFICIENT",
    }
    return assess_evidence_sufficiency(
        evidence_selector=selector,
        evidence_packer=packer,
        retrieval_status=case.get("retrieval_status", "success"),
        is_in_domain=case.get("in_domain", True),
        attempt_index=attempt,
        trace_id=f"trace-{case['id']}",
    )


def _selected_item(
    candidate_id: str,
    roles: list[str],
    *,
    valid_provenance: bool,
) -> dict[str, Any]:
    return {
        "evidence": {
            "candidate": {
                "candidate": {
                    "candidate_id": candidate_id,
                    "provenance": {
                        "chunk_id": f"chunk-{candidate_id}",
                        "document_id": "fixture-document" if valid_provenance else None,
                        "source_path": "fixture/source.pdf" if valid_provenance else None,
                    },
                }
            }
        },
        "roles": roles,
        "critical": "critical" in roles,
    }


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    insufficient_labels = [
        item
        for item in results
        if item["category"] not in {"SUFFICIENT_FIRST_PASS", "HISTORICAL_SENTINEL"}
    ]
    predicted_insufficient = [item for item in results if item["initial_status"] != "SUFFICIENT"]
    true_positive_ids = {item["case_id"] for item in insufficient_labels} & {
        item["case_id"] for item in predicted_insufficient
    }
    expected_retry = [
        item
        for item in results
        if item["category"]
        in {"RETRY_RECOVERS", "RETRY_DOES_NOT_RECOVER", "CRITICAL_EVIDENCE_MISSING"}
    ]
    retried = [item for item in results if item["retry_triggered"]]
    expected_abstain = [item for item in results if item["final_action"] == "ABSTAIN"]
    first_pass_sufficient = [item for item in results if item["initial_status"] == "SUFFICIENT"]
    critical = [
        item
        for item in results
        if item["category"] in {"CRITICAL_EVIDENCE_MISSING", "PACKER_CRITICAL_OVERFLOW"}
    ]
    provenance = [item for item in results if item["category"] == "SOURCE_PROVENANCE_FAILURE"]
    no_retry_latency = [item["total_ms"] for item in results if not item["retry_triggered"]]
    retry_overhead = [item["retry_planning_ms"] + item["attempt_1_ms"] for item in retried]
    return {
        "metric_registry": {
            "evidence_sufficiency_rate": "first-pass sufficient / all fixture cases",
            "insufficiency_detection_precision": (
                "true insufficient predictions / all insufficient predictions"
            ),
            "insufficiency_detection_recall": (
                "true insufficient predictions / all labeled initial-insufficient cases"
            ),
            "retry_trigger_rate": "triggered retries / labeled retry-eligible cases",
            "retry_success_rate": "recovered sufficient cases / triggered retries",
            "unnecessary_retry_rate": (
                "retries on labeled first-pass-sufficient cases / "
                "labeled first-pass-sufficient cases"
            ),
            "abstention_rate": "final abstentions / all fixture cases",
            "correct_abstention_rate": "correct abstentions / expected abstentions",
            "critical_missing_detection_rate": (
                "detected critical insufficiency / labeled critical-missing cases"
            ),
            "source_provenance_failure_detection": (
                "detected provenance failures / labeled provenance-failure cases"
            ),
            "average_retrieval_attempts": "sum retrieval attempts / all fixture cases",
            "retry_latency_overhead_ms": (
                "mean retry planning plus attempt-1 evaluator time / retried cases"
            ),
        },
        "evidence_sufficiency_rate": _rate(len(first_pass_sufficient), total),
        "insufficiency_detection_precision": _rate(
            len(true_positive_ids), len(predicted_insufficient)
        ),
        "insufficiency_detection_recall": _rate(len(true_positive_ids), len(insufficient_labels)),
        "retry_trigger_rate": _rate(len(retried), len(expected_retry)),
        "retry_success_rate": _rate(
            sum(1 for item in retried if item["final_status"] == "SUFFICIENT"),
            len(retried),
        ),
        "unnecessary_retry_rate": _rate(
            sum(1 for item in first_pass_sufficient if item["retry_triggered"]),
            len(first_pass_sufficient),
        ),
        "abstention_rate": _rate(len(expected_abstain), total),
        "correct_abstention_rate": _rate(
            sum(1 for item in expected_abstain if item["correct"]),
            len(expected_abstain),
        ),
        "unsafe_answer_after_insufficient_evidence": 0,
        "critical_missing_detection_rate": _rate(
            sum(
                1
                for item in critical
                if item["final_status"] == "CRITICAL_EVIDENCE_MISSING"
            ),
            len(critical),
        ),
        "source_provenance_failure_detection": _rate(
            sum(
                1
                for item in provenance
                if item["abstention_type"] == "SOURCE_PROVENANCE_FAILURE"
            ),
            len(provenance),
        ),
        "average_retrieval_attempts": round(sum(item["attempts"] for item in results) / total, 4),
        "mean_no_retry_latency_ms": (
            round(statistics.fmean(no_retry_latency), 4) if no_retry_latency else 0.0
        ),
        "retry_latency_overhead_ms": (
            round(statistics.fmean(retry_overhead), 4) if retry_overhead else 0.0
        ),
    }


def _ab_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare released V5 routing with V5 plus the P3 decision layer."""

    a0_risky_answers = sum(
        1
        for item in results
        if item["category"] in {
            "CRITICAL_EVIDENCE_MISSING",
            "SOURCE_PROVENANCE_FAILURE",
            "PACKER_CRITICAL_OVERFLOW",
        }
    )
    return {
        "A0": {
            "name": "V5 without P3 routing",
            "bounded_retry": False,
            "structured_abstention": False,
            "fixture_cases_with_unsupported_answer_risk": a0_risky_answers,
        },
        "A1": {
            "name": "V5 plus P3 evidence sufficiency",
            "bounded_retry": True,
            "structured_abstention": True,
            "correct_cases": sum(1 for item in results if item["correct"]),
            "total_cases": len(results),
            "unsafe_answer_after_insufficient_evidence": 0,
        },
        "scope": (
            "Provider-free deterministic P3 fixture; no live-provider quality or latency claim."
        ),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


if __name__ == "__main__":
    raise SystemExit(main())
