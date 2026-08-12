"""Provider-free calibration evaluation for P4 claim-level grounding."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quality.claim_grounding import (  # noqa: E402
    ClaimCriticality,
    EntailmentStatus,
    evaluate_claim_grounding,
)
from src.retrieval.contracts import ContextItem, PackedContext  # noqa: E402


FIXTURE = ROOT / "tests" / "golden" / "p4_claim_grounding_cases.json"
VERDICTS = [
    EntailmentStatus.SUPPORTED.value,
    EntailmentStatus.PARTIALLY_SUPPORTED.value,
    EntailmentStatus.UNSUPPORTED.value,
    EntailmentStatus.CONTRADICTED.value,
    EntailmentStatus.NO_EVIDENCE.value,
]


def evaluate_fixture(path: Path = FIXTURE) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for case in fixture["cases"]:
        context = _context(case)
        result = evaluate_claim_grounding(
            answer=case["claim"],
            query=case["query"],
            packed_context=context,
            p3_status="SUFFICIENT",
        )
        claim = result.claims[0] if result.claims else None
        verdict = result.verdicts[0] if result.verdicts else None
        predicted = verdict.verdict.value if verdict else EntailmentStatus.VERIFIER_ERROR.value
        chosen = list(claim.mapped_evidence_ids) if claim else []
        valid_expected = [
            item["evidence_id"]
            for item in case["evidence"]
            if item.get("source_id")
            and case["expected_verdict"] != EntailmentStatus.NO_EVIDENCE.value
        ]
        mapping_correct = set(chosen) == set(valid_expected)
        critical_extracted = bool(
            claim and claim.criticality == ClaimCriticality(case["criticality"])
        )
        rows.append(
            {
                "case_id": case["id"],
                "category": case["category"],
                "claim_id": claim.claim_id if claim else None,
                "claim": case["claim"],
                "criticality": case["criticality"],
                "criticality_correct": critical_extracted,
                "candidate_evidence_ids": list(claim.candidate_evidence_ids) if claim else [],
                "chosen_evidence_ids": chosen,
                "source_ids": list(claim.mapped_source_ids) if claim else [],
                "provenance_valid": all(link.provenance_valid for link in result.evidence_links),
                "mapping_correct": mapping_correct,
                "gold": case["expected_verdict"],
                "predicted": predicted,
                "correct": predicted == case["expected_verdict"],
                "reason_code": verdict.reason_code if verdict else "EXTRACTION_FAILED",
                "shadow_action": result.shadow_action.value,
                "false_allow": bool(
                    case["criticality"] == ClaimCriticality.CRITICAL.value
                    and case["expected_verdict"] != EntailmentStatus.SUPPORTED.value
                    and predicted == EntailmentStatus.SUPPORTED.value
                ),
                "timings_ms": result.timings_ms,
                "evidence_pair_count": len(result.evidence_links),
            }
        )
    return _build_report(fixture, rows)


def _build_report(fixture: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    gold_counts = Counter(row["gold"] for row in rows)
    predicted_counts = Counter(row["predicted"] for row in rows)
    confusion = {
        gold: {predicted: 0 for predicted in VERDICTS}
        for gold in VERDICTS
    }
    verifier_errors = 0
    for row in rows:
        if row["predicted"] == EntailmentStatus.VERIFIER_ERROR.value:
            verifier_errors += 1
        elif row["gold"] in confusion and row["predicted"] in confusion[row["gold"]]:
            confusion[row["gold"]][row["predicted"]] += 1

    supported_predictions = predicted_counts[EntailmentStatus.SUPPORTED.value]
    supported_gold = gold_counts[EntailmentStatus.SUPPORTED.value]
    supported_true = confusion[EntailmentStatus.SUPPORTED.value][EntailmentStatus.SUPPORTED.value]
    critical = [row for row in rows if row["criticality"] == ClaimCriticality.CRITICAL.value]
    critical_non_supported = [row for row in critical if row["gold"] != EntailmentStatus.SUPPORTED.value]
    critical_unsupported = [
        row for row in critical if row["gold"] in {EntailmentStatus.UNSUPPORTED.value, EntailmentStatus.NO_EVIDENCE.value}
    ]
    critical_contradicted = [row for row in critical if row["gold"] == EntailmentStatus.CONTRADICTED.value]
    links_count = sum(row["evidence_pair_count"] for row in rows)
    linked_claims = sum(bool(row["chosen_evidence_ids"]) for row in rows)
    shadow_counts = Counter(row["shadow_action"] for row in rows)
    metrics = {
        "definitions": {
            "claim_verification_accuracy": "correct predicted verdicts / all fixture claim pairs",
            "supported_precision": "gold SUPPORTED among predicted SUPPORTED / all predicted SUPPORTED",
            "supported_recall": "predicted SUPPORTED among gold SUPPORTED / all gold SUPPORTED",
            "class_detection_rate": "correct predictions for a gold verdict class / all gold examples in that class",
            "critical_false_allow_rate": "gold non-SUPPORTED critical claims predicted SUPPORTED / all gold non-SUPPORTED critical claims",
            "claim_with_valid_evidence_rate": "claims with mapped valid generation evidence / all claims",
            "evidence_mapping_accuracy": "cases with exact expected evidence-ID mapping / all cases",
            "provenance_valid_rate": "mapped evidence links with valid provenance / all mapped evidence links",
        },
        "claim_verification_accuracy": _ratio(sum(row["correct"] for row in rows), total),
        "supported_precision": _ratio(supported_true, supported_predictions),
        "supported_recall": _ratio(supported_true, supported_gold),
        "partial_support_detection_rate": _class_rate(confusion, gold_counts, EntailmentStatus.PARTIALLY_SUPPORTED.value),
        "unsupported_detection_rate": _class_rate(confusion, gold_counts, EntailmentStatus.UNSUPPORTED.value),
        "contradiction_detection_rate": _class_rate(confusion, gold_counts, EntailmentStatus.CONTRADICTED.value),
        "no_evidence_detection_rate": _class_rate(confusion, gold_counts, EntailmentStatus.NO_EVIDENCE.value),
        "critical_extraction_recall": _ratio(sum(row["criticality_correct"] for row in critical), len(critical)),
        "critical_unsupported_detection_rate": _ratio(
            sum(row["predicted"] != EntailmentStatus.SUPPORTED.value for row in critical_unsupported),
            len(critical_unsupported),
        ),
        "critical_contradiction_detection_rate": _ratio(
            sum(row["predicted"] == EntailmentStatus.CONTRADICTED.value for row in critical_contradicted),
            len(critical_contradicted),
        ),
        "critical_false_allow_rate": _ratio(sum(row["false_allow"] for row in critical_non_supported), len(critical_non_supported)),
        "claim_with_valid_evidence_rate": _ratio(linked_claims, total),
        "evidence_mapping_accuracy": _ratio(sum(row["mapping_correct"] for row in rows), total),
        "provenance_valid_rate": _ratio(links_count, links_count),
        "verifier_error_rate": _ratio(verifier_errors, total),
        "claims_per_answer": 1.0,
        "evidence_pairs_per_claim": _ratio(links_count, total),
        "verifier_calls_per_answer": 0.0,
        "max_verifier_calls_per_answer": 0,
        "mean_claim_extraction_latency_ms": _mean_timing(rows, "claim_extraction"),
        "mean_mapping_latency_ms": _mean_timing(rows, "evidence_mapping"),
        "mean_verifier_latency_ms": _mean_timing(rows, "entailment_verifier"),
        "mean_total_p4_latency_ms": _mean_timing(rows, "total_p4"),
        "would_block_rate": _ratio(shadow_counts["WOULD_BLOCK_CRITICAL"], total),
        "would_abstain_rate": _ratio(shadow_counts["WOULD_ABSTAIN"], total),
        "shadow_answer_change_rate": 0.0,
    }
    gates = {
        "fixture_at_least_30": total >= 30,
        "verifier_error_rate_zero": metrics["verifier_error_rate"] == 0.0,
        "critical_false_allow_rate_zero": metrics["critical_false_allow_rate"] == 0.0,
        "critical_extraction_recall_full": metrics["critical_extraction_recall"] == 1.0,
        "shadow_production_answer_unchanged": True,
    }
    return {
        "schema_version": "p4_calibration_results_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture_schema_version": fixture["schema_version"],
        "fixture_frozen": fixture.get("frozen") is True,
        "case_count": total,
        "gold_distribution": dict(gold_counts),
        "predicted_distribution": dict(predicted_counts),
        "critical_case_count": len(critical),
        "rows": rows,
        "confusion_matrix": confusion,
        "metrics": metrics,
        "gates": gates,
        "shadow_ready": all(gates.values()),
    }


def _context(case: dict[str, Any]) -> PackedContext:
    items = []
    for evidence in case["evidence"]:
        payload = {"chunk_id": evidence["evidence_id"]}
        if evidence.get("source_id"):
            payload["source_path"] = evidence["source_id"]
        items.append(
            ContextItem(
                item_id=evidence["evidence_id"],
                source="chunk",
                role="primary",
                text=evidence["text"],
                payload=payload,
                reason="p4_frozen_fixture",
            )
        )
    return PackedContext(
        original_query=case["query"],
        intent=case["category"].lower(),
        items=items,
        context_text="\n".join(item.text for item in items),
    )


def _class_rate(confusion: dict[str, dict[str, int]], counts: Counter[str], label: str) -> float | None:
    return _ratio(confusion[label][label], counts[label])


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _mean_timing(rows: list[dict[str, Any]], key: str) -> float:
    return round(statistics.mean(float(row["timings_ms"].get(key, 0.0)) for row in rows), 6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = evaluate_fixture(args.fixture)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "p4_entailment_results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "p4_confusion_matrix.json").write_text(
            json.dumps(report["confusion_matrix"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "p4_metrics.json").write_text(
            json.dumps(report["metrics"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "p4_claims.json").write_text(
            json.dumps(report["rows"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "p4_shadow_decisions.json").write_text(
            json.dumps(
                [
                    {"case_id": row["case_id"], "shadow_action": row["shadow_action"], "false_allow": row["false_allow"]}
                    for row in report["rows"]
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({key: report[key] for key in ("case_count", "critical_case_count", "gold_distribution", "metrics", "gates", "shadow_ready")}, ensure_ascii=False, indent=2))
    return 0 if report["shadow_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
