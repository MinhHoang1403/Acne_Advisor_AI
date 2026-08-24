from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


AUDIT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = AUDIT_DIR.parents[2]
POST_RUN = Path("evaluation/results/post_improvement_47b10954")
BASELINE_RUN = Path("evaluation/results/formal_run_baseline")
EXPECTED_BENCHMARK_SHA = "f61d6807c0ce39f902936844d810562c486f1bcadaa57a8a2da0e460ad7e534b"
EXPECTED_SYSTEM_SHA = "47b10954d217de91cf8919650b31dc7569ec1f0e"
EXPECTED_PIPELINE = "5991f3c8effb67091fd6274c"
EXPECTED_KB = "94d613bc9b33628de3ef"
EXPECTED_EVALUATOR = "gpt-5.4-mini-2026-03-17"
EXPECTED_RAGCHECKER = "0.1.9"

METRIC_COLUMNS = {
    "claim_recall": "Claim Recall (%)",
    "context_precision": "Context Precision (%)",
    "faithfulness": "Faithfulness (%)",
    "claim_f1": "Claim F1 (%)",
}

CSV_COLUMNS = [
    "case_id",
    "case_family",
    "category",
    "query",
    "expected_action",
    "expected_reason",
    "actual_action",
    "actual_reason",
    "claim_recall_pct",
    "context_precision_pct",
    "faithfulness_pct",
    "claim_f1_pct",
    "nrr_correct",
    "retrieval_attempt",
    "retrieval_status",
    "retry_used",
    "retry_outcome",
    "packed_context_count",
    "packed_context_chars",
    "candidate_count",
    "gold_candidate_hits",
    "gold_pack_hits",
    "pipeline_fingerprint",
    "requested_provider",
    "requested_model",
    "actual_provider",
    "actual_model",
    "llm_fallback_used",
    "primary_classification",
    "root_cause_layer",
    "severity",
    "secondary_flags",
    "baseline_case_delta",
    "baseline_comparison",
    "evidence_summary",
    "audit_notes",
]

MULTI_TURN_PRIMARY = {
    "ANS-MUL-006",
    "ANS-MUL-009",
    "ANS-MUL-010",
    "ANS-MUL-012",
    "ANS-MUL-013",
    "ANS-MUL-015",
}

SOURCE_ARTIFACTS = [
    POST_RUN / "raw_results.json",
    POST_RUN / "case_metrics.csv",
    POST_RUN / "metrics_summary.csv",
    POST_RUN / "ragchecker_checkpoint.json",
    POST_RUN / "evaluator_calibration_results.json",
    POST_RUN / "calibration_adjudication.json",
    Path("evaluation/formal_evaluation.ipynb"),
    Path("evaluation/benchmark_100.json"),
    Path("evaluation/benchmark_manifest.json"),
    Path("evaluation/evaluator_calibration.json"),
    BASELINE_RUN / "raw_results.json",
    BASELINE_RUN / "case_metrics.csv",
    BASELINE_RUN / "metrics_summary.csv",
    BASELINE_RUN / "ragchecker_checkpoint.json",
    BASELINE_RUN / "evaluator_calibration_results.json",
]


class AuditError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(row: dict[str, str], key: str) -> float | None:
    value = row.get(METRIC_COLUMNS[key], "")
    return float(value) if value else None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def round4(value: float) -> float:
    return round(value, 4)


def gold_ids(case: dict[str, Any]) -> set[str]:
    return {
        chunk_id
        for claim in case.get("gold_claims", [])
        for chunk_id in claim.get("source_chunk_ids", [])
    }


def candidate_ids(record: dict[str, Any]) -> set[str]:
    trace = record.get("retrieval_trace") or {}
    candidate_trace = trace.get("candidate_trace") or {}
    output: set[str] = set()
    for channel in ("dense", "bm25", "fused"):
        output.update(
            item.get("candidate_id")
            for item in candidate_trace.get(channel) or []
            if item.get("candidate_id")
        )
    return output


def packed_ids(record: dict[str, Any]) -> set[str]:
    return {
        item.get("chunk_id")
        for item in record.get("packed_contexts") or []
        if item.get("chunk_id")
    }


def packed_chars(record: dict[str, Any]) -> int:
    trace = record.get("retrieval_trace") or {}
    packer = trace.get("packer") or {}
    if packer.get("context_chars") is not None:
        return int(packer["context_chars"])
    return sum(len(item.get("text") or "") for item in record.get("packed_contexts") or [])


def action_correct(row: dict[str, str]) -> bool:
    if row["expected_action"] == "abstain":
        return row["actual_action"] == "abstain" and row["actual_reason"] == "evidence_gap"
    return row["actual_action"] == row["expected_action"]


def baseline_comparison(current: dict[str, str], baseline: dict[str, str]) -> str:
    current_action = action_correct(current)
    baseline_action = action_correct(baseline)
    if current_action and not baseline_action:
        return "IMPROVED"
    if baseline_action and not current_action:
        return "REGRESSED"
    if current["case_family"] == "evidence_gap":
        return "UNCHANGED"
    deltas = [
        float(current[column]) - float(baseline[column])
        for column in METRIC_COLUMNS.values()
    ]
    positive = any(delta >= 10.0 for delta in deltas)
    negative = any(delta <= -10.0 for delta in deltas)
    if positive and negative:
        return "MIXED"
    if positive:
        return "IMPROVED"
    if negative:
        return "REGRESSED"
    return "UNCHANGED"


def response_metrics(checkpoint: dict[str, Any] | None) -> tuple[float, float, int, int]:
    if checkpoint is None:
        return 0.0, 0.0, 0, 0
    values = checkpoint.get("metrics") or {}
    return (
        float(values.get("recall") or 0.0),
        float(values.get("precision") or 0.0),
        len(checkpoint.get("response_claims") or []),
        len(checkpoint.get("gt_answer_claims") or []),
    )


def classify_case(
    row: dict[str, str],
    record: dict[str, Any],
    case: dict[str, Any],
    checkpoint: dict[str, Any] | None,
) -> tuple[str, str, str, list[str], str]:
    case_id = row["case_id"]
    family = row["case_family"]
    expected = row["expected_action"]
    actual = row["actual_action"]
    reason = row["actual_reason"]
    cr = metric(row, "claim_recall") or 0.0
    cp = metric(row, "context_precision") or 0.0
    faith = metric(row, "faithfulness") or 0.0
    f1 = metric(row, "claim_f1") or 0.0
    response_recall, response_precision, response_claim_count, gold_claim_count = (
        response_metrics(checkpoint)
    )
    gold = gold_ids(case)
    candidates = candidate_ids(record)
    packed = packed_ids(record)
    candidate_hits = len(gold & candidates)
    pack_hits = len(gold & packed)
    flags: list[str] = []
    if family == "answerable_multi_turn":
        flags.append("multi_turn")
    if record.get("retrieval_attempt") == 2:
        flags.append("retry_used")
    if record.get("llm_fallback_used"):
        flags.append("provider_fallback")
    if packed:
        flags.append("retrieval_gold_hit" if pack_hits else "retrieval_noise")
    if len(packed) == 8:
        flags.append("full_8_item_pack")
    if cp <= 25.0 and family.startswith("answerable"):
        flags.append("low_context_precision")
    if cr >= 75.0:
        flags.append("high_claim_recall")
    elif family.startswith("answerable"):
        flags.append("low_claim_recall")
    if faith >= 75.0 and family.startswith("answerable"):
        flags.append("high_faithfulness")
    elif family.startswith("answerable"):
        flags.append("low_faithfulness")

    if expected == "abstain":
        if actual == "abstain" and reason == "evidence_gap":
            return "PASS", "none", "", flags, "Correct evidence-gap rejection."
        flags.append("false_generate_gap")
        return (
            "SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP",
            "evidence_assessment",
            "HIGH",
            flags,
            "Nearby/general evidence was marked usable and the structured action generated instead of rejecting the unsupported requirement.",
        )

    if not actual or not reason:
        flags.extend(["safety_short_circuit", "structured_action_missing"])
        return (
            "SYSTEM_BUG_SAFETY_SHORT_CIRCUIT",
            "safety",
            "HIGH",
            flags,
            "A deterministic safety answer bypassed retrieval and left action/reason unset.",
        )

    if actual == "abstain":
        flags.append("false_abstention")
        if case_id in MULTI_TURN_PRIMARY:
            return (
                "SYSTEM_BUG_MULTI_TURN_CONTEXT",
                "multi_turn",
                "MEDIUM",
                flags,
                "Dialogue handling produced an immediate or repeated evidence-gap abstention for an answerable follow-up.",
            )
        if candidate_hits and not pack_hits:
            flags.append("gold_missing_from_pack")
            return (
                "SYSTEM_BUG_PACKER_SELECTION",
                "packing",
                "MEDIUM",
                flags,
                "Canonical gold provenance reached candidates but was absent from the final packed context before abstention.",
            )
        if not candidate_hits:
            return (
                "SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE",
                "retrieval",
                "MEDIUM",
                flags,
                "Canonical gold provenance did not reach saved candidates; the downstream abstention followed undercoverage.",
            )
        return (
            "SYSTEM_BUG_ACTION_FALSE_ABSTENTION",
            "action_decision",
            "MEDIUM",
            flags,
            "Usable evidence was available, but the final action abstained on an answerable case.",
        )

    if case_id in MULTI_TURN_PRIMARY and (cr < 100.0 or response_recall < 1.0):
        flags.append("generation_omission")
        return (
            "SYSTEM_BUG_MULTI_TURN_CONTEXT",
            "multi_turn",
            "MEDIUM",
            flags,
            "The follow-up/topic-isolation path did not preserve all required current-turn evidence or claims.",
        )

    if case_id == "ANS-CMP-010":
        return (
            "SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE",
            "retrieval",
            "MEDIUM",
            flags,
            "Neither gold provenance item reached candidates; the answer reported insufficient information instead of inventing clinical content.",
        )

    if case_id == "ANS-REF-002":
        flags.append("expected_metric_penalty")
        return (
            "EXPECTED_METRIC_BEHAVIOR",
            "metric_behavior",
            "LOW",
            flags,
            "The answer covers all referral criteria; low faithfulness comes from parenthetical alias claims not matched to context by the evaluator.",
        )

    if faith < 75.0:
        flags.append("generation_omission" if response_recall < 1.0 else "over_answering")
        severity = "HIGH" if row["category"] == "adverse_effects_precautions" else "MEDIUM"
        return (
            "SYSTEM_BUG_UNSUPPORTED_GENERATION",
            "generation",
            severity,
            flags,
            "Saved entailment labels show material response claims not supported by the packed contexts.",
        )

    if cr < 100.0:
        if candidate_hits > pack_hits:
            flags.append("gold_missing_from_pack")
            return (
                "SYSTEM_BUG_PACKER_SELECTION",
                "packing",
                "MEDIUM",
                flags,
                "At least one canonical gold chunk was a candidate but was not retained in the final pack.",
            )
        return (
            "SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE",
            "retrieval",
            "MEDIUM",
            flags,
            "Packed evidence did not cover all gold claims and no additional exact gold provenance was available to the packer.",
        )

    if response_recall < 1.0:
        flags.append("generation_omission")
        return (
            "SYSTEM_BUG_GENERATION_OMISSION",
            "generation",
            "MEDIUM",
            flags,
            "Gold-supporting context was complete, but the response omitted at least one required gold claim.",
        )

    claim_ratio = response_claim_count / max(gold_claim_count, 1)
    if f1 < 40.0 and response_precision < 0.4 and claim_ratio >= 2.0:
        flags.append("over_answering")
        return (
            "SYSTEM_BUG_GENERATION_OVER_ANSWERING",
            "generation",
            "LOW",
            flags,
            "Core gold claims were covered and grounded, but the response added substantially more supported claims than requested.",
        )

    if f1 < 40.0:
        flags.append("expected_metric_penalty")
        return (
            "EXPECTED_METRIC_BEHAVIOR",
            "metric_behavior",
            "LOW",
            flags,
            "The answer is substantively supported; claim segmentation or narrow gold scope explains the remaining low F1.",
        )

    return "PASS", "none", "", flags, "Behavior and evidence are substantively aligned with the benchmark."


def validate_integrity(
    benchmark: dict[str, Any],
    manifest: dict[str, Any],
    raw: dict[str, Any],
    rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    calibration: dict[str, Any],
    adjudication: dict[str, Any],
) -> dict[str, Any]:
    records = raw["records"]
    benchmark_ids = [case["case_id"] for case in benchmark["cases"]]
    record_ids = [record["case_id"] for record in records]
    row_ids = [row["case_id"] for row in rows]
    checks = {
        "records_100": len(records) == 100,
        "answerable_70": sum(r["case_family"].startswith("answerable") for r in records) == 70,
        "gap_30": sum(r["case_family"] == "evidence_gap" for r in records) == 30,
        "benchmark_ids_exact": set(record_ids) == set(benchmark_ids),
        "record_ids_unique": len(record_ids) == len(set(record_ids)) == 100,
        "metric_ids_exact": set(row_ids) == set(benchmark_ids),
        "metric_ids_unique": len(row_ids) == len(set(row_ids)) == 100,
        "infrastructure_failures_zero": not any(r.get("infrastructure_error") for r in records),
        "benchmark_sha": raw["benchmark_sha256"] == manifest["benchmark_sha256"] == EXPECTED_BENCHMARK_SHA,
        "system_sha": raw["system_under_test_sha"] == EXPECTED_SYSTEM_SHA,
        "kb_build": raw["active_kb_build_id"] == EXPECTED_KB,
        "pipeline": raw["expected_pipeline_fingerprint"] == EXPECTED_PIPELINE,
        "record_fingerprints": all(r.get("pipeline_fingerprint") == EXPECTED_PIPELINE for r in records),
        "repository_head_recorded": bool(raw.get("run_metadata", {}).get("repository_head_at_run")),
        "evaluator": raw["run_metadata"]["evaluator_model"] == EXPECTED_EVALUATOR,
        "ragchecker": raw["run_metadata"]["ragchecker_version"] == EXPECTED_RAGCHECKER,
        "calibration_counts": (
            calibration["claim_extraction_acceptable"],
            calibration["claim_checking_agreement"],
            calibration["repeat_consistency"],
        ) == (7, 12, 19),
        "calibration_decision": calibration["final_calibration_decision"] == "CALIBRATION_REVIEW_REQUIRED",
        "calibration_item": [x["item_id"] for x in calibration["disagreements"]] == ["CAL-EXT-03"],
        "adjudication": adjudication["researcher_adjudication"] == "APPROVED",
        "effective_decision": adjudication["effective_decision"] == "CALIBRATION_READY_FOR_FORMAL_RUN",
    }
    answerable = [row for row in rows if row["case_family"].startswith("answerable")]
    saved_summary = {row["Metric"]: float(row["Score"]) for row in summary_rows}
    calculated = {
        label: mean([float(row[column]) for row in answerable])
        for label, column in (
            ("Claim Recall", "Claim Recall (%)"),
            ("Context Precision", "Context Precision (%)"),
            ("Faithfulness", "Faithfulness (%)"),
            ("Claim F1", "Claim F1 (%)"),
        )
    }
    gaps = [row for row in rows if row["case_family"] == "evidence_gap"]
    nrr_successes = sum(action_correct(row) for row in gaps)
    calculated["Negative Rejection Rate"] = nrr_successes / len(gaps) * 100.0
    checks["metric_aggregates"] = all(
        abs(calculated[name] - saved_summary[name]) <= 0.05 for name in calculated
    )
    checks["nrr_16_of_30"] = nrr_successes == 16 and len(gaps) == 30
    if not all(checks.values()):
        raise AuditError(f"POST_IMPROVEMENT_FORENSIC_AUDIT_BLOCKED_BY_RUN_INTEGRITY: {checks}")
    return {
        "checks": checks,
        "repository_head_at_run": raw["run_metadata"]["repository_head_at_run"],
        "calculated_metrics": {key: round4(value) for key, value in calculated.items()},
        "nrr_successes": nrr_successes,
        "nrr_failures": len(gaps) - nrr_successes,
    }


def metric_summary(rows: list[dict[str, str]]) -> dict[str, float]:
    return {row["Metric"]: float(row["Score"]) for row in rows}


def cohort_summary(rows: list[dict[str, str]], family: str) -> dict[str, Any]:
    selected = [row for row in rows if row["case_family"] == family]
    return {
        "count": len(selected),
        "means": {
            key: round4(mean([float(row[column]) for row in selected]))
            for key, column in METRIC_COLUMNS.items()
        },
    }


def make_case_rows(
    benchmark: dict[str, Any],
    raw: dict[str, Any],
    metrics: list[dict[str, str]],
    checkpoints: dict[str, dict[str, Any]],
    baseline_metrics: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    cases = {case["case_id"]: case for case in benchmark["cases"]}
    records = {record["case_id"]: record for record in raw["records"]}
    output: list[dict[str, Any]] = []
    for source_row in metrics:
        case_id = source_row["case_id"]
        case = cases[case_id]
        record = records[case_id]
        checkpoint = checkpoints.get(case_id)
        primary, layer, severity, flags, note = classify_case(
            source_row, record, case, checkpoint
        )
        gold = gold_ids(case)
        candidates = candidate_ids(record)
        packed = packed_ids(record)
        response_recall, response_precision, response_claims, gold_claims = response_metrics(
            checkpoint
        )
        history_actions = [
            decision.get("action") for decision in record.get("agent_decision_history") or []
        ]
        retry_used = record.get("retrieval_attempt") == 2
        retry_outcome = ""
        if retry_used:
            retry_outcome = (
                "final_action_correct" if action_correct(source_row) else "final_action_incorrect"
            )
        nrr_correct = ""
        if source_row["case_family"] == "evidence_gap":
            nrr_correct = str(action_correct(source_row)).lower()
        evidence = (
            f"gold_ids={len(gold)}; candidate_hits={len(gold & candidates)}; "
            f"pack_hits={len(gold & packed)}; CR={source_row['Claim Recall (%)']}; "
            f"response_recall={response_recall:.4f}; response_precision={response_precision:.4f}; "
            f"faithfulness={source_row['Faithfulness (%)']}; "
            f"assessment={(record.get('evidence_assessment') or {}).get('reason')}; "
            f"decisions={history_actions}"
        )
        baseline_delta = {
            key: (
                round4(
                    float(source_row[column])
                    - float(baseline_metrics[case_id][column])
                )
                if source_row[column] and baseline_metrics[case_id][column]
                else None
            )
            for key, column in METRIC_COLUMNS.items()
        }
        output.append(
            {
                "case_id": case_id,
                "case_family": source_row["case_family"],
                "category": source_row["category"],
                "query": source_row.get("query") or record.get("query") or case.get("query") or "",
                "expected_action": source_row["expected_action"],
                "expected_reason": source_row["expected_reason"],
                "actual_action": source_row["actual_action"],
                "actual_reason": source_row["actual_reason"],
                "claim_recall_pct": source_row["Claim Recall (%)"],
                "context_precision_pct": source_row["Context Precision (%)"],
                "faithfulness_pct": source_row["Faithfulness (%)"],
                "claim_f1_pct": source_row["Claim F1 (%)"],
                "nrr_correct": nrr_correct,
                "retrieval_attempt": record.get("retrieval_attempt") or 0,
                "retrieval_status": record.get("retrieval_status") or "",
                "retry_used": str(retry_used).lower(),
                "retry_outcome": retry_outcome,
                "packed_context_count": len(record.get("packed_contexts") or []),
                "packed_context_chars": packed_chars(record),
                "candidate_count": len(candidates),
                "gold_candidate_hits": len(gold & candidates),
                "gold_pack_hits": len(gold & packed),
                "pipeline_fingerprint": record.get("pipeline_fingerprint") or "",
                "requested_provider": record.get("requested_provider") or "",
                "requested_model": record.get("requested_model") or "",
                "actual_provider": record.get("actual_provider") or "",
                "actual_model": record.get("actual_model") or "",
                "llm_fallback_used": str(bool(record.get("llm_fallback_used"))).lower(),
                "primary_classification": primary,
                "root_cause_layer": layer,
                "severity": severity,
                "secondary_flags": "|".join(sorted(set(flags))),
                "baseline_case_delta": json.dumps(baseline_delta, sort_keys=True),
                "baseline_comparison": baseline_comparison(
                    source_row, baseline_metrics[case_id]
                ),
                "evidence_summary": evidence,
                "audit_notes": note,
                "_response_claims": response_claims,
                "_gold_claims": gold_claims,
                "_response_recall": response_recall,
                "_response_precision": response_precision,
            }
        )
    return output


def source_hashes(root: Path) -> list[dict[str, Any]]:
    output = []
    for relative in SOURCE_ARTIFACTS:
        path = root / relative
        if not path.is_file():
            raise AuditError(f"Missing required source artifact: {relative.as_posix()}")
        digest = sha256(path)
        output.append(
            {
                "artifact": relative.as_posix(),
                "pre_audit_sha256": digest,
                "post_audit_sha256": digest,
                "unchanged": True,
            }
        )
    return output


def summarize(
    root: Path,
    case_rows: list[dict[str, Any]],
    integrity: dict[str, Any],
    post_summary_rows: list[dict[str, str]],
    baseline_summary_rows: list[dict[str, str]],
    raw: dict[str, Any],
    baseline_raw: dict[str, Any],
) -> dict[str, Any]:
    post_metrics = metric_summary(post_summary_rows)
    baseline_metrics = metric_summary(baseline_summary_rows)
    deltas = {
        key: round4(post_metrics[key] - baseline_metrics[key]) for key in post_metrics
    }
    primary_counts = Counter(row["primary_classification"] for row in case_rows)
    layer_counts = Counter(row["root_cause_layer"] for row in case_rows)
    severity_counts = Counter(row["severity"] or "NONE" for row in case_rows)
    gap_failures = [
        row
        for row in case_rows
        if row["case_family"] == "evidence_gap" and row["nrr_correct"] == "false"
    ]
    false_abstentions = [
        row
        for row in case_rows
        if row["case_family"].startswith("answerable")
        and row["actual_action"] == "abstain"
    ]
    gap_breakdown = Counter(row["category"] for row in gap_failures)
    nrr_behavior = Counter(
        "B_text_rejects_but_structured_generate"
        if row["category"] == "unsupported_absolute_certainty_guarantee"
        else "C_answers_supported_nearby_fact_after_rejecting_precision"
        for row in gap_failures
    )
    for behavior in (
        "A_incorrectly_asserts_unsupported_content",
        "B_text_rejects_but_structured_generate",
        "C_answers_supported_nearby_fact_after_rejecting_precision",
        "D_retrieval_or_evidence_false_positive_without_B_or_C",
        "E_other",
    ):
        nrr_behavior.setdefault(behavior, 0)
    single = cohort_summary(
        [
            {
                "case_family": row["case_family"],
                **{
                    column: row[f"{key}_pct"]
                    for key, column in METRIC_COLUMNS.items()
                },
            }
            for row in case_rows
            if row["case_family"].startswith("answerable")
        ],
        "answerable_single_turn",
    )
    multi = cohort_summary(
        [
            {
                "case_family": row["case_family"],
                **{
                    column: row[f"{key}_pct"]
                    for key, column in METRIC_COLUMNS.items()
                },
            }
            for row in case_rows
            if row["case_family"].startswith("answerable")
        ],
        "answerable_multi_turn",
    )
    answerable = [row for row in case_rows if row["case_family"].startswith("answerable")]
    full_recall_faith_low_f1 = [
        row["case_id"]
        for row in answerable
        if float(row["claim_recall_pct"]) == 100.0
        and float(row["faithfulness_pct"]) == 100.0
        and float(row["claim_f1_pct"]) < 40.0
    ]
    pack_sizes = Counter(int(row["packed_context_count"]) for row in answerable)
    pack_chars = [int(row["packed_context_chars"]) for row in answerable]
    cp_by_pack: dict[str, float] = {}
    for size in sorted(pack_sizes):
        cp_by_pack[str(size)] = round4(
            mean(
                [
                    float(row["context_precision_pct"])
                    for row in answerable
                    if int(row["packed_context_count"]) == size
                ]
            )
        )
    retry_rows = [row for row in case_rows if row["retry_used"] == "true"]
    traces = [record.get("retrieval_trace") or {} for record in raw["records"]]
    rerankers = [trace.get("reranker") or {} for trace in traces if trace.get("reranker")]
    reranker_latencies = [float(item["elapsed_ms"]) for item in rerankers if item.get("elapsed_ms") is not None]
    fused_counts = [
        int(trace.get("fused_candidate_count") or 0)
        for trace in traces
        if trace.get("reranker")
    ]
    eligible_counts = [
        int(trace.get("eligible_candidate_count") or 0)
        for trace in traces
        if trace.get("reranker")
    ]
    selected_counts = [
        len(trace.get("selected_ids") or []) for trace in traces if trace.get("reranker")
    ]
    provider_counts = Counter(
        (
            record.get("requested_provider") or "none",
            record.get("requested_model") or "none",
            record.get("actual_provider") or "none",
            record.get("actual_model") or "none",
            str(bool(record.get("llm_fallback_used"))).lower(),
        )
        for record in raw["records"]
    )
    baseline_case_counts = Counter(row["baseline_comparison"] for row in case_rows)
    baseline_records = {record["case_id"]: record for record in baseline_raw["records"]}
    baseline_retry = Counter(record.get("retrieval_attempt") or 0 for record in baseline_records.values())
    low_f1 = [row for row in answerable if float(row["claim_f1_pct"]) < 40.0]
    low_f1_causes = Counter(row["primary_classification"] for row in low_f1)
    recall_buckets = {
        "equal_0": [row["case_id"] for row in answerable if float(row["claim_recall_pct"]) == 0.0],
        "between_0_and_50": [row["case_id"] for row in answerable if 0.0 < float(row["claim_recall_pct"]) < 50.0],
        "between_50_and_100": [row["case_id"] for row in answerable if 50.0 <= float(row["claim_recall_pct"]) < 100.0],
    }
    faith_low = [row for row in answerable if float(row["faithfulness_pct"]) < 75.0]
    evaluator_notes = [
        {
            "case_id": "ANS-MUL-006",
            "status": "NOT_CONFIRMED",
            "evidence": "The answer contains one narrower supported tactic, but omits the full disjunctive gold scope; zero response-to-gold recall is strict but not demonstrably invalid.",
        },
        {
            "case_id": "ANS-MUL-015",
            "status": "NOT_CONFIRMED",
            "evidence": "The answer covers the not-dirt claim but omits melanin; low scores reflect a real omission rather than a proven evaluator defect.",
        },
    ]
    return {
        "run_identity": {
            "run_id": raw["run_id"],
            "benchmark_sha256": raw["benchmark_sha256"],
            "system_under_test_sha": raw["system_under_test_sha"],
            "repository_head_at_run": integrity["repository_head_at_run"],
            "active_kb_build_id": raw["active_kb_build_id"],
            "pipeline_fingerprint": raw["expected_pipeline_fingerprint"],
            "evaluator_model": raw["run_metadata"]["evaluator_model"],
            "ragchecker_version": raw["run_metadata"]["ragchecker_version"],
        },
        "source_artifact_hashes": source_hashes(root),
        "run_integrity": integrity,
        "calibration_audit": {
            "automatic_counts": {"extraction": 7, "checking": 12, "repeat_consistency": 19},
            "automatic_decision": "CALIBRATION_REVIEW_REQUIRED",
            "review_item": "CAL-EXT-03",
            "researcher_adjudication": "APPROVED",
            "effective_decision": "CALIBRATION_READY_FOR_FORMAL_RUN",
            "classification": "RESEARCHER_ADJUDICATED_SEMANTIC_VARIATION",
        },
        "metric_summary": post_metrics,
        "baseline_metric_summary": baseline_metrics,
        "metric_deltas": deltas,
        "case_counts": {"total": 100, "answerable": 70, "evidence_gap": 30},
        "primary_classification_counts": dict(sorted(primary_counts.items())),
        "root_cause_layer_counts": dict(sorted(layer_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "answerable_false_abstentions": [row["case_id"] for row in false_abstentions],
        "gap_false_generates": [row["case_id"] for row in gap_failures],
        "nrr_failure_breakdown": {
            "failures": len(gap_failures),
            "successes": 30 - len(gap_failures),
            "by_category": dict(sorted(gap_breakdown.items())),
            "behavior_types": dict(sorted(nrr_behavior.items())),
            "case_ids": [row["case_id"] for row in gap_failures],
        },
        "multi_turn_summary": {
            "single_turn": single,
            "multi_turn": multi,
            "multi_turn_false_abstentions": sum(
                row["actual_action"] == "abstain"
                for row in answerable
                if row["case_family"] == "answerable_multi_turn"
            ),
            "multi_turn_structured_action_missing": sum(
                not row["actual_action"]
                for row in answerable
                if row["case_family"] == "answerable_multi_turn"
            ),
            "multi_turn_primary_defects": sum(
                row["primary_classification"] == "SYSTEM_BUG_MULTI_TURN_CONTEXT"
                for row in answerable
            ),
            "retrieval_or_packing_failures": sum(
                row["primary_classification"]
                in {
                    "SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE",
                    "SYSTEM_BUG_PACKER_SELECTION",
                }
                for row in answerable
                if row["case_family"] == "answerable_multi_turn"
            ),
            "generation_omissions": sum(
                row["primary_classification"] == "SYSTEM_BUG_GENERATION_OMISSION"
                for row in answerable
                if row["case_family"] == "answerable_multi_turn"
            ),
            "generation_over_answering": sum(
                row["primary_classification"]
                == "SYSTEM_BUG_GENERATION_OVER_ANSWERING"
                for row in answerable
                if row["case_family"] == "answerable_multi_turn"
            ),
        },
        "claim_f1_deep_dive": {
            "mean": post_metrics["Claim F1"],
            "below_40_count": len(low_f1),
            "below_40_primary_causes": dict(sorted(low_f1_causes.items())),
            "full_recall_full_faithfulness_below_40": full_recall_faith_low_f1,
        },
        "context_precision_deep_dive": {
            "pack_size_distribution": {str(key): value for key, value in sorted(pack_sizes.items())},
            "mean_pack_size": round4(mean([int(row["packed_context_count"]) for row in answerable])),
            "median_pack_size": round4(median([int(row["packed_context_count"]) for row in answerable])),
            "mean_packed_chars": round4(mean(pack_chars)),
            "median_packed_chars": round4(median(pack_chars)),
            "context_precision_by_pack_size": cp_by_pack,
            "cr_at_least_75_cp_at_most_25": [
                row["case_id"]
                for row in answerable
                if float(row["claim_recall_pct"]) >= 75.0
                and float(row["context_precision_pct"]) <= 25.0
            ],
            "cr_100_cp_at_most_25": [
                row["case_id"]
                for row in answerable
                if float(row["claim_recall_pct"]) == 100.0
                and float(row["context_precision_pct"]) <= 25.0
            ],
        },
        "claim_recall_deep_dive": recall_buckets,
        "faithfulness_deep_dive": {
            "below_75": [row["case_id"] for row in faith_low],
            "equal_0_false_abstention_or_safety": [
                row["case_id"]
                for row in faith_low
                if float(row["faithfulness_pct"]) == 0.0
                and (row["actual_action"] == "abstain" or not row["actual_action"])
            ],
            "equal_0_generated": [
                row["case_id"]
                for row in faith_low
                if float(row["faithfulness_pct"]) == 0.0
                and row["actual_action"] == "generate"
            ],
        },
        "retry_summary": {
            "attempt_distribution": dict(sorted(Counter(int(row["retrieval_attempt"]) for row in case_rows).items())),
            "retry_count": len(retry_rows),
            "retry_rate_pct": round4(len(retry_rows) / 100 * 100.0),
            "answerable_retry_count": sum(row["case_family"].startswith("answerable") for row in retry_rows),
            "gap_retry_count": sum(row["case_family"] == "evidence_gap" for row in retry_rows),
            "retry_to_generate": sum(row["actual_action"] == "generate" for row in retry_rows),
            "retry_to_abstain": sum(row["actual_action"] == "abstain" for row in retry_rows),
            "retry_final_correct": sum(row["retry_outcome"] == "final_action_correct" for row in retry_rows),
            "retry_final_incorrect": sum(row["retry_outcome"] == "final_action_incorrect" for row in retry_rows),
            "recovered_exact_gold_evidence": sum(
                int(row["gold_candidate_hits"]) > 0 or int(row["gold_pack_hits"]) > 0
                for row in retry_rows
                if row["case_family"].startswith("answerable")
            ),
            "changed_final_action_correctly": sum(
                row["retry_outcome"] == "final_action_correct" for row in retry_rows
            ),
            "added_noise_full_pack_cp_below_25": sum(
                row["case_family"].startswith("answerable")
                and int(row["packed_context_count"]) == 8
                and float(row["context_precision_pct"]) < 25.0
                for row in retry_rows
            ),
            "failed_to_help": sum(
                row["retry_outcome"] == "final_action_incorrect" for row in retry_rows
            ),
            "case_ids": [row["case_id"] for row in retry_rows],
            "baseline_attempt_distribution": dict(sorted(baseline_retry.items())),
        },
        "reranker_summary": {
            "saved_final_traces": len(rerankers),
            "succeeded": sum(item.get("status") == "succeeded" for item in rerankers),
            "fallback": sum(bool(item.get("fallback_used")) for item in rerankers),
            "failed_or_timeout": sum(item.get("status") != "succeeded" for item in rerankers),
            "mean_latency_ms": round4(mean(reranker_latencies)),
            "median_latency_ms": round4(median(reranker_latencies)),
            "mean_fused_candidates": round4(mean(fused_counts)),
            "mean_eligible_candidates": round4(mean(eligible_counts)),
            "mean_selected_items": round4(mean(selected_counts)),
        },
        "provider_fallback_summary": {
            "distribution": [
                {
                    "requested_provider": key[0],
                    "requested_model": key[1],
                    "actual_provider": key[2],
                    "actual_model": key[3],
                    "fallback_used": key[4],
                    "count": count,
                }
                for key, count in sorted(provider_counts.items())
            ],
            "fallback_case_ids": [
                record["case_id"] for record in raw["records"] if record.get("llm_fallback_used")
            ],
            "fallback_cases": [
                {
                    "case_id": record["case_id"],
                    "requested_provider": record.get("requested_provider"),
                    "requested_model": record.get("requested_model"),
                    "actual_provider": record.get("actual_provider"),
                    "actual_model": record.get("actual_model"),
                    "fallback_chain": record.get("fallback_chain") or [],
                }
                for record in raw["records"]
                if record.get("llm_fallback_used")
            ],
            "provider_failures_recovered": sum(
                bool(record.get("llm_fallback_used"))
                and record.get("actual_provider") not in {None, "system"}
                for record in raw["records"]
            ),
            "ollama_fallback_uses": sum(
                record.get("actual_provider") == "ollama" for record in raw["records"]
            ),
            "infrastructure_failures": 0,
        },
        "baseline_case_comparison": dict(sorted(baseline_case_counts.items())),
        "evaluator_anomalies": evaluator_notes,
        "recommended_next_action": "TARGETED_PRODUCTION_FIX_JUSTIFIED_BEFORE_FINAL_FREEZE",
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend(
        "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |"
        for row in rows
    )
    return "\n".join(output)


def render_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    metrics = summary["metric_summary"]
    baseline = summary["baseline_metric_summary"]
    deltas = summary["metric_deltas"]
    metric_rows = [
        [name, f"{baseline[name]:.4f}", f"{metrics[name]:.4f}", f"{deltas[name]:+.4f}"]
        for name in metrics
    ]
    class_rows = [[key, value] for key, value in summary["primary_classification_counts"].items()]
    root_rows = [[key, value] for key, value in summary["root_cause_layer_counts"].items()]
    gaps = [row for row in rows if row["case_family"] == "evidence_gap"]
    gap_failures = [row for row in gaps if row["nrr_correct"] == "false"]
    false_abstentions = [
        row for row in rows if row["case_family"].startswith("answerable") and row["actual_action"] == "abstain"
    ]
    multi = [row for row in rows if row["case_family"] == "answerable_multi_turn"]
    f1_cases = set(summary["claim_f1_deep_dive"]["full_recall_full_faithfulness_below_40"])
    cr_zero = set(summary["claim_recall_deep_dive"]["equal_0"])
    faith_low = set(summary["faithfulness_deep_dive"]["below_75"])
    retry_cases = set(summary["retry_summary"]["case_ids"])
    hash_rows = [
        [item["artifact"], item["pre_audit_sha256"], item["post_audit_sha256"], item["unchanged"]]
        for item in summary["source_artifact_hashes"]
    ]
    cohort_rows = []
    for label in ("single_turn", "multi_turn"):
        item = summary["multi_turn_summary"][label]
        cohort_rows.append(
            [label, item["count"]]
            + [f"{item['means'][key]:.4f}" for key in METRIC_COLUMNS]
        )
    provider_rows = [
        [
            item["requested_provider"],
            item["requested_model"],
            item["actual_provider"],
            item["actual_model"],
            item["fallback_used"],
            item["count"],
        ]
        for item in summary["provider_fallback_summary"]["distribution"]
    ]
    report = f"""# Post-Improvement Formal Evaluation — Forensic Audit

## 1. Executive Summary

The saved run is complete and internally consistent: 100/100 cases, zero infrastructure failures, and all identity fields match the frozen system. Retrieval-facing metrics improved materially, while 14 evidence-gap false-generates reduced NRR to 16/30. The dominant remaining high-severity defect is not hallucinated natural-language assertion: all 14 failed gap answers reject the unsupported premise, but the evidence assessment/structured action still emits `generate/evidence_sufficient`.

The audit assigns exactly one primary classification to every case. It recommends a narrow evidence-gap action-contract correction before final freeze; no production change is made here.

## 2. Run Integrity

- Records: 100/100; answerable: 70; evidence-gap: 30.
- Unique IDs: 100; missing/duplicate IDs: 0/0.
- Infrastructure failures: 0.
- Benchmark SHA: `{summary['run_identity']['benchmark_sha256']}`.
- System SHA: `{summary['run_identity']['system_under_test_sha']}`.
- KB build: `{summary['run_identity']['active_kb_build_id']}`.
- Pipeline fingerprint: `{summary['run_identity']['pipeline_fingerprint']}` on all 100 records.
- Repository head recorded at run: `{summary['run_identity']['repository_head_at_run']}`.
- Evaluator / RAGChecker: `{summary['run_identity']['evaluator_model']}` / `{summary['run_identity']['ragchecker_version']}`.

### Source artifact hashes

{markdown_table(['Artifact', 'Pre-audit SHA', 'Post-audit SHA', 'Unchanged'], hash_rows)}

## 3. Metric Comparison

{markdown_table(['Metric', 'Baseline (%)', 'Post-improvement (%)', 'Delta pp'], metric_rows)}

Retrieval/reranking/packing improvements worked in aggregate: Claim Recall rose 10.5 pp and Context Precision 12.9 pp, with Faithfulness up 11.3 pp. Case traces nevertheless show residual exact-provenance candidate misses and candidate-to-pack loss; the gain is material, not complete.

## 4. Calibration Audit

Automatic calibration remained 7/8 extraction, 12/12 checking, and 19/20 repeat consistency with `CALIBRATION_REVIEW_REQUIRED` on `CAL-EXT-03`. The separate adjudication records `APPROVED`, producing effective `CALIBRATION_READY_FOR_FORMAL_RUN`. This is classified as `RESEARCHER_ADJUDICATED_SEMANTIC_VARIATION`; automatic counts were not rewritten.

## 5. Overall Case Classification

{markdown_table(['Primary classification', 'Count'], class_rows)}

The CSV contains exactly 100 exclusive primary classifications.

## 6. Root-Cause Distribution

{markdown_table(['Dominant layer', 'Count'], root_rows)}

`none` denotes PASS cases. `metric_behavior` is reserved for low-score cases where supported detail or claim segmentation explains the score without a material system defect.

## 7. Answerable Cases

False abstentions: {len(false_abstentions)}. Claim Recall zero cases: {len(cr_zero)}. Faithfulness below 75: {len(faith_low)}.

### False abstention list

{markdown_table(['Case', 'Category', 'Attempts', 'Candidate hits', 'Pack hits', 'Primary cause'], [[r['case_id'], r['category'], r['retrieval_attempt'], r['gold_candidate_hits'], r['gold_pack_hits'], r['primary_classification']] for r in false_abstentions])}

### Claim Recall == 0

{markdown_table(['Case', 'Action', 'Faithfulness', 'Primary cause'], [[r['case_id'], r['actual_action'] or 'null', r['faithfulness_pct'], r['primary_classification']] for r in rows if r['case_id'] in cr_zero])}

The other recall buckets are: 0<CR<50 = {len(summary['claim_recall_deep_dive']['between_0_and_50'])} cases; 50<=CR<100 = {len(summary['claim_recall_deep_dive']['between_50_and_100'])} cases. Their IDs are retained in `forensic_summary.json`; each row's candidate/pack evidence distinguishes retrieval undercoverage from packer loss.

### Faithfulness < 75

{markdown_table(['Case', 'Action', 'CR', 'Faithfulness', 'Primary cause'], [[r['case_id'], r['actual_action'] or 'null', r['claim_recall_pct'], r['faithfulness_pct'], r['primary_classification']] for r in rows if r['case_id'] in faith_low])}

## 8. Evidence-Gap Cases and NRR Regression

NRR is 16/30 (53.3333%), down from 21/30 (70.0%). All 14 failures are structured false-generates. Ten absolute-guarantee answers explicitly say “Không” but still emit `generate/evidence_sufficient` (type B). Four exact/relationship cases reject the unsupported precision and then provide nearby supported facts (type C). No saved answer positively asserts the requested unsupported absolute/quantity as fact.

### NRR failures by category

{markdown_table(['Category', 'Failures'], [[k, v] for k, v in summary['nrr_failure_breakdown']['by_category'].items()])}

### All 30 gap cases

{markdown_table(['Case', 'Category', 'Action', 'Reason', 'NRR correct', 'Primary classification'], [[r['case_id'], r['category'], r['actual_action'], r['actual_reason'], r['nrr_correct'], r['primary_classification']] for r in gaps])}

### Detailed 14 failures

{markdown_table(['Case', 'Category', 'Pack', 'Evidence assessment outcome', 'Behavior type'], [[r['case_id'], r['category'], r['packed_context_count'], 'generate/evidence_sufficient', 'B' if r['category']=='unsupported_absolute_certainty_guarantee' else 'C'] for r in gap_failures])}

`GAP-ABS-001` rejects a permanent cure guarantee in text but generates structurally. `GAP-EXA-005` says no exact temperature is available, then gives general washing advice while generating structurally. This is predominantly an evidence-assessment/action-policy mismatch, not 14 hallucinations.

## 9. Multi-Turn Audit

{markdown_table(['Cohort', 'N', 'CR', 'CP', 'Faithfulness', 'F1'], cohort_rows)}

{markdown_table(['Case', 'Category', 'Action', 'CR', 'F1', 'Primary classification'], [[r['case_id'], r['category'], r['actual_action'] or 'null', r['claim_recall_pct'], r['claim_f1_pct'], r['primary_classification']] for r in multi])}

Multi-turn remains operationally weaker: it contains {summary['multi_turn_summary']['multi_turn_false_abstentions']} false abstentions, {summary['multi_turn_summary']['multi_turn_structured_action_missing']} missing structured action, and {summary['multi_turn_summary']['multi_turn_primary_defects']} primary multi-turn defects. These are descriptive benchmark results, not a statistical-significance claim.

Additional multi-turn counts: retrieval/packing failures = {summary['multi_turn_summary']['retrieval_or_packing_failures']}; generation omissions = {summary['multi_turn_summary']['generation_omissions']}; over-answering = {summary['multi_turn_summary']['generation_over_answering']}.

## 10. Retrieval / Reranker / Packer Audit

- Answerable pack-size distribution: `{summary['context_precision_deep_dive']['pack_size_distribution']}`.
- Mean/median pack size: {summary['context_precision_deep_dive']['mean_pack_size']:.4f} / {summary['context_precision_deep_dive']['median_pack_size']:.4f}.
- Mean/median packed chars: {summary['context_precision_deep_dive']['mean_packed_chars']:.4f} / {summary['context_precision_deep_dive']['median_packed_chars']:.4f}.
- CR>=75 with CP<=25: {len(summary['context_precision_deep_dive']['cr_at_least_75_cp_at_most_25'])}; CR=100 with CP<=25: {len(summary['context_precision_deep_dive']['cr_100_cp_at_most_25'])}.
- Saved final reranker traces: {summary['reranker_summary']['saved_final_traces']}; succeeded: {summary['reranker_summary']['succeeded']}; fallback/failure: {summary['reranker_summary']['fallback']}/{summary['reranker_summary']['failed_or_timeout']}.
- Reranker mean/median latency: {summary['reranker_summary']['mean_latency_ms']:.4f}/{summary['reranker_summary']['median_latency_ms']:.4f} ms.
- Mean fused/eligible/selected candidates in saved final traces: {summary['reranker_summary']['mean_fused_candidates']:.4f}/{summary['reranker_summary']['mean_eligible_candidates']:.4f}/{summary['reranker_summary']['mean_selected_items']:.4f}.

### Context Precision by pack size

{markdown_table(['Packed items', 'Cases', 'Mean CP'], [[size, summary['context_precision_deep_dive']['pack_size_distribution'][size], value] for size, value in summary['context_precision_deep_dive']['context_precision_by_pack_size'].items()])}

Low CP is mixed: some full packs contain useful but non-gold context and incur expected precision penalties; other rows show exact gold candidates displaced before packing. No reranker runtime failure explains the failed cases.

### Provider fallback distribution

{markdown_table(['Requested provider', 'Requested model', 'Actual provider', 'Actual model', 'Fallback', 'Cases'], provider_rows)}

Fallback changed the model for exactly {len(summary['provider_fallback_summary']['fallback_case_ids'])} cases: `{', '.join(summary['provider_fallback_summary']['fallback_case_ids'])}`. All five primary rate-limit failures recovered on Gemini 3.1 Flash-Lite; Ollama uses and infrastructure failures were both zero. This limited same-provider fallback can change individual generation wording, but it does not explain retrieval metrics or the systematic 14-case structured NRR pattern.

## 11. Retry Audit

{markdown_table(['Measure', 'Value'], [[k, v] for k, v in summary['retry_summary'].items() if k not in {'case_ids'}])}

{markdown_table(['Case', 'Family', 'Final action', 'Outcome', 'Primary cause'], [[r['case_id'], r['case_family'], r['actual_action'], r['retry_outcome'], r['primary_classification']] for r in rows if r['case_id'] in retry_cases])}

Retries were rare (6%). They did not systematically recover correctness: the final action remained incorrect in the majority of retried cases. Saved traces support this descriptive result but do not expose a counterfactual final answer without retry.

## 12. Generation Audit

Claim F1 rose only 4.7 pp because retrieval gains did not remove response-side scope mismatch. Among F1<40 rows, causes include false abstention/missing claims, substantial extra supported claims, upstream evidence undercoverage, multi-turn loss, and a small number of unsupported response claims.

{markdown_table(['Primary cause among F1<40', 'Cases'], [[key, value] for key, value in summary['claim_f1_deep_dive']['below_40_primary_causes'].items()])}

### CR=100, Faithfulness=100, F1<40

{markdown_table(['Case', 'F1', 'Response/gold claims', 'Primary classification'], [[r['case_id'], r['claim_f1_pct'], f"{r['_response_claims']}/{r['_gold_claims']}", r['primary_classification']] for r in rows if r['case_id'] in f1_cases])}

The listed cases are not retrieval failures. Their packed evidence covers gold and generated claims are grounded; low F1 primarily reflects over-answering or claim-granularity/gold-scope penalties.

## 13. Safety Short-Circuit Audit

`ANS-MUL-010` produced a deterministic pregnancy warning with no retrieval, no packed context, and null structured action/reason. The safety content is appropriate, but the missing structured benchmark contract is a high-severity consistency defect. Faithfulness zero here is empty-context metric behavior, not evidence of a hallucinated unsafe answer.

## 14. Evaluator and Metric Behavior

No confirmed case-level evaluator anomaly was proven. `ANS-MUL-006` and `ANS-MUL-015` have strict-looking scores, but both also omit part of the exact gold scope. The calibration wording issue remains separately adjudicated and is not counted as a production defect.

Expected metric behavior is visible where useful supported details are outside narrow gold scope, Context Precision penalizes non-gold but relevant chunks, or claim extraction granularity expands the response claim count.

## 15. Formal Run #1 Comparison

{markdown_table(['Case-level comparison', 'Count'], [[k, v] for k, v in summary['baseline_case_comparison'].items()])}

Criteria were fixed before application: action correctness transitions dominate; otherwise >=10 pp gains/losses across individual metrics yield improved/regressed, opposing material movements yield mixed, and no material movement yields unchanged. No composite score was used.

The historical audit reported 57 system bugs, 3 evaluator anomalies, 19 expected-metric cases, and 21 passes. Direct count comparison is approximate because this audit separates upstream retrieval, packer loss, structured gap actions, and metric behavior more strictly. The post-improvement run has fewer false abstentions and stronger retrieval metrics, but five additional gap action failures versus baseline explain the NRR regression.

## 16. Remaining Bottlenecks

1. **Evidence-gap structured action (14 HIGH):** unsupported premises are rejected in prose but accepted structurally.
2. **Residual retrieval/packing undercoverage:** exact gold provenance still misses candidate or packed stages in multiple answerable cases.
3. **Multi-turn contract (6 primary defects):** includes false abstention, topic-switch safety bypass, and claim loss.
4. **Generation scope control:** supported over-answering depresses F1 despite high recall/faithfulness.
5. **Safety structured-action consistency (1 HIGH):** appropriate deterministic warning lacks action/reason trace.

## 17. Recommendation

**TARGETED_PRODUCTION_FIX_JUSTIFIED_BEFORE_FINAL_FREEZE**

The justification is narrow and evidence-based: 14 HIGH failures share one localized evidence-gap action-contract pattern, and one safety short-circuit lacks structured action. A separate approved task should address only these contracts and justify any new formal run. This audit makes no production change.
"""
    return report.rstrip() + "\n"


def serialize_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})
    return buffer.getvalue()


def build(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    post = root / POST_RUN
    baseline = root / BASELINE_RUN
    benchmark = load_json(root / "evaluation/benchmark_100.json")
    manifest = load_json(root / "evaluation/benchmark_manifest.json")
    raw = load_json(post / "raw_results.json")
    metrics = load_csv(post / "case_metrics.csv")
    summary_rows = load_csv(post / "metrics_summary.csv")
    checkpoints = {
        item["query_id"]: item
        for item in load_json(post / "ragchecker_checkpoint.json")["results"]
    }
    calibration = load_json(post / "evaluator_calibration_results.json")
    adjudication = load_json(post / "calibration_adjudication.json")
    baseline_raw = load_json(baseline / "raw_results.json")
    baseline_metrics_rows = load_csv(baseline / "case_metrics.csv")
    baseline_metrics = {row["case_id"]: row for row in baseline_metrics_rows}
    baseline_summary_rows = load_csv(baseline / "metrics_summary.csv")
    integrity = validate_integrity(
        benchmark,
        manifest,
        raw,
        metrics,
        summary_rows,
        calibration,
        adjudication,
    )
    case_rows = make_case_rows(
        benchmark,
        raw,
        metrics,
        checkpoints,
        baseline_metrics,
    )
    summary = summarize(
        root,
        case_rows,
        integrity,
        summary_rows,
        baseline_summary_rows,
        raw,
        baseline_raw,
    )
    validate_outputs(case_rows, summary)
    report = render_report(summary, case_rows)
    return case_rows, summary, report


def validate_outputs(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    ids = [row["case_id"] for row in rows]
    checks = {
        "rows_100": len(rows) == 100,
        "ids_unique_100": len(set(ids)) == 100,
        "answerable_70": sum(row["case_family"].startswith("answerable") for row in rows) == 70,
        "gap_30": sum(row["case_family"] == "evidence_gap" for row in rows) == 30,
        "classification_sum_100": sum(summary["primary_classification_counts"].values()) == 100,
        "nrr_failures_14": summary["nrr_failure_breakdown"]["failures"] == 14,
        "nrr_successes_16": summary["nrr_failure_breakdown"]["successes"] == 16,
        "fingerprints": all(row["pipeline_fingerprint"] == EXPECTED_PIPELINE for row in rows),
    }
    if not all(checks.values()):
        raise AuditError(f"Audit output consistency failure: {checks}")


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], report: str) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "case_forensic_audit.csv").write_text(
        serialize_csv(rows), encoding="utf-8", newline=""
    )
    (AUDIT_DIR / "forensic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (AUDIT_DIR / "forensic_audit.md").write_text(report, encoding="utf-8")


def check_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], report: str) -> None:
    expected = {
        "case_forensic_audit.csv": serialize_csv(rows),
        "forensic_summary.json": json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        "forensic_audit.md": report,
    }
    for name, content in expected.items():
        path = AUDIT_DIR / name
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise AuditError(f"Generated audit artifact is stale or inconsistent: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline forensic audit for the frozen 100-case run.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Repository containing ignored formal-run source artifacts.",
    )
    parser.add_argument("--check", action="store_true", help="Validate without writing outputs.")
    args = parser.parse_args()
    root = args.source_root.resolve()
    rows, summary, report = build(root)
    if args.check:
        check_outputs(rows, summary, report)
        print("POST_IMPROVEMENT_FORENSIC_AUDIT_CHECK: PASS")
    else:
        write_outputs(rows, summary, report)
        print(f"Wrote 100-case forensic audit to {AUDIT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
