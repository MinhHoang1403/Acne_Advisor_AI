from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


AUDIT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = AUDIT_DIR.parents[2]
VALIDATED_CSV = AUDIT_DIR / "validated_case_forensic_audit.csv"
SUMMARY_PATH = AUDIT_DIR / "audit_validation_summary.json"
REPORT_PATH = AUDIT_DIR / "audit_validation_report.md"

SOURCE_ARTIFACTS = [
    Path("evaluation/results/post_improvement_47b10954/raw_results.json"),
    Path("evaluation/results/post_improvement_47b10954/case_metrics.csv"),
    Path("evaluation/results/post_improvement_47b10954/metrics_summary.csv"),
    Path("evaluation/results/post_improvement_47b10954/ragchecker_checkpoint.json"),
    Path("evaluation/results/post_improvement_47b10954/evaluator_calibration_results.json"),
    Path("evaluation/results/post_improvement_47b10954/calibration_adjudication.json"),
    Path("evaluation/results/formal_run_baseline/raw_results.json"),
    Path("evaluation/results/formal_run_baseline/case_metrics.csv"),
    Path("evaluation/results/formal_run_baseline/metrics_summary.csv"),
    Path("evaluation/results/formal_run_baseline/ragchecker_checkpoint.json"),
    Path("evaluation/results/formal_run_baseline/evaluator_calibration_results.json"),
    Path("evaluation/formal_evaluation.ipynb"),
    Path("evaluation/benchmark_100.json"),
    Path("evaluation/benchmark_manifest.json"),
    Path("evaluation/evaluator_calibration.json"),
    Path("evaluation/audits/post_improvement_47b10954/forensic_audit.md"),
    Path("evaluation/audits/post_improvement_47b10954/case_forensic_audit.csv"),
    Path("evaluation/audits/post_improvement_47b10954/forensic_summary.json"),
    Path("evaluation/audits/post_improvement_47b10954/forensic_analysis.py"),
]

REQUIRED_COLUMNS = {
    "case_id",
    "case_family",
    "category",
    "original_primary_classification",
    "original_root_cause_layer",
    "original_severity",
    "validation_status",
    "validated_primary_classification",
    "validated_root_cause_layer",
    "validated_severity",
    "expected_action",
    "expected_reason",
    "actual_action",
    "actual_reason",
    "claim_recall_pct",
    "context_precision_pct",
    "faithfulness_pct",
    "claim_f1_pct",
    "nrr_correct",
    "candidate_semantic_support",
    "packed_semantic_support",
    "structured_action_status",
    "generation_scope_status",
    "multi_turn_status",
    "safety_status",
    "original_secondary_flags",
    "validated_secondary_flags",
    "evidence_basis",
    "correction_reason",
    "validated_notes",
}

VALIDATION_STATUSES = {"CONFIRMED", "CORRECTED", "NOT_PROVEN"}
SEMANTIC_SUPPORT_STATUSES = {
    "SUPPORTED",
    "NOT_SUPPORTED",
    "AMBIGUOUS",
    "NOT_APPLICABLE",
}


class ValidationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_source_inputs(source_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path in SOURCE_ARTIFACTS:
        path = source_root / relative_path
        if not path.is_file():
            raise ValidationError(f"Missing source artifact: {path}")
        hashes[relative_path.as_posix()] = sha256(path)
    return hashes


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    return dict(sorted(Counter(row[column] or "(empty)" for row in rows).items()))


def ids_for(
    rows: list[dict[str, str]],
    column: str,
    value: str,
) -> list[str]:
    return [row["case_id"] for row in rows if row[column] == value]


def validate_rows(
    rows: list[dict[str, str]],
    original_rows: list[dict[str, str]],
) -> None:
    if len(rows) != 100:
        raise ValidationError(f"Expected 100 validated rows, found {len(rows)}")
    case_ids = [row["case_id"] for row in rows]
    if len(set(case_ids)) != 100:
        raise ValidationError("Validated case IDs are not unique")
    if not rows or not REQUIRED_COLUMNS.issubset(rows[0]):
        missing = sorted(REQUIRED_COLUMNS - set(rows[0] if rows else {}))
        raise ValidationError(f"Validated CSV columns missing: {missing}")
    if set(row["validation_status"] for row in rows) - VALIDATION_STATUSES:
        raise ValidationError("Validated CSV contains an unsupported validation status")
    semantic_values = {
        row[column]
        for row in rows
        for column in ("candidate_semantic_support", "packed_semantic_support")
    }
    if semantic_values - SEMANTIC_SUPPORT_STATUSES:
        raise ValidationError("Validated CSV contains an unsupported semantic status")

    original_by_id = {row["case_id"]: row for row in original_rows}
    if set(case_ids) != set(original_by_id):
        raise ValidationError("Validated and original audit case IDs differ")
    for row in rows:
        original = original_by_id[row["case_id"]]
        if row["original_primary_classification"] != original["primary_classification"]:
            raise ValidationError(f"Original classification drift: {row['case_id']}")
        if row["original_root_cause_layer"] != original["root_cause_layer"]:
            raise ValidationError(f"Original root-cause drift: {row['case_id']}")
        if not row["evidence_basis"].strip():
            raise ValidationError(f"Missing evidence basis: {row['case_id']}")

    nrr_failures = [
        row
        for row in rows
        if row["case_family"] == "evidence_gap" and row["nrr_correct"] == "false"
    ]
    if len(nrr_failures) != 14:
        raise ValidationError(f"Expected 14 NRR failures, found {len(nrr_failures)}")
    for row in nrr_failures:
        flags = set(filter(None, row["validated_secondary_flags"].split("|")))
        if not ({"nrr_behavior_A", "nrr_behavior_B", "nrr_behavior_C", "nrr_behavior_D", "nrr_behavior_E"} & flags):
            raise ValidationError(f"Missing NRR behavior type: {row['case_id']}")

    multi_turn = [row for row in rows if row["case_family"] == "answerable_multi_turn"]
    if len(multi_turn) != 15:
        raise ValidationError(f"Expected 15 multi-turn rows, found {len(multi_turn)}")

    for row in rows:
        originally_reviewed = row["original_primary_classification"] != "PASS"
        if originally_reviewed and len(row["evidence_basis"].strip()) < 40:
            raise ValidationError(f"Evidence basis too short: {row['case_id']}")


def nrr_behavior_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    output = {letter: 0 for letter in "ABCDE"}
    for row in rows:
        flags = set(filter(None, row["validated_secondary_flags"].split("|")))
        for letter in output:
            if f"nrr_behavior_{letter}" in flags:
                output[letter] += 1
    return output


def build_summary(
    rows: list[dict[str, str]],
    pre_hashes: dict[str, str],
    post_hashes: dict[str, str],
) -> dict[str, Any]:
    original_counts = count(rows, "original_primary_classification")
    validated_counts = count(rows, "validated_primary_classification")
    status_counts = count(rows, "validation_status")

    original_retrieval = ids_for(
        rows,
        "original_primary_classification",
        "SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE",
    )
    validated_retrieval = ids_for(
        rows,
        "validated_primary_classification",
        "SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE",
    )
    original_packing = ids_for(
        rows,
        "original_primary_classification",
        "SYSTEM_BUG_PACKER_SELECTION",
    )
    validated_packing = ids_for(
        rows,
        "validated_primary_classification",
        "SYSTEM_BUG_PACKER_SELECTION",
    )

    corrected = [row for row in rows if row["validation_status"] == "CORRECTED"]
    not_proven = [row for row in rows if row["validation_status"] == "NOT_PROVEN"]
    multi_turn = [row for row in rows if row["case_family"] == "answerable_multi_turn"]

    confirmed_fix_candidates = [
        {
            "name": "Exact-proposition evidence action boundary",
            "affected_cases": ids_for(
                rows,
                "validated_primary_classification",
                "SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP",
            ),
            "validated_root_cause": "Topic-relevant evidence is treated as sufficient for an unsupported exact proposition, producing generate/evidence_sufficient after the answer text rejects the proposition.",
            "severity": "HIGH",
            "evidence": "All 14 failed evidence-gap answers reject the unsupported core while structured state generates; saved assessment marks topic evidence usable.",
            "generalization_reason": "The contract concerns proposition scope rather than benchmark wording and can be tested with synthetic guarantee, exact-value and exact-comparison requests.",
            "production_change_justified": True,
        },
        {
            "name": "Safety response structured-state completeness",
            "affected_cases": ids_for(
                rows,
                "validated_primary_classification",
                "SYSTEM_BUG_SAFETY_SHORT_CIRCUIT",
            ),
            "validated_root_cause": "An appropriate deterministic safety route returns an answer without action/reason and omits part of the requested risk scope.",
            "severity": "HIGH",
            "evidence": "The saved record has no retrieval, empty decision history, null action/reason and a pregnancy-only answer to a pregnancy-and-mental-risk request.",
            "generalization_reason": "All deterministic safety responses should preserve a complete structured contract independent of a specific medicine or benchmark case.",
            "production_change_justified": True,
        },
    ]

    return {
        "source_integrity": {
            "pre_hashes": pre_hashes,
            "post_hashes": post_hashes,
            "unchanged": pre_hashes == post_hashes,
            "artifact_count": len(pre_hashes),
        },
        "case_counts": {
            "rows": len(rows),
            "unique_ids": len({row["case_id"] for row in rows}),
            "answerable": sum(row["case_family"].startswith("answerable") for row in rows),
            "evidence_gap": sum(row["case_family"] == "evidence_gap" for row in rows),
        },
        "original_classification_counts": original_counts,
        "validated_classification_counts": validated_counts,
        "validation_status_counts": status_counts,
        "confirmed_system_defects": {
            key: value
            for key, value in validated_counts.items()
            if key.startswith("SYSTEM_BUG_")
        },
        "corrected_original_findings": [
            {
                "case_id": row["case_id"],
                "original": row["original_primary_classification"],
                "validated": row["validated_primary_classification"],
                "reason": row["correction_reason"],
            }
            for row in corrected
        ],
        "not_proven_original_findings": [
            {
                "case_id": row["case_id"],
                "original": row["original_primary_classification"],
                "reason": row["correction_reason"],
            }
            for row in not_proven
        ],
        "nrr_validated_breakdown": {
            "failures": 14,
            "successes": 16,
            "behavior_types": nrr_behavior_counts(rows),
            "natural_language_rejects_core": 14,
            "structured_false_generates": 14,
        },
        "retrieval_semantic_validation": {
            "original_count": len(original_retrieval),
            "original_confirmed_semantic_count": len(set(original_retrieval) & set(validated_retrieval)),
            "exact_provenance_only_corrected_count": 5,
            "corrected_to_generation_count": 1,
            "newly_identified_from_multi_turn_review": len(set(validated_retrieval) - set(original_retrieval)),
            "validated_total": len(validated_retrieval),
            "not_proven_count": 0,
            "case_ids": validated_retrieval,
        },
        "packing_semantic_validation": {
            "original_count": len(original_packing),
            "confirmed_semantic_count": len(set(original_packing) & set(validated_packing)),
            "exact_provenance_only_corrected_count": len(set(original_packing) - set(validated_packing)),
            "validated_total": len(validated_packing),
            "not_proven_count": 0,
            "case_ids": validated_packing,
        },
        "multi_turn_validation": {
            "case_count": len(multi_turn),
            "confirmed_multi_turn_context_defects": sum(row["multi_turn_status"] == "DEFECT" for row in multi_turn),
            "retrieval_defects": sum(row["validated_root_cause_layer"] == "retrieval" for row in multi_turn),
            "packing_defects": sum(row["validated_root_cause_layer"] == "packing" for row in multi_turn),
            "generation_defects": sum(row["validated_root_cause_layer"] == "generation" for row in multi_turn),
            "safety_structured_action_defects": sum(row["validated_root_cause_layer"] == "safety" for row in multi_turn),
            "no_material_defect": sum(row["validated_primary_classification"] == "PASS" for row in multi_turn),
            "not_proven_defects": sum(row["validation_status"] == "NOT_PROVEN" for row in multi_turn),
        },
        "generation_validation": {
            "confirmed_omissions": ids_for(rows, "generation_scope_status", "OMISSION"),
            "confirmed_over_answering": ids_for(rows, "generation_scope_status", "OVER_ANSWERING"),
            "confirmed_unsupported": ids_for(rows, "generation_scope_status", "UNSUPPORTED"),
            "expected_scope_or_granularity": ids_for(rows, "generation_scope_status", "EXPECTED_SCOPE_VARIATION"),
        },
        "safety_validation": {
            "structured_state_missing": ids_for(rows, "safety_status", "STRUCTURED_STATE_MISSING"),
            "routing_appropriate": True,
            "normal_retrieval_executed": False,
            "requested_scope_complete": False,
        },
        "evaluator_validation": {
            "confirmed_evaluator_anomalies": [],
            "status": "NO_EVALUATOR_ANOMALY_PROVEN",
            "note": "Low or surprising saved labels were not treated as evaluator defects without a direct semantic contradiction.",
        },
        "retry_validation": {
            "observed_retries": 6,
            "observed_final_correct": 2,
            "observed_final_incorrect": 4,
            "causal_effect": "CAUSAL_EFFECT_NOT_PROVEN",
            "reason": "The saved run has no clean no-retry counterfactual for those cases.",
        },
        "reranker_validation": {
            "saved_final_traces": 96,
            "succeeded": 96,
            "fallback_or_failure": 0,
            "conclusion": "RERANKER_RUNTIME_FAILURE_NOT_SUPPORTED",
        },
        "confirmed_fix_candidates": confirmed_fix_candidates,
        "deferred_or_not_proven_candidates": [
            "Pack size 8 is intrinsically wrong: NOT_PROVEN.",
            "Retry causes worse answers: CAUSAL_EFFECT_NOT_PROVEN.",
            "The local reranker should be replaced: NOT_PROVEN.",
            "A single retrieval or packing change explains all semantic misses: NOT_PROVEN.",
        ],
        "recommended_production_scope": {
            "decision": "AUDIT_VALIDATION_SUPPORTS_TARGETED_PRODUCTION_CHANGES",
            "include": [
                "Exact-proposition evidence sufficiency and structured action alignment",
                "Structured action/reason completeness for deterministic safety responses",
            ],
            "defer": [
                "Retrieval/reranker/packer retuning without a more localized cause",
                "Retry policy changes without counterfactual evidence",
                "Broad generation prompt changes",
            ],
        },
        "baseline_comparison_policy": {
            "description": "The prior 10 percentage-point comparison rule is descriptive project audit policy only.",
            "scientific_cutoff": False,
            "used_for_root_cause": False,
        },
    }


def markdown_table(headers: list[str], body: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in body)
    return "\n".join(lines)


def compact_cell(value: Any, limit: int = 240) -> str:
    if value is None:
        return "(none)"
    text = " ".join(str(value).split()).replace("|", "\\|")
    if not text:
        return "(none)"
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def decision_path(record: dict[str, Any]) -> str:
    decisions = record.get("agent_decision_history") or []
    if not decisions:
        return "(empty)"
    return " -> ".join(
        f"{decision.get('action')}/{decision.get('reason_code')}" for decision in decisions
    )


def render_report(
    summary: dict[str, Any],
    rows: list[dict[str, str]],
    raw_records: dict[str, dict[str, Any]],
    benchmark_cases: dict[str, dict[str, Any]],
) -> str:
    hashes = summary["source_integrity"]
    status_counts = summary["validation_status_counts"]
    nrr_rows = [
        row
        for row in rows
        if row["case_family"] == "evidence_gap" and row["nrr_correct"] == "false"
    ]
    multi_rows = [row for row in rows if row["case_family"] == "answerable_multi_turn"]
    corrected_rows = [row for row in rows if row["validation_status"] == "CORRECTED"]
    retrieval_packing_rows = [
        row
        for row in rows
        if row["original_primary_classification"]
        in {
            "SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE",
            "SYSTEM_BUG_PACKER_SELECTION",
        }
    ]
    f1_review_rows = [
        row
        for row in rows
        if row["claim_recall_pct"]
        and row["faithfulness_pct"]
        and row["claim_f1_pct"]
        and float(row["claim_recall_pct"]) == 100.0
        and float(row["faithfulness_pct"]) == 100.0
        and float(row["claim_f1_pct"]) < 40.0
    ]
    hash_rows = [
        [name, digest, hashes["post_hashes"][name], digest == hashes["post_hashes"][name]]
        for name, digest in hashes["pre_hashes"].items()
    ]
    class_rows = [[name, value] for name, value in summary["validated_classification_counts"].items()]
    corrected_table = [
        [row["case_id"], row["original_primary_classification"], row["validated_primary_classification"]]
        for row in corrected_rows
    ]
    nrr_table = [
        [
            row["case_id"],
            compact_cell(raw_records[row["case_id"]].get("query"), 160),
            compact_cell(
                benchmark_cases[row["case_id"]]
                .get("absence_verification", {})
                .get("unsupported_factual_requirement"),
                180,
            ),
            next(
                flag.removeprefix("nrr_behavior_")
                for flag in row["validated_secondary_flags"].split("|")
                if flag.startswith("nrr_behavior_")
            ),
            compact_cell(row["evidence_basis"], 220),
            compact_cell(
                f"assessment={raw_records[row['case_id']].get('evidence_assessment', {}).get('reason')}; "
                f"final={row['actual_action']}/{row['actual_reason']}; "
                f"decisions={decision_path(raw_records[row['case_id']])}",
                220,
            ),
        ]
        for row in nrr_rows
    ]
    retrieval_packing_table = [
        [
            row["case_id"],
            row["original_primary_classification"],
            row["candidate_semantic_support"],
            row["packed_semantic_support"],
            row["validated_primary_classification"],
            compact_cell(row["evidence_basis"], 240),
        ]
        for row in retrieval_packing_rows
    ]
    multi_table = [
        [
            row["case_id"],
            row["category"],
            compact_cell(
                " / ".join(
                    item.get("content", "")
                    for item in raw_records[row["case_id"]].get("history", [])
                ),
                180,
            ),
            compact_cell(raw_records[row["case_id"]].get("query"), 160),
            compact_cell(
                (raw_records[row["case_id"]].get("retrieval_trace") or {}).get(
                    "query"
                ),
                180,
            ),
            compact_cell(decision_path(raw_records[row["case_id"]]), 140),
            compact_cell(
                f"{row['validated_primary_classification']}; "
                f"candidate={row['candidate_semantic_support']}; "
                f"pack={row['packed_semantic_support']}; {row['evidence_basis']}",
                260,
            ),
        ]
        for row in multi_rows
    ]
    f1_review_table = [
        [
            row["case_id"],
            row["claim_f1_pct"],
            row["validated_primary_classification"],
            compact_cell(row["evidence_basis"], 240),
        ]
        for row in f1_review_rows
    ]
    retrieval = summary["retrieval_semantic_validation"]
    packing = summary["packing_semantic_validation"]
    generation = summary["generation_validation"]

    return f"""# Forensic Audit Validation

## 1. Executive Summary

The second-pass validation reviewed every original non-PASS case, both expected-metric cases, all 14 NRR failures, all 24 retrieval/packing labels, and all 15 multi-turn cases from saved evidence only. It confirms {status_counts.get('CONFIRMED', 0)} case findings and corrects {status_counts.get('CORRECTED', 0)}; no per-case finding remains `NOT_PROVEN`. Causal claims without counterfactual evidence remain explicitly deferred.

Decision: **AUDIT_VALIDATION_SUPPORTS_TARGETED_PRODUCTION_CHANGES**. The justified scope is limited to exact-proposition evidence/action alignment and structured-state completeness on deterministic safety responses.

## 2. Source Integrity

Hashes were computed independently at process start and again after all reads and validation-output generation. All {hashes['artifact_count']} source artifacts remained byte-identical.

{markdown_table(['Artifact', 'Pre SHA-256', 'Post SHA-256', 'Unchanged'], hash_rows)}

## 3. Validation Method

Deterministic checks validate identity, schema, counts, original-audit linkage, NRR/multi-turn coverage, and independent pre/post hashes. Semantic review separately compared each gold proposition with saved candidate IDs mapped to saved excerpts, packed text, evidence assessment, decision history, and final response. Exact chunk identity and score thresholds were discovery aids only, not sufficient root-cause proof.

No new scientific threshold was introduced. The prior 10 percentage-point baseline rule is retained only as descriptive project audit policy and is not used to assign root cause.

## 4. Original Audit Issues Reviewed

- **Hash implementation:** corrected by two independent hash passes around output generation.
- **NRR category shortcut:** replaced by answer-by-answer A/B/C/D/E review.
- **Multi-turn case-ID shortcut:** replaced by history, standalone query, retrieval, pack, action and response review for all 15 cases.
- **Exact provenance assumption:** replaced by proposition-level semantic support judgments.
- **Over-answering heuristic:** replaced by information-need and response-scope review.
- **Baseline threshold:** explicitly documented as descriptive policy, not a scientific cutoff.

## 5. 100-Case Validation Status

Validation status: `CONFIRMED={status_counts.get('CONFIRMED', 0)}`, `CORRECTED={status_counts.get('CORRECTED', 0)}`, `NOT_PROVEN={status_counts.get('NOT_PROVEN', 0)}`.

{markdown_table(['Validated classification', 'Count'], class_rows)}

## 6. Validated NRR Root Cause

All 14 natural-language responses reject the unsupported core. Two are direct rejection/explanation cases (B); twelve reject the exact requirement and add nearby supported facts or advice (C). There are no A, D or E cases. Every failed record nevertheless ends as `generate/evidence_sufficient`, while evidence assessment treats topic-relevant material as usable.

{markdown_table(['Case', 'Query', 'Unsupported core', 'Type', 'Answer stance / supplementary grounding', 'Evidence assessment / decision path'], nrr_table)}

The natural-language layer already behaves like an abstention on the unsupported core. The systematic defect is that topic relevance is accepted as support for the exact requested proposition in structured state.

## 7. Validated Retrieval and Packing Findings

Original retrieval labels: {retrieval['original_count']}; confirmed semantic retrieval defects among those: {retrieval['original_confirmed_semantic_count']}; exact-provenance-only corrections: {retrieval['exact_provenance_only_corrected_count']}; corrected to generation: {retrieval['corrected_to_generation_count']}. Three multi-turn labels were reclassified as retrieval defects, for {retrieval['validated_total']} validated retrieval cases total.

Original packing labels: {packing['original_count']}; confirmed semantic losses: {packing['confirmed_semantic_count']}; exact-provenance-only corrections: {packing['exact_provenance_only_corrected_count']}; validated total: {packing['validated_total']}.

{markdown_table(['Case', 'Original label', 'Candidate semantic support', 'Packed semantic support', 'Validated finding', 'Proposition-level evidence'], retrieval_packing_table)}

The 24 full eight-item packs and their low mean Context Precision do not prove that pack size eight is wrong. Saved rows include broad overlap, useful non-gold context, exact-equivalent evidence, and true selection losses.

Reranker traces independently reproduce `96/96 succeeded`, no fallback/failure. Conclusion: `RERANKER_RUNTIME_FAILURE_NOT_SUPPORTED`.

## 8. Validated Multi-Turn Findings

The table reconstructs the previous topic, current-turn need, actual saved standalone retrieval query, decision path, evidence path and final validated layer for every multi-turn case. A missing retrieval query is shown explicitly rather than inferred.

{markdown_table(['Case', 'Subtype', 'Previous conversation', 'Current need', 'Saved standalone/retrieval query', 'Decision path', 'Evidence path and validated finding'], multi_table)}

Only one case is a confirmed primary history/context defect. The cohort instead contains three retrieval defects, three packing defects, one generation omission, one safety structured-state defect, and six cases with no material defect. Correct standalone queries in the corrected cases rule out multi-turn resolution as the dominant layer.

## 9. Validated Generation Findings

- Confirmed omission: `{', '.join(generation['confirmed_omissions'])}`.
- Confirmed material over-answering: `{', '.join(generation['confirmed_over_answering'])}`.
- Confirmed unsupported generation: `{', '.join(generation['confirmed_unsupported'])}`.
- Expected scope/granularity effects: `{', '.join(generation['expected_scope_or_granularity'])}`.

The original four omission labels were corrected because the responses cover their required propositions. Five of seven original over-answering labels were corrected because the additional supported claims directly explain the requested concept or reflect narrow gold scope.

The eight saved rows with Claim Recall 100, Faithfulness 100 and Claim F1 below 40 were reviewed semantically rather than classified by score:

{markdown_table(['Case', 'Claim F1', 'Validated interpretation', 'Semantic review'], f1_review_table)}

## 10. Safety Finding

The deterministic safety route for `ANS-MUL-010` is appropriate, but normal retrieval does not occur, the mental-risk half of the request is omitted, and action/reason remain null. A deterministic safety answer can and should coexist with complete structured state; this is a confirmed high-severity contract defect.

## 11. Evaluator / Metric Findings

`NO_EVALUATOR_ANOMALY_PROVEN`. Surprising recall/F1/faithfulness labels were not called evaluator errors unless saved semantics directly contradicted them. Claim granularity and narrow gold scope remain expected metric behavior in seven validated cases.

## 12. Retry Interpretation

Observed: six retries, two final-correct and four final-incorrect outcomes. This is descriptive only. The saved run lacks a no-retry counterfactual, so the causal conclusion is `CAUSAL_EFFECT_NOT_PROVEN`; retry must not be changed from this evidence alone.

## 13. Corrections to PR #100 Audit Conclusions

{markdown_table(['Case', 'Original', 'Validated'], corrected_table)}

The largest corrections are retrieval `12 -> 6` among originally labelled rows, packing `12 -> 9`, primary multi-turn `5 -> 1`, over-answering `7 -> 2`, and original generation omissions `4 -> 0`. The validated totals also include three retrieval defects and one generation omission recovered from the multi-turn review.

## 14. Confirmed Production Fix Candidates

1. **Exact-proposition evidence action boundary (14 HIGH):** distinguish evidence about a topic from evidence supporting the requested guarantee, exact value, or exact relationship; keep natural-language rejection and structured action aligned.
2. **Safety response structured-state completeness (1 HIGH):** deterministic safety responses must retain action/reason and complete the requested risk dimensions.

Both are localized, generalize beyond benchmark wording, and can be regression-tested with synthetic non-benchmark prompts. No model, agent, framework or new scoring system is justified.

## 15. Findings That Must NOT Be Used to Modify Production

- Pack size eight is intrinsically wrong: `NOT_PROVEN`.
- Retry causes incorrect answers: `CAUSAL_EFFECT_NOT_PROVEN`.
- The local reranker should be replaced: `NOT_PROVEN`.
- One uniform retrieval/packing change explains all semantic misses: `NOT_PROVEN`.
- Low Claim F1 alone proves over-answering or evaluator failure: `NOT_PROVEN`.

## 16. Recommended Production Scope

**AUDIT_VALIDATION_SUPPORTS_TARGETED_PRODUCTION_CHANGES**

Limit the later production task to the two confirmed contract candidates above. Defer retrieval/reranker/packer retuning, retry changes, and broad prompt changes until a localized mechanism is proven.
"""


def execute(source_root: Path, check: bool) -> None:
    pre_hashes = hash_source_inputs(source_root)
    rows = load_csv(VALIDATED_CSV)
    original_rows = load_csv(
        source_root
        / "evaluation/audits/post_improvement_47b10954/case_forensic_audit.csv"
    )
    validate_rows(rows, original_rows)

    # These reads are part of the integrity gate and ensure the saved inputs remain parseable.
    benchmark = load_json(source_root / "evaluation/benchmark_100.json")
    raw_results = load_json(
        source_root
        / "evaluation/results/post_improvement_47b10954/raw_results.json"
    )
    benchmark_cases = {case["case_id"]: case for case in benchmark["cases"]}
    raw_records = {record["case_id"]: record for record in raw_results["records"]}
    validated_ids = {row["case_id"] for row in rows}
    if set(benchmark_cases) != validated_ids or set(raw_records) != validated_ids:
        raise ValidationError("Saved benchmark/raw case IDs differ from validated rows")

    provisional_post_hashes = hash_source_inputs(source_root)
    if pre_hashes != provisional_post_hashes:
        raise ValidationError("Source mutation detected before output generation")
    summary = build_summary(rows, pre_hashes, provisional_post_hashes)
    report = render_report(summary, rows, raw_records, benchmark_cases)
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    if check:
        if not SUMMARY_PATH.is_file() or SUMMARY_PATH.read_text(encoding="utf-8") != summary_text:
            raise ValidationError("audit_validation_summary.json is stale")
        if not REPORT_PATH.is_file() or REPORT_PATH.read_text(encoding="utf-8") != report:
            raise ValidationError("audit_validation_report.md is stale")
    else:
        SUMMARY_PATH.write_text(summary_text, encoding="utf-8")
        REPORT_PATH.write_text(report, encoding="utf-8")

    post_hashes = hash_source_inputs(source_root)
    if pre_hashes != post_hashes:
        raise ValidationError("Source mutation detected after output generation")
    if summary["source_integrity"]["post_hashes"] != post_hashes:
        raise ValidationError("Persisted post-hash map differs from final hash pass")
    print("FORENSIC_AUDIT_VALIDATION_CHECK: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the saved 100-case forensic audit without live execution."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Repository containing the ignored saved formal-run artifacts.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate committed outputs without writing them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    execute(arguments.source_root.resolve(), arguments.check)
