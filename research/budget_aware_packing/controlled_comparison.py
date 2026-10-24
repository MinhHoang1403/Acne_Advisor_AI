"""Run the single preregistered offline paired packing comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from research.budget_aware_packing.baseline_replay import (
    ARTIFACT_COMMIT,
    ARTIFACT_REF,
    BENCHMARK_CANONICAL_SHA256,
    FORENSIC_SHA,
    FROZEN_PRODUCTION_SHA,
    FROZEN_RAW_BYTE_SHA256,
    FROZEN_RAW_CANONICAL_SHA256,
    FROZEN_RUN_ID,
    KB_BUILD_ID,
    PIPELINE_FINGERPRINT,
    SOURCE_ARTIFACTS,
    FrozenInputs,
    ReplayInputError,
    candidate_from_cache,
    canonical_json_bytes,
    canonical_json_sha256,
    load_candidate_cache,
    load_frozen_inputs,
    sha256_bytes,
    sha256_text,
    validate_frozen_identity,
)
from research.budget_aware_packing.budget_aware_selector import (
    SELECTOR_NAME,
    SelectionResult,
    select_budget_aware,
)
from src.retrieval.context_packer import _render_block
from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate


JSONDict = dict[str, Any]
DETERMINISTIC_OUTPUT_NAMES = (
    "development_comparison.json",
    "per_case_comparison.csv",
    "holdout_eligibility.json",
    "holdout_comparison.json",
    "regression_audit.json",
    "experiment_identity.json",
)
PRIOR_DEVELOPMENT_CASES = (
    "bp_antibiotic",
    "adapalene_class",
    "clindamycin_monotherapy",
    "adapalene_bp_comparison",
    "retinoid_pregnancy",
    "dermatologist_referral",
    "historical_02",
    "paraphrase_inflammatory_lesions",
)
PRIOR_REGRESSION_CASES = (
    "ANS-MUL-002",
    "ANS-MUL-006",
    "ANS-MUL-008",
    "ANS-MUL-012",
    "ANS-MUL-014",
)
PER_CASE_COLUMNS = (
    "case_id",
    "historically_exposed",
    "retrieval_attempts",
    "final_candidate_count",
    "required_evidence_count",
    "acquired_required_evidence",
    "baseline_hits",
    "treatment_hits",
    "outcome",
    "baseline_complete",
    "treatment_complete",
    "baseline_selected_count",
    "treatment_selected_count",
    "baseline_context_chars",
    "treatment_context_chars",
    "baseline_selected_ids",
    "treatment_selected_ids",
    "added_candidate_ids",
    "removed_candidate_ids",
    "gained_requirement_ids",
    "lost_requirement_ids",
)


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _load_json(path: Path) -> JSONDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplayInputError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_csv(path: Path, rows: list[JSONDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_CASE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _json_cell(row[key]) if isinstance(row[key], (list, dict)) else row[key]
                    for key in PER_CASE_COLUMNS
                }
            )


def _requirements(gold: JSONDict) -> tuple[list[JSONDict], dict[str, list[JSONDict]]]:
    requirements: list[JSONDict] = []
    seen: set[tuple[str, str]] = set()
    by_case: dict[str, list[JSONDict]] = defaultdict(list)
    for claim in gold.get("gold_claim_adjudications") or []:
        case_id = str(claim.get("case_id") or "")
        for evidence_id in claim.get("current_evidence_ids") or []:
            chunk_id = str(evidence_id)
            key = (case_id, chunk_id)
            requirement_id = f"{case_id}::{chunk_id}"
            if key in seen:
                existing = next(
                    item for item in by_case[case_id] if item["required_chunk_id"] == chunk_id
                )
                existing["claim_ids"].append(str(claim.get("claim_id") or ""))
                continue
            seen.add(key)
            row = {
                "requirement_id": requirement_id,
                "case_id": case_id,
                "required_chunk_id": chunk_id,
                "claim_ids": [str(claim.get("claim_id") or "")],
            }
            requirements.append(row)
            by_case[case_id].append(row)
    return requirements, dict(by_case)


def _historical_stage(record: JSONDict, candidate_id: str) -> str:
    if candidate_id not in set(record.get("acquisition_candidate_ids") or []):
        return "ACQUISITION_MISS"
    final_fused = {
        str(item.get("candidate_id")): item
        for item in ((record.get("retrieval_trace") or {}).get("candidate_trace") or {}).get(
            "fused"
        )
        or []
    }
    item = final_fused.get(candidate_id) or {}
    if item.get("packed"):
        return "PACKED"
    if item.get("drop_reason") == "character_limit":
        return "CHARACTER_BUDGET_LOSS"
    if item.get("not_examined_reason") == "item_limit_reached":
        return "ITEM_BUDGET_LOSS"
    return "UNKNOWN"


def _attempt_candidates(attempt: JSONDict, cache: JSONDict) -> list[RetrievedCandidate]:
    traces = (attempt.get("candidate_trace") or {}).get("fused") or []
    return [candidate_from_cache(trace, cache) for trace in traces]


def _normalized_query(record: JSONDict, attempt: JSONDict) -> NormalizedQuery:
    query = str(attempt.get("overall_rerank_query") or attempt.get("rerank_query") or "")
    return NormalizedQuery(
        original_query=str(attempt.get("overall_rerank_query") or record.get("query") or ""),
        normalized_text=query,
    )


def run_treatment_without_gold(raw: JSONDict, cache: JSONDict) -> list[JSONDict]:
    """Selection phase. Gold/evaluation artifacts are intentionally not arguments."""

    outputs: list[JSONDict] = []
    for record in raw.get("records") or []:
        attempts = record.get("retrieval_attempt_traces") or []
        for attempt_offset, attempt in enumerate(attempts):
            candidates = _attempt_candidates(attempt, cache)
            result = select_budget_aware(
                _normalized_query(record, attempt),
                candidates,
                max_items=9,
                max_chars=7000,
            )
            outputs.append(
                {
                    "case_id": str(record["case_id"]),
                    "attempt_index": int(attempt.get("attempt_index") or attempt_offset + 1),
                    "is_final_attempt": attempt_offset == len(attempts) - 1,
                    "candidate_ids": [candidate.candidate_id for candidate in candidates],
                    "candidate_text_sha256": {
                        candidate.candidate_id: sha256_text(candidate.text)
                        for candidate in candidates
                    },
                    "candidate_source_ids": {
                        candidate.candidate_id: str(candidate.payload.get("source_id") or "")
                        for candidate in candidates
                    },
                    "candidate_rendered_costs": {
                        candidate.candidate_id: len(_render_block(candidate, 1))
                        for candidate in candidates
                    },
                    "baseline_selected_ids": list(
                        (attempt.get("packer") or {}).get("selected_ids") or []
                    ),
                    "baseline_context_chars": (attempt.get("packer") or {}).get("context_chars"),
                    "treatment_selected_ids": list(result.selected_candidate_ids),
                    "treatment_selected_positions": list(result.selected_original_positions),
                    "treatment_context_chars": len(result.packed_context.context_text),
                    "treatment_total_utility": result.total_utility,
                    "transitions_considered": result.transitions_considered,
                    "peak_state_count": result.peak_state_count,
                    "provenance_violations": _provenance_violations(candidates, result),
                }
            )
    return outputs


def _provenance_violations(
    candidates: list[RetrievedCandidate], result: SelectionResult
) -> list[JSONDict]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    violations: list[JSONDict] = []
    if len(result.selected_candidate_ids) != len(set(result.selected_candidate_ids)):
        violations.append({"kind": "duplicate_candidate_id"})
    for item in result.packed_context.items:
        source = by_id.get(item.item_id)
        if source is None:
            violations.append({"kind": "candidate_outside_pool", "candidate_id": item.item_id})
            continue
        if item.text != source.text.strip():
            violations.append({"kind": "text_changed", "candidate_id": item.item_id})
        if str(item.payload.get("chunk_id") or item.item_id) != str(
            source.payload.get("chunk_id") or source.candidate_id
        ):
            violations.append({"kind": "chunk_id_changed", "candidate_id": item.item_id})
        if str(item.payload.get("source_id") or "") != str(source.payload.get("source_id") or ""):
            violations.append({"kind": "source_id_changed", "candidate_id": item.item_id})
    if len(result.packed_context.items) > 9:
        violations.append({"kind": "item_budget_exceeded"})
    if len(result.packed_context.context_text) > 7000:
        violations.append({"kind": "character_budget_exceeded"})
    return violations


def _position_distribution(outputs: list[JSONDict], arm: str) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for output in outputs:
        positions = {
            candidate_id: index for index, candidate_id in enumerate(output["candidate_ids"], 1)
        }
        for candidate_id in output[f"{arm}_selected_ids"]:
            counts[positions[candidate_id]] += 1
    return {str(key): counts[key] for key in sorted(counts)}


def _percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def evaluate_development(
    inputs: FrozenInputs,
    outputs: list[JSONDict],
) -> tuple[JSONDict, list[JSONDict], JSONDict]:
    records = {str(item["case_id"]): item for item in inputs.raw.get("records") or []}
    final_outputs = {output["case_id"]: output for output in outputs if output["is_final_attempt"]}
    requirements, requirements_by_case = _requirements(inputs.gold)
    requirement_rows: list[JSONDict] = []
    for requirement in requirements:
        case_id = requirement["case_id"]
        candidate_id = requirement["required_chunk_id"]
        output = final_outputs[case_id]
        baseline_hit = candidate_id in set(output["baseline_selected_ids"])
        treatment_hit = candidate_id in set(output["treatment_selected_ids"])
        requirement_rows.append(
            {
                **requirement,
                "historical_stage": _historical_stage(records[case_id], candidate_id),
                "baseline_hit": baseline_hit,
                "treatment_hit": treatment_hit,
            }
        )

    baseline_hits = sum(row["baseline_hit"] for row in requirement_rows)
    treatment_hits = sum(row["treatment_hit"] for row in requirement_rows)
    gains = [row for row in requirement_rows if not row["baseline_hit"] and row["treatment_hit"]]
    losses = [row for row in requirement_rows if row["baseline_hit"] and not row["treatment_hit"]]
    unchanged = len(requirement_rows) - len(gains) - len(losses)
    char_gains = [row for row in gains if row["historical_stage"] == "CHARACTER_BUDGET_LOSS"]
    item_gains = [row for row in gains if row["historical_stage"] == "ITEM_BUDGET_LOSS"]

    case_rows: list[JSONDict] = []
    for record in inputs.raw.get("records") or []:
        case_id = str(record["case_id"])
        output = final_outputs[case_id]
        case_requirements = requirements_by_case.get(case_id, [])
        required_ids = {item["required_chunk_id"] for item in case_requirements}
        acquisition = set(record.get("acquisition_candidate_ids") or [])
        baseline_ids = list(output["baseline_selected_ids"])
        treatment_ids = list(output["treatment_selected_ids"])
        baseline_set = set(baseline_ids)
        treatment_set = set(treatment_ids)
        baseline_case_hits = len(required_ids & baseline_set)
        treatment_case_hits = len(required_ids & treatment_set)
        case_gains = [
            item["requirement_id"]
            for item in case_requirements
            if item["required_chunk_id"] not in baseline_set
            and item["required_chunk_id"] in treatment_set
        ]
        case_losses = [
            item["requirement_id"]
            for item in case_requirements
            if item["required_chunk_id"] in baseline_set
            and item["required_chunk_id"] not in treatment_set
        ]
        outcome = (
            "GAIN" if case_gains and not case_losses else "REGRESSION" if case_losses else "TIE"
        )
        if case_gains and case_losses:
            outcome = "MIXED"
        case_rows.append(
            {
                "case_id": case_id,
                "historically_exposed": True,
                "retrieval_attempts": len(record.get("retrieval_attempt_traces") or []),
                "final_candidate_count": len(output["candidate_ids"]),
                "required_evidence_count": len(required_ids),
                "acquired_required_evidence": len(required_ids & acquisition),
                "baseline_hits": baseline_case_hits,
                "treatment_hits": treatment_case_hits,
                "outcome": outcome,
                "baseline_complete": bool(required_ids) and baseline_case_hits == len(required_ids),
                "treatment_complete": bool(required_ids)
                and treatment_case_hits == len(required_ids),
                "baseline_selected_count": len(baseline_ids),
                "treatment_selected_count": len(treatment_ids),
                "baseline_context_chars": output["baseline_context_chars"],
                "treatment_context_chars": output["treatment_context_chars"],
                "baseline_selected_ids": baseline_ids,
                "treatment_selected_ids": treatment_ids,
                "added_candidate_ids": [item for item in treatment_ids if item not in baseline_set],
                "removed_candidate_ids": [
                    item for item in baseline_ids if item not in treatment_set
                ],
                "gained_requirement_ids": case_gains,
                "lost_requirement_ids": case_losses,
            }
        )

    answerable_rows = [row for row in case_rows if row["required_evidence_count"]]
    baseline_complete = sum(row["baseline_complete"] for row in answerable_rows)
    treatment_complete = sum(row["treatment_complete"] for row in answerable_rows)
    all_violations = [
        {"case_id": output["case_id"], "attempt_index": output["attempt_index"], **violation}
        for output in outputs
        for violation in output["provenance_violations"]
    ]
    baseline_chunks = sum(len(output["baseline_selected_ids"]) for output in outputs)
    treatment_chunks = sum(len(output["treatment_selected_ids"]) for output in outputs)
    baseline_chars = sum(int(output["baseline_context_chars"]) for output in outputs)
    treatment_chars = sum(int(output["treatment_context_chars"]) for output in outputs)
    transition_values = [int(output["transitions_considered"]) for output in outputs]
    peak_state_values = [int(output["peak_state_count"]) for output in outputs]

    comparison = {
        "schema": "budget_aware_development_comparison_v1",
        "status": "EXPERIMENT_DEVELOPMENT_COMPLETE",
        "split": "DEVELOPMENT_RESULT",
        "historically_exposed": True,
        "selector": SELECTOR_NAME,
        "gold_fields_passed_to_selector": False,
        "cases": len(case_rows),
        "answerable_cases": len(answerable_rows),
        "retrieval_attempts": len(outputs),
        "primary_metric": {
            "name": "Packed Required-Evidence Coverage",
            "baseline_hits": baseline_hits,
            "treatment_hits": treatment_hits,
            "total_requirements": len(requirement_rows),
            "baseline_rate": round(baseline_hits / len(requirement_rows), 12),
            "treatment_rate": round(treatment_hits / len(requirement_rows), 12),
            "absolute_delta_hits": treatment_hits - baseline_hits,
            "absolute_delta_percentage_points": round(
                100 * (treatment_hits - baseline_hits) / len(requirement_rows), 6
            ),
            "gained_requirements": len(gains),
            "lost_requirements": len(losses),
            "unchanged_requirements": unchanged,
        },
        "complete_case_coverage": {
            "baseline": baseline_complete,
            "treatment": treatment_complete,
            "denominator": len(answerable_rows),
            "delta": treatment_complete - baseline_complete,
        },
        "historical_loss_recovery": {
            "character_budget_reference": 20,
            "character_budget_recovered": len(char_gains),
            "character_budget_remaining": 20 - len(char_gains),
            "item_budget_reference": 7,
            "item_budget_recovered": len(item_gains),
            "item_budget_remaining": 7 - len(item_gains),
            "new_losses_of_previously_packed_requirements": len(losses),
            "note": (
                "Treatment omissions are not reclassified as greedy character/item drop reasons; "
                "they are paired selection regressions."
            ),
        },
        "resource_use_all_attempts": {
            "baseline_selected_chunks": baseline_chunks,
            "treatment_selected_chunks": treatment_chunks,
            "delta_selected_chunks": treatment_chunks - baseline_chunks,
            "baseline_context_chars": baseline_chars,
            "treatment_context_chars": treatment_chars,
            "delta_context_chars": treatment_chars - baseline_chars,
            "baseline_character_utilization": round(baseline_chars / (len(outputs) * 7000), 12),
            "treatment_character_utilization": round(treatment_chars / (len(outputs) * 7000), 12),
            "baseline_selected_rank_distribution": _position_distribution(outputs, "baseline"),
            "treatment_selected_rank_distribution": _position_distribution(outputs, "treatment"),
            "candidate_churn_added": sum(len(row["added_candidate_ids"]) for row in case_rows),
            "candidate_churn_removed": sum(len(row["removed_candidate_ids"]) for row in case_rows),
        },
        "deterministic_compute_proxy": {
            "transitions_p50": _percentile(transition_values, 0.50),
            "transitions_p95": _percentile(transition_values, 0.95),
            "transitions_max": max(transition_values),
            "peak_states_p50": _percentile(peak_state_values, 0.50),
            "peak_states_p95": _percentile(peak_state_values, 0.95),
            "peak_states_max": max(peak_state_values),
        },
        "gained_requirement_ids": [row["requirement_id"] for row in gains],
        "lost_requirement_ids": [row["requirement_id"] for row in losses],
        "character_budget_recovered_requirement_ids": [row["requirement_id"] for row in char_gains],
        "item_budget_recovered_requirement_ids": [row["requirement_id"] for row in item_gains],
        "hard_invariants": {
            "provenance_violations": len(all_violations),
            "whole_chunk_violations": sum(
                violation["kind"] == "text_changed" for violation in all_violations
            ),
            "budget_violations": sum(
                violation["kind"] in {"item_budget_exceeded", "character_budget_exceeded"}
                for violation in all_violations
            ),
            "duplicate_id_violations": sum(
                violation["kind"] == "duplicate_candidate_id" for violation in all_violations
            ),
            "candidate_pool_violations": sum(
                violation["kind"] == "candidate_outside_pool" for violation in all_violations
            ),
        },
        "interpretation_limit": (
            "Identity-based packed recall does not measure answer quality, context coherence, "
            "unlabeled supporting evidence, or clinical safety."
        ),
    }
    regression_audit = _build_regression_audit(
        records=records,
        final_outputs=final_outputs,
        case_rows=case_rows,
        gains=gains,
        losses=losses,
        violations=all_violations,
    )
    return comparison, case_rows, regression_audit


def _build_regression_audit(
    *,
    records: dict[str, JSONDict],
    final_outputs: dict[str, JSONDict],
    case_rows: list[JSONDict],
    gains: list[JSONDict],
    losses: list[JSONDict],
    violations: list[JSONDict],
) -> JSONDict:
    row_by_case = {row["case_id"]: row for row in case_rows}
    changed_cases = sorted({item["case_id"] for item in [*gains, *losses]})
    audited: list[JSONDict] = []
    for case_id in changed_cases:
        output = final_outputs[case_id]
        case = row_by_case[case_id]
        positions = {
            candidate_id: index for index, candidate_id in enumerate(output["candidate_ids"], 1)
        }
        case_gains = [item for item in gains if item["case_id"] == case_id]
        case_losses = [item for item in losses if item["case_id"] == case_id]
        added = list(case["added_candidate_ids"])
        removed = list(case["removed_candidate_ids"])
        audited.append(
            {
                "case_id": case_id,
                "frozen_candidate_pool_sha256": sha256_text(
                    "\n".join(output["candidate_ids"]) + "\n"
                ),
                "baseline_selected_ids": list(output["baseline_selected_ids"]),
                "treatment_selected_ids": list(output["treatment_selected_ids"]),
                "added_candidate_ids": added,
                "removed_candidate_ids": removed,
                "gained_requirement_ids": [item["requirement_id"] for item in case_gains],
                "lost_requirement_ids": [item["requirement_id"] for item in case_losses],
                "gained_evidence": [
                    {
                        "requirement_id": item["requirement_id"],
                        "chunk_id": item["required_chunk_id"],
                        "original_rank": positions[item["required_chunk_id"]],
                        "source_id": output["candidate_source_ids"].get(item["required_chunk_id"]),
                        "historical_stage": item["historical_stage"],
                    }
                    for item in case_gains
                ],
                "lost_evidence": [
                    {
                        "requirement_id": item["requirement_id"],
                        "chunk_id": item["required_chunk_id"],
                        "original_rank": positions[item["required_chunk_id"]],
                        "source_id": output["candidate_source_ids"].get(item["required_chunk_id"]),
                    }
                    for item in case_losses
                ],
                "baseline_context_chars": output["baseline_context_chars"],
                "treatment_context_chars": output["treatment_context_chars"],
                "potential_qualifier_displacement": bool(removed),
                "qualifier_assessment": (
                    "UNASSESSED_UNLABELED_OR_QUALIFIER_RISK"
                    if removed
                    else "NO_CANDIDATE_DISPLACEMENT"
                ),
                "known_limitations": [
                    "Gold labels do not enumerate every useful supporting chunk.",
                    "No generation or semantic answer-quality evaluation was run.",
                ],
                "frozen_retrieval_attempts": len(
                    records[case_id].get("retrieval_attempt_traces") or []
                ),
            }
        )
    return {
        "schema": "budget_aware_regression_audit_v1",
        "status": "COMPLETE",
        "gain_count": len(gains),
        "regression_count": len(losses),
        "changed_case_count": len(changed_cases),
        "hard_invariant_violation_count": len(violations),
        "hard_invariant_violations": violations,
        "changed_cases": audited,
    }


def holdout_artifacts() -> tuple[JSONDict, JSONDict]:
    eligibility = {
        "schema": "budget_aware_holdout_eligibility_v1",
        "status": "INDEPENDENT_HOLDOUT_UNAVAILABLE",
        "eligible": False,
        "repository_refs_audited": True,
        "historically_exposed_frozen_cases": 100,
        "prior_selector_development_case_ids": list(PRIOR_DEVELOPMENT_CASES),
        "prior_selector_regression_case_ids": list(PRIOR_REGRESSION_CASES),
        "findings": [
            "The frozen 100-case benchmark was fully exposed by forensic analysis.",
            "No separate reviewer-approved evidence-requirement holdout with frozen replayable candidates exists.",
            "Files named holdout on local-ragchecker-extraction-contract concern claim-extraction contract research, not this packing experiment.",
            "A random split of the exposed 100 cases is not an independent holdout.",
        ],
        "holdout_case_ids": [],
        "holdout_case_ids_sha256": None,
    }
    comparison = {
        "schema": "budget_aware_holdout_comparison_v1",
        "status": "NOT_RUN",
        "reason": "INDEPENDENT_HOLDOUT_UNAVAILABLE",
        "cases": 0,
        "scores": None,
        "selector_executions": 0,
    }
    return eligibility, comparison


def experiment_identity(
    *,
    repository: Path,
    inputs: FrozenInputs,
    cache: JSONDict,
    baseline: JSONDict,
    protocol_path: Path,
    selector_path: Path,
) -> JSONDict:
    case_ids = [str(record["case_id"]) for record in inputs.raw.get("records") or []]
    protocol_bytes = protocol_path.read_bytes()
    selector_bytes = selector_path.read_bytes()
    cache_canonical = canonical_json_bytes(cache)
    return {
        "schema": "budget_aware_packing_experiment_identity_v1",
        "repository": "MinhHoang1403/Acne_Advisor_AI",
        "research_branch": _git(repository, "branch", "--show-current"),
        "research_base_sha": _git(repository, "rev-parse", "HEAD"),
        "forensic_sha": FORENSIC_SHA,
        "frozen_production_sha": FROZEN_PRODUCTION_SHA,
        "frozen_run_id": FROZEN_RUN_ID,
        "pipeline_fingerprint": PIPELINE_FINGERPRINT,
        "kb_build_id": KB_BUILD_ID,
        "source_artifact_ref": ARTIFACT_REF,
        "source_artifact_commit": ARTIFACT_COMMIT,
        "source_artifacts": SOURCE_ARTIFACTS,
        "raw_results": {
            "canonical_sha256": canonical_json_sha256(inputs.raw),
            "raw_byte_sha256": sha256_bytes(inputs.raw_bytes),
            "expected_canonical_sha256": FROZEN_RAW_CANONICAL_SHA256,
            "expected_raw_byte_sha256": FROZEN_RAW_BYTE_SHA256,
        },
        "benchmark_canonical_sha256": canonical_json_sha256(inputs.benchmark),
        "expected_benchmark_canonical_sha256": BENCHMARK_CANONICAL_SHA256,
        "baseline_implementation_identity": baseline["baseline_implementation_identity"],
        "experiment_protocol": {
            "path": protocol_path.relative_to(repository).as_posix(),
            "sha256": sha256_bytes(protocol_bytes),
            "changed_after_development": False,
        },
        "candidate_data_source": {
            **dict(cache.get("source") or {}),
            "cache_canonical_sha256": sha256_bytes(cache_canonical),
        },
        "reconstruction_status": baseline["status"],
        "development": {
            "historically_exposed": True,
            "case_count": len(case_ids),
            "case_ids": case_ids,
            "case_ids_sha256": sha256_text("\n".join(case_ids) + "\n"),
        },
        "holdout": {
            "status": "INDEPENDENT_HOLDOUT_UNAVAILABLE",
            "case_ids": [],
            "case_ids_sha256": None,
        },
        "selector": {
            "name": SELECTOR_NAME,
            "implementation_path": selector_path.relative_to(repository).as_posix(),
            "implementation_sha256": sha256_bytes(selector_bytes),
            "gold_fields_in_input_contract": False,
        },
        "test_environment": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "algorithm_network_calls": 0,
            "provider_calls": 0,
            "database_writes": 0,
        },
    }


def measure_runtime(raw: JSONDict, cache: JSONDict) -> JSONDict:
    elapsed_ms: list[float] = []
    tracemalloc.start()
    for record in raw.get("records") or []:
        for attempt in record.get("retrieval_attempt_traces") or []:
            candidates = _attempt_candidates(attempt, cache)
            started = time.perf_counter_ns()
            select_budget_aware(
                _normalized_query(record, attempt),
                candidates,
                max_items=9,
                max_chars=7000,
            )
            elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ordered = sorted(elapsed_ms)
    return {
        "schema": "budget_aware_selector_runtime_measurement_v1",
        "deterministic_output": False,
        "reason_excluded_from_reproducibility_hashes": "Wall-clock timing is nondeterministic.",
        "selector": SELECTOR_NAME,
        "attempts": len(ordered),
        "p50_ms": round(statistics.median(ordered), 6),
        "p95_ms": round(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)], 6),
        "max_ms": round(max(ordered), 6),
        "mean_ms": round(statistics.fmean(ordered), 6),
        "python_tracemalloc_peak_bytes": peak_bytes,
        "provider_calls": 0,
        "retrieval_calls": 0,
        "database_writes": 0,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_reproducibility_manifest(output_dir: Path) -> JSONDict:
    tracked_names = (
        "reconstruction_manifest.json",
        "baseline_replay_results.json",
        "baseline_replay_mismatches.json",
        *DETERMINISTIC_OUTPUT_NAMES,
    )
    files = {
        name: {
            "sha256": _file_sha256(output_dir / name),
            "bytes": (output_dir / name).stat().st_size,
        }
        for name in tracked_names
    }
    manifest = {
        "schema": "budget_aware_deterministic_output_manifest_v1",
        "status": "HASHES_RECORDED",
        "deterministic_files": files,
        "excluded_nondeterministic_files": ["selector_runtime_measurements.json"],
    }
    _write_json(output_dir / "deterministic_output_manifest.json", manifest)
    return manifest


def run(
    *,
    repository: Path,
    cache_path: Path,
    output_dir: Path,
    measure: bool,
) -> JSONDict:
    baseline = _load_json(output_dir / "baseline_replay_results.json")
    if baseline.get("status") != "EXACT_BASELINE_REPLAY_CONFIRMED" or not baseline.get(
        "gate_passed"
    ):
        raise ReplayInputError("BASELINE_REPLAY_BLOCKED: treatment is not authorized")
    inputs = load_frozen_inputs(repository)
    validate_frozen_identity(inputs)
    cache = load_candidate_cache(cache_path)
    outputs = run_treatment_without_gold(inputs.raw, cache)
    if len(outputs) != 104:
        raise ReplayInputError(f"Expected 104 treatment attempts, received {len(outputs)}")
    comparison, case_rows, regression = evaluate_development(inputs, outputs)
    eligibility, holdout = holdout_artifacts()
    identity = experiment_identity(
        repository=repository,
        inputs=inputs,
        cache=cache,
        baseline=baseline,
        protocol_path=repository / "research/BUDGET_AWARE_PACKING_PROTOCOL.md",
        selector_path=repository / "research/budget_aware_packing/budget_aware_selector.py",
    )
    _write_json(output_dir / "development_comparison.json", comparison)
    _write_csv(output_dir / "per_case_comparison.csv", case_rows)
    _write_json(output_dir / "holdout_eligibility.json", eligibility)
    _write_json(output_dir / "holdout_comparison.json", holdout)
    _write_json(output_dir / "regression_audit.json", regression)
    _write_json(output_dir / "experiment_identity.json", identity)
    if measure:
        _write_json(
            output_dir / "selector_runtime_measurements.json", measure_runtime(inputs.raw, cache)
        )
    manifest = write_reproducibility_manifest(output_dir)
    return {"comparison": comparison, "holdout": holdout, "manifest": manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/cache/research_budget_aware_packing/candidates.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/results/budget_aware_packing"),
    )
    parser.add_argument("--measure-runtime", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repository = args.repository.resolve()
    try:
        result = run(
            repository=repository,
            cache_path=(repository / args.cache_path).resolve(),
            output_dir=(repository / args.output_dir).resolve(),
            measure=args.measure_runtime,
        )
    except (ReplayInputError, OSError, subprocess.CalledProcessError) as exc:
        print(f"EXPERIMENT_INCONCLUSIVE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["comparison"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
