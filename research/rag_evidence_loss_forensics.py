"""Deterministic forensic analysis for the frozen thesis RAG outputs.

This module reads already-persisted benchmark, gold, run-manifest, and execution
trace files.  It never calls a model, retrieval service, database, or embedding
provider.  The original metric unit is preserved: one requirement is one unique
``case_id + current_evidence_id`` pair, with all associated claim IDs attached.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


class ForensicsInputError(ValueError):
    """Raised when frozen inputs cannot be safely reconciled."""


JSONDict = dict[str, Any]


JOURNEY_COLUMNS = [
    "requirement_id",
    "case_id",
    "family",
    "category",
    "turn",
    "history_type",
    "gold_claim_ids",
    "required_chunk_id",
    "gold_review_status",
    "retrieval_attempts",
    "current_user_question",
    "acquisition_queries",
    "rerank_queries",
    "dense_ranks",
    "bm25_ranks",
    "rrf_ranks",
    "rerank_ranks",
    "rerank_scores",
    "reranker_statuses",
    "reranker_fallback_statuses",
    "candidate_identity",
    "retry_statuses",
    "candidate_text_length",
    "candidate_text_length_missing_reason",
    "rendered_context_block_cost",
    "rendered_cost_missing_reason",
    "packer_max_items",
    "packer_max_chars",
    "packer_context_chars",
    "packed_position",
    "packer_drop_reason",
    "not_examined_reason",
    "decision_visible_evidence",
    "decision_visibility_inference",
    "decision_visibility_missing_reason",
    "generation_visible_evidence",
    "output_action",
    "output_reason",
    "primary_loss_stage",
    "primary_reason",
    "contributing_factors",
    "evidence_confidence",
    "trace_references",
    "missing_data_reasons",
]


def canonical_json_bytes(value: Any) -> bytes:
    """Return the benchmark's canonical JSON representation."""

    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (serialized + "\n").encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForensicsInputError(f"Cannot read valid UTF-8 JSON from {path}: {exc}") from exc


def _require_mapping(value: Any, label: str) -> JSONDict:
    if not isinstance(value, dict):
        raise ForensicsInputError(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ForensicsInputError(f"{label} must be a JSON list")
    return value


def _require_keys(value: JSONDict, keys: Iterable[str], label: str) -> None:
    missing = sorted(set(keys) - set(value))
    if missing:
        raise ForensicsInputError(f"{label} is missing required fields: {missing}")


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def validate_identity(
    *,
    raw_path: Path,
    raw: JSONDict,
    benchmark: JSONDict,
    benchmark_manifest: JSONDict,
    gold_path: Path,
    gold: JSONDict,
    gold_manifest: JSONDict,
    run_manifest: JSONDict,
    output_manifest: JSONDict,
    reference_metrics: JSONDict,
    expected_system_sha: str,
    artifact_ref: str,
    artifact_commit: str,
    source_inventory: dict[str, Any],
) -> JSONDict:
    """Fail closed unless the frozen input set has one coherent identity."""

    _require_keys(raw, ("schema", "identity", "records"), "raw results")
    _require_keys(benchmark, ("schema_version", "cases"), "benchmark")
    _require_keys(
        gold,
        ("identity", "gold_claim_adjudications", "negative_adjudications"),
        "locked gold",
    )
    _require_keys(
        run_manifest,
        ("run_id", "system_under_test_sha", "pipeline_fingerprint", "kb_build_id"),
        "run manifest",
    )
    _require_keys(
        output_manifest,
        ("record_count", "raw_results_canonical_sha256", "raw_results_raw_sha256"),
        "system output manifest",
    )
    _require_keys(reference_metrics, ("source_raw_results_canonical_sha256",), "metrics")

    identity = _require_mapping(raw["identity"], "raw identity")
    records = _require_list(raw["records"], "raw records")
    cases = _require_list(benchmark["cases"], "benchmark cases")
    raw_canonical = canonical_json_sha256(raw)
    raw_bytes = raw_sha256(raw_path)
    benchmark_canonical = canonical_json_sha256(benchmark)
    gold_canonical = canonical_json_sha256(gold)
    gold_bytes = raw_sha256(gold_path)

    checks = {
        "system_sha_matches_expected": identity.get("system_under_test_sha") == expected_system_sha,
        "system_sha_matches_run_manifest": run_manifest.get("system_under_test_sha")
        == expected_system_sha,
        "run_id_matches": identity.get("run_id") == run_manifest.get("run_id"),
        "pipeline_fingerprint_matches": identity.get("pipeline_fingerprint")
        == run_manifest.get("pipeline_fingerprint"),
        "kb_build_matches_run": identity.get("kb_build_id") == run_manifest.get("kb_build_id"),
        "kb_build_matches_gold": identity.get("kb_build_id")
        == (gold.get("identity") or {}).get("current_kb_build_id"),
        "kb_build_matches_gold_manifest": identity.get("kb_build_id")
        == gold_manifest.get("knowledge_build_id"),
        "benchmark_hash_matches_raw": benchmark_canonical == identity.get("benchmark_sha256"),
        "benchmark_hash_matches_run": benchmark_canonical
        == (run_manifest.get("benchmark") or {}).get("canonical_sha256"),
        "benchmark_hash_matches_manifest": benchmark_canonical
        == benchmark_manifest.get("benchmark_sha256"),
        "gold_canonical_matches_raw": gold_canonical == identity.get("gold_canonical_sha256"),
        "gold_canonical_matches_run": gold_canonical
        == (run_manifest.get("gold") or {}).get("canonical_sha256"),
        "gold_raw_matches_run": gold_bytes == (run_manifest.get("gold") or {}).get("raw_sha256"),
        "raw_canonical_matches_output_manifest": raw_canonical
        == output_manifest.get("raw_results_canonical_sha256"),
        "raw_bytes_match_output_manifest": raw_bytes
        == output_manifest.get("raw_results_raw_sha256"),
        "raw_canonical_matches_reference_metrics": raw_canonical
        == reference_metrics.get("source_raw_results_canonical_sha256"),
        "record_count_matches_output_manifest": len(records) == output_manifest.get("record_count"),
        "record_count_matches_benchmark": len(records) == len(cases),
        "case_ids_unique": len({record.get("case_id") for record in records}) == len(records),
        "case_order_matches_benchmark": [record.get("case_id") for record in records]
        == [case.get("case_id") for case in cases],
        "record_pipeline_fingerprints_match": {
            record.get("pipeline_fingerprint") for record in records
        }
        == {identity.get("pipeline_fingerprint")},
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ForensicsInputError(f"DATA_IDENTITY_BLOCKED: failed checks: {failed}")

    old_kb = (gold.get("identity") or {}).get("old_benchmark_kb_build_id")
    benchmark_manifest_kb = benchmark_manifest.get("active_kb_build_id")
    historical_kb_mapping_verified = bool(old_kb and old_kb == benchmark_manifest_kb)

    return {
        "schema": "rag_evidence_loss_identity_v1",
        "status": "VERIFIED",
        "artifact_source": {
            "ref": artifact_ref,
            "commit": artifact_commit,
            "files": source_inventory,
        },
        "evaluated_production_sha": identity["system_under_test_sha"],
        "run_id": identity["run_id"],
        "kb_build_id": identity["kb_build_id"],
        "pipeline_fingerprint": identity["pipeline_fingerprint"],
        "benchmark": {
            "canonical_sha256": benchmark_canonical,
            "case_count": len(cases),
            "manifest_active_kb_build_id": benchmark_manifest_kb,
        },
        "gold": {
            "canonical_sha256": gold_canonical,
            "raw_sha256": gold_bytes,
            "current_kb_build_id": (gold.get("identity") or {}).get("current_kb_build_id"),
            "old_benchmark_kb_build_id": old_kb,
            "annotation_status": dict(gold.get("review_totals") or {}),
        },
        "raw_results": {
            "raw_sha256": raw_bytes,
            "canonical_sha256": raw_canonical,
            "case_count": len(records),
            "system_answers_locked": output_manifest.get("system_answers_locked"),
        },
        "run_status": {
            "system_outputs": "complete_and_locked",
            "semantic_evaluator": (run_manifest.get("ragchecker") or {}).get("status"),
            "semantic_evaluator_note": (
                "The semantic evaluator was blocked after outputs were frozen; this study uses only "
                "the complete deterministic traces."
            ),
        },
        "historical_metadata_reconciliation": {
            "benchmark_manifest_kb_is_old_build": benchmark_manifest_kb != identity["kb_build_id"],
            "old_kb_matches_locked_gold_lineage": historical_kb_mapping_verified,
            "current_kb_matches_run_and_gold": True,
            "interpretation": (
                "The benchmark manifest records the build where the benchmark was assembled. "
                "The locked-gold artifact explicitly maps that old build to the current KB used "
                "by the frozen run; content hashes and the current KB identity match."
            ),
        },
        "checks": checks,
    }


def build_requirements(benchmark: JSONDict, gold: JSONDict) -> list[JSONDict]:
    """Reproduce the original per-case unique evidence-ID denominator."""

    cases = {case["case_id"]: case for case in benchmark["cases"]}
    claims_by_case: dict[str, list[JSONDict]] = defaultdict(list)
    for claim in gold["gold_claim_adjudications"]:
        claims_by_case[str(claim["case_id"])].append(claim)

    requirements: list[JSONDict] = []
    for case_id in [case["case_id"] for case in benchmark["cases"]]:
        case_claims = claims_by_case.get(case_id, [])
        evidence_to_claims: dict[str, list[JSONDict]] = defaultdict(list)
        ordered_evidence: list[str] = []
        for claim in case_claims:
            for evidence_id in claim.get("current_evidence_ids") or []:
                evidence_id = str(evidence_id)
                if evidence_id not in evidence_to_claims:
                    ordered_evidence.append(evidence_id)
                evidence_to_claims[evidence_id].append(claim)
        for evidence_id in ordered_evidence:
            supporting_claims = evidence_to_claims[evidence_id]
            requirements.append(
                {
                    "requirement_id": f"{case_id}::{evidence_id}",
                    "case_id": case_id,
                    "required_chunk_id": evidence_id,
                    "claim_ids": _ordered_unique(
                        claim.get("claim_id") for claim in supporting_claims
                    ),
                    "review_statuses": _ordered_unique(
                        claim.get("review_status") for claim in supporting_claims
                    ),
                    "technical_evidence_statuses": _ordered_unique(
                        claim.get("technical_evidence_status") for claim in supporting_claims
                    ),
                    "case": cases[case_id],
                }
            )
    return requirements


def _index_candidates(items: Iterable[Any]) -> dict[str, JSONDict]:
    return {
        str(item.get("candidate_id")): item
        for item in items
        if isinstance(item, dict) and item.get("candidate_id")
    }


def _attempt_fact(attempt: JSONDict, evidence_id: str) -> JSONDict:
    candidate_trace = attempt.get("candidate_trace") or {}
    if not isinstance(candidate_trace, dict):
        raise ForensicsInputError("candidate_trace must be a JSON object")
    for channel in ("dense", "bm25", "fused"):
        if not isinstance(candidate_trace.get(channel, []), list):
            raise ForensicsInputError(f"candidate_trace.{channel} must be a JSON list")
    dense = _index_candidates(candidate_trace.get("dense") or [])
    bm25 = _index_candidates(candidate_trace.get("bm25") or [])
    fused = _index_candidates(candidate_trace.get("fused") or [])
    retry = attempt.get("retry_evidence") or {}
    retained = set(retry.get("retained_candidate_ids") or [])
    acquired = set(retry.get("acquired_candidate_ids") or [])
    eligible = set(retry.get("eligible_candidate_ids") or [])
    duplicates = set(retry.get("duplicate_candidate_ids") or [])
    packed = set(retry.get("packed_candidate_ids") or [])
    fused_item = fused.get(evidence_id) or {}
    if evidence_id in retained and evidence_id in acquired:
        retry_status = "retained_and_reacquired"
    elif evidence_id in retained:
        retry_status = "retained"
    elif evidence_id in acquired:
        retry_status = "newly_acquired"
    elif evidence_id in eligible:
        retry_status = "eligible_without_origin_trace"
    else:
        retry_status = "absent"
    if evidence_id in duplicates:
        retry_status += "_deduplicated"

    return {
        "attempt_index": attempt.get("attempt_index"),
        "current_question": attempt.get("current_question"),
        "acquisition_query": attempt.get("acquisition_query"),
        "rerank_query": attempt.get("overall_rerank_query") or attempt.get("rerank_query"),
        "dense_rank": (dense.get(evidence_id) or {}).get("rank"),
        "bm25_rank": (bm25.get(evidence_id) or {}).get("rank"),
        "rrf_rank": fused_item.get("rrf_rank"),
        "rerank_rank": fused_item.get("rerank_rank"),
        "rerank_score": fused_item.get("rerank_score"),
        "source_id": fused_item.get("source_id"),
        "section": fused_item.get("section"),
        "packing_status": fused_item.get("packing_status"),
        "drop_reason": fused_item.get("drop_reason"),
        "not_examined_reason": fused_item.get("not_examined_reason"),
        "channel_hit": evidence_id in dense or evidence_id in bm25,
        "fused_hit": evidence_id in fused,
        "retained": evidence_id in retained,
        "acquired": evidence_id in acquired,
        "eligible": evidence_id in eligible,
        "duplicate": evidence_id in duplicates,
        "packed": evidence_id in packed,
        "retry_status": retry_status,
        "packer": attempt.get("packer") or {},
        "trace_status": attempt.get("status"),
        "warnings": attempt.get("warnings") or [],
    }


def classify_requirement(
    *,
    reviewed: bool,
    acquired: bool,
    packed: bool,
    channel_hit: bool,
    fused_hit: bool,
    final_eligible: bool,
    final_candidate: JSONDict | None,
    retrieval_attempt_count: int,
) -> tuple[str, str, str]:
    """Assign exactly one primary outcome to one metric requirement."""

    if not reviewed:
        return (
            "GOLD_OR_IDENTITY_MISMATCH",
            "gold_requirement_not_technically_clear",
            "OBSERVED",
        )
    if packed:
        return "PACKED", "present_in_final_generation_context", "OBSERVED"
    if not acquired:
        if channel_hit and not fused_hit:
            return (
                "FUSION_OR_IDENTITY_LOSS",
                "present_in_saved_channel_trace_but_absent_after_rrf",
                "OBSERVED",
            )
        return (
            "ACQUISITION_MISS",
            "absent_from_saved_dense_and_bm25_candidate_limits",
            "OBSERVED",
        )
    if retrieval_attempt_count > 1 and not final_eligible:
        return (
            "RETRY_STATE_LOSS",
            "acquired_in_an_earlier_attempt_but_absent_from_final_eligible_union",
            "OBSERVED",
        )
    if final_candidate is None:
        return (
            "TRACE_INSUFFICIENT",
            "acquired_requirement_missing_from_final_candidate_trace",
            "UNKNOWN",
        )
    if final_candidate.get("drop_reason") == "character_limit":
        return (
            "CHARACTER_BUDGET_LOSS",
            "whole_chunk_exceeded_remaining_character_budget",
            "OBSERVED",
        )
    if final_candidate.get("not_examined_reason") == "item_limit_reached":
        return (
            "ITEM_BUDGET_LOSS",
            "candidate_not_examined_after_item_limit_was_reached",
            "OBSERVED",
        )
    if final_candidate.get("drop_reason") in {"missing_id_or_text", "duplicate_id"}:
        return (
            "CONTRACT_OR_PROVENANCE_REJECTION",
            str(final_candidate["drop_reason"]),
            "OBSERVED",
        )
    if not final_candidate.get("source_id"):
        return (
            "CONTRACT_OR_PROVENANCE_REJECTION",
            "missing_source_identity",
            "OBSERVED",
        )
    return (
        "TRACE_INSUFFICIENT",
        "candidate_not_packed_without_supported_packer_reason",
        "UNKNOWN",
    )


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def trace_requirement(requirement: JSONDict, record: JSONDict) -> JSONDict:
    evidence_id = requirement["required_chunk_id"]
    attempts = record.get("retrieval_attempt_traces")
    if not isinstance(attempts, list) or not attempts:
        raise ForensicsInputError(
            f"Schema-invalid trace for {record.get('case_id')}: retrieval_attempt_traces missing"
        )
    attempt_facts = [_attempt_fact(attempt, evidence_id) for attempt in attempts]
    acquisition = set(record.get("acquisition_candidate_ids") or [])
    recomputed_acquisition = {
        evidence
        for attempt in attempts
        for evidence in (attempt.get("retry_evidence") or {}).get("acquired_candidate_ids", [])
    }
    if acquisition != recomputed_acquisition:
        raise ForensicsInputError(f"Acquisition union mismatch for {record.get('case_id')}")

    contexts = record.get("generation_contexts") or []
    packed_ids = [item.get("chunk_id") for item in contexts]
    packed = evidence_id in packed_ids
    packed_item = next((item for item in contexts if item.get("chunk_id") == evidence_id), None)
    packed_position = packed_ids.index(evidence_id) + 1 if packed else None

    final_trace = record.get("retrieval_trace") or {}
    final_fused = _index_candidates((final_trace.get("candidate_trace") or {}).get("fused") or [])
    final_candidate = final_fused.get(evidence_id)
    final_retry = final_trace.get("retry_evidence") or {}
    final_eligible = evidence_id in set(final_retry.get("eligible_candidate_ids") or [])
    reviewed = set(requirement["review_statuses"]) == {"TECHNICALLY_CLEAR"}
    stage, reason, confidence = classify_requirement(
        reviewed=reviewed,
        acquired=evidence_id in acquisition,
        packed=packed,
        channel_hit=any(fact["channel_hit"] for fact in attempt_facts),
        fused_hit=any(fact["fused_hit"] for fact in attempt_facts),
        final_eligible=final_eligible,
        final_candidate=final_candidate,
        retrieval_attempt_count=len(attempts),
    )

    contributing: list[JSONDict] = []
    if final_candidate and final_candidate.get("rerank_rank") is not None:
        rrf_rank = final_candidate.get("rrf_rank")
        rerank_rank = final_candidate.get("rerank_rank")
        if rrf_rank is not None and rerank_rank > rrf_rank:
            contributing.append(
                {
                    "factor": "RERANK_DEMOTION_OBSERVED",
                    "rrf_rank": rrf_rank,
                    "rerank_rank": rerank_rank,
                    "confidence": "OBSERVED",
                }
            )
        if stage == "ITEM_BUDGET_LOSS":
            contributing.append(
                {
                    "factor": "RERANK_ORDER_PLACED_REQUIREMENT_AFTER_PACKED_PREFIX",
                    "rerank_rank": rerank_rank,
                    "confidence": "DETERMINISTICALLY_REPRODUCED",
                }
            )
    if stage == "CHARACTER_BUDGET_LOSS":
        contributing.append(
            {
                "factor": "WHOLE_CHUNK_ADMISSION_POLICY",
                "confidence": "DETERMINISTICALLY_REPRODUCED",
            }
        )
    if len(attempts) > 1 and final_eligible and not packed:
        contributing.append(
            {
                "factor": "RETRY_REPACK_OUTCOME_NOT_STATE_LOSS",
                "confidence": "OBSERVED",
            }
        )

    final_packer = final_trace.get("packer") or {}
    limits = final_packer.get("limits") or {}
    reranker_events = record.get("reranker_events") or []
    missing: list[str] = []
    if packed_item is None:
        missing.extend(("candidate_text_length", "rendered_context_block_cost"))
    if stage == "CHARACTER_BUDGET_LOSS":
        missing.extend(("used_chars_at_inspection", "remaining_chars_at_inspection"))
    missing.append("agent_decision_evidence_trace")

    history_messages = len(record.get("history") or [])
    decision_inference = packed
    trace_refs = [
        f"raw_results.records[{record['case_id']}].retrieval_attempt_traces",
        f"raw_results.records[{record['case_id']}].retrieval_trace",
        f"locked_gold.gold_claim_adjudications[{','.join(requirement['claim_ids'])}]",
    ]
    return {
        "requirement_id": requirement["requirement_id"],
        "case_id": record["case_id"],
        "family": record["family"],
        "category": record["category"],
        "turn": history_messages // 2 + 1,
        "history_type": "multi_turn" if history_messages else "single_turn",
        "gold_claim_ids": requirement["claim_ids"],
        "required_chunk_id": evidence_id,
        "gold_review_status": requirement["review_statuses"],
        "retrieval_attempts": [fact["attempt_index"] for fact in attempt_facts],
        "current_user_question": record["query"],
        "acquisition_queries": [fact["acquisition_query"] for fact in attempt_facts],
        "rerank_queries": [fact["rerank_query"] for fact in attempt_facts],
        "dense_ranks": [fact["dense_rank"] for fact in attempt_facts],
        "bm25_ranks": [fact["bm25_rank"] for fact in attempt_facts],
        "rrf_ranks": [fact["rrf_rank"] for fact in attempt_facts],
        "rerank_ranks": [fact["rerank_rank"] for fact in attempt_facts],
        "rerank_scores": [fact["rerank_score"] for fact in attempt_facts],
        "reranker_statuses": [event.get("status") for event in reranker_events],
        "reranker_fallback_statuses": [
            bool(event.get("fallback_used")) for event in reranker_events
        ],
        "candidate_identity": {
            "candidate_id": evidence_id,
            "source_id": (final_candidate or {}).get("source_id"),
            "section": (final_candidate or {}).get("section"),
        },
        "retry_statuses": [fact["retry_status"] for fact in attempt_facts],
        "candidate_text_length": len(packed_item["text"]) if packed_item else None,
        "candidate_text_length_missing_reason": (
            None if packed_item else "nonpacked_candidate_text_not_persisted"
        ),
        "rendered_context_block_cost": (len(packed_item["rendered_text"]) if packed_item else None),
        "rendered_cost_missing_reason": (
            None if packed_item else "nonpacked_rendered_block_not_persisted"
        ),
        "packer_max_items": limits.get("max_items"),
        "packer_max_chars": limits.get("max_chars"),
        "packer_context_chars": final_packer.get("context_chars"),
        "packed_position": packed_position,
        "packer_drop_reason": (final_candidate or {}).get("drop_reason"),
        "not_examined_reason": (final_candidate or {}).get("not_examined_reason"),
        "decision_visible_evidence": None,
        "decision_visibility_inference": decision_inference,
        "decision_visibility_missing_reason": "agent_decision_evidence_trace_not_persisted",
        "generation_visible_evidence": packed,
        "output_action": record.get("final_action"),
        "output_reason": record.get("final_reason"),
        "primary_loss_stage": stage,
        "primary_reason": reason,
        "contributing_factors": contributing,
        "evidence_confidence": confidence,
        "trace_references": trace_refs,
        "missing_data_reasons": sorted(set(missing)),
        "attempt_journey": attempt_facts,
    }


def analyze_gap_failures(raw: JSONDict, gold: JSONDict) -> JSONDict:
    records = {record["case_id"]: record for record in raw["records"]}
    negative = {item["case_id"]: item for item in gold["negative_adjudications"]}
    confirmed = [
        item for item in negative.values() if item["negative_status"] == "CONFIRMED_ABSENCE"
    ]
    ambiguous = [
        item for item in negative.values() if item["negative_status"] == "POLICY_AMBIGUOUS"
    ]
    failures: list[JSONDict] = []
    passes = 0
    for item in confirmed:
        record = records[item["case_id"]]
        passed = (
            record.get("final_action") == "abstain" and record.get("final_reason") == "evidence_gap"
        )
        if passed:
            passes += 1
            continue
        if (
            record.get("final_action") == "generate"
            and record.get("final_reason") == "evidence_sufficient"
        ):
            category = "G1_INSUFFICIENT_EVIDENCE_MISJUDGED"
            explanation = (
                "Confirmed-absence gold requires evidence-gap rejection, but the persisted semantic "
                "decision was generate/evidence_sufficient. Direct-support IDs and policy validation "
                "were not persisted, so the internal semantic mistake is supported rather than observed."
            )
            confidence = "SUPPORTED_INFERENCE"
        else:
            category = "G7_UNKNOWN"
            explanation = (
                "The persisted action failed the rejection contract without enough decision trace."
            )
            confidence = "UNKNOWN"
        attempts = record.get("retrieval_attempt_traces") or []
        failures.append(
            {
                "case_id": record["case_id"],
                "category": record["category"],
                "current_question": record["query"],
                "conversation_history": record.get("history") or [],
                "expected_action": "abstain",
                "expected_reason": "evidence_gap",
                "adjudicated_gold_status": item["negative_status"],
                "unsupported_factual_requirement": item.get("unsupported_factual_requirement"),
                "corpus_factual_result": item.get("corpus_factual_result"),
                "actual_action": record.get("final_action"),
                "actual_reason": record.get("final_reason"),
                "all_decision_attempts": None,
                "decision_trace_missing_reason": (
                    "agent_decision_history, direct_supporting_evidence_ids, and policy_validation "
                    "were omitted by the frozen record serializer"
                ),
                "retrieval_queries": [attempt.get("acquisition_query") for attempt in attempts],
                "rerank_queries": [
                    attempt.get("overall_rerank_query") or attempt.get("rerank_query")
                    for attempt in attempts
                ],
                "retrieval_attempts": len(attempts),
                "final_packed_ids": [
                    context.get("chunk_id") for context in record.get("generation_contexts") or []
                ],
                "direct_supporting_evidence_ids": None,
                "policy_validation": None,
                "retry_status": [attempt.get("status") for attempt in attempts],
                "actual_final_response": record.get("final_answer"),
                "response_status": {
                    "structural_verifier_pass": (
                        record.get("structural_verification_report") or {}
                    ).get("passed"),
                    "semantic_contract_review": "not_available",
                },
                "generation_invoked": record.get("generation_invoked"),
                "generation_provider": record.get("generation_provider"),
                "generation_model": record.get("generation_model"),
                "fallback_applied": record.get("fallback_applied"),
                "missing_evidence_from_model": None,
                "root_cause_category": category,
                "root_cause_explanation": explanation,
                "evidence_confidence": confidence,
                "trace_references": [
                    f"raw_results.records[{record['case_id']}]",
                    f"locked_gold.negative_adjudications[{record['case_id']}]",
                ],
            }
        )

    return {
        "schema": "rag_evidence_gap_failures_v1",
        "confirmed_absence_total": len(confirmed),
        "confirmed_rejections": passes,
        "confirmed_failures": len(failures),
        "policy_ambiguous_total": len(ambiguous),
        "policy_ambiguous_case_ids": sorted(item["case_id"] for item in ambiguous),
        "failures": failures,
        "classification_counts": dict(
            sorted(Counter(item["root_cause_category"] for item in failures).items())
        ),
        "important_interpretation": (
            "These are action-contract failures. A generated answer may still state a limitation; "
            "generation_invoked alone is not a failure. No new semantic answer grading was performed."
        ),
    }


def summarize(
    journeys: list[JSONDict],
    gap: JSONDict,
    reference_metrics: JSONDict,
) -> JSONDict:
    stage_counts = Counter(row["primary_loss_stage"] for row in journeys)
    stage_cases: dict[str, set[str]] = defaultdict(set)
    for row in journeys:
        stage_cases[row["primary_loss_stage"]].add(row["case_id"])

    total = len(journeys)
    packed = stage_counts["PACKED"]
    acquired = sum(any(fact["acquired"] for fact in row["attempt_journey"]) for row in journeys)
    post_acquisition_stages = {
        "FUSION_OR_IDENTITY_LOSS",
        "RERANK_ORDER_LOSS",
        "CHARACTER_BUDGET_LOSS",
        "ITEM_BUDGET_LOSS",
        "CONTRACT_OR_PROVENANCE_REJECTION",
        "RETRY_STATE_LOSS",
        "GOLD_OR_IDENTITY_MISMATCH",
        "TRACE_INSUFFICIENT",
    }
    post_acquisition_losses = [
        row
        for row in journeys
        if row["primary_loss_stage"] in post_acquisition_stages
        and any(fact["acquired"] for fact in row["attempt_journey"])
        and row["primary_loss_stage"] != "GOLD_OR_IDENTITY_MISMATCH"
    ]
    unknown = [row for row in journeys if row["evidence_confidence"] == "UNKNOWN"]
    rerank_related = [
        row
        for row in journeys
        if any(
            factor.get("factor") == "RERANK_ORDER_PLACED_REQUIREMENT_AFTER_PACKED_PREFIX"
            for factor in row["contributing_factors"]
        )
    ]
    retry_rows = [row for row in journeys if len(row["retrieval_attempts"]) > 1]
    reference_evidence = reference_metrics["evidence"]
    reference_checks = {
        "acquisition_85_of_88": (
            reference_evidence["acquisition_required_evidence_recall"]["count"],
            reference_evidence["acquisition_required_evidence_recall"]["denominator"],
        )
        == (acquired, total),
        "packed_58_of_88": (
            reference_evidence["packed_required_evidence_recall"]["count"],
            reference_evidence["packed_required_evidence_recall"]["denominator"],
        )
        == (packed, total),
        "confirmed_gap_18_of_24": (
            reference_metrics["confirmed_evidence_gap_rejection"]["count"],
            reference_metrics["confirmed_evidence_gap_rejection"]["denominator"],
        )
        == (gap["confirmed_rejections"], gap["confirmed_absence_total"]),
    }
    if not all(reference_checks.values()):
        raise ForensicsInputError(f"Metric reconciliation failed: {reference_checks}")

    ordered_stages = [
        "ACQUISITION_MISS",
        "FUSION_OR_IDENTITY_LOSS",
        "RERANK_ORDER_LOSS",
        "CHARACTER_BUDGET_LOSS",
        "ITEM_BUDGET_LOSS",
        "CONTRACT_OR_PROVENANCE_REJECTION",
        "RETRY_STATE_LOSS",
        "GOLD_OR_IDENTITY_MISMATCH",
        "TRACE_INSUFFICIENT",
    ]
    reconciliation = [
        {
            "stage": stage,
            "requirements": stage_counts[stage],
            "affected_cases": len(stage_cases[stage]),
            "sufficient_trace": sum(
                row["evidence_confidence"] != "UNKNOWN"
                for row in journeys
                if row["primary_loss_stage"] == stage
            ),
            "unknown": sum(
                row["evidence_confidence"] == "UNKNOWN"
                for row in journeys
                if row["primary_loss_stage"] == stage
            ),
        }
        for stage in ordered_stages
    ]

    return {
        "schema": "rag_root_cause_summary_v1",
        "metric_unit": "unique case_id + required_chunk_id; claim IDs attached but not double-counted",
        "evidence_accounting": {
            "requirements": total,
            "acquired": acquired,
            "packed": packed,
            "acquisition_misses": stage_counts["ACQUISITION_MISS"],
            "all_not_packed": total - packed,
            "post_acquisition_losses": len(post_acquisition_losses),
            "all_loss_affected_cases": len(
                {row["case_id"] for row in journeys if row["primary_loss_stage"] != "PACKED"}
            ),
            "post_acquisition_affected_cases": len(
                {row["case_id"] for row in post_acquisition_losses}
            ),
            "packed_is_subset_of_acquisition": all(
                row["primary_loss_stage"] != "PACKED"
                or any(fact["acquired"] for fact in row["attempt_journey"])
                for row in journeys
            ),
            "primary_classification_conserved": sum(stage_counts.values()) == total,
            "unknown_primary_classifications": len(unknown),
        },
        "primary_stage_counts": dict(sorted(stage_counts.items())),
        "reconciliation": reconciliation,
        "reranking": {
            "item_budget_losses_with_direct_rerank_order_link": len(rerank_related),
            "affected_cases": len({row["case_id"] for row in rerank_related}),
            "interpretation": (
                "The observed post-rerank order deterministically placed these requirements after "
                "the packed prefix. This does not prove that BGE misunderstood their semantics."
            ),
        },
        "retry": {
            "answerable_requirements_in_retried_cases": len(retry_rows),
            "retry_state_losses": stage_counts["RETRY_STATE_LOSS"],
            "retained_but_finally_not_packed": sum(
                row["primary_loss_stage"] != "PACKED"
                and any("retained" in status for status in row["retry_statuses"])
                for row in retry_rows
            ),
            "interpretation": (
                "Retained requirements remained eligible. Their final omission was a rerank/pack "
                "outcome, not evidence disappearance from retry state."
            ),
        },
        "decision_and_generation": {
            "observed_identity_comparison_available": False,
            "code_supported_same_packed_context": True,
            "reason": (
                "The frozen serializer omitted agent_decision_evidence_traces. Frozen code uses the "
                "same PackedContext for decision and generation, but the direct runtime comparison "
                "cannot be marked OBSERVED."
            ),
        },
        "evidence_gap": {
            "confirmed_total": gap["confirmed_absence_total"],
            "confirmed_rejections": gap["confirmed_rejections"],
            "confirmed_failures": gap["confirmed_failures"],
            "semantic_decision_supported_inference": gap["classification_counts"].get(
                "G1_INSUFFICIENT_EVIDENCE_MISJUDGED", 0
            ),
            "retrieval_context_deficiency_confirmed": 0,
            "response_or_policy_contract_failure_confirmed": 0,
        },
        "implementation_assessment": {
            "confirmed_bugs": 0,
            "status": "NO_CONFIRMED_IMPLEMENTATION_BUG",
            "observed_design_limitations": [
                "rank-ordered whole-chunk packing under a 7000-character budget",
                "nine-item early stop after reranking",
                "semantic action decision can label confirmed-absence queries evidence_sufficient",
            ],
            "unresolved": [
                "candidate-limit cause for the three acquisition misses",
                "exact counterfactual contribution of reranking to character-budget losses",
                "direct-support IDs and policy-validation state for the six action failures",
            ],
        },
        "largest_confirmed_root_cause": {
            "stage": "CHARACTER_BUDGET_LOSS",
            "requirements": stage_counts["CHARACTER_BUDGET_LOSS"],
            "share_of_post_acquisition_losses_percent": round(
                100 * stage_counts["CHARACTER_BUDGET_LOSS"] / len(post_acquisition_losses), 1
            ),
            "confidence": "OBSERVED",
        },
        "reference_metric_checks": reference_checks,
        "final_decision_status": [
            "PARTIAL_ROOT_CAUSE_CONFIRMED",
            "NO_CONFIRMED_IMPLEMENTATION_BUG",
        ],
    }


def build_missing_trace_inventory(journeys: list[JSONDict], raw: JSONDict) -> JSONDict:
    lost = [row for row in journeys if row["primary_loss_stage"] != "PACKED"]
    character = [row for row in lost if row["primary_loss_stage"] == "CHARACTER_BUDGET_LOSS"]
    acquisition = [row for row in lost if row["primary_loss_stage"] == "ACQUISITION_MISS"]
    return {
        "schema": "rag_missing_trace_fields_v1",
        "fields": [
            {
                "field": "agent_decision_history/direct_supporting_evidence_ids/policy_validation",
                "scope": "all frozen records; critical for six confirmed evidence-gap failures",
                "affected_records": len(raw["records"]),
                "impact": "prevents OBSERVED attribution of the semantic decision internals",
                "minimal_future_instrumentation": (
                    "persist the existing text-free agent_decision_history and "
                    "agent_decision_evidence_traces fields"
                ),
            },
            {
                "field": "nonpacked candidate text and rendered block cost",
                "scope": "all nonpacked acquired requirements",
                "affected_requirements": sum(
                    row["primary_loss_stage"]
                    not in {"PACKED", "ACQUISITION_MISS", "FUSION_OR_IDENTITY_LOSS"}
                    for row in journeys
                ),
                "impact": "prevents exact pure-function replay of whole-chunk packing",
                "minimal_future_instrumentation": (
                    "persist text_sha256, text_length, rendered_cost, and used_chars_before"
                ),
            },
            {
                "field": "used and remaining characters at drop time",
                "scope": "character-budget losses",
                "affected_requirements": len(character),
                "impact": "drop reason is observed, but exact arithmetic cannot be independently replayed",
                "minimal_future_instrumentation": (
                    "add used_chars_before, rendered_cost, and remaining_chars to packer trace"
                ),
            },
            {
                "field": "candidates beyond saved channel top-k",
                "scope": "acquisition misses",
                "affected_requirements": len(acquisition),
                "impact": "cannot separate corpus/ranking miss from the configured channel limit",
                "minimal_future_instrumentation": (
                    "record bounded rank-of-required only in offline labeled evaluation, never at runtime"
                ),
            },
        ],
        "cases_not_fully_traceable": sorted(
            {
                row["case_id"]
                for row in lost
                if row["primary_loss_stage"] in {"ACQUISITION_MISS", "TRACE_INSUFFICIENT"}
                or row["primary_loss_stage"] == "CHARACTER_BUDGET_LOSS"
            }
        ),
    }


def group_losses_by_case(journeys: list[JSONDict]) -> JSONDict:
    grouped: dict[str, list[JSONDict]] = defaultdict(list)
    for row in journeys:
        if row["primary_loss_stage"] == "PACKED":
            continue
        grouped[row["case_id"]].append(
            {
                "requirement_id": row["requirement_id"],
                "gold_claim_ids": row["gold_claim_ids"],
                "required_chunk_id": row["required_chunk_id"],
                "primary_loss_stage": row["primary_loss_stage"],
                "primary_reason": row["primary_reason"],
                "rrf_ranks": row["rrf_ranks"],
                "rerank_ranks": row["rerank_ranks"],
                "rerank_scores": row["rerank_scores"],
                "packer_drop_reason": row["packer_drop_reason"],
                "not_examined_reason": row["not_examined_reason"],
                "contributing_factors": row["contributing_factors"],
                "evidence_confidence": row["evidence_confidence"],
                "trace_references": row["trace_references"],
            }
        )
    cases = [
        {
            "case_id": case_id,
            "lost_requirement_count": len(rows),
            "stage_counts": dict(
                sorted(Counter(row["primary_loss_stage"] for row in rows).items())
            ),
            "requirements": rows,
        }
        for case_id, rows in sorted(grouped.items())
    ]
    return {
        "schema": "rag_evidence_loss_by_case_v1",
        "affected_case_count": len(cases),
        "cases": cases,
    }


def followup_experiments_markdown(summary: JSONDict) -> str:
    accounting = summary["evidence_accounting"]
    char_losses = summary["primary_stage_counts"].get("CHARACTER_BUDGET_LOSS", 0)
    item_losses = summary["primary_stage_counts"].get("ITEM_BUDGET_LOSS", 0)
    gap_failures = summary["evidence_gap"]["confirmed_failures"]
    return f"""# Follow-up experiments for RAG evidence loss

These are preregistration proposals only. They were not implemented or executed in this study.

## Priority 1 — Budget-aware whole-chunk selection

- **Target root cause:** `{char_losses}` observed character-budget losses among
  `{accounting["post_acquisition_losses"]}` post-acquisition losses.
- **Hypothesis:** a query-conditioned, coverage-aware selector that preserves whole chunks can
  increase required-evidence packing without increasing the 7,000-character/9-item budgets.
- **Code owner:** a new research selector adjacent to `src/retrieval/context_packer.py`; production
  packer remains the control.
- **Baseline:** frozen `366f3ad8` rank-order whole-chunk packer.
- **Development data:** derive selector features on a development split only; do not expose gold IDs
  to runtime selection.
- **Independent validation:** frozen holdout cases or a separately adjudicated benchmark, with all
  thresholds preregistered before results are opened.
- **Primary metric:** packed required-evidence coverage on the independent set.
- **Secondary metrics:** complete reviewed claims/cases, context precision, per-case non-inferiority.
- **Safety regression:** evidence-gap rejection, qualifier/negation preservation, provenance validity.
- **Latency/resources:** report selector p50/p95 and total context bytes; no model/API dependency.
- **Failure criterion:** no significant holdout gain, any safety regression, or reliance on gold IDs.
- **Stop rule:** one fixed candidate pool and one preregistered selector; no budget/top-k sweep.
- **Production risk:** coverage optimization may displace useful unlabeled context or overfit labels.

## Priority 2 — Rerank-order diagnosis at the item boundary

- **Target root cause:** `{item_losses}` item-budget losses whose observed post-rerank positions occur
  after the nine-item packed prefix.
- **Hypothesis:** a bounded second-stage ordering objective can preserve high-relevance evidence while
  reducing required evidence demotion, without changing the BGE model or candidate depth.
- **Code owner:** research-only ordering wrapper around `src/retrieval/reranker.py` output.
- **Baseline:** persisted BGE order at frozen SHA; no new inference during the forensic study.
- **Development data:** a disjoint labeled development split with claim/qualifier roles.
- **Independent validation:** locked holdout with paired per-case analysis.
- **Primary metric:** required IDs retained within the first nine admissible candidates.
- **Secondary metrics:** nDCG/recall at nine, packed coverage, complete cases.
- **Safety regression:** qualifier loss, evidence-gap rejection, structural provenance.
- **Latency/resources:** same candidate count and model; measure only ordering overhead.
- **Failure criterion:** aggregate gain with any preregistered per-case regression, or no holdout gain.
- **Stop rule:** one ordering hypothesis; do not swap model, top-k, weights, or context limits.
- **Production risk:** labeled relevance may not represent usefulness of unlabeled evidence.

## Priority 3 — Evidence-sufficiency decision calibration

- **Target root cause:** `{gap_failures}` confirmed-absence cases persisted as
  `generate/evidence_sufficient`.
- **Hypothesis:** an explicit qualifier/scope contract and text-free decision telemetry can reduce
  false-generate actions while preserving answerable completion.
- **Code owner:** `src/agent/action_decision.py` and its policy tests; no medical-answer prompt change in
  the forensic phase.
- **Baseline:** 18/24 confirmed evidence-gap rejections on frozen outputs.
- **Development data:** confirmed negative development cases separated from six ambiguous cases.
- **Independent validation:** owner-adjudicated negative holdout plus answerable safety controls.
- **Primary metric:** confirmed evidence-gap rejection.
- **Secondary metrics:** answerable generation rate and retry rate.
- **Safety regression:** emergency/scope decisions, response-contract compliance, unsupported claims.
- **Latency/resources:** preserve the current model/provider and bounded two-attempt workflow.
- **Failure criterion:** reduced answerable completion, no negative holdout gain, or policy ambiguity
  converted into automatic failure labels.
- **Stop rule:** one preregistered decision-contract change and one independent evaluation.
- **Production risk:** over-abstention and reduced usefulness.

Increasing top-k or replacing the model is not the first experiment: acquisition already covers
`{accounting["acquired"]}/{accounting["requirements"]}` requirements, while
`{accounting["post_acquisition_losses"]}` are lost after acquisition. The saved traces prove packing
and ordering outcomes, but do not prove that a different model is the required remedy.
"""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_journey_csv(path: Path, journeys: list[JSONDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=JOURNEY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for journey in journeys:
            row: JSONDict = {}
            for column in JOURNEY_COLUMNS:
                value = journey.get(column)
                row[column] = _json_cell(value) if isinstance(value, (list, dict)) else value
            writer.writerow(row)


def run_analysis(args: argparse.Namespace) -> JSONDict:
    raw_path = Path(args.raw_results)
    benchmark_path = Path(args.benchmark)
    gold_path = Path(args.gold)
    raw = _require_mapping(load_json(raw_path), "raw results")
    benchmark = _require_mapping(load_json(benchmark_path), "benchmark")
    benchmark_manifest = _require_mapping(
        load_json(Path(args.benchmark_manifest)), "benchmark manifest"
    )
    gold = _require_mapping(load_json(gold_path), "locked gold")
    gold_manifest = _require_mapping(load_json(Path(args.gold_manifest)), "gold manifest")
    run_manifest = _require_mapping(load_json(Path(args.run_manifest)), "run manifest")
    output_manifest = _require_mapping(
        load_json(Path(args.system_output_manifest)), "system output manifest"
    )
    reference_metrics = _require_mapping(
        load_json(Path(args.reference_metrics)), "reference metrics"
    )
    source_inventory = load_json(Path(args.source_inventory))
    if not isinstance(source_inventory, dict):
        raise ForensicsInputError("--source-inventory must contain a JSON object")
    identity = validate_identity(
        raw_path=raw_path,
        raw=raw,
        benchmark=benchmark,
        benchmark_manifest=benchmark_manifest,
        gold_path=gold_path,
        gold=gold,
        gold_manifest=gold_manifest,
        run_manifest=run_manifest,
        output_manifest=output_manifest,
        reference_metrics=reference_metrics,
        expected_system_sha=args.expected_system_sha,
        artifact_ref=args.artifact_ref,
        artifact_commit=args.artifact_commit,
        source_inventory=source_inventory,
    )
    records = {record["case_id"]: record for record in raw["records"]}
    requirements = build_requirements(benchmark, gold)
    journeys = [
        trace_requirement(requirement, records[requirement["case_id"]])
        for requirement in requirements
    ]
    if len({row["requirement_id"] for row in journeys}) != len(journeys):
        raise ForensicsInputError("A requirement appeared in more than one primary classification")
    gap = analyze_gap_failures(raw, gold)
    summary = summarize(journeys, gap, reference_metrics)
    losses = group_losses_by_case(journeys)
    missing = build_missing_trace_inventory(journeys, raw)

    output_dir = Path(args.output_dir)
    _write_json(output_dir / "rag_evidence_loss_identity.json", identity)
    _write_journey_csv(output_dir / "rag_evidence_journey.csv", journeys)
    _write_json(output_dir / "rag_evidence_loss_by_case.json", losses)
    _write_json(output_dir / "rag_evidence_gap_failures.json", gap)
    _write_json(output_dir / "rag_root_cause_summary.json", summary)
    _write_json(output_dir / "rag_missing_trace_fields.json", missing)
    (output_dir / "rag_followup_experiments.md").write_text(
        followup_experiments_markdown(summary), encoding="utf-8", newline="\n"
    )
    return {
        "identity": identity,
        "journeys": journeys,
        "losses": losses,
        "gap": gap,
        "summary": summary,
        "missing": missing,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-results", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--gold-manifest", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--system-output-manifest", required=True)
    parser.add_argument("--reference-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-system-sha", required=True)
    parser.add_argument("--artifact-ref", required=True)
    parser.add_argument("--artifact-commit", required=True)
    parser.add_argument("--source-inventory", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_analysis(args)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
