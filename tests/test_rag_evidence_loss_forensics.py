from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.rag_evidence_loss_forensics import (
    ForensicsInputError,
    _attempt_fact,
    analyze_gap_failures,
    build_requirements,
    canonical_json_sha256,
    classify_requirement,
    trace_requirement,
    validate_identity,
)


def _requirement(evidence_id: str = "gold-1", *, reviewed: bool = True) -> dict:
    return {
        "requirement_id": f"CASE-1::{evidence_id}",
        "case_id": "CASE-1",
        "required_chunk_id": evidence_id,
        "claim_ids": ["CASE-1-C01"],
        "review_statuses": ["TECHNICALLY_CLEAR" if reviewed else "PENDING"],
        "technical_evidence_statuses": ["DIRECT_SUPPORT"],
        "case": {"case_id": "CASE-1"},
    }


def _attempt(
    evidence_id: str = "gold-1",
    *,
    attempt_index: int = 1,
    dense: bool = True,
    bm25: bool = False,
    acquired: bool = True,
    retained: bool = False,
    duplicate: bool = False,
    eligible: bool = True,
    packed: bool = False,
    rrf_rank: int = 2,
    rerank_rank: int | None = 3,
    drop_reason: str | None = None,
    not_examined_reason: str | None = None,
    source_id: str | None = "source-1",
) -> dict:
    fused = []
    if eligible:
        fused.append(
            {
                "candidate_id": evidence_id,
                "rank": rrf_rank,
                "rrf_rank": rrf_rank,
                "fused_score": 0.02,
                "rerank_rank": rerank_rank,
                "rerank_score": 0.5 if rerank_rank is not None else None,
                "dense_rank": 1 if dense else None,
                "bm25_rank": 1 if bm25 else None,
                "seen_in_attempts": [attempt_index],
                "rank_by_attempt": {str(attempt_index): rrf_rank},
                "source_id": source_id,
                "section": "section",
                "packed": packed,
                "packing_status": (
                    "selected"
                    if packed
                    else "dropped"
                    if drop_reason
                    else "not_examined"
                    if not_examined_reason
                    else "unknown"
                ),
                "drop_reason": drop_reason,
                "not_examined_reason": not_examined_reason,
            }
        )
    return {
        "attempt_index": attempt_index,
        "current_question": "question",
        "acquisition_query": f"query-{attempt_index}",
        "overall_rerank_query": f"query-{attempt_index}",
        "status": "ok",
        "candidate_trace": {
            "dense": (
                [{"candidate_id": evidence_id, "rank": 1, "native_score": 0.8}] if dense else []
            ),
            "bm25": (
                [{"candidate_id": evidence_id, "rank": 1, "native_score": 3.0}] if bm25 else []
            ),
            "fused": fused,
        },
        "retry_evidence": {
            "retained_candidate_ids": [evidence_id] if retained else [],
            "acquired_candidate_ids": [evidence_id] if acquired else [],
            "duplicate_candidate_ids": [evidence_id] if duplicate else [],
            "eligible_candidate_ids": [evidence_id] if eligible else [],
            "reranked_candidate_ids": [evidence_id] if eligible else [],
            "packed_candidate_ids": [evidence_id] if packed else [],
        },
        "packer": {
            "limits": {"max_items": 9, "max_chars": 7000},
            "context_chars": 6000,
            "selected_ids": [evidence_id] if packed else [],
            "dropped": (
                [{"candidate_id": evidence_id, "reason": drop_reason}] if drop_reason else []
            ),
            "not_examined": (
                [{"candidate_id": evidence_id, "reason": not_examined_reason}]
                if not_examined_reason
                else []
            ),
        },
        "warnings": [],
    }


def _record(attempts: list[dict], *, packed: bool = False, evidence_id: str = "gold-1") -> dict:
    acquisition = list(
        dict.fromkeys(
            item
            for attempt in attempts
            for item in attempt["retry_evidence"]["acquired_candidate_ids"]
        )
    )
    final = attempts[-1]
    final_trace = {
        "candidate_trace": final["candidate_trace"],
        "retry_evidence": final["retry_evidence"],
        "packer": final["packer"],
        "reranker": {"status": "succeeded", "fallback_used": False},
    }
    contexts = (
        [
            {
                "chunk_id": evidence_id,
                "text": "evidence text",
                "rendered_text": "[Evidence 1]\nevidence text",
                "source_id": "source-1",
            }
        ]
        if packed
        else []
    )
    return {
        "case_id": "CASE-1",
        "family": "answerable_single_turn",
        "category": "fixture",
        "query": "question",
        "history": [],
        "acquisition_candidate_ids": acquisition,
        "generation_contexts": contexts,
        "retrieval_attempt_traces": attempts,
        "retrieval_trace": final_trace,
        "final_action": "generate",
        "final_reason": "evidence_sufficient",
    }


def test_01_evidence_in_dense_and_bm25() -> None:
    fact = _attempt_fact(_attempt(dense=True, bm25=True), "gold-1")
    assert (fact["dense_rank"], fact["bm25_rank"]) == (1, 1)


def test_02_evidence_only_in_dense() -> None:
    fact = _attempt_fact(_attempt(dense=True, bm25=False), "gold-1")
    assert fact["dense_rank"] == 1 and fact["bm25_rank"] is None


def test_03_evidence_only_in_bm25() -> None:
    fact = _attempt_fact(_attempt(dense=False, bm25=True), "gold-1")
    assert fact["dense_rank"] is None and fact["bm25_rank"] == 1


def test_04_good_rrf_rank_low_rerank_rank() -> None:
    attempt = _attempt(rrf_rank=2, rerank_rank=15, not_examined_reason="item_limit_reached")
    row = trace_requirement(_requirement(), _record([attempt]))
    assert row["primary_loss_stage"] == "ITEM_BUDGET_LOSS"
    assert any(item["factor"] == "RERANK_DEMOTION_OBSERVED" for item in row["contributing_factors"])


def test_05_character_limit() -> None:
    attempt = _attempt(drop_reason="character_limit")
    row = trace_requirement(_requirement(), _record([attempt]))
    assert row["primary_loss_stage"] == "CHARACTER_BUDGET_LOSS"


def test_06_item_limit() -> None:
    attempt = _attempt(not_examined_reason="item_limit_reached")
    row = trace_requirement(_requirement(), _record([attempt]))
    assert row["primary_loss_stage"] == "ITEM_BUDGET_LOSS"


def test_07_duplicate_candidate_identity() -> None:
    attempt = _attempt(duplicate=True, drop_reason="duplicate_id")
    row = trace_requirement(_requirement(), _record([attempt]))
    assert row["primary_loss_stage"] == "CONTRACT_OR_PROVENANCE_REJECTION"
    assert row["primary_reason"] == "duplicate_id"


def test_08_missing_source_identity() -> None:
    attempt = _attempt(source_id=None)
    row = trace_requirement(_requirement(), _record([attempt]))
    assert row["primary_loss_stage"] == "CONTRACT_OR_PROVENANCE_REJECTION"
    assert row["primary_reason"] == "missing_source_identity"


def test_09_two_attempt_retry_retains_union() -> None:
    first = _attempt(attempt_index=1, packed=False)
    second = _attempt(
        attempt_index=2,
        retained=True,
        acquired=False,
        packed=True,
        rerank_rank=1,
    )
    row = trace_requirement(_requirement(), _record([first, second], packed=True))
    assert row["primary_loss_stage"] == "PACKED"
    assert row["retry_statuses"] == ["newly_acquired", "retained"]


def test_10_first_attempt_packed_then_final_pack_omits_without_state_loss() -> None:
    first = _attempt(attempt_index=1, packed=True)
    second = _attempt(
        attempt_index=2,
        retained=True,
        acquired=False,
        packed=False,
        not_examined_reason="item_limit_reached",
    )
    row = trace_requirement(_requirement(), _record([first, second], packed=False))
    assert row["primary_loss_stage"] == "ITEM_BUDGET_LOSS"
    assert row["primary_loss_stage"] != "RETRY_STATE_LOSS"


def test_11_multiple_claims_share_one_chunk() -> None:
    benchmark = {"cases": [{"case_id": "CASE-1"}]}
    gold = {
        "gold_claim_adjudications": [
            {
                "case_id": "CASE-1",
                "claim_id": "C1",
                "current_evidence_ids": ["shared"],
                "review_status": "TECHNICALLY_CLEAR",
                "technical_evidence_status": "DIRECT_SUPPORT",
            },
            {
                "case_id": "CASE-1",
                "claim_id": "C2",
                "current_evidence_ids": ["shared"],
                "review_status": "TECHNICALLY_CLEAR",
                "technical_evidence_status": "DIRECT_SUPPORT",
            },
        ]
    }
    requirements = build_requirements(benchmark, gold)
    assert len(requirements) == 1
    assert requirements[0]["claim_ids"] == ["C1", "C2"]


def test_12_multiple_current_evidence_ids_are_distinct_requirements() -> None:
    benchmark = {"cases": [{"case_id": "CASE-1"}]}
    gold = {
        "gold_claim_adjudications": [
            {
                "case_id": "CASE-1",
                "claim_id": "C1",
                "current_evidence_ids": ["alternative-a", "alternative-b"],
                "review_status": "TECHNICALLY_CLEAR",
                "technical_evidence_status": "DIRECT_SUPPORT",
            }
        ]
    }
    requirements = build_requirements(benchmark, gold)
    assert [item["required_chunk_id"] for item in requirements] == [
        "alternative-a",
        "alternative-b",
    ]


def test_13_missing_trace_field_is_rejected() -> None:
    record = _record([_attempt()])
    record["retrieval_attempt_traces"] = []
    with pytest.raises(ForensicsInputError, match="retrieval_attempt_traces missing"):
        trace_requirement(_requirement(), record)


def test_14_schema_invalid_trace_is_rejected() -> None:
    attempt = _attempt()
    attempt["candidate_trace"]["fused"] = {"not": "a list"}
    with pytest.raises(ForensicsInputError, match="candidate_trace.fused"):
        trace_requirement(_requirement(), _record([attempt]))


def test_15_pending_gold_is_not_treated_as_runtime_loss() -> None:
    stage, _, confidence = classify_requirement(
        reviewed=False,
        acquired=True,
        packed=False,
        channel_hit=True,
        fused_hit=True,
        final_eligible=True,
        final_candidate={"source_id": "source"},
        retrieval_attempt_count=1,
    )
    assert (stage, confidence) == ("GOLD_OR_IDENTITY_MISMATCH", "OBSERVED")


def _negative_record(action: str, reason: str, *, generation_invoked: bool) -> dict:
    return {
        "case_id": "GAP-1",
        "category": "negative",
        "query": "unsupported question",
        "history": [],
        "final_action": action,
        "final_reason": reason,
        "retrieval_attempt_traces": [],
        "generation_contexts": [],
        "structural_verification_report": {"passed": True},
        "generation_invoked": generation_invoked,
        "generation_provider": "fixture",
        "generation_model": "fixture",
        "fallback_applied": False,
        "final_answer": "The evidence is insufficient.",
    }


def test_16_ambiguous_negative_is_not_counted_as_failure() -> None:
    raw = {
        "records": [_negative_record("generate", "evidence_sufficient", generation_invoked=True)]
    }
    gold = {"negative_adjudications": [{"case_id": "GAP-1", "negative_status": "POLICY_AMBIGUOUS"}]}
    result = analyze_gap_failures(raw, gold)
    assert result["confirmed_failures"] == 0
    assert result["policy_ambiguous_total"] == 1


def test_17_evidence_gap_abstain_may_invoke_generation_and_pass() -> None:
    raw = {"records": [_negative_record("abstain", "evidence_gap", generation_invoked=True)]}
    gold = {
        "negative_adjudications": [{"case_id": "GAP-1", "negative_status": "CONFIRMED_ABSENCE"}]
    }
    result = analyze_gap_failures(raw, gold)
    assert result["confirmed_rejections"] == 1
    assert result["confirmed_failures"] == 0


def _identity_fixture(tmp_path: Path) -> tuple[dict, dict]:
    benchmark = {"schema_version": "fixture", "cases": [{"case_id": "CASE-1"}]}
    gold = {
        "identity": {
            "current_kb_build_id": "kb-current",
            "old_benchmark_kb_build_id": "kb-old",
        },
        "gold_claim_adjudications": [],
        "negative_adjudications": [],
        "review_totals": {},
    }
    raw = {
        "schema": "fixture",
        "identity": {
            "system_under_test_sha": "system-sha",
            "run_id": "run-1",
            "pipeline_fingerprint": "pipeline",
            "kb_build_id": "kb-current",
            "benchmark_sha256": canonical_json_sha256(benchmark),
            "gold_canonical_sha256": canonical_json_sha256(gold),
        },
        "records": [{"case_id": "CASE-1", "pipeline_fingerprint": "pipeline"}],
    }
    raw_path = tmp_path / "raw.json"
    gold_path = tmp_path / "gold.json"
    raw_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    gold_path.write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    gold_hash = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    kwargs = {
        "raw_path": raw_path,
        "raw": raw,
        "benchmark": benchmark,
        "benchmark_manifest": {
            "benchmark_sha256": canonical_json_sha256(benchmark),
            "active_kb_build_id": "kb-old",
        },
        "gold_path": gold_path,
        "gold": gold,
        "gold_manifest": {"knowledge_build_id": "kb-current"},
        "run_manifest": {
            "run_id": "run-1",
            "system_under_test_sha": "system-sha",
            "pipeline_fingerprint": "pipeline",
            "kb_build_id": "kb-current",
            "benchmark": {"canonical_sha256": canonical_json_sha256(benchmark)},
            "gold": {
                "canonical_sha256": canonical_json_sha256(gold),
                "raw_sha256": gold_hash,
            },
        },
        "output_manifest": {
            "record_count": 1,
            "raw_results_canonical_sha256": canonical_json_sha256(raw),
            "raw_results_raw_sha256": raw_hash,
            "system_answers_locked": True,
        },
        "reference_metrics": {"source_raw_results_canonical_sha256": canonical_json_sha256(raw)},
        "expected_system_sha": "system-sha",
        "artifact_ref": "fixture",
        "artifact_commit": "fixture",
        "source_inventory": {},
    }
    return kwargs, raw


def test_18_input_hash_mismatch_is_blocking(tmp_path: Path) -> None:
    kwargs, _ = _identity_fixture(tmp_path)
    kwargs["output_manifest"]["raw_results_raw_sha256"] = "bad"
    with pytest.raises(ForensicsInputError, match="raw_bytes_match_output_manifest"):
        validate_identity(**kwargs)


def test_19_baseline_or_kb_identity_mismatch_is_blocking(tmp_path: Path) -> None:
    kwargs, _ = _identity_fixture(tmp_path)
    kwargs["run_manifest"]["kb_build_id"] = "wrong-kb"
    with pytest.raises(ForensicsInputError, match="kb_build_matches_run"):
        validate_identity(**kwargs)


def test_20_repeated_analysis_is_deterministic() -> None:
    attempt = _attempt(drop_reason="character_limit", rrf_rank=4, rerank_rank=6)
    first = trace_requirement(_requirement(), _record([attempt]))
    second = trace_requirement(_requirement(), _record([attempt]))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
