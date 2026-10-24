from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.formal_evaluation_support import EvaluationBlocked
from evaluation.thesis_evaluation_support import (
    CALIBRATION_APPROVAL_RUN_SHA,
    EXPECTED_BENCHMARK_SHA256,
    EXPECTED_GOLD_CANONICAL_SHA256,
    EXPECTED_KB_BUILD_ID,
    EXPECTED_PIPELINE_FINGERPRINT,
    SYSTEM_UNDER_TEST_SHA,
    _case_deterministic_row,
    _context_records,
    _latency_stats,
    _proportion,
    adapter_contract_sha256,
    raw_sha256,
    validate_benchmark_and_gold,
    validate_checkpoint_prefix,
    validate_complete_raw,
)


def test_locked_identities_and_gold_contract() -> None:
    report = validate_benchmark_and_gold()

    assert SYSTEM_UNDER_TEST_SHA == "366f3ad836c36be36565caad4a94ad17a7a6278f"
    assert EXPECTED_PIPELINE_FINGERPRINT == "e4c420923b79b37c992cf36d"
    assert EXPECTED_KB_BUILD_ID == "d4a1819fe7fb77fe1f40"
    assert report["benchmark_canonical_sha256"] == EXPECTED_BENCHMARK_SHA256
    assert report["gold_canonical_sha256"] == EXPECTED_GOLD_CANONICAL_SHA256
    assert report["reviewed_claims"] == 104
    assert report["confirmed_absence"] == 24
    assert report["policy_ambiguous"] == 6
    assert report["source_canonical_match"] is True
    assert report["source_manifest_canonical_match"] is True
    assert report["benchmark_schema_validation"] == "passed"


def test_calibration_adapter_contract_is_unchanged() -> None:
    assert adapter_contract_sha256() == adapter_contract_sha256(CALIBRATION_APPROVAL_RUN_SHA)


def test_checkpoint_resume_accepts_only_exact_prefix() -> None:
    cases = [{"case_id": "A"}, {"case_id": "B"}, {"case_id": "C"}]
    payload = {
        "identity": {"protocol_commit": "abc"},
        "records": [{"case_id": "A", "error": None}, {"case_id": "B", "error": None}],
    }
    assert validate_checkpoint_prefix(payload, cases, "abc") == 2

    payload["records"][1]["case_id"] = "C"
    with pytest.raises(EvaluationBlocked, match="benchmark-order prefix"):
        validate_checkpoint_prefix(payload, cases, "abc")

    payload["records"] = [{"case_id": "A", "error": {"type": "TimeoutError"}}]
    with pytest.raises(EvaluationBlocked, match="infrastructure-error"):
        validate_checkpoint_prefix(payload, cases, "abc")


def test_complete_raw_requires_all_cases_once_in_order() -> None:
    cases = [{"case_id": f"CASE-{index:03d}"} for index in range(100)]
    raw = {"records": [{"case_id": case["case_id"], "error": None} for case in cases]}
    validate_complete_raw(raw, cases)

    raw["records"][-1]["case_id"] = raw["records"][-2]["case_id"]
    with pytest.raises(EvaluationBlocked, match="records=100"):
        validate_complete_raw(raw, cases)


def test_exact_generation_context_uses_every_packed_item() -> None:
    from src.retrieval.context_packer import pack_context
    from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate

    candidates = [
        RetrievedCandidate(
            candidate_id=f"chunk-{index}",
            collection="acne_knowledge",
            text=f"Evidence text {index}",
            payload={"chunk_id": f"chunk-{index}", "source_id": f"source-{index}"},
            score=1.0 / index,
            rank=index,
        )
        for index in range(1, 10)
    ]
    packed = pack_context(
        NormalizedQuery(original_query="query", normalized_text="query"),
        candidates,
        max_items=9,
        max_chars=7000,
    ).model_dump(mode="json")
    trace = {"packed_evidence": [{"item_id": item["item_id"]} for item in packed["items"]]}

    contexts, context_text, context_hash = _context_records(
        packed, generation_invoked=True, generation_trace=trace
    )

    assert len(contexts) == 9
    assert [item["chunk_id"] for item in contexts] == [f"chunk-{index}" for index in range(1, 10)]
    assert context_text == packed["context_text"]
    assert len(context_hash) == 64


def test_deterministic_evidence_formulas_use_reviewed_ids() -> None:
    case = {"case_id": "A", "family": "answerable_single_turn", "category": "demo"}
    record = {
        "final_action": "generate",
        "final_reason": "evidence_sufficient",
        "acquisition_candidate_ids": ["a", "b", "other"],
        "generation_contexts": [
            {"chunk_id": "a", "text": "A", "source_id": "s1"},
            {"chunk_id": "other", "text": "Other", "source_id": "s2"},
        ],
        "sources": ["s1", "s2"],
        "source_allowlist": [{"source_id": "s1"}, {"source_id": "s2"}],
        "source_validation": {"invalid_source_name_count": 0},
        "structural_verification_report": {"passed": True},
        "retrieval_attempt_count": 1,
        "reranker_events": [],
        "cache": {"hit": False},
        "timings_ms": {"agent_total": 100.0},
        "error": None,
    }
    claims = [
        {"current_evidence_ids": ["a"]},
        {"current_evidence_ids": ["a", "b"]},
    ]

    row = _case_deterministic_row(case, record, claims, None)

    assert row["required_evidence_count"] == 2
    assert row["acquired_required_evidence_count"] == 2
    assert row["packed_required_evidence_count"] == 1
    assert row["complete_reviewed_claim_count"] == 1
    assert row["complete_case"] is False
    assert row["strict_reviewed_id_context_count"] == 1


def test_proportion_and_latency_contracts() -> None:
    assert _proportion(1, 3) == {"count": 1, "denominator": 3, "percentage": 33.3}
    assert _latency_stats([10, 20, 30, 40, 50]) == {
        "count": 5,
        "min_ms": 10.0,
        "median_ms": 30.0,
        "p95_ms": 50.0,
        "max_ms": 50.0,
    }


def test_json_serialization_and_raw_hash(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    payload = {"text": "mụn trứng cá", "values": [1, 2, 3]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert len(raw_sha256(path)) == 64
