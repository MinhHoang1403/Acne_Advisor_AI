from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
import sys

from scripts.answer_quality_diagnostics import (
    _channel_trace,
    _fused_candidate_trace,
    assess_retrieval_coverage,
    load_diagnostic_cases,
    run_live_retrieval_diagnostic,
)
from src.retrieval.context_packer import pack_context
from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate


CASES_PATH = Path(__file__).parent / "fixtures" / "answer_quality_diagnostic_cases.json"


def test_diagnostic_cases_have_traceable_evidence_and_required_case_groups() -> None:
    cases = load_diagnostic_cases(CASES_PATH)
    by_id = {case["case_id"]: case for case in cases}

    assert {
        "bp_antibiotic",
        "adapalene_class",
        "clindamycin_monotherapy",
        "adapalene_bp_comparison",
        "retinoid_pregnancy",
        "dermatologist_referral",
        "isotretinoin_pregnancy",
        "historical_02",
        "historical_03",
        "historical_04",
        "historical_16",
        "conversation_adapalene_pregnancy",
        "conversation_entity_reference",
        "safety_cross_subject",
        "safety_same_subject",
        "safety_resolved_history",
    } <= by_id.keys()

    for case in cases:
        assert case["expected_behavior"]
        assert case["assessment_mode"]
        if case["assessment_mode"] == "retrieval_evidence":
            assert case["evidence_groups"]
            assert all(group["any_of"] for group in case["evidence_groups"])

    historical_mask = by_id["historical_16"]
    assert historical_mask["assessment_mode"] == "source_scope_review"
    assert "not masks or face coverings" in historical_mask["review_notes"]


def test_retrieval_coverage_classifies_candidate_and_context_loss_without_semantic_claims() -> None:
    cases = {case["case_id"]: case for case in load_diagnostic_cases(CASES_PATH)}
    retinoid_case = cases["retinoid_pregnancy"]
    comparison_case = cases["adapalene_bp_comparison"]

    retrieval_miss = assess_retrieval_coverage(
        retinoid_case,
        candidate_ids=["unrelated-chunk"],
        packed_ids=["unrelated-chunk"],
    )
    assert retrieval_miss["classification"] == "retrieval_miss"
    assert retrieval_miss["semantic_truth_checked"] is False

    context_loss = assess_retrieval_coverage(
        comparison_case,
        candidate_ids=["460222d6-c4ba-56f9-a26c-bf255b6afb39", "db4632c5-44ed-5dea-80c2-addb0a2534b3", "66d47093-ac23-5331-8955-8867287370cb"],
        packed_ids=["460222d6-c4ba-56f9-a26c-bf255b6afb39", "66d47093-ac23-5331-8955-8867287370cb"],
    )
    assert context_loss["classification"] == "context_missing_required_fact"

    evidence_packed = assess_retrieval_coverage(
        comparison_case,
        candidate_ids=["460222d6-c4ba-56f9-a26c-bf255b6afb39", "db4632c5-44ed-5dea-80c2-addb0a2534b3", "66d47093-ac23-5331-8955-8867287370cb"],
        packed_ids=["460222d6-c4ba-56f9-a26c-bf255b6afb39", "db4632c5-44ed-5dea-80c2-addb0a2534b3", "66d47093-ac23-5331-8955-8867287370cb"],
    )
    assert evidence_packed["classification"] == "evidence_packed"


def test_fused_candidate_trace_reports_packer_budget_without_reselecting() -> None:
    candidates = [
        RetrievedCandidate(
            candidate_id="selected",
            collection="knowledge",
            text="Selected evidence.",
            payload={"source_id": "source-a"},
            rank=1,
            debug={"dense_rank": 1, "bm25_rank": None},
        ),
        RetrievedCandidate(
            candidate_id="dropped",
            collection="knowledge",
            text="Later evidence.",
            payload={"source_id": "source-b"},
            rank=2,
            debug={"dense_rank": 2, "bm25_rank": None},
        ),
    ]
    packed = pack_context(
        NormalizedQuery(original_query="question", normalized_text="question"),
        candidates,
        max_items=1,
        max_chars=1000,
    )

    trace = _fused_candidate_trace(candidates, packed)

    assert trace[0]["packed"] is True
    assert trace[0]["drop_reason"] is None
    assert trace[1]["packed"] is False
    assert trace[1]["drop_reason"] == "item_limit"
    assert trace[1]["cumulative_chars_before"] > 0
    assert trace[1]["remaining_chars_before"] < 1000


def test_channel_trace_preserves_rank_and_native_score() -> None:
    trace = _channel_trace(
        [
            {"id": "first", "score": 0.9},
            {"chunk_id": "second", "score": 0.5},
        ]
    )

    assert trace == [
        {"candidate_id": "first", "rank": 1, "native_score": 0.9},
        {"candidate_id": "second", "rank": 2, "native_score": 0.5},
    ]


def test_source_scope_review_does_not_invoke_retrieval(monkeypatch) -> None:
    class _NoRetrieval:
        called = False

        async def close(self) -> None:
            return None

    cases = {case["case_id"]: case for case in load_diagnostic_cases(CASES_PATH)}
    retriever = _NoRetrieval()
    monkeypatch.setattr(
        "scripts.answer_quality_diagnostics.EvidenceRetriever",
        lambda: retriever,
    )

    observations = asyncio.run(run_live_retrieval_diagnostic([cases["historical_16"]]))

    assert retriever.called is False
    assert observations == [
        {
            "case_id": "historical_16",
            "classification": "not_run",
            "reason": "source_scope_review does not treat analogous evidence as direct retrieval support.",
            "semantic_truth_checked": False,
        }
    ]


def test_diagnostic_cli_runs_directly_from_repository_root() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/answer_quality_diagnostics.py", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Read-only source-grounded retrieval diagnostic" in completed.stdout
