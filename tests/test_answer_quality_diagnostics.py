from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from scripts.answer_quality_diagnostics import assess_retrieval_coverage, load_diagnostic_cases


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


def test_retrieval_coverage_classifies_candidate_and_context_loss_without_semantic_claims() -> None:
    cases = {case["case_id"]: case for case in load_diagnostic_cases(CASES_PATH)}
    mask_case = cases["historical_16"]
    comparison_case = cases["adapalene_bp_comparison"]

    retrieval_miss = assess_retrieval_coverage(
        mask_case,
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
