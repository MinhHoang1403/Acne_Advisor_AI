from __future__ import annotations

import json

from evaluation.models import EvaluationFailure, FailureCategory, FailureSeverity, ReleaseStatus
from evaluation.release_contract import (
    aggregate_severity,
    build_medical_release_contract,
    failure_records_for_row,
    summarize_case_failures,
)


_CRITICAL_GATES = {
    "emergency_detection": True,
    "emergency_first_action": True,
    "pregnancy_safety": True,
    "antibiotic_stewardship": True,
    "critical_forbidden_claims": True,
    "critical_unsupported_claims": True,
    "invalid_source_names": True,
}


def _record(
    severity: FailureSeverity,
    *,
    metric: str = "fixture",
    critical: bool = False,
) -> dict[str, object]:
    return EvaluationFailure(
        metric=metric,
        category=FailureCategory.OTHER,
        severity=severity,
        critical_failure=critical,
        message="fixture finding",
        evidence={"fixture": True},
    ).as_json()


def _contract(
    rows: list[dict[str, object]],
    *,
    quality_passed: bool = True,
    judge_average: float | None = None,
    deterministic_pass_rate: float | None = None,
    hard_gate_overrides: dict[str, bool] | None = None,
) -> dict[str, object]:
    hard_gates = {**_CRITICAL_GATES, **(hard_gate_overrides or {})}
    return build_medical_release_contract(
        severity_summary=aggregate_severity(rows),
        base_hard_gates=hard_gates,
        quality_targets={"fixture_quality": {"passed": quality_passed}},
        judge_average=judge_average,
        deterministic_pass_rate=deterministic_pass_rate,
    )


def test_s4_failure_causes_release_fail() -> None:
    contract = _contract([{"case_id": "case-s4", "failure_records": [_record(FailureSeverity.S4, critical=True)]}])

    assert contract["release_status"] == ReleaseStatus.FAIL.value
    assert contract["hard_gates"]["S4_zero"] is False


def test_s3_failure_causes_release_fail() -> None:
    contract = _contract([{"case_id": "case-s3", "failure_records": [_record(FailureSeverity.S3, critical=True)]}])

    assert contract["release_status"] == ReleaseStatus.FAIL.value
    assert contract["hard_gates"]["S3_zero"] is False


def test_high_judge_score_cannot_override_s4() -> None:
    contract = _contract(
        [{"case_id": "case-s4", "failure_records": [_record(FailureSeverity.S4, critical=True)]}],
        judge_average=99.9,
    )

    assert contract["release_status"] == ReleaseStatus.FAIL.value
    assert contract["aggregate_scores_recorded_only"]["used_for_hard_gate_decision"] is False


def test_high_deterministic_score_cannot_override_s3() -> None:
    contract = _contract(
        [{"case_id": "case-s3", "failure_records": [_record(FailureSeverity.S3, critical=True)]}],
        deterministic_pass_rate=99.9,
    )

    assert contract["release_status"] == ReleaseStatus.FAIL.value
    assert contract["aggregate_scores_recorded_only"]["deterministic_pass_rate"] == 99.9


def test_s0_only_defect_is_not_critical_medical_failure() -> None:
    summary = summarize_case_failures(
        [
            EvaluationFailure(
                metric="format",
                category=FailureCategory.FORMAT,
                severity=FailureSeverity.S0,
                critical_failure=False,
                message="format",
            )
        ]
    )

    assert summary["highest_severity"] == "S0"
    assert summary["critical_failure"] is False


def test_s1_only_defect_is_not_critical_medical_failure() -> None:
    summary = summarize_case_failures(
        [
            EvaluationFailure(
                metric="instruction",
                category=FailureCategory.INSTRUCTION,
                severity=FailureSeverity.S1,
                critical_failure=False,
                message="instruction",
            )
        ]
    )

    assert summary["highest_severity"] == "S1"
    assert summary["critical_failure"] is False


def test_s2_is_material_quality_debt_not_an_unsafe_advice_gate() -> None:
    contract = _contract([{"case_id": "case-s2", "failure_records": [_record(FailureSeverity.S2)]}], quality_passed=False)

    assert contract["hard_gates"]["S3_zero"] is True
    assert contract["hard_gates"]["S4_zero"] is True
    assert contract["release_status"] == ReleaseStatus.PASS_WITH_QUALITY_LIMITATIONS.value


def test_emergency_miss_maps_to_s4_and_hard_gate_failure() -> None:
    records = failure_records_for_row({"case_id": "emergency", "failure_reasons": ["emergency_action"]})
    contract = _contract([{"case_id": "emergency", "failure_records": [record.as_json() for record in records]}])

    assert records[0].severity is FailureSeverity.S4
    assert records[0].category is FailureCategory.EMERGENCY
    assert contract["hard_gates"]["S4_zero"] is False


def test_critical_pregnancy_violation_maps_to_s4_and_hard_gate_failure() -> None:
    records = failure_records_for_row({"case_id": "pregnancy", "failure_reasons": ["pregnancy_safety"]})
    contract = _contract([{"case_id": "pregnancy", "failure_records": [record.as_json() for record in records]}])

    assert records[0].severity is FailureSeverity.S4
    assert records[0].category is FailureCategory.PREGNANCY_LACTATION
    assert contract["hard_gates"]["S4_zero"] is False


def test_critical_antibiotic_violation_maps_to_s3_and_hard_gate_failure() -> None:
    records = failure_records_for_row({"case_id": "antibiotic", "failure_reasons": ["antibiotic_stewardship"]})
    contract = _contract([{"case_id": "antibiotic", "failure_records": [record.as_json() for record in records]}])

    assert records[0].severity is FailureSeverity.S3
    assert records[0].category is FailureCategory.ANTIBIOTIC_STEWARDSHIP
    assert contract["hard_gates"]["S3_zero"] is False


def test_unsupported_critical_medical_claim_causes_hard_gate_failure() -> None:
    records = failure_records_for_row(
        {"case_id": "unsupported", "critical_case": True, "failure_reasons": ["unsupported_assumption"]}
    )
    contract = _contract(
        [{"case_id": "unsupported", "failure_records": [record.as_json() for record in records]}],
        quality_passed=True,
        hard_gate_overrides={"critical_unsupported_claims": False},
    )

    assert records[0].severity is FailureSeverity.S4
    assert records[0].critical_failure is True
    assert contract["hard_gates"]["critical_unsupported_claims"] is False
    assert contract["hard_gates"]["S4_zero"] is False


def test_no_failures_and_passing_contracts_allow_pass() -> None:
    contract = _contract([])

    assert contract["release_status"] == ReleaseStatus.PASS.value
    assert contract["hard_gates_passed"] is True


def test_quality_failure_with_hard_gates_passing_is_pass_with_quality_limitations() -> None:
    contract = _contract([], quality_passed=False)

    assert contract["hard_gates_passed"] is True
    assert contract["release_status"] == ReleaseStatus.PASS_WITH_QUALITY_LIMITATIONS.value


def test_failure_count_and_affected_case_count_are_distinct() -> None:
    summary = aggregate_severity(
        [
            {"case_id": "one", "failure_records": [_record(FailureSeverity.S2), _record(FailureSeverity.S2, metric="second")]},
            {"case_id": "two", "failure_records": [_record(FailureSeverity.S2)]},
        ]
    )

    assert summary["failure_counts"]["S2"] == 3
    assert summary["cases_affected"]["S2"] == 2


def test_severity_serialization_is_stable_and_machine_readable() -> None:
    record = EvaluationFailure(
        metric="format",
        category=FailureCategory.FORMAT,
        severity=FailureSeverity.S0,
        critical_failure=False,
        message="minor Markdown defect",
        evidence={"line": 1},
    )

    payload = record.as_json()
    assert json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True)) == payload
    assert payload["severity"] == "S0"
    assert payload["category"] == "FORMAT"
