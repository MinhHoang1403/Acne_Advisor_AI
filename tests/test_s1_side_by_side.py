import asyncio
import hashlib
import json
from dataclasses import replace

import pytest

from scripts.evaluate_s1_minimal_rag import (
    CURRENT_SYSTEM_ID,
    MINIMAL_RAG_SYSTEM_ID,
    P45_CASES,
    EvaluationCase,
    SystemRunResult,
    load_frozen_cases,
    retrieval_metrics,
    run_provider_free_current,
    run_provider_free_minimal,
    run_side_by_side,
    write_outputs,
)


def _result(system_id: str, case_id: str, evidence_ids=()) -> SystemRunResult:
    return SystemRunResult(
        system_id=system_id,
        case_id=case_id,
        evidence_ids=tuple(evidence_ids),
        source_ids=tuple(f"source/{value}" for value in evidence_ids),
        answer="",
        citations=(),
        latency_ms={"total": 1.0},
        call_counts={"neo4j": 0},
        status="retrieved",
    )


@pytest.mark.asyncio
async def test_provider_free_harness_uses_same_case_ids_and_is_deterministic():
    cases = load_frozen_cases()[:3]

    first_current, first_minimal = await run_side_by_side(
        cases,
        run_provider_free_current,
        run_provider_free_minimal,
    )
    second_current, second_minimal = await run_side_by_side(
        cases,
        run_provider_free_current,
        run_provider_free_minimal,
    )

    expected_ids = [case.case_id for case in cases]
    assert [result.case_id for result in first_current] == expected_ids
    assert [result.case_id for result in first_minimal] == expected_ids
    assert [result.evidence_ids for result in first_current] == [
        result.evidence_ids for result in second_current
    ]
    assert [result.evidence_ids for result in first_minimal] == [
        result.evidence_ids for result in second_minimal
    ]
    assert all(result.call_counts["neo4j"] == 0 for result in first_minimal)


def test_retrieval_metrics_use_exact_denominators_and_missing_gold_is_na():
    labeled = EvaluationCase(
        case_id="labeled",
        question="q",
        category="test",
        language="VI",
        critical=True,
        positive_evidence_ids=("positive",),
        dataset="test",
        current_fixture={},
    )
    unlabeled = replace(labeled, case_id="unlabeled", positive_evidence_ids=())
    metrics = retrieval_metrics(
        [labeled, unlabeled],
        [_result(MINIMAL_RAG_SYSTEM_ID, "labeled", ("other", "positive")), _result(MINIMAL_RAG_SYSTEM_ID, "unlabeled")],
    )

    assert metrics["recall@1"] == {"numerator": 0, "denominator": 1, "ratio": 0.0}
    assert metrics["recall@3"] == {"numerator": 1, "denominator": 1, "ratio": 1.0}
    assert metrics["mrr"] == 0.5
    assert metrics["mean_positive_rank"] == 2
    assert metrics["labeled_case_denominator"] == 1
    assert retrieval_metrics([unlabeled], [_result(MINIMAL_RAG_SYSTEM_ID, "unlabeled")])["mrr"] == "N/A"


@pytest.mark.asyncio
async def test_harness_isolates_failure_and_timeout_per_system():
    case = load_frozen_cases()[0]

    async def failed(_case):
        raise RuntimeError("private detail")

    async def slow(_case):
        await asyncio.sleep(0.05)
        return _result(MINIMAL_RAG_SYSTEM_ID, case.case_id)

    current, minimal = await run_side_by_side([case], failed, slow, timeout_seconds=0.001)

    assert current[0].status == "failed"
    assert current[0].error == "RuntimeError"
    assert minimal[0].status == "timeout"
    assert minimal[0].error == "TimeoutError"


@pytest.mark.asyncio
async def test_harness_serializes_required_outputs_without_mutating_p45(tmp_path):
    before = hashlib.sha256(P45_CASES.read_bytes()).hexdigest()
    cases = load_frozen_cases()[:2]
    current, minimal = await run_side_by_side(
        cases,
        run_provider_free_current,
        run_provider_free_minimal,
    )

    summary = write_outputs(tmp_path, cases, current, minimal, mode="provider-free")

    required = {
        "s1_cases.json",
        "s1_current_results.json",
        "s1_minimal_results.json",
        "s1_retrieval_metrics.json",
        "s1_complexity_metrics.json",
        "s1_call_counts.json",
    }
    assert required == {path.name for path in tmp_path.iterdir()}
    assert json.loads((tmp_path / "s1_cases.json").read_text(encoding="utf-8"))[0]["case_id"] == cases[0].case_id
    assert summary["case_count"] == 2
    assert hashlib.sha256(P45_CASES.read_bytes()).hexdigest() == before


def test_shared_result_schema_stays_compact():
    result = _result(CURRENT_SYSTEM_ID, "case")
    assert len(result.__dataclass_fields__) == 10


def test_live_outputs_do_not_score_static_fixture_ids_as_corpus_gold(tmp_path):
    case = load_frozen_cases()[0]
    current = [_result(CURRENT_SYSTEM_ID, case.case_id, ("real-qdrant-point",))]
    minimal = [_result(MINIMAL_RAG_SYSTEM_ID, case.case_id, ("another-real-point",))]

    summary = write_outputs(tmp_path, [case], current, minimal, mode="live")

    assert summary["objective_gold_scope"].startswith("N/A")
    assert summary["current"]["retrieval"]["recall@1"].startswith("N/A")
    assert summary["minimal"]["retrieval"]["retrieval_miss_count"].startswith("N/A")
