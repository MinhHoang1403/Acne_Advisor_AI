"""Regression coverage for the locked, provider-free Retrieval V5 release gate."""

from __future__ import annotations

from scripts.eval_retrieval_v5_release import run_locked_dual_run


def test_locked_dual_run_preserves_source_and_critical_evidence() -> None:
    summary = run_locked_dual_run()

    assert summary["release_decision"] == "V5_RELEASE_READY"
    assert summary["passed"] is True
    assert summary["case_count"] >= 16
    assert all(summary["medical_gates"].values())
    assert all(summary["performance_gates"].values())
    for name, metric in summary["metrics"].items():
        assert metric["denominator"] > 0, name
        assert metric["passed"] is True, name
        assert "v4_reference" in metric, name
        assert "v5_candidate" in metric, name
    assert summary["latency_ms"]["cold"]["total_pre_generation"] >= 0
    assert summary["latency_ms"]["warm"]["total_pre_generation"] >= 0
    assert summary["latency_ms"]["cold"]["v5_total_pre_generation"] >= 0
    assert summary["latency_ms"]["warm"]["v4_total_pre_generation"] >= 0


def test_locked_dual_run_keeps_structural_signals_out_of_source_evidence() -> None:
    summary = run_locked_dual_run()
    record = next(item for item in summary["cases"] if item["id"] == "pregnancy_adapalene")

    assert record["entity_signals"]
    assert record["graph_signals"]
    assert all(item_id.endswith((":primary", ":support")) for item_id in record["selected_evidence"]["ids"])
    assert record["expected"]["critical_source_id"] in record["packed_evidence"]["critical_ids"]
