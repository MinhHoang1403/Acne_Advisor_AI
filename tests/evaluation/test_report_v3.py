from __future__ import annotations

from pathlib import Path

from evaluation.report_vi import render_report


def test_report_v3_uses_single_vietnamese_main_report(tmp_path: Path) -> None:
    metrics = {
        "hard_gates_passed": True,
        "reliability": {"request_success_rate": {"value": 100, "denominator": 300}, "provider_provenance_rate": {"value": 100, "denominator": 300}},
        "retrieval_and_grounding": {key: {"value": 100, "denominator": 1} for key in ("source_hit_rate", "source_traceability_validity", "entity_hit_rate", "alias_resolution_accuracy")},
        "answer_quality": {"concept_recall": {"value": 100}, "entity_preservation": {"value": 100}, "polarity_accuracy": {"value": 100}, "comparison_completeness": {"value": 100}, "forbidden_claim_count": 0, "source_requirement_pass": {"value": 100}},
        "safety_and_scope": {key: {"value": 100, "denominator": 1} for key in ("emergency_first_action_accuracy", "pregnancy_safety_pass", "antibiotic_stewardship_pass", "ood_precision", "ood_recall")} | {"false_emergency_escalation_rate": {"value": 0, "denominator": 1}},
        "instruction_format_conversation": {key: {"value": 100, "denominator": 1} for key in ("format_pass_rate", "exact_count_pass", "multi_turn_context_accuracy")},
        "naturalness_user_experience": {key: {"value": 0 if key != "markdown_readability_rate" else 100, "denominator": 1} for key in ("repeated_disclaimer_rate", "internal_error_leakage_rate", "judgmental_wording_rate", "markdown_readability_rate")},
        "performance": {"average_latency_ms": 1, "p50_latency_ms": 1, "p95_latency_ms": 1, "p99_latency_ms": 1},
    }
    path = render_report(
        tmp_path,
        {"dataset_sha256": "abc", "metrics_version": "evaluation_metrics_v4", "judge_rubric_version": "route_aware_gemini_v3", "live_provider": "ollama", "live_model": "qwen3:8b", "judge_provider": "gemini", "judge_model": "gemini-3.1-flash-lite"},
        metrics,
        {"judge_final_errors": 0, "average_score_1_to_5": 4.5, "judge_success_rate": 100, "agreement_rate": 100},
        [],
        [],
    )

    assert path.name == "BAO_CAO_DANH_GIA_HE_THONG_V3.md"
    assert "## 11. Kết luận" in path.read_text(encoding="utf-8")
