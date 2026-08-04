from __future__ import annotations

from pathlib import Path

from src.evaluation.comprehensive import (
    ComprehensiveRunner,
    EvaluationConfig,
    acquire_run_lock,
    create_plots,
    deterministic_result,
    judge_prompt,
    origin_from_response,
    release_run_lock,
    render_report,
    summarize_metrics,
)


def _config(tmp_path: Path) -> EvaluationConfig:
    return EvaluationConfig(
        dataset_path=Path("notebooks/eval_data/acne_rag_eval_comprehensive_v1.jsonl"),
        report_root=tmp_path,
        run_live=True,
        run_judge=True,
        bypass_cache=True,
    )


def _case(**overrides):
    case = {
        "id": "case-1",
        "category": "comparison",
        "question": "Có nên dùng clindamycin đơn độc không?",
        "expected_route": "llm_generated",
        "expected_safety_level": "normal",
        "expected_concepts": ["clindamycin", "không nên"],
        "expected_entities": ["clindamycin"],
        "forbidden_concepts": [],
        "accepted_sources": [],
        "source_required": False,
        "format_contract": {"type": "short_answer"},
        "critical_case": False,
    }
    case.update(overrides)
    return case


def _raw(answer: str, *, origin: str = "llm", provider: str = "ollama"):
    return {
        "ok": True,
        "http_status": 200,
        "latency_ms": 12.0,
        "runtime_attempts": 1,
        "raw_response": {
            "answer": answer,
            "sources": ["web_raw_dataset.json"],
            "metadata": {
                "provider": provider,
                "model": "qwen3:8b" if provider == "ollama" else None,
                "requested_provider": "ollama",
                "requested_model": "qwen3:8b",
                "response_origin": origin,
                "fallback_applied": origin == "safe_fallback",
                "cache": {"hit": False},
            },
        },
    }


def test_origin_mapping_keeps_llm_fallback_and_guardrail_separate() -> None:
    payload = _raw("a")["raw_response"]
    payload["metadata"]["guardrail"] = "in_domain_rule"
    assert origin_from_response(payload, "ollama") == "llm_generated"
    assert origin_from_response(_raw("a", origin="safe_fallback", provider="system")["raw_response"], "ollama") == "system_safe_fallback"
    payload = _raw("a", origin="guardrail", provider="system")["raw_response"]
    payload["metadata"]["guardrail_applied"] = True
    assert origin_from_response(payload, "ollama") == "guardrail"


def test_deterministic_result_checks_direct_negative_polarity() -> None:
    result = deterministic_result(_raw("Không. Clindamycin không nên được dùng đơn độc."), _case(), _config(Path(".")))
    assert result["polarity_pass"] is True
    assert result["provider_provenance_ok"] is True
    assert result["actual_origin"] == "llm_generated"
    assert result["graph_enrichment"] == []
    assert result["safety_flags"] == []


def test_route_aware_judge_prompt_changes_for_safe_fallback() -> None:
    result = deterministic_result(_raw("Tôi cần thêm thông tin.", origin="safe_fallback", provider="system"), _case(expected_route="system_safe_fallback"), _config(Path(".")))
    prompt = judge_prompt(_case(expected_route="system_safe_fallback"), result)
    assert "Không phạt chỉ vì fallback ngắn" in prompt
    assert "ROUTE: system_safe_fallback" in prompt


def test_metrics_and_report_artifacts_are_rendered(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = deterministic_result(_raw("Không. Clindamycin không nên dùng đơn độc và cần hỏi bác sĩ."), _case(), config)
    judge = {
        "case_id": "case-1", "category": "comparison", "origin": "llm_generated", "status": "ok",
        "overall_score": 82.0, "pass": True, "retry_count": 0, "deterministic_score": result["concept_recall"],
    }
    component = {"backend_health": {"passed": True}}
    metrics = summarize_metrics([result], [judge], component)
    assert metrics["macro"]["request_success_rate"]["value"] == 100.0
    assert metrics["safety"]["critical_forbidden_claim_rate"]["value"] is None
    plots = create_plots(tmp_path, metrics, [result], [judge])
    manifest = {
        "live_model": "qwen3:8b", "judge_model": "gemini-3.1-flash-lite", "dataset_sha256": "abc",
        "dataset_schema_version": "comprehensive_eval_v1", "runtime_config": {"bypass_cache": True},
    }
    render_report(tmp_path, manifest, metrics, [result], [judge], plots)
    assert (tmp_path / "BAO_CAO_DANH_GIA_HE_THONG.md").exists()
    assert len(list((tmp_path / "plots").glob("*.png"))) == 9


def test_smoke_selection_is_stratified_across_all_categories(tmp_path: Path) -> None:
    config = EvaluationConfig(
        dataset_path=Path("notebooks/eval_data/acne_rag_eval_comprehensive_v1.jsonl"),
        report_root=tmp_path,
        question_limit=15,
        smoke=True,
        run_live=True,
        run_judge=True,
    )
    runner = ComprehensiveRunner(config, Path("."))
    assert len(runner.cases) == 15
    assert len({case["category"] for case in runner.cases}) == 15


def test_run_lock_rejects_a_second_active_writer(tmp_path: Path) -> None:
    lock_path = acquire_run_lock(tmp_path)
    try:
        try:
            acquire_run_lock(tmp_path)
        except RuntimeError as exc:
            assert "already active" in str(exc)
        else:
            raise AssertionError("Expected an active run lock to reject a second writer.")
    finally:
        release_run_lock(lock_path)
    assert not lock_path.exists()
