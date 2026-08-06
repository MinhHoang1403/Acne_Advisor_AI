from __future__ import annotations

import pytest

from evaluation.checkpoint import assert_resume_compatible
from evaluation.live_eval import run_live_case, run_live_case_async
from evaluation.models import EvaluationConfig


def test_resume_rejects_different_dataset_or_model() -> None:
    manifest = {
        "dataset_sha256": "one",
        "live_provider": "ollama",
        "live_model": "qwen3:8b",
        "metrics_version": "evaluation_metrics_v4",
    }
    with pytest.raises(ValueError, match="dataset_sha256"):
        assert_resume_compatible(
            manifest,
            dataset_sha256="two",
            provider="ollama",
            model="qwen3:8b",
            version="evaluation_metrics_v4",
            stage="live",
        )


def test_live_runner_uses_internal_evaluation_mode_and_no_persistence(monkeypatch, tmp_path) -> None:
    received = {}
    cleanup_calls = 0

    async def fake_agent(**kwargs):
        received.update(kwargs)
        return {
            "answer": "Benzoyl peroxide là hoạt chất bôi trị mụn.",
            "actual_provider": "ollama",
            "actual_model": "qwen3:8b",
            "sources": [],
        }

    async def fake_cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    monkeypatch.setattr("evaluation.live_eval.run_clinical_agent", fake_agent)
    monkeypatch.setattr("evaluation.live_eval.close_evaluation_runtime_clients", fake_cleanup)
    row = run_live_case(
        {"id": "v3_case", "category": "active_ingredients", "question": "BPO là gì?", "conversation_history": []},
        EvaluationConfig(dataset_path=tmp_path / "data.jsonl", report_root=tmp_path),
    )

    assert row["ok"] is True
    assert row["persistence_enabled"] is False
    assert row["cache_read_enabled"] is False
    assert row["cache_write_enabled"] is False
    assert received["evaluation_mode"] is True
    assert received["bypass_cache"] is True
    assert cleanup_calls == 1


@pytest.mark.asyncio
async def test_live_case_async_uses_the_callers_event_loop(monkeypatch, tmp_path) -> None:
    received = {}

    async def fake_agent(**kwargs):
        received.update(kwargs)
        return {"answer": "Benzoyl peroxide là hoạt chất bôi trị mụn."}

    monkeypatch.setattr("evaluation.live_eval.run_clinical_agent", fake_agent)
    row = await run_live_case_async(
        {"id": "v3_case_async", "category": "active_ingredients", "question": "BPO là gì?", "conversation_history": []},
        EvaluationConfig(dataset_path=tmp_path / "data.jsonl", report_root=tmp_path),
    )

    assert row["ok"] is True
    assert received["evaluation_mode"] is True
