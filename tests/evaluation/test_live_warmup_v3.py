from __future__ import annotations

import evaluation.runner as runner_module
from evaluation.models import EvaluationConfig
from evaluation.runner import FinalEvaluationRunner


class _FakeReranker:
    available = True

    def __init__(self) -> None:
        self.warmup_calls = 0

    def warmup(self) -> None:
        self.warmup_calls += 1


def test_live_runner_warms_the_semantic_reranker_before_cases(monkeypatch, tmp_path) -> None:
    fake = _FakeReranker()
    config = EvaluationConfig(dataset_path=tmp_path / "data.jsonl", report_root=tmp_path)
    runner = object.__new__(FinalEvaluationRunner)
    runner.config = config
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("RERANK_PROVIDER", "hybrid")
    monkeypatch.setattr(runner_module, "get_cached_semantic_reranker_from_env", lambda: fake)

    result = runner._prewarm_semantic_reranker()

    assert result["status"] == "ready"
    assert result["requested_provider"] == "hybrid"
    assert result["duration_ms"] >= 0
    assert fake.warmup_calls == 1


def test_live_runner_skips_warmup_for_local_rules(monkeypatch, tmp_path) -> None:
    config = EvaluationConfig(dataset_path=tmp_path / "data.jsonl", report_root=tmp_path)
    runner = object.__new__(FinalEvaluationRunner)
    runner.config = config
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("RERANK_PROVIDER", "local_rules")

    result = runner._prewarm_semantic_reranker()

    assert result == {
        "enabled": True,
        "requested_provider": "local_rules",
        "status": "skipped",
    }
