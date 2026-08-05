from __future__ import annotations

import json
from pathlib import Path

import pytest

import evaluation.models as models_module
import evaluation.runner as runner_module
from evaluation.checkpoint import append_jsonl, read_jsonl
from evaluation.models import EvaluationConfig
from evaluation.runner import FinalEvaluationRunner


def _cases(count: int) -> list[dict[str, object]]:
    return [
        {
            "id": f"case-{index}",
            "category": "core_knowledge",
            "question": f"Question {index}",
        }
        for index in range(count)
    ]


def _result(case: dict[str, object]) -> dict[str, object]:
    return {
        "case_id": case["id"],
        "category": case["category"],
        "actual_origin": "llm_generated",
        "deterministic_score": 100.0,
        "deterministic_pass": True,
    }


def _judge_row(case: dict[str, object], result: dict[str, object], _config: EvaluationConfig) -> dict[str, object]:
    return {
        "case_id": case["id"],
        "category": case["category"],
        "origin": result["actual_origin"],
        "rubric": "llm_generated_or_cautious_answer",
        "rubric_version": "route_aware_gemini_v3",
        "provider": "gemini",
        "model": "gemini-3.1-flash-lite",
        "retry_count": 0,
        "final_error": None,
        "status": "ok",
        "checkpoint_status": "success",
        "deterministic_score": result["deterministic_score"],
        "deterministic_pass": result["deterministic_pass"],
        "scores": {"vietnamese_naturalness": 5},
        "overall_1_to_5": 5,
        "overall_0_to_100": 100.0,
        "pass": True,
        "short_reason_vi": "Đạt.",
    }


def _config(tmp_path: Path, *, judge_limit: int, judge_model: str = "gemini-3.1-flash-lite", count: int = 5) -> EvaluationConfig:
    report_root = tmp_path / "reports"
    return EvaluationConfig(
        dataset_path=tmp_path / "dataset.jsonl",
        report_root=report_root,
        run_dir=report_root / "official-run",
        question_limit=count,
        judge_limit=judge_limit,
        judge_model=judge_model,
    )


def _prepare_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, count: int = 5) -> tuple[FinalEvaluationRunner, Path]:
    cases = _cases(count)
    monkeypatch.setattr(runner_module, "load_cases", lambda _path: cases)
    monkeypatch.setattr(runner_module, "validate_cases", lambda *_args: None)
    monkeypatch.setattr(runner_module, "sha256_file", lambda _path: "dataset-sha-one")
    runner = FinalEvaluationRunner(_config(tmp_path, judge_limit=2, count=count), tmp_path)
    run_dir, manifest = runner._ensure_run(resume=False)
    manifest["live_stage"] = "live_completed"
    runner._write_manifest(run_dir, manifest)
    for case in cases:
        append_jsonl(run_dir / "case_results.jsonl", _result(case))
    return runner, run_dir


def test_resume_fingerprint_ignores_judge_limit_but_records_invocation_metadata(tmp_path: Path) -> None:
    smoke = _config(tmp_path, judge_limit=2)
    full = _config(tmp_path, judge_limit=5)

    assert smoke.resume_identity_json() == full.resume_identity_json()
    assert smoke.as_json()["judge_limit"] == 2
    assert full.as_json()["judge_limit"] == 5


def test_judge_smoke_then_full_processes_remaining_cases_without_duplicates(monkeypatch, tmp_path) -> None:
    runner, run_dir = _prepare_run(monkeypatch, tmp_path)
    calls: list[str] = []

    def fake_judge(case, result, config):
        calls.append(str(case["id"]))
        return _judge_row(case, result, config)

    monkeypatch.setattr(runner_module, "judge_case", fake_judge)
    runner.run_judge()

    manifest = json.loads((run_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    smoke_summary = json.loads((run_dir / "judge_summary.json").read_text(encoding="utf-8"))
    assert calls == ["case-0", "case-1"]
    assert manifest["judge_stage"] == "judge_in_progress"
    assert smoke_summary["target_total_cases"] == 5
    assert smoke_summary["successful_unique_cases"] == 2
    assert smoke_summary["remaining_case_count"] == 3
    assert manifest["invocations"][-1]["config"]["judge_limit"] == 2

    full_runner = FinalEvaluationRunner(_config(tmp_path, judge_limit=5), tmp_path)
    full_runner.run_judge()
    rows = read_jsonl(run_dir / "judge_results.jsonl")
    manifest = json.loads((run_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))

    assert calls == ["case-0", "case-1", "case-2", "case-3", "case-4"]
    assert len(rows) == len({row["case_id"] for row in rows}) == 5
    assert manifest["judge_stage"] == "judge_completed"
    assert manifest["judge_successful_unique_cases"] == 5
    assert manifest["invocations"][-1]["config"]["judge_limit"] == 5

    full_runner.run_judge()
    assert len(calls) == 5


def test_real_three_to_three_hundred_limits_share_semantic_resume_identity(tmp_path: Path) -> None:
    smoke = _config(tmp_path, judge_limit=3, count=300)
    full = _config(tmp_path, judge_limit=300, count=300)

    assert smoke.resume_identity_json() == full.resume_identity_json()
    assert smoke.judge_limit == 3
    assert full.judge_limit == 300


def test_resume_rejects_dataset_model_and_rubric_changes(monkeypatch, tmp_path) -> None:
    _runner, _run_dir = _prepare_run(monkeypatch, tmp_path)

    monkeypatch.setattr(runner_module, "sha256_file", lambda _path: "dataset-sha-two")
    with pytest.raises(ValueError, match="semantic evaluation identity"):
        FinalEvaluationRunner(_config(tmp_path, judge_limit=5), tmp_path).run_judge()

    monkeypatch.setattr(runner_module, "sha256_file", lambda _path: "dataset-sha-one")
    with pytest.raises(ValueError, match="semantic evaluation identity"):
        FinalEvaluationRunner(_config(tmp_path, judge_limit=5, judge_model="other-model"), tmp_path).run_judge()

    monkeypatch.setattr(models_module, "JUDGE_RUBRIC_VERSION", "route_aware_gemini_v4")
    with pytest.raises(ValueError, match="semantic evaluation identity"):
        FinalEvaluationRunner(_config(tmp_path, judge_limit=5), tmp_path).run_judge()


def test_finalize_rejects_partial_judge_and_explicit_run_dir_isolated(monkeypatch, tmp_path) -> None:
    runner, run_dir = _prepare_run(monkeypatch, tmp_path)
    monkeypatch.setattr(runner_module, "judge_case", _judge_row)
    (tmp_path / "reports" / "old_final_evaluation_v3").mkdir()
    runner.run_judge()

    assert run_dir.name == "official-run"
    with pytest.raises(RuntimeError, match="Finalize requires completed"):
        runner.finalize()


def test_finalize_rejects_unresolved_judge_final_error(monkeypatch, tmp_path) -> None:
    runner, run_dir = _prepare_run(monkeypatch, tmp_path, count=1)

    def final_error(case, _result, _config):
        return {
            "case_id": case["id"],
            "status": "error",
            "checkpoint_status": "final_error",
            "final_error": "RuntimeError: invalid request",
        }

    monkeypatch.setattr(runner_module, "judge_case", final_error)
    runner.run_judge()

    manifest = json.loads((run_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["judge_stage"] == "judge_blocked"
    with pytest.raises(RuntimeError, match="Finalize requires completed"):
        runner.finalize()


def test_explicit_run_dir_rejects_path_outside_report_root(monkeypatch, tmp_path) -> None:
    cases = _cases(1)
    monkeypatch.setattr(runner_module, "load_cases", lambda _path: cases)
    monkeypatch.setattr(runner_module, "validate_cases", lambda *_args: None)
    monkeypatch.setattr(runner_module, "sha256_file", lambda _path: "dataset-sha-one")
    config = EvaluationConfig(
        dataset_path=tmp_path / "dataset.jsonl",
        report_root=tmp_path / "reports",
        run_dir=tmp_path / "outside" / "run",
        question_limit=1,
        judge_limit=1,
    )
    with pytest.raises(ValueError, match="inside report root"):
        FinalEvaluationRunner(config, tmp_path)._ensure_run(resume=False)


def test_transient_error_is_checkpointed_and_can_resume(monkeypatch, tmp_path) -> None:
    runner, run_dir = _prepare_run(monkeypatch, tmp_path, count=1)
    calls: list[str] = []

    def transient(case, _result, _config):
        calls.append(str(case["id"]))
        return {
            "case_id": case["id"],
            "status": "error",
            "checkpoint_status": "transient_error",
            "final_error": "TimeoutError: timed out",
        }

    monkeypatch.setattr(runner_module, "judge_case", transient)
    runner.run_judge()
    manifest = json.loads((run_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["judge_stage"] == "judge_in_progress"
    assert manifest["judge_checkpoint"]["cases"]["case-0"]["status"] == "transient_error"
    assert read_jsonl(run_dir / "judge_results.jsonl") == []

    monkeypatch.setattr(runner_module, "judge_case", _judge_row)
    runner.run_judge()
    assert calls == ["case-0"]
    assert len(read_jsonl(run_dir / "judge_results.jsonl")) == 1
