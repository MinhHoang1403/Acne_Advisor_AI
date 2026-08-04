from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.render_generation_focused_report import (
    REQUIRED_REPORT_HEADINGS,
    render_generation_focused_report,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_render_generation_focused_report_writes_required_vietnamese_sections(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "results.csv",
        [
            {
                "case_id": "generation-001",
                "category": "core_knowledge_generation",
                "question": "Mụn trứng cá là gì?",
                "overall_score": 92,
                "overall_pass": True,
                "failure_reasons": "",
            }
        ],
    )
    (tmp_path / "raw_responses.jsonl").write_text(
        json.dumps(
            {
                "runtime_verification": "ollama_llm",
                "raw_response": {"metadata": {"provider": "ollama", "response_origin": "llm"}},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        tmp_path / "judge_results.csv",
        [
            {
                "category": "core_knowledge_generation",
                "judge_status": "ok",
                "judge_pass": True,
                "judge_score_100": 88,
            }
        ],
    )
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "timestamp": "20260731T000000Z",
                "live_eval_provider": "ollama",
                "live_eval_model": "qwen3:8b",
                "cache_bypass_for_live_eval": True,
                "live_eval_max_runtime_attempts": 3,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary_metrics.json").write_text(json.dumps({"overall_score": 92.0}), encoding="utf-8")
    (tmp_path / "judge_summary.json").write_text(
        json.dumps(
            {
                "judge_provider": "gemini",
                "judge_model": "gemini-3.1-flash-lite",
                "judge_successful_cases": 1,
                "judge_error_cases": 0,
                "judge_avg_score": 88.0,
                "judge_pass_rate": 100.0,
                "judge_agreement_rate": 100.0,
            }
        ),
        encoding="utf-8",
    )

    report_path = render_generation_focused_report(tmp_path)
    report = report_path.read_text(encoding="utf-8")
    assert all(heading in report for heading in REQUIRED_REPORT_HEADINGS)
    assert "acne_rag_eval_generation_focused.jsonl" in report
    assert (tmp_path / "plots" / "judge_pass_rate_by_category.png").exists()
    assert (tmp_path / "plots" / "origin_distribution.png").exists()
