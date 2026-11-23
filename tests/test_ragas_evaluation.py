from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.ragas_runner import (
    DEFAULT_DATASET_PATH,
    ProjectGeminiEmbeddings,
    export_evaluation_artifacts,
    load_evaluation_dataset,
    render_markdown_report,
    summarize_evaluation,
)


def test_ragas_dataset_schema_is_valid_and_balanced() -> None:
    meta, cases = load_evaluation_dataset(DEFAULT_DATASET_PATH)

    assert meta["dataset_name"] == "acne_advisor_ragas_benchmark"
    assert len(cases) >= 14
    answer_cases = [c for c in cases if c.expected_behavior == "answer"]
    abstain_cases = [c for c in cases if c.expected_behavior == "abstain"]
    assert len(answer_cases) >= 11
    assert len(abstain_cases) >= 3


def test_project_gemini_embeddings_delegates_to_project_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_embed_texts_sync(texts: list[str], **kwargs: Any) -> list[list[float]]:
        captured["texts"] = list(texts)
        captured["kwargs"] = dict(kwargs)
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("src.integrations.google_genai.embed_texts_sync", fake_embed_texts_sync)
    emb = ProjectGeminiEmbeddings(model="models/gemini-embedding-2", dimensions=3)

    single = emb.embed_text("mụn đầu trắng")
    batch = emb.embed_texts(["a", "b"])

    assert single == [0.1, 0.2, 0.3]
    assert batch == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert captured["kwargs"]["expected_dimensions"] == 3
    assert captured["kwargs"]["output_dimensions"] == 3


def test_summarize_and_export_markdown_report(tmp_path: Path) -> None:
    scored_records: list[dict[str, Any]] = [
        {
            "case_id": "SYM-01",
            "category": "symptoms_and_types",
            "expected_behavior": "answer",
            "user_input": "Mụn đầu trắng là gì?",
            "reference": "Mụn nhân đóng do tắc lỗ chân lông.",
            "response": "Mụn đầu trắng là mụn nhân đóng do tắc lỗ chân lông.",
            "retrieved_contexts": ["Mụn đầu trắng là mụn nhân đóng."],
            "sources": ["SRC_demo"],
            "action": "generate",
            "fallback_applied": False,
            "latency_ms": 8500.0,
            "token_usage": {"input": 1000, "output": 100, "total": 1100},
            "error": None,
            "abstention_correct": None,
            "ragas_scores": {
                "faithfulness": 0.95,
                "answer_relevancy": 0.91,
                "context_precision": 0.88,
                "context_recall": 0.90,
                "factual_correctness": 0.86,
            },
            "metric_errors": {},
        },
        {
            "case_id": "ABS-01",
            "category": "safe_abstention",
            "expected_behavior": "abstain",
            "user_input": "Thời tiết hôm nay?",
            "reference": "Từ chối ngoài phạm vi.",
            "response": "Câu hỏi nằm ngoài phạm vi mụn và chăm sóc da.",
            "retrieved_contexts": [],
            "sources": [],
            "action": "abstain",
            "fallback_applied": True,
            "latency_ms": 4200.0,
            "token_usage": {},
            "error": None,
            "abstention_correct": True,
            "ragas_scores": {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
                "factual_correctness": None,
            },
            "metric_errors": {},
        },
    ]

    summary = summarize_evaluation(
        scored_records,
        dataset_meta={"dataset_name": "unit_test_suite"},
        kb_build_id="b768fae999391f47760a",
    )
    assert summary["ragas_summary"]["faithfulness"]["mean"] == 0.95
    assert summary["safe_abstention"] == {"correct": 1, "total": 1, "accuracy": 1.0}
    assert summary["token_usage"] == {"input": 1000, "output": 100, "total": 1100}

    md = render_markdown_report(summary)
    assert "# Báo cáo Đánh giá Hệ thống Acne Advisor AI (RAGAS)" in md
    assert "SYM-01" in md
    assert "ABS-01" in md
    assert "0.950" in md

    report_path, raw_path = export_evaluation_artifacts(summary, output_dir=tmp_path)
    assert report_path.exists()
    assert raw_path.exists()
    parsed = json.loads(raw_path.read_text(encoding="utf-8"))
    assert parsed["kb_build_id"] == "b768fae999391f47760a"
