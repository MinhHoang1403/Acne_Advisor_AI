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
    assert len(cases) >= 50
    answer_cases = [c for c in cases if c.expected_behavior == "answer"]
    abstain_cases = [c for c in cases if c.expected_behavior == "abstain"]
    assert len(answer_cases) >= 40
    assert len(abstain_cases) >= 8


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
                "answer_correctness": 0.86,
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
                "answer_correctness": None,
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
    assert "Answer Correctness" in md
    assert "SYM-01" in md
    assert "ABS-01" in md
    assert "0.950" in md

    report_path, raw_path = export_evaluation_artifacts(summary, output_dir=tmp_path)
    assert report_path.exists()
    assert raw_path.exists()
    parsed = json.loads(raw_path.read_text(encoding="utf-8"))
    assert parsed["kb_build_id"] == "b768fae999391f47760a"


def test_strip_presentation_disclaimer_removes_canonical_and_legacy() -> None:
    from evaluation.ragas_runner import _strip_presentation_disclaimer
    from src.agent.answer_formatting import CANONICAL_DISCLAIMER

    body = "Mụn đầu trắng là mụn nhân đóng."
    with_disclaimer = f"{body}\n\n{CANONICAL_DISCLAIMER}"
    assert _strip_presentation_disclaimer(with_disclaimer) == body


def test_prioritize_supporting_contexts_keeps_direct_evidence_in_order() -> None:
    from src.agent.nodes.reason import _prioritize_supporting_contexts

    contexts = [
        {"id": "c-noise", "text": "Đoạn nhiễu", "source_id": "SRC_1"},
        {"id": "c-2", "text": "Bằng chứng 2", "source_id": "SRC_2"},
        {"id": "c-1", "text": "Bằng chứng 1", "source_id": "SRC_3"},
    ]
    packed = {
        "context_text": "old",
        "chunk_items_count": 3,
        "items": [
            {"item_id": "c-noise", "text": "Đoạn nhiễu", "payload": {"source_id": "SRC_1", "chunk_id": "c-noise"}},
            {"item_id": "c-2", "text": "Bằng chứng 2", "payload": {"source_id": "SRC_2", "chunk_id": "c-2"}},
            {"item_id": "c-1", "text": "Bằng chứng 1", "payload": {"source_id": "SRC_3", "chunk_id": "c-1"}},
        ],
    }

    narrowed_contexts, narrowed_packed = _prioritize_supporting_contexts(contexts, packed, ["c-1", "c-2"])
    assert [c["id"] for c in narrowed_contexts] == ["c-1", "c-2"]
    assert [i["item_id"] for i in narrowed_packed["items"]] == ["c-1", "c-2"]
    assert narrowed_packed["chunk_items_count"] == 2
    assert "Đoạn nhiễu" not in narrowed_packed["context_text"]
    assert narrowed_packed["context_text"].index("Bằng chứng 1") < narrowed_packed["context_text"].index("Bằng chứng 2")


def test_remove_greetings_capitalizes_remaining_vietnamese_sentence() -> None:
    from src.agent.answer_formatting import _remove_greetings

    assert _remove_greetings("Chào bạn, tôi rất hiểu lo lắng của bạn.") == "Tôi rất hiểu lo lắng của bạn."
    assert _remove_greetings("Xin chào! tình trạng mụn cần theo dõi.") == "Tình trạng mụn cần theo dõi."
    assert _remove_greetings("Mụn đầu trắng là mụn nhân đóng.") == "Mụn đầu trắng là mụn nhân đóng."


def test_daily_quota_error_detector_distinguishes_rpd_from_rpm() -> None:
    from evaluation.ragas_runner import _is_daily_quota_error, _is_rate_limit_error

    rpd_exc = RuntimeError("429 RESOURCE_EXHAUSTED: GenerateRequestsPerDayPerProjectPerModel-FreeTier limit 20")
    rpm_exc = RuntimeError("503 UNAVAILABLE: temporary spike")
    assert _is_daily_quota_error(rpd_exc) is True
    assert _is_daily_quota_error(rpm_exc) is False
    assert _is_rate_limit_error(rpm_exc) is True
