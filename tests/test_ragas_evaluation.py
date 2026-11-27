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
    project_rpd_msg = "Google GenAI daily project quota is exhausted (HTTP 429)."
    rpm_exc = RuntimeError("503 UNAVAILABLE: temporary spike")
    assert _is_daily_quota_error(rpd_exc) is True
    assert _is_daily_quota_error(project_rpd_msg) is True
    assert _is_daily_quota_error(rpm_exc) is False
    assert _is_rate_limit_error(rpm_exc) is True


@pytest.mark.asyncio
async def test_run_system_on_dataset_fails_fast_on_daily_quota_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.ragas_runner import EvaluationCase, run_system_on_dataset

    calls = 0

    async def fake_warmup() -> None:
        return None

    async def fake_run_agent(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "answer": "Xin lỗi, hệ thống tạm bận.",
            "fallback_applied": True,
            "fallback_reason_code": "provider_unavailable",
            "fallback_reason": "Google GenAI daily project quota is exhausted (HTTP 429).",
            "agent_decision": {"action": "abstain", "reason_code": "provider_unavailable"},
        }

    monkeypatch.setattr("evaluation.ragas_runner._warmup_reranker", fake_warmup)
    monkeypatch.setattr("src.agent.graph.run_clinical_agent", fake_run_agent)

    cases = [
        EvaluationCase(
            case_id="NEW-SYM-01",
            category="symptoms_and_types",
            user_input="Mụn nang là gì?",
            reference="Mụn nang là tổn thương viêm sâu.",
            expected_behavior="answer",
        )
    ]

    with pytest.raises(RuntimeError, match="cạn hạn mức theo ngày"):
        await run_system_on_dataset(
            cases,
            llm_provider="gemini",
            llm_model="gemini-3.5-flash-lite",
            max_provider_retries=2,
            provider_retry_delay_seconds=0.01,
        )

    assert calls == 1


def test_build_ragas_evaluators_nvidia_uses_md_json_and_disables_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import instructor
    from evaluation.ragas_runner import (
        DEFAULT_JUDGE_MODEL,
        DEFAULT_NVIDIA_BASE_URL,
        build_ragas_evaluators,
    )

    captured: dict[str, Any] = {}

    def fake_llm_factory(model: str, **kwargs: Any) -> dict[str, Any]:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return {"fake_llm": model}

    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
    monkeypatch.setenv("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL)
    monkeypatch.setattr("ragas.llms.llm_factory", fake_llm_factory)

    llm, emb = build_ragas_evaluators(judge_provider="nvidia")

    assert llm == {"fake_llm": DEFAULT_JUDGE_MODEL}
    assert captured["model"] == DEFAULT_JUDGE_MODEL
    assert captured["kwargs"]["provider"] == "openai"
    assert captured["kwargs"]["mode"] == instructor.Mode.MD_JSON
    assert captured["kwargs"]["max_tokens"] == 8192
    assert captured["kwargs"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert emb.dimensions == 3072


@pytest.mark.asyncio
async def test_multi_turn_case_passes_history_and_closes_judge_clients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evaluation.ragas_runner import (
        EvaluationCase,
        _evaluate_safe_abstention,
        load_evaluation_dataset,
        run_system_on_dataset,
        score_records_with_ragas,
    )

    sample_ds = tmp_path / "mini_dataset.json"
    sample_ds.write_text(
        json.dumps(
            {
                "dataset_name": "acne_advisor_ragas_benchmark",
                "cases": [
                    {
                        "case_id": "MT-01",
                        "category": "safety_and_precautions",
                        "dialogue_type": "multi_turn",
                        "expected_behavior": "answer",
                        "conversation_history": [
                            {"role": "user", "content": "Bác sĩ vừa kê isotretinoin đường uống trị mụn nang cho mình."},
                            {"role": "assistant", "content": "Isotretinoin đường uống được dùng cho mụn trứng cá nặng."},
                        ],
                        "user_input": "Thuốc đó có dùng chung với doxycycline được không?",
                        "standalone_question": "Isotretinoin đường uống có dùng chung với kháng sinh doxycycline được không?",
                        "reference": "Không được phối hợp isotretinoin đường uống với doxycycline do nguy cơ tăng áp lực nội sọ lành tính.",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _, loaded_cases = load_evaluation_dataset(sample_ds)
    assert len(loaded_cases) == 1
    assert loaded_cases[0].dialogue_type == "multi_turn"
    assert len(loaded_cases[0].conversation_history) == 2

    captured_agent_kwargs: dict[str, Any] = {}

    async def fake_warmup() -> None:
        return None

    async def fake_run_agent(**kwargs: Any) -> dict[str, Any]:
        captured_agent_kwargs.update(kwargs)
        return {
            "answer": "Không dùng chung isotretinoin với doxycycline.",
            "packed_context": {"items": [{"text": "Tránh phối hợp isotretinoin và doxycycline."}]},
            "sources": ["SRC_test"],
            "agent_decision": {"action": "generate", "reason_code": "evidence_sufficient"},
            "fallback_applied": False,
        }

    monkeypatch.setattr("evaluation.ragas_runner._warmup_reranker", fake_warmup)
    monkeypatch.setattr("src.agent.graph.run_clinical_agent", fake_run_agent)

    sys_records = await run_system_on_dataset(loaded_cases, case_delay_seconds=0.0)
    assert len(captured_agent_kwargs["conversation_history"]) == 2
    assert sys_records[0]["standalone_question"].startswith("Isotretinoin")

    # Kiểm tra loại trừ lỗi sập hạ tầng khỏi kết quả từ chối an toàn
    assert _evaluate_safe_abstention(
        {"action": "abstain", "fallback_applied": True, "fallback_reason_code": "provider_unavailable"}
    ) is False
    assert _evaluate_safe_abstention(
        {"action": "abstain", "fallback_applied": True, "fallback_reason_code": "out_of_scope"}
    ) is True

    closed_count = 0

    class DummyInnerHttpClient:
        async def close(self) -> None:
            nonlocal closed_count
            closed_count += 1

    class DummyInstructorWrapper:
        def __init__(self) -> None:
            self.client = DummyInnerHttpClient()

    class DummyJudgeLLM:
        def __init__(self) -> None:
            self.client = DummyInstructorWrapper()

    class DummyMetricResult:
        def __init__(self, val: float) -> None:
            self.value = val

    captured_questions: list[str] = []

    class DummyMetric:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.prompt = type("P", (), {"instruction": "Generate a question."})()

        async def ascore(self, **kwargs: Any) -> DummyMetricResult:
            if "user_input" in kwargs:
                captured_questions.append(kwargs["user_input"])
            return DummyMetricResult(0.92)

    monkeypatch.setattr(
        "evaluation.ragas_runner.build_ragas_evaluators",
        lambda **_kw: (DummyJudgeLLM(), ProjectGeminiEmbeddings(api_key="k")),
    )
    for m_name in ("Faithfulness", "AnswerRelevancy", "ContextPrecision", "ContextRecall", "AnswerCorrectness"):
        monkeypatch.setattr(f"ragas.metrics.collections.{m_name}", DummyMetric)

    scored = await score_records_with_ragas(
        sys_records,
        judge_provider="nvidia",
        judge_model="nvidia/nemotron-3-ultra-550b-a55b",
        fallback_judge_models=("mistralai/mistral-large-4-1t-a52b",),
        request_delay_seconds=0.0,
    )
    assert scored[0]["ragas_scores"]["faithfulness"] == 0.92
    assert all(q.startswith("Isotretinoin") for q in captured_questions)
    assert closed_count == 2
    assert isinstance(EvaluationCase("X", "cat", "answer", "q", "r"), EvaluationCase)


def test_lexical_graph_brand_bridges_and_mixed_accent_facets() -> None:
    from src.knowledge.lexical_graph import (
        extract_clinical_facets_from_text,
        extract_query_graph_anchors,
    )
    from src.quality.answer_verifier import verify_answer_quality
    from src.quality.safe_fallback import GENERIC_NON_SAFETY_FALLBACK_ANSWER

    winlevi_anchors = extract_query_graph_anchors("Kem bôi Winlevi 1% trị mụn trứng cá")
    assert "clascoterone" in winlevi_anchors.entity_names

    duac_anchors = extract_query_graph_anchors("Vì sao bác sĩ khuyên dùng gel Duac trị mụn viêm?")
    assert {"clindamycin", "benzoyl_peroxide"}.issubset(set(duac_anchors.entity_names))

    mixed_facets = extract_clinical_facets_from_text(
        "Mình bị mun an và sẹo lõm lâu năm", allow_ascii_fallback=True
    )
    assert "comedonal_acne" in mixed_facets.get("acne_subtype", [])

    # Không bắt nhầm "mụn nặng" thành "mun nang" hay "bi mun an uong gi" thành "mụn ẩn"
    severe_facets = extract_clinical_facets_from_text(
        "Mình bị mụn nặng ở hai má", allow_ascii_fallback=True
    )
    assert "nodulocystic_acne" not in severe_facets.get("acne_subtype", [])
    diet_facets = extract_clinical_facets_from_text(
        "bi mun an uong gi cho mat da", allow_ascii_fallback=True
    )
    assert "comedonal_acne" not in diet_facets.get("acne_subtype", [])

    # Safe fallback không bị gắn cờ sai lệch thực thể bởi answer_verifier
    report = verify_answer_quality(
        query="Tinh chất nọc ong pha rượu gạo bôi isotretinoin liều bao nhiêu?",
        answer=GENERIC_NON_SAFETY_FALLBACK_ANSWER,
        fallback_applied=True,
    )
    assert report.passed is True
