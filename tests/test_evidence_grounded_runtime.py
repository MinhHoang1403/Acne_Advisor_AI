from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from src.agent import action_decision
from src.agent import graph as graph_module
from src.agent.nodes import reason, workflow
from src.agent.nodes.quality import answer_quality_node
from src.agent.nodes.respond import finalize_response_node
from src.agent.prompts.medical_answer import MEDICAL_RAG_SYSTEM_PROMPT
from src.api.app import _response_origin, retrieve_endpoint
from src.retrieval.context_packer import pack_context
from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate


def _state(draft: str) -> dict:
    return {
        "user_question": "Câu hỏi kiểm soát?",
        "standalone_question": "Câu hỏi kiểm soát?",
        "is_in_domain": True,
        "fallback_applied": False,
        "fallback_type": "none",
        "safety_severity": None,
        "source_allowlist": [{"source_id": "source-a", "display_name": "Source A"}],
        "sources": ["source-a"],
        "vector_contexts": [{"text": "Evidence A", "source_id": "source-a"}],
        "draft_answer": draft,
        "performance_timings": {},
    }


@pytest.mark.asyncio
async def test_controlled_llm_draft_survives_normal_finalization() -> None:
    marker = "Nội dung kiểm soát độc nhất từ model."
    result = await finalize_response_node(_state(marker))
    quality = await answer_quality_node({**_state(marker), **result})
    assert marker in quality["final_answer"]


@pytest.mark.asyncio
async def test_different_drafts_with_same_query_and_evidence_remain_different() -> None:
    first = await finalize_response_node(_state("Model chọn cách giải thích A."))
    second = await finalize_response_node(_state("Model chọn cách giải thích B."))
    assert first["final_answer"] != second["final_answer"]
    assert "A" in first["final_answer"]
    assert "B" in second["final_answer"]


@pytest.mark.asyncio
async def test_emergency_override_clears_retrieved_attribution_and_cache_eligibility() -> None:
    result = await workflow.guard_node(
        {"normalized_question": "Sau khi bôi thuốc tôi khó thở và sưng môi."}
    )
    assert result["actual_provider"] == "system"
    assert result["sources"] == []
    assert result["fallback_cache_eligible"] is False
    assert _response_origin(result, True) == "deterministic_safety"


@pytest.mark.asyncio
async def test_deterministic_prescription_boundary_reports_system_without_model() -> None:
    result = await workflow.guard_node(
        {"normalized_question": "Hãy kê đơn isotretinoin cho tôi."}
    )
    assert result["actual_provider"] == "system"
    assert result["actual_model"] is None


@pytest.mark.asyncio
async def test_no_evidence_abstains_without_blind_second_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def identical_retry(**_: object) -> dict:
        return {
            "text": (
                '{"action":"retry","retrieval_query":"Câu hỏi giống nhau",'
                '"reason_code":"evidence_gap"}'
            ),
            "provider": "test",
            "model": "decision-model",
            "fallback_used": False,
        }

    monkeypatch.setattr(action_decision, "generate_llm_response", identical_retry)
    state = {
        "is_in_domain": True,
        "user_question": "Câu hỏi giống nhau",
        "standalone_question": "Câu hỏi giống nhau",
        "retrieval_attempt": 1,
        "retrieval_status": "no_evidence",
        "retry_history": [{"query": "Câu hỏi giống nhau"}],
        "evidence_assessment": {"usable": False},
    }
    assert (await workflow.decide_node(state))["next_action"] == "abstain"


@pytest.mark.asyncio
async def test_generation_uses_true_system_instruction_and_canonical_packed_evidence(monkeypatch) -> None:
    captured: dict = {}

    async def fake_generate_llm_response(**kwargs):
        captured.update(kwargs)
        return {
            "text": "Câu trả lời từ model.",
            "requested_provider": "mock",
            "requested_model": "mock-model",
            "provider": "mock",
            "model": "mock-model",
            "fallback_used": False,
            "fallback_provider": None,
            "fallback_model": None,
        }

    monkeypatch.setattr(reason, "generate_llm_response", fake_generate_llm_response)
    packed_text = "[Evidence 1 | source=source-a | chunk=chunk-1]\nCanonical evidence only."
    result = await reason.generate_answer_node(
        {
            **_state(""),
            "packed_context": {"context_text": packed_text},
            "llm_provider": "mock",
            "llm_model": "mock-model",
            "allow_model_fallback": False,
        }
    )
    assert captured["system_prompt"].startswith(MEDICAL_RAG_SYSTEM_PROMPT.strip())
    assert "<CURRENT_QUESTION>" in captured["prompt"]
    assert packed_text in captured["prompt"]
    assert MEDICAL_RAG_SYSTEM_PROMPT not in captured["prompt"]
    assert result["generation_evidence_trace"] == {
        "current_question": "Câu hỏi kiểm soát?",
        "conversation_history_messages": 0,
        "answer_context_ids": [None],
        "packed_evidence": [],
    }


def test_packed_item_and_rendered_evidence_obey_actual_character_limit() -> None:
    packed = pack_context(
        NormalizedQuery(original_query="q", normalized_text="q"),
        [
            RetrievedCandidate(
                candidate_id="chunk-1",
                collection="acne_knowledge",
                text="x" * 5000,
                payload={
                    "source_id": "source-a",
                    "chunk_id": "chunk-1",
                    "content": "x" * 5000,
                },
                rank=1,
            )
        ],
        max_chars=512,
    )
    assert len(packed.context_text) <= 512
    assert packed.items[0].text in packed.context_text
    assert len(packed.items[0].text) < 5000
    assert "content" not in packed.items[0].payload


@pytest.mark.asyncio
async def test_fallback_requires_global_enable_and_request_opt_in(monkeypatch) -> None:
    captured: list[bool] = []

    class FakeGraph:
        async def ainvoke(self, state):
            captured.append(state["allow_model_fallback"])
            return {**state, "final_answer": "ok"}

    monkeypatch.setattr(graph_module, "clinical_graph", FakeGraph())
    monkeypatch.setenv("LLM_PROVIDER_FALLBACK_ENABLED", "false")
    await graph_module.run_clinical_agent("q", allow_model_fallback=True)
    monkeypatch.setenv("LLM_PROVIDER_FALLBACK_ENABLED", "true")
    await graph_module.run_clinical_agent("q", allow_model_fallback=False)
    await graph_module.run_clinical_agent("q", allow_model_fallback=True)
    assert captured == [False, False, True]


@pytest.mark.asyncio
async def test_retry_is_bounded_and_has_concrete_reason() -> None:
    assert workflow.MAX_RETRIEVAL_ATTEMPTS == 2
    assert not hasattr(workflow, "_next_retrieval_query")


@pytest.mark.asyncio
async def test_diagnostic_retrieve_is_hidden_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_DIAGNOSTIC_RETRIEVE", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        await retrieve_endpoint("query", 5)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_operator_can_explicitly_enable_diagnostic_retrieve(monkeypatch) -> None:
    class FakeRetriever:
        async def retrieve(self, query, top_k):
            class Result:
                vector_contexts = []
                sources = []
                metadata = {"retrieval_status": "no_evidence"}

            assert query == "query"
            assert top_k == 3
            return Result()

        async def close(self):
            return None

    monkeypatch.setenv("ENABLE_DIAGNOSTIC_RETRIEVE", "true")
    monkeypatch.setattr("src.retrieval.service.EvidenceRetriever", FakeRetriever)
    result = await retrieve_endpoint("query", 3)
    assert result.metadata["retrieval_status"] == "no_evidence"


def test_response_origin_distinguishes_llm_and_deterministic_safety() -> None:
    assert _response_origin({"actual_provider": "gemini"}, True) == "llm"
    assert _response_origin({"safety_decision": {"rule_id": "test"}}, True) == (
        "deterministic_safety"
    )


def test_removed_medical_engines_cannot_reenter_normal_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    formatter = (root / "src/agent/answer_formatting.py").read_text(encoding="utf-8")
    verifier = (root / "src/quality/answer_verifier.py").read_text(encoding="utf-8")
    assert "DrugEntityNormalizer" not in formatter
    assert "src.knowledge" not in formatter
    assert "grounded_entity_relation_answer" not in formatter
    assert "DomainProposition" not in verifier
    assert "proposition_detector" not in verifier
    assert not (root / "src/quality/proposition_detector.py").exists()
    assert not (root / "src/retrieval/query_normalization.py").exists()
