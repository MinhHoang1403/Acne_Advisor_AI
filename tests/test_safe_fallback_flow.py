from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.agent import graph as graph_module
from src.agent.emergency_contract import first_sentence_has_immediate_emergency_action
from src.agent.nodes import fallback as fallback_node
from src.agent.nodes import reason as reason_node
from src.agent.nodes.cache import cache_lookup_node, cache_store_node
from src.agent.nodes.workflow import abstain_node, decide_node
from src.api.app import ChatRequest, chat_endpoint, _http_status_for_resilience_error
from src.observability.trace_exporter import build_observability_event
from src.observability.versioning import (
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
    get_answer_cache_version,
)
from src.quality.safe_fallback import (
    SAFE_FALLBACK_FLOW_VERSION,
    build_safe_fallback_answer,
    decide_generation_fallback,
    sanitize_fallback_reason,
)
from src.quality.severity_guard import apply_severity_aware_answer_guard
from src.resilience.exceptions import (
    AgentTimeoutError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StageTimeoutError,
)


def test_empty_query_fallback_is_deterministic_and_short():
    answer = build_safe_fallback_answer("empty_query")

    assert "vấn đề da hoặc loại mụn" in answer
    assert "Thông tin này chỉ nhằm định hướng" in answer


def test_insufficient_context_fallback_copy_remains_available():
    assert "chưa đủ bằng chứng" in build_safe_fallback_answer("insufficient_context")


@pytest.mark.asyncio
async def test_sufficient_assessment_routes_to_generation():
    result = await decide_node({"is_in_domain": True, "evidence_assessment": {"sufficient": True}})

    assert result == {"next_action": "generate"}


@pytest.mark.asyncio
async def test_retrieval_error_abstention_sanitizes_reason():
    reason = sanitize_fallback_reason("boom token=secret-value path C:/tmp")
    decision = await abstain_node(
        {
            "standalone_question": "mụn viêm",
            "retrieval_error": reason,
        }
    )

    assert "secret-value" not in reason
    assert decision["fallback_type"] == "retrieval_error"
    assert decision["fallback_cache_eligible"] is False


def test_fallback_reason_redacts_password_in_connection_url():
    reason = sanitize_fallback_reason(
        "connection failed: postgresql://user:secret-password@localhost/database"
    )

    assert "secret-password" not in reason
    assert "[REDACTED]" in reason


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", "empty_generation"),
        ("   ", "empty_generation"),
        (None, "invalid_generation"),
        ({"text": "bad"}, "invalid_generation"),
    ],
)
def test_generation_output_validation_invalid_cases(value, expected):
    decision = decide_generation_fallback(value)

    assert decision.fallback_applied is True
    assert decision.fallback_type == expected
    assert decision.fallback_cache_eligible is False


def test_generation_output_validation_accepts_valid_answer():
    decision = decide_generation_fallback("Câu trả lời có nội dung.")

    assert decision.fallback_applied is False
    assert decision.fallback_type == "none"


@pytest.mark.asyncio
async def test_generation_fallback_repairs_complete_terminal_sentence_before_fallback():
    result = await fallback_node.generation_fallback_decision_node(
        {"draft_answer": "Mụn đầu đen là tổn thương không viêm thường gặp ở da dầu"}
    )

    assert result["fallback_applied"] is False
    assert result["draft_answer"].endswith(".")


@pytest.mark.asyncio
async def test_generation_fallback_removes_an_empty_markdown_heading_before_fallback():
    result = await fallback_node.generation_fallback_decision_node(
        {"draft_answer": "Thông tin này đủ để trả lời câu hỏi hiện tại.\n\n**Lưu ý**"}
    )

    assert result["fallback_applied"] is False
    assert "**Lưu ý**" not in result["draft_answer"]


@pytest.mark.asyncio
async def test_generation_fallback_recovers_verified_entity_relation_instead_of_refusal():
    result = await fallback_node.generation_fallback_decision_node(
        {
            "standalone_question": "Tazarotene có alias tazaroten không?",
            "draft_answer": "",
        }
    )

    assert result["fallback_type"] == "grounded_direct_recovery"
    assert "alias" in result["draft_answer"].lower()


@pytest.mark.asyncio
async def test_fallback_nodes_apply_answer_and_metadata():
    decision = await abstain_node(
        {
            "standalone_question": "mụn đầu đen là gì?",
        }
    )

    assert decision["fallback_applied"] is True
    assert decision["fallback_type"] == "no_retrieval_evidence"
    assert decision["actual_provider"] == "system"
    assert decision["actual_model"] is None
    assert decision["llm_fallback_used"] is False
    assert decision["fallback_provider"] is None
    assert decision["fallback_model"] is None
    assert decision["fallback_cache_eligible"] is False


@pytest.mark.asyncio
async def test_retrieval_error_fallback_keeps_emergency_action_first():
    query = "Môi và lưỡi sưng sau khi dùng thuốc mới trị mụn, nên làm gì ngay?"
    decision = await abstain_node(
        {
            "standalone_question": query,
            "retrieval_error": "SSLEOFError: provider disconnected",
        }
    )

    assert decision["medical_severity"] == "emergency"
    assert first_sentence_has_immediate_emergency_action(decision["draft_answer"]) is True


@pytest.mark.asyncio
async def test_generate_empty_string_routes_to_generation_fallback(monkeypatch):
    async def fake_generate_llm_response(**kwargs):
        return {
            "text": "   ",
            "provider": "gemini",
            "model": "test",
            "fallback_used": False,
            "fallback_provider": None,
            "fallback_model": None,
            "resilience": {},
        }

    monkeypatch.setattr(reason_node, "generate_llm_response", fake_generate_llm_response)
    state = {
        "user_question": "mụn đầu đen là gì?",
        "standalone_question": "mụn đầu đen là gì?",
        "vector_contexts": [{"text": "Mụn đầu đen là dạng nhân mụn mở.", "source_file": "doc.pdf"}],
        "graph_facts": [],
        "safety_flags": [],
        "symptoms": [],
        "conversation_history": [],
        "is_in_domain": True,
    }
    generated = await reason_node.generate_answer_node(state)
    decision = await fallback_node.generation_fallback_decision_node(generated)

    assert generated["draft_answer"] == "   "
    assert decision["fallback_type"] == "empty_generation"


@pytest.mark.asyncio
async def test_generate_invalid_none_routes_to_generation_fallback(monkeypatch):
    async def fake_generate_llm_response(**kwargs):
        return {
            "text": None,
            "provider": "gemini",
            "model": "test",
            "fallback_used": False,
            "fallback_provider": None,
            "fallback_model": None,
            "resilience": {},
        }

    monkeypatch.setattr(reason_node, "generate_llm_response", fake_generate_llm_response)
    generated = await reason_node.generate_answer_node(
        {
            "user_question": "mụn đầu đen là gì?",
            "standalone_question": "mụn đầu đen là gì?",
            "vector_contexts": [{"text": "Mụn đầu đen là dạng nhân mụn mở."}],
            "graph_facts": [],
            "safety_flags": [],
            "symptoms": [],
            "conversation_history": [],
            "is_in_domain": True,
        }
    )
    decision = await fallback_node.generation_fallback_decision_node(generated)

    assert generated["draft_answer"] is None
    assert decision["fallback_type"] == "invalid_generation"


@pytest.mark.asyncio
async def test_generate_valid_provider_fallback_success_is_preserved(monkeypatch):
    async def fake_generate_llm_response(**kwargs):
        return {
            "text": "Câu trả lời hợp lệ từ provider fallback.",
            "provider": "gemini",
            "model": "gemini-test",
            "fallback_used": True,
            "fallback_provider": "ollama",
            "fallback_model": "qwen3:8b",
            "resilience": {"ok": True},
        }

    monkeypatch.setattr(reason_node, "generate_llm_response", fake_generate_llm_response)
    result = await reason_node.generate_answer_node(
        {
            "user_question": "mụn đầu đen là gì?",
            "standalone_question": "mụn đầu đen là gì?",
            "vector_contexts": [{"text": "Mụn đầu đen là dạng nhân mụn mở."}],
            "graph_facts": [],
            "safety_flags": [],
            "symptoms": [],
            "conversation_history": [],
            "is_in_domain": True,
        }
    )

    assert result["draft_answer"].startswith("Câu trả lời")
    assert result["llm_fallback_used"] is True
    assert result["fallback_provider"] == "ollama"


@pytest.mark.asyncio
async def test_provider_unavailable_and_timeout_not_swallowed(monkeypatch):
    async def unavailable(**kwargs):
        raise ValueError("no providers")

    async def timeout(**kwargs):
        raise ProviderTimeoutError("slow")

    monkeypatch.setattr(reason_node, "generate_llm_response", unavailable)
    with pytest.raises(ProviderUnavailableError):
        await reason_node.generate_answer_node(
            {
                "user_question": "mụn",
                "standalone_question": "mụn",
                "vector_contexts": [{"text": "evidence"}],
                "graph_facts": [],
                "safety_flags": [],
                "symptoms": [],
                "conversation_history": [],
                "is_in_domain": True,
            }
        )

    monkeypatch.setattr(reason_node, "generate_llm_response", timeout)
    with pytest.raises(ProviderTimeoutError):
        await reason_node.generate_answer_node(
            {
                "user_question": "mụn",
                "standalone_question": "mụn",
                "vector_contexts": [{"text": "evidence"}],
                "graph_facts": [],
                "safety_flags": [],
                "symptoms": [],
                "conversation_history": [],
                "is_in_domain": True,
            }
        )


def test_structured_http_status_for_runtime_errors():
    assert _http_status_for_resilience_error(AgentTimeoutError("agent")) == 504
    assert _http_status_for_resilience_error(StageTimeoutError("stage")) == 504
    assert _http_status_for_resilience_error(ProviderTimeoutError("provider")) == 504
    assert _http_status_for_resilience_error(ProviderUnavailableError("provider")) == 503


def test_severity_precedence_over_generic_fallback():
    generic = build_safe_fallback_answer("no_retrieval_evidence")
    emergency = apply_severity_aware_answer_guard(
        "Bôi thuốc xong tôi khó thở, sưng môi và nổi mề đay",
        generic,
    )
    urgent = apply_severity_aware_answer_guard(
        "Tôi đang mang thai, có dùng isotretinoin trị mụn được không?",
        generic,
    )
    caution = apply_severity_aware_answer_guard(
        "Da tôi đỏ rát nhẹ khi dùng benzoyl peroxide",
        generic,
    )
    routine = apply_severity_aware_answer_guard("Mụn đầu đen là gì?", generic)

    assert emergency.classification.severity == "emergency"
    assert "cấp cứu" in emergency.answer
    assert urgent.classification.severity == "urgent"
    assert "24-48" in urgent.answer
    assert caution.answer.count("**Lưu ý an toàn**") <= 1
    assert routine.answer == generic


@pytest.mark.asyncio
async def test_fallback_answer_is_not_cached(monkeypatch):
    called = False

    async def fake_set_answer_cache(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("src.agent.nodes.cache.set_answer_cache", fake_set_answer_cache)
    result = await cache_store_node(
        {
            "cache_hit": False,
            "bypass_cache": False,
            "conversation_history": [],
            "cache_reason": "miss",
            "medical_severity": "routine",
            "severity_guard_modified": False,
            "severity_guard_cache_eligible": True,
            "fallback_applied": True,
            "fallback_type": "no_retrieval_evidence",
            "fallback_cache_eligible": False,
            "retrieval_status": "no_evidence",
        }
    )

    assert called is False
    assert result["cache_reason"] == "safe_fallback_no_retrieval_evidence"


@pytest.mark.asyncio
async def test_routine_valid_answer_still_can_store(monkeypatch):
    captured = {}

    async def fake_set_answer_cache(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("src.agent.nodes.cache.set_answer_cache", fake_set_answer_cache)
    monkeypatch.setenv("CACHE_MIN_ANSWER_CHARS", "10")
    monkeypatch.setenv("CACHE_REQUIRED_ENTITY_CHECK", "false")
    await cache_store_node(
        {
            "cache_hit": False,
            "bypass_cache": False,
            "conversation_history": [],
            "cache_reason": "miss",
            "medical_severity": "routine",
            "severity_guard_modified": False,
            "severity_guard_cache_eligible": True,
            "fallback_applied": False,
            "fallback_type": "none",
            "fallback_cache_eligible": True,
            "retrieval_status": "success",
            "is_in_domain": True,
            "use_history_context": False,
            "errors": [],
            "llm_fallback": False,
            "llm_fallback_used": False,
            "fallback_provider": None,
            "guardrail": "in_domain",
            "user_question": "Mụn đầu đen là gì?",
            "final_answer": "Mụn đầu đen là dạng nhân mụn mở liên quan bít tắc nang lông.",
            "sources": ["source.pdf"],
            "actual_provider": "gemini",
            "actual_model": "gemini-test",
            "answer_quality_report": {"passed": True, "issues": []},
            "pipeline_manifest": {"phase": "phase2e"},
            "pipeline_fingerprint": "fp",
        }
    )

    assert captured["metadata"]["retrieval_status"] == "success"
    assert captured["metadata"]["fallback_applied"] is False


@pytest.mark.asyncio
async def test_cache_hit_does_not_run_retrieval_fallback():
    result = await cache_lookup_node(
        {
            "user_question": "Mụn ở gần mắt bị sưng đỏ đau và chảy mủ",
            "conversation_history": [],
            "is_in_domain": True,
            "bypass_cache": False,
            "medical_severity": "urgent",
        }
    )

    assert result["cache_hit"] is False
    assert result["cache_reason"] == "severity_urgent"
    assert "fallback_applied" not in result


def test_graph_routes_and_compile_safe_fallback_flow():
    assert graph_module.route_agent_action({"next_action": "retrieve"}) == "retrieve"
    assert graph_module.route_agent_action({"next_action": "generate"}) == "generate"
    assert graph_module.route_agent_action({}) == "abstain"
    assert graph_module.build_clinical_graph() is not None


@pytest.mark.asyncio
async def test_run_clinical_agent_exposes_fallback_metadata(monkeypatch):
    class FakeGraph:
        async def ainvoke(self, state):
            assert state["fallback_applied"] is False
            return {
                **state,
                "final_answer": "Fallback answer",
                "retrieval_status": "no_evidence",
                "fallback_applied": True,
                "fallback_type": "no_retrieval_evidence",
                "fallback_reason": "No evidence",
                "fallback_cache_eligible": False,
            }

    monkeypatch.setattr(graph_module, "clinical_graph", FakeGraph())
    result = await graph_module.run_clinical_agent("mụn đầu đen")

    assert result["retrieval_status"] == "no_evidence"
    assert result["fallback_applied"] is True
    assert result["fallback_type"] == "no_retrieval_evidence"
    assert result["fallback_cache_eligible"] is False


@pytest.mark.asyncio
async def test_chat_response_exposes_fallback_metadata(monkeypatch):
    async def fake_run_clinical_agent(**kwargs):
        return {
            "answer": "Fallback answer",
            "session_id": kwargs.get("session_id"),
            "sources": [],
            "symptoms": [],
            "safety_flags": [],
            "graph_facts": [],
            "retrieval_status": "no_evidence",
            "fallback_applied": True,
            "fallback_type": "no_retrieval_evidence",
            "fallback_reason": "No evidence",
            "fallback_cache_eligible": False,
            "is_in_domain": True,
            "actual_provider": "system",
            "actual_model": None,
            "cache_checked": True,
            "cache_hit": False,
            "cache_reason": "safe_fallback_no_retrieval_evidence",
            "pipeline_fingerprint": "fp",
            "pipeline_manifest": {
                "phase": "phase2e",
                "answer_cache_version": "v5",
                "safe_fallback_flow_version": "safe_fallback_flow_v1",
            },
            "answer_quality_report": {"passed": True, "issues": []},
        }

    async def fake_persist(*args, **kwargs):
        return None

    monkeypatch.setattr("src.api.app.run_clinical_agent", fake_run_clinical_agent)
    monkeypatch.setattr("src.api.app._persist_chat_to_db", fake_persist)
    response = await chat_endpoint(ChatRequest(message="mụn đầu đen", user_id="u1"))

    assert response.metadata.retrieval_status == "no_evidence"
    assert response.metadata.provider == "system"
    assert response.metadata.model is None
    assert response.metadata.fallback_applied is True
    assert response.metadata.fallback_type == "no_retrieval_evidence"
    assert response.metadata.cache.reason == "safe_fallback_no_retrieval_evidence"


@pytest.mark.asyncio
async def test_chat_runtime_resilience_error_stays_http_503(monkeypatch):
    async def fake_run_clinical_agent(**kwargs):
        raise ProviderUnavailableError("provider down")

    monkeypatch.setattr("src.api.app.run_clinical_agent", fake_run_clinical_agent)

    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoint(ChatRequest(message="mụn đầu đen", user_id="u1"))

    assert exc_info.value.status_code == 503


def test_observability_event_has_safe_fallback_metadata():
    event = build_observability_event(
        query="mụn đầu đen",
        state={
            "retrieval_status": "no_evidence",
            "fallback_applied": True,
            "fallback_type": "no_retrieval_evidence",
            "fallback_reason": "No evidence token=secret",
            "fallback_cache_eligible": False,
            "medical_severity": "routine",
            "cache_reason": "safe_fallback_no_retrieval_evidence",
        },
        pipeline_manifest={"phase": "phase2e", "safe_fallback_flow_version": SAFE_FALLBACK_FLOW_VERSION},
        pipeline_fingerprint="fp",
    )
    metadata = event.summary.metadata

    assert metadata["retrieval_status"] == "no_evidence"
    assert metadata["fallback_applied"] is True
    assert metadata["fallback_type"] == "no_retrieval_evidence"
    assert event.safe_payload["fallback_type"] == "no_retrieval_evidence"


def test_versioning_changes_fingerprint_without_cache_version_change():
    old = build_pipeline_version_manifest({"SAFE_FALLBACK_FLOW_VERSION": ""})
    new = build_pipeline_version_manifest({"SAFE_FALLBACK_FLOW_VERSION": SAFE_FALLBACK_FLOW_VERSION})

    assert new["safe_fallback_flow_version"] == SAFE_FALLBACK_FLOW_VERSION
    assert compute_pipeline_fingerprint(old) != compute_pipeline_fingerprint(new)
    assert get_answer_cache_version({"CACHE_ANSWER_VERSION": "v5"}) == "v6"
