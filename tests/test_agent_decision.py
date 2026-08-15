from __future__ import annotations

import pytest

from src.agent import action_decision as decision_module
from src.agent.action_decision import AgentDecision, select_agent_action, validate_agent_decision


async def _model(monkeypatch: pytest.MonkeyPatch, payload: str) -> None:
    async def fake_generate(**_: object) -> dict:
        return {
            "text": payload,
            "provider": "test",
            "model": "decision-model",
            "fallback_used": False,
        }

    monkeypatch.setattr(decision_module, "generate_llm_response", fake_generate)


@pytest.mark.asyncio
async def test_agent_chooses_retrieval_before_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    await _model(
        monkeypatch,
        '{"action":"retrieve","retrieval_query":"benzoyl peroxide acne","reason_code":"needs_evidence"}',
    )
    result = await select_agent_action(
        {"normalized_question": "Benzoyl peroxide là gì?", "retrieval_attempt": 0}
    )
    assert result["next_action"] == "retrieve"
    assert result["standalone_question"] == "benzoyl peroxide acne"
    assert result["agent_decision"]["provider"] == "test"


@pytest.mark.asyncio
async def test_agent_chooses_generation_after_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    await _model(
        monkeypatch,
        '{"action":"generate","retrieval_query":null,"reason_code":"evidence_sufficient"}',
    )
    result = await select_agent_action(
        {
            "normalized_question": "Mụn đầu đen là gì?",
            "retrieval_attempt": 1,
            "evidence_assessment": {"usable": True},
            "vector_contexts": [{"text": "source text", "source_file": "source.pdf"}],
        }
    )
    assert result["next_action"] == "generate"


@pytest.mark.asyncio
async def test_agent_chooses_materially_different_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    await _model(
        monkeypatch,
        '{"action":"retry","retrieval_query":"adapalene topical retinoid pregnancy","reason_code":"evidence_gap"}',
    )
    result = await select_agent_action(
        {
            "normalized_question": "Còn thuốc đó khi mang thai?",
            "retrieval_attempt": 1,
            "retrieval_status": "no_evidence",
            "retry_history": [{"query": "adapalene"}],
            "evidence_assessment": {"usable": True},
            "vector_contexts": [{"text": "unrelated", "source_file": "source.pdf"}],
        }
    )
    assert result["next_action"] == "retry"
    assert result["standalone_question"] != "adapalene"


def test_exhausted_retry_and_identical_retry_abstain() -> None:
    decision = AgentDecision(
        action="retry", retrieval_query="adapalene", reason_code="evidence_gap"
    )
    exhausted = validate_agent_decision(
        decision,
        {"retrieval_attempt": 2, "evidence_assessment": {"usable": True}},
    )
    duplicate = validate_agent_decision(
        decision,
        {
            "retrieval_attempt": 1,
            "retrieval_status": "no_evidence",
            "evidence_assessment": {"usable": True},
            "retry_history": [{"query": "Adapalene?"}],
        },
    )
    assert exhausted.action == "abstain"
    assert duplicate.action == "abstain"


@pytest.mark.asyncio
async def test_invalid_schema_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    await _model(monkeypatch, '{"action":"run_python","reason":"because"}')
    result = await select_agent_action({"normalized_question": "q", "retrieval_attempt": 0})
    assert result["next_action"] == "abstain"
    assert result["agent_decision"]["reason_code"] == "cannot_safely_proceed"


def test_generate_without_evidence_is_rejected() -> None:
    decision = AgentDecision(
        action="generate", retrieval_query=None, reason_code="evidence_sufficient"
    )
    assert validate_agent_decision(decision, {"retrieval_attempt": 0}).action == "abstain"


@pytest.mark.asyncio
async def test_multiturn_history_has_one_semantic_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    async def fake_generate(**kwargs: object) -> dict:
        captured["prompt"] = str(kwargs["prompt"])
        return {
            "text": '{"action":"retrieve","retrieval_query":"benzoyl peroxide dùng buổi tối","reason_code":"needs_evidence"}',
            "provider": "test",
            "model": "decision-model",
            "fallback_used": False,
        }

    monkeypatch.setattr(decision_module, "generate_llm_response", fake_generate)
    result = await select_agent_action(
        {
            "normalized_question": "còn buổi tối?",
            "conversation_context": {
                "messages": [{"role": "user", "content": "Benzoyl peroxide dùng thế nào?"}]
            },
            "retrieval_attempt": 0,
        }
    )
    assert "Benzoyl peroxide" in captured["prompt"]
    assert result["standalone_question"] == "benzoyl peroxide dùng buổi tối"
