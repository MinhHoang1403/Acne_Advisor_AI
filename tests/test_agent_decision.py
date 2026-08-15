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
    assert result["performance_timings"]["agent_decision_1"] >= 0


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
            "performance_timings": {"agent_decision_1": 1.25},
        }
    )
    assert result["next_action"] == "generate"
    assert result["performance_timings"]["agent_decision_1"] == 1.25
    assert result["performance_timings"]["agent_decision_2"] >= 0


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


def test_retrieval_transition_contract_enforces_action_and_budget() -> None:
    retrieve = AgentDecision(
        action="retrieve", retrieval_query="benzoyl peroxide", reason_code="needs_evidence"
    )
    retry = AgentDecision(
        action="retry", retrieval_query="benzoyl peroxide antimicrobial", reason_code="evidence_gap"
    )
    generate = AgentDecision(
        action="generate", retrieval_query=None, reason_code="evidence_sufficient"
    )

    first = validate_agent_decision(retrieve, {"retrieval_attempt": 0})
    retrieve_as_retry = validate_agent_decision(retrieve, {"retrieval_attempt": 1})
    valid_retry = validate_agent_decision(
        retry,
        {
            "retrieval_attempt": 1,
            "evidence_assessment": {"usable": True},
            "retry_history": [{"query": "benzoyl peroxide"}],
        },
    )
    exhausted_retrieve = validate_agent_decision(retrieve, {"retrieval_attempt": 2})
    exhausted_retry = validate_agent_decision(
        retry,
        {"retrieval_attempt": 2, "evidence_assessment": {"usable": True}},
    )
    exhausted_generate = validate_agent_decision(
        generate,
        {"retrieval_attempt": 2, "evidence_assessment": {"usable": True}},
    )

    assert first.action == "retrieve"
    assert retrieve_as_retry.action == "abstain"
    assert valid_retry.action == "retry"
    assert exhausted_retrieve.action == "abstain"
    assert exhausted_retry.action == "abstain"
    assert exhausted_generate.action == "generate"


def test_retry_requires_a_prior_retrieval_even_without_evidence() -> None:
    decision = AgentDecision(
        action="retry", retrieval_query="adapalene pregnancy", reason_code="evidence_gap"
    )

    before_first_retrieval = validate_agent_decision(decision, {"retrieval_attempt": 0})
    after_first_retrieval = validate_agent_decision(
        decision,
        {
            "retrieval_attempt": 1,
            "evidence_assessment": {"usable": False},
            "retry_history": [{"query": "adapalene"}],
        },
    )

    assert before_first_retrieval.action == "abstain"
    assert after_first_retrieval.action == "retry"


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
