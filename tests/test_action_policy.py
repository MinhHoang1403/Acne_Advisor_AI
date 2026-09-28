from __future__ import annotations

import hashlib
import json

import pytest

from src.agent import action_decision as decision_module
from src.agent.action_decision import AgentDecision, build_agent_decision_prompt
from src.agent.action_policy import (
    LEGAL_REASONS_BY_ACTION,
    MAX_RETRIEVAL_ATTEMPTS,
    ActionPolicyContext,
    ActionProposal,
    validate_action_proposal,
)
from src.agent.nodes import workflow


ACTIONS = ("retrieve", "retry", "generate", "abstain")
REASONS = (
    "needs_evidence",
    "evidence_sufficient",
    "evidence_gap",
    "out_of_scope",
    "cannot_safely_proceed",
)


def _valid_proposal(action: str, reason: str) -> ActionProposal:
    return ActionProposal(
        action=action,  # type: ignore[arg-type]
        retrieval_query=(
            "initial acne query"
            if action == "retrieve"
            else "targeted acne evidence gap"
            if action == "retry"
            else None
        ),
        missing_evidence="specific unsupported relation" if action == "retry" else None,
        reason_code=reason,  # type: ignore[arg-type]
        direct_supporting_evidence_ids=("evidence-1",) if action == "generate" else None,
    )


def _valid_context(action: str) -> ActionPolicyContext:
    return ActionPolicyContext(
        attempt_index=0 if action == "retrieve" else 1,
        evidence_available=action in {"retry", "generate"},
        retrieval_status="ok",
        visible_evidence_ids=frozenset({"evidence-1"}),
        previous_queries=("initial acne query",),
    )


@pytest.mark.parametrize(
    ("action", "reason"),
    [(action, reason) for action in ACTIONS for reason in REASONS],
)
def test_legal_action_reason_matrix_is_owned_by_deterministic_policy(
    action: str,
    reason: str,
) -> None:
    result = validate_action_proposal(_valid_proposal(action, reason), _valid_context(action))

    if reason in LEGAL_REASONS_BY_ACTION[action]:  # type: ignore[index]
        assert result.decision.action == action
        assert result.decision.reason_code == reason
    else:
        assert result.decision == ActionProposal(
            action="abstain",
            retrieval_query=None,
            missing_evidence=None,
            reason_code="evidence_gap",
            direct_supporting_evidence_ids=None,
        )
        assert result.policy_reason_code == "illegal_action_reason"


@pytest.mark.parametrize(
    ("proposal", "context", "expected_reason"),
    [
        (
            ActionProposal("retrieve", None, None, "needs_evidence", None),
            ActionPolicyContext(0, False),
            "invalid_initial_retrieval",
        ),
        (
            ActionProposal("retrieve", "   ", None, "needs_evidence", None),
            ActionPolicyContext(0, False),
            "invalid_initial_retrieval",
        ),
        (
            ActionProposal("retrieve", "query", "gap", "needs_evidence", None),
            ActionPolicyContext(0, False),
            "invalid_initial_retrieval",
        ),
        (
            ActionProposal("retrieve", "query", None, "needs_evidence", None),
            ActionPolicyContext(1, False),
            "invalid_initial_retrieval",
        ),
        (
            ActionProposal("retrieve", "query", None, "needs_evidence", None),
            ActionPolicyContext(0, True),
            "invalid_initial_retrieval",
        ),
        (
            ActionProposal("retry", "new query", "gap", "evidence_gap", None),
            ActionPolicyContext(0, True),
            "invalid_retry_transition",
        ),
        (
            ActionProposal("retry", "new query", "gap", "evidence_gap", None),
            ActionPolicyContext(MAX_RETRIEVAL_ATTEMPTS, True),
            "retry_budget_exhausted",
        ),
        (
            ActionProposal("retry", None, "gap", "evidence_gap", None),
            ActionPolicyContext(1, True),
            "invalid_retry_transition",
        ),
        (
            ActionProposal("retry", "new query", "   ", "evidence_gap", None),
            ActionPolicyContext(1, True),
            "invalid_retry_transition",
        ),
        (
            ActionProposal("retry", "new query", "gap", "evidence_gap", None),
            ActionPolicyContext(1, False, retrieval_status="failed"),
            "invalid_retry_transition",
        ),
        (
            ActionProposal("retry", "Same query?", "gap", "evidence_gap", None),
            ActionPolicyContext(1, True, previous_queries=("same query",)),
            "retry_query_not_revised",
        ),
        (
            ActionProposal("generate", None, None, "evidence_sufficient", ("evidence-1",)),
            ActionPolicyContext(1, False, visible_evidence_ids=frozenset({"evidence-1"})),
            "invalid_generate_shape",
        ),
        (
            ActionProposal("generate", None, "gap", "evidence_sufficient", ("evidence-1",)),
            ActionPolicyContext(1, True, visible_evidence_ids=frozenset({"evidence-1"})),
            "invalid_generate_shape",
        ),
        (
            ActionProposal("generate", None, None, "evidence_sufficient", None),
            ActionPolicyContext(1, True, visible_evidence_ids=frozenset({"evidence-1"})),
            "invalid_generate_shape",
        ),
        (
            ActionProposal("generate", None, None, "evidence_sufficient", ("invented",)),
            ActionPolicyContext(1, True, visible_evidence_ids=frozenset({"evidence-1"})),
            "invalid_direct_evidence_ids",
        ),
    ],
)
def test_policy_rejects_important_invalid_transition_shapes(
    proposal: ActionProposal,
    context: ActionPolicyContext,
    expected_reason: str,
) -> None:
    result = validate_action_proposal(proposal, context)

    assert result.decision.action == "abstain"
    assert result.decision.reason_code == "evidence_gap"
    assert result.policy_reason_code == expected_reason
    assert result.changed_by_policy is True


def test_policy_normalizes_fields_without_creating_semantic_content() -> None:
    retry = validate_action_proposal(
        ActionProposal(
            "retry",
            "  targeted   query  ",
            "  specific   gap  ",
            "evidence_gap",
            ("must-be-removed",),
        ),
        ActionPolicyContext(1, True, previous_queries=("initial query",)),
    )
    generate = validate_action_proposal(
        ActionProposal(
            "generate",
            "must-be-removed",
            None,
            "evidence_sufficient",
            ("evidence-1", "evidence-1", "evidence-2"),
        ),
        ActionPolicyContext(
            1,
            True,
            visible_evidence_ids=frozenset({"evidence-1", "evidence-2"}),
        ),
    )
    abstain = validate_action_proposal(
        ActionProposal(
            "abstain",
            "must-be-removed",
            "  exact   gap  ",
            "evidence_gap",
            ("must-be-removed",),
        ),
        ActionPolicyContext(1, True),
    )

    assert retry.decision == ActionProposal(
        "retry", "targeted query", "specific gap", "evidence_gap", None
    )
    assert generate.decision == ActionProposal(
        "generate",
        None,
        None,
        "evidence_sufficient",
        ("evidence-1", "evidence-2"),
    )
    assert abstain.decision == ActionProposal("abstain", None, "exact gap", "evidence_gap", None)


def test_prompt_schema_and_budget_match_pre_extraction_baseline() -> None:
    state = {
        "normalized_question": "Adapalene và benzoyl peroxide khác nhau thế nào?",
        "conversation_context": {
            "messages": [{"role": "user", "content": "Tôi đang tìm hiểu điều trị mụn."}]
        },
        "retrieval_attempt": 1,
        "retrieval_status": "ok",
        "evidence_assessment": {"usable": True},
        "packed_context": {
            "context_text": "packed",
            "items": [
                {
                    "item_id": "chunk-1",
                    "text": "Evidence text.",
                    "payload": {"source_id": "source.pdf", "chunk_id": "chunk-1"},
                }
            ],
            "debug": {"limits": {"max_items": 9, "max_chars": 7000}},
        },
        "retry_history": [{"query": "overall information need"}],
    }
    system_prompt, payload = build_agent_decision_prompt(state)
    schema = json.dumps(
        AgentDecision.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
    )

    assert hashlib.sha256(system_prompt.encode()).hexdigest() == (
        "a7b970c50fc1b2ca3c847b05cdbe6b42f56f2f12712293f40c6d32442b391bc9"
    )
    assert hashlib.sha256(payload.encode()).hexdigest() == (
        "502f9114c170990e4415d45e2a56680ef271aab33ff343637fd04224af57ac59"
    )
    assert hashlib.sha256(schema.encode()).hexdigest() == (
        "6ccfc5c40cf4e159c40b6a9e250364077e803de4a507f4893b03bf1a87c8ad8c"
    )
    assert MAX_RETRIEVAL_ATTEMPTS == 2


@pytest.mark.asyncio
async def test_facade_calls_provider_once_and_records_bounded_policy_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def fake_generate(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        assert kwargs["response_schema"] is AgentDecision
        return {
            "text": json.dumps(
                {
                    "action": "retrieve",
                    "retrieval_query": "benzoyl peroxide acne",
                    "missing_evidence": None,
                    "reason_code": "needs_evidence",
                    "direct_supporting_evidence_ids": None,
                }
            ),
            "provider": "test",
            "model": "decision-model",
            "fallback_used": False,
        }

    monkeypatch.setattr(decision_module, "generate_llm_response", fake_generate)
    result = await decision_module.select_agent_action(
        {"normalized_question": "Benzoyl peroxide là gì?", "retrieval_attempt": 0}
    )

    assert calls == 1
    assert result["next_action"] == "retrieve"
    assert result["agent_decision"]["policy_validation"] == {
        "changed_by_policy": False,
        "policy_reason_code": "accepted",
        "attempt_index": 0,
    }


@pytest.mark.asyncio
async def test_safety_and_cache_precedence_skip_decision_while_normal_miss_calls_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def fake_select(_state: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "next_action": "abstain",
            "agent_decision": {
                "action": "abstain",
                "reason_code": "evidence_gap",
                "retrieval_query": None,
                "missing_evidence": None,
            },
        }

    monkeypatch.setattr(workflow, "select_agent_action", fake_select)

    assert (await workflow.decide_node({"safety_override": True}))["next_action"] == "finalize"
    assert (await workflow.decide_node({"cache_hit": True}))["next_action"] == "finalize"
    assert calls == 0
    assert (await workflow.decide_node({"retrieval_attempt": 0}))["next_action"] == "abstain"
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError("timeout"), RuntimeError("unavailable")])
async def test_provider_failures_preserve_provider_unavailable_fallback(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    async def fail(**_: object) -> dict[str, object]:
        raise error

    monkeypatch.setattr(decision_module, "generate_llm_response", fail)
    result = await decision_module.select_agent_action(
        {"normalized_question": "Mụn là gì?", "retrieval_attempt": 0}
    )

    assert result["next_action"] == "abstain"
    assert result["agent_decision"]["reason_code"] == "cannot_safely_proceed"
    assert result["fallback_reason_code"] == "provider_unavailable"
    assert result["agent_decision"]["policy_validation"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "state", "policy_reason"),
    [
        ("not-json", {"retrieval_attempt": 0}, None),
        ('{"action":"retrieve"}', {"retrieval_attempt": 0}, None),
        (
            '{"action":"unknown","reason_code":"needs_evidence"}',
            {"retrieval_attempt": 0},
            None,
        ),
        (
            '{"action":"retrieve","retrieval_query":"q","reason_code":"evidence_gap"}',
            {"retrieval_attempt": 0},
            "illegal_action_reason",
        ),
        (
            '{"action":"retrieve","retrieval_query":null,"reason_code":"needs_evidence"}',
            {"retrieval_attempt": 0},
            "invalid_initial_retrieval",
        ),
        (
            '{"action":"generate","reason_code":"evidence_sufficient",'
            '"direct_supporting_evidence_ids":["invented"]}',
            {
                "retrieval_attempt": 1,
                "evidence_assessment": {"usable": True},
                "vector_contexts": [
                    {"id": "evidence-1", "text": "Evidence", "source_id": "source"}
                ],
            },
            "invalid_direct_evidence_ids",
        ),
    ],
)
async def test_invalid_model_outputs_preserve_fail_closed_outcome(
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
    state: dict[str, object],
    policy_reason: str | None,
) -> None:
    async def fake_generate(**_: object) -> dict[str, object]:
        return {
            "text": payload,
            "provider": "test",
            "model": "decision-model",
            "fallback_used": False,
        }

    monkeypatch.setattr(decision_module, "generate_llm_response", fake_generate)
    result = await decision_module.select_agent_action({"normalized_question": "question", **state})

    assert result["next_action"] == "abstain"
    assert result["agent_decision"]["reason_code"] == "evidence_gap"
    assert result["fallback_reason_code"] == "insufficient_evidence"
    validation = result["agent_decision"]["policy_validation"]
    if policy_reason is None:
        assert validation is None
    else:
        assert validation["policy_reason_code"] == policy_reason
