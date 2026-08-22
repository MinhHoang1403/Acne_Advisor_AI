from __future__ import annotations

import json

import pytest

from src.agent import action_decision as decision_module
from src.agent.action_decision import (
    AgentDecision,
    parse_agent_decision,
    select_agent_action,
    validate_agent_decision,
)
from src.agent.nodes import workflow


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
        '{"action":"retrieve","retrieval_query":"benzoyl peroxide acne",'
        '"missing_evidence":null,"reason_code":"needs_evidence"}',
    )
    result = await select_agent_action(
        {"normalized_question": "Benzoyl peroxide là gì?", "retrieval_attempt": 0}
    )
    assert result["next_action"] == "retrieve"
    assert result["retrieval_query"] == "benzoyl peroxide acne"
    assert "standalone_question" not in result
    assert result["agent_decision"]["provider"] == "test"
    assert result["agent_decision"]["model_decision"] == {
        "action": "retrieve",
        "retrieval_query": "benzoyl peroxide acne",
        "missing_evidence": None,
        "reason_code": "needs_evidence",
    }
    assert result["agent_decision"]["topic_reset_applied"] is False
    assert result["agent_decision"]["validation_changed"] is False
    assert result["performance_timings"]["agent_decision_1"] >= 0


@pytest.mark.asyncio
async def test_agent_chooses_generation_after_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    await _model(
        monkeypatch,
        '{"action":"generate","retrieval_query":null,"missing_evidence":null,'
        '"reason_code":"evidence_sufficient"}',
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
async def test_decision_evidence_trace_matches_bounded_prompt_view_without_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _model(
        monkeypatch,
        '{"action":"generate","retrieval_query":null,"missing_evidence":null,'
        '"reason_code":"evidence_sufficient"}',
    )
    qualifier = "not recommended during pregnancy"
    contexts = [
        {
            "id": f"chunk-{index}",
            "source_id": "guideline",
            "header": f"Section {index}",
            "text": (("x" * 1250) + qualifier) if index == 1 else f"Evidence {index}",
        }
        for index in range(1, 8)
    ]
    state = {
        "normalized_question": "Mụn là gì?",
        "retrieval_attempt": 1,
        "evidence_assessment": {"usable": True},
        "vector_contexts": contexts,
        "packed_context": {
            "context_text": "\n\n".join(
                f"[Evidence {index} | source=guideline | chunk=chunk-{index}]\n{context['text']}"
                for index, context in enumerate(contexts, 1)
            ),
            "items": [
                {
                    "item_id": context["id"],
                    "text": context["text"],
                    "payload": {
                        "source_id": context["source_id"],
                        "header": context["header"],
                    },
                }
                for context in contexts
            ],
            "debug": {"limits": {"max_items": 8, "max_chars": 6000}},
        },
    }

    _, prompt = decision_module.build_agent_decision_prompt(state)
    result = await select_agent_action(state)
    payload = json.loads(prompt)
    trace = result["agent_decision"]["evidence_trace"]

    assert qualifier in payload["evidence_for_relevance_check"]
    assert payload["evidence_for_relevance_check"] == state["packed_context"]["context_text"]
    assert trace["packed_evidence_count"] == 7
    assert trace["packed_evidence_ids"] == [f"chunk-{index}" for index in range(1, 8)]
    assert trace["decision_visible_evidence_ids"] == [
        f"chunk-{index}" for index in range(1, 8)
    ]
    assert trace["decision_visible_items"][0] == {
        "item_id": "chunk-1",
        "source_id": "guideline",
        "section": "Section 1",
        "position_in_packed_context": 1,
        "original_text_length": 1250 + len(qualifier),
        "decision_visible_text_length": 1250 + len(qualifier),
        "truncated_for_decision": False,
    }
    assert trace["uses_generation_packed_context"] is True
    assert trace["limits"] == {"max_items": 8, "max_chars": 6000}
    assert all("text" not in item for item in trace["decision_visible_items"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("conversation_context", "referral_position"),
    [
        ({"messages": [], "message_count": 0}, 5),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Adapalene và benzoyl peroxide khác nhau thế nào?",
                    },
                    {"role": "assistant", "content": "Hai hoạt chất có cơ chế khác nhau."},
                    {
                        "role": "user",
                        "content": "Clindamycin bôi có nên dùng một mình không?",
                    },
                    {"role": "assistant", "content": "Không nên dùng đơn trị liệu."},
                ],
                "message_count": 4,
            },
            6,
        ),
    ],
)
async def test_referral_evidence_remains_visible_for_standalone_and_multiturn_decisions(
    monkeypatch: pytest.MonkeyPatch,
    conversation_context: dict[str, object],
    referral_position: int,
) -> None:
    async def evidence_aware_model(**kwargs: object) -> dict:
        payload = json.loads(str(kwargs["prompt"]))
        visible_text = payload["evidence_for_relevance_check"]
        if "đi khám bác sĩ da liễu" in visible_text:
            decision = {
                "action": "generate",
                "retrieval_query": None,
                "missing_evidence": None,
                "reason_code": "evidence_sufficient",
            }
        else:
            decision = {
                "action": "retry",
                "retrieval_query": "chỉ định đi khám bác sĩ điều trị mụn",
                "missing_evidence": "chỉ định cụ thể cần đi khám bác sĩ da liễu",
                "reason_code": "evidence_gap",
            }
        return {
            "text": json.dumps(decision, ensure_ascii=False),
            "provider": "test",
            "model": "decision-model",
            "fallback_used": False,
        }

    monkeypatch.setattr(decision_module, "generate_llm_response", evidence_aware_model)
    contexts = [
        {
            "id": f"chunk-{index}",
            "source_id": "guideline",
            "text": (
                "Người bị mụn nên đi khám bác sĩ da liễu khi tự chăm sóc không hiệu quả."
                if index == referral_position
                else f"Thông tin mụn chung {index}."
            ),
        }
        for index in range(1, 9)
    ]

    result = await select_agent_action(
        {
            "normalized_question": (
                "Khi nào người bị mụn nên đi khám bác sĩ thay vì tự chăm sóc ở nhà?"
            ),
            "conversation_context": conversation_context,
            "retrieval_attempt": 1,
            "retrieval_status": "ok",
            "evidence_assessment": {"usable": True},
            "vector_contexts": contexts,
        }
    )

    assert result["next_action"] == "generate"
    assert result["agent_decision"]["reason_code"] == "evidence_sufficient"
    assert result["agent_decision"]["evidence_trace"]["decision_visible_evidence_ids"] == [
        f"chunk-{index}" for index in range(1, 9)
    ]


@pytest.mark.asyncio
async def test_agent_chooses_lexically_distinct_normalized_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _model(
        monkeypatch,
        '{"action":"retry","retrieval_query":"adapalene topical retinoid pregnancy",'
        '"missing_evidence":"whether adapalene is appropriate during pregnancy",'
        '"reason_code":"evidence_gap"}',
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
    assert result["retrieval_query"] == "adapalene topical retinoid pregnancy"
    assert result["missing_evidence"] == "whether adapalene is appropriate during pregnancy"
    assert result["agent_decision"]["missing_evidence"] == result["missing_evidence"]
    assert "standalone_question" not in result


def test_exhausted_retry_and_identical_retry_abstain() -> None:
    decision = AgentDecision(
        action="retry",
        retrieval_query="adapalene",
        missing_evidence="adapalene pregnancy contraindication",
        reason_code="evidence_gap",
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
        action="retrieve",
        retrieval_query="benzoyl peroxide",
        missing_evidence=None,
        reason_code="needs_evidence",
    )
    retry = AgentDecision(
        action="retry",
        retrieval_query="benzoyl peroxide antimicrobial",
        missing_evidence="benzoyl peroxide antimicrobial mechanism",
        reason_code="evidence_gap",
    )
    generate = AgentDecision(
        action="generate",
        retrieval_query=None,
        missing_evidence=None,
        reason_code="evidence_sufficient",
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


def test_retry_requires_prior_usable_evidence() -> None:
    decision = AgentDecision(
        action="retry",
        retrieval_query="adapalene pregnancy",
        missing_evidence="adapalene pregnancy contraindication",
        reason_code="evidence_gap",
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
    assert after_first_retrieval.action == "abstain"


def test_decision_schema_requires_explicit_missing_evidence_key() -> None:
    with pytest.raises(ValueError, match="bounded schema"):
        parse_agent_decision(
            '{"action":"generate","retrieval_query":null,'
            '"reason_code":"evidence_sufficient"}'
        )


def test_purposeful_retry_requires_specific_gap_and_revised_query() -> None:
    state = {
        "retrieval_attempt": 1,
        "evidence_assessment": {"usable": True},
        "retry_history": [{"query": "benzoyl peroxide antimicrobial acne"}],
    }
    valid = AgentDecision(
        action="retry",
        retrieval_query="benzoyl peroxide antibiotic resistance combination acne",
        missing_evidence="whether benzoyl peroxide limits antibiotic resistance in combination",
        reason_code="evidence_gap",
    )
    missing_gap = valid.model_copy(update={"missing_evidence": None})
    duplicate_query = valid.model_copy(
        update={"retrieval_query": "Benzoyl peroxide antimicrobial acne?"}
    )

    assert validate_agent_decision(valid, state).action == "retry"
    assert validate_agent_decision(missing_gap, state).action == "abstain"
    assert validate_agent_decision(duplicate_query, state).action == "abstain"


def test_generate_fails_closed_when_missing_evidence_remains() -> None:
    decision = AgentDecision(
        action="generate",
        retrieval_query=None,
        missing_evidence="pregnancy contraindication for the named treatment",
        reason_code="evidence_sufficient",
    )

    result = validate_agent_decision(
        decision,
        {"retrieval_attempt": 1, "evidence_assessment": {"usable": True}},
    )

    assert result.model_dump() == {
        "action": "abstain",
        "retrieval_query": None,
        "missing_evidence": None,
        "reason_code": "evidence_gap",
    }


def test_semantic_gap_choice_remains_with_model_under_structural_validation() -> None:
    state = {
        "retrieval_attempt": 1,
        "evidence_assessment": {"usable": True},
        "retry_history": [{"query": "benzoyl peroxide antimicrobial action"}],
    }
    supported = AgentDecision(
        action="generate",
        retrieval_query=None,
        missing_evidence=None,
        reason_code="evidence_sufficient",
    )
    unsupported_aspect = AgentDecision(
        action="retry",
        retrieval_query="benzoyl peroxide antibiotic resistance acne combination",
        missing_evidence="effect on antibiotic resistance when combined with acne antibiotics",
        reason_code="evidence_gap",
    )

    assert validate_agent_decision(supported, state).action == "generate"
    assert validate_agent_decision(unsupported_aspect, state).action == "retry"


@pytest.mark.asyncio
async def test_invalid_schema_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    await _model(monkeypatch, '{"action":"run_python","reason":"because"}')
    result = await select_agent_action({"normalized_question": "q", "retrieval_attempt": 0})
    assert result["next_action"] == "abstain"
    assert result["agent_decision"]["reason_code"] == "evidence_gap"
    assert result["fallback_reason_code"] == "insufficient_evidence"


@pytest.mark.asyncio
async def test_action_provider_failure_keeps_provider_unavailable_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(**_: object) -> dict:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(decision_module, "generate_llm_response", unavailable)
    result = await select_agent_action(
        {"normalized_question": "Mụn đầu đen là gì?", "retrieval_attempt": 0}
    )

    assert result["next_action"] == "abstain"
    assert result["fallback_reason_code"] == "provider_unavailable"


def test_generate_without_evidence_is_rejected() -> None:
    decision = AgentDecision(
        action="generate",
        retrieval_query=None,
        missing_evidence=None,
        reason_code="evidence_sufficient",
    )
    assert validate_agent_decision(decision, {"retrieval_attempt": 0}).action == "abstain"


@pytest.mark.parametrize(
    ("action", "reason_code", "state"),
    [
        ("generate", "out_of_scope", {"retrieval_attempt": 1, "evidence_assessment": {"usable": True}}),
        ("generate", "evidence_gap", {"retrieval_attempt": 1, "evidence_assessment": {"usable": True}}),
        ("retrieve", "evidence_sufficient", {"retrieval_attempt": 0}),
        ("retry", "out_of_scope", {"retrieval_attempt": 1}),
    ],
)
def test_invalid_action_reason_pairs_fail_closed(
    action: str,
    reason_code: str,
    state: dict,
) -> None:
    decision = AgentDecision(
        action=action,
        retrieval_query="acne evidence" if action in {"retrieve", "retry"} else None,
        missing_evidence=None,
        reason_code=reason_code,
    )

    result = validate_agent_decision(decision, state)

    assert result.model_dump() == {
        "action": "abstain",
        "retrieval_query": None,
        "missing_evidence": None,
        "reason_code": "evidence_gap",
    }


@pytest.mark.parametrize(
    ("action", "reason_code"),
    [
        ("retrieve", "needs_evidence"),
        ("retry", "evidence_gap"),
        ("generate", "evidence_sufficient"),
        ("abstain", "evidence_gap"),
        ("abstain", "out_of_scope"),
        ("abstain", "cannot_safely_proceed"),
    ],
)
def test_action_reason_contract_accepts_only_legal_semantic_pairs(
    action: str,
    reason_code: str,
) -> None:
    query = "different acne query" if action in {"retrieve", "retry"} else None
    missing_evidence = "specific missing treatment relation" if action == "retry" else None
    state = {
        "retrieval_attempt": 0 if action == "retrieve" else 1,
        "evidence_assessment": {"usable": action in {"retry", "generate"}},
        "retrieval_status": "no_evidence",
        "retry_history": [{"query": "original query"}],
    }
    decision = AgentDecision(
        action=action,
        retrieval_query=query,
        missing_evidence=missing_evidence,
        reason_code=reason_code,
    )

    result = validate_agent_decision(decision, state)

    assert result.action == action
    assert result.reason_code == reason_code


def test_decision_prompt_distinguishes_in_scope_evidence_gap_from_out_of_scope() -> None:
    system_prompt, _ = decision_module.build_agent_decision_prompt(
        {"normalized_question": "Mụn là gì?", "retrieval_attempt": 1}
    )

    assert "unrelated to acne or related skincare" in system_prompt
    assert "requested specificity is not supported by the evidence" in system_prompt
    assert "use evidence_gap with retry or abstain" in system_prompt
    assert "demand absolute certainty, guarantees, or permanent outcomes" in system_prompt
    assert "remain in scope" in system_prompt


def test_decision_prompt_encodes_proposition_grounding_and_epistemic_boundaries() -> None:
    system_prompt, _ = decision_module.build_agent_decision_prompt(
        {"normalized_question": "Benzoyl peroxide có hạn chế kháng thuốc không?"}
    )

    assert "directly supports the requested factual propositions" in system_prompt
    assert "Sharing the same topic" in system_prompt
    assert "is not sufficient by itself" in system_prompt
    assert "Absence of supporting evidence is not evidence that a proposition is false" in system_prompt
    assert "explicitly states that evidence is insufficient" in system_prompt
    assert "runtime merely failed to find support" in system_prompt
    assert "missing_evidence names the specific unsupported relationship" in system_prompt
    assert "self-contained search query" in system_prompt
    assert "Treat the current question as authoritative" in system_prompt


def test_decision_prompt_treats_repeated_history_as_context_not_evidence() -> None:
    question = "Khi nào người bị mụn nên đi khám bác sĩ thay vì tự chăm sóc ở nhà?"
    system_prompt, payload = decision_module.build_agent_decision_prompt(
        {
            "normalized_question": question,
            "conversation_context": {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "Nên đi khám khi mụn nặng."},
                ]
            },
            "retrieval_attempt": 0,
        }
    )

    assert "never retrieval evidence" in system_prompt
    assert "A repeated current question remains a normal request" in system_prompt
    assert "repetition alone is not out_of_scope" in system_prompt
    assert "is not cannot_safely_proceed" in system_prompt
    parsed = json.loads(payload)
    assert parsed["current_question"] == question
    assert parsed["bounded_history"] == []
    assert parsed["evidence_presence"] == {
        "provenance_complete": False,
        "item_count": 0,
    }


@pytest.mark.asyncio
async def test_unsupported_absolute_certainty_about_acne_stays_in_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _model(
        monkeypatch,
        '{"action":"retrieve","retrieval_query":"acne treatment outcomes evidence",'
        '"missing_evidence":null,"reason_code":"needs_evidence"}',
    )

    result = await select_agent_action(
        {
            "normalized_question": (
                "Theo tài liệu, phương pháp nào bảo đảm một kết quả trị mụn vĩnh viễn?"
            ),
            "retrieval_attempt": 0,
        }
    )

    assert result["is_in_domain"] is True
    assert result["next_action"] == "retrieve"
    assert result["agent_decision"]["reason_code"] != "out_of_scope"

    await _model(
        monkeypatch,
        '{"action":"abstain","retrieval_query":null,'
        '"missing_evidence":"evidence about guaranteed permanent acne outcomes",'
        '"reason_code":"evidence_gap"}',
    )
    unsupported = await select_agent_action(
        {
            "normalized_question": (
                "Theo tài liệu, phương pháp nào bảo đảm một kết quả trị mụn vĩnh viễn?"
            ),
            "retrieval_attempt": 1,
            "retrieval_status": "ok",
            "evidence_assessment": {"usable": True},
            "vector_contexts": [
                {
                    "id": "outcome-evidence",
                    "source_id": "guideline",
                    "text": "Điều trị mụn cần được lựa chọn theo tình trạng cụ thể.",
                }
            ],
        }
    )

    assert unsupported["is_in_domain"] is True
    assert unsupported["next_action"] == "abstain"
    assert unsupported["agent_decision"]["reason_code"] == "evidence_gap"
    assert unsupported["fallback_reason_code"] == "insufficient_evidence"


@pytest.mark.asyncio
async def test_repeated_referral_turns_do_not_require_safety_abstention_with_valid_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _model(
        monkeypatch,
        '{"action":"generate","retrieval_query":null,"missing_evidence":null,'
        '"reason_code":"evidence_sufficient"}',
    )
    question = "Khi nào người bị mụn nên đi khám bác sĩ thay vì tự chăm sóc ở nhà?"
    histories = [
        [],
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Nên đi khám khi mụn nặng."},
        ],
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Nên đi khám khi mụn nặng."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Nên đi khám khi tự chăm sóc không hiệu quả."},
        ],
    ]

    for history in histories:
        result = await select_agent_action(
            {
                "normalized_question": question,
                "conversation_context": {"messages": history},
                "retrieval_attempt": 1,
                "retrieval_status": "ok",
                "evidence_assessment": {"usable": True},
                "vector_contexts": [
                    {
                        "id": "referral-evidence",
                        "source_id": "guideline",
                        "text": "Nên đi khám bác sĩ khi mụn nặng hoặc tự chăm sóc không hiệu quả.",
                    }
                ],
            }
        )

        assert result["next_action"] == "generate"
        assert result["agent_decision"]["reason_code"] == "evidence_sufficient"
        assert result.get("fallback_reason_code") != "cannot_safely_proceed"


@pytest.mark.asyncio
async def test_validation_trace_distinguishes_model_decision_from_fail_closed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _model(
        monkeypatch,
        '{"action":"generate","retrieval_query":null,"missing_evidence":null,'
        '"reason_code":"evidence_sufficient"}',
    )

    result = await select_agent_action(
        {"normalized_question": "Mụn là gì?", "retrieval_attempt": 0}
    )

    assert result["agent_decision"]["model_decision"]["action"] == "generate"
    assert result["agent_decision"]["model_decision"]["reason_code"] == "evidence_sufficient"
    assert result["agent_decision"]["action"] == "abstain"
    assert result["agent_decision"]["reason_code"] == "evidence_gap"
    assert result["fallback_reason_code"] == "insufficient_evidence"
    assert result["agent_decision"]["validation_changed"] is True


def test_repeated_pronoun_question_keeps_earlier_entity_context() -> None:
    question = "Nó có phải kháng sinh không?"
    _, payload = decision_module.build_agent_decision_prompt(
        {
            "normalized_question": question,
            "conversation_context": {
                "messages": [
                    {"role": "user", "content": "Benzoyl peroxide là gì?"},
                    {
                        "role": "assistant",
                        "content": "Benzoyl peroxide là một hoạt chất bôi trị mụn.",
                    },
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": "Benzoyl peroxide không phải là kháng sinh.",
                    },
                ]
            },
            "retrieval_attempt": 0,
        }
    )

    history = json.loads(payload)["bounded_history"]
    assert [item["content"] for item in history] == [
        "Benzoyl peroxide là gì?",
        "Benzoyl peroxide là một hoạt chất bôi trị mụn.",
    ]


def test_decision_prompt_prioritizes_explicit_current_topic_switch() -> None:
    system_prompt, payload = decision_module.build_agent_decision_prompt(
        {
            "normalized_question": (
                "Bỏ qua adapalene. Benzoyl peroxide có gây kích ứng không?"
            ),
            "conversation_context": {
                "messages": [
                    {"role": "user", "content": "Adapalene là gì?"},
                    {"role": "assistant", "content": "Adapalene là một retinoid bôi."},
                ]
            },
            "retrieval_attempt": 0,
        }
    )

    assert "Treat the current question as authoritative" in system_prompt
    assert "do not carry the superseded topic into the retrieval query" in system_prompt
    parsed = json.loads(payload)
    assert parsed["current_question"] == "Benzoyl peroxide có gây kích ứng không?"
    assert parsed["bounded_history"][0]["content"] == "Adapalene là gì?"


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_prefix", ["Bỏ qua adapalene.", "Ignore adapalene;"])
async def test_explicit_topic_switch_cannot_restore_superseded_topic_in_initial_query(
    monkeypatch: pytest.MonkeyPatch,
    reset_prefix: str,
) -> None:
    await _model(
        monkeypatch,
        (
            '{"action":"retrieve","retrieval_query":'
            '"adapalene benzoyl peroxide combination acne treatment",'
            '"missing_evidence":null,"reason_code":"needs_evidence"}'
        ),
    )
    result = await select_agent_action(
        {
            "normalized_question": (
                f"{reset_prefix} Benzoyl peroxide có gây kích ứng không?"
            ),
            "conversation_context": {
                "messages": [
                    {"role": "user", "content": "Adapalene là gì?"},
                    {"role": "assistant", "content": "Adapalene là một retinoid bôi."},
                ]
            },
            "retrieval_attempt": 0,
        }
    )

    assert result["next_action"] == "retrieve"
    assert result["retrieval_query"] == "Benzoyl peroxide có gây kích ứng không?"
    assert result["agent_decision"]["model_decision"]["retrieval_query"] == (
        "adapalene benzoyl peroxide combination acne treatment"
    )
    assert result["agent_decision"]["topic_reset_applied"] is True


@pytest.mark.asyncio
async def test_multiturn_history_has_one_semantic_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    async def fake_generate(**kwargs: object) -> dict:
        captured["prompt"] = str(kwargs["prompt"])
        return {
            "text": (
                '{"action":"retrieve","retrieval_query":"benzoyl peroxide dùng buổi tối",'
                '"missing_evidence":null,"reason_code":"needs_evidence"}'
            ),
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
    assert result["retrieval_query"] == "benzoyl peroxide dùng buổi tối"
    assert "standalone_question" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current_question", "history", "model_query", "expected_query"),
    [
        (
            "Nó có gây kích ứng không?",
            [{"role": "user", "content": "Adapalene có tác dụng gì?"}],
            "adapalene irritation adverse effects acne",
            "adapalene irritation adverse effects acne",
        ),
        (
            "Còn kháng thuốc?",
            [{"role": "user", "content": "Benzoyl peroxide có tác dụng thế nào?"}],
            "benzoyl peroxide antibiotic resistance acne combination",
            "benzoyl peroxide antibiotic resistance acne combination",
        ),
        (
            "Còn thời gian dùng thì sao?",
            [{"role": "user", "content": "Clindamycin bôi dùng thế nào?"}],
            "topical clindamycin acne treatment duration",
            "topical clindamycin acne treatment duration",
        ),
        (
            "Bỏ qua adapalene. Benzoyl peroxide có gây kích ứng không?",
            [{"role": "user", "content": "Adapalene có gây khô da không?"}],
            "adapalene benzoyl peroxide irritation",
            "Benzoyl peroxide có gây kích ứng không?",
        ),
        (
            "Benzoyl peroxide có phải kháng sinh không?",
            [
                {"role": "user", "content": "Benzoyl peroxide có phải kháng sinh không?"},
                {"role": "assistant", "content": "Câu trả lời trước."},
            ],
            "benzoyl peroxide antibiotic classification acne",
            "benzoyl peroxide antibiotic classification acne",
        ),
    ],
)
async def test_model_standalone_query_reaches_retrieval_for_multiturn_contracts(
    monkeypatch: pytest.MonkeyPatch,
    current_question: str,
    history: list[dict[str, str]],
    model_query: str,
    expected_query: str,
) -> None:
    captured: dict[str, str] = {}

    async def fake_generate(**_: object) -> dict:
        return {
            "text": json.dumps(
                {
                    "action": "retrieve",
                    "retrieval_query": model_query,
                    "missing_evidence": None,
                    "reason_code": "needs_evidence",
                }
            ),
            "provider": "test",
            "model": "decision-model",
            "fallback_used": False,
        }

    async def fake_retrieve(payload: dict[str, object]) -> dict:
        captured["query"] = str(payload["query"])
        return {
            "vector_contexts": [],
            "sources": [],
            "metadata": {"retrieval_status": "no_evidence", "retrieval_trace": {}},
        }

    class FakeTool:
        ainvoke = staticmethod(fake_retrieve)

    monkeypatch.setattr(decision_module, "generate_llm_response", fake_generate)
    monkeypatch.setattr(workflow, "retrieve_evidence", FakeTool())
    state = {
        "normalized_question": current_question,
        "conversation_context": {"messages": history},
        "retrieval_attempt": 0,
        "retry_history": [],
    }
    decision = await select_agent_action(state)
    await workflow.retrieve_node({**state, **decision})

    assert decision["next_action"] == "retrieve"
    assert decision["retrieval_query"] == expected_query
    assert captured["query"] == expected_query


@pytest.mark.asyncio
async def test_retrieval_rewrite_does_not_replace_original_temporal_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _model(
        monkeypatch,
        '{"action":"retrieve","retrieval_query":"isotretinoin breathing adverse effect",'
        '"missing_evidence":null,"reason_code":"needs_evidence"}',
    )
    original = "Hôm qua tôi khó thở, nhưng đã hết và hiện tại tôi bình thường."
    result = await select_agent_action(
        {
            "user_question": original,
            "normalized_question": original,
            "standalone_question": original,
            "retrieval_attempt": 0,
        }
    )

    assert result["retrieval_query"] == "isotretinoin breathing adverse effect"
    assert "standalone_question" not in result
