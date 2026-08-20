import json

import pytest

from src.agent import graph as graph_module
from src.agent import action_decision as decision_module
from src.agent.graph import clinical_graph, route_agent_action
from src.agent.state import ClinicalState
from src.agent.nodes import workflow


def test_graph_has_eight_semantic_nodes() -> None:
    nodes = set(clinical_graph.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {"prepare", "guard", "decide", "retrieve", "assess", "generate", "abstain", "finalize"}


@pytest.mark.asyncio
async def test_agent_decision_is_bounded_and_meaningful() -> None:
    assert (await workflow.decide_node({"safety_override": True}))["next_action"] == "finalize"
    assert (await workflow.decide_node({"cache_hit": True}))["next_action"] == "finalize"


@pytest.mark.asyncio
async def test_prepare_and_guard_record_only_executed_stage_timings() -> None:
    prepared = await workflow.prepare_node(
        {"user_question": "Mụn là gì?", "conversation_history": [], "performance_timings": {}}
    )
    guarded = await workflow.guard_node(
        {
            **prepared,
            "user_question": "Mụn là gì?",
            "bypass_cache": True,
        }
    )
    skipped_decision = await workflow.decide_node({"cache_hit": True})

    assert prepared["performance_timings"]["prepare"] >= 0
    assert guarded["performance_timings"]["prepare"] >= 0
    assert guarded["performance_timings"]["guard"] >= 0
    assert "performance_timings" not in skipped_decision


def test_route_reads_only_explicit_agent_action() -> None:
    assert route_agent_action({"next_action": "retrieve"}) == "retrieve"
    assert route_agent_action({}) == "abstain"


def test_clinical_state_has_no_retired_fallback_or_error_fields() -> None:
    fields = ClinicalState.__annotations__

    assert "answerability" not in fields
    assert "errors" not in fields
    assert "prompt_budget" in fields


@pytest.mark.asyncio
async def test_run_clinical_agent_returns_prompt_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGraph:
        async def ainvoke(self, state):
            return {
                **state,
                "final_answer": "Nguồn đã được đóng gói.",
                "prompt_budget": {"accounting_mode": "observation_only"},
            }

    monkeypatch.setattr(graph_module, "clinical_graph", FakeGraph())
    result = await graph_module.run_clinical_agent("Mụn đầu đen là gì?")

    assert result["prompt_budget"] == {"accounting_mode": "observation_only"}
    assert "answerability" not in result
    assert "errors" not in result
    assert "graph_relation_found" not in result
    assert result["performance_timings"]["agent_total"] >= 0
    assert "agent_decision_1" not in result["performance_timings"]


@pytest.mark.asyncio
async def test_generation_diagnostics_are_explicitly_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGraph:
        async def ainvoke(self, state):
            return {
                **state,
                "draft_answer": "Raw grounded draft.",
                "final_answer": "Presented grounded answer.",
                "answer_quality_report": {
                    "passed": True,
                    "checked_answer": "Pre-verifier grounded answer.",
                    "issues": [],
                },
            }

    monkeypatch.setattr(graph_module, "clinical_graph", FakeGraph())

    regular = await graph_module.run_clinical_agent("Question")
    diagnostic = await graph_module.run_clinical_agent(
        "Question",
        include_generation_diagnostics=True,
    )

    assert "generation_diagnostics" not in regular
    assert diagnostic["generation_diagnostics"] == {
        "raw_generated_answer": "Raw grounded draft.",
        "pre_verifier_answer": "Pre-verifier grounded answer.",
    }


@pytest.mark.asyncio
async def test_generation_diagnostics_fall_back_to_final_for_safety_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGraph:
        async def ainvoke(self, state):
            return {
                **state,
                "draft_answer": "Deterministic safety draft.",
                "final_answer": "Presented safety answer.",
                "answer_quality_report": {"passed": True, "issues": []},
            }

    monkeypatch.setattr(graph_module, "clinical_graph", FakeGraph())

    result = await graph_module.run_clinical_agent(
        "Safety question",
        include_generation_diagnostics=True,
    )

    assert result["generation_diagnostics"]["pre_verifier_answer"] == (
        "Presented safety answer."
    )


@pytest.mark.asyncio
async def test_assessment_requires_text_and_provenance() -> None:
    missing_source = await workflow.assess_evidence_node(
        {"vector_contexts": [{"text": "Medical text"}], "retrieval_attempt": 1}
    )
    complete = await workflow.assess_evidence_node(
        {"vector_contexts": [{"text": "Medical text", "source_id": "guideline"}], "retrieval_attempt": 1}
    )

    assert missing_source["evidence_assessment"]["usable"] is False
    assert complete["evidence_assessment"]["usable"] is True
    assert complete["evidence_assessment"]["assessment_kind"] == "provenance_complete_evidence_presence"
    assert complete["evidence_assessment"]["source_ids"] == ["guideline"]


@pytest.mark.asyncio
async def test_decide_node_records_post_retrieval_decision_visible_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_select(_state: ClinicalState) -> dict:
        return {
            "next_action": "generate",
            "agent_decision": {
                "action": "generate",
                "reason_code": "evidence_sufficient",
                "retrieval_query": None,
                "provider": "test",
                "model": "decision-model",
                "fallback_used": False,
                "evidence_trace": {
                    "packed_evidence_count": 2,
                    "packed_evidence_ids": ["chunk-1", "chunk-2"],
                    "decision_visible_evidence_count": 1,
                    "decision_visible_evidence_ids": ["chunk-1"],
                    "decision_visible_items": [
                        {
                            "item_id": "chunk-1",
                            "source_id": "guideline",
                            "section": "Treatment",
                            "position_in_packed_context": 1,
                            "original_text_length": 1400,
                            "decision_visible_text_length": 1200,
                            "truncated_for_decision": True,
                        }
                    ],
                    "limits": {"max_items": 5, "max_chars_per_item": 1200},
                },
            },
        }

    monkeypatch.setattr(workflow, "select_agent_action", fake_select)
    result = await workflow.decide_node(
        {
            "retrieval_attempt": 1,
            "agent_decision_history": [{"action": "retrieve"}],
            "agent_decision_evidence_traces": [],
        }
    )

    assert result["agent_decision_evidence_traces"] == [
        {
            "decision_index": 2,
            "retrieval_attempts_used": 1,
            "packed_evidence_count": 2,
            "packed_evidence_ids": ["chunk-1", "chunk-2"],
            "decision_visible_evidence_count": 1,
            "decision_visible_evidence_ids": ["chunk-1"],
            "decision_visible_items": [
                {
                    "item_id": "chunk-1",
                    "source_id": "guideline",
                    "section": "Treatment",
                    "position_in_packed_context": 1,
                    "original_text_length": 1400,
                    "decision_visible_text_length": 1200,
                    "truncated_for_decision": True,
                }
            ],
            "limits": {"max_items": 5, "max_chars_per_item": 1200},
            "action": "generate",
            "reason_code": "evidence_sufficient",
            "retrieval_query": None,
            "provider": "test",
            "model": "decision-model",
            "requested_provider": None,
            "requested_model": None,
            "provider_fallback_attempted": False,
            "provider_fallback_used": False,
            "fallback_reason_code": None,
        }
    ]


@pytest.mark.asyncio
async def test_retrieve_action_uses_tool_and_never_injects_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ainvoke(_payload):
        return {
            "vector_contexts": [
                {
                    "id": "chunk-1",
                    "chunk_id": "chunk-1",
                    "text": "Source evidence",
                    "source_id": "guideline",
                }
            ],
            "sources": ["guideline"],
            "metadata": {
                "retrieval_status": "ok",
                "retrieval_trace": {
                    "selected_ids": ["chunk-1"],
                    "candidate_trace": {
                        "dense": [{"candidate_id": "chunk-1", "rank": 1}],
                        "bm25": [],
                        "fused": [{"candidate_id": "chunk-1", "rank": 1}],
                    },
                    "packer": {"limits": {"max_items": 8, "max_chars": 6000}},
                },
                "packed_context": {
                    "items": [
                        {
                            "item_id": "chunk-1",
                            "payload": {"source_id": "guideline", "header": "Treatment"},
                        }
                    ]
                },
            },
        }

    class FakeTool:
        ainvoke = staticmethod(fake_ainvoke)

    monkeypatch.setattr(workflow, "retrieve_evidence", FakeTool())
    result = await workflow.retrieve_node(
        {"user_question": "Mụn là gì?", "standalone_question": "Mụn là gì?", "retrieval_attempt": 0}
    )

    assert result["retrieval_attempt"] == 1
    assert "graph_facts" not in result
    assert "graph_relation_found" not in result
    assert result["source_allowlist"][0]["source_id"] == "guideline"
    trace = result["retrieval_attempt_traces"]
    assert len(trace) == 1
    assert trace[0]["attempt_index"] == 1
    assert trace[0]["candidate_trace"]["dense"][0]["candidate_id"] == "chunk-1"
    assert trace[0]["packed_evidence"] == [
        {"item_id": "chunk-1", "source_id": "guideline", "section": "Treatment"}
    ]


@pytest.mark.asyncio
async def test_graph_cannot_execute_a_third_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    retrieve_calls = 0

    async def fake_generate(**kwargs: object) -> dict:
        attempt = json.loads(str(kwargs["prompt"]))["retrieval_attempt"]
        if attempt == 0:
            payload = {
                "action": "retrieve",
                "retrieval_query": "initial acne query",
                "reason_code": "needs_evidence",
            }
        elif attempt == 1:
            payload = {
                "action": "retry",
                "retrieval_query": "changed acne evidence query",
                "reason_code": "evidence_gap",
            }
        else:
            payload = {
                "action": "retrieve",
                "retrieval_query": "attempt to bypass retry budget",
                "reason_code": "needs_evidence",
            }
        return {
            "text": json.dumps(payload),
            "provider": "test",
            "model": "decision-model",
            "fallback_used": False,
        }

    async def fake_prepare(state: ClinicalState) -> dict:
        return {
            "normalized_question": state["user_question"],
            "standalone_question": state["user_question"],
            "conversation_context": {"messages": []},
        }

    async def fake_cache_lookup(_state: ClinicalState) -> dict:
        return {"cache_checked": True, "cache_hit": False}

    async def fake_retrieve(_payload: dict) -> dict:
        nonlocal retrieve_calls
        retrieve_calls += 1
        return {
            "vector_contexts": [],
            "sources": [],
            "metadata": {"retrieval_status": "no_evidence", "retrieval_trace": {}},
        }

    async def fake_fallback(_state: ClinicalState) -> dict:
        return {"draft_answer": "Không đủ bằng chứng nguồn.", "fallback_applied": True}

    async def fake_finalize(state: ClinicalState) -> dict:
        return {"final_answer": state.get("draft_answer", "")}

    async def no_updates(_state: ClinicalState) -> dict:
        return {}

    class FakeTool:
        ainvoke = staticmethod(fake_retrieve)

    monkeypatch.setattr(decision_module, "generate_llm_response", fake_generate)
    monkeypatch.setattr(workflow, "prepare_request_node", fake_prepare)
    monkeypatch.setattr(workflow, "cache_lookup_node", fake_cache_lookup)
    monkeypatch.setattr(workflow, "retrieve_evidence", FakeTool())
    monkeypatch.setattr(workflow, "safe_fallback_node", fake_fallback)
    monkeypatch.setattr(workflow, "finalize_response_node", fake_finalize)
    monkeypatch.setattr(workflow, "answer_quality_node", no_updates)
    monkeypatch.setattr(workflow, "cache_store_node", no_updates)
    monkeypatch.setattr(workflow, "observability_export_node", no_updates)

    bounded_graph = graph_module.build_clinical_graph()
    result = await bounded_graph.ainvoke(
        {
            "user_question": "What evidence is available?",
            "retrieval_attempt": 0,
            "retry_history": [],
            "vector_contexts": [],
            "sources": [],
            "performance_timings": {},
        }
    )

    assert retrieve_calls == 2
    assert result["retrieval_attempt"] == 2
    assert result["agent_decision"]["action"] == "abstain"
    assert [item["action"] for item in result["agent_decision_history"]] == [
        "retrieve",
        "retry",
        "abstain",
    ]
    assert [item["attempt_index"] for item in result["retrieval_attempt_traces"]] == [1, 2]
    assert result["performance_timings"]["agent_decision_1"] >= 0
    assert result["performance_timings"]["agent_decision_2"] >= 0
    assert result["performance_timings"]["agent_decision_3"] >= 0
    assert result["performance_timings"]["finalize"] >= 0
