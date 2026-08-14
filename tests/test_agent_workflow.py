import pytest

from src.agent import graph as graph_module
from src.agent.graph import clinical_graph, route_agent_action
from src.agent.state import ClinicalState
from src.agent.nodes import workflow


def test_final_graph_has_eight_semantic_nodes() -> None:
    nodes = set(clinical_graph.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {"prepare", "guard", "decide", "retrieve", "assess", "generate", "abstain", "finalize"}


@pytest.mark.asyncio
async def test_agent_decision_is_bounded_and_meaningful() -> None:
    assert (await workflow.decide_node({"is_in_domain": False}))["next_action"] == "finalize"
    assert (await workflow.decide_node({"is_in_domain": True, "cache_hit": True}))["next_action"] == "finalize"
    assert (await workflow.decide_node({"is_in_domain": True, "retrieval_attempt": 0}))["next_action"] == "retrieve"
    assert (
        await workflow.decide_node(
            {"is_in_domain": True, "retrieval_attempt": 1, "evidence_assessment": {"sufficient": True}}
        )
    )["next_action"] == "generate"
    assert (
        await workflow.decide_node(
            {"is_in_domain": True, "retrieval_attempt": 2, "evidence_assessment": {"sufficient": False}}
        )
    )["next_action"] == "abstain"


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


@pytest.mark.asyncio
async def test_assessment_requires_text_and_provenance() -> None:
    missing_source = await workflow.assess_evidence_node(
        {"vector_contexts": [{"text": "Medical text"}], "retrieval_attempt": 1}
    )
    complete = await workflow.assess_evidence_node(
        {"vector_contexts": [{"text": "Medical text", "source_id": "guideline"}], "retrieval_attempt": 1}
    )

    assert missing_source["evidence_assessment"]["sufficient"] is False
    assert complete["evidence_assessment"]["sufficient"] is True
    assert complete["evidence_assessment"]["source_ids"] == ["guideline"]


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
                "retrieval_trace": {"selected_ids": ["chunk-1"]},
                "packed_context": {"items": [{"item_id": "chunk-1"}]},
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
