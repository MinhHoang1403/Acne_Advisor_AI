from __future__ import annotations

import pytest

from src.agent import graph as graph_module
from src.agent.nodes.fallback import generation_fallback_decision_node, safe_fallback_node
from src.agent.nodes.workflow import abstain_node
from src.observability.versioning import (
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
    get_answer_cache_version,
)
from src.quality.safe_fallback import (
    SAFE_FALLBACK_FLOW_VERSION,
    build_safe_fallback_answer,
    decide_generation_fallback,
)


def test_generation_fallback_accepts_valid_text_and_rejects_empty_output() -> None:
    assert decide_generation_fallback("Câu trả lời có nội dung.").fallback_applied is False
    empty = decide_generation_fallback("   ")
    assert empty.fallback_applied is True
    assert empty.fallback_type == "empty_generation"


@pytest.mark.asyncio
async def test_generation_fallback_node_preserves_valid_draft() -> None:
    result = await generation_fallback_decision_node({"draft_answer": "Câu trả lời hợp lệ."})
    assert result["draft_answer"] == "Câu trả lời hợp lệ."
    assert result["fallback_applied"] is False


@pytest.mark.asyncio
async def test_safe_fallback_metadata_is_truthful_system_origin() -> None:
    result = await safe_fallback_node(
        {
            "fallback_type": "retrieval_error",
            "fallback_reason": "provider unavailable",
        }
    )
    assert result["actual_provider"] == "system"
    assert result["actual_model"] is None
    assert result["llm_fallback_used"] is False
    assert result["fallback_provider"] is None
    assert result["fallback_model"] is None
    assert result["sources"] == []
    assert result["fallback_cache_eligible"] is False


@pytest.mark.asyncio
async def test_abstention_never_manufactures_evidence_or_cache_eligibility() -> None:
    result = await abstain_node({"retrieval_status": "no_evidence"})
    assert result["fallback_type"] == "no_retrieval_evidence"
    assert result["sources"] == []
    assert result["fallback_cache_eligible"] is False


def test_safe_fallback_answers_are_generic_infrastructure_messages() -> None:
    for fallback_type in ("no_retrieval_evidence", "retrieval_error", "invalid_generation"):
        answer = build_safe_fallback_answer(fallback_type)
        assert answer
        assert "isotretinoin" not in answer.casefold()
        assert "benzoyl peroxide" not in answer.casefold()
        assert "retrieval" not in answer.casefold()
        assert "truy xuất" not in answer.casefold()
        assert "context" not in answer.casefold()


def test_graph_routes_all_bounded_actions_and_compiles() -> None:
    for action in ("retrieve", "retry", "generate", "abstain", "finalize"):
        assert graph_module.route_agent_action({"next_action": action}) == action
    assert graph_module.route_agent_action({}) == "abstain"
    assert graph_module.build_clinical_graph() is not None


def test_safe_fallback_version_partitions_pipeline_fingerprint() -> None:
    old = build_pipeline_version_manifest({"SAFE_FALLBACK_FLOW_VERSION": "legacy"})
    new = build_pipeline_version_manifest(
        {"SAFE_FALLBACK_FLOW_VERSION": SAFE_FALLBACK_FLOW_VERSION}
    )
    assert new["safe_fallback_flow_version"] == SAFE_FALLBACK_FLOW_VERSION
    assert compute_pipeline_fingerprint(old) != compute_pipeline_fingerprint(new)
    assert get_answer_cache_version({"CACHE_ANSWER_VERSION": "v5"}) == "v9"
