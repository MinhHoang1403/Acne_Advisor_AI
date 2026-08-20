from __future__ import annotations

import pytest

from src.agent import graph as graph_module
from src.agent.nodes import workflow
from src.agent.nodes.fallback import generation_fallback_decision_node, safe_fallback_node
from src.agent.nodes.workflow import abstain_node
from src.observability.versioning import (
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
    get_answer_cache_version,
)
from src.quality.safe_fallback import (
    GENERIC_NON_SAFETY_FALLBACK_ANSWER,
    SAFE_FALLBACK_FLOW_VERSION,
    build_safe_fallback_answer,
    decide_generation_fallback,
    fallback_reason_label,
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
async def test_generation_failure_keeps_invocation_identity_separate_from_system_fallback(
    monkeypatch,
) -> None:
    async def fake_generate(_state):
        return {
            "draft_answer": "",
            "actual_provider": "gemini",
            "actual_model": "gemini-3.5-flash-lite",
        }

    monkeypatch.setattr(workflow, "generate_answer_node", fake_generate)

    result = await workflow.generate_node({})

    assert result["actual_provider"] == "system"
    assert result["actual_model"] is None
    assert result["generation_invoked"] is True
    assert result["generation_provider"] == "gemini"
    assert result["generation_model"] == "gemini-3.5-flash-lite"
    assert result["fallback_reason_code"] == "generation_unavailable"


@pytest.mark.asyncio
async def test_abstention_never_manufactures_evidence_or_cache_eligibility() -> None:
    result = await abstain_node({"retrieval_status": "no_evidence"})
    assert result["fallback_type"] == "no_retrieval_evidence"
    assert result["sources"] == []
    assert result["fallback_cache_eligible"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason_code", "expected_type"),
    [
        ("provider_unavailable", "provider_error"),
        ("out_of_scope", "out_of_scope"),
        ("insufficient_evidence", "no_retrieval_evidence"),
        ("cannot_safely_proceed", "cannot_safely_proceed"),
        ("retrieval_unavailable", "retrieval_error"),
    ],
)
async def test_abstention_preserves_structured_cause(
    reason_code: str,
    expected_type: str,
) -> None:
    result = await abstain_node(
        {
            "fallback_reason_code": reason_code,
            "agent_decision": {"reason_code": "evidence_gap"},
        }
    )
    assert result["fallback_reason_code"] == reason_code
    assert result["fallback_type"] == expected_type


@pytest.mark.asyncio
async def test_out_of_scope_fallback_does_not_claim_missing_provenance() -> None:
    result = await abstain_node(
        {"agent_decision": {"reason_code": "out_of_scope"}}
    )
    assert result["fallback_reason_code"] == "out_of_scope"
    assert "provenance-complete" not in str(result["fallback_reason"]).casefold()


def test_safe_fallback_answers_are_generic_infrastructure_messages() -> None:
    for fallback_type in (
        "no_retrieval_evidence",
        "provider_error",
        "retrieval_error",
        "invalid_generation",
    ):
        answer = build_safe_fallback_answer(fallback_type)
        assert answer
        assert "isotretinoin" not in answer.casefold()
        assert "benzoyl peroxide" not in answer.casefold()
        assert "retrieval" not in answer.casefold()
        assert "truy xuất" not in answer.casefold()
        assert "context" not in answer.casefold()


@pytest.mark.parametrize(
    ("fallback_type", "reason_code"),
    [
        ("no_retrieval_evidence", "insufficient_evidence"),
        ("provider_error", "provider_unavailable"),
        ("retrieval_error", "retrieval_unavailable"),
        ("invalid_generation", "generation_unavailable"),
    ],
)
def test_non_safety_fallbacks_share_the_canonical_user_message(
    fallback_type: str,
    reason_code: str,
) -> None:
    assert (
        build_safe_fallback_answer(fallback_type, reason_code=reason_code)
        == GENERIC_NON_SAFETY_FALLBACK_ANSWER
    )


def test_safety_and_scope_fallbacks_remain_distinct() -> None:
    safety = build_safe_fallback_answer(
        "cannot_safely_proceed",
        reason_code="cannot_safely_proceed",
    )
    out_of_scope = build_safe_fallback_answer("out_of_scope", reason_code="out_of_scope")

    assert safety != GENERIC_NON_SAFETY_FALLBACK_ANSWER
    assert "an toàn" in safety
    assert out_of_scope != GENERIC_NON_SAFETY_FALLBACK_ANSWER
    assert "mụn" in out_of_scope


@pytest.mark.parametrize(
    ("reason_code", "label"),
    [
        ("insufficient_evidence", "Chưa đủ bằng chứng"),
        ("provider_unavailable", "Dịch vụ mô hình tạm thời không khả dụng"),
        ("cannot_safely_proceed", "Không thể tiếp tục an toàn"),
        ("out_of_scope", "Ngoài phạm vi hỗ trợ"),
    ],
)
def test_fallback_reason_labels_do_not_replace_internal_codes(
    reason_code: str,
    label: str,
) -> None:
    assert fallback_reason_label(reason_code) == label


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
    assert get_answer_cache_version({"CACHE_ANSWER_VERSION": "v5"}) == "v10"
