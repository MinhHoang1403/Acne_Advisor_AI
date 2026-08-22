from __future__ import annotations

import json

import pytest

from src.agent import action_decision as decision_module
from src.agent.action_decision import select_agent_action
from src.agent.nodes import workflow
from src.agent.prompts.medical_answer import build_medical_system_instruction


def _evidence_state(question: str) -> dict:
    return {
        "normalized_question": question,
        "retrieval_attempt": 1,
        "retrieval_status": "ok",
        "evidence_assessment": {"usable": True},
        "vector_contexts": [
            {
                "id": "evidence-1",
                "source_id": "synthetic-guideline",
                "text": (
                    "Treatment A is available at 5% and can improve inflammatory lesions."
                ),
            }
        ],
    }


async def _mock_decision(
    monkeypatch: pytest.MonkeyPatch,
    *,
    action: str,
    reason_code: str,
    direct_supporting_evidence_ids: list[str] | None,
    missing_evidence: str | None = None,
) -> None:
    async def fake_generate(**_: object) -> dict:
        return {
            "text": json.dumps(
                {
                    "action": action,
                    "retrieval_query": None,
                    "missing_evidence": missing_evidence,
                    "reason_code": reason_code,
                    "direct_supporting_evidence_ids": direct_supporting_evidence_ids,
                }
            ),
            "provider": "test",
            "model": "decision-model",
            "fallback_used": False,
        }

    monkeypatch.setattr(decision_module, "generate_llm_response", fake_generate)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "Liệu gel A có đem lại cùng một kết quả lâu dài cho từng người sử dụng?",
        "Nhiệt độ chính xác nào khiến sản phẩm A mất hoàn toàn tác dụng?",
        "Mức chênh hiệu quả chính xác giữa liệu pháp A và B là bao nhiêu?",
        "Có trường hợp nào kết quả điều trị A không duy trì suốt đời hay không?",
    ],
)
async def test_topic_evidence_without_direct_proposition_support_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
) -> None:
    await _mock_decision(
        monkeypatch,
        action="generate",
        reason_code="evidence_sufficient",
        direct_supporting_evidence_ids=[],
    )

    result = await select_agent_action(_evidence_state(question))

    assert result["next_action"] == "abstain"
    assert result["agent_decision"]["reason_code"] == "evidence_gap"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "Nồng độ được nêu trực tiếp cho hoạt chất A là bao nhiêu?",
        "Hoạt chất A có vai trò gì trong chăm sóc da mụn?",
    ],
)
async def test_directly_supported_proposition_can_generate(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
) -> None:
    await _mock_decision(
        monkeypatch,
        action="generate",
        reason_code="evidence_sufficient",
        direct_supporting_evidence_ids=["evidence-1"],
    )

    result = await select_agent_action(_evidence_state(question))

    assert result["next_action"] == "generate"
    assert result["agent_decision"]["reason_code"] == "evidence_sufficient"
    assert result["agent_decision"]["direct_supporting_evidence_ids"] == [
        "evidence-1"
    ]


@pytest.mark.asyncio
async def test_generate_rejects_support_identifier_not_visible_to_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _mock_decision(
        monkeypatch,
        action="generate",
        reason_code="evidence_sufficient",
        direct_supporting_evidence_ids=["invented-evidence"],
    )

    result = await select_agent_action(_evidence_state("Hoạt chất A có vai trò gì?"))

    assert result["next_action"] == "abstain"
    assert result["agent_decision"]["reason_code"] == "evidence_gap"


@pytest.mark.asyncio
async def test_evidence_gap_can_keep_grounded_context_without_changing_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    async def fake_generate(state: dict) -> dict:
        captured.update(state)
        return {
            "draft_answer": (
                "Bằng chứng hiện có không xác lập kết luận được hỏi. "
                "Nguồn vẫn cho biết liệu pháp A có thể hỗ trợ tổn thương viêm."
            ),
            "actual_provider": "test",
            "actual_model": "answer-model",
        }

    monkeypatch.setattr(workflow, "generate_answer_node", fake_generate)
    state = {
        **_evidence_state("Liệu pháp A có đem lại kết quả giống nhau cho tất cả?"),
        "agent_decision": {
            "action": "abstain",
            "reason_code": "evidence_gap",
        },
    }

    result = await workflow.abstain_node(state)

    assert captured["response_contract"] == "evidence_gap_with_related_context"
    assert result["fallback_applied"] is False
    assert result["fallback_cache_eligible"] is False
    assert result["generation_invoked"] is True
    assert "liệu pháp A" in result["draft_answer"]
    assert state["agent_decision"]["action"] == "abstain"


def test_evidence_gap_generation_instruction_preserves_epistemic_boundary() -> None:
    instruction = build_medical_system_instruction(
        "Câu hỏi synthetic",
        response_contract="evidence_gap_with_related_context",
    )

    assert "action=abstain" in instruction
    assert "reason=evidence_gap" in instruction
    assert "không xác lập" in instruction
    assert "không đồng nghĩa mệnh đề đó đã được chứng minh là sai" in instruction
    assert "thông tin liên quan" in instruction
