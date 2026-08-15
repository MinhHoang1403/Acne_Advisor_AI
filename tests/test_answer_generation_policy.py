from __future__ import annotations

from src.agent.prompts.medical_answer import (
    MEDICAL_RAG_SYSTEM_PROMPT,
    build_medical_prompt,
    build_medical_system_instruction,
)


def test_static_system_policy_is_not_an_ordinary_medical_answer_key() -> None:
    policy = MEDICAL_RAG_SYSTEM_PROMPT.casefold()
    forbidden_facts = (
        "benzoyl peroxide không phải",
        "clindamycin",
        "adapalene",
        "epiduo",
        "differin",
        "dalacin",
        "doxycycline",
    )
    assert not any(fact in policy for fact in forbidden_facts)
    assert "mọi nội dung y khoa thông thường" in policy
    assert "evidence" in policy


def test_system_instruction_contains_policy_and_shape_not_user_evidence() -> None:
    system = build_medical_system_instruction("A và B khác nhau thế nào?")
    assert MEDICAL_RAG_SYSTEM_PROMPT.strip() in system
    assert "đối chiếu" in system
    assert "<EVIDENCE>" not in system


def test_user_prompt_delimits_question_history_sources_and_evidence_as_data() -> None:
    prompt = build_medical_prompt(
        question="Câu hỏi hiện tại",
        contexts=[],
        conversation_history=[{"role": "user", "content": "lịch sử"}],
        available_sources=[{"source_id": "source-a", "display_name": "Source A"}],
        packed_context_text="canonical evidence",
    )
    assert "<USER_DATA>" in prompt
    assert "<CURRENT_QUESTION>\nCâu hỏi hiện tại" in prompt
    assert "<CONVERSATION_HISTORY>" in prompt
    assert "source_id=source-a" in prompt
    assert "<EVIDENCE>\ncanonical evidence\n</EVIDENCE>" in prompt
    assert MEDICAL_RAG_SYSTEM_PROMPT not in prompt


def test_prompt_contract_has_no_graph_fact_input_channel() -> None:
    prompt = build_medical_prompt(
        question="q",
        contexts=[],
        packed_context_text="source evidence",
    )
    assert "graph_facts" not in prompt
