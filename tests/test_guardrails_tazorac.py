from __future__ import annotations

import asyncio

from src.agent.nodes.guardrails import domain_guard_node


def test_tazorac_questions_are_rule_based_in_domain() -> None:
    questions = [
        "Tazorac chứa hoạt chất gì?",
        "Tazarotene thuộc nhóm thuốc nào?",
        "tazorac co hoat chat gi",
    ]

    for question in questions:
        result = asyncio.run(domain_guard_node({"user_question": question}))

        assert result["is_in_domain"] is True
        assert result["guardrail"] == "in_domain_rule"


def test_safety_constrained_comparison_is_not_misclassified_as_prescription_request() -> None:
    result = asyncio.run(
        domain_guard_node(
            {
                "user_question": (
                    "Isotretinoin và kháng sinh uống: hãy so sánh bối cảnh cần bác sĩ theo dõi, không kê đơn."
                )
            }
        )
    )

    assert result["is_in_domain"] is True
    assert result["guardrail"] == "in_domain_rule"


def test_context_dependent_active_ingredient_followup_stays_in_domain() -> None:
    result = asyncio.run(
        domain_guard_node(
            {
                "user_question": "Nên làm gì với các hoạt chất lúc này?",
                "conversation_history": [
                    {"role": "user", "content": "Tôi đã dùng routine nhiều bước và bị rát."},
                ],
            }
        )
    )

    assert result["is_in_domain"] is True
    assert result["guardrail"] == "in_domain_followup_rule"
