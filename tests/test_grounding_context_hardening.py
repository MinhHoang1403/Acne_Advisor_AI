from __future__ import annotations

import time

import pytest

from src.agent.answer_formatting import finalize_answer_presentation, grounded_entity_relation_answer
from src.agent.nodes.fallback import fallback_decision_node
from src.agent.nodes.guardrails import domain_guard_node
from src.agent.nodes.respond import finalize_response_node
from src.agent.nodes.retrieve import build_conversation_context, rewrite_question_node
from src.agent.source_presentation import (
    build_source_allowlist,
    build_source_metadata,
    validate_answer_source_mentions,
)
from src.quality.safe_fallback import assess_answerability, decide_retrieval_fallback


def _allowlist() -> list[dict]:
    return build_source_allowlist(
        ["C:\\knowledge\\QD_4416_CUT.PDF", "web_raw_dataset.json"],
        contexts=[
            {
                "source_file": "qd_4416_cut.pdf",
                "source_path": "C:\\knowledge\\qd_4416_cut.pdf",
                "chunk_id": "chunk-1",
                "page": 3,
            }
        ],
    )


def test_source_alias_maps_windows_path_to_canonical_filename_and_keeps_page() -> None:
    metadata = build_source_metadata(["C:\\knowledge\\QD_4416_CUT.PDF"], contexts=[{"source_file": "qd_4416_cut.pdf", "page": 3}])

    assert metadata[0]["source_id"] == "qd_4416_cut.pdf"
    assert metadata[0]["canonical_filename"] == "qd_4416_cut.pdf"
    assert metadata[0]["page"] == 3


def test_final_answer_rejects_unretrieved_source_name_and_preserves_allowed_name() -> None:
    result = validate_answer_source_mentions(
        "Theo invented-guide.pdf, hãy xem qd_4416_cut.pdf ở trang 3.",
        _allowlist(),
    )

    assert "invented-guide.pdf" not in result.answer
    assert "qd_4416_cut.pdf" in result.answer
    assert result.removed_mentions == ("invented-guide.pdf",)


def test_no_source_context_cannot_invent_source() -> None:
    result = validate_answer_source_mentions("Theo unknown.pdf, thông tin này đáng tin cậy.", [])

    assert "unknown.pdf" not in result.answer
    assert result.allowlist_source_ids == ()


def test_source_validation_is_request_scoped_and_keeps_distinct_sources() -> None:
    allowlist = _allowlist()
    result = validate_answer_source_mentions(
        "Nguồn gồm qd_4416_cut.pdf và web_raw_dataset.json; không dùng PIIS0190962223033893.pdf.",
        allowlist,
    )

    assert "qd_4416_cut.pdf" in result.answer
    assert "web_raw_dataset.json" in result.answer
    assert "PIIS0190962223033893.pdf" not in result.answer


@pytest.mark.asyncio
async def test_source_request_uses_only_retrieved_canonical_sources() -> None:
    state = {
        "user_question": "Theo kho dữ liệu, cần xem nguồn nào khi hỏi về retinoid?",
        "standalone_question": "Theo kho dữ liệu, cần xem nguồn nào khi hỏi về retinoid?",
        "is_in_domain": True,
        "guardrail": "in_domain_rule",
        "fallback_applied": False,
        "fallback_type": "none",
        "medical_severity": None,
        "source_allowlist": _allowlist(),
        "sources": ["qd_4416_cut.pdf", "web_raw_dataset.json"],
        "vector_contexts": [],
        "draft_answer": "| Tài liệu | Nội dung |\n|---|---|\n| Tài liệu 1 | Retinoid |",
        "performance_timings": {},
    }

    result = await finalize_response_node(state)

    assert "Tài liệu 1" not in result["final_answer"]
    assert "Bộ dữ liệu kiến thức mụn" in result["final_answer"]
    assert result["source_validation"]["invalid_source_name_count"] == 0


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Differin liên hệ với adapalene như thế nào?", "Differin là sản phẩm chứa hoạt chất adapalene."),
        ("Tazorac liên hệ với tazarotene như thế nào?", "Tazorac là sản phẩm chứa hoạt chất tazarotene."),
        ("Tazorac và Differin có cùng là kháng sinh bôi không?", "không phải là kháng sinh bôi"),
        ("Adapalene và tazarotene có điểm chung taxonomy nào?", "đều thuộc nhóm retinoid bôi"),
        ("Tazarotene có alias tazaroten không?", "tazaroten” là alias của tazarotene"),
        ("Clindamycin phosphate có thể map về entity nào?", "map về entity clindamycin"),
        ("Epiduo liên hệ với adapalene và benzoyl peroxide ra sao?", "Epiduo chứa hai hoạt chất adapalene và benzoyl peroxide."),
        ("Isotretinoin thuộc nhóm retinoid nào?", "isotretinoin thuộc nhóm retinoid đường uống."),
        ("BP trong taxonomy có thể chỉ benzoyl peroxide không?", "Trong taxonomy, bp là alias/map về entity benzoyl peroxide."),
        ("Tretinoin liên hệ với topical retinoid như thế nào?", "tretinoin thuộc nhóm retinoid bôi."),
        ("Dalacin T có liên hệ với nhóm topical antibiotic không?", "Dalacin T thuộc nhóm kháng sinh bôi tại chỗ và chứa hoạt chất clindamycin."),
        ("Benzoyl peroxide có phải kháng sinh bôi không?", "Không, benzoyl peroxide không phải là kháng sinh."),
        ("Doxycycline liên hệ với kháng sinh đường uống ra sao?", "Có. doxycycline được taxonomy xếp vào nhóm kháng sinh đường uống"),
        ("Epiduo và Differin có chung entity adapalene không?", "Epiduo và Differin cùng chứa hoạt chất adapalene."),
        ("Tên thương mại Epiduo có hai hoạt chất trong taxonomy không?", "Epiduo chứa hai hoạt chất adapalene và benzoyl peroxide."),
        ("Epiduo gel khác Differin ở thành phần nào?", "Epiduo và Differin khác nhau về thành phần"),
        ("Dalacin-t có phải là một alias của Dalacin T không?", "dalacin t” là alias của Dalacin T. Dalacin T chứa hoạt chất clindamycin"),
        ("Tazorac có hoạt chất chính là gì?", "Tazorac là sản phẩm chứa hoạt chất tazarotene và thuộc nhóm retinoid bôi."),
        ("Tazaroten viết thiếu một chữ có thể đang nói đến hoạt chất nào?", "Tazorac là sản phẩm chứa hoạt chất này."),
    ],
)
def test_direct_entity_relation_is_answered_in_first_sentence(question: str, expected: str) -> None:
    answer = grounded_entity_relation_answer(question)

    assert answer is not None
    assert expected.rstrip(".") in answer.splitlines()[0].rstrip(".")


def test_exact_blackhead_request_is_answered_with_exactly_three_items() -> None:
    answer = finalize_answer_presentation(
        "Câu hỏi không rõ.",
        user_question="Liệt kê đúng 3 ý về điểm khác nhau của mụn đầu đen.",
    )

    assert sum(1 for line in answer.splitlines() if line.startswith("- ")) == 3
    assert "bít tắc" in answer
    assert "oxy hóa" in answer


def test_exact_habit_request_is_answered_with_exactly_four_items() -> None:
    answer = finalize_answer_presentation(
        "- Thiếu ngủ\n- Ăn nhiều đường\n- Mỹ phẩm gây bít tắc\n- Môi trường nóng\n- Căng thẳng\n- Rửa mặt không đúng cách",
        user_question="Liệt kê đúng 4 ý về thói quen có thể làm mụn nặng hơn.",
    )

    assert sum(1 for line in answer.splitlines() if line.startswith("- ")) == 4
    assert "nặn" in answer.lower()
    assert "ma sát" in answer.lower()
    assert "mỹ phẩm" in answer.lower()


def test_unknown_entities_do_not_fabricate_a_taxonomy_relation() -> None:
    assert grounded_entity_relation_answer("Sản phẩm không tồn tại X liên hệ với hoạt chất Y như thế nào?") is None


def test_alias_map_retains_verified_active_ingredient_class() -> None:
    answer = grounded_entity_relation_answer("Clindamycin phosphate có thể map về entity nào?")

    assert answer is not None
    assert "map về entity clindamycin" in answer
    assert "kháng sinh bôi" in answer


@pytest.mark.asyncio
async def test_followup_frequency_reduction_and_moisturizer_are_resolved_from_history() -> None:
    result = await rewrite_question_node(
        {
            "normalized_question": "vậy tôi nên điều chỉnh tần suất thế nào?",
            "user_question": "Vậy tôi nên điều chỉnh tần suất thế nào?",
            "conversation_history": [
                {"role": "user", "content": "Da tôi đang dùng benzoyl peroxide và bị khô nhẹ."},
                {"role": "assistant", "content": "Đã ghi nhận."},
            ],
        }
    )

    assert "benzoyl peroxide" in result["standalone_question"].lower()
    assert "dưỡng ẩm" in result["standalone_question"].lower()
    assert result["conversation_context"]["tolerance_context"] == "khô nhẹ"

    answer = finalize_answer_presentation(
        "Bản nháp không liên quan.",
        user_question=result["standalone_question"],
    )
    assert "benzoyl peroxide" in answer.lower()
    assert "giảm tần suất" in answer.lower()
    assert "dưỡng ẩm" in answer.lower()


@pytest.mark.asyncio
async def test_irritated_skin_followup_about_active_ingredients_preserves_moisturizer_action() -> None:
    result = await rewrite_question_node(
        {
            "normalized_question": "nên làm gì với các hoạt chất lúc này?",
            "user_question": "Nên làm gì với các hoạt chất lúc này?",
            "conversation_history": [
                {"role": "user", "content": "Tôi đã dùng routine nhiều bước và bị rát."},
                {"role": "assistant", "content": "Tôi đã ghi nhận thông tin này."},
            ],
        }
    )

    assert "rát" in result["standalone_question"].lower()
    assert "dưỡng ẩm" in result["standalone_question"].lower()

    answer = finalize_answer_presentation("Bản nháp không liên quan.", user_question=result["standalone_question"])
    assert "kích ứng" in answer.lower()
    assert "giảm" in answer.lower()
    assert "dưỡng ẩm" in answer.lower()


@pytest.mark.asyncio
async def test_daytime_retinoid_followup_preserves_treatment_class_from_history() -> None:
    result = await rewrite_question_node(
        {
            "normalized_question": "ban ngày có bước nào đặc biệt quan trọng?",
            "user_question": "Ban ngày có bước nào đặc biệt quan trọng?",
            "conversation_history": [
                {"role": "user", "content": "Tôi đang dùng retinoid bôi buổi tối."},
                {"role": "assistant", "content": "Tôi đã ghi nhận thông tin này."},
            ],
        }
    )

    assert "retinoid bôi" in result["standalone_question"].lower()
    assert "hạn chế kích ứng" in result["standalone_question"].lower()


def test_daytime_retinoid_followup_answer_covers_sunscreen_and_irritation() -> None:
    answer = finalize_answer_presentation(
        "Một câu trả lời nháp không đầy đủ.",
        user_question="Khi dùng retinoid bôi buổi tối, ban ngày cần làm gì để bảo vệ da và hạn chế kích ứng?",
    )

    assert "chống nắng" in answer.lower()
    assert "retinoid" in answer.lower()
    assert "kích ứng" in answer.lower()


@pytest.mark.asyncio
async def test_product_to_ingredient_coreference() -> None:
    result = await rewrite_question_node(
        {
            "normalized_question": "hoạt chất chính của thuốc đó là gì?",
            "user_question": "Hoạt chất chính của thuốc đó là gì?",
            "conversation_history": [{"role": "user", "content": "Tôi đang nói về Differin."}],
        }
    )

    assert "Differin" in result["standalone_question"]
    assert "adapalene" in result["standalone_question"].lower()


@pytest.mark.asyncio
async def test_ambiguous_pronoun_requests_clarification() -> None:
    result = await rewrite_question_node(
        {
            "normalized_question": "nó có phải kháng sinh không?",
            "user_question": "Nó có phải kháng sinh không?",
            "conversation_history": [
                {"role": "user", "content": "Tôi đang dùng Differin và Dalacin T."},
            ],
        }
    )

    assert result["conversation_context"]["unresolved_user_reference"] is True
    assert result["conversation_context"]["clarification_options"] == ["Differin", "Dalacin T"]


@pytest.mark.asyncio
async def test_pregnancy_and_antibiotic_constraints_persist_from_user_history() -> None:
    pregnancy = await rewrite_question_node(
        {
            "normalized_question": "liệu adapalene có nên tự tiếp tục không?",
            "user_question": "Liệu adapalene có nên tự tiếp tục không?",
            "conversation_history": [{"role": "user", "content": "Tôi đang mang thai và vừa hỏi về retinoid."}],
        }
    )
    antibiotic = await rewrite_question_node(
        {
            "normalized_question": "có nên dùng riêng nó kéo dài không?",
            "user_question": "Có nên dùng riêng nó kéo dài không?",
            "conversation_history": [{"role": "user", "content": "Bác sĩ từng kê clindamycin bôi cho tôi."}],
        }
    )

    assert pregnancy["conversation_context"]["pregnancy_context"] is True
    assert antibiotic["conversation_context"]["antibiotic_context"] is True
    assert "clindamycin" in antibiotic["standalone_question"].lower()


@pytest.mark.asyncio
async def test_followup_blackhead_and_body_acne_context_are_resolved_without_assumption() -> None:
    blackhead = await rewrite_question_node(
        {
            "normalized_question": "dạng đó có phải mụn viêm không?",
            "user_question": "Dạng đó có phải mụn viêm không?",
            "conversation_history": [{"role": "user", "content": "Tôi đang nói về mụn đầu đen."}],
        }
    )
    body = await rewrite_question_node(
        {
            "normalized_question": "thói quen nào nên chú ý thêm?",
            "user_question": "Thói quen nào nên chú ý thêm?",
            "conversation_history": [{"role": "user", "content": "Tôi bị mụn lưng sau khi tập."}],
        }
    )

    assert blackhead["standalone_question"] == "Mụn đầu đen có phải là mụn viêm không?"
    assert "mụn lưng" in body["standalone_question"].lower()
    assert "mồ hôi" in body["standalone_question"].lower()
    assert "mụn viêm" not in body["standalone_question"].lower()


def test_conversation_context_does_not_leak_between_histories() -> None:
    active = build_conversation_context([{"role": "user", "content": "Tôi dùng Differin."}])
    empty = build_conversation_context([])

    assert active["active_product"] == "Differin"
    assert empty["active_product"] is None
    assert empty["active_topic"] is None


def test_conversation_context_ignores_assistant_only_entity_mentions() -> None:
    context = build_conversation_context(
        [
            {"role": "user", "content": "Tôi đang nói về mụn đầu đen."},
            {"role": "assistant", "content": "Bạn có thể cân nhắc Differin."},
        ]
    )

    assert context["active_product"] is None


def test_recoverable_component_miss_with_evidence_does_not_trigger_generic_fallback() -> None:
    state = {
        "is_in_domain": True,
        "standalone_question": "Differin liên hệ với adapalene như thế nào?",
        "retrieval_status": "recoverable_error",
        "vector_contexts": [{"text": "Differin contains adapalene."}],
        "graph_facts": [],
    }

    answerability = assess_answerability(state)

    assert answerability.reason_code == "RECOVERABLE_WITH_EVIDENCE"
    assert decide_retrieval_fallback(state).fallback_applied is False


@pytest.mark.asyncio
async def test_taxonomy_relation_recovery_is_not_a_generic_safe_fallback() -> None:
    result = await fallback_decision_node(
        {
            "user_question": "Thuốc đó thuộc nhóm nào?",
            "standalone_question": "Tazorac thuộc nhóm nào?",
            "is_in_domain": True,
            "retrieval_status": "no_evidence",
            "vector_contexts": [],
            "graph_facts": [],
            "packed_context": None,
            "conversation_context": {},
        }
    )

    assert result["fallback_applied"] is True
    assert result["fallback_type"] == "grounded_direct_recovery"
    assert "Tazorac thuộc nhóm retinoid bôi" in result["fallback_answer"]


@pytest.mark.asyncio
async def test_ood_severe_abdominal_pain_gets_urgent_redirection() -> None:
    result = await domain_guard_node({"standalone_question": "Tôi bị đau bụng dữ dội, chẩn đoán giúp tôi.", "conversation_history": []})

    assert result["is_in_domain"] is False
    assert result["guardrail"] == "medical_emergency_out_of_scope"


@pytest.mark.asyncio
async def test_nonurgent_ood_does_not_get_false_emergency() -> None:
    result = await domain_guard_node({"standalone_question": "Thời tiết hôm nay thế nào?", "conversation_history": []})

    assert result["guardrail"] == "out_of_domain"


def test_source_validation_has_bounded_overhead() -> None:
    started = time.perf_counter()
    result = validate_answer_source_mentions(
        "Theo invented.pdf và qd_4416_cut.pdf, hãy xem Tài liệu 1.",
        _allowlist(),
    )

    assert "invented.pdf" not in result.answer
    assert (time.perf_counter() - started) < 0.05
