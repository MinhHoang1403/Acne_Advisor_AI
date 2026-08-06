"""
Tests for Phase 2 answer-generation policy without calling external LLMs.
"""

from __future__ import annotations

import pytest

from src.agent.graph import route_after_guard
from src.agent.nodes.respond import finalize_response_node
from src.agent.nodes.reason import _select_answer_contexts, filter_graph_facts_for_prompt
from src.agent.nodes.retrieve import rewrite_question_node
from src.agent.prompts.medical_answer import MEDICAL_RAG_SYSTEM_PROMPT


def test_medical_prompt_contains_required_medication_rules():
    prompt = MEDICAL_RAG_SYSTEM_PROMPT.lower()

    assert "benzoyl peroxide is not an antibiotic" in prompt
    assert "không khuyến cáo dùng kháng sinh bôi đơn trị liệu" in prompt
    assert "không kê liều cụ thể" in prompt
    assert "combined oral contraceptives" in prompt
    assert "spironolactone" in prompt
    assert "cần bác sĩ kê đơn/đánh giá" in prompt


def test_medical_prompt_contains_direct_answer_and_entity_preservation_rules():
    prompt = MEDICAL_RAG_SYSTEM_PROMPT.lower()

    assert "direct answer first" in prompt
    assert "primary entity preservation" in prompt
    assert "benzoyl peroxide không phải là kháng sinh" in prompt
    assert "không được viết \"có, ... không nên\"" in prompt
    assert "câu trả lời phải cover đầy đủ cả a và b" in prompt
    assert "ưu tiên bảng hoặc bullet đối chiếu" in prompt
    assert "phải trả lời từng ý hoặc từng câu hỏi con" in prompt
    assert "phải xử lý từng thuốc người dùng nêu" in prompt
    assert "header bảng phải giữ đủ các cột" in prompt
    assert "ý nghĩ tự làm hại bản thân" in prompt


def test_graph_fact_filter_removes_document_code_and_empty_fact():
    facts = [
        {
            "entity": "ng198",
            "relationship": "TREATS",
            "related_entity": "acne",
            "description": "guideline code",
            "evidence": "citation",
        },
        {
            "entity": "benzoyl peroxide",
            "relationship": "TREATS",
            "related_entity": "mụn trứng cá",
            "description": "",
            "evidence": "",
        },
        {
            "entity": "benzoyl peroxide",
            "relationship": "TREATS",
            "related_entity": "mụn trứng cá",
            "description": "Hoạt chất bôi trị mụn.",
            "evidence": "AAD recommendation",
        },
    ]

    filtered = filter_graph_facts_for_prompt(
        query="benzoyl peroxide trị mụn",
        contexts=[{"text": "benzoyl peroxide là hoạt chất bôi trị mụn.", "graph_nodes": ["benzoyl peroxide"]}],
        graph_facts=facts,
    )

    assert len(filtered) == 1
    assert filtered[0]["entity"] == "benzoyl peroxide"
    assert filtered[0]["evidence"] == "AAD recommendation"


def test_graph_fact_filter_removes_mechanism_treats_noise_without_context_support():
    facts = [
        {
            "entity": "C. acnes",
            "relationship": "TREATS",
            "related_entity": "mụn trứng cá",
            "description": "Noisy graph edge.",
            "evidence": "citation",
        }
    ]

    filtered = filter_graph_facts_for_prompt(
        query="cơ chế mụn",
        contexts=[{"text": "Mụn liên quan đến bít tắc nang lông và tăng tiết bã."}],
        graph_facts=facts,
    )

    assert filtered == []


def test_context_selection_prefers_clinical_chunk_over_abbreviation():
    contexts = [
        {
            "header": "Abbreviations",
            "text": "BP: benzoyl peroxide",
            "score": 0.99,
        },
        {
            "header": "Treatment recommendations",
            "text": "Benzoyl peroxide may be used as an acne treatment and can cause irritation.",
            "score": 0.75,
        },
    ]

    selected = _select_answer_contexts(contexts, limit=1)

    assert selected[0]["header"] == "Treatment recommendations"


def test_context_selection_prefers_bp_not_antibiotic_chunk_over_oral_antibiotics():
    contexts = [
        {
            "header": "Oral antibiotics",
            "text": "Oral antibiotics such as doxycycline may be used for acne in selected cases.",
            "score": 0.95,
        },
        {
            "header": "Benzoyl peroxide (BP)",
            "text": "Benzoyl peroxide (BP) does not contain antibiotics and may reduce antibiotic resistance.",
            "score": 0.70,
        },
    ]

    selected = _select_answer_contexts(
        contexts,
        limit=1,
        query="Benzoyl peroxide có phải kháng sinh không?",
    )

    assert selected[0]["header"] == "Benzoyl peroxide (BP)"


@pytest.mark.asyncio
async def test_finalize_bp_antibiotic_question_direct_answer_first():
    result = await finalize_response_node(
        {
            "user_question": "Benzoyl peroxide có phải kháng sinh không?",
            "draft_answer": "Không nên tự uống kháng sinh để trị mụn.",
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"]
    assert answer.lower().startswith("không, benzoyl peroxide không phải là kháng sinh.")
    assert "benzoyl peroxide không phải là kháng sinh" in answer.lower()
    assert "Không nên tự uống kháng sinh" not in answer.splitlines()[0]


@pytest.mark.asyncio
async def test_finalize_clindamycin_monotherapy_polarity_is_no():
    result = await finalize_response_node(
        {
            "user_question": "Có nên dùng clindamycin đơn độc để trị mụn không?",
            "draft_answer": "Có, clindamycin không nên được dùng đơn độc để trị mụn theo tài liệu hiện có.",
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"]
    assert answer.startswith("Không. Clindamycin không nên")
    assert "Có, clindamycin không nên" not in answer
    assert "**Tóm tắt ngắn**" not in answer


@pytest.mark.asyncio
async def test_finalize_antibiotic_multi_intent_does_not_collapse_to_bp_identity():
    result = await finalize_response_node(
        {
            "user_question": (
                "Vì sao không nên dùng clindamycin bôi hoặc kháng sinh uống đơn độc để điều trị mụn kéo dài? "
                "Benzoyl peroxide có vai trò gì khi phối hợp với kháng sinh?"
            ),
            "draft_answer": "Không, benzoyl peroxide không phải là kháng sinh.",
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"].lower()
    assert answer.startswith("không nên dùng clindamycin")
    assert "clindamycin" in answer
    assert "kháng sinh uống" in answer
    assert "đơn độc" in answer
    assert "kéo dài" in answer
    assert "kháng kháng sinh" in answer
    assert "benzoyl peroxide" in answer
    assert "tăng hiệu quả" in answer
    assert not answer.startswith("không, benzoyl peroxide không phải")


@pytest.mark.asyncio
async def test_finalize_multi_entity_pregnancy_safety_covers_all_named_meds():
    result = await finalize_response_node(
        {
            "user_question": (
                "Tôi đang có thai và hiện dùng adapalene, tazarotene và doxycycline để trị mụn. "
                "Tôi nên làm gì?"
            ),
            "draft_answer": "Nên ngừng adapalene và hỏi bác sĩ.",
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"]
    lowered = answer.lower()
    assert "adapalene" in lowered
    assert "tazarotene" in lowered
    assert "doxycycline" in lowered
    assert "## Với từng thuốc" in answer
    assert "Không tự tăng liều" in answer
    assert "bác sĩ sản khoa" in answer
    assert "có thể tiếp tục" not in lowered


@pytest.mark.asyncio
async def test_finalize_adapalene_bp_comparison_covers_both_entities():
    result = await finalize_response_node(
        {
            "user_question": "Adapalene và benzoyl peroxide khác nhau thế nào?",
            "draft_answer": "Adapalene là retinoid bôi.",
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"].lower()
    assert "adapalene" in answer
    assert "benzoyl peroxide" in answer
    assert "| tiêu chí | adapalene | benzoyl peroxide |" in answer
    assert "retinoid bôi" in answer
    assert "không phải kháng sinh" in answer
    assert "có thể được phối hợp" in answer
    assert "bạc màu" in answer
    assert "**khi nào nên gặp bác sĩ**" not in answer


@pytest.mark.asyncio
async def test_finalize_self_harm_question_prioritizes_crisis_action():
    result = await finalize_response_node(
        {
            "user_question": (
                "Mụn của tôi không quá nặng nhưng khiến tôi né tránh mọi người, mất ngủ và gần đây "
                "tôi đã nghĩ đến việc tự làm hại bản thân. Tôi nên làm gì?"
            ),
            "draft_answer": "Bạn nên chăm sóc da dịu nhẹ và gặp bác sĩ da liễu.",
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"].lower()
    assert answer.startswith("điều quan trọng nhất lúc này là an toàn")
    assert "cấp cứu" in answer
    assert "người đáng tin cậy" in answer
    assert answer.index("an toàn") < answer.index("điều trị mụn")
    assert "chẩn đoán tình trạng tâm thần" in answer


@pytest.mark.asyncio
async def test_finalize_acne_fulminans_like_question_keeps_urgency():
    result = await finalize_response_node(
        {
            "user_question": (
                "Một nam thiếu niên đột ngột xuất hiện nhiều cục và nang viêm lớn, trợt loét, "
                "đóng vảy xuất huyết, kèm sốt và đau khớp. Đây có thể là tình trạng gì và cần xử trí với mức độ khẩn cấp ra sao?"
            ),
            "draft_answer": "Đây là mụn nặng, nên khám khi có thời gian.",
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"].lower()
    assert "nghi acne fulminans" in answer
    assert "không thể chẩn đoán chắc chắn" in answer
    assert "trong ngày" in answer
    assert "24 giờ" in answer


@pytest.mark.asyncio
async def test_finalize_dedupes_duplicate_section_headers():
    result = await finalize_response_node(
        {
            "user_question": "Mụn viêm khi nào nên gặp bác sĩ?",
            "draft_answer": (
                "**Tóm tắt ngắn**\nNên gặp bác sĩ nếu mụn nặng.\n\n"
                "**Khi nào nên gặp bác sĩ**\nNếu mụn đau nhiều.\n\n"
                "**Khi nào nên gặp bác sĩ**\nNếu mụn kéo dài.\n\n"
                "**Lưu ý**\n"
            ),
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    assert result["final_answer"].count("**Khi nào nên gặp bác sĩ**") == 0
    assert result["final_answer"].count("Nếu mụn đau nhiều.") == 1


@pytest.mark.asyncio
async def test_finalize_antibiotic_review_question_adds_one_actionable_stewardship_note():
    result = await finalize_response_node(
        {
            "user_question": "Dấu hiệu nào cho thấy cần đánh giá lại kế hoạch kháng sinh trị mụn?",
            "draft_answer": "- Mụn không cải thiện sau một liệu trình phù hợp.\n- Mụn tái phát hoặc nặng hơn.",
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"].lower()
    assert "không tự bắt đầu, kéo dài, ngừng hoặc đổi kháng sinh" in answer
    assert "bác sĩ da liễu" in answer
    assert answer.count("không tự bắt đầu, kéo dài, ngừng hoặc đổi kháng sinh") == 1


@pytest.mark.asyncio
async def test_finalize_bpo_clindamycin_comparison_resolves_bpo_alias_and_uses_table():
    result = await finalize_response_node(
        {
            "user_question": "BPO khác clindamycin ở chỗ nào và có thể phối hợp không?",
            "draft_answer": "Clindamycin có thể điều trị mụn.",
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"].lower()
    assert "| tiêu chí | benzoyl peroxide | clindamycin |" in answer
    assert "không phải kháng sinh" in answer
    assert "kháng kháng sinh" in answer


@pytest.mark.asyncio
async def test_finalize_epiduo_dalacin_comparison_keeps_both_products_and_stewardship():
    result = await finalize_response_node(
        {
            "user_question": "Epiduo khác Dalacin T ở hoạt chất và kháng kháng sinh ra sao?",
            "draft_answer": "Epiduo có adapalene.",
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"].lower()
    assert "| tiêu chí | epiduo | dalacin t |" in answer
    assert "adapalene + benzoyl peroxide" in answer
    assert "clindamycin" in answer
    assert "kháng kháng sinh" in answer


@pytest.mark.asyncio
async def test_finalize_retinoid_bpo_comparison_uses_complete_table():
    result = await finalize_response_node(
        {
            "user_question": "Retinoid bôi và benzoyl peroxide khác nhau theo cơ chế nào?",
            "draft_answer": "Retinoid có thể điều trị mụn.",
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"].lower()
    assert "| tiêu chí | retinoid bôi | benzoyl peroxide |" in answer
    assert "không phải kháng sinh" in answer


@pytest.mark.asyncio
async def test_finalize_boolean_cardinality_answer_starts_with_clear_negative():
    result = await finalize_response_node(
        {
            "user_question": "Epiduo có phải chỉ chứa một hoạt chất không?",
            "draft_answer": "Epiduo chứa hai hoạt chất chính là adapalene và benzoyl peroxide.",
            "is_in_domain": True,
        }
    )

    assert result["final_answer"].startswith("Không. Epiduo chứa hai hoạt chất")


@pytest.mark.asyncio
async def test_finalize_tazarotene_adapalene_comparison_handles_which_drug_question():
    result = await finalize_response_node(
        {
            "user_question": "Tazarotene và adapalene: thuốc nào thuộc nhóm gì và cần lưu ý gì về kích ứng?",
            "draft_answer": "Cả hai có thể dùng cho mụn.",
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"].lower()
    assert "| tiêu chí | tazarotene | adapalene |" in answer
    assert "retinoid bôi" in answer
    assert "kích ứng" in answer


@pytest.mark.asyncio
async def test_finalize_retinoid_bpo_comparison_honors_doi_chieu_table_request():
    result = await finalize_response_node(
        {
            "user_question": "Retinoid bôi và benzoyl peroxide tác động vào mụn theo hai cơ chế gì? Hãy đối chiếu bằng bảng.",
            "draft_answer": "Hai hoạt chất có tác dụng khác nhau.",
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"].lower()
    assert "| tiêu chí | retinoid bôi | benzoyl peroxide |" in answer
    assert "không phải kháng sinh" in answer


@pytest.mark.asyncio
async def test_finalize_morning_evening_routine_comparison_uses_safe_table():
    result = await finalize_response_node(
        {
            "user_question": "Routine trị mụn buổi sáng và buổi tối nên khác nhau ở những bước nào? Lập bảng giúp tôi.",
            "draft_answer": "Bạn nên chăm sóc da phù hợp.",
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"].lower()
    assert "| bước | buổi sáng | buổi tối |" in answer
    assert "chống nắng" in answer
    assert "không thêm nhiều hoạt chất mạnh cùng lúc" in answer


@pytest.mark.asyncio
async def test_finalize_back_and_face_acne_comparison_keeps_both_locations_in_table():
    result = await finalize_response_node(
        {
            "user_question": "Mụn ở lưng khác mụn ở mặt ở vị trí chăm sóc và dấu hiệu cần khám như thế nào?",
            "draft_answer": "Cần chăm sóc phù hợp.",
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"].lower()
    assert "| tiêu chí | mụn ở mặt | mụn ở lưng |" in answer
    assert "khi cần khám" in answer


@pytest.mark.asyncio
async def test_rewrite_preserves_explicit_primary_entity_with_history():
    result = await rewrite_question_node(
        {
            "user_question": "Benzoyl peroxide có phải kháng sinh không?",
            "normalized_question": "benzoyl peroxide có phải kháng sinh không?",
            "conversation_history": [{"role": "user", "content": "Tôi bị mụn viêm."}],
        }
    )

    assert result["standalone_question"] == "benzoyl peroxide có phải kháng sinh không?"
    assert result["use_history_context"] is False


def test_out_of_domain_guard_routes_to_finalize_without_retrieval():
    assert route_after_guard({"is_in_domain": False}) == "finalize"
