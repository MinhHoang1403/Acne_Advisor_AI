from __future__ import annotations

import pytest

from src.agent.answer_formatting import (
    ANSWER_FORMATTING_CONTRACT,
    ANSWER_FORMATTING_CONTRACT_VERSION,
    assess_structural_quality,
    answer_format_instruction_for_question,
    finalize_answer_presentation,
    normalize_answer_markdown,
    repair_terminal_punctuation,
    strip_leading_question_echo,
)
from src.agent.nodes import reason as reason_node
from src.agent.nodes.respond import finalize_response_node
from src.agent.prompts.medical_answer import build_medical_prompt, observe_medical_prompt_budget
from src.agent.source_presentation import build_source_metadata, display_names_for_sources
from src.quality.safe_fallback import build_safe_fallback_answer


def test_formatting_contract_avoids_mandatory_boilerplate_sections():
    prompt = build_medical_prompt(
        question="Differin thuộc nhóm thuốc gì?",
        symptoms=[],
        safety_flags=[],
        contexts=[{"text": "Differin contains adapalene, a topical retinoid.", "source_file": "doc.pdf"}],
        graph_facts=[],
    )

    assert ANSWER_FORMATTING_CONTRACT_VERSION == "answer_formatting_contract_v11"
    assert ANSWER_FORMATTING_CONTRACT in prompt
    assert "ANSWER PRESENTATION CONTRACT V11" in prompt
    assert "provider không được quyết định format" in prompt
    assert "Multi-intent" in prompt
    assert "Structured request" in prompt
    assert "Signs/symptoms vs causes" in prompt
    assert "Emergency immediate action" in prompt
    assert "3-5 đoạn ngắn gồm tóm tắt" not in prompt
    assert "Mục **Lưu ý** phải có câu" not in prompt


def test_comparison_prompt_requires_complete_entity_coverage_and_table_format():
    instruction = answer_format_instruction_for_question(
        "Mụn đầu đen và mụn đầu trắng khác nhau như thế nào?"
    )

    assert "CÂU SO SÁNH" in instruction
    assert "cover đầy đủ từng entity" in instruction
    assert "bảng Markdown GFM" in instruction


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Differin thuộc nhóm thuốc gì?", "CÂU YES/NO HOẶC ĐỊNH DANH"),
        ("Epiduo có BPO không?", "CÂU YES/NO HOẶC ĐỊNH DANH"),
        ("Tôi đang mang thai, có dùng adapalene được không?", "CÂU YES/NO HOẶC ĐỊNH DANH"),
        ("Tôi có nhiều cục mụn sâu, đau và bắt đầu để lại sẹo.", "CÂU AN TOÀN"),
    ],
)
def test_intent_specific_formatting_hints_cover_smoke_cases(question: str, expected: str):
    assert expected in answer_format_instruction_for_question(question)


def test_markdown_cleanup_removes_blank_lines_inside_tables_and_empty_headings():
    messy = (
        "| A | B |\n"
        "|---|---|\n"
        "\n"
        "| one | two |\n\n"
        "**Khi nào nên gặp bác sĩ**\n\n"
        "**Lưu ý**\n"
        "Thông tin mang tính tham khảo."
    )

    cleaned = normalize_answer_markdown(messy)

    assert "|---|---|\n| one | two |" in cleaned
    assert "**Khi nào nên gặp bác sĩ**" not in cleaned
    assert "**Lưu ý**" in cleaned


def test_markdown_cleanup_removes_leading_english_yes_marker():
    cleaned = normalize_answer_markdown("**Yes** Mụn đầu đen là một dạng mụn nhân mở.")

    assert cleaned == "Mụn đầu đen là một dạng mụn nhân mở."


@pytest.mark.asyncio
async def test_finalize_does_not_wrap_short_valid_answer_in_generic_template():
    result = await finalize_response_node(
        {
            "user_question": "Differin thuộc nhóm thuốc gì?",
            "draft_answer": "Differin chứa adapalene và thuộc nhóm retinoid bôi.",
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"]
    assert answer.startswith("Differin thuộc nhóm retinoid bôi")
    assert "**Tóm tắt ngắn**" not in answer
    assert "Thông tin trả lời dựa trên phần tài liệu y khoa" not in answer
    assert "Thông tin mang tính tham khảo" not in answer


@pytest.mark.asyncio
async def test_finalize_blackhead_whitehead_comparison_uses_table():
    result = await finalize_response_node(
        {
            "user_question": "Mụn đầu đen và mụn đầu trắng khác nhau như thế nào?",
            "draft_answer": "Mụn đầu đen là nhân mụn mở. Mụn đầu trắng là nhân mụn đóng.",
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"]
    assert answer.startswith("Mụn đầu đen và mụn đầu trắng")
    assert "| Tiêu chí | Mụn đầu đen | Mụn đầu trắng |" in answer
    assert "**Tóm tắt ngắn**" not in answer


@pytest.mark.asyncio
async def test_finalize_differin_identity_removes_yes_prefix_and_stays_concise():
    result = await finalize_response_node(
        {
            "user_question": "Differin thuộc nhóm thuốc gì?",
            "draft_answer": "Có, Differin thuộc nhóm topical_retinoid.",
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"]
    assert answer.startswith("Differin thuộc nhóm retinoid bôi")
    assert "Có, Differin" not in answer
    assert "adapalene" in answer
    assert "**Tóm tắt ngắn**" not in answer


@pytest.mark.asyncio
async def test_finalize_epiduo_composition_covers_both_ingredients():
    result = await finalize_response_node(
        {
            "user_question": "Epiduo có BPO không?",
            "draft_answer": "Có. Epiduo có benzoyl peroxide.",
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"]
    assert answer.startswith("Có. Epiduo chứa")
    assert "adapalene" in answer
    assert "benzoyl peroxide" in answer
    assert "không phải kháng sinh" in answer


@pytest.mark.asyncio
async def test_finalize_retinoid_shared_class_repairs_wrong_negative_generation():
    result = await finalize_response_node(
        {
            "user_question": "Adapalene, tretinoin và isotretinoin có cùng nhóm thuốc không?",
            "draft_answer": (
                "Không, adapalene, tretinoin và isotretinoin không thuộc cùng một nhóm thuốc. "
                "Sự khác biệt giữa các hoạt chất này rất lớn."
            ),
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"]
    assert answer.startswith("Có. Các hoạt chất này đều thuộc nhóm retinoid")
    assert "khác đường dùng" in answer
    assert "không thuộc cùng một nhóm" not in answer
    assert "| Hoạt chất | Nhóm chung | Điểm cần phân biệt |" in answer


@pytest.mark.asyncio
async def test_finalize_pregnancy_safety_preserves_warning():
    result = await finalize_response_node(
        {
            "user_question": "Đang mang thai có dùng adapalene được không?",
            "draft_answer": "Có thể dùng adapalene nếu muốn.",
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"]
    first_line = answer.splitlines()[0]
    assert "adapalene" in first_line
    assert "tránh hoặc ngừng" in first_line
    assert "nếu chưa được bác sĩ" not in first_line
    assert "bác sĩ da liễu hoặc bác sĩ sản khoa" in answer
    assert "Cảnh báo nghiêm trọng" not in answer
    assert "cho con bú" not in answer
    assert answer.count("thai kỳ") == 2
    assert "- Tạm ngưng" in answer
    assert "Thông tin mang tính tham khảo" not in answer


@pytest.mark.asyncio
async def test_finalize_severe_acne_keeps_escalation():
    result = await finalize_response_node(
        {
            "user_question": "Tôi có nhiều cục mụn sâu, đau và đang để lại sẹo.",
            "draft_answer": (
                "Mụn cục sâu, đau và để lại sẹo là dấu hiệu cần được bác sĩ da liễu đánh giá sớm.\n\n"
                "Không tự nặn hoặc tự dùng isotretinoin/kháng sinh uống khi chưa được kê đơn."
            ),
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"]
    assert "bác sĩ da liễu" in answer
    assert "sẹo" in answer
    assert "## Việc nên làm" in answer
    assert "- Không nặn" in answer
    assert "## Trong lúc chờ khám" in answer
    assert "- Giữ routine" in answer
    assert "Thông tin trả lời dựa trên phần tài liệu y khoa" not in answer


def test_safe_fallback_format_is_short_and_not_cache_boilerplate():
    answer = build_safe_fallback_answer("insufficient_context")

    assert "Tài liệu hiện có chưa đủ bằng chứng" in answer
    assert answer.count("**Tóm tắt ngắn**") == 1
    assert "Thông tin trả lời dựa trên phần tài liệu y khoa" not in answer
    assert len(answer.split()) < 90


def test_strip_leading_question_echo_exact_and_partial():
    question = "Tôi có nhiều cục mụn sâu, đau, sưng đỏ và bắt đầu để lại sẹo. Tôi nên làm gì?"

    exact = strip_leading_question_echo(question + "\n\nBạn nên khám da liễu sớm.", question)
    partial = strip_leading_question_echo("Tôi nên làm gì?\nBạn nên khám da liễu sớm.", question)

    assert exact.startswith("Bạn nên khám")
    assert partial.startswith("Bạn nên khám")


def test_finalizer_removes_empty_heading_duplicate_disclaimer_and_legacy_template():
    answer = finalize_answer_presentation(
        (
            "**Tóm tắt ngắn**\nNên chăm sóc da dịu nhẹ.\n\n"
            "**Giải thích/cơ chế**\nMụn viêm liên quan phản ứng viêm.\n\n"
            "**Chăm sóc/điều trị thường gặp**\nRửa mặt nhẹ.\n\n"
            "**Lưu ý an toàn/tác dụng phụ**\nTheo dõi kích ứng.\n\n"
            "**Khi nào nên gặp bác sĩ**\n\n"
            "Thông tin mang tính tham khảo và không thay thế chẩn đoán của bác sĩ.\n\n"
            "Thông tin mang tính tham khảo và không thay thế chẩn đoán của bác sĩ."
        ),
        user_question="Mụn viêm nhẹ nên chăm sóc da hằng ngày như thế nào?",
        response_profile="treatment",
    )

    assert "**Tóm tắt ngắn**" not in answer
    assert "**Khi nào nên gặp bác sĩ**" not in answer
    assert answer.count("Thông tin mang tính tham khảo") == 0


def test_structural_quality_detects_incomplete_and_truncated_output():
    issues = assess_structural_quality(
        "Mụn viêm có thể làm da nhạy cảm với ánh nắng",
        user_question="Mụn viêm nhẹ nên chăm sóc da hằng ngày như thế nào?",
    )
    codes = {issue["code"] for issue in issues}

    assert "incomplete_terminal_sentence" in codes

    truncated = assess_structural_quality("Câu trả lời\n...[truncated_generation]", user_question="Mụn là gì?")
    assert "truncated_generation" in {issue["code"] for issue in truncated}


def test_terminal_punctuation_repair_keeps_true_truncation_blocking():
    complete_without_stop = "Mụn đầu đen là tổn thương không viêm thường gặp ở da dầu"
    assert repair_terminal_punctuation(complete_without_stop).endswith(".")

    dangling_fragment = "Mụn đầu đen có thể xuất hiện khi da tiết dầu nhiều và"
    assert repair_terminal_punctuation(dangling_fragment) == dangling_fragment


def test_non_boolean_composition_does_not_start_with_yes_prefix():
    answer = finalize_answer_presentation(
        "Có. Epiduo chứa adapalene và benzoyl peroxide.",
        user_question="Epiduo gồm những hoạt chất nào và mỗi hoạt chất có tác dụng gì?",
    )

    assert answer.startswith("Epiduo chứa hai hoạt chất")
    assert not answer.startswith("Có.")


def test_differin_epiduo_comparison_does_not_collapse_to_epiduo_only():
    answer = finalize_answer_presentation(
        "Epiduo chứa adapalene và benzoyl peroxide.",
        user_question="Differin và Epiduo khác nhau ở thành phần nào?",
    )

    assert "Differin" in answer
    assert "Epiduo" in answer
    assert "adapalene" in answer
    assert "benzoyl peroxide" in answer
    assert "| Thuốc | Thành phần chính | Ý nghĩa |" in answer


def test_rewritten_epiduo_differin_comparison_uses_safe_product_specific_answer():
    answer = finalize_answer_presentation(
        "Benzoyl peroxide có thể sử dụng dưới dạng thuốc bôi hoặc thuốc uống.",
        user_question="Epiduo khác Differin ở điểm nào?",
    )

    assert "Differin" in answer
    assert "Epiduo" in answer
    assert "adapalene" in answer
    assert "benzoyl peroxide" in answer
    assert "thuốc uống" not in answer


def test_tretinoin_adapalene_comparison_uses_a_complete_markdown_table():
    answer = finalize_answer_presentation(
        "Tretinoin và adapalene là retinoid.",
        user_question="Tretinoin khác adapalene như thế nào về nhóm thuốc và khả năng gây kích ứng? Hãy tóm tắt dạng bảng.",
    )

    assert "| Tiêu chí | Adapalene | Tretinoin |" in answer
    assert "adapalene" in answer.lower()
    assert "tretinoin" in answer.lower()
    assert "kích ứng" in answer.lower()


@pytest.mark.asyncio
async def test_finalize_tazorac_differin_epiduo_comparison_keeps_all_products():
    result = await finalize_response_node(
        {
            "user_question": "Tazorac, Differin và Epiduo khác nhau về hoạt chất như thế nào?",
            "draft_answer": "Differin chứa adapalene, còn Epiduo chứa adapalene và benzoyl peroxide.",
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
        }
    )

    answer = result["final_answer"]
    assert answer.startswith("Các sản phẩm này khác nhau chủ yếu ở hoạt chất chính.")
    assert "| Sản phẩm | Hoạt chất chính | Nhóm thuốc |" in answer
    assert "| Tazorac | Tazarotene | Retinoid bôi/topical retinoid. |" in answer
    assert "| Differin | Adapalene | Retinoid bôi/topical retinoid. |" in answer
    assert "| Epiduo | Adapalene + benzoyl peroxide |" in answer
    assert "benzoyl peroxide không phải kháng sinh" in answer


def test_tazorac_differin_comparison_does_not_introduce_epiduo_or_bpo():
    answer = finalize_answer_presentation(
        "Differin chứa adapalene.",
        user_question="Tazorac và Differin khác nhau thế nào?",
    )

    assert "Tazorac" in answer
    assert "Tazarotene" in answer
    assert "Differin" in answer
    assert "Adapalene" in answer
    assert "Epiduo" not in answer
    assert "benzoyl peroxide" not in answer.lower()


def test_tazorac_group_identity_preserves_product_and_active_ingredient():
    answer = finalize_answer_presentation(
        "Tazorac la retinoid boi.",
        user_question="Tazorac (tazarotene) thuộc nhóm thuốc nào?",
    )

    assert "Tazorac" in answer
    assert "tazarotene" in answer.lower()
    assert "retinoid bôi" in answer.lower()


@pytest.mark.asyncio
async def test_finalize_mild_acne_skincare_uses_bullets_and_fixed_wording():
    result = await finalize_response_node(
        {
            "user_question": "Mụn viêm nhẹ nên chăm sóc da hằng ngày như thế nào?",
            "draft_answer": "Rửa mặt dịu nhẹ và dùng dưỡng ẩm.",
            "conversation_history": [],
            "use_history_context": False,
            "is_in_domain": True,
            "guardrail": "in_domain",
        }
    )

    answer = result["final_answer"]
    assert "## Chăm sóc hằng ngày" in answer
    assert "- Rửa mặt nhẹ nhàng" in answer
    assert "ngưng dùng và hỏi bác sĩ nếu kích ứng rõ" in answer
    assert "ngưng hỏi bác sĩ" not in answer
    assert "Benzoyl peroxide hoặc salicylic acid không bị cấm mặc định" in answer
    assert result["response_profile"] == "treatment"


def test_source_metadata_keeps_raw_ids_but_exposes_friendly_display_names():
    metadata = build_source_metadata(
        ["web_raw_dataset.json", "entity:active_ingredient", "qd_4416_cut.pdf", "entity:active_ingredient"],
        contexts=[
            {"source_file": "qd_4416_cut.pdf", "document_title": "Guideline Acne Treatment"},
            {"source_file": "web_raw_dataset.json", "source_type": "web_json"},
        ],
    )

    assert [item["source_id"] for item in metadata] == [
        "entity:active_ingredient",
        "web_raw_dataset.json",
        "qd_4416_cut.pdf",
    ]
    assert [item["display_name"] for item in metadata] == [
        "Cơ sở tri thức hoạt chất",
        "Bộ dữ liệu kiến thức mụn",
        "Tài liệu tiếng Việt về mụn trứng cá",
    ]


def test_source_display_fallback_humanizes_without_fabricating_title():
    displays = display_names_for_sources(["qd_4416_cut.pdf", "PIIS0190962223033893.pdf", "unknown-source"])

    assert displays == [
        "Tài liệu chuyên môn về điều trị mụn",
        "Tài liệu tiếng Việt về mụn trứng cá",
        "Unknown Source",
    ]


def test_comparison_detector_handles_khac_the_nao_variant():
    instruction = answer_format_instruction_for_question(
        "Adapalene và benzoyl peroxide khác nhau thế nào?"
    )

    assert "CÂU SO SÁNH" in instruction


def test_boolean_question_keeps_valid_yes_prefix():
    answer = finalize_answer_presentation(
        "Có. Epiduo chứa benzoyl peroxide.",
        user_question="Epiduo có BPO không?",
    )

    assert answer.startswith("Có.")


@pytest.mark.parametrize("provider", ["gemini", "ollama"])
@pytest.mark.asyncio
async def test_provider_generation_uses_same_formatting_contract(monkeypatch, provider: str):
    captured: dict[str, str] = {}

    async def fake_generate_llm_response(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return {
            "text": "Mụn đầu đen là nhân mụn mở; mụn đầu trắng là nhân mụn đóng.",
            "provider": provider,
            "model": "mock-model",
            "fallback_used": False,
            "fallback_provider": None,
            "fallback_model": None,
            "resilience": {"provider": provider},
        }

    monkeypatch.setattr(reason_node, "generate_llm_response", fake_generate_llm_response)

    result = await reason_node.generate_answer_node(
        {
            "user_question": "Mụn đầu đen và mụn đầu trắng khác nhau như thế nào?",
            "vector_contexts": [
                {
                    "text": "Blackheads are open comedones; whiteheads are closed comedones.",
                    "source_file": "doc.pdf",
                    "score": 0.9,
                }
            ],
            "graph_facts": [],
            "safety_flags": [],
            "symptoms": [],
            "conversation_history": [],
            "use_history_context": False,
            "ignored_out_of_domain_part": False,
            "is_in_domain": True,
            "llm_provider": provider,
            "llm_model": "mock-model",
            "allow_model_fallback": False,
        }
    )

    assert ANSWER_FORMATTING_CONTRACT in captured["prompt"]
    assert "bảng Markdown GFM" in captured["prompt"]
    assert result["prompt_budget"]["accounting_mode"] == "observation_only"
    assert result["prompt_budget"]["estimated_total_tokens"] > 0


def test_full_prompt_budget_observation_accounts_for_graph_and_all_rendered_text() -> None:
    prompt = build_medical_prompt(
        question="Tazorac có dùng khi mang thai không?",
        symptoms=[],
        safety_flags=["pregnancy"],
        contexts=[{"text": "Source-backed pregnancy evidence.", "source_file": "source.pdf"}],
        graph_facts=[
            {
                "entity": "tazarotene",
                "entity_type": "ActiveIngredient",
                "relationship": "CONTRAINDICATED_IN",
                "related_entity": "pregnancy",
                "related_type": "SafetyContext",
                "evidence": "taxonomy",
            }
        ],
        conversation_history=[{"role": "user", "content": "Tôi đang mang thai."}],
    )

    observation = observe_medical_prompt_budget(prompt)

    assert observation.total_characters == len(prompt)
    assert observation.estimated_total_tokens == (len(prompt) + 3) // 4
    assert observation.evidence_characters > 0
    assert observation.graph_characters > 0
    assert observation.non_evidence_characters > 0
    assert observation.token_count_mode == "approximate_chars_div_4"
    assert observation.accounting_mode == "observation_only"
    assert observation.enforced_max_tokens is None


def test_s4b_packed_context_order_is_not_reranked_after_budgeting() -> None:
    contexts = [
        {
            "text": "First packed block",
            "score": 0.1,
            "context_pack_reason": "rrf_rank",
            "context_role": "medical_evidence",
        },
        {
            "text": "Second packed block",
            "score": 0.9,
            "context_pack_reason": "rrf_rank",
            "context_role": "medical_evidence",
        },
    ]

    selected = reason_node._select_answer_contexts(contexts, query="adapalene")

    assert [item["text"] for item in selected] == ["First packed block", "Second packed block"]


@pytest.mark.asyncio
async def test_generation_uses_configured_runtime_provider_when_request_omits_one(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_generate_llm_response(**kwargs):
        captured["provider"] = kwargs["provider"]
        return {
            "text": "Mụn đầu đen là nhân mụn mở.",
            "provider": "ollama",
            "model": "qwen3:8b",
            "fallback_used": False,
            "fallback_provider": None,
            "fallback_model": None,
            "resilience": {"provider": "ollama"},
        }

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(reason_node, "generate_llm_response", fake_generate_llm_response)

    await reason_node.generate_answer_node(
        {
            "user_question": "Mụn đầu đen là gì?",
            "vector_contexts": [],
            "graph_facts": [],
            "safety_flags": [],
            "symptoms": [],
            "conversation_history": [],
            "use_history_context": False,
            "ignored_out_of_domain_part": False,
            "is_in_domain": True,
            "llm_provider": None,
            "llm_model": None,
            "allow_model_fallback": False,
        }
    )

    assert captured["provider"] == "ollama"
