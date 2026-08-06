"""Shared answer-presentation contract and deterministic Markdown cleanup."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from src.agent.emergency_contract import (
    build_anaphylaxis_like_emergency_answer,
    build_generic_emergency_answer,
    is_anaphylaxis_like_emergency_query,
)
from src.agent.requested_structure import parse_requested_structure
from src.knowledge import DrugEntityNormalizer


ResponseProfile = Literal[
    "routine",
    "comparison",
    "drug_identity",
    "treatment",
    "safety",
    "urgent",
    "out_of_domain_emergency",
    "safe_fallback",
]

ANSWER_FORMATTING_CONTRACT_VERSION = "answer_formatting_contract_v11"

CANONICAL_DISCLAIMER = "Thông tin mang tính tham khảo và không thay thế chẩn đoán của bác sĩ."
LEGACY_DISCLAIMER = "Thông tin này chỉ mang tính tham khảo và không thay thế tư vấn y khoa chuyên nghiệp."

LEGACY_BOILERPLATE_HEADINGS = (
    "**Tóm tắt ngắn**",
    "**Giải thích/cơ chế**",
    "**Chăm sóc/điều trị thường gặp**",
    "**Lưu ý an toàn/tác dụng phụ**",
    "**Khi nào nên gặp bác sĩ**",
)

ANSWER_FORMATTING_CONTRACT = """\
ANSWER PRESENTATION CONTRACT V11:
- Dùng cùng một chuẩn trình bày cho Gemini, Gemini fallback, Ollama, cache hit, guardrail, severity guard và safe fallback; provider không được quyết định format.
- Không lặp lại hoặc dùng nguyên câu hỏi của người dùng làm tiêu đề. Bắt đầu ngay bằng câu trả lời.
- Trả lời trực tiếp trước, sau đó mới giải thích. Chỉ bắt đầu bằng "Có." hoặc "Không." khi câu hỏi thật sự là yes/no.
- Chọn cấu trúc theo response profile, không nối template nhiều mục vào mọi câu trả lời.
- Routine factual: ngắn gọn, không heading nếu chỉ hỏi định danh/thành phần; không thêm disclaimer trong thân answer khi UI đã có footer chung.
- Routine skincare: dùng heading Markdown ngắn và bullet hành động rõ ràng; không gắn nhãn Guardrail nếu vẫn là câu hỏi in-domain.
- Comparison: trả lời trực tiếp rồi dùng bảng Markdown GFM hoặc bullet đối chiếu; cover đủ các entity được hỏi.
- Multi-intent: nếu câu hỏi có nhiều ý hoặc nhiều câu hỏi con, phải trả lời từng ý; không được rút toàn bộ câu hỏi về một câu định danh đơn lẻ.
- Structured request: nếu người dùng yêu cầu bảng/cột/mục cụ thể, giữ đúng cấu trúc đó; không đảo cột thành hàng hoặc bỏ entity/item đã nêu.
- Signs/symptoms vs causes: nếu người dùng hỏi dấu hiệu/triệu chứng/biểu hiện, chỉ liệt kê dấu hiệu quan sát được; không thay bằng nguyên nhân, thói quen hoặc yếu tố làm nặng.
- Exact-count request: nếu người dùng yêu cầu đúng N ý, trả đúng N bullet; với dấu hiệu/triệu chứng/biểu hiện, chỉ nêu biểu hiện quan sát được và không thêm danh sách xử trí dài thứ hai.
- Emergency immediate action: nếu có khó thở kèm sưng quanh mắt/môi/mặt/họng hoặc phát ban sau thuốc/sản phẩm, câu đầu phải khuyên gọi cấp cứu/đi cấp cứu ngay; không dùng wording yếu như "có thể cần", "nên cân nhắc", "theo dõi thêm" hoặc chờ 24-48 giờ.
- Markdown style request: nếu người dùng yêu cầu tiêu đề đậm/in đậm/bold, dùng Markdown hợp lệ như **Tiêu đề**; frontend sẽ render, không được né instruction này.
- Drug identity/composition: trả lời trực tiếp tên nhóm/hoạt chất, giải thích ngắn vai trò; bullet khi có nhiều hoạt chất; không mở đầu "Có." nếu câu hỏi không phải yes/no.
- Safety/pregnancy: câu đầu nêu trực tiếp đối tượng được hỏi; mọi thuốc người dùng nêu phải được xử lý riêng; wording phải ưu tiên tránh/ngừng trong thai kỳ, không để người dùng hiểu rằng có thể tự tiếp tục dùng; chỉ render một warning.
- Crisis/self-harm: ưu tiên hành động an toàn tức thời trước lời khuyên trị mụn; khuyên gọi cấp cứu/cơ sở y tế khẩn cấp nếu có nguy cơ hành động ngay và nhờ người tin cậy ở bên.
- Urgent/severe acne: dùng heading Markdown và bullet; nêu rõ cần bác sĩ da liễu đánh giá sớm, tránh nặn/bóp, không tự dùng isotretinoin hoặc thuốc kê đơn.
- Acne fulminans-like: không chẩn đoán chắc chắn, nhưng nêu nghi ngờ acne fulminans khi có cục/nang viêm lớn, trợt loét/vảy xuất huyết, sốt hoặc đau khớp; khuyên khám/chuyển khẩn trong ngày hoặc đánh giá trong 24 giờ.
- Out-of-domain emergency: ngắn, trực tiếp, khuyên tìm trợ giúp y tế khẩn cấp; không dùng template mụn năm phần.
- Không có heading rỗng, heading lặp, disclaimer lặp, cảnh báo lặp, câu ghép hỏng hoặc câu bị cắt.
- Không đưa "Nguồn:" vào thân câu trả lời; hệ thống hiển thị nguồn riêng từ metadata.
- Luôn giữ tiếng Việt UTF-8 tự nhiên, không lộ prompt, context, JSON hoặc quy trình nội bộ.
- Không dùng giọng lên lớp, cụm từ máy móc như "provider failure"/"verification failed", hoặc phần mở đầu dài trước câu trả lời chính.
- Safe fallback phải nói nhẹ nhàng là tài liệu hiện có chưa đủ căn cứ, gợi ý cách làm rõ câu hỏi hoặc trao đổi với chuyên gia khi phù hợp; không lặp disclaimer.
"""


def answer_format_instruction_for_question(question: str) -> str:
    """Return intent-specific formatting hints without changing medical policy."""

    normalized = (question or "").lower()
    structure = parse_requested_structure(question)
    structure_hints: list[str] = []
    if structure.wants_table:
        if structure.required_columns:
            columns = ", ".join(structure.required_columns)
            count_text = (
                f"đúng {structure.exact_column_count} cột"
                if structure.exact_column_count
                else "các cột người dùng yêu cầu"
            )
            structure_hints.append(
                "FORMAT RIÊNG CHO BẢNG: dùng bảng Markdown GFM; header phải giữ "
                f"{count_text}: {columns}. Không đảo các cột này thành hàng."
            )
        else:
            structure_hints.append("FORMAT RIÊNG CHO BẢNG: dùng bảng Markdown GFM hợp lệ.")
        if structure.required_rows:
            rows = ", ".join(structure.required_rows)
            structure_hints.append(
                f"Các entity/hàng được hỏi phải xuất hiện trong answer: {rows}."
            )
    if structure.exact_item_count:
        structure_hints.append(
            f"Nếu trả lời dạng list, giữ đúng {structure.exact_item_count} ý chính như người dùng yêu cầu."
        )
    if structure.semantic_intent == "signs_symptoms":
        structure_hints.append(
            "Người dùng hỏi dấu hiệu/triệu chứng: chỉ nêu biểu hiện quan sát được, không thay bằng nguyên nhân/thói quen. "
            "Nếu có yêu cầu đúng N dấu hiệu/triệu chứng/biểu hiện, trả đúng N bullet và không thêm danh sách xử trí dài thứ hai."
        )
    if "bold_headings" in structure.style_constraints:
        structure_hints.append(
            "Người dùng yêu cầu tiêu đề đậm: dùng Markdown **Tiêu đề** cho các heading ngắn, không để lộ ký tự thừa."
        )

    if _is_comparison_question(normalized):
        base = (
            "FORMAT RIÊNG CHO CÂU SO SÁNH: Bắt đầu bằng một câu tóm tắt khác biệt chính. "
            "Sau đó dùng một bảng Markdown GFM hoặc bullet đối chiếu để cover đầy đủ từng entity trong câu hỏi. "
            "Nếu tài liệu chỉ đủ cho một entity, vẫn nhắc entity còn lại và nói rõ tài liệu hiện có chưa đủ thông tin về entity đó."
        )
        return _join_instruction_hints(structure_hints, base)
    if _is_direct_question(normalized):
        base = (
            "FORMAT RIÊNG CHO CÂU YES/NO HOẶC ĐỊNH DANH: Câu đầu tiên phải là câu trả lời trực tiếp. "
            "Không lặp câu hỏi. Không dùng template nhiều mục nếu câu trả lời đã đủ rõ."
        )
        return _join_instruction_hints(structure_hints, base)
    if _is_high_safety_question(normalized):
        base = (
            "FORMAT RIÊNG CHO CÂU AN TOÀN: Trả lời thận trọng, nêu điều không nên làm, "
            "và chỉ thêm mục 'Khi nào cần trao đổi với bác sĩ' nếu có dấu hiệu cần khám hoặc cấp cứu."
        )
        return _join_instruction_hints(structure_hints, base)
    base = (
        "FORMAT RIÊNG CHO CÂU THƯỜNG: Trả lời gọn, theo đúng intent, 2-4 đoạn ngắn hoặc bullet. "
        "Không thêm khung template dài nếu người dùng chỉ hỏi một ý."
    )
    return _join_instruction_hints(structure_hints, base)


def _join_instruction_hints(hints: list[str], base: str) -> str:
    if not hints:
        return base
    return " ".join(hints + [base])


def infer_response_profile(
    question: str,
    *,
    severity: str | None = None,
    guardrail: str | None = None,
    fallback_type: str | None = None,
) -> ResponseProfile:
    """Infer presentation profile from intent/severity/guardrail, never provider."""

    text = _fold(question)
    if fallback_type and fallback_type != "none":
        return "safe_fallback"
    if guardrail in {"medical_emergency_out_of_scope", "medical_emergency_allergy"}:
        return "out_of_domain_emergency"
    if guardrail and guardrail not in {"in_domain", "in_domain_rule", "in_domain_followup_rule", "partial_out_of_domain", None}:
        if "emergency" in guardrail or ("dau nguc" in text and "kho tho" in text):
            return "out_of_domain_emergency"
    if severity == "emergency":
        return "out_of_domain_emergency"
    if severity == "urgent" or _is_severe_acne_question(text):
        return "urgent"
    if _is_high_safety_question(text):
        return "safety"
    if _is_comparison_question(text):
        return "comparison"
    if _is_drug_identity_or_composition(text):
        return "drug_identity"
    if any(marker in text for marker in ["cham soc", "routine", "dieu tri", "tri mun"]):
        return "treatment"
    return "routine"


def finalize_answer_presentation(
    answer: str,
    *,
    user_question: str = "",
    response_profile: ResponseProfile | None = None,
    severity: str | None = None,
    guardrail: str | None = None,
    fallback_type: str | None = None,
    add_disclaimer: bool | None = None,
) -> str:
    """Apply deterministic presentation policy without inventing arbitrary medical content."""

    profile = response_profile or infer_response_profile(
        user_question,
        severity=severity,
        guardrail=guardrail,
        fallback_type=fallback_type,
    )
    question_folded = _fold(user_question)
    draft = _normalize_newlines(answer)
    draft = _remove_known_disclaimers(draft)

    replacement = _deterministic_profile_answer(
        user_question=user_question,
        profile=profile,
        guardrail=guardrail,
        draft=draft,
    )
    if replacement:
        draft = replacement

    draft = _ensure_boolean_cardinality_polarity(draft, question_folded)
    draft = _ensure_antibiotic_stewardship_guidance(draft, question_folded, profile)
    draft = strip_leading_question_echo(draft, user_question)
    if not _is_boolean_question(question_folded):
        draft = _strip_unexpected_boolean_prefix(draft)
    draft = _remove_source_lines(draft)
    draft = _normalize_common_surface_errors(draft)
    draft = normalize_answer_markdown(draft, disclaimer=CANONICAL_DISCLAIMER)
    draft = _remove_legacy_boilerplate_headings(draft, profile)
    draft = _repair_requested_structure_answer(draft, user_question)
    draft = _dedupe_exact_paragraphs(draft)
    draft = _trim_incomplete_terminal_paragraph(draft)
    draft = normalize_answer_markdown(draft, disclaimer=CANONICAL_DISCLAIMER)

    if not draft:
        draft = "Tài liệu hiện có chưa đủ thông tin để trả lời chắc chắn."

    should_add_disclaimer = _should_add_answer_disclaimer(profile) if add_disclaimer is None else add_disclaimer
    if should_add_disclaimer and profile not in {"out_of_domain_emergency", "safe_fallback"}:
        draft = _append_disclaimer_once(draft, CANONICAL_DISCLAIMER)

    return draft.strip()


def normalize_answer_markdown(text: str, *, disclaimer: str | None = None) -> str:
    """Apply safe formatting cleanup without rewriting medical claims."""

    answer = _normalize_newlines(text)
    answer = _remove_greetings(answer)
    answer = _normalize_table_spacing(answer)
    answer = _remove_empty_markdown_headings(answer)
    answer = _dedupe_exact_headings(answer)
    if disclaimer:
        answer = _dedupe_disclaimer(answer, disclaimer)
    answer = re.sub(r"[ \t]+\n", "\n", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def strip_leading_question_echo(answer: str, user_question: str) -> str:
    """Remove exact/high-confidence question echo only from the answer opening."""

    if not answer or not user_question:
        return answer
    lines = _normalize_newlines(answer).splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""

    question_norm = _normalized_match_text(user_question)
    first_norm = _normalized_match_text(lines[0])
    if first_norm and first_norm == question_norm:
        return "\n".join(lines[1:]).lstrip()

    partials = _question_tail_candidates(user_question)
    if first_norm in partials:
        return "\n".join(lines[1:]).lstrip()

    # Some local models echo the final clause and continue on the same line.
    for partial in partials:
        if partial and first_norm.startswith(partial + " "):
            raw = lines[0].strip()
            raw_norm = _normalized_match_text(raw)
            if raw_norm.startswith(partial + " "):
                words_to_remove = len(partial.split())
                lines[0] = " ".join(raw.split()[words_to_remove:]).lstrip(" .:;-")
                return "\n".join(lines).lstrip()
    return "\n".join(lines)


def assess_structural_quality(
    answer: str,
    *,
    user_question: str = "",
    response_profile: ResponseProfile | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic structural presentation issues."""

    text = _normalize_newlines(answer)
    profile = response_profile or infer_response_profile(user_question)
    issues: list[dict[str, Any]] = []
    folded = _fold(text)

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if user_question and _normalized_match_text(first_line) == _normalized_match_text(user_question):
        issues.append(_struct_issue("leading_question_echo", "error", "Answer starts by repeating the full user question."))
    elif user_question and _normalized_match_text(first_line) in _question_tail_candidates(user_question):
        issues.append(_struct_issue("partial_question_echo", "error", "Answer starts by repeating the tail of the user question."))

    heading_counts: dict[str, int] = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if _is_heading(stripped):
            heading_counts[stripped.lower()] = heading_counts.get(stripped.lower(), 0) + 1
            lookahead = index + 1
            while lookahead < len(lines) and not lines[lookahead].strip():
                lookahead += 1
            if lookahead >= len(lines) or _is_heading(lines[lookahead]):
                issues.append(_struct_issue("empty_heading", "error", f"Heading has no body: {stripped}"))
    if any(count > 1 for count in heading_counts.values()):
        issues.append(_struct_issue("duplicate_heading", "warning", "Answer repeats a Markdown heading."))

    if text.count(CANONICAL_DISCLAIMER) > 1 or text.count(LEGACY_DISCLAIMER) > 1:
        issues.append(_struct_issue("duplicate_disclaimer", "warning", "Answer repeats the medical disclaimer."))

    warning_lines = [
        _normalized_match_text(line)
        for line in lines
        if any(marker in _fold(line) for marker in ["luu y", "canh bao", "khong tu", "cap cuu", "mang thai"])
    ]
    if len(warning_lines) != len(set(warning_lines)):
        issues.append(_struct_issue("duplicate_warning", "warning", "Answer repeats the same warning."))

    legacy_count = sum(1 for heading in LEGACY_BOILERPLATE_HEADINGS if heading in text)
    if legacy_count >= 4:
        issues.append(_struct_issue("legacy_boilerplate", "error", "Answer uses the legacy five-section template."))

    if profile not in {"safe_fallback"} and not _is_boolean_question(_fold(user_question)):
        if re.match(r"^\s*(Có|Không)[,.]\s+", text, flags=re.IGNORECASE):
            issues.append(_struct_issue("unexpected_boolean_prefix", "warning", "Non-boolean answer starts with Có./Không."))

    if _has_incomplete_terminal_sentence(text):
        issues.append(_struct_issue("incomplete_terminal_sentence", "error", "Answer appears to end mid-sentence."))

    if any(marker in folded for marker in ["...[truncated]", "truncated_generation", "[truncated"]):
        issues.append(_struct_issue("truncated_generation", "error", "Answer contains truncation marker."))

    if re.search(r"\b(mụn|tình trạng|da|điều trị)\s+Chỉ dựa vào", text):
        issues.append(_struct_issue("malformed_sentence_join", "error", "Answer contains a malformed sentence join."))

    return issues


def _deterministic_profile_answer(
    *,
    user_question: str,
    profile: ResponseProfile,
    guardrail: str | None,
    draft: str,
) -> str | None:
    text = _fold(user_question)
    if profile == "out_of_domain_emergency":
        if "dau nguc" in text and "kho tho" in text:
            return (
                "Đau ngực kèm khó thở không phải biểu hiện điển hình của mụn. "
                "Bạn cần gọi cấp cứu hoặc đến cơ sở cấp cứu ngay nếu triệu chứng đang xảy ra hoặc nặng lên.\n\n"
                "Không tự quy triệu chứng này cho mụn hoặc tự điều trị bằng thuốc trị mụn."
            )
        if is_anaphylaxis_like_emergency_query(user_question):
            return build_anaphylaxis_like_emergency_answer()
        return build_generic_emergency_answer()

    if _is_self_harm_crisis_question(text):
        return (
            "Điều quan trọng nhất lúc này là an toàn của bạn, trước khi bàn tiếp về điều trị mụn.\n\n"
            "## Việc cần làm ngay\n"
            "- Nếu bạn thấy mình có thể tự làm hại bản thân hoặc không an toàn khi ở một mình, hãy gọi cấp cứu hoặc đến cơ sở y tế khẩn cấp ngay.\n"
            "- Hãy liên hệ một người đáng tin cậy và nhờ họ ở bên bạn ngay lúc này; cố gắng không ở một mình khi nguy cơ tăng.\n"
            "- Tránh rượu, chất kích thích và cất xa vật dụng có thể gây hại nếu có thể làm an toàn.\n\n"
            "## Sau khi đã an toàn\n"
            "- Mụn có thể ảnh hưởng mạnh đến giấc ngủ, giao tiếp và tâm lý, nên bạn cũng nên hẹn bác sĩ da liễu hoặc bác sĩ tâm lý/bác sĩ gia đình để được hỗ trợ.\n"
            "- Tôi không thể chẩn đoán tình trạng tâm thần qua chat, nhưng ý nghĩ tự làm hại bản thân là dấu hiệu cần được hỗ trợ trực tiếp."
        )

    structure = parse_requested_structure(user_question)
    structured_answer = _structured_requested_answer(text, structure)
    if structured_answer:
        return structured_answer

    followup_answer = _structured_followup_answer(text)
    if followup_answer:
        return followup_answer

    # Comparison-specific contracts below carry richer tables and safety
    # guidance. Use the taxonomy comparison only when none of them applies.
    defer_taxonomy_comparison = _is_comparison_question(text) or "khac" in text
    if not defer_taxonomy_comparison:
        taxonomy_answer = grounded_entity_relation_answer(user_question)
        if taxonomy_answer:
            return taxonomy_answer

    if "tazorac" in text and not _is_comparison_question(text) and any(
        marker in text for marker in ["thuoc nhom", "nhom thuoc", "nhom nao", "nhom gi"]
    ):
        return (
            "Tazorac là tên thương mại chứa tazarotene và thuộc nhóm retinoid bôi ngoài da.\n\n"
            "Tazarotene giúp điều hòa sừng hóa nang lông và thường cần được sử dụng theo hướng dẫn của bác sĩ da liễu. "
            "Đây không phải là kháng sinh."
        )

    if _is_retinoid_shared_class_question(text):
        return (
            "Có. Các hoạt chất này đều thuộc nhóm retinoid, nhưng khác đường dùng và bối cảnh chỉ định.\n\n"
            "| Hoạt chất | Nhóm chung | Điểm cần phân biệt |\n"
            "|---|---|---|\n"
            "| Adapalene | Retinoid | Thường là retinoid bôi, hay dùng cho mụn trứng cá và giảm bít tắc nang lông. |\n"
            "| Tretinoin | Retinoid | Cũng là retinoid bôi trong nhiều phác đồ, nhưng không đồng nghĩa hoàn toàn với adapalene. |\n"
            "| Isotretinoin | Retinoid | Thường là retinoid đường uống cho mụn nặng/có nguy cơ sẹo và cần bác sĩ da liễu theo dõi. |\n\n"
            "Vì vậy, không nên kết luận chúng “hoàn toàn khác nhóm”; điểm khác quan trọng là dạng dùng, mức độ cần theo dõi và chỉ định lâm sàng."
        )

    if _is_pregnancy_context(text):
        pregnancy_entities = _mentioned_treatment_entities(text)
        if len(pregnancy_entities) >= 2:
            return _pregnancy_multi_entity_answer(pregnancy_entities)
        if "isotretinoin" in text:
            return (
                "Có. Thai kỳ làm thay đổi cách đánh giá isotretinoin vì đây là retinoid đường uống "
                "có nguy cơ cao cho thai nhi. Isotretinoin không được tự dùng; không tự bắt đầu hoặc tiếp tục dùng thuốc khi đang, "
                "có thể đang hoặc chuẩn bị mang thai.\n\n"
                "Hãy liên hệ bác sĩ da liễu hoặc bác sĩ sản khoa càng sớm càng tốt để được hướng dẫn phù hợp. "
                "Nếu đã dùng thuốc, ghi lại tên thuốc, liều và thời điểm dùng để trao đổi với bác sĩ; không tự thay đổi liều theo đơn cũ."
            )
        if "adapalene" in text or "adapalen" in text or "retinoid" in text:
            return (
                "Nên tránh hoặc ngừng dùng adapalene trong thai kỳ và trao đổi với bác sĩ da liễu hoặc bác sĩ sản khoa.\n\n"
                "## Việc nên làm\n"
                "- Tạm ngưng hoặc chưa bắt đầu adapalene cho đến khi được bác sĩ xác nhận lựa chọn an toàn hơn.\n"
                "- Ưu tiên chăm sóc nền dịu nhẹ: rửa mặt nhẹ, dưỡng ẩm phù hợp và chống nắng.\n"
                "- Nếu đã lỡ dùng, hãy ghi lại thời gian, sản phẩm/nồng độ và trao đổi với bác sĩ để được tư vấn cụ thể.\n\n"
                "## Khi nào cần trao đổi với bác sĩ\n"
                "- Khi bạn cần chọn thuốc trị mụn trong thai kỳ.\n"
                "- Khi mụn viêm nhiều, đau, lan rộng hoặc có nguy cơ để lại sẹo."
            )

    if _is_acne_fulminans_like_question(text):
        return (
            "Mô tả này gợi ý tình trạng mụn rất nặng, có thể nghi acne fulminans, nhưng không thể chẩn đoán chắc chắn qua chat.\n\n"
            "## Mức độ khẩn cấp\n"
            "- Cần được bác sĩ da liễu hoặc cơ sở y tế đánh giá khẩn trong ngày.\n"
            "- Nếu có sốt, đau khớp, tổn thương trợt loét hoặc vảy xuất huyết, nên được đánh giá trong vòng 24 giờ.\n"
            "- Không tự nặn, cạy, dùng isotretinoin, kháng sinh uống hoặc thuốc kê đơn khi chưa được bác sĩ chỉ định.\n\n"
            "## Khi chờ đi khám\n"
            "- Giữ vùng da sạch nhẹ nhàng, tránh chà xát và theo dõi sốt/đau tăng nhanh.\n"
            "- Mang theo danh sách thuốc/sản phẩm đang dùng để bác sĩ đánh giá."
        )

    if _is_severe_acne_question(text):
        return (
            "Các dấu hiệu này cần được bác sĩ da liễu đánh giá sớm vì mụn cục sâu, đau, sưng đỏ và bắt đầu để lại sẹo có nguy cơ tiến triển nặng hơn.\n\n"
            "## Việc nên làm\n"
            "- Sắp xếp khám da liễu sớm để đánh giá mức độ mụn và nguy cơ sẹo.\n"
            "- Không tự dùng isotretinoin, kháng sinh uống hoặc thuốc kê đơn khi chưa được bác sĩ chỉ định.\n"
            "- Không nặn, bóp hoặc cạy các cục mụn sâu vì dễ làm viêm nặng hơn và tăng nguy cơ sẹo.\n\n"
            "## Trong lúc chờ khám\n"
            "- Giữ routine dịu nhẹ, tránh chà xát mạnh.\n"
            "- Tạm ngưng phối hợp nhiều hoạt chất dễ kích ứng nếu da đang đỏ rát."
        )

    if _is_mild_inflammatory_routine_question(text):
        return (
            "Mụn viêm nhẹ thường nên bắt đầu bằng routine đơn giản, dịu nhẹ và theo dõi đáp ứng của da thay vì phối hợp nhiều hoạt chất cùng lúc.\n\n"
            "## Chăm sóc hằng ngày\n"
            "- Rửa mặt nhẹ nhàng, tránh chà xát hoặc tẩy rửa quá mạnh.\n"
            "- Dưỡng ẩm phù hợp, không gây bít tắc và dùng chống nắng ban ngày.\n"
            "- Không nặn/cạy mụn viêm vì dễ làm đỏ lâu, thâm hoặc sẹo.\n\n"
            "## Khi cân nhắc hoạt chất\n"
            "- Benzoyl peroxide hoặc salicylic acid không bị cấm mặc định, nhưng có thể gây khô rát/kích ứng.\n"
            "- Nếu dùng, nên bắt đầu thận trọng, từng sản phẩm một và ngưng dùng và hỏi bác sĩ nếu kích ứng rõ."
        )

    if "tazorac" in text and _is_comparison_question(text):
        includes_epiduo = "epiduo" in text
        rows = [
            "| Sản phẩm | Hoạt chất chính | Nhóm thuốc |",
            "|---|---|---|",
            "| Tazorac | Tazarotene | Retinoid bôi/topical retinoid. |",
        ]
        if "differin" in text:
            rows.append("| Differin | Adapalene | Retinoid bôi/topical retinoid. |")
        if includes_epiduo:
            rows.append("| Epiduo | Adapalene + benzoyl peroxide | Phối hợp retinoid bôi và benzoyl peroxide. |")
        if includes_epiduo:
            note = (
                "Tazarotene và adapalene đều thuộc nhóm retinoid bôi, còn benzoyl peroxide không phải kháng sinh; "
                "đây là hoạt chất bôi có tác dụng kháng khuẩn/antimicrobial và hỗ trợ giảm bít tắc nhẹ."
            )
        else:
            note = (
                "Tazarotene và adapalene đều thuộc nhóm retinoid bôi, nhưng là hai hoạt chất khác nhau; "
                "lựa chọn sản phẩm nên dựa trên chỉ định, mức độ dung nạp và tư vấn của bác sĩ."
            )
        return (
            "Các sản phẩm này khác nhau chủ yếu ở hoạt chất chính.\n\n"
            + "\n".join(rows)
            + "\n\n"
            + note
        )

    if "differin" in text and "epiduo" in text and _is_comparison_question(text):
        return (
            "Hai sản phẩm khác nhau chủ yếu ở thành phần: Differin chứa adapalene, còn Epiduo chứa adapalene kết hợp benzoyl peroxide.\n\n"
            "| Thuốc | Thành phần chính | Ý nghĩa |\n"
            "|---|---|---|\n"
            "| Differin | Adapalene | Adapalene là retinoid bôi, giúp điều hòa sừng hóa nang lông, giảm bít tắc/nhân mụn và hỗ trợ chống viêm. |\n"
            "| Epiduo | Adapalene + benzoyl peroxide | Có cùng adapalene như Differin, thêm benzoyl peroxide để hỗ trợ tác dụng kháng khuẩn/antimicrobial với C. acnes và tiêu sừng nhẹ. |\n\n"
            "Benzoyl peroxide không phải kháng sinh; khi phối hợp trong sản phẩm như Epiduo, nó bổ sung cơ chế tác động khác với adapalene."
        )

    if "tretinoin" in text and ("adapalene" in text or "adapalen" in text) and _is_comparison_question(text):
        return (
            "Tretinoin và adapalene đều là retinoid bôi, nhưng khả năng dung nạp có thể khác nhau giữa từng người.\n\n"
            "| Tiêu chí | Adapalene | Tretinoin |\n"
            "|---|---|---|\n"
            "| Nhóm thuốc | Retinoid bôi. | Retinoid bôi. |\n"
            "| Vai trò | Giúp điều hòa sừng hóa nang lông, giảm bít tắc và hỗ trợ chống viêm. | Giúp điều hòa sừng hóa nang lông và hỗ trợ giảm bít tắc. |\n"
            "| Kích ứng | Có thể gây khô, đỏ, rát hoặc bong tróc, nhất là khi mới dùng. | Cũng có thể gây khô, đỏ, rát hoặc bong tróc; mức độ kích ứng cần theo dõi theo từng người và công thức. |\n\n"
            "Nên bắt đầu thận trọng, không tự tăng tần suất khi da đang kích ứng. Retinoid cần được bác sĩ đánh giá nếu bạn đang mang thai hoặc dự định mang thai."
        )

    if "isotretinoin" in text and _mentions_oral_antibiotic(text) and _is_comparison_question(text):
        return (
            "Isotretinoin và kháng sinh uống đều cần bác sĩ đánh giá, nhưng thường được cân nhắc trong bối cảnh khác nhau.\n\n"
            "| Tiêu chí | Isotretinoin | Kháng sinh uống |\n"
            "|---|---|---|\n"
            "| Bối cảnh | Retinoid đường uống thường được cân nhắc cho mụn nặng, nguy cơ sẹo hoặc không đáp ứng điều trị chuẩn. | Có thể được cân nhắc cho mụn viêm phù hợp theo bác sĩ đánh giá. |\n"
            "| Theo dõi | Cần bác sĩ da liễu đánh giá chỉ định, tác dụng phụ và thai kỳ khi có khả năng mang thai. | Cần bác sĩ kê đơn, đánh giá đáp ứng và tránh dùng kéo dài tùy tiện. |\n"
            "| Điều không nên làm | Không tự bắt đầu, đổi liều hoặc dùng lại đơn cũ. | Không tự mua, tự kéo dài hoặc phối hợp nhiều kháng sinh. |\n\n"
            "Bảng này không thay thế việc kê đơn; lựa chọn cụ thể cần dựa trên mức độ mụn và đánh giá trực tiếp của bác sĩ."
        )

    if "tazarotene" in text and ("adapalene" in text or "adapalen" in text) and _is_comparison_question(text):
        return (
            "Tazarotene và adapalene đều là retinoid bôi, nhưng dung nạp có thể khác nhau giữa từng người.\n\n"
            "| Tiêu chí | Tazarotene | Adapalene |\n"
            "|---|---|---|\n"
            "| Nhóm thuốc | Retinoid bôi. | Retinoid bôi. |\n"
            "| Vai trò | Hỗ trợ điều hòa sừng hóa nang lông và giảm bít tắc trong điều trị mụn khi phù hợp. | Hỗ trợ điều hòa sừng hóa nang lông, giảm bít tắc/nhân mụn và chống viêm. |\n"
            "| Kích ứng | Có thể gây khô, đỏ, rát hoặc bong tróc; cần theo dõi dung nạp. | Cũng có thể gây khô, đỏ, rát hoặc bong tróc, nhất là khi mới dùng. |\n\n"
            "Không tự tăng tần suất khi da đang kích ứng; retinoid cần được bác sĩ đánh giá nếu bạn đang mang thai hoặc dự định mang thai."
        )

    if _is_morning_evening_routine_comparison(text):
        return (
            "Routine buổi sáng và buổi tối đều nên ưu tiên làm sạch dịu nhẹ, dưỡng ẩm phù hợp, nhưng có mục tiêu khác nhau.\n\n"
            "| Bước | Buổi sáng | Buổi tối |\n"
            "|---|---|---|\n"
            "| Làm sạch | Rửa mặt dịu nhẹ, tránh chà xát quá mức. | Rửa mặt dịu nhẹ để loại bỏ kem chống nắng, bụi bẩn và dầu thừa trong ngày. |\n"
            "| Bảo vệ/điều trị | Dưỡng ẩm phù hợp và dùng chống nắng ban ngày. | Có thể dùng hoạt chất trị mụn theo kế hoạch đang dung nạp; không thêm nhiều hoạt chất mạnh cùng lúc. |\n"
            "| Khi kích ứng | Giảm các sản phẩm dễ gây rát và ưu tiên phục hồi hàng rào da. | Giảm tần suất hoặc tạm ngưng hoạt chất gây kích ứng rõ, rồi trao đổi với bác sĩ nếu không cải thiện. |\n\n"
            "Điều chỉnh từng bước để theo dõi dung nạp, thay vì đổi toàn bộ routine cùng lúc."
        )

    if _is_back_vs_face_acne_comparison(text):
        return (
            "Mụn ở lưng và mụn ở mặt đều cần chăm sóc dịu nhẹ, nhưng nên chú ý các yếu tố tại vùng da bị ảnh hưởng.\n\n"
            "| Tiêu chí | Mụn ở mặt | Mụn ở lưng |\n"
            "|---|---|---|\n"
            "| Chăm sóc tại chỗ | Dùng sản phẩm phù hợp da mặt, tránh cạy hoặc nặn tổn thương viêm. | Giữ vùng lưng sạch sau khi ra mồ hôi, tránh ma sát kéo dài từ quần áo hoặc dụng cụ. |\n"
            "| Theo dõi | Lưu ý tổn thương viêm kéo dài, đau hoặc có nguy cơ để lại sẹo. | Cũng cần lưu ý mụn viêm lan rộng, đau nhiều hoặc để lại sẹo. |\n"
            "| Khi cần khám | Khám da liễu khi mụn viêm nặng, đau, có sẹo hoặc không cải thiện. | Khám da liễu với các dấu hiệu tương tự, nhất là khi vùng tổn thương rộng hoặc đau nhiều. |\n\n"
            "Không tự nặn/bóp mụn hoặc tự dùng thuốc kê đơn; bác sĩ có thể đánh giá kế hoạch phù hợp cho từng vùng da."
        )

    if "epiduo" in text and "dalacin" in text and _is_comparison_question(text):
        return (
            "Epiduo và Dalacin T khác nhau chủ yếu ở hoạt chất và vai trò trong điều trị mụn.\n\n"
            "| Tiêu chí | Epiduo | Dalacin T |\n"
            "|---|---|---|\n"
            "| Hoạt chất chính | Adapalene + benzoyl peroxide. | Clindamycin. |\n"
            "| Nhóm/vai trò | Phối hợp retinoid bôi và benzoyl peroxide; benzoyl peroxide không phải kháng sinh. | Kháng sinh bôi, có thể hỗ trợ mụn viêm khi được chỉ định. |\n"
            "| Lưu ý về kháng kháng sinh | Benzoyl peroxide có thể hỗ trợ giảm nguy cơ kháng kháng sinh khi phối hợp phù hợp. | Không nên dùng đơn trị liệu hoặc kéo dài tùy tiện; cần bác sĩ đánh giá. |\n\n"
            "Cả hai có thể gây kích ứng tại chỗ; không tự thay thế hoặc phối hợp thuốc nếu chưa được bác sĩ hướng dẫn."
        )

    if _mentions_benzoyl_peroxide(text) and "clindamycin" in text and _is_comparison_question(text):
        return (
            "Benzoyl peroxide và clindamycin đều có thể xuất hiện trong điều trị mụn, nhưng không cùng nhóm thuốc.\n\n"
            "| Tiêu chí | Benzoyl peroxide | Clindamycin |\n"
            "|---|---|---|\n"
            "| Bản chất | Không phải kháng sinh; là hoạt chất bôi có tác dụng kháng khuẩn/antimicrobial và hỗ trợ tiêu sừng nhẹ. | Kháng sinh bôi. |\n"
            "| Vai trò | Tác động lên C. acnes và hỗ trợ giảm bít tắc. | Có thể hỗ trợ mụn viêm khi bác sĩ chỉ định. |\n"
            "| Khi phối hợp | Có thể hỗ trợ tăng hiệu quả và giảm nguy cơ kháng kháng sinh. | Không nên dùng đơn trị liệu hoặc kéo dài tùy tiện. |\n\n"
            "Không tự bắt đầu, kéo dài hoặc đổi kháng sinh; hãy trao đổi với bác sĩ da liễu nếu cần phối hợp thuốc."
        )

    if _mentions_benzoyl_peroxide(text) and _mentions_topical_retinoid(text) and _is_comparison_question(text):
        return (
            "Retinoid bôi và benzoyl peroxide cùng dùng cho mụn nhưng tác động vào các cơ chế khác nhau.\n\n"
            "| Tiêu chí | Retinoid bôi | Benzoyl peroxide |\n"
            "|---|---|---|\n"
            "| Vai trò chính | Điều hòa sừng hóa nang lông, hỗ trợ giảm bít tắc và nhân mụn. | Không phải kháng sinh; có tác dụng kháng khuẩn/antimicrobial với C. acnes và tiêu sừng nhẹ. |\n"
            "| Lưu ý | Có thể gây khô, đỏ, bong tróc; cần cẩn trọng trong thai kỳ. | Có thể gây khô, đỏ, bong tróc và làm bạc màu vải/tóc. |\n"
            "| Phối hợp | Có thể phối hợp trong một số phác đồ khi da dung nạp và có hướng dẫn phù hợp. | Bổ sung cơ chế khác với retinoid bôi. |"
        )

    if _is_antibiotic_monotherapy_and_bp_role_question(text):
        return (
            "Không nên dùng clindamycin bôi hoặc kháng sinh uống đơn độc/kéo dài để điều trị mụn nếu chưa có bác sĩ đánh giá.\n\n"
            "- Clindamycin bôi là kháng sinh bôi; dùng đơn trị liệu hoặc kéo dài có thể làm tăng nguy cơ kháng kháng sinh.\n"
            "- Kháng sinh uống cũng không nên dùng đơn độc hoặc kéo dài tùy tiện; thường cần bác sĩ kê đơn, theo dõi và phối hợp với điều trị bôi phù hợp.\n"
            "- Benzoyl peroxide không phải kháng sinh. Khi phối hợp với kháng sinh trị mụn, benzoyl peroxide giúp tăng hiệu quả và giảm nguy cơ kháng kháng sinh.\n\n"
            "Vì vậy, trọng tâm không phải chỉ là “có dùng kháng sinh hay không”, mà là tránh đơn trị liệu/kéo dài và dùng phối hợp đúng chỉ định."
        )

    if "epiduo" in text and any(marker in text for marker in ["gom", "hoat chat", "thanh phan", "moi hoat chat"]):
        return (
            "Epiduo chứa hai hoạt chất chính là adapalene và benzoyl peroxide.\n\n"
            "- Adapalene là retinoid bôi, giúp điều hòa sừng hóa nang lông, giảm bít tắc/nhân mụn và hỗ trợ chống viêm.\n"
            "- Benzoyl peroxide không phải kháng sinh; đây là hoạt chất bôi có tác dụng kháng khuẩn/antimicrobial với C. acnes và hỗ trợ tiêu sừng nhẹ.\n\n"
            "Hai hoạt chất này có thể gây khô, đỏ, rát hoặc bong tróc; benzoyl peroxide còn có thể làm bạc màu vải/tóc. Nếu da kích ứng mạnh, nên ngưng dùng và hỏi bác sĩ."
        )

    if "epiduo" in text and any(marker in text for marker in ["bpo", "benzoyl peroxide", "benzoyl"]):
        return (
            "Có. Epiduo chứa benzoyl peroxide và adapalene.\n\n"
            "Adapalene là retinoid bôi, giúp giảm bít tắc nang lông và hỗ trợ chống viêm. "
            "Benzoyl peroxide không phải kháng sinh; đây là hoạt chất bôi có tác dụng kháng khuẩn/antimicrobial với C. acnes và tiêu sừng nhẹ.\n\n"
            "Hai hoạt chất này có thể gây khô, đỏ, rát hoặc bong tróc; benzoyl peroxide còn có thể làm bạc màu vải/tóc."
        )

    if "adapalene" in text and _mentions_benzoyl_peroxide(text) and _is_comparison_question(text):
        return (
            "Adapalene và benzoyl peroxide đều là hoạt chất bôi trị mụn, nhưng tác động lên các cơ chế khác nhau.\n\n"
            "| Tiêu chí | Adapalene | Benzoyl peroxide |\n"
            "|---|---|---|\n"
            "| Nhóm/ bản chất | Retinoid bôi. | Không phải kháng sinh; là hoạt chất bôi có tác dụng antimicrobial/kháng khuẩn. |\n"
            "| Vai trò chính | Điều hòa sừng hóa nang lông, giảm bít tắc/nhân mụn và hỗ trợ chống viêm. | Tác động lên C. acnes và hỗ trợ tiêu sừng nhẹ/giảm bít tắc. |\n"
            "| Lưu ý | Có thể gây khô, đỏ, bong tróc; cần cẩn trọng thai kỳ. | Có thể gây khô, đỏ, bong tróc và làm bạc màu vải/tóc. |\n\n"
            "Hai hoạt chất này có thể được phối hợp trong một số phác đồ vì tác động lên các cơ chế khác nhau."
        )

    if _mentions_benzoyl_peroxide(text) and ("khang sinh" in text or "antibiotic" in text):
        return (
            "Không, benzoyl peroxide không phải là kháng sinh.\n\n"
            "Benzoyl peroxide là hoạt chất bôi trị mụn có tác dụng kháng khuẩn/antimicrobial với C. acnes và hỗ trợ giảm bít tắc nang lông/tiêu sừng nhẹ. "
            "Clindamycin hoặc erythromycin mới là kháng sinh bôi; khi phối hợp với kháng sinh bôi, benzoyl peroxide giúp tăng hiệu quả và giảm nguy cơ kháng kháng sinh."
        )

    if "clindamycin" in text and any(marker in text for marker in ["don doc", "don tri lieu", "monotherapy"]):
        return (
            "Không. Clindamycin không nên được dùng đơn độc để trị mụn.\n\n"
            "Clindamycin là kháng sinh bôi; dùng đơn trị liệu hoặc kéo dài có thể làm tăng nguy cơ kháng kháng sinh. Nếu bác sĩ chỉ định kháng sinh bôi, thuốc thường được phối hợp với benzoyl peroxide để tăng hiệu quả và giảm nguy cơ kháng kháng sinh."
        )

    if "differin" in text and any(
        marker in text
        for marker in ["thuoc gi", "thuoc nhom", "thuoc nhom gi", "nhom thuoc", "nhom gi", "hoat chat", "thuoc gì"]
    ):
        return (
            "Differin thuộc nhóm retinoid bôi ngoài da. Hoạt chất chính của Differin là adapalene.\n\n"
            "Adapalene giúp điều hòa sừng hóa nang lông, giảm bít tắc/nhân mụn và có tác dụng chống viêm. Hoạt chất này không phải là kháng sinh."
        )

    if "mun dau den" in text and "mun dau trang" in text and _is_comparison_question(text):
        return (
            "Mụn đầu đen và mụn đầu trắng đều là mụn nhân do bít tắc nang lông, nhưng khác nhau ở việc nhân mụn mở hay đóng trên bề mặt da.\n\n"
            "| Tiêu chí | Mụn đầu đen | Mụn đầu trắng |\n"
            "|---|---|---|\n"
            "| Bề mặt | Nhân mụn mở, bã nhờn/tế bào sừng tiếp xúc không khí nên sẫm màu. | Nhân mụn đóng, bề mặt bị che phủ nên nhìn trắng hoặc màu da. |\n"
            "| Chăm sóc | Tránh nặn mạnh; ưu tiên làm sạch dịu nhẹ và sản phẩm không gây bít tắc. | Tương tự, tránh cạy/nặn và theo dõi kích ứng khi dùng hoạt chất. |\n\n"
            "Nếu mụn viêm nhiều, đau hoặc để lại sẹo, nên khám bác sĩ da liễu."
        )

    if defer_taxonomy_comparison:
        taxonomy_answer = grounded_entity_relation_answer(user_question)
        if taxonomy_answer:
            return taxonomy_answer

    return None


def _structured_followup_answer(text: str) -> str | None:
    """Provide bounded, evidence-safe replies for resolved follow-up intents."""

    folded = _fold(text)
    if any(marker in folded for marker in ("hoat chat", "tam ngung")) and re.search(
        r"\b(?:rat|kich ung|bong troc|kho)\b",
        folded,
    ):
        return (
            "Khi da đang kích ứng hoặc rát, nên giảm tần suất hoặc tạm ngưng các hoạt chất bôi gây rát rõ thay vì tiếp tục dùng nhiều sản phẩm cùng lúc. "
            "Ưu tiên dưỡng ẩm dịu nhẹ và chỉ dùng lại từng hoạt chất, từ từ, khi da đã ổn hơn; nếu kích ứng tăng hoặc kéo dài, nên hỏi bác sĩ da liễu."
        )
    if "retinoid" in folded and any(marker in folded for marker in ("ban ngay", "chong nang")):
        return (
            "Nếu bạn dùng retinoid bôi vào buổi tối, bước đặc biệt quan trọng ban ngày là dùng kem chống nắng phổ rộng. "
            "Retinoid có thể gây khô, đỏ, bong tróc hoặc kích ứng; dưỡng ẩm phù hợp và tránh chà xát giúp da dễ dung nạp hơn."
        )
    if "mun dau den" in folded and "mun viem" in folded:
        return (
            "Không. Mụn đầu đen thường là tổn thương không viêm do bít tắc nang lông mở. "
            "Màu sẫm xuất hiện khi chất trong nhân mụn tiếp xúc không khí và bị oxy hóa; "
            "điều này không tự nó cho thấy mụn viêm."
        )
    if "mun lung" in folded and any(marker in folded for marker in ("thoi quen", "mo hoi", "ma sat")):
        return (
            "Với mụn lưng sau khi tập, nên tắm hoặc thay áo sớm để giảm mồ hôi lưu trên da và tránh quần áo, ba lô hoặc dụng cụ gây ma sát kéo dài. "
            "Ưu tiên quần áo thoáng, giặt sạch đồ tập và không cạy/nặn tổn thương."
        )
    if any(marker in folded for marker in ("dieu chinh tan suat", "tan suat the nao", "giam tan suat")) and any(
        marker in folded for marker in ("kho", "rat", "bong troc", "kich ung")
    ):
        subject = "benzoyl peroxide" if _mentions_benzoyl_peroxide(folded) else "hoạt chất bôi"
        return (
            f"Nếu da khô hoặc rát khi dùng {subject}, hãy giảm tần suất xuống mức da dung nạp được và bổ sung dưỡng ẩm dịu nhẹ. "
            "Chỉ tăng lại từ từ khi kích ứng đã ổn; nếu đỏ rát tăng rõ hoặc kéo dài, nên tạm ngưng và hỏi bác sĩ da liễu."
        )
    return None


def grounded_entity_relation_answer(question: str) -> str | None:
    """Preserve product, ingredient and taxonomy relations in the first sentence.

    These facts are read from the project's canonical taxonomy rather than
    inferred from the model response, so a noisy graph lookup cannot turn a
    simple relation question into a broad treatment answer.
    """

    folded = _fold(question)
    relation_markers = (
        "alias",
        "map",
        "lien he",
        "co cung",
        "chung entity",
        "diem chung",
        "taxonomy",
        "khang sinh",
        "thuoc nhom",
        "nhom retinoid",
        "topical retinoid",
        "oral retinoid",
        "co the chi",
        "hai hoat chat",
        "hoat chat nao",
        "hoat chat chinh",
        "thanh phan",
        "khac",
        "viet thieu",
        "co the dang noi",
    )
    if not any(marker in folded for marker in relation_markers):
        return None
    try:
        normalizer = DrugEntityNormalizer()
        cards = [
            card
            for card in normalizer.match_alias(question)
            if card.entity_type in {"drug_product", "active_ingredient"}
        ]
    except Exception:
        return None
    cards = _dedupe_taxonomy_cards(cards)
    if not cards:
        return None

    if "alias" in folded and len(cards) == 1:
        card = cards[0]
        alias = _matched_noncanonical_alias(question, card.aliases, card.canonical_name)
        if alias:
            canonical_name = _display_card_name(question, card)
            details: list[str] = []
            if card.entity_type == "drug_product":
                ingredients = _taxonomy_ingredient_names(normalizer, card.active_ingredients)
                if ingredients:
                    details.append(f"{canonical_name} chứa hoạt chất {_human_join(ingredients)}")
                classes = _card_classes(card)
                if classes:
                    details.append(f"thuộc nhóm {_taxonomy_class_label(classes[0])}")
            suffix = f" {', '.join(details)}." if details else ""
            return f"Có. “{alias}” là alias của {canonical_name}.{suffix}"

    if ("map" in folded or "co the chi" in folded or "co the dang noi" in folded) and len(cards) == 1:
        card = cards[0]
        mention = _matched_noncanonical_alias(question, card.aliases, card.canonical_name)
        mention = mention or _matched_entity_mention(question, card) or card.canonical_name
        canonical_name = str(card.canonical_name).replace("_", " ")
        if card.entity_type == "active_ingredient" and _is_spelling_variant_question(folded):
            related_products = _products_containing_ingredient(normalizer, card)
            if related_products:
                product_name = str(related_products[0].canonical_name).replace("_", " ")
                return (
                    f"Có. Trong taxonomy, {mention} là alias/map về entity {canonical_name}. "
                    f"{product_name} là sản phẩm chứa hoạt chất này."
                )
        classes = _card_classes(card)
        class_suffix = f" {canonical_name} thuộc nhóm {_taxonomy_class_label(classes[0])}." if classes else ""
        return f"Có. Trong taxonomy, {mention} là alias/map về entity {canonical_name}.{class_suffix}"

    product_cards = [card for card in cards if card.entity_type == "drug_product"]
    ingredient_cards = [card for card in cards if card.entity_type == "active_ingredient"]
    product_cards = _cards_in_question_order(question, product_cards)
    if len(product_cards) >= 2 and (_is_comparison_question(folded) or "khac" in folded) and any(
        marker in folded for marker in ("thanh phan", "hoat chat", "khac")
    ):
        first, second = product_cards[:2]
        first_name = _display_card_name(question, first)
        second_name = _display_card_name(question, second)
        first_ingredients = _taxonomy_ingredient_names(normalizer, first.active_ingredients)
        second_ingredients = _taxonomy_ingredient_names(normalizer, second.active_ingredients)
        return (
            f"{first_name} và {second_name} khác nhau về thành phần: "
            f"{first_name} chứa {_human_join(first_ingredients)}, còn {second_name} chứa {_human_join(second_ingredients)}."
        )
    if len(product_cards) >= 2 and any(marker in folded for marker in ("co cung", "chung entity", "diem chung")):
        first, second = product_cards[:2]
        shared_ingredients = [
            key
            for key in first.active_ingredients
            if key in set(second.active_ingredients)
        ]
        if shared_ingredients:
            names = _taxonomy_ingredient_names(normalizer, shared_ingredients)
            return f"Có. {first.canonical_name} và {second.canonical_name} cùng chứa hoạt chất {_human_join(names)}."

    class_markers = (
        "thuoc nhom",
        "nhom nao",
        "nhom gi",
        "nhom retinoid",
        "topical retinoid",
        "oral retinoid",
        "topical antibiotic",
        "oral antibiotic",
    )
    if len(cards) == 1 and any(marker in folded for marker in class_markers):
        card = cards[0]
        classes = _card_classes(card)
        if classes:
            class_label = _taxonomy_class_label(classes[0])
            display_name = _display_card_name(question, card)
            answer = f"{display_name} thuộc nhóm {class_label}."
            if card.entity_type == "drug_product":
                ingredients = _taxonomy_ingredient_names(normalizer, card.active_ingredients)
                if ingredients:
                    answer = f"{display_name} thuộc nhóm {class_label} và chứa hoạt chất {_human_join(ingredients)}."
            if classes[0] == "topical_antibiotic":
                answer += " Hoạt chất này được dùng theo chỉ định trong điều trị mụn."
            if classes[0] == "oral_retinoid":
                answer += " Đây là thuốc cần bác sĩ da liễu đánh giá và theo dõi."
            return answer

    if product_cards and any(marker in folded for marker in class_markers):
        product = product_cards[0]
        classes = _card_classes(product)
        ingredients = _taxonomy_ingredient_names(normalizer, product.active_ingredients)
        if classes:
            display_name = _display_card_name(question, product)
            answer = f"{display_name} thuộc nhóm {_taxonomy_class_label(classes[0])}."
            if ingredients:
                answer = f"{display_name} thuộc nhóm {_taxonomy_class_label(classes[0])} và chứa hoạt chất {_human_join(ingredients)}."
            return answer

    if product_cards and any(
        marker in folded
        for marker in ("lien he", "hai hoat chat", "hoat chat nao", "hoat chat chinh", "thanh phan")
    ):
        product = product_cards[0]
        product_ingredients = _taxonomy_ingredient_names(normalizer, product.active_ingredients)
        if product_ingredients:
            display_name = _display_card_name(question, product)
            classes = _card_classes(product)
            class_suffix = f" và thuộc nhóm {_taxonomy_class_label(classes[0])}" if classes else ""
            if len(product_ingredients) == 1:
                return f"{display_name} là sản phẩm chứa hoạt chất {product_ingredients[0]}{class_suffix}."
            prefix = "hai hoạt chất" if len(product_ingredients) == 2 else "các hoạt chất"
            return f"{display_name} chứa {prefix} {_human_join(product_ingredients)}{class_suffix}."

    if "khang sinh" in folded and len(cards) >= 2:
        classes = {class_name for card in cards for class_name in _card_classes(card)}
        labels = _human_join([card.canonical_name for card in cards[:2]])
        if "topical_antibiotic" not in classes:
            details = "; ".join(_product_ingredient_class_summary(card) for card in cards[:2])
            return f"Không. {labels} không phải là kháng sinh bôi. {details}"
        if all("topical_antibiotic" in _card_classes(card) for card in cards[:2]):
            return f"Có. {labels} đều được taxonomy xếp vào nhóm kháng sinh bôi tại chỗ."

    if "khang sinh" in folded and len(cards) == 1:
        card = cards[0]
        classes = _card_classes(card)
        canonical_name = str(card.canonical_name).replace("_", " ")
        if "topical_antibiotic" in classes:
            return f"Có. {canonical_name} được taxonomy xếp vào nhóm kháng sinh bôi tại chỗ."
        if "oral_antibiotic" in classes:
            return f"Có. {canonical_name} được taxonomy xếp vào nhóm kháng sinh đường uống và cần bác sĩ đánh giá, kê đơn."
        return f"Không, {canonical_name} không phải là kháng sinh. Đây cũng không phải là kháng sinh bôi."

    shared_class = _shared_taxonomy_class(cards)
    if len(cards) >= 2 and shared_class and any(marker in folded for marker in ("co cung", "diem chung", "taxonomy", "cung nhom")):
        labels = _human_join([card.canonical_name for card in cards[:2]])
        return f"Có. {labels} đều thuộc nhóm {_taxonomy_class_label(shared_class)}."
    return None


def _taxonomy_ingredient_names(normalizer: DrugEntityNormalizer, ingredient_keys: list[str]) -> list[str]:
    names: list[str] = []
    for ingredient_key in ingredient_keys:
        card = normalizer.get_entity_card("active_ingredient", ingredient_key)
        name = card.canonical_name if card else str(ingredient_key)
        names.append(str(name).replace("_", " "))
    return names


def _products_containing_ingredient(normalizer: DrugEntityNormalizer, ingredient_card: Any) -> list[Any]:
    ingredient_key = _canonical_taxonomy_key(ingredient_card)
    products = normalizer.cards_by_type.get("drug_product", {}).values()
    return [
        product
        for product in _dedupe_taxonomy_cards(list(products))
        if ingredient_key in {_fold_key(value) for value in product.active_ingredients}
    ]


def _fold_key(value: str) -> str:
    return str(value).replace(" ", "_").casefold()


def _is_spelling_variant_question(text: str) -> bool:
    return any(marker in text for marker in ("viet thieu", "viet sai", "typo", "spelling"))


def _display_card_name(question: str, card: Any) -> str:
    return str(_matched_entity_mention(question, card) or card.canonical_name).replace("_", " ")


def _cards_in_question_order(question: str, cards: list[Any]) -> list[Any]:
    folded_question = _fold(question)

    def first_mention(card: Any) -> int:
        aliases = [getattr(card, "canonical_name", ""), *(getattr(card, "aliases", []) or [])]
        positions = [folded_question.find(_fold(alias)) for alias in aliases if _fold(alias) in folded_question]
        return min((position for position in positions if position >= 0), default=len(folded_question))

    return sorted(cards, key=first_mention)


def _dedupe_taxonomy_cards(cards: list[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for card in cards:
        key = f"{card.entity_type}:{_canonical_taxonomy_key(card)}"
        if key not in seen:
            seen.add(key)
            output.append(card)
    return output


def _canonical_taxonomy_key(card: Any) -> str:
    metadata = getattr(card, "metadata", {}) or {}
    return str(metadata.get("taxonomy_key") or getattr(card, "canonical_name", "")).replace(" ", "_").casefold()


def _card_classes(card: Any) -> list[str]:
    return [str(value).casefold() for value in getattr(card, "drug_class", []) or [] if value]


def _shared_taxonomy_class(cards: list[Any]) -> str | None:
    classes = [set(_card_classes(card)) for card in cards[:2]]
    if len(classes) < 2:
        return None
    return next(iter(classes[0] & classes[1]), None)


def _taxonomy_class_label(value: str) -> str:
    return {
        "topical_retinoid": "retinoid bôi",
        "oral_retinoid": "retinoid đường uống",
        "topical_antibiotic": "kháng sinh bôi tại chỗ",
        "oral_antibiotic": "kháng sinh đường uống",
        "benzoyl_peroxide": "nhóm benzoyl peroxide",
    }.get(value, value.replace("_", " "))


def _matched_noncanonical_alias(question: str, aliases: list[str], canonical_name: str) -> str | None:
    folded_question = _fold(question)
    canonical = _fold(canonical_name)
    matches: list[tuple[int, int, str, bool]] = []
    for alias in aliases:
        alias_key = _fold(alias)
        if not alias_key:
            continue
        alias_pattern = re.escape(alias_key).replace(r"\ ", r"[-_\s]+")
        match = re.search(rf"(?<![a-z0-9]){alias_pattern}(?![a-z0-9])", folded_question)
        if match:
            matches.append((match.start(), -len(alias_key), alias, alias_key == canonical))
    if not matches:
        return None
    noncanonical = [match for match in matches if not match[3]]
    if noncanonical:
        first_noncanonical_position = min(match[0] for match in noncanonical)
        equivalent_spelling = [
            match
            for match in matches
            if match[3] and match[0] == first_noncanonical_position
        ]
        # Preserve a full hyphenated/canonical spelling at the same position
        # instead of collapsing it to a shorter alias prefix.
        return min(equivalent_spelling or noncanonical)[2]
    return min(matches)[2]


def _matched_entity_mention(question: str, card: Any) -> str | None:
    for value in [getattr(card, "canonical_name", ""), *(getattr(card, "aliases", []) or [])]:
        if _contains_entity_alias(question, value):
            return value
    return None


def _product_ingredient_class_summary(card: Any) -> str:
    ingredients = [str(value).replace("_", " ") for value in getattr(card, "active_ingredients", []) or []]
    classes = _card_classes(card)
    ingredient_text = _human_join(ingredients) if ingredients else card.canonical_name
    class_text = _taxonomy_class_label(classes[0]) if classes else "nhóm thuốc liên quan"
    return f"{card.canonical_name} chứa {ingredient_text} và thuộc nhóm {class_text}"


def _human_join(values: list[str]) -> str:
    values = [value for value in values if value]
    if len(values) <= 1:
        return values[0] if values else ""
    if len(values) == 2:
        return f"{values[0]} và {values[1]}"
    return ", ".join(values[:-1]) + f" và {values[-1]}"


def _structured_requested_answer(text: str, structure: Any) -> str | None:
    if not getattr(structure, "has_constraints", False):
        return None

    if (
        getattr(structure, "exact_item_count", None) == 3
        and "mun dau den" in text
    ):
        return (
            "- Mụn đầu đen là nhân mụn mở, hình thành khi nang lông bị bít tắc.\n"
            "- Phần nhân tiếp xúc không khí nên bị oxy hóa và có màu sẫm.\n"
            "- Đây thường là tổn thương không viêm; khác với sẩn hoặc mụn mủ đỏ, đau."
        )

    if _is_bp_antimicrobial_mechanism_followup(text):
        return (
            "Benzoyl peroxide có tác dụng kháng khuẩn vì nó tạo môi trường oxy hóa bất lợi cho C. acnes, "
            "vi khuẩn liên quan đến mụn viêm.\n\n"
            "- Benzoyl peroxide không phải là kháng sinh, nên cơ chế này khác với clindamycin/erythromycin.\n"
            "- Ngoài tác dụng kháng khuẩn/antimicrobial, benzoyl peroxide còn hỗ trợ tiêu sừng nhẹ và giảm bít tắc nang lông.\n"
            "- Khi phối hợp với kháng sinh bôi, benzoyl peroxide có thể giúp tăng hiệu quả và giảm nguy cơ kháng kháng sinh."
        )

    if structure.exact_item_count and _asks_acne_aggravating_habits(text):
        return _acne_aggravating_habits_list(structure.exact_item_count)

    if structure.wants_table and structure.semantic_intent == "treatment_summary":
        return _treatment_summary_table(structure)

    if structure.wants_table and structure.required_columns and structure.required_rows:
        return _entity_table_for_requested_structure(structure)

    if structure.semantic_intent == "signs_symptoms" and structure.exact_item_count:
        return _signs_symptoms_list(structure.exact_item_count, text)

    if "bold_headings" in structure.style_constraints and _asks_acne_mechanism(text):
        return (
            "**Bã nhờn**\n"
            "Tăng tiết bã nhờn làm môi trường nang lông dễ bị bít tắc và thuận lợi cho mụn hình thành.\n\n"
            "**Dày sừng cổ nang lông**\n"
            "Tế bào sừng tích tụ ở cổ nang lông làm nhân mụn dễ xuất hiện.\n\n"
            "**C. acnes và viêm**\n"
            "C. acnes cùng phản ứng viêm góp phần làm mụn đỏ, đau hoặc có mủ."
        )

    return None


def _asks_acne_aggravating_habits(text: str) -> bool:
    return "thoi quen" in text and any(
        marker in text
        for marker in ("mun nang hon", "lam nang mun", "lam mun nang", "mun tram trong hon")
    )


def _acne_aggravating_habits_list(expected_count: int) -> str:
    habits = [
        "- Nặn, bóp hoặc cạy mụn có thể làm tổn thương viêm nặng hơn và tăng nguy cơ thâm hoặc sẹo.",
        "- Ma sát kéo dài từ khẩu trang, mũ bảo hiểm, quần áo chật hoặc dụng cụ tập luyện có thể làm vùng da mụn kích ứng hơn.",
        "- Dùng mỹ phẩm hoặc sản phẩm tóc có xu hướng gây bít tắc có thể làm nhân mụn tăng lên.",
        "- Rửa mặt quá mạnh, chà xát hoặc thay quá nhiều hoạt chất cùng lúc có thể làm da kích ứng và khiến mụn khó kiểm soát hơn.",
    ]
    return "\n".join(habits[:expected_count])


def _repair_requested_structure_answer(answer: str, user_question: str) -> str:
    structure = parse_requested_structure(user_question)
    if (
        structure.exact_item_count
        and structure.semantic_intent != "signs_symptoms"
        and _count_markdown_items(answer) > structure.exact_item_count
    ):
        return _limit_markdown_items(answer, structure.exact_item_count)
    if (
        structure.semantic_intent == "signs_symptoms"
        and structure.exact_item_count
        and _sign_symptom_exact_count_needs_repair(answer, structure.exact_item_count)
    ):
        return _signs_symptoms_list(structure.exact_item_count, _fold(user_question))
    return answer


def _limit_markdown_items(answer: str, expected_count: int) -> str:
    """Keep the first requested list items without synthesizing medical claims."""

    kept: list[str] = []
    item_count = 0
    for line in answer.splitlines():
        if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line):
            item_count += 1
            if item_count > expected_count:
                break
        kept.append(line)
    return "\n".join(kept).strip()


def _sign_symptom_exact_count_needs_repair(answer: str, expected_count: int) -> bool:
    item_count = _count_markdown_items(answer)
    if item_count != expected_count:
        return True
    folded = _fold(answer)
    if _has_sign_symptom_forbidden_cause_terms(folded):
        return True
    return _has_second_management_list(folded)


def _has_sign_symptom_forbidden_cause_terms(folded_answer: str) -> bool:
    return any(
        marker in folded_answer
        for marker in (
            "thoi quen",
            "nguyen nhan",
            "thuc khuya",
            "an do ngot",
            "stress",
            "my pham gay bit tac",
            "boi qua day",
            "boi qua nhieu",
            "doi san pham lien tuc",
            "khong duong am",
        )
    )


def _has_second_management_list(folded_answer: str) -> bool:
    return any(
        marker in folded_answer
        for marker in (
            "viec nen lam",
            "cach xu ly",
            "xu tri",
            "khi nao nen gap bac si",
            "khi nao can gap bac si",
        )
    )


def _count_markdown_items(text: str) -> int:
    return len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text or ""))


def _treatment_summary_table(structure: Any) -> str:
    headers = _requested_headers_or_default(
        structure,
        ["Thuốc", "Đường dùng", "Ưu điểm", "Lưu ý an toàn"],
    )
    rows = [
        {
            "thuoc": "Mụn nhẹ-trung bình: benzoyl peroxide hoặc retinoid bôi như adapalene",
            "duong dung": "Bôi ngoài da",
            "uu diem": "Tác động lên bít tắc nang lông và/hoặc C. acnes; có thể dùng làm nền trong nhiều phác đồ.",
            "luu y an toan": "Có thể gây khô, đỏ, bong tróc; retinoid cần cẩn trọng thai kỳ.",
        },
        {
            "thuoc": "Mụn trung bình-nặng: phối hợp điều trị bôi; cân nhắc kháng sinh uống hoặc isotretinoin khi bác sĩ đánh giá",
            "duong dung": "Bôi và/hoặc đường uống tùy chỉ định",
            "uu diem": "Phù hợp hơn khi mụn viêm nhiều, đau hoặc có nguy cơ sẹo.",
            "luu y an toan": "Không tự dùng kháng sinh uống/isotretinoin; cần bác sĩ kê đơn và theo dõi.",
        },
    ]
    return _markdown_table(headers, rows) + (
        "\n\nBảng này là tóm tắt định hướng theo tài liệu hiện có; lựa chọn cụ thể vẫn cần bác sĩ đánh giá mức độ mụn, nguy cơ sẹo, thai kỳ và khả năng dung nạp."
    )


def _entity_table_for_requested_structure(structure: Any) -> str:
    headers = _requested_headers_or_default(
        structure,
        ["Hoạt chất", "Vai trò chính", "Tác dụng phụ thường gặp", "Lưu ý sử dụng"],
    )
    rows = []
    for row_name in structure.required_rows:
        facts = _entity_fact_cells(row_name)
        if not facts:
            continue
        rows.append(facts)
    if not rows:
        return ""
    return _markdown_table(headers, rows)


def _requested_headers_or_default(structure: Any, default: list[str]) -> list[str]:
    if not structure.required_columns:
        return default
    return [_display_column_label(column) for column in structure.required_columns]


def _markdown_table(headers: list[str], rows: list[dict[str, str]]) -> str:
    keys = [_column_key(header) for header in headers]
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [row.get(key) or row.get(_fallback_column_key(key), "Tài liệu hiện có chưa đủ thông tin.") for key in keys]
        output.append("| " + " | ".join(cells) + " |")
    return "\n".join(output)


def _display_column_label(column: str) -> str:
    labels = {
        "thuoc": "Thuốc",
        "hoat chat": "Hoạt chất",
        "duong dung": "Đường dùng",
        "uu diem": "Ưu điểm",
        "vai tro chinh": "Vai trò chính",
        "tac dung phu thuong gap": "Tác dụng phụ thường gặp",
        "luu y su dung": "Lưu ý sử dụng",
        "luu y an toan": "Lưu ý an toàn",
    }
    return labels.get(column, column[:1].upper() + column[1:])


def _column_key(header: str) -> str:
    return _fold(header).replace("/", " ").strip()


def _fallback_column_key(key: str) -> str:
    if key in {"ten thuoc", "lua chon"}:
        return "thuoc"
    if key in {"thanh phan"}:
        return "hoat chat"
    if key in {"loi ich"}:
        return "uu diem"
    if key in {"canh bao", "luu y"}:
        return "luu y an toan"
    if key in {"vai tro"}:
        return "vai tro chinh"
    if key in {"tac dung phu"}:
        return "tac dung phu thuong gap"
    return key


def _entity_fact_cells(row_name: str) -> dict[str, str] | None:
    key = _fold(row_name)
    facts: dict[str, dict[str, str]] = {
        "adapalene": {
            "hoat chat": "Adapalene",
            "thuoc": "Adapalene",
            "vai tro chinh": "Retinoid bôi, giúp điều hòa sừng hóa nang lông, giảm bít tắc/nhân mụn và hỗ trợ chống viêm.",
            "tac dung phu thuong gap": "Khô, đỏ, rát hoặc bong tróc.",
            "luu y su dung": "Cần cẩn trọng thai kỳ; không tự phối hợp nhiều hoạt chất khi da kích ứng.",
            "luu y an toan": "Cần cẩn trọng thai kỳ; không tự phối hợp nhiều hoạt chất khi da kích ứng.",
            "duong dung": "Bôi ngoài da",
            "uu diem": "Hữu ích cho nhân mụn/bít tắc và duy trì điều trị.",
        },
        "benzoyl peroxide": {
            "hoat chat": "Benzoyl peroxide",
            "thuoc": "Benzoyl peroxide",
            "vai tro chinh": "Không phải kháng sinh; có tác dụng kháng khuẩn/antimicrobial với C. acnes và tiêu sừng nhẹ.",
            "tac dung phu thuong gap": "Khô, đỏ, rát, bong tróc; có thể làm bạc màu vải/tóc.",
            "luu y su dung": "Tránh để dây vào vải/tóc; giảm tần suất hoặc hỏi bác sĩ nếu kích ứng rõ.",
            "luu y an toan": "Tránh để dây vào vải/tóc; giảm tần suất hoặc hỏi bác sĩ nếu kích ứng rõ.",
            "duong dung": "Bôi ngoài da",
            "uu diem": "Bổ sung cơ chế kháng khuẩn và giúp giảm nguy cơ kháng kháng sinh khi phối hợp kháng sinh bôi.",
        },
        "salicylic acid": {
            "hoat chat": "Salicylic acid",
            "thuoc": "Salicylic acid",
            "vai tro chinh": "Hoạt chất tiêu sừng/keratolytic, hỗ trợ giảm bít tắc bề mặt.",
            "tac dung phu thuong gap": "Khô, châm chích hoặc kích ứng nếu dùng quá dày/không phù hợp.",
            "luu y su dung": "Bắt đầu thận trọng, tránh phối hợp dồn dập nhiều hoạt chất dễ kích ứng.",
            "luu y an toan": "Bắt đầu thận trọng, tránh phối hợp dồn dập nhiều hoạt chất dễ kích ứng.",
            "duong dung": "Bôi ngoài da",
            "uu diem": "Có thể hữu ích với bít tắc/nhân mụn nhẹ nếu da dung nạp.",
        },
        "clindamycin": {
            "hoat chat": "Clindamycin",
            "thuoc": "Clindamycin",
            "vai tro chinh": "Kháng sinh bôi, nhắm vào vi khuẩn liên quan mụn viêm.",
            "tac dung phu thuong gap": "Kích ứng tại chỗ; nguy cơ kháng kháng sinh nếu dùng đơn độc/kéo dài.",
            "luu y su dung": "Không khuyến cáo dùng đơn trị liệu; thường cần phối hợp benzoyl peroxide khi bác sĩ chỉ định.",
            "luu y an toan": "Không khuyến cáo dùng đơn trị liệu; thường cần phối hợp benzoyl peroxide khi bác sĩ chỉ định.",
            "duong dung": "Bôi ngoài da",
            "uu diem": "Có thể hỗ trợ mụn viêm khi dùng đúng chỉ định.",
        },
    }
    return facts.get(key)


def _signs_symptoms_list(count: int, text: str) -> str:
    signs = [
        "Đỏ rát rõ ở vùng bôi.",
        "Khô căng hoặc bong tróc tăng lên.",
        "Châm chích/bỏng rát kéo dài thay vì chỉ thoáng qua.",
        "Sưng, nổi mẩn hoặc phát ban quanh vùng dùng sản phẩm.",
        "Đau nhiều, rỉ dịch hoặc phồng rộp.",
    ]
    selected = signs[: max(1, min(count, len(signs)))]
    intro = "Các dấu hiệu/triệu chứng có thể quan sát gồm:"
    if "tac dung phu" in text or "side effect" in text:
        intro = "Các dấu hiệu tác dụng phụ/kích ứng có thể quan sát gồm:"
    return intro + "\n\n" + "\n".join(f"- {item}" for item in selected)


def _is_bp_antimicrobial_mechanism_followup(text: str) -> bool:
    has_bp = "benzoyl peroxide" in text or re.search(r"(?<![a-z0-9])bp(?![a-z0-9])", text)
    asks_why = any(marker in text for marker in ["tai sao", "vi sao", "vì sao", "why", "how"])
    antimicrobial = any(marker in text for marker in ["khang khuan", "antimicrobial", "c. acnes", "vi khuan"])
    return bool(has_bp and asks_why and antimicrobial)


def _mentions_benzoyl_peroxide(text: str) -> bool:
    return "benzoyl peroxide" in text or bool(re.search(r"(?<![a-z0-9])bpo?(?![a-z0-9])", text))


def _mentions_oral_antibiotic(text: str) -> bool:
    return any(
        marker in text
        for marker in ("khang sinh uong", "khang sinh duong uong", "oral antibiotic", "doxycycline")
    )


def _mentions_topical_retinoid(text: str) -> bool:
    return any(marker in text for marker in ("retinoid boi", "topical retinoid"))


def _ensure_boolean_cardinality_polarity(draft: str, question_folded: str) -> str:
    """Make a negative one-versus-many answer explicit without changing medical facts."""

    if not re.search(r"\bco phai\s+chi(?:\s+co|\s+chua)?\s+.*\bkhong\b", question_folded):
        return draft
    opening = _fold(next((line for line in draft.splitlines() if line.strip()), ""))
    if opening.startswith(("khong", "co")):
        return draft
    if any(marker in opening for marker in ("hai hoat chat", "2 hoat chat", "nhieu hoat chat")):
        return f"Không. {draft.lstrip()}"
    return draft


def _ensure_antibiotic_stewardship_guidance(
    draft: str,
    question_folded: str,
    profile: ResponseProfile,
) -> str:
    """Add one concise action when an antibiotic answer lacks stewardship guidance."""

    if profile in {"out_of_domain_emergency", "safe_fallback"}:
        return draft
    antibiotic_markers = ("khang sinh", "antibiotic", "clindamycin", "erythromycin", "doxycycline")
    if not any(marker in question_folded for marker in antibiotic_markers):
        return draft
    folded_draft = _fold(draft)
    if any(marker in folded_draft for marker in ("khong tu", "khong nen", "bac si", "khang khang sinh")):
        return draft
    guidance = (
        "Không tự bắt đầu, kéo dài, ngừng hoặc đổi kháng sinh; hãy trao đổi với bác sĩ da liễu "
        "hoặc bác sĩ kê đơn để được đánh giá lại."
    )
    return f"{draft.rstrip()}\n\n{guidance}".strip()


def _asks_acne_mechanism(text: str) -> bool:
    return any(marker in text for marker in ["co che", "cơ chế", "vi sao gay mun", "hinh thanh mun"]) or (
        "mun" in text and any(marker in text for marker in ["ba nhon", "c. acnes", "viem", "day sung"])
    )


def _normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _remove_known_disclaimers(text: str) -> str:
    output = text
    for disclaimer in (CANONICAL_DISCLAIMER, LEGACY_DISCLAIMER):
        output = output.replace(disclaimer, "")
    return output.strip()


def _append_disclaimer_once(text: str, disclaimer: str) -> str:
    text = _remove_known_disclaimers(text).rstrip()
    return f"{text}\n\n{disclaimer}".strip()


def _remove_greetings(text: str) -> str:
    text = re.sub(r"^(Chào bạn,?|Xin chào,?|Chào bạn!|Xin chào!)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(\*\*)?Yes(\*\*)?\s*[,.:;–-]?\s+", "", text, flags=re.IGNORECASE)
    return re.sub(r"^(Hy vọng|Mong rằng) thông tin.*?$", "", text, flags=re.IGNORECASE | re.MULTILINE)


def _strip_unexpected_boolean_prefix(text: str) -> str:
    return re.sub(r"^\s*(Có|Không)[,.]\s+", "", text, count=1, flags=re.IGNORECASE)


def _normalize_common_surface_errors(text: str) -> str:
    replacements = {
        "đổ mồ hồ": "đổ mồ hôi",
        "nặn hoặc chèn": "nặn hoặc bóp",
        "mụn Chỉ dựa": "mụn. Chỉ dựa",
        "tình trạng mụn Chỉ dựa": "tình trạng mụn. Chỉ dựa",
        "ngưng hỏi bác sĩ": "ngưng dùng và hỏi bác sĩ",
    }
    output = text
    for bad, good in replacements.items():
        output = output.replace(bad, good)
    return output


def _remove_source_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.strip().lower().startswith("nguồn:"))


def _normalize_table_spacing(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        previous_is_table = bool(output and _is_table_row(output[-1]))
        next_is_table = bool(index + 1 < len(lines) and _is_table_row(lines[index + 1]))
        if stripped == "" and previous_is_table and next_is_table:
            continue
        output.append(line.rstrip())
    return "\n".join(output)


def _remove_empty_markdown_headings(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _is_heading(line):
            lookahead = index + 1
            while lookahead < len(lines) and not lines[lookahead].strip():
                lookahead += 1
            if lookahead >= len(lines) or _is_heading(lines[lookahead]):
                index = lookahead
                continue
        output.append(line)
        index += 1
    return "\n".join(output)


def _remove_legacy_boilerplate_headings(text: str, profile: ResponseProfile) -> str:
    if profile == "safe_fallback":
        return text
    lines = []
    for line in text.splitlines():
        if line.strip() in LEGACY_BOILERPLATE_HEADINGS:
            continue
        lines.append(line)
    return "\n".join(lines)


def _dedupe_exact_headings(text: str) -> str:
    seen: set[str] = set()
    output: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _is_heading(stripped):
            key = stripped.lower()
            if key in seen:
                continue
            seen.add(key)
        output.append(line)
    return "\n".join(output)


def _dedupe_exact_paragraphs(text: str) -> str:
    seen: set[str] = set()
    output: list[str] = []
    for paragraph in re.split(r"\n{2,}", text):
        key = _normalized_match_text(paragraph)
        if key and key in seen:
            continue
        seen.add(key)
        output.append(paragraph.strip())
    return "\n\n".join(part for part in output if part)


def _dedupe_disclaimer(text: str, disclaimer: str) -> str:
    parts = text.split(disclaimer)
    if len(parts) <= 2:
        return text
    return disclaimer.join(parts[:2]) + "".join(parts[2:])


def _should_add_answer_disclaimer(profile: ResponseProfile) -> bool:
    # The frontend already renders a global medical-information footer. Keep
    # answer-level disclaimers opt-in to avoid repeating generic safety text.
    return False


def _trim_incomplete_terminal_paragraph(text: str) -> str:
    if not _has_incomplete_terminal_sentence(text):
        return text
    paragraphs = [part for part in re.split(r"\n{2,}", text.strip()) if part.strip()]
    if len(paragraphs) <= 1:
        return text.strip() + "."
    return "\n\n".join(paragraphs[:-1]).strip()


def repair_terminal_punctuation(text: str) -> str:
    """Repair a complete final sentence that only lacks terminal punctuation.

    Fragments with a dangling connective may be genuine truncations and are
    deliberately left unchanged for the safe fallback flow to handle.
    """

    if not _has_incomplete_terminal_sentence(text):
        return text
    last = _remove_known_disclaimers(text).strip().splitlines()[-1].strip()
    if _is_heading(last) or _is_table_row(last):
        return text
    folded = _fold(last)
    dangling_endings = (
        " va",
        " nhung",
        " co the",
        " voi",
        " do",
        " vi",
        " anh nang",
        " nhay cam voi anh nang",
        " trong khi",
    )
    if any(folded.endswith(ending) for ending in dangling_endings):
        return text
    if len(last.split()) >= 8 and not re.search(r"[.!?…)]$", last):
        return text.rstrip() + "."
    return text


def _has_incomplete_terminal_sentence(text: str) -> bool:
    clean = _remove_known_disclaimers(text).strip()
    if not clean:
        return False
    last = clean.splitlines()[-1].strip()
    if _is_heading(last):
        return True
    if _is_table_row(last):
        return False
    folded = _fold(last)
    dangling_endings = (
        " va",
        " nhung",
        " co the",
        " voi",
        " do",
        " vi",
        " anh nang",
        " nhay cam voi anh nang",
        " trong khi",
    )
    if any(folded.endswith(ending) for ending in dangling_endings):
        return True
    if len(last.split()) >= 8 and not re.search(r"[.!?…)]$", last):
        return True
    return False


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if re.fullmatch(r"\*\*[^*\n]{2,80}\*\*", stripped):
        return True
    return bool(re.fullmatch(r"#{1,4}\s+\S.{0,80}", stripped))


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_comparison_question(text: str) -> bool:
    folded = _fold(text)
    markers = [
        "khác nhau",
        "khac nhau",
        "khác gì",
        "khac gi",
        "khác thế nào",
        "khac the nao",
        "khác nhau ở",
        "khac nhau o",
        "so sánh",
        "so sanh",
        "đối chiếu",
        "doi chieu",
        " vs ",
        "versus",
    ]
    if any(marker in text or marker in folded for marker in markers):
        return True
    if "thuoc nao" in folded and " va " in f" {folded} ":
        return True
    return bool(re.search(r"\bkhac\b.{0,80}\b(?:the nao|o diem nao|o cho nao|ra sao|gi)\b", folded))


def _is_morning_evening_routine_comparison(text: str) -> bool:
    return _is_comparison_question(text) and "buoi sang" in text and "buoi toi" in text and any(
        marker in text for marker in ("routine", "cham soc", "tri mun")
    )


def _is_back_vs_face_acne_comparison(text: str) -> bool:
    return _is_comparison_question(text) and "mun o lung" in text and "mun o mat" in text


def _is_mild_inflammatory_routine_question(text: str) -> bool:
    folded = _fold(text)
    has_inflammatory_acne = "mun viem" in folded or "mụn viêm" in text
    has_mild_marker = any(marker in folded for marker in ["nhe", "muc do nhe", "nhẹ"])
    asks_routine = any(marker in folded for marker in ["cham soc", "hang ngay", "hằng ngày", "routine"])
    return has_inflammatory_acne and has_mild_marker and asks_routine


def _is_retinoid_shared_class_question(text: str) -> bool:
    folded = _fold(text)
    retinoids = ["adapalene", "adapalen", "tretinoin", "isotretinoin"]
    mentioned_count = sum(1 for value in retinoids if value in folded)
    asks_group = any(
        marker in folded
        for marker in [
            "cung nhom",
            "cùng nhóm",
            "nhom thuoc",
            "nhóm thuốc",
            "thuoc nhom",
            "thuộc nhóm",
            "retinoid",
        ]
    )
    return mentioned_count >= 2 and asks_group and "isotretinoin" in folded


def _is_direct_question(text: str) -> bool:
    return any(marker in text for marker in ["có phải", "co phai", "phải là", "phai la", "là gì", "la gi", "thuộc nhóm", "thuoc nhom", "có nên", "co nen", "không?", "khong?"])


def _is_boolean_question(text: str) -> bool:
    folded = _fold(text)
    return any(
        marker in folded
        for marker in [
            "co phai",
            "co nen",
            "duoc khong",
            "dung duoc khong",
            "khong?",
            "khong",
            "is ",
        ]
    ) and not any(marker in folded for marker in ["gom", "hoat chat nao", "thanh phan nao", "moi hoat chat"])


def _is_high_safety_question(text: str) -> bool:
    folded = _fold(text)
    return any(
        marker in folded
        for marker in [
            "mang thai",
            "co thai",
            "co bau",
            "dang co thai",
            "cho con bu",
            "isotretinoin",
            "mun sau",
            "mun cuc",
            "de lai seo",
            "dau nhieu",
            "sot",
            "tu lam hai",
            "tu hai",
            "self harm",
        ]
    )


_TREATMENT_ENTITY_RULES: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "adapalene",
        "Adapalene",
        ("adapalene", "adapalen", "differin"),
        "retinoid bôi; nên tránh hoặc ngừng trong thai kỳ trừ khi bác sĩ da liễu/sản khoa đánh giá trực tiếp.",
    ),
    (
        "tazarotene",
        "Tazarotene",
        ("tazarotene", "tazaroten", "tazorac"),
        "retinoid bôi nguy cơ cao hơn trong thai kỳ; không nên tự tiếp tục dùng khi đang có thai.",
    ),
    (
        "tretinoin",
        "Tretinoin",
        ("tretinoin",),
        "retinoid bôi; cần tránh tự dùng trong thai kỳ và hỏi bác sĩ về lựa chọn thay thế.",
    ),
    (
        "isotretinoin",
        "Isotretinoin",
        ("isotretinoin",),
        "retinoid đường uống nguy cơ cao; không được tự dùng khi đang hoặc có thể mang thai và cần bác sĩ quản lý nguy cơ.",
    ),
    (
        "doxycycline",
        "Doxycycline",
        ("doxycycline", "doxycyclin"),
        "kháng sinh uống nhóm tetracycline; không tự tiếp tục dùng trong thai kỳ nếu chưa được bác sĩ sản khoa/da liễu xác nhận.",
    ),
    (
        "clindamycin",
        "Clindamycin",
        ("clindamycin", "dalacin"),
        "kháng sinh bôi; cần bác sĩ đánh giá khi đang mang thai, đặc biệt nếu đang phối hợp nhiều thuốc trị mụn.",
    ),
    (
        "benzoyl_peroxide",
        "Benzoyl peroxide",
        ("benzoyl peroxide", "benzoyl peroxid", "bpo", "bp"),
        "hoạt chất bôi không phải kháng sinh; vẫn nên hỏi bác sĩ khi đang mang thai nếu cần dùng thuốc trị mụn.",
    ),
)


def _mentioned_treatment_entities(text: str) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for key, label, aliases, pregnancy_note in _TREATMENT_ENTITY_RULES:
        if any(_contains_entity_alias(text, alias) for alias in aliases):
            found.append((key, label, pregnancy_note))
    return found


def _contains_entity_alias(text: str, alias: str) -> bool:
    alias = _fold(alias)
    if not alias:
        return False
    alias_pattern = re.escape(alias).replace(r"\ ", r"[-_\s]+")
    return re.search(rf"(?<![a-z0-9]){alias_pattern}(?![a-z0-9])", _fold(text)) is not None


def _pregnancy_multi_entity_answer(entities: list[tuple[str, str, str]]) -> str:
    rows = "\n".join(f"- {label}: {note}" for _, label, note in entities)
    return (
        "Trong thai kỳ, bạn không nên tự tiếp tục dùng các thuốc trị mụn đã nêu; hãy liên hệ bác sĩ sản khoa hoặc bác sĩ da liễu càng sớm càng tốt để được hướng dẫn cụ thể.\n\n"
        "## Với từng thuốc\n"
        f"{rows}\n\n"
        "## Việc nên làm ngay\n"
        "- Không tự tăng liều, đổi thuốc hoặc dùng tiếp theo đơn cũ khi chưa được bác sĩ xác nhận.\n"
        "- Ghi lại tên thuốc, dạng dùng, nồng độ/liều nếu có và thời điểm đã dùng để trao đổi với bác sĩ.\n"
        "- Trong lúc chờ tư vấn, ưu tiên chăm sóc nền dịu nhẹ như rửa mặt nhẹ, dưỡng ẩm phù hợp và chống nắng."
    )


def _is_pregnancy_context(text: str) -> bool:
    folded = _fold(text)
    return any(
        marker in folded
        for marker in [
            "mang thai",
            "co thai",
            "dang co thai",
            "co bau",
            "dang bau",
            "thai ky",
            "pregnancy",
            "pregnant",
        ]
    )


def _is_self_harm_crisis_question(text: str) -> bool:
    folded = _fold(text)
    self_harm = any(
        marker in folded
        for marker in [
            "tu lam hai",
            "tu hai",
            "lam hai ban than",
            "hai ban than",
            "self harm",
            "suicide",
            "tu sat",
        ]
    )
    distress = any(marker in folded for marker in ["mat ngu", "ne tranh", "tuyet vong", "tram cam", "lo au"])
    return self_harm and (distress or "mun" in folded or "acne" in folded)


def _is_acne_fulminans_like_question(text: str) -> bool:
    folded = _fold(text)
    severe_lesions = any(marker in folded for marker in ["cuc", "nang viem", "mun nang", "mun cuc"])
    erosive = any(marker in folded for marker in ["trot loet", "loet", "vay xuat huyet", "dong vay xuat huyet"])
    systemic = any(marker in folded for marker in ["sot", "dau khop", "dot ngot"])
    return severe_lesions and erosive and systemic


def _is_antibiotic_monotherapy_and_bp_role_question(text: str) -> bool:
    folded = _fold(text)
    topical_antibiotic = any(marker in folded for marker in ["clindamycin", "erythromycin", "khang sinh boi"])
    oral_antibiotic = any(
        marker in folded
        for marker in [
            "khang sinh uong",
            "khang sinh duong uong",
            "oral antibiotic",
            "doxycycline",
            "lymecycline",
            "minocycline",
            "sarecycline",
        ]
    )
    monotherapy_or_long = any(marker in folded for marker in ["don doc", "don tri lieu", "monotherapy", "keo dai"])
    bp_role = "benzoyl peroxide" in folded and any(marker in folded for marker in ["phoi hop", "vai tro", "ket hop"])
    return bp_role and monotherapy_or_long and (topical_antibiotic or oral_antibiotic)


def _is_severe_acne_question(text: str) -> bool:
    folded = _fold(text)
    return any(marker in folded for marker in ["mun cuc", "mun sau", "cuc mun sau", "de lai seo", "nguy co seo"]) and any(
        marker in folded for marker in ["dau", "sung", "do", "viem"]
    )


def _is_drug_identity_or_composition(text: str) -> bool:
    return any(marker in text for marker in ["differin", "epiduo", "benzoyl peroxide", "adapalene", "clindamycin", "dalacin"]) and any(
        marker in text
        for marker in ["la gi", "thuoc nhom", "hoat chat", "thanh phan", "chua", "gom", "co phai"]
    )


def _question_tail_candidates(question: str) -> set[str]:
    parts = re.split(r"[?.!。！？]\s*", question.strip())
    candidates: set[str] = set()
    for part in parts[-2:]:
        norm = _normalized_match_text(part)
        if 2 <= len(norm.split()) <= 8:
            candidates.add(norm)
    words = _normalized_match_text(question).split()
    if 2 <= len(words[-5:]) <= 8:
        candidates.add(" ".join(words[-5:]))
    return {candidate for candidate in candidates if candidate}


def _fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", text or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("đ", "d")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _normalized_match_text(text: str) -> str:
    folded = _fold(text)
    folded = re.sub(r"[^\w\s]", " ", folded, flags=re.UNICODE)
    return re.sub(r"\s+", " ", folded).strip()


def _struct_issue(code: str, severity: str, message: str) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "evidence": {}, "suggested_fix": None}


__all__ = [
    "ANSWER_FORMATTING_CONTRACT",
    "ANSWER_FORMATTING_CONTRACT_VERSION",
    "CANONICAL_DISCLAIMER",
    "ResponseProfile",
    "answer_format_instruction_for_question",
    "assess_structural_quality",
    "finalize_answer_presentation",
    "grounded_entity_relation_answer",
    "infer_response_profile",
    "normalize_answer_markdown",
    "strip_leading_question_echo",
]
