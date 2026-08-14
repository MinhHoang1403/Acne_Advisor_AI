"""
src/agent/nodes/guardrails.py
=============================
Topic guardrail for Acne Advisor AI.
"""

import json
import logging
import re
from src.agent.llm.provider import generate_llm_response

from src.agent.state import ClinicalState
from src.resilience.budget import DeadlineBudget
from src.resilience.contracts import RuntimeResilienceSettings, runtime_resilience_settings_from_env
from src.quality.safe_fallback import sanitize_fallback_reason
from src.quality.vietnamese_text import build_matching_views

logger = logging.getLogger(__name__)


def _has_unsafe_prescription_request(query: str, markers: list[str]) -> bool:
    """Recognize a request to prescribe while preserving safety-constrained questions."""

    _, folded_query = build_matching_views(query)
    for marker in markers:
        _, folded_marker = build_matching_views(marker)
        start = 0
        while (position := folded_query.find(folded_marker, start)) >= 0:
            prefix = folded_query[max(0, position - 24) : position].strip()
            if not re.search(r"(?:^|\s)(?:khong|dung)(?:\s+can|\s+yeu\s+cau)?$", prefix):
                return True
            start = position + len(folded_marker)
    return False


def _runtime_settings(state: ClinicalState) -> RuntimeResilienceSettings:
    configured = state.get("runtime_resilience_settings")
    if isinstance(configured, dict):
        return RuntimeResilienceSettings(**configured)
    return runtime_resilience_settings_from_env()


def _runtime_budget(state: ClinicalState, settings: RuntimeResilienceSettings) -> DeadlineBudget:
    budget = state.get("runtime_budget")
    if isinstance(budget, DeadlineBudget):
        return budget
    return DeadlineBudget.from_timeout(settings.agent_total_timeout_seconds)

async def domain_guard_node(state: ClinicalState) -> dict:
    """
    Check if the question is in-domain (acne, dermatology, skincare).
    Classifies into: in_domain, out_of_domain, partial.
    If partial, rewrites the standalone_question to only include the acne part.
    """
    # The question could have been rewritten by the history context node
    # Use standalone_question if available, otherwise user_question
    query = state.get("standalone_question") or state.get("user_question", "")
    query_lower = query.lower()
    history_text = " ".join(
        str(msg.get("content", ""))
        for msg in (state.get("conversation_history") or [])
    ).lower()
    
    logger.info("Running domain guardrail...")

    if not query_lower.strip():
        return {
            "is_in_domain": True,
            "guardrail": "empty_query",
            "ignored_out_of_domain_part": False,
            "domain_reason": "Empty query routed to safe fallback.",
            "refusal_message": None,
        }

    prompt_injection_markers = [
        "bỏ qua hướng dẫn",
        "ignore previous",
        "ignore instructions",
        "bỏ qua quy tắc",
        "không cần tuân thủ",
        "jailbreak",
        "giả vờ bạn là bác sĩ",
        "giả vờ là bác sĩ",
        "pretend you are a doctor",
    ]
    unsafe_cyber_markers = ["hack tài khoản", "hack facebook", "đánh cắp mật khẩu", "lấy mật khẩu"]
    prescription_markers = [
        "kê isotretinoin",
        "kê đơn",
        "kê thuốc",
        "cho tôi đơn",
        "cho tôi toa",
        "toa thuốc",
        "liều cao",
        "cho tôi liều",
    ]
    high_risk_meds = ["isotretinoin", "kháng sinh", "antibiotic", "retinoid"]

    if any(marker in query_lower for marker in unsafe_cyber_markers):
        return {
            "is_in_domain": False,
            "guardrail": "unsafe_out_of_domain",
            "ignored_out_of_domain_part": False,
            "domain_reason": "Unsafe cyber request.",
            "refusal_message": (
                "Tôi không thể hỗ trợ viết code hack, chiếm quyền tài khoản hoặc xâm nhập hệ thống. "
                "Tôi chỉ hỗ trợ câu hỏi về mụn trứng cá, chăm sóc da mụn và thông tin da liễu an toàn."
            )
        }

    if any(marker in query_lower for marker in prompt_injection_markers) and any(marker in query_lower for marker in prescription_markers):
        return {
            "is_in_domain": False,
            "guardrail": "unsafe_prescription_request",
            "ignored_out_of_domain_part": False,
            "domain_reason": "Prompt-injection style prescription request.",
            "refusal_message": (
                "Tôi không thể giả vờ là bác sĩ, kê đơn hoặc chọn thuốc trị mụn thay bác sĩ. "
                "Nếu bạn cần thuốc kê đơn, hãy gặp bác sĩ da liễu để được đánh giá trực tiếp."
            )
        }

    if any(marker in query_lower for marker in prompt_injection_markers) and any(med in query_lower for med in high_risk_meds):
        return {
            "is_in_domain": False,
            "guardrail": "unsafe_prescription_request",
            "ignored_out_of_domain_part": False,
            "domain_reason": "Prompt-injection style request for high-risk medication.",
            "refusal_message": (
                "Tôi không thể bỏ qua hướng dẫn an toàn hoặc kê liều thuốc trị mụn nguy cơ cao. "
                "Isotretinoin, retinoid và kháng sinh cần được bác sĩ đánh giá, kê đơn và theo dõi khi phù hợp."
            )
        }

    if _has_unsafe_prescription_request(query, prescription_markers) and any(med in query_lower for med in high_risk_meds):
        return {
            "is_in_domain": False,
            "guardrail": "unsafe_prescription_request",
            "ignored_out_of_domain_part": False,
            "domain_reason": "Unsafe prescription or dosage request.",
            "refusal_message": (
                "Tôi không thể kê đơn, chọn liều hoặc hướng dẫn dùng liều cao thuốc trị mụn. "
                "Bạn cần bác sĩ da liễu đánh giá trực tiếp nếu đang cân nhắc thuốc kê đơn."
            )
        }

    allergy_markers = ["sưng môi", "sưng mặt", "sưng lưỡi", "nổi mề đay", "mề đay", "sốc phản vệ"]
    systemic_emergency_markers = ["khó thở", "ngất", "choáng", "tim đập nhanh", "tím tái"]
    if any(marker in query_lower for marker in allergy_markers) and any(marker in query_lower for marker in systemic_emergency_markers):
        return {
            "is_in_domain": False,
            "guardrail": "medical_emergency_allergy",
            "ignored_out_of_domain_part": False,
            "domain_reason": "Possible allergic reaction or anaphylaxis.",
            "refusal_message": (
                "Các dấu hiệu như sưng môi, khó thở, nổi mề đay, choáng hoặc tim đập nhanh sau khi dùng thuốc/sản phẩm có thể là phản ứng dị ứng nặng. "
                "Bạn nên ngừng sản phẩm nghi ngờ và đi cấp cứu hoặc gọi hỗ trợ y tế khẩn cấp ngay nếu triệu chứng đang xảy ra."
            )
        }

    urgent_skin_eye_markers = ["sốt cao", "đau lan quanh mắt", "quanh mắt", "nhìn mờ", "đỏ lan nhanh", "sưng đau quanh mắt"]
    if ("mụn" in query_lower or "da" in query_lower) and (
        ("sốt cao" in query_lower and any(marker in query_lower for marker in urgent_skin_eye_markers))
        or "nhìn mờ" in query_lower
        or "đỏ lan nhanh" in query_lower
    ):
        return {
            "is_in_domain": False,
            "guardrail": "urgent_skin_eye_infection",
            "ignored_out_of_domain_part": False,
            "domain_reason": "Possible urgent skin/eye infection signs.",
            "refusal_message": (
                "Mụn hoặc vùng da sưng đau kèm sốt cao, đau lan quanh mắt, nhìn mờ hoặc đỏ lan nhanh là dấu hiệu cần khám khẩn. "
                "Bạn nên đi khám cấp cứu hoặc cơ sở y tế ngay, đặc biệt nếu vùng quanh mắt sưng đau hoặc thị lực thay đổi."
            )
        }

    emergency_markers = [
        "đau ngực",
        "khó thở",
        "tức ngực",
        "ngất",
        "choáng",
        "tim đập nhanh",
        "đau bụng dữ dội",
        "đau bụng rất dữ",
        "severe abdominal pain",
        "chảy máu nhiều",
        "yếu liệt đột ngột",
    ]
    if any(marker in query_lower for marker in emergency_markers):
        return {
            "is_in_domain": False,
            "guardrail": "medical_emergency_out_of_scope",
            "ignored_out_of_domain_part": False,
            "domain_reason": "Potential urgent non-acne symptom.",
            "refusal_message": (
                "Các triệu chứng bạn mô tả không phải biểu hiện điển hình của mụn. "
                "Nếu triệu chứng đang xảy ra hoặc nặng lên, bạn nên đi cấp cứu hoặc liên hệ cơ sở y tế ngay."
            )
        }
    
    # 1. Rule-based out-of-domain checks first to save LLM calls and avoid hallucination/errors on common OOD queries
    ood_exact_matches = [
        "tôi tên gì", "tôi tên gì?", "tên tôi là gì", "tên tôi là gì?", "bạn biết tên tôi không",
        "bạn tên là gì", "bạn tên là gì?", "tên bạn là gì", "tên bạn là gì?",
        "thời tiết hôm nay thế nào", "thời tiết hôm nay thế nào?", "thời tiết hôm nay",
        "thời tiết ở", "thời tiết tp.hcm", "thời tiết tphcm",
        "hôm nay ngày mấy", "hôm nay là ngày mấy", "hôm nay ngày mấy?",
        "bạn là ai", "bạn là ai?", "kể chuyện cười", "viết code", "làm toán"
    ]
    
    for match in ood_exact_matches:
        if match in query_lower:
            return {
                "is_in_domain": False,
                "guardrail": "out_of_domain",
                "ignored_out_of_domain_part": False,
                "domain_reason": "Rule-based out of domain match",
                "refusal_message": "Xin lỗi, tôi chỉ hỗ trợ các câu hỏi liên quan đến mụn trứng cá, chăm sóc da mụn và thông tin da liễu liên quan. Bạn có thể hỏi tôi về triệu chứng mụn, hoạt chất trị mụn, routine chăm sóc da hoặc khi nào nên gặp bác sĩ da liễu."
            }

    in_domain_markers = [
        "mụn",
        "trứng cá",
        "acne",
        "benzoyl peroxide",
        "adapalene",
        "clindamycin",
        "erythromycin",
        "doxycycline",
        "dalacin",
        "azelaic acid",
        "salicylic acid",
        "differin",
        "epiduo",
        "tazarotene",
        "tazaroten",
        "tazorac",
        "retinoid",
        "isotretinoin",
        "kháng sinh",
        "da dầu",
        "mụn đầu đen",
        "kem đánh răng",
        "mang thai",
        "cho con bú",
        "chăm sóc da",
        "routine",
        "bôi thuốc",
        "thuốc trị mụn",
    ]
    followup_markers = [
        "vậy",
        "chăm sóc",
        "bắt đầu",
        "routine",
        "có cần",
        "nhắc lại",
        "tình trạng da",
        "hoạt chất",
        "lúc này",
        "nên làm gì",
    ]
    if any(marker in query_lower for marker in in_domain_markers):
        return {
            "is_in_domain": True,
            "guardrail": "in_domain_rule",
            "ignored_out_of_domain_part": False,
            "domain_reason": "Rule-based in-domain dermatology/acne match.",
            "refusal_message": None
        }

    if any(marker in history_text for marker in in_domain_markers) and any(marker in query_lower for marker in followup_markers):
        return {
            "is_in_domain": True,
            "guardrail": "in_domain_followup_rule",
            "ignored_out_of_domain_part": False,
            "domain_reason": "Rule-based acne follow-up matched from conversation history.",
            "refusal_message": None
        }

    prompt = f"""
Bạn là một bộ lọc chủ đề cho Acne Advisor AI - một trợ lý y khoa chuyên về mụn trứng cá, chăm sóc da mụn, và da liễu.
Nhiệm vụ của bạn là phân loại câu hỏi sau đây thuộc loại nào trong 3 loại:
1. "in_domain": Câu hỏi hoàn toàn liên quan đến mụn trứng cá, chăm sóc da, thuốc bôi, bệnh da liễu, hỏi thông tin bác sĩ da liễu, hoặc triệu chứng trên da. (Ví dụ: "Bôi BHA có đẩy mụn không?")
2. "out_of_domain": Câu hỏi hoàn toàn KHÔNG liên quan đến mụn/da liễu. (Ví dụ: "Tổng thống Mỹ là ai?", "Thời tiết hôm nay?", "Viết code Python", "Tôi tên gì?")
3. "partial": Câu hỏi kết hợp cả ngoài lề và một phần về mụn/da liễu. (Ví dụ: "Dạo này kinh tế đi xuống quá, mà mặt tôi lại mọc đầy mụn viêm, tôi phải làm sao?")

CHỈ TRẢ VỀ CHUỖI JSON ĐÚNG ĐỊNH DẠNG SAU, KHÔNG CÓ MARKDOWN HAY CHỮ GÌ KHÁC:
{{
  "classification": "in_domain" | "out_of_domain" | "partial",
  "acne_query": "NẾU partial, hãy trích xuất hoặc viết lại CHỈ phần câu hỏi liên quan đến mụn/da liễu. NẾU in_domain, để nguyên câu hỏi. NẾU out_of_domain, để rỗng",
  "reason": "Lý do phân loại ngắn gọn"
}}

Câu hỏi cần phân loại:
"{query}"
"""

    try:
        llm_provider = state.get("llm_provider", "gemini")
        llm_model = state.get("llm_model")
        allow_model_fallback = state.get("allow_model_fallback", True)
        settings = _runtime_settings(state)
        
        response_data = await generate_llm_response(
            prompt=prompt,
            provider=llm_provider,
            model=llm_model,
            temperature=0.0,
            allow_fallback=allow_model_fallback,
            use_sync=False,
            budget=_runtime_budget(state, settings),
            resilience_settings=settings,
        )
        
        result_str = response_data["text"].strip()
        # Clean markdown code block if present
        if result_str.startswith("```json"):
            result_str = result_str[7:]
        if result_str.startswith("```"):
            result_str = result_str[3:]
        if result_str.endswith("```"):
            result_str = result_str[:-3]
        result_str = result_str.strip()
            
        result = json.loads(result_str)
        
        classification = result.get("classification")
        acne_query = result.get("acne_query", "")
        reason = result.get("reason", "")
        
        logger.info(f"Guardrail classification: {classification}. Reason: {reason}")
        
        if classification == "out_of_domain":
            return {
                "is_in_domain": False,
                "guardrail": "out_of_domain",
                "ignored_out_of_domain_part": False,
                "domain_reason": reason,
                "refusal_message": "Xin lỗi, tôi chỉ hỗ trợ các câu hỏi liên quan đến mụn trứng cá, chăm sóc da mụn và thông tin da liễu liên quan. Bạn có thể hỏi tôi về triệu chứng mụn, hoạt chất trị mụn, routine chăm sóc da hoặc khi nào nên gặp bác sĩ da liễu."
            }
        elif classification == "partial":
            return {
                "is_in_domain": True,
                "guardrail": "partial_out_of_domain",
                "ignored_out_of_domain_part": True,
                "domain_reason": reason,
                "standalone_question": acne_query if acne_query else query, # Rewrite standalone to focus on acne
                "refusal_message": None
            }
        else:
            return {
                "is_in_domain": True,
                "guardrail": "in_domain",
                "ignored_out_of_domain_part": False,
                "domain_reason": reason,
                "refusal_message": None
            }
            
    except Exception as e:
        logger.error(
            "Guardrail failed (%s). Defaulting to safe fallback (in_domain).",
            sanitize_fallback_reason(e),
        )
        # Safe fallback: assume in-domain if LLM fails, safety checks later might catch some issues
        # Or simple rule check
        keyword_fallback = any(k in query.lower() for k in ["mụn", "acne", "da", "sẹo", "skincare", "rửa mặt", "tác dụng phụ", "bôi", "uống", "thuốc", "loại đó", "cái đó", "cách dùng", "điều trị"])
        
        if keyword_fallback:
             return {
                "is_in_domain": True,
                "guardrail": "in_domain_fallback",
                "ignored_out_of_domain_part": False,
                "domain_reason": "Fallback due to LLM error, keyword matched.",
                "refusal_message": None
            }
        else:
            # If no keyword, block to be safe when API fails
             return {
                "is_in_domain": False,
                "guardrail": "out_of_domain_fallback",
                "ignored_out_of_domain_part": False,
                "domain_reason": "Fallback due to LLM error, no keyword matched.",
                "refusal_message": "Xin lỗi, tôi chỉ hỗ trợ các câu hỏi liên quan đến mụn trứng cá, chăm sóc da mụn và thông tin da liễu liên quan. Bạn có thể hỏi tôi về triệu chứng mụn, hoạt chất trị mụn, routine chăm sóc da hoặc khi nào nên gặp bác sĩ da liễu."
            }
