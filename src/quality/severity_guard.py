"""Deterministic severity-aware answer guard for dermatology/acne questions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.agent.emergency_contract import (
    build_anaphylaxis_like_emergency_answer,
    build_generic_emergency_answer,
    is_anaphylaxis_like_emergency_query,
)
from src.quality.vietnamese_text import build_matching_views


MedicalSeverity = Literal["routine", "caution", "urgent", "emergency"]

SAFETY_POLICY_PROVENANCE = {
    "emergency": [
        "NHS_ANAPHYLAXIS",
        "DAILYMED_ISOTRETINOIN_MEDICATION_GUIDE",
    ],
    "self_harm": ["WHO_SUICIDE_QA"],
    "acne_fulminans": ["NICE_NG198_RECOMMENDATION_1_4_1"],
    "isotretinoin_pregnancy": ["NICE_NG198_RECOMMENDATION_1_5_22", "MHRA_ISOTRETINOIN_PPP"],
}


class SeverityClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: MedicalSeverity
    matched_rules: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class SeverityGuardResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    original_answer: str
    classification: SeverityClassification
    modified: bool
    modification_reason: str | None = None
    cache_eligible: bool = True


EMERGENCY_TEMPLATE = build_generic_emergency_answer()
ANAPHYLAXIS_LIKE_EMERGENCY_TEMPLATE = build_anaphylaxis_like_emergency_answer()

ISOTRETINOIN_NEURO_EMERGENCY_TEMPLATE = (
    "**Tóm tắt ngắn**\n"
    "Bạn nên liên hệ bác sĩ/cơ sở y tế khẩn cấp ngay. Đau đầu dữ dội kèm nhìn mờ và buồn nôn/nôn khi đang dùng isotretinoin là nhóm triệu chứng cần được đánh giá ngay, không nên chờ 24-48 giờ.\n\n"
    "**Việc nên làm ngay**\n"
    "- Ngừng tự uống liều tiếp theo cho đến khi được bác sĩ đang điều trị hướng dẫn.\n"
    "- Không tự tăng/giảm liều hoặc dùng thêm thuốc khác để xử lý triệu chứng.\n"
    "- Mang theo tên thuốc, liều đang dùng và thời điểm xuất hiện triệu chứng khi đi khám.\n\n"
    "**Lưu ý**\n"
    "Tôi không thể chẩn đoán qua chat, nhưng đây không phải tình huống nên chỉ theo dõi tại nhà."
)


SELF_HARM_CRISIS_TEMPLATE = (
    "**Tóm tắt ngắn**\n"
    "An toàn của bạn là ưu tiên trước điều trị mụn. Ý nghĩ tự làm hại bản thân là dấu hiệu cần được hỗ trợ trực tiếp, không nên ở một mình với nguy cơ này.\n\n"
    "**Việc nên làm ngay**\n"
    "- Nếu bạn có nguy cơ hành động ngay hoặc thấy mình không an toàn, hãy gọi cấp cứu hoặc đến cơ sở y tế khẩn cấp ngay.\n"
    "- Hãy gọi hoặc nhắn cho một người đáng tin cậy và nhờ họ ở bên bạn ngay lúc này.\n"
    "- Tránh rượu/chất kích thích và cất xa vật dụng có thể gây hại nếu làm được an toàn.\n\n"
    "**Sau khi đã an toàn**\n"
    "Bạn có thể hẹn bác sĩ da liễu để xử lý mụn và trao đổi thêm với bác sĩ tâm lý/bác sĩ gia đình về mất ngủ, né tránh giao tiếp hoặc ý nghĩ tự làm hại. Tôi không thể chẩn đoán tâm thần qua chat."
)


ACNE_FULMINANS_URGENT_TEMPLATE = (
    "**Tóm tắt ngắn**\n"
    "Mô tả này gợi ý mụn rất nặng và có thể nghi acne fulminans, nhưng không thể chẩn đoán chắc chắn qua chat.\n\n"
    "**Mức độ khẩn cấp**\n"
    "- Bạn nên được bác sĩ da liễu hoặc cơ sở y tế đánh giá/chuyển khẩn trong ngày.\n"
    "- Nếu có sốt, đau khớp, tổn thương trợt loét hoặc vảy xuất huyết, nên được đánh giá trong vòng 24 giờ.\n"
    "- Không tự dùng isotretinoin, kháng sinh uống hoặc thuốc kê đơn khi chưa được bác sĩ chỉ định.\n\n"
    "**Trong lúc chờ khám**\n"
    "Tránh nặn/cạy, không chà xát mạnh và mang theo danh sách thuốc/sản phẩm đang dùng khi đi khám."
)


ISOTRETINOIN_PREGNANCY_URGENT_NOTE = (
    "Isotretinoin không được tự dùng khi đang mang thai, chuẩn bị mang thai hoặc nghi ngờ có thai; "
    "cần bác sĩ chuyên khoa đánh giá và quản lý nguy cơ."
)


def classify_medical_severity(query: str) -> SeverityClassification:
    """Classify query severity with Vietnamese-first deterministic rules."""

    text, accentless = build_matching_views(query or "")
    rules: list[str] = []
    evidence: list[str] = []

    def mark(rule: str, *items: str) -> None:
        rules.append(rule)
        evidence.extend(item for item in items if item)

    # Emergency: airway/allergy, severe drug rash, systemic infection/necrosis.
    drug_exposure = _has_any(
        accentless,
        [
            "sau khi dung thuoc",
            "sau dung thuoc",
            "sau mot thuoc",
            "sau thuoc",
            "thuoc moi",
            "uong thuoc",
            "boi thuoc",
            "dung isotretinoin",
        ],
    )
    drug_reaction_context = drug_exposure or _has_any(
        accentless,
        [
            "sau khi dung",
            "sau khi uong",
            "sau khi boi",
            "sau san pham moi",
            "sau phan ung thuoc",
            "sau thuoc tri mun",
        ],
    )
    airway_or_systemic_alarm = _has_any(
        accentless,
        [
            "sung moi",
            "sung luoi",
            "sung hong",
            "moi sung",
            "luoi sung",
            "hong sung",
            "kho tho",
            "tho gap",
            "kho khe",
            "ngat",
            "choang",
            "khong tinh tao",
        ],
    )
    if drug_reaction_context and _has_any(accentless, ["phan ve", "phan ung di ung nang"]):
        mark("emergency_explicit_anaphylaxis_or_severe_allergic_reaction", "nghi phản vệ/phản ứng dị ứng nặng sau thuốc")
        return SeverityClassification(severity="emergency", matched_rules=rules, evidence=evidence)

    if drug_reaction_context and _has_any(accentless, ["goi cap cuu", "can cap cuu"]):
        mark("emergency_drug_reaction_emergency_action_question", "hỏi dấu hiệu cần gọi cấp cứu sau thuốc")
        return SeverityClassification(severity="emergency", matched_rules=rules, evidence=evidence)

    if drug_reaction_context and airway_or_systemic_alarm:
        mark("emergency_airway_or_systemic_drug_reaction", "dấu hiệu đường thở/toàn thân sau dùng thuốc")
        if is_anaphylaxis_like_emergency_query(query):
            mark("emergency_anaphylaxis_like_reaction", "khó thở kèm sưng/phát ban")
        return SeverityClassification(severity="emergency", matched_rules=rules, evidence=evidence)

    if drug_exposure and _has_any(
        accentless,
        ["sung moi", "sung luoi", "sung hong", "moi sung", "luoi sung", "hong sung"],
    ):
        mark("emergency_airway_swelling_after_drug", "sưng môi/lưỡi/họng sau dùng thuốc")
        return SeverityClassification(severity="emergency", matched_rules=rules, evidence=evidence)

    if is_anaphylaxis_like_emergency_query(query):
        mark("emergency_anaphylaxis_like_reaction", "khó thở kèm sưng/phát ban")
        return SeverityClassification(severity="emergency", matched_rules=rules, evidence=evidence)

    if _has_any(accentless, ["kho tho", "tho gap", "tuc nguc"]) and _has_any(
        accentless,
        ["sung moi", "sung mat", "sung hong", "sung luoi", "me day", "phat ban"],
    ):
        mark("emergency_anaphylaxis_like_reaction", "khó thở/sưng/phát ban")
        return SeverityClassification(severity="emergency", matched_rules=rules, evidence=evidence)

    severe_mucocutaneous = _has_any(
        accentless,
        ["phong rop", "bong troc da", "bong troc dien rong", "loet mieng", "loet mat", "loet sinh duc", "niem mac"],
    )
    systemic_or_eye_alarm = _has_any(accentless, ["sot", "mat dau", "dau rat mat", "dau mat"])
    if severe_mucocutaneous and drug_exposure and (systemic_or_eye_alarm or _has_any(accentless, ["phong rop", "loet mieng", "niem mac"])):
        mark("emergency_severe_drug_rash_sjs_ten_like", "phồng rộp/loét/sốt sau dùng thuốc")
        return SeverityClassification(severity="emergency", matched_rules=rules, evidence=evidence)

    if _has_any(accentless, ["phat ban toan than", "noi me day toan than"]) and _has_any(
        accentless,
        ["sau khi dung thuoc", "uong thuoc", "boi thuoc", "kho tho", "sung moi", "sung mat"],
    ):
        mark("emergency_generalized_drug_rash", "phát ban toàn thân sau dùng thuốc")
        return SeverityClassification(severity="emergency", matched_rules=rules, evidence=evidence)

    if _has_any(accentless, ["da tim den", "hoai tu"]) or (
        _has_any(accentless, ["chay mu", "mu nhieu"])
        and _has_any(accentless, ["sot", "dau du doi", "lan nhanh", "sung nhanh"])
    ):
        mark("emergency_severe_skin_infection_or_necrosis", "da tím đen/hoại tử/chảy mủ kèm dấu hiệu nặng")
        return SeverityClassification(severity="emergency", matched_rules=rules, evidence=evidence)

    if _has_any(accentless, ["sot cao"]) and _has_any(accentless, ["phat ban nang", "phat ban lan nhanh"]):
        mark("emergency_high_fever_with_severe_rash", "sốt cao kèm phát ban nặng")
        return SeverityClassification(severity="emergency", matched_rules=rules, evidence=evidence)

    if (
        "isotretinoin" in accentless
        and _has_any(accentless, ["dau dau du doi", "dau dau nang"])
        and _has_any(accentless, ["nhin mo", "mo mat", "giam thi luc"])
        and _has_any(accentless, ["buon non", "non"])
    ):
        mark("emergency_isotretinoin_neurologic_symptoms", "isotretinoin kèm đau đầu dữ dội/nhìn mờ/buồn nôn")
        return SeverityClassification(severity="emergency", matched_rules=rules, evidence=evidence)

    if _has_any(
        accentless,
        ["tu lam hai", "tu hai", "lam hai ban than", "hai ban than", "tu sat", "self harm", "suicide"],
    ):
        mark("urgent_self_harm_ideation", "ý nghĩ tự làm hại bản thân")
        return SeverityClassification(severity="urgent", matched_rules=rules, evidence=evidence)

    # Urgent: same-day/24-48h clinician review.
    if (
        _has_any(accentless, ["cuc", "nang viem", "mun nang", "mun cuc"])
        and _has_any(accentless, ["trot loet", "loet", "vay xuat huyet", "dong vay xuat huyet"])
        and _has_any(accentless, ["sot", "dau khop", "dot ngot"])
    ):
        mark("urgent_acne_fulminans_like", "mụn cục/nang trợt loét kèm sốt/đau khớp")
        return SeverityClassification(severity="urgent", matched_rules=rules, evidence=evidence)

    if _has_any(accentless, ["quanh mat", "gan mat", "mi mat", "sung mi", "dau mat"]) and _has_any(
        accentless,
        ["sung", "do", "dau", "chay mu", "nhin mo"],
    ):
        mark("urgent_eye_area_acne_or_infection", "mụn/vùng da gần mắt sưng đau/chảy mủ")
        return SeverityClassification(severity="urgent", matched_rules=rules, evidence=evidence)

    if "isotretinoin" in accentless and _has_any(
        accentless,
        ["dau dau du doi", "nhin mo", "dau bung nang", "vang da", "tram cam nang", "y nghi tu hai", "tu hai"],
    ):
        mark("urgent_isotretinoin_concerning_symptoms", "isotretinoin kèm triệu chứng đáng lo")
        return SeverityClassification(severity="urgent", matched_rules=rules, evidence=evidence)

    if _has_any(accentless, ["mang thai", "co thai", "co bau", "dang bau", "chuan bi mang thai", "cho con bu"]) and _has_any(
        accentless,
        ["isotretinoin", "retinoid duong uong", "thuoc nguy co cao"],
    ):
        mark("urgent_pregnancy_high_risk_acne_medication", "thai kỳ/cho bú với isotretinoin hoặc retinoid nguy cơ cao")
        return SeverityClassification(severity="urgent", matched_rules=rules, evidence=evidence)

    if _has_any(accentless, ["ap xe", "nghi nhiem trung", "sau nan mun"]) or (
        _has_any(accentless, ["sung to", "dau nhieu", "do nong dau", "chay mu"])
        and _has_any(accentless, ["lan nhanh", "mun", "not viem", "viem"])
    ):
        mark("urgent_possible_skin_infection_or_abscess", "áp xe/nhiễm trùng hoặc nốt viêm nặng")
        return SeverityClassification(severity="urgent", matched_rules=rules, evidence=evidence)

    if _has_any(accentless, ["tre so sinh", "tre nho", "em be"]) and _has_any(accentless, ["nhiem trung", "chay mu", "sot"]):
        mark("urgent_child_skin_infection", "trẻ nhỏ/sơ sinh có dấu hiệu nhiễm trùng da")
        return SeverityClassification(severity="urgent", matched_rules=rules, evidence=evidence)

    # Caution: active ingredients, mild irritation, pregnancy/breastfeeding routine care.
    if _has_any(
        accentless,
        [
            "benzoyl peroxide",
            "bpo",
            " bp ",
            "adapalene",
            "tretinoin",
            "retinoid",
            "retinol",
            "aha",
            "bha",
            "clindamycin",
            "erythromycin",
            "khang sinh boi",
            "khang sinh uong",
            "antibiotic",
        ],
    ):
        mark("caution_acne_active_or_antibiotic_question", "hoạt chất/kháng sinh trị mụn")
        return SeverityClassification(severity="caution", matched_rules=rules, evidence=evidence)

    if _has_any(accentless, ["do rat nhe", "kich ung nhe", "bong troc nhe", "kho da", "cham chich"]):
        mark("caution_mild_irritation", "kích ứng nhẹ")
        return SeverityClassification(severity="caution", matched_rules=rules, evidence=evidence)

    if _has_any(accentless, ["di ung thuoc", "di ung my pham", "tien su di ung", "mang thai", "co thai", "cho con bu"]):
        mark("caution_history_or_pregnancy_context", "tiền sử dị ứng/thai kỳ/cho bú")
        return SeverityClassification(severity="caution", matched_rules=rules, evidence=evidence)

    return SeverityClassification(severity="routine", matched_rules=["routine_default"], evidence=[])


def apply_severity_aware_answer_guard(query: str, answer: str) -> SeverityGuardResult:
    """Ensure the final answer matches the medical severity of the user query."""

    classification = classify_medical_severity(query)
    answer = answer or ""

    if classification.severity == "routine":
        return SeverityGuardResult(
            answer=answer,
            original_answer=answer,
            classification=classification,
            modified=False,
            cache_eligible=True,
        )

    if classification.severity == "emergency":
        template = (
            ISOTRETINOIN_NEURO_EMERGENCY_TEMPLATE
            if "emergency_isotretinoin_neurologic_symptoms" in classification.matched_rules
            else ANAPHYLAXIS_LIKE_EMERGENCY_TEMPLATE
            if "emergency_anaphylaxis_like_reaction" in classification.matched_rules
            else EMERGENCY_TEMPLATE
        )
        return SeverityGuardResult(
            answer=template,
            original_answer=answer,
            classification=classification,
            modified=True,
            modification_reason="severity_emergency_safety_fallback",
            cache_eligible=False,
        )

    if classification.severity == "urgent":
        if "urgent_self_harm_ideation" in classification.matched_rules:
            return SeverityGuardResult(
                answer=SELF_HARM_CRISIS_TEMPLATE,
                original_answer=answer,
                classification=classification,
                modified=True,
                modification_reason="severity_self_harm_crisis_preface",
                cache_eligible=False,
            )
        if "urgent_acne_fulminans_like" in classification.matched_rules:
            return SeverityGuardResult(
                answer=ACNE_FULMINANS_URGENT_TEMPLATE,
                original_answer=answer,
                classification=classification,
                modified=True,
                modification_reason="severity_acne_fulminans_urgent_preface",
                cache_eligible=False,
            )
        if "urgent_pregnancy_high_risk_acne_medication" in classification.matched_rules:
            return SeverityGuardResult(
                answer=_prepend_once(ISOTRETINOIN_PREGNANCY_URGENT_NOTE, answer),
                original_answer=answer,
                classification=classification,
                modified=True,
                modification_reason="severity_pregnancy_high_risk_medication_preface",
                cache_eligible=False,
            )
        return SeverityGuardResult(
            answer=answer,
            original_answer=answer,
            classification=classification,
            modified=False,
            cache_eligible=False,
        )

    # Caution is metadata-only. Ordinary medication meaning must come from
    # retrieved evidence and the LLM, never from a deterministic suffix table.
    return SeverityGuardResult(
        answer=answer,
        original_answer=answer,
        classification=classification,
        modified=False,
        cache_eligible=True,
    )


def _has_any(text: str, needles: list[str]) -> bool:
    padded = f" {text} "
    return any(needle in padded for needle in needles)


def _prepend_once(prefix: str, answer: str) -> str:
    if not answer.strip():
        return prefix
    if prefix.strip() in answer:
        return answer
    return prefix.rstrip() + "\n\n**Thông tin thêm**\n" + answer.strip()


__all__ = [
    "ANAPHYLAXIS_LIKE_EMERGENCY_TEMPLATE",
    "EMERGENCY_TEMPLATE",
    "ISOTRETINOIN_PREGNANCY_URGENT_NOTE",
    "ISOTRETINOIN_NEURO_EMERGENCY_TEMPLATE",
    "MedicalSeverity",
    "SAFETY_POLICY_PROVENANCE",
    "SeverityClassification",
    "SeverityGuardResult",
    "apply_severity_aware_answer_guard",
    "classify_medical_severity",
]
