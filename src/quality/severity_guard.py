"""Deterministic severity-aware answer guard for dermatology/acne questions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.agent.emergency_contract import (
    build_anaphylaxis_like_emergency_answer,
    build_generic_emergency_answer,
    is_anaphylaxis_like_emergency_query,
)
from src.quality.vietnamese_text import build_matching_views


MedicalSeverity = Literal["routine", "caution", "urgent", "emergency"]


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


URGENT_TEMPLATE = (
    "**Tóm tắt ngắn**\n"
    "Trường hợp này không nên chỉ xử lý như mụn thông thường. Bạn nên đi khám bác sĩ/da liễu sớm, "
    "tốt nhất trong 24-48 giờ hoặc sớm hơn nếu triệu chứng tăng nhanh.\n\n"
    "**Lưu ý an toàn**\n"
    "Trong lúc chờ khám, tránh nặn/cạy, ngưng các hoạt chất dễ kích ứng và theo dõi dấu hiệu nặng lên. "
    "Nếu có khó thở, sưng môi/mặt/họng, sốt cao, đau dữ dội, nhìn mờ, phát ban lan nhanh hoặc da phồng rộp/bong tróc, hãy đi cấp cứu ngay."
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


CAUTION_TEMPLATE = (
    "**Lưu ý an toàn**\n"
    "Nếu da đỏ rát, khô bong hoặc châm chích tăng, hãy giảm tần suất hoặc tạm ngưng hoạt chất dễ kích ứng. "
    "Nên hỏi bác sĩ/dược sĩ nếu đang mang thai, cho con bú, có tiền sử dị ứng, hoặc cần phối hợp nhiều hoạt chất trị mụn."
)

TOPICAL_IRRITATION_CAUTION_TEMPLATE = CAUTION_TEMPLATE

ORAL_ISOTRETINOIN_CAUTION_TEMPLATE = (
    "**Lưu ý an toàn**\n"
    "Isotretinoin đường uống cần bác sĩ đủ thẩm quyền chỉ định và theo dõi; không tự dùng, tự tăng liều hoặc dùng theo đơn cũ. "
    "Nếu đang hoặc có thể mang thai, cần báo bác sĩ ngay vì thuốc có nguy cơ cao trong thai kỳ."
)

ANTIBIOTIC_CAUTION_TEMPLATE = (
    "**Lưu ý an toàn**\n"
    "Không tự dùng kháng sinh trị mụn hoặc dùng kéo dài khi chưa có bác sĩ đánh giá. Kháng sinh bôi thường không nên dùng đơn trị liệu; benzoyl peroxide có thể được phối hợp để giảm nguy cơ kháng kháng sinh khi phù hợp."
)

PREGNANCY_CAUTION_TEMPLATE = (
    "**Lưu ý an toàn**\n"
    "Khi đang mang thai, chuẩn bị mang thai hoặc cho con bú, nên hỏi bác sĩ sản khoa/da liễu trước khi dùng thuốc trị mụn, đặc biệt với retinoid hoặc thuốc kê đơn."
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
    answer_text, answer_accentless = build_matching_views(answer)

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
        urgent_text = URGENT_TEMPLATE
        if "urgent_pregnancy_high_risk_acne_medication" in classification.matched_rules:
            urgent_text += "\n\n" + ISOTRETINOIN_PREGNANCY_URGENT_NOTE
        has_required_referral = _has_any(
            answer_accentless,
            ["kham bac si", "gap bac si", "da lieu", "24-48", "24 48", "trong ngay", "kham som"],
        )
        has_isotretinoin_pregnancy_caution = (
            "urgent_pregnancy_high_risk_acne_medication" not in classification.matched_rules
            or _has_any(answer_accentless, ["khong tu dung isotretinoin", "isotretinoin khong duoc tu dung", "chong chi dinh trong thai ky"])
        )
        if has_required_referral and has_isotretinoin_pregnancy_caution:
            return SeverityGuardResult(
                answer=answer,
                original_answer=answer,
                classification=classification,
                modified=False,
                cache_eligible=False,
            )
        return SeverityGuardResult(
            answer=_prepend_once(urgent_text, answer),
            original_answer=answer,
            classification=classification,
            modified=True,
            modification_reason="severity_urgent_safety_preface",
            cache_eligible=False,
        )

    caution_template = _select_caution_template(query, classification)
    has_caution = _answer_already_has_relevant_caution(answer_accentless, caution_template)
    if has_caution:
        return SeverityGuardResult(
            answer=answer,
            original_answer=answer,
            classification=classification,
            modified=False,
            cache_eligible=True,
        )
    return SeverityGuardResult(
        answer=_append_once(answer, caution_template),
        original_answer=answer,
        classification=classification,
        modified=True,
        modification_reason="severity_caution_safety_note",
        cache_eligible=False,
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


def _append_once(answer: str, suffix: str) -> str:
    if suffix.strip() in answer:
        return answer
    if not answer.strip():
        return suffix
    return answer.rstrip() + "\n\n" + suffix


def _select_caution_template(query: str, classification: SeverityClassification) -> str:
    _, accentless = build_matching_views(query or "")
    if "isotretinoin" in accentless:
        return ORAL_ISOTRETINOIN_CAUTION_TEMPLATE
    if _has_any(accentless, ["khang sinh", "antibiotic", "clindamycin", "erythromycin", "doxycycline"]):
        return ANTIBIOTIC_CAUTION_TEMPLATE
    if _has_any(accentless, ["mang thai", "co thai", "co bau", "cho con bu", "thai ky"]):
        return PREGNANCY_CAUTION_TEMPLATE
    return TOPICAL_IRRITATION_CAUTION_TEMPLATE


def _answer_already_has_relevant_caution(answer_accentless: str, template: str) -> bool:
    _, template_accentless = build_matching_views(template)
    if template == TOPICAL_IRRITATION_CAUTION_TEMPLATE:
        return _has_any(
            answer_accentless,
            ["giam tan suat", "tam ngung", "theo doi kich ung", "kich ung", "do rat", "cham chich"],
        )
    if "isotretinoin" in template_accentless:
        return _has_any(answer_accentless, ["isotretinoin"]) and _has_any(
            answer_accentless,
            ["bac si", "theo doi", "khong tu dung", "khong duoc tu dung", "chi dinh"],
        )
    if "khang sinh" in template_accentless:
        return _has_any(answer_accentless, ["khang sinh", "clindamycin", "erythromycin"]) and _has_any(
            answer_accentless,
            ["don tri lieu", "don doc", "khang khang sinh", "khong tu dung", "bac si"],
        )
    if "mang thai" in template_accentless or "thai ky" in template_accentless:
        return _has_any(answer_accentless, ["mang thai", "thai ky", "cho con bu"]) and _has_any(
            answer_accentless,
            ["bac si", "san khoa", "da lieu", "khong tu"],
        )
    return _has_any(answer_accentless, ["bac si", "hoi bac si", "khong tu dung", "theo doi"])


__all__ = [
    "CAUTION_TEMPLATE",
    "ANAPHYLAXIS_LIKE_EMERGENCY_TEMPLATE",
    "EMERGENCY_TEMPLATE",
    "ISOTRETINOIN_PREGNANCY_URGENT_NOTE",
    "ISOTRETINOIN_NEURO_EMERGENCY_TEMPLATE",
    "MedicalSeverity",
    "SeverityClassification",
    "SeverityGuardResult",
    "URGENT_TEMPLATE",
    "apply_severity_aware_answer_guard",
    "classify_medical_severity",
]
