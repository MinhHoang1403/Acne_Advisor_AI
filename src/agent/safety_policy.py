"""Các safety override deterministic, hẹp và gắn với nguồn cụ thể.

Module chuẩn hóa câu hỏi để so khớp một inventory hữu hạn các tình huống cần
phản hồi cố định. Nó không phải bộ phân loại y khoa tổng quát, không chẩn đoán và
không thay thế retrieval/LLM cho câu hỏi thông thường. Rule đầu tiên khớp sẽ được
áp dụng; response và source IDs là contract runtime nên không dịch hoặc sửa khi
chỉ bảo trì comment.

Muốn thêm safety case phải bắt đầu từ ``SAFETY_RULES`` và test cả khẳng định lẫn
phủ định để tránh false positive từ phrase matching.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Literal

from src.agent.semantic_signals import (
    BREATHING_DIFFICULTY_CONCEPTS,
    contains_bounded_sequence,
    has_active_symptom,
    has_first_person_reference,
    has_local_concept_groups,
    has_medication_related_active_symptom,
    is_prescription_execution_request,
    normalize_text,
)

SafetySeverity = Literal["policy", "urgent", "emergency"]
SAFETY_POLICY_VERSION = "source_mapped_safety_policy_v6"


@dataclass(frozen=True)
class SafetyRule:
    """Một rule bất biến gồm trigger, action, response và provenance cố định."""
    rule_id: str
    severity: SafetySeverity
    trigger: Callable[[str], bool]
    action: str
    response: str
    source_ids: tuple[str, ...]
    source_urls: tuple[str, ...]


@dataclass(frozen=True)
class SafetyDecision:
    """Kết quả public của rule đầu tiên khớp câu hỏi đã chuẩn hóa."""
    rule_id: str
    severity: SafetySeverity
    action: str
    response: str
    source_ids: tuple[str, ...]
    source_urls: tuple[str, ...]


def evaluate_safety(query: str) -> SafetyDecision | None:
    """Trả một override hẹp; semantics thông thường vẫn thuộc Agent."""

    text = normalize_text(query, preserve_boundaries=True)
    for rule in SAFETY_RULES:
        if rule.trigger(text):
            return SafetyDecision(
                rule_id=rule.rule_id,
                severity=rule.severity,
                action=rule.action,
                response=rule.response,
                source_ids=rule.source_ids,
                source_urls=rule.source_urls,
            )
    return None


def safety_rule_inventory() -> tuple[SafetyRule, ...]:
    return SAFETY_RULES


def _anaphylaxis(text: str) -> bool:
    return has_local_concept_groups(
        text,
        (
            BREATHING_DIFFICULTY_CONCEPTS,
            (
                "sung moi",
                "sung mieng",
                "sung luoi",
                "sung hong",
                "noi me day",
                "phat ban ngua lan nhanh",
                "soc phan ve",
            ),
        ),
    )


def _breathing_after_medication(text: str) -> bool:
    return has_medication_related_active_symptom(text, BREATHING_DIFFICULTY_CONCEPTS)


def _significant_bleeding_after_acne_manipulation(text: str) -> bool:
    return has_local_concept_groups(
        text,
        (
            ("nan mun", "bop mun", "choc mun"),
            (
                "chay mau nhieu",
                "mau chay nhieu",
                "van chay nhieu",
                "khong cam duoc",
                "khong cam mau duoc",
                "chay mai",
                "chay khong dung",
                "chay lien tuc",
            ),
        ),
    )


def _chest_breathing_emergency(text: str) -> bool:
    return has_local_concept_groups(
        text,
        (
            ("dau nguc", "tuc nguc", "nguc bi ep chat"),
            BREATHING_DIFFICULTY_CONCEPTS,
        ),
    )


def _self_harm(text: str) -> bool:
    concepts = (
        "tu sat",
        "tu tu",
        "lam hai ban than",
        "suicide",
        "kill myself",
        "harm myself",
    )
    if not has_active_symptom(text, concepts):
        return False
    if not has_first_person_reference(text, concepts):
        return False
    direct_intent = any(
        contains_bounded_sequence(text, (intent, concept), max_gap=1)
        for intent in ("muon", "sap", "se", "want", "going")
        for concept in concepts
    )
    active_thought = any(
        contains_bounded_sequence(text, ("dang", "nghi", concept), max_gap=4)
        for concept in concepts
    ) or any(
        contains_bounded_sequence(text, ("thinking", "about", concept), max_gap=2)
        for concept in concepts
    )
    return (direct_intent or active_thought) and not _has_reference_purpose(text, concepts)


def _acne_fulminans(text: str) -> bool:
    return has_local_concept_groups(
        text,
        (
            ("mun", "acne"),
            ("loet", "trot loet", "mun cuc", "mun nang", "bung phat rat nhanh"),
            ("sot", "dau khop"),
        ),
    )


def _isotretinoin_pregnancy(text: str) -> bool:
    negative_pregnancy = (
        "khong mang thai",
        "khong co thai",
        "khong co bau",
        "khong dang mang thai",
        "not pregnant",
    )
    concept_groups = (
        ("isotretinoin",),
        (
            "mang thai",
            "co thai",
            "co bau",
            "dang bau",
            "du dinh mang thai",
            "planning pregnancy",
            "pregnant",
        ),
    )
    if has_first_person_reference(text, negative_pregnancy):
        return any(
            has_local_concept_groups(
                text,
                concept_groups,
                active_group_indexes=(1,),
                required_owner=owner,
            )
            for owner in (True, False)
        )
    return has_local_concept_groups(
        text,
        concept_groups,
        active_group_indexes=(1,),
    )


def _isotretinoin_neurologic(text: str) -> bool:
    return has_local_concept_groups(
        text,
        (
            ("isotretinoin",),
            ("dau dau du doi", "dau dau rat nang", "severe headache"),
            ("nhin mo", "mo mat", "blurred vision", "buon non", "non"),
        ),
        active_group_indexes=(1, 2),
    )


def _prescription_execution(text: str) -> bool:
    return is_prescription_execution_request(text)


def _has_reference_purpose(text: str, concepts: tuple[str, ...]) -> bool:
    reference_terms = r"(?:information|policy|prevention|research)"
    return any(
        re.search(
            rf"(?:{reference_terms}(?:\s+\w+){{0,2}}\s+{re.escape(concept)}|"
            rf"{re.escape(concept)}(?:\s+\w+){{0,2}}\s+{reference_terms})",
            text,
        )
        is not None
        for concept in concepts
    )


SAFETY_RULES: tuple[SafetyRule, ...] = (
    SafetyRule(
        "anaphylaxis_like_emergency",
        "emergency",
        _anaphylaxis,
        "seek_emergency_help_now",
        "**Hành động ngay**\nKhó thở kèm sưng môi, miệng, lưỡi hoặc họng, hoặc nổi mề đay lan nhanh, có thể là phản ứng phản vệ. Hãy gọi cấp cứu địa phương ngay và không tự lái xe. Nếu có bút tiêm adrenaline đã được kê cho bạn, hãy dùng theo hướng dẫn đã được cấp.",
        ("NHS_ANAPHYLAXIS",),
        ("https://www.nhs.uk/conditions/anaphylaxis/",),
    ),
    SafetyRule(
        "breathing_difficulty_after_medication",
        "emergency",
        _breathing_after_medication,
        "seek_emergency_help_now",
        "**Hành động ngay**\nKhó thở đang xảy ra sau khi dùng thuốc có thể là phản ứng nghiêm trọng. Hãy gọi cấp cứu địa phương ngay và không tự lái xe. Mang theo thuốc hoặc bao bì thuốc để nhân viên y tế kiểm tra; không chờ tự xử lý tại nhà khi bạn vẫn khó thở.",
        ("NHS_ANAPHYLAXIS_MEDICINE_TRIGGER", "NHS_SHORTNESS_OF_BREATH"),
        (
            "https://www.nhs.uk/conditions/anaphylaxis/",
            "https://www.nhs.uk/symptoms/shortness-of-breath/",
        ),
    ),
    SafetyRule(
        "significant_bleeding_after_acne_manipulation",
        "emergency",
        _significant_bleeding_after_acne_manipulation,
        "control_bleeding_and_seek_emergency_help",
        "**Hành động ngay**\nNgừng nặn hoặc chạm vào vùng da. Dùng gạc hay khăn sạch ép chắc và liên tục lên chỗ chảy máu; không nhấc miếng đang ép chỉ để kiểm tra. Nếu máu thấm qua, đặt thêm một miếng sạch lên trên và tiếp tục ép. Vì bạn mô tả máu chảy nhiều hoặc không cầm, hãy gọi cấp cứu địa phương ngay.",
        ("NHS_FIRST_AID_HEAVY_BLEEDING",),
        ("https://www.nhs.uk/tests-and-treatments/first-aid/",),
    ),
    SafetyRule(
        "chest_pain_with_breathlessness",
        "emergency",
        _chest_breathing_emergency,
        "seek_emergency_help_now",
        "**Hành động ngay**\nĐau hoặc tức ngực kèm khó thở cần được đánh giá cấp cứu. Hãy gọi cấp cứu địa phương ngay và không tự lái xe.",
        ("NHS_CHEST_PAIN",),
        ("https://www.nhs.uk/symptoms/chest-pain/",),
    ),
    SafetyRule(
        "self_harm_immediate_risk",
        "emergency",
        _self_harm,
        "contact_emergency_or_crisis_support",
        "**Hành động ngay**\nNếu bạn có nguy cơ làm hại bản thân ngay lúc này, hãy gọi cấp cứu địa phương hoặc đường dây hỗ trợ khủng hoảng, và ở cùng một người bạn tin cậy. Đừng ở một mình; hãy nói trực tiếp với nhân viên y tế hoặc người thân ngay bây giờ.",
        ("WHO_SUICIDE_QA",),
        ("https://www.who.int/news-room/questions-and-answers/item/suicide",),
    ),
    SafetyRule(
        "acne_fulminans_systemic_features",
        "urgent",
        _acne_fulminans,
        "same_day_urgent_dermatology_referral",
        "**Cần khám khẩn trong ngày**\nMụn bùng phát nặng với tổn thương cục/nang hoặc trợt loét kèm sốt hay đau khớp phù hợp với dấu hiệu cần loại trừ acne fulminans. Hãy đến cơ sở y tế trong ngày để được đánh giá; không tự bắt đầu thuốc kê đơn.",
        ("NICE_NG198_RECOMMENDATION_1_4_1",),
        ("https://www.nice.org.uk/guidance/ng198/chapter/Recommendations",),
    ),
    SafetyRule(
        "isotretinoin_pregnancy",
        "urgent",
        _isotretinoin_pregnancy,
        "do_not_use_and_contact_prescriber",
        "**Không dùng isotretinoin trong thai kỳ**\nIsotretinoin có thể gây tổn hại nghiêm trọng cho thai nhi. Nếu bạn đang mang thai, nghi ngờ có thai hoặc dự định mang thai, không dùng thuốc và hãy liên hệ bác sĩ hoặc đơn vị kê đơn ngay để được hướng dẫn theo chương trình phòng ngừa thai.",
        ("NICE_NG198_RECOMMENDATION_1_5_22", "MHRA_ISOTRETINOIN_PREGNANCY_PREVENTION"),
        (
            "https://www.nice.org.uk/guidance/ng198/chapter/Recommendations",
            "https://www.gov.uk/drug-safety-update/oral-retinoids-pregnancy-prevention-reminder-of-measures-to-minimise-teratogenic-risk",
        ),
    ),
    SafetyRule(
        "isotretinoin_severe_headache_visual_symptoms",
        "emergency",
        _isotretinoin_neurologic,
        "stop_and_seek_urgent_medical_assessment",
        "**Cần được đánh giá y tế ngay**\nĐau đầu dữ dội kèm nhìn mờ hoặc buồn nôn/nôn khi đang dùng isotretinoin là dấu hiệu được hướng dẫn phải ngừng thuốc và liên hệ nhân viên y tế ngay. Nếu triệu chứng đang nặng hoặc tiến triển, hãy đến cấp cứu.",
        ("DAILYMED_ISOTRETINOIN_MEDICATION_GUIDE",),
        ("https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=72867c88-070f-4608-bfef-cc5225ebce6d",),
    ),
    SafetyRule(
        "prescription_execution_request",
        "policy",
        _prescription_execution,
        "refuse_prescription_execution",
        "**Giới hạn hỗ trợ**\nTôi không thể kê đơn, chọn thuốc hoặc chọn liều cá nhân. Tôi có thể giải thích thông tin trong nguồn hiện có để bạn chuẩn bị câu hỏi cho bác sĩ hoặc dược sĩ.",
        ("ENGINEERING_POLICY_NO_PRESCRIPTION",),
        (),
    ),
)

__all__ = [
    "SAFETY_POLICY_VERSION",
    "SAFETY_RULES",
    "SafetyDecision",
    "SafetyRule",
    "evaluate_safety",
    "safety_rule_inventory",
]
