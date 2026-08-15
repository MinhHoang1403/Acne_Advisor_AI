"""Narrow, source-mapped deterministic safety overrides."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Callable, Literal

SafetySeverity = Literal["policy", "urgent", "emergency"]


@dataclass(frozen=True)
class SafetyRule:
    rule_id: str
    severity: SafetySeverity
    trigger: Callable[[str], bool]
    action: str
    response: str
    source_ids: tuple[str, ...]
    source_urls: tuple[str, ...]


@dataclass(frozen=True)
class SafetyDecision:
    rule_id: str
    severity: SafetySeverity
    action: str
    response: str
    source_ids: tuple[str, ...]
    source_urls: tuple[str, ...]


def evaluate_safety(query: str) -> SafetyDecision | None:
    """Return one narrow override; ordinary semantics remain with the agent."""

    text = _fold(query)
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
    return _has(text, "kho tho", "kho khe", "nghet tho", "tho rit") and _has(
        text,
        "sung moi",
        "sung mieng",
        "sung luoi",
        "sung hong",
        "noi me day",
        "phat ban ngua lan nhanh",
        "soc phan ve",
    )


def _chest_breathing_emergency(text: str) -> bool:
    return _has_unnegated(text, "dau nguc", "tuc nguc", "nguc bi ep chat") and _has_unnegated(
        text, "kho tho", "hut hoi", "tho khong ra hoi"
    )


def _self_harm(text: str) -> bool:
    return _has(
        text,
        "tu sat",
        "tu tu",
        "tu lam hai",
        "lam hai ban than",
        "ket lieu doi minh",
        "self harm",
        "suicide",
    )


def _acne_fulminans(text: str) -> bool:
    acne = _has(text, "mun", "acne")
    systemic = _has_unnegated(text, "sot", "dau khop")
    severe = _has_unnegated(
        text, "loet", "trot loet", "mun cuc", "mun nang", "bung phat rat nhanh"
    )
    return acne and systemic and severe


def _isotretinoin_pregnancy(text: str) -> bool:
    if _has(
        text,
        "khong mang thai",
        "khong co thai",
        "khong co bau",
        "khong dang mang thai",
        "not pregnant",
    ):
        return False
    return "isotretinoin" in text and _has_unnegated(
        text,
        "mang thai",
        "co thai",
        "co bau",
        "dang bau",
        "du dinh mang thai",
        "planning pregnancy",
        "pregnant",
    )


def _isotretinoin_neurologic(text: str) -> bool:
    severe_headache = _has(text, "dau dau du doi", "dau dau rat nang", "severe headache")
    visual_or_gi = _has(text, "nhin mo", "mo mat", "blurred vision", "buon non", "non")
    return "isotretinoin" in text and severe_headache and visual_or_gi


def _prescription_execution(text: str) -> bool:
    return bool(
        re.search(
            r"(?:^|\s)(ke don|ke thuoc|cho toi toa|cho toi don thuoc|chon lieu|cho toi lieu)(?:\s|$)",
            text,
        )
    )


def _has(text: str, *phrases: str) -> bool:
    return any(phrase in text for phrase in phrases)


def _has_unnegated(text: str, *phrases: str) -> bool:
    for phrase in phrases:
        for match in re.finditer(rf"(?:^|\s){re.escape(phrase)}(?:\s|$)", text):
            prefix_tokens = text[: match.start()].split()[-2:]
            if "khong" not in prefix_tokens and "not" not in prefix_tokens:
                return True
    return False


def _fold(value: str) -> str:
    normalized = unicodedata.normalize(
        "NFD", str(value or "").casefold().replace("đ", "d")
    )
    accentless = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", accentless).split())


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

__all__ = ["SAFETY_RULES", "SafetyDecision", "SafetyRule", "evaluate_safety", "safety_rule_inventory"]
