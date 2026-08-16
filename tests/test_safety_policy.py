from __future__ import annotations

import pytest

from src.agent.nodes.workflow import decide_node, guard_node
from src.agent.safety_policy import evaluate_safety, safety_rule_inventory


@pytest.mark.parametrize(
    ("query", "rule_id"),
    [
        ("Sau thuốc tôi khó thở, sưng môi và nổi mề đay", "anaphylaxis_like_emergency"),
        ("Tôi đau ngực và khó thở", "chest_pain_with_breathlessness"),
        ("Tôi muốn tự làm hại bản thân ngay", "self_harm_immediate_risk"),
        ("Tôi đang nghĩ đến việc tự sát", "self_harm_immediate_risk"),
        ("I am thinking about suicide", "self_harm_immediate_risk"),
        ("Mụn nang bùng phát rất nhanh, loét, sốt và đau khớp", "acne_fulminans_systemic_features"),
        ("Tôi đang mang thai và dùng isotretinoin", "isotretinoin_pregnancy"),
        ("Đang dùng isotretinoin, đau đầu dữ dội và nhìn mờ", "isotretinoin_severe_headache_visual_symptoms"),
        ("Tôi đang uống isotretinoin và hôm nay bị đau đầu dữ dội kèm nhìn mờ.", "isotretinoin_severe_headache_visual_symptoms"),
        ("Kê đơn thuốc trị mụn cho tôi", "prescription_execution_request"),
    ],
)
def test_each_retained_rule_has_positive_trigger_and_source(query: str, rule_id: str) -> None:
    decision = evaluate_safety(query)
    assert decision is not None
    assert decision.rule_id == rule_id
    assert decision.source_ids
    if not rule_id.startswith("prescription_"):
        assert decision.source_urls


@pytest.mark.parametrize(
    "query",
    [
        "Tôi hơi khó thở sau khi chạy bộ nhưng không sưng hay phát ban",
        "Tôi đau ngực nhẹ đã hết và không khó thở",
        "Mụn nang nhưng không sốt, không đau khớp",
        "Tôi dùng isotretinoin và hơi đau đầu",
        "Isotretinoin có dùng khi mang thai không? Tôi không đang mang thai",
        "Isotretinoin có liên quan đến suicide không?",
        "Các nghiên cứu nói gì về self harm?",
        "Tôi không muốn tự sát.",
        "Tôi chưa bao giờ muốn tự tử.",
        "Tôi từng đọc về suicide.",
    ],
)
def test_nearby_negative_boundaries_do_not_trigger(query: str) -> None:
    assert evaluate_safety(query) is None


@pytest.mark.parametrize(
    "query",
    [
        "Tôi không khó thở nhưng bị sưng môi.",
        "Tôi từng khó thở nhưng hiện chỉ còn sưng môi.",
        "Tôi từng bị khó thở, hiện chỉ còn sưng môi.",
        "Tôi khó thở trước đây nhưng đã hết, giờ chỉ sưng môi.",
        "Tôi bị sưng môi nhưng thở bình thường.",
    ],
)
def test_anaphylaxis_requires_current_unnegated_breathing_difficulty(query: str) -> None:
    assert evaluate_safety(query) is None


@pytest.mark.parametrize(
    "query",
    [
        "Tôi dùng isotretinoin nhưng không đau đầu dữ dội, chỉ hơi mỏi mắt và nhìn mờ.",
        "Tôi từng đau đầu dữ dội, hiện đã hết nhưng còn hơi nhìn mờ khi dùng isotretinoin.",
        "Tôi từng bị đau đầu dữ dội khi dùng isotretinoin, hiện chỉ còn hơi nhìn mờ.",
        "Tôi dùng isotretinoin, đau đầu dữ dội đã hết nhưng mắt còn hơi mờ.",
        "Tôi dùng isotretinoin và chỉ hơi mỏi mắt.",
    ],
)
def test_isotretinoin_neurologic_rule_requires_active_unnegated_severe_headache(
    query: str,
) -> None:
    assert evaluate_safety(query) is None


def test_resolved_other_symptom_does_not_suppress_active_isotretinoin_neurologic_rule() -> None:
    decision = evaluate_safety(
        "Tôi dùng isotretinoin, hiện đã hết buồn nôn nhưng vẫn đau đầu dữ dội và nhìn mờ."
    )

    assert decision is not None
    assert decision.rule_id == "isotretinoin_severe_headache_visual_symptoms"


@pytest.mark.parametrize(
    "query",
    [
        "Tôi đang khó thở sau khi uống thuốc trị mụn.",
        "Uống thuốc xong tôi thấy khó thở.",
        "Sau khi dùng thuốc tôi đang bị hụt hơi.",
        "Tôi vừa uống thuốc trị mụn xong thì hiện tại đang khó thở.",
    ],
)
def test_current_breathing_difficulty_after_medication_triggers_emergency(query: str) -> None:
    decision = evaluate_safety(query)

    assert decision is not None
    assert decision.rule_id == "breathing_difficulty_after_medication"
    assert decision.severity == "emergency"
    assert decision.source_ids == (
        "NHS_ANAPHYLAXIS_MEDICINE_TRIGGER",
        "NHS_SHORTNESS_OF_BREATH",
    )


@pytest.mark.parametrize(
    "query",
    [
        "Thuốc trị mụn có gây khó thở không?",
        "Tôi không bị khó thở sau khi uống thuốc.",
        "Tôi từng đọc rằng thuốc này có thể gây khó thở.",
        "Tôi khó thở sau khi chạy bộ.",
    ],
)
def test_breathing_after_medication_rule_rejects_non_current_or_unrelated_queries(
    query: str,
) -> None:
    assert evaluate_safety(query) is None


@pytest.mark.parametrize(
    "query",
    [
        "Tôi uống thuốc trị mụn mới được khoảng 20 phút thì hiện thấy rất khó thở.",
        "Hiện tôi rất khó thở, tình trạng bắt đầu ít phút sau lúc bôi thuốc trị mụn.",
        "TOI vua UONG, mot vien THUOC tri mun; bay gio dang HUT HOI!",
    ],
)
def test_medication_breathing_emergency_accepts_modifier_order_and_surface_variants(
    query: str,
) -> None:
    decision = evaluate_safety(query)
    assert decision is not None
    assert decision.rule_id == "breathing_difficulty_after_medication"


@pytest.mark.parametrize(
    "query",
    [
        "Hôm qua tôi khó thở sau khi uống thuốc, nhưng đã hết và hiện tại tôi bình thường.",
        "Hôm qua tôi từng hơi khó thở sau khi uống thuốc nhưng giờ hết hoàn toàn và đang thở bình thường.",
        "Bạn tôi đang khó thở sau khi uống thuốc trị mụn.",
        "Nếu uống thuốc rồi khó thở thì phải làm gì?",
        "Tôi uống thuốc nhưng hiện không khó thở.",
    ],
)
def test_medication_breathing_emergency_rejects_resolved_third_person_or_hypothetical(
    query: str,
) -> None:
    assert evaluate_safety(query) is None


@pytest.mark.parametrize(
    "query",
    [
        "Kê cho tôi một đơn thuốc trị mụn.",
        "Kê cho tôi một đơn thuốc trị mụn và chọn luôn liều adapalene phù hợp.",
        "Bạn chọn giúp tôi liều phù hợp.",
        "Cho tôi liều dùng isotretinoin phù hợp.",
    ],
)
def test_prescription_policy_accepts_natural_request_order(query: str) -> None:
    decision = evaluate_safety(query)
    assert decision is not None
    assert decision.rule_id == "prescription_execution_request"


@pytest.mark.parametrize(
    "query",
    [
        "Đừng kê đơn cho tôi.",
        "Bác sĩ đã kê đơn cho tôi.",
        "Khi nào bác sĩ kê đơn?",
        "Tài liệu nói gì về việc kê đơn?",
    ],
)
def test_prescription_policy_rejects_non_execution_mentions(query: str) -> None:
    assert evaluate_safety(query) is None


@pytest.mark.parametrize(
    "query",
    [
        "Sau khi nặn mụn tôi bị chảy máu nhiều.",
        "Tôi vừa nặn mụn và máu vẫn chảy nhiều.",
        "Chỗ nặn mụn đang chảy máu không cầm được.",
    ],
)
def test_significant_bleeding_after_acne_manipulation_triggers_emergency(query: str) -> None:
    decision = evaluate_safety(query)

    assert decision is not None
    assert decision.rule_id == "significant_bleeding_after_acne_manipulation"
    assert decision.severity == "emergency"
    assert decision.source_ids == ("NHS_FIRST_AID_HEAVY_BLEEDING",)


@pytest.mark.parametrize(
    "query",
    [
        "Nặn mụn có thể chảy máu không?",
        "Tôi nặn mụn và có một chấm máu nhỏ nhưng đã cầm.",
        "Tôi không bị chảy máu sau khi nặn mụn.",
        "Mụn đỏ có phải là máu không?",
    ],
)
def test_bleeding_rule_rejects_minor_resolved_hypothetical_or_negated_queries(
    query: str,
) -> None:
    assert evaluate_safety(query) is None


def test_rule_inventory_has_unique_ids_and_specific_source_mapping() -> None:
    rules = safety_rule_inventory()
    assert len(rules) == 9
    assert len({rule.rule_id for rule in rules}) == len(rules)
    anaphylaxis = next(rule for rule in rules if rule.rule_id == "anaphylaxis_like_emergency")
    neurologic = next(
        rule for rule in rules if rule.rule_id == "isotretinoin_severe_headache_visual_symptoms"
    )
    assert anaphylaxis.source_ids == ("NHS_ANAPHYLAXIS",)
    assert neurologic.source_ids == ("DAILYMED_ISOTRETINOIN_MEDICATION_GUIDE",)
    prescription = next(
        rule for rule in rules if rule.rule_id == "prescription_execution_request"
    )
    assert prescription.severity == "policy"
    assert prescription.source_ids == ("ENGINEERING_POLICY_NO_PRESCRIPTION",)
    assert prescription.source_urls == ()


@pytest.mark.parametrize(
    ("query", "rule_id"),
    [
        ("Sưng lưỡi kèm lúc này rất khó thở.", "anaphylaxis_like_emergency"),
        ("Tôi vừa bôi thuốc trị mụn và hiện đang hụt hơi.", "breathing_difficulty_after_medication"),
        ("Máu vẫn chảy liên tục dù tôi vừa bóp mụn.", "significant_bleeding_after_acne_manipulation"),
        ("Hiện tôi khó thở và ngực bị ép chặt.", "chest_pain_with_breathlessness"),
        ("Tôi thực sự muốn làm hại bản thân ngay.", "self_harm_immediate_risk"),
        ("Tôi sốt, đau khớp và có mụn nang trợt loét.", "acne_fulminans_systemic_features"),
        ("ISOTRETINOIN: tôi có thai.", "isotretinoin_pregnancy"),
        ("Tôi đau đầu rất nặng và buồn nôn khi dùng isotretinoin.", "isotretinoin_severe_headache_visual_symptoms"),
        ("Bạn hãy chọn giúp tôi liều thuốc phù hợp.", "prescription_execution_request"),
    ],
)
def test_all_nine_rules_accept_meaningful_surface_variants(query: str, rule_id: str) -> None:
    decision = evaluate_safety(query)
    assert decision is not None
    assert decision.rule_id == rule_id


@pytest.mark.parametrize(
    "query",
    [
        "Tôi từng khó thở và sưng lưỡi nhưng hiện đã bình thường.",
        "Nếu dùng thuốc rồi khó thở thì cần làm gì?",
        "Tôi nặn mụn nhưng không chảy máu.",
        "Hôm qua tôi đau ngực và khó thở nhưng giờ đã hết.",
        "Tôi không muốn làm hại bản thân.",
        "Mụn nang nhưng không loét, không sốt và không đau khớp.",
        "Tôi dùng isotretinoin nhưng không mang thai.",
        "Đau đầu dữ dội khi dùng isotretinoin đã hết; hiện chỉ hơi mỏi mắt.",
        "Bác sĩ đã chọn liều thuốc cho tôi.",
    ],
)
def test_all_nine_rules_keep_negative_resolved_or_reference_boundaries(query: str) -> None:
    assert evaluate_safety(query) is None


@pytest.mark.asyncio
async def test_safety_override_precedes_agent_and_is_not_cacheable() -> None:
    guarded = await guard_node(
        {"normalized_question": "Sau thuốc tôi khó thở và sưng lưỡi", "bypass_cache": False}
    )
    assert guarded["safety_override"] is True
    assert guarded["fallback_cache_eligible"] is False
    assert guarded["sources"] == []
    assert (await decide_node(guarded))["next_action"] == "finalize"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "Tôi đang khó thở sau khi uống thuốc trị mụn.",
        "Sau khi nặn mụn tôi bị chảy máu nhiều và máu vẫn không cầm.",
    ],
)
async def test_new_safety_gaps_bypass_agent_retrieval_and_cache(query: str) -> None:
    guarded = await guard_node({"normalized_question": query, "bypass_cache": False})

    assert guarded["safety_override"] is True
    assert guarded["fallback_cache_eligible"] is False
    assert guarded["sources"] == []
    assert (await decide_node(guarded))["next_action"] == "finalize"
