from __future__ import annotations

import pytest

from src.agent.emergency_contract import first_sentence_has_immediate_emergency_action
from src.agent.nodes.quality import answer_quality_node
from src.quality.severity_guard import apply_severity_aware_answer_guard
from src.quality.severity_guard import SAFETY_POLICY_PROVENANCE


@pytest.mark.parametrize(
    "query",
    [
        "Sau khi bôi thuốc trị mụn, mắt tôi sưng và tôi bắt đầu khó thở.",
        "Sau khi dùng thuốc, môi tôi sưng, nghẹn họng và khó thở.",
    ],
)
def test_anaphylaxis_like_emergency_is_deterministic_and_immediate(query: str) -> None:
    guarded = apply_severity_aware_answer_guard(query, "Model draft")
    assert guarded.classification.severity == "emergency"
    assert guarded.modified is True
    assert guarded.cache_eligible is False
    assert first_sentence_has_immediate_emergency_action(guarded.answer)


def test_caution_classification_never_appends_medical_facts() -> None:
    draft = "Controlled answer from evidence."
    guarded = apply_severity_aware_answer_guard("Da đỏ rát nhẹ khi dùng một sản phẩm", draft)
    assert guarded.classification.severity == "caution"
    assert guarded.answer == draft
    assert guarded.modified is False


@pytest.mark.asyncio
async def test_full_safety_replacement_has_truthful_system_origin() -> None:
    result = await answer_quality_node(
        {
            "user_question": "Sau khi bôi thuốc tôi khó thở và sưng môi.",
            "final_answer": "Model draft",
            "sources": ["source-a"],
            "vector_contexts": [{"source_id": "source-a", "text": "evidence"}],
        }
    )
    assert result["actual_provider"] == "system"
    assert result["actual_model"] is None
    assert result["sources"] == []
    assert result["fallback_cache_eligible"] is False


def test_acne_fulminans_like_contract_keeps_same_day_urgency() -> None:
    guarded = apply_severity_aware_answer_guard(
        "Đột ngột có nhiều cục nang viêm trợt loét, vảy xuất huyết, sốt và đau khớp.",
        "Model draft",
    )
    assert guarded.classification.severity == "urgent"
    assert "trong ngày" in guarded.answer
    assert "24 giờ" in guarded.answer


def test_every_retained_full_safety_override_has_named_authoritative_provenance() -> None:
    assert SAFETY_POLICY_PROVENANCE == {
        "emergency": ["NHS_ANAPHYLAXIS", "DAILYMED_ISOTRETINOIN_MEDICATION_GUIDE"],
        "self_harm": ["WHO_SUICIDE_QA"],
        "acne_fulminans": ["NICE_NG198_RECOMMENDATION_1_4_1"],
        "isotretinoin_pregnancy": [
            "NICE_NG198_RECOMMENDATION_1_5_22",
            "MHRA_ISOTRETINOIN_PPP",
        ],
    }
