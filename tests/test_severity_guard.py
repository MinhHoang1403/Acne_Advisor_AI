from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent import graph as graph_module
from src.agent.nodes import cache as cache_node
from src.agent.nodes import quality as quality_node
from src.agent.nodes.cache import cache_lookup_node, cache_store_node
from src.agent.nodes.severity import severity_classification_node
from src.agent.emergency_contract import first_sentence_has_immediate_emergency_action
from src.resilience.exceptions import AgentTimeoutError
from src.quality.severity_guard import apply_severity_aware_answer_guard, classify_medical_severity


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Tôi bôi thuốc trị mụn xong bị sưng môi, khó thở và nổi mề đay toàn thân", "emergency"),
        ("Uống thuốc trị mụn xong bị phồng rộp da, loét miệng và sốt cao", "emergency"),
        ("Sau khi dùng thuốc trị mụn, tôi bị phát ban toàn thân", "emergency"),
        ("Mặt tôi sưng nhanh, đau dữ dội, da tím đen và chảy mủ", "emergency"),
        ("Tôi bị sốt cao kèm phát ban lan nhanh", "emergency"),
        ("Mụn ở gần mắt bị sưng đỏ đau và chảy mủ", "urgent"),
        ("Tôi đang uống isotretinoin mà bị đau đầu dữ dội và nhìn mờ", "urgent"),
        ("Tôi đang mang thai, có dùng isotretinoin trị mụn được không?", "urgent"),
        ("Mụn làm tôi mất ngủ và tôi nghĩ đến việc tự làm hại bản thân", "urgent"),
        ("Nhiều cục và nang viêm lớn trợt loét, đóng vảy xuất huyết, kèm sốt và đau khớp", "urgent"),
        ("Sau nặn mụn có vẻ áp xe và đau nhiều", "urgent"),
        ("Em bé bị nghi nhiễm trùng da ở má", "urgent"),
        ("Da tôi bị đỏ rát nhẹ khi dùng benzoyl peroxide", "caution"),
        ("Tôi muốn dùng BHA với retinol chung được không?", "caution"),
        ("Da tôi bị châm chích khi dùng treatment", "caution"),
        ("Đang cho con bú thì dùng skincare trị mụn thế nào?", "caution"),
        ("Da dầu mụn ẩn nên rửa mặt và dưỡng ẩm thế nào?", "routine"),
        ("Mụn đầu đen ở mũi xử lý sao?", "routine"),
        ("Thâm sau mụn nên chăm sóc thế nào?", "routine"),
    ],
)
def test_medical_severity_classifier_required_cases(query: str, expected: str):
    assert classify_medical_severity(query).severity == expected


def test_emergency_guard_replaces_routine_skincare_with_emergency_action():
    result = apply_severity_aware_answer_guard(
        query="Tôi bôi thuốc trị mụn xong bị sưng môi, khó thở và nổi mề đay toàn thân",
        answer="Bạn nên rửa mặt dịu nhẹ, dưỡng ẩm và dùng chống nắng.",
    )

    assert result.classification.severity == "emergency"
    assert result.modified is True
    assert result.cache_eligible is False
    assert "cấp cứu" in result.answer
    assert "cơ sở y tế" in result.answer
    assert "rửa mặt dịu nhẹ" not in result.answer


def test_sjs_like_emergency_uses_explicit_urgent_care_wording():
    result = apply_severity_aware_answer_guard(
        query="Uống thuốc trị mụn xong bị phồng rộp da, loét miệng và sốt cao",
        answer="Bạn nên rửa mặt dịu nhẹ, dưỡng ẩm và dùng chống nắng.",
    )

    assert result.classification.severity == "emergency"
    assert result.modified is True
    assert "khẩn cấp" in result.answer
    assert "cấp cứu" in result.answer


@pytest.mark.parametrize(
    "query",
    [
        "Môi và lưỡi sưng sau khi dùng thuốc mới trị mụn, nên làm gì ngay?",
        "Sau khi dùng thuốc mụn, da bong tróc diện rộng và mắt đau rát.",
        "Có sốt, loét miệng và ban phồng rộp sau một thuốc điều trị mụn.",
        "Phồng rộp niêm mạc miệng kèm sốt sau thuốc trị mụn có phải cần cấp cứu không?",
        "Một người dùng isotretinoin có dấu hiệu sưng lưỡi và thở khó, nên làm gì?",
    ],
)
def test_v1_emergency_regressions_route_to_immediate_emergency_action(query: str):
    result = apply_severity_aware_answer_guard(query=query, answer="Bạn nên theo dõi thêm tại nhà.")

    assert result.classification.severity == "emergency"
    assert result.modified is True
    assert first_sentence_has_immediate_emergency_action(result.answer) is True


def test_all_retained_emergency_regressions_route_to_immediate_emergency_action():
    fixture_path = Path("tests/fixtures/safety_emergency_regression.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    emergency_cases = fixture["cases"]

    assert fixture["schema_version"] == "safety_emergency_regression_v1"
    assert "not clinical gold" in fixture["description"]
    assert len(emergency_cases) == 20
    for case in emergency_cases:
        result = apply_severity_aware_answer_guard(
            query=case["question"],
            answer="Bạn nên theo dõi thêm tại nhà.",
        )
        assert result.classification.severity == "emergency", case["id"]
        assert first_sentence_has_immediate_emergency_action(result.answer), case["id"]


def test_anaphylaxis_like_emergency_keeps_immediate_action_and_urgent_wording():
    result = apply_severity_aware_answer_guard(
        query="Tôi bôi thuốc trị mụn xong bị sưng môi, khó thở và nổi mề đay toàn thân",
        answer="Bạn nên theo dõi thêm tại nhà.",
    )

    assert result.classification.severity == "emergency"
    assert "khẩn cấp" in result.answer
    assert "cấp cứu" in result.answer


def test_urgent_guard_adds_clinician_review_and_isotretinoin_pregnancy_warning():
    result = apply_severity_aware_answer_guard(
        query="Tôi đang mang thai, có dùng isotretinoin trị mụn được không?",
        answer="Bạn có thể chăm sóc da nhẹ nhàng.",
    )

    assert result.classification.severity == "urgent"
    assert result.modified is True
    assert result.cache_eligible is False
    assert "bác sĩ" in result.answer
    assert "24-48" in result.answer
    assert "Isotretinoin không được tự dùng" in result.answer


def test_self_harm_guard_prioritizes_crisis_actions_over_acne_advice():
    result = apply_severity_aware_answer_guard(
        query=(
            "Mụn của tôi không quá nặng nhưng khiến tôi né tránh mọi người, mất ngủ và gần đây "
            "tôi đã nghĩ đến việc tự làm hại bản thân. Tôi nên làm gì?"
        ),
        answer="Bạn nên rửa mặt dịu nhẹ và gặp bác sĩ da liễu.",
    )

    assert result.classification.severity == "urgent"
    assert result.modified is True
    assert result.cache_eligible is False
    assert result.modification_reason == "severity_self_harm_crisis_preface"
    assert result.answer.startswith("**Tóm tắt ngắn**")
    assert "cấp cứu" in result.answer
    assert "người đáng tin cậy" in result.answer
    assert result.answer.index("An toàn") < result.answer.index("điều trị mụn")


def test_acne_fulminans_guard_keeps_same_day_and_24_hour_urgency():
    result = apply_severity_aware_answer_guard(
        query=(
            "Một nam thiếu niên đột ngột xuất hiện nhiều cục và nang viêm lớn, trợt loét, "
            "đóng vảy xuất huyết, kèm sốt và đau khớp."
        ),
        answer="Có thể là mụn nặng.",
    )

    assert result.classification.severity == "urgent"
    assert result.modified is True
    assert result.cache_eligible is False
    assert result.modification_reason == "severity_acne_fulminans_urgent_preface"
    assert "nghi acne fulminans" in result.answer
    assert "trong ngày" in result.answer
    assert "24 giờ" in result.answer
    assert "không thể chẩn đoán chắc chắn" in result.answer


def test_caution_guard_adds_mild_irritation_safety_note():
    result = apply_severity_aware_answer_guard(
        query="Da tôi bị đỏ rát nhẹ khi dùng benzoyl peroxide",
        answer="Benzoyl peroxide có thể hỗ trợ giảm mụn viêm.",
    )

    assert result.classification.severity == "caution"
    assert result.modified is True
    assert "giảm tần suất" in result.answer
    assert "tạm ngưng" in result.answer


def test_routine_guard_does_not_force_urgent_or_emergency_warning():
    answer = "Có thể rửa mặt dịu nhẹ, dưỡng ẩm phù hợp và chống nắng đều."
    result = apply_severity_aware_answer_guard(
        query="Da dầu mụn ẩn nên rửa mặt và dưỡng ẩm thế nào?",
        answer=answer,
    )

    assert result.classification.severity == "routine"
    assert result.modified is False
    assert result.answer == answer
    assert "cấp cứu" not in result.answer
    assert "24-48" not in result.answer


@pytest.mark.asyncio
async def test_urgent_and_emergency_queries_skip_cache_lookup():
    result = await cache_lookup_node(
        {
            "user_question": "Mụn ở gần mắt bị sưng đỏ đau và chảy mủ",
            "standalone_question": None,
            "conversation_history": [],
            "is_in_domain": True,
            "bypass_cache": False,
            "medical_severity": "urgent",
        }
    )

    assert result["cache_hit"] is False
    assert result["cache_reason"] == "severity_urgent"


@pytest.mark.asyncio
async def test_severity_classification_node_prefers_standalone_question():
    result = await severity_classification_node(
        {
            "user_question": "Nó thì sao?",
            "standalone_question": "Tôi đang uống isotretinoin mà bị nhìn mờ",
        }
    )

    assert result["medical_severity"] == "urgent"
    assert result["severity_guard"]["matched_rules"] == ["urgent_isotretinoin_concerning_symptoms"]


def test_urgent_guard_does_not_duplicate_sufficient_referral_answer():
    answer = "Bạn nên khám bác sĩ da liễu trong 24-48 giờ và theo dõi dấu hiệu nặng lên."
    result = apply_severity_aware_answer_guard(
        query="Mụn gần mắt bị sưng đỏ đau",
        answer=answer,
    )

    assert result.classification.severity == "urgent"
    assert result.modified is False
    assert result.cache_eligible is False
    assert result.answer == answer


def test_caution_guard_does_not_duplicate_existing_safety_note():
    answer = "Nếu kích ứng tăng, hãy giảm tần suất và hỏi bác sĩ trước khi phối hợp hoạt chất."
    result = apply_severity_aware_answer_guard(
        query="Tôi muốn dùng BHA với retinol chung được không?",
        answer=answer,
    )

    assert result.classification.severity == "caution"
    assert result.modified is False
    assert result.cache_eligible is True
    assert result.answer == answer


def test_guard_handles_empty_answers_with_preface_or_append_only():
    urgent = apply_severity_aware_answer_guard(
        query="Mụn gần mắt bị sưng đỏ đau",
        answer="",
    )
    caution = apply_severity_aware_answer_guard(
        query="Da tôi bị đỏ rát nhẹ",
        answer="",
    )

    assert urgent.answer == urgent.answer.strip()
    assert urgent.answer.startswith("**Tóm tắt ngắn**")
    assert "**Thông tin thêm**" not in urgent.answer
    assert caution.answer.startswith("**Lưu ý an toàn**")


def test_caution_guard_does_not_append_duplicate_template():
    base = (
        "**Lưu ý an toàn**\n"
        "Nếu da đỏ rát, khô bong hoặc châm chích tăng, hãy giảm tần suất hoặc tạm ngưng hoạt chất dễ kích ứng. "
        "Nên hỏi bác sĩ/dược sĩ nếu đang mang thai, cho con bú, có tiền sử dị ứng, hoặc cần phối hợp nhiều hoạt chất trị mụn."
    )
    result = apply_severity_aware_answer_guard(
        query="Tôi muốn phối hợp retinol và BHA",
        answer=base,
    )

    assert result.modified is False
    assert result.answer.count("**Lưu ý an toàn**") == 1


def test_emergency_rules_have_priority_over_caution_active_ingredient():
    result = apply_severity_aware_answer_guard(
        query="Bôi benzoyl peroxide xong tôi khó thở, sưng môi và nổi mề đay",
        answer="Benzoyl peroxide có thể gây khô da.",
    )

    assert result.classification.severity == "emergency"
    assert result.modified is True
    assert result.cache_eligible is False
    assert result.answer.startswith("**Tóm tắt ngắn**")


def test_graph_routing_uses_the_explicit_agent_action():
    assert graph_module.route_agent_action({"next_action": "finalize"}) == "finalize"
    assert graph_module.route_agent_action({"next_action": "retrieve"}) == "retrieve"
    assert graph_module.route_agent_action({}) == "abstain"


def test_cache_model_key_resolution_variants(monkeypatch):
    monkeypatch.setenv("GOOGLE_MODEL", "gemini-1.5-flash")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")

    assert cache_node._resolve_cache_model_key({"llm_provider": "gemini"}) == (
        "gemini",
        "gemini-3.5-flash",
    )
    assert cache_node._resolve_cache_model_key({"llm_provider": "local", "llm_model": "qwen3"}) == (
        "ollama",
        "qwen3:latest",
    )
    assert cache_node._resolve_cache_model_key({"llm_provider": "custom"}) == ("custom", "unknown")


@pytest.mark.asyncio
async def test_run_clinical_agent_returns_severity_fields(monkeypatch):
    captured_state = {}

    class FakeGraph:
        async def ainvoke(self, state):
            captured_state.update(state)
            return {
                **state,
                "final_answer": "Câu trả lời đã qua guard.",
                "standalone_question": "Mụn đầu đen là gì?",
                "symptoms": ["comedone"],
                "safety_flags": [],
                "sources": ["source.pdf"],
                "graph_facts": [{"fact": "test"}],
                "retrieval_trace": {"trace": "ok"},
                "packed_context": {"chunks": []},
                "observability_exported": {"ok": True},
                "answer_quality_report": {"passed": True},
                "answer_guard_modified": False,
                "answer_guard_mode": "metadata_only",
                "medical_severity": "routine",
                "severity_guard": {"severity": "routine"},
                "severity_guard_modified": False,
                "severity_guard_cache_eligible": True,
                "errors": [],
                "is_in_domain": True,
                "guardrail": "in_domain",
                "ignored_out_of_domain_part": None,
                "domain_reason": "ok",
                "cache_checked": True,
                "cache_hit": False,
                "cache_reason": "miss",
                "cache_metadata": None,
                "llm_fallback": False,
                "fallback_reason": None,
                "actual_provider": "gemini",
                "actual_model": "gemini-2.5-flash",
                "llm_fallback_used": False,
                "fallback_provider": None,
                "fallback_model": None,
            }

    monkeypatch.setattr(graph_module, "clinical_graph", FakeGraph())
    result = await graph_module.run_clinical_agent(
        "Mụn đầu đen là gì?",
        user_id="u1",
        session_id="s1",
        llm_provider="gemini",
        llm_model="gemini-2.5-flash",
        bypass_cache=True,
    )

    assert captured_state["medical_severity"] is None
    assert captured_state["severity_guard_cache_eligible"] is None
    assert captured_state["bypass_cache"] is True
    assert result["answer"] == "Câu trả lời đã qua guard."
    assert result["medical_severity"] == "routine"
    assert result["severity_guard_cache_eligible"] is True
    assert result["cache_reason"] == "miss"


@pytest.mark.asyncio
async def test_run_clinical_agent_wraps_timeout(monkeypatch):
    class TimeoutGraph:
        async def ainvoke(self, state):
            raise TimeoutError("expired")

    monkeypatch.setattr(graph_module, "clinical_graph", TimeoutGraph())

    with pytest.raises(AgentTimeoutError):
        await graph_module.run_clinical_agent("Mụn đầu đen là gì?")


@pytest.mark.asyncio
async def test_severity_modified_answers_are_not_cached(monkeypatch):
    called = False

    async def fake_set_answer_cache(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("src.agent.nodes.cache.set_answer_cache", fake_set_answer_cache)
    result = await cache_store_node(
        {
            "cache_hit": False,
            "bypass_cache": False,
            "conversation_history": [],
            "cache_reason": "miss",
            "medical_severity": "caution",
            "severity_guard_modified": True,
            "severity_guard_cache_eligible": False,
        }
    )

    assert result == {}
    assert called is False


@pytest.mark.asyncio
async def test_cache_lookup_hit_and_invalid_metadata(monkeypatch):
    calls: list[dict] = []

    async def fake_get_exact_cache(*args, **kwargs):
        kwargs["normalized_question"] = args[0] if args else kwargs.get("normalized_question")
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "answer": "Có thể chăm sóc da dịu nhẹ và theo dõi đáp ứng.",
                "sources": ["source.pdf"],
                "metadata": {
                    "quality_passed": True,
                    "answer_version": "v6",
                    "pipeline_fingerprint": "fp123",
                },
                "model_name": "gemini-2.5-flash",
            }
        return {
            "answer": "cache stale",
            "sources": [],
            "metadata": {
                "quality_passed": False,
                "answer_version": "v6",
                "pipeline_fingerprint": "fp123",
            },
        }

    monkeypatch.setattr("src.agent.nodes.cache.get_exact_cache", fake_get_exact_cache)

    base_state = {
        "user_question": "Mụn đầu đen là gì?",
        "conversation_history": [],
        "is_in_domain": True,
        "bypass_cache": False,
        "medical_severity": "routine",
        "pipeline_manifest": {"phase": "phase2e"},
        "pipeline_fingerprint": "fp123",
        "llm_provider": "gemini",
        "llm_model": "gemini-2.5-flash",
    }
    hit = await cache_lookup_node(base_state)
    invalid = await cache_lookup_node(base_state)

    assert hit["cache_hit"] is True
    assert hit["actual_provider"] == "cache"
    assert hit["pipeline_fingerprint"] == "fp123"
    assert invalid["cache_hit"] is False
    assert invalid["cache_reason"] == "invalid_cache_metadata"
    assert calls[0]["provider"] == "gemini"


@pytest.mark.asyncio
async def test_cache_lookup_bypass_history_and_not_cacheable(monkeypatch):
    bypass = await cache_lookup_node(
        {
            "user_question": "Mụn đầu đen là gì?",
            "conversation_history": [],
            "is_in_domain": True,
            "bypass_cache": True,
        }
    )
    history = await cache_lookup_node(
        {
            "user_question": "Nó thì sao?",
            "standalone_question": "Benzoyl peroxide dùng thế nào?",
            "conversation_history": [{"role": "user", "content": "Benzoyl peroxide là gì?"}],
            "is_in_domain": True,
            "bypass_cache": False,
            "medical_severity": "routine",
        }
    )
    monkeypatch.setattr(
        "src.agent.nodes.cache.is_cacheable_question",
        lambda question, history, guard_status: (False, "not_cacheable_test"),
    )
    not_cacheable = await cache_lookup_node(
        {
            "user_question": "Hãy kê đơn thuốc trị mụn cho tôi",
            "conversation_history": [],
            "is_in_domain": True,
            "bypass_cache": False,
            "medical_severity": "routine",
        }
    )

    assert bypass["cache_checked"] is False
    assert history["cache_reason"] == "history_present"
    assert not_cacheable["cache_reason"] == "not_cacheable_test"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state_patch",
    [
        {"cache_hit": True},
        {"bypass_cache": True},
        {"conversation_history": [{"role": "user", "content": "hi"}]},
        {"cache_reason": "history_present"},
        {"medical_severity": "emergency"},
        {"is_in_domain": False},
        {"use_history_context": True},
        {"user_question": "Bạn kê đơn thuốc trị mụn cho tôi"},
        {"errors": ["timeout"]},
        {"llm_fallback": True},
        {"final_answer": ""},
        {"final_answer": "quá ngắn"},
        {"llm_fallback_used": True},
        {"fallback_provider": "rule_based"},
        {"sources": []},
    ],
)
async def test_cache_store_skip_policies(monkeypatch, state_patch):
    called = False

    async def fake_set_answer_cache(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("src.agent.nodes.cache.set_answer_cache", fake_set_answer_cache)
    monkeypatch.setenv("CACHE_MIN_ANSWER_CHARS", "20")
    monkeypatch.setenv("CACHE_REQUIRED_ENTITY_CHECK", "false")

    base_state = {
        "cache_hit": False,
        "bypass_cache": False,
        "conversation_history": [],
        "cache_reason": "miss",
        "medical_severity": "routine",
        "severity_guard_modified": False,
        "severity_guard_cache_eligible": True,
        "is_in_domain": True,
        "use_history_context": False,
        "errors": [],
        "llm_fallback": False,
        "llm_fallback_used": False,
        "fallback_provider": None,
        "guardrail": "in_domain",
        "user_question": "Mụn đầu đen là gì?",
        "final_answer": "Mụn đầu đen là dạng nhân mụn mở liên quan bít tắc nang lông.",
        "sources": ["source.pdf"],
    }
    base_state.update(state_patch)

    result = await cache_store_node(base_state)

    assert result == {}
    assert called is False


@pytest.mark.asyncio
async def test_cache_store_rejects_generic_entity_miss_and_unasked_dosage(monkeypatch):
    called = False

    async def fake_set_answer_cache(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("src.agent.nodes.cache.set_answer_cache", fake_set_answer_cache)
    monkeypatch.setenv("CACHE_MIN_ANSWER_CHARS", "20")
    monkeypatch.setenv("CACHE_REQUIRED_ENTITY_CHECK", "true")

    common_state = {
        "cache_hit": False,
        "bypass_cache": False,
        "conversation_history": [],
        "cache_reason": "miss",
        "medical_severity": "routine",
        "severity_guard_modified": False,
        "severity_guard_cache_eligible": True,
        "is_in_domain": True,
        "use_history_context": False,
        "errors": [],
        "llm_fallback": False,
        "llm_fallback_used": False,
        "fallback_provider": None,
        "guardrail": "in_domain",
        "actual_provider": "gemini",
        "sources": ["source.pdf"],
    }

    generic = await cache_store_node(
        {
            **common_state,
            "user_question": "Benzoyl peroxide là gì?",
            "final_answer": (
                "Dựa trên các thông tin bạn cung cấp, đây là những lưu ý chăm sóc da cơ bản. "
                "Giữ vệ sinh da sạch sẽ và sử dụng kem dưỡng ẩm phù hợp."
            ),
        }
    )
    dosage = await cache_store_node(
        {
            **common_state,
            "user_question": "Mụn viêm là gì?",
            "final_answer": "Mụn viêm là tổn thương viêm. Có thể bôi sau khi dưỡng ẩm nếu được hướng dẫn.",
        }
    )

    assert generic == {}
    assert dosage == {}
    assert called is False


@pytest.mark.asyncio
async def test_answer_quality_node_adds_severity_metadata(monkeypatch):
    monkeypatch.setenv("ANSWER_VERIFIER_ENABLED", "true")
    result = await quality_node.answer_quality_node(
        {
            "user_question": "Tôi đang mang thai, có dùng isotretinoin trị mụn được không?",
            "final_answer": "Bạn có thể chăm sóc da nhẹ nhàng.",
        }
    )

    assert result["medical_severity"] == "urgent"
    assert result["severity_guard_modified"] is True
    assert result["severity_guard_cache_eligible"] is False
    assert "Isotretinoin không được tự dùng" in result["final_answer"]
    assert result["answer_quality_report"]["metadata"]["severity_guard"]["version"] == (
        "severity_aware_answer_guard_v1"
    )


@pytest.mark.asyncio
async def test_emergency_quality_guard_reattributes_replaced_answer_to_system(monkeypatch):
    monkeypatch.setenv("ANSWER_VERIFIER_ENABLED", "true")
    result = await quality_node.answer_quality_node(
        {
            "user_question": "Môi và lưỡi sưng sau khi dùng thuốc mới trị mụn, nên làm gì ngay?",
            "final_answer": "Bạn nên theo dõi thêm tại nhà.",
            "actual_provider": "ollama",
            "actual_model": "qwen3:8b",
        }
    )

    assert result["medical_severity"] == "emergency"
    assert result["actual_provider"] == "system"
    assert result["actual_model"] is None
    assert result["fallback_applied"] is True
    assert result["fallback_type"] == "severity_emergency_safety_fallback"
    assert "gọi cấp cứu" in result["final_answer"]


@pytest.mark.asyncio
async def test_answer_quality_node_disabled_and_runtime_error(monkeypatch):
    monkeypatch.setenv("ANSWER_VERIFIER_ENABLED", "false")
    disabled = await quality_node.answer_quality_node(
        {"user_question": "Mụn đầu đen là gì?", "final_answer": "Một dạng nhân mụn mở."}
    )

    monkeypatch.setenv("ANSWER_VERIFIER_ENABLED", "true")

    def raise_error(query):
        raise RuntimeError("normalization failed token=secret-value")

    monkeypatch.setattr(quality_node, "normalize_query", raise_error)
    failed = await quality_node.answer_quality_node(
        {"user_question": "Mụn đầu đen là gì?", "final_answer": "Một dạng nhân mụn mở."}
    )

    assert disabled == {"answer_quality_report": None, "answer_guard_modified": False}
    assert failed["answer_quality_report"]["passed"] is False
    assert failed["answer_quality_report"]["issues"][0]["code"] == "answer_verifier_runtime_error"
    assert "secret-value" not in failed["answer_quality_report"]["issues"][0]["message"]
    assert "token=[REDACTED]" in failed["answer_quality_report"]["issues"][0]["message"]
