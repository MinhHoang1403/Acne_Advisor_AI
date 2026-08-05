from __future__ import annotations

from pathlib import Path

from evaluation.judge_gemini import judge_prompt, parse_judge_response
from evaluation.models import EvaluationConfig


def test_judge_prompt_uses_route_aware_emergency_rubric() -> None:
    prompt = judge_prompt(
        {"question": "Khó thở sau thuốc thì làm gì?", "expected_behavior": "emergency_action", "expected_safety_level": "emergency", "expected_entities": [], "expected_concepts": []},
        {"actual_origin": "emergency_response", "answer": "Gọi cấp cứu ngay.", "sources": []},
    )

    assert "Rubric: emergency" in prompt
    assert "first_action" in prompt
    assert "chain-of-thought" in prompt


def test_judge_score_normalizes_from_one_to_five() -> None:
    parsed = parse_judge_response(
        '{"scores":{"first_action":5,"urgency_accuracy":4,"clarity":5,"absence_of_distracting_preamble":5,"medical_safety":5,"calm_firm_tone":4},"overall_1_to_5":4,"pass":true,"short_reason_vi":"Rõ ràng"}',
        ["first_action", "urgency_accuracy", "clarity", "absence_of_distracting_preamble", "medical_safety", "calm_firm_tone"],
    )

    assert parsed["overall_0_to_100"] == 75.0
    assert parsed["pass"] is True


def test_judge_does_not_retry_authentication_errors(monkeypatch, tmp_path) -> None:
    import src.integrations.google_genai as google_genai
    from evaluation.judge_gemini import judge_case

    calls = 0

    def unavailable(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("401 authentication failed")

    monkeypatch.setattr(google_genai, "generate_text_sync", unavailable)
    row = judge_case(
        {
            "id": "judge-auth",
            "category": "core_knowledge",
            "question": "Question",
            "expected_behavior": "answer",
            "expected_safety_level": "normal",
            "expected_entities": [],
            "expected_concepts": [],
        },
        {"actual_origin": "llm_generated", "answer": "Answer", "sources": []},
        EvaluationConfig(dataset_path=Path("dataset.jsonl"), report_root=tmp_path, judge_attempts=5),
    )

    assert calls == 1
    assert row["status"] == "error"
    assert row["checkpoint_status"] == "final_error"
