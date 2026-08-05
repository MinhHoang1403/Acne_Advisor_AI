from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from evaluation.validators import load_cases, validate_cases


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "evaluation" / "data" / "acne_system_eval_v3.jsonl"


def test_dataset_v3_is_balanced_and_valid() -> None:
    rows = load_cases(DATASET)
    counts = validate_cases(rows, ROOT)

    assert len(rows) == 300
    assert set(counts) == {
        "core_knowledge",
        "active_ingredients",
        "product_entity_alias",
        "comparison",
        "skincare_routine",
        "treatment_plan_reference",
        "retrieval_source_traceability",
        "entity_graph_relation",
        "multi_turn_context",
        "exact_format_instruction",
        "antibiotic_stewardship",
        "pregnancy_lactation",
        "mild_adverse_false_escalation",
        "urgent_emergency",
        "out_of_domain_insufficient_evidence",
    }
    assert all(value == 20 for value in counts.values())


def test_dataset_v3_has_required_behavior_distribution() -> None:
    rows = load_cases(DATASET)
    behavior = Counter(row["expected_behavior"] for row in rows)

    assert behavior["answer"] + behavior["cautious_answer"] == 240
    assert behavior["emergency_action"] == 20
    assert behavior["refuse_or_redirect"] == 20
    assert behavior["safe_insufficient_evidence"] == 20


def test_dataset_v3_rejects_lossy_internal_question_markers() -> None:
    rows = load_cases(DATASET)
    rows[0] = {**rows[0], "question": "Benzoyl peroxide v? kháng sinh?"}

    with pytest.raises(ValueError, match="encoding placeholder"):
        validate_cases(rows, ROOT)
