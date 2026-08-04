"""Validate the generation-focused Acne Advisor evaluation dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "notebooks" / "eval_data" / "acne_rag_eval_generation_focused.jsonl"

CATEGORY_TARGETS = {
    "core_knowledge_generation": 50,
    "active_ingredients_generation": 45,
    "product_entity_generation": 40,
    "comparison_generation": 40,
    "treatment_plan_reference": 45,
    "routine_skincare_generation": 35,
    "multi_turn_like_generation": 25,
    "exact_format_light": 15,
    "mild_safety_caution": 5,
}

EMERGENCY_TERMS = (
    "khó thở",
    "sưng môi",
    "phồng rộp",
    "sốc phản vệ",
    "stevens-johnson",
    "sjs",
    "gọi 115",
)
PLACEHOLDER_TERMS = ("todo", "tbd", "placeholder", "lorem ipsum", "{topic}", "{ingredient}")


def load_rows(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Line {line_number} is not a JSON object")
        rows.append(value)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> Counter[str]:
    if len(rows) != 300:
        raise ValueError(f"Expected exactly 300 rows, got {len(rows)}")

    ids = [str(row.get("id") or "").strip() for row in rows]
    questions = [str(row.get("question") or "").strip() for row in rows]
    if any(not item for item in ids):
        raise ValueError("Every row must have a non-empty id")
    if any(not item for item in questions):
        raise ValueError("Every row must have a non-empty question")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids found")
    if len(questions) != len(set(questions)):
        raise ValueError("Duplicate questions found")

    counts = Counter(str(row.get("category") or "") for row in rows)
    unknown = set(counts) - set(CATEGORY_TARGETS)
    if unknown:
        raise ValueError(f"Unknown categories: {sorted(unknown)}")
    if counts != CATEGORY_TARGETS:
        raise ValueError(f"Unexpected category distribution: {dict(counts)}")

    for row in rows:
        question = str(row["question"]).strip()
        lowered = question.lower()
        if not isinstance(row.get("expected_keywords"), list) or not row["expected_keywords"]:
            raise ValueError(f"Case {row['id']} must have non-empty expected_keywords")
        if not isinstance(row.get("forbidden_keywords"), list):
            raise ValueError(f"Case {row['id']} must have forbidden_keywords list")
        if any(term in lowered for term in EMERGENCY_TERMS):
            raise ValueError(f"Emergency term found in {row['id']}: {question}")
        if any(term in lowered for term in PLACEHOLDER_TERMS):
            raise ValueError(f"Placeholder found in {row['id']}: {question}")
        if row.get("requires_emergency_action") or row.get("requires_out_of_domain_refusal"):
            raise ValueError(f"Disallowed emergency/OOD flag found in {row['id']}")
    return counts


def main() -> None:
    rows = load_rows()
    counts = validate_rows(rows)
    print("GENERATION_FOCUSED_EVAL_SET: PASS")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Rows: {len(rows)}")
    print(f"Category distribution: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
