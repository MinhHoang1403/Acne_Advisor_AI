"""Validate the canonical comprehensive evaluation dataset without runtime calls."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "notebooks" / "eval_data" / "acne_rag_eval_comprehensive_v1.jsonl"
CATEGORIES = {
    "core_knowledge",
    "active_ingredients",
    "product_entity_alias",
    "comparison",
    "treatment_plan_reference",
    "skincare_routine",
    "multi_turn_context",
    "exact_format_instruction",
    "retrieval_source_traceability",
    "entity_graph_relation",
    "antibiotic_stewardship",
    "pregnancy_lactation",
    "mild_adverse_false_escalation",
    "urgent_emergency",
    "out_of_domain_insufficient_evidence",
}
ROUTES = {"llm_generated", "system_safe_fallback", "guardrail", "any_safe"}
SAFETY_LEVELS = {"normal", "caution", "urgent", "emergency", "out_of_domain"}
REQUIRED_FIELDS = {
    "id",
    "category",
    "question",
    "conversation_history",
    "expected_route",
    "expected_safety_level",
    "expected_entities",
    "expected_concepts",
    "forbidden_concepts",
    "accepted_sources",
    "source_required",
    "format_contract",
    "judge_eligible",
    "critical_case",
    "notes",
}
PLACEHOLDER_PATTERN = re.compile(r"\{[^}]+\}|\b(todo|tbd|placeholder|xxx)\b", re.IGNORECASE)
KNOWN_SOURCE_FILES = {
    "web_raw_dataset.json",
    "acne-vulgaris-management-pdf-66142088866501.pdf",
    "PIIS0190962223033893.pdf",
    "qd_4416_cut.pdf",
}


def normalize(value: str) -> str:
    text = unicodedata.normalize("NFD", value.casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Line {line_number} must be a JSON object")
        rows.append(value)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> Counter[str]:
    errors: list[str] = []
    if len(rows) != 300:
        errors.append(f"Expected 300 rows, found {len(rows)}")
    ids: list[str] = []
    normalized_questions: list[str] = []
    category_counts: Counter[str] = Counter()

    for index, row in enumerate(rows, 1):
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            errors.append(f"row {index}: missing fields {missing}")
            continue
        case_id = row.get("id")
        question = row.get("question")
        category = row.get("category")
        if not isinstance(case_id, str) or not case_id.strip():
            errors.append(f"row {index}: id must be non-empty")
        else:
            ids.append(case_id)
        if not isinstance(question, str) or not question.strip():
            errors.append(f"row {index}: question must be non-empty")
        else:
            normalized_questions.append(normalize(question))
            if PLACEHOLDER_PATTERN.search(question):
                errors.append(f"row {index}: question contains a placeholder")
        if category not in CATEGORIES:
            errors.append(f"row {index}: invalid category {category!r}")
        else:
            category_counts[category] += 1
        if row.get("expected_route") not in ROUTES:
            errors.append(f"row {index}: invalid expected_route")
        if row.get("expected_safety_level") not in SAFETY_LEVELS:
            errors.append(f"row {index}: invalid expected_safety_level")
        for field in ("expected_entities", "expected_concepts", "forbidden_concepts", "accepted_sources"):
            if not isinstance(row.get(field), list) or not all(isinstance(item, str) and item.strip() for item in row[field]):
                errors.append(f"row {index}: {field} must be a list of non-empty strings")
        if not row.get("expected_concepts"):
            errors.append(f"row {index}: expected_concepts cannot be empty")
        if not isinstance(row.get("source_required"), bool):
            errors.append(f"row {index}: source_required must be bool")
        if row.get("source_required") and not row.get("accepted_sources"):
            errors.append(f"row {index}: source-required case lacks accepted_sources")
        for source in row.get("accepted_sources", []):
            if source not in KNOWN_SOURCE_FILES:
                errors.append(f"row {index}: accepted source is not a known document-level source: {source}")
        if not isinstance(row.get("format_contract"), dict) or not row["format_contract"].get("type"):
            errors.append(f"row {index}: format_contract.type is required")
        if not isinstance(row.get("judge_eligible"), bool) or not isinstance(row.get("critical_case"), bool):
            errors.append(f"row {index}: judge_eligible and critical_case must be bool")
        history = row.get("conversation_history")
        if not isinstance(history, list):
            errors.append(f"row {index}: conversation_history must be a list")
        elif any(
            not isinstance(turn, dict)
            or turn.get("role") not in {"user", "assistant"}
            or not isinstance(turn.get("content"), str)
            or not turn["content"].strip()
            for turn in history
        ):
            errors.append(f"row {index}: invalid conversation_history schema")
        required = {normalize(item) for item in row.get("expected_concepts", [])}
        forbidden = {normalize(item) for item in row.get("forbidden_concepts", [])}
        contradiction = required & forbidden
        if contradiction:
            errors.append(f"row {index}: required/forbidden contradiction {sorted(contradiction)}")
        if category == "urgent_emergency":
            if row.get("expected_safety_level") != "emergency" or not row.get("critical_case"):
                errors.append(f"row {index}: emergency case must be critical/emergency")
        elif row.get("expected_safety_level") == "emergency":
            errors.append(f"row {index}: emergency safety level outside urgent_emergency")
        if category == "out_of_domain_insufficient_evidence":
            if row.get("expected_safety_level") != "out_of_domain" or row.get("expected_route") != "guardrail":
                errors.append(f"row {index}: OOD case must expect guardrail/out_of_domain")
        elif row.get("expected_safety_level") == "out_of_domain":
            errors.append(f"row {index}: out_of_domain safety level outside OOD category")
        if category == "pregnancy_lactation" and not row.get("critical_case"):
            errors.append(f"row {index}: pregnancy case must be critical")
        if category == "mild_adverse_false_escalation" and any(
            term in normalize(question or "") for term in ("kho tho", "sung moi", "phong rop", "sjs", "phan ve")
        ):
            errors.append(f"row {index}: mild case contains emergency trigger")

    if len(ids) != len(set(ids)):
        errors.append("duplicate id")
    if len(normalized_questions) != len(set(normalized_questions)):
        errors.append("duplicate normalized question")
    if set(category_counts) != CATEGORIES:
        errors.append(f"category coverage mismatch: {dict(category_counts)}")
    for category in CATEGORIES:
        if category_counts[category] != 20:
            errors.append(f"category {category} must contain 20 cases, found {category_counts[category]}")
    if errors:
        raise ValueError("Comprehensive evaluation dataset validation failed:\n- " + "\n- ".join(errors))
    return category_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    args = parser.parse_args()
    rows = read_jsonl(args.dataset)
    counts = validate_rows(rows)
    print("COMPREHENSIVE_EVAL_SET: PASS")
    print(f"Dataset: {args.dataset.resolve()}")
    print(f"Rows: {len(rows)}")
    print(f"Category distribution: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
