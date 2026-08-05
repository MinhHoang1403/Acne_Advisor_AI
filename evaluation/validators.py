"""Dataset and ground-truth validation for the canonical Evaluation V3 set."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .models import ACCEPTABLE_ORIGINS, CATEGORIES, EXPECTED_BEHAVIORS, SAFETY_LEVELS


PLACEHOLDER_PATTERN = re.compile(r"\{[^}]+\}|\b(todo|tbd|placeholder|xxx)\b", re.IGNORECASE)
REQUIRED_FIELDS = {
    "id",
    "category",
    "question",
    "conversation_history",
    "expected_behavior",
    "acceptable_origins",
    "expected_safety_level",
    "expected_entities",
    "expected_concepts",
    "forbidden_claims",
    "accepted_sources",
    "source_required",
    "format_contract",
    "naturalness_applicable",
    "critical_case",
    "notes",
}

# The Phase 1 manifest is intentionally runtime-only and absent in clean CI.
# These are the document-level source names used by the frozen V3 dataset and
# were verified against the current Phase 1 manifest before it was backed up.
CANONICAL_SOURCE_REFERENCES = frozenset(
    {
        "PIIS0190962223033893.pdf",
        "acne-vulgaris-management-pdf-66142088866501.pdf",
        "qd_4416_cut.pdf",
        "web_raw_dataset.json",
    }
)


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or "").casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number} must be a JSON object")
        rows.append(row)
    return rows


def _known_references(project_root: Path) -> tuple[set[str], set[str]]:
    alias_path = project_root / "data" / "taxonomy" / "drug_aliases.yaml"
    taxonomy_path = project_root / "data" / "taxonomy" / "drug_taxonomy_v2.yaml"
    entities: set[str] = set()
    for path in (alias_path, taxonomy_path):
        if not path.exists():
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for item in payload.get("entities", []) or []:
            if isinstance(item, dict):
                entities.add(normalize(item.get("canonical_name")))
                entities.update(normalize(alias) for alias in item.get("aliases", []) or [])
    manifest_path = project_root / "data" / "ingestion_manifest.json"
    sources: set[str] = set(CANONICAL_SOURCE_REFERENCES)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in (manifest.get("documents") or {}).values():
            if isinstance(entry, dict):
                for key in ("source_file", "source_path", "document_id"):
                    value = entry.get(key)
                    if value:
                        sources.add(str(value))
                        sources.add(Path(str(value)).name)
    return entities, sources


def _near_duplicate(left: str, right: str) -> bool:
    left_tokens, right_tokens = set(normalize(left).split()), set(normalize(right).split())
    if not left_tokens or not right_tokens:
        return False
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) >= 0.88


def validate_cases(rows: list[dict[str, Any]], project_root: Path) -> Counter[str]:
    errors: list[str] = []
    entity_refs, source_refs = _known_references(project_root)
    ids: list[str] = []
    questions: list[str] = []
    category_counts: Counter[str] = Counter()

    if len(rows) != 300:
        errors.append(f"Expected exactly 300 cases, found {len(rows)}")

    for index, row in enumerate(rows, 1):
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            errors.append(f"row {index}: missing {missing}")
            continue
        case_id = row.get("id")
        question = row.get("question")
        if not isinstance(case_id, str) or not case_id.strip():
            errors.append(f"row {index}: id must be non-empty")
        else:
            ids.append(case_id)
        if not isinstance(question, str) or not question.strip():
            errors.append(f"row {index}: question must be non-empty")
        else:
            questions.append(question)
            if PLACEHOLDER_PATTERN.search(question):
                errors.append(f"row {index}: question contains placeholder text")
            # These shapes signal lossy shell encoding such as "v? c?". A
            # normal Vietnamese question can still contain "? Trình bày..."
            # before a capitalized follow-up instruction.
            if "??" in question or re.search(r"\?(?=[A-Za-z])|\?\s+[a-zà-ỹ]", question):
                errors.append(f"row {index}: question contains an internal encoding placeholder")
        category = row.get("category")
        if category not in CATEGORIES:
            errors.append(f"row {index}: invalid category {category!r}")
        else:
            category_counts[category] += 1
        if row.get("expected_behavior") not in EXPECTED_BEHAVIORS:
            errors.append(f"row {index}: invalid expected_behavior")
        if row.get("expected_safety_level") not in SAFETY_LEVELS:
            errors.append(f"row {index}: invalid expected_safety_level")
        origins = row.get("acceptable_origins")
        if not isinstance(origins, list) or not origins or not set(origins) <= ACCEPTABLE_ORIGINS:
            errors.append(f"row {index}: invalid acceptable_origins")
        for field in ("expected_entities", "expected_concepts", "forbidden_claims", "accepted_sources"):
            values = row.get(field)
            if not isinstance(values, list) or not all(isinstance(value, str) and value.strip() for value in values):
                errors.append(f"row {index}: {field} must be a list of non-empty strings")
        if not isinstance(row.get("source_required"), bool):
            errors.append(f"row {index}: source_required must be boolean")
        if row.get("source_required") and not row.get("accepted_sources"):
            errors.append(f"row {index}: source-required case lacks accepted_sources")
        if not isinstance(row.get("format_contract"), dict):
            errors.append(f"row {index}: format_contract must be object")
        if not isinstance(row.get("naturalness_applicable"), bool) or not isinstance(row.get("critical_case"), bool):
            errors.append(f"row {index}: naturalness_applicable and critical_case must be booleans")
        history = row.get("conversation_history")
        if not isinstance(history, list) or any(
            not isinstance(turn, dict)
            or turn.get("role") not in {"user", "assistant"}
            or not isinstance(turn.get("content"), str)
            or not turn["content"].strip()
            for turn in history
        ):
            errors.append(f"row {index}: invalid conversation_history")
        required = {normalize(value) for value in row.get("expected_concepts", [])}
        forbidden = {normalize(value) for value in row.get("forbidden_claims", [])}
        if required & forbidden:
            errors.append(f"row {index}: required/forbidden contradiction")
        for entity in row.get("expected_entities", []):
            if normalize(entity) not in entity_refs:
                errors.append(f"row {index}: unknown entity reference {entity!r}")
        for source in row.get("accepted_sources", []):
            if source not in source_refs and Path(source).name not in source_refs:
                errors.append(f"row {index}: unknown source reference {source!r}")
        if category == "urgent_emergency":
            if row.get("expected_behavior") != "emergency_action" or row.get("expected_safety_level") != "emergency" or not row.get("critical_case"):
                errors.append(f"row {index}: emergency case must be critical emergency_action")
        if category == "mild_adverse_false_escalation" and row.get("expected_safety_level") == "emergency":
            errors.append(f"row {index}: mild adverse case cannot be emergency")
        if category == "out_of_domain_insufficient_evidence" and row.get("expected_safety_level") == "out_of_domain":
            if row.get("expected_behavior") != "refuse_or_redirect":
                errors.append(f"row {index}: OOD case must request refusal/redirection")
        elif row.get("expected_safety_level") == "out_of_domain":
            errors.append(f"row {index}: out_of_domain safety only allowed in OOD category")

    if len(ids) != len(set(ids)):
        errors.append("duplicate case IDs")
    normalized_questions = [normalize(question) for question in questions]
    if len(normalized_questions) != len(set(normalized_questions)):
        errors.append("duplicate normalized questions")
    for index, question in enumerate(questions):
        for other in questions[index + 1 :]:
            if _near_duplicate(question, other):
                errors.append(f"near-duplicate questions: {question!r} / {other!r}")
                break
    if set(category_counts) != set(CATEGORIES):
        errors.append(f"category coverage mismatch: {dict(category_counts)}")
    for category in CATEGORIES:
        if category_counts[category] != 20:
            errors.append(f"{category} must contain 20 cases, got {category_counts[category]}")

    if errors:
        raise ValueError("Evaluation V3 dataset validation failed:\n- " + "\n- ".join(errors))
    return category_counts
