"""Build and score the human-review package for P4.5 shadow calibration.

The harness is evaluation-only. It invokes the current clinical graph without
changing retrieval, P3, P4, prompts, or provider settings. Generated records
are checkpointed after every case so provider work is never repeated silently.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.graph import run_clinical_agent  # noqa: E402
from src.observability.versioning import (  # noqa: E402
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
)


QUESTION_SET = ROOT / "tests" / "golden" / "p45_production_shadow_questions.json"
P45_SCHEMA_VERSION = "p45_calibration_package_v1"
P45_REVIEW_SCHEMA_VERSION = "p45_human_evidence_support_review_v1"
VERDICTS = (
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "CONTRADICTED",
    "NO_EVIDENCE",
)
HUMAN_VERDICTS = VERDICTS + ("REVIEW_UNCERTAIN",)
CRITICALITIES = ("CRITICAL", "NON_CRITICAL")
ORIGINS = ("HUMAN_WRITTEN", "HISTORICAL_USER_STYLE", "CURATED_PRODUCTION_LIKE", "EDGE_CASE")
LANGUAGES = ("VI", "EN", "MIXED")
ERROR_ROOT_CAUSES = (
    "CLAIM_EXTRACTION_ERROR",
    "CRITICALITY_ERROR",
    "EVIDENCE_MAPPING_ERROR",
    "ENTAILMENT_ERROR",
    "PROVENANCE_ERROR",
    "POLICY_ERROR",
    "HUMAN_LABEL_UNCERTAIN",
    "OTHER",
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CalibrationQuestion(FrozenModel):
    id: str
    question: str
    language: str
    category: str
    safety_category: str
    origin: str
    critical_expected: bool

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if value not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}")
        return value

    @field_validator("origin")
    @classmethod
    def validate_origin(cls, value: str) -> str:
        if value not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}")
        return value


class QuestionSet(FrozenModel):
    schema_version: str
    dataset_version: str
    frozen: bool
    questions: tuple[CalibrationQuestion, ...]


class HumanReviewRecord(BaseModel):
    """Editable review row. Null human fields are an intentional template."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = P45_REVIEW_SCHEMA_VERSION
    record_type: str
    case_id: str
    claim_id: str | None = None
    question: str
    answer: str
    claim_text: str | None = None
    claim_type: str | None = None
    criticality: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_text: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    provenance_valid: bool | None = None
    p4_verdict: str | None = None
    p4_shadow_action: str | None = None
    human_verdict: str | None = None
    human_criticality: str | None = None
    mapping_correct: bool | None = None
    claim_extraction_correct: bool | None = None
    missing_claims: list[str] = Field(default_factory=list)
    missing_critical_claims: list[str] = Field(default_factory=list)
    incorrectly_split: bool | None = None
    unrelated_fragments: list[str] = Field(default_factory=list)
    reviewer_confidence: str | None = None
    reviewer_notes: str | None = None
    reviewer_id: str | None = None
    reviewer_qualification: str | None = None
    adjudication_status: str = "PENDING"
    adjudicated_verdict: str | None = None
    error_root_cause: str | None = None
    error_subtype: str | None = None

    @field_validator("human_verdict", "adjudicated_verdict")
    @classmethod
    def validate_human_verdict(cls, value: str | None) -> str | None:
        if value is not None and value not in HUMAN_VERDICTS:
            raise ValueError(f"human verdict must be one of {HUMAN_VERDICTS}")
        return value

    @field_validator("human_criticality")
    @classmethod
    def validate_human_criticality(cls, value: str | None) -> str | None:
        if value is not None and value not in CRITICALITIES:
            raise ValueError(f"human criticality must be one of {CRITICALITIES}")
        return value

    @field_validator("error_root_cause")
    @classmethod
    def validate_root_cause(cls, value: str | None) -> str | None:
        if value is not None and value not in ERROR_ROOT_CAUSES:
            raise ValueError(f"error_root_cause must be one of {ERROR_ROOT_CAUSES}")
        return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_question_set(path: Path = QUESTION_SET) -> QuestionSet:
    return QuestionSet.model_validate_json(path.read_text(encoding="utf-8"))


def validate_question_set(question_set: QuestionSet) -> dict[str, Any]:
    questions = list(question_set.questions)
    ids = [question.id for question in questions]
    texts = [question.question.casefold().strip() for question in questions]
    critical_count = sum(question.critical_expected for question in questions)
    category_counts = Counter(question.category for question in questions)
    language_counts = Counter(question.language for question in questions)
    origin_counts = Counter(question.origin for question in questions)
    errors: list[str] = []
    if not question_set.frozen:
        errors.append("question set must be frozen")
    if not 50 <= len(questions) <= 100:
        errors.append("question count must be between 50 and 100")
    if len(ids) != len(set(ids)):
        errors.append("question IDs must be unique")
    if len(texts) != len(set(texts)):
        errors.append("question text must be unique")
    if len(category_counts) < 15:
        errors.append("at least 15 categories are required")
    if set(language_counts) != set(LANGUAGES):
        errors.append("VI, EN, and MIXED language coverage is required")
    if set(origin_counts) != set(ORIGINS):
        errors.append("all declared origin categories are required")
    critical_ratio = critical_count / len(questions) if questions else 0.0
    if not 0.20 <= critical_ratio <= 0.25:
        errors.append("critical question ratio must be between 20% and 25%")
    return {
        "passed": not errors,
        "errors": errors,
        "question_count": len(questions),
        "critical_question_count": critical_count,
        "critical_question_ratio": round(critical_ratio, 6),
        "categories": dict(sorted(category_counts.items())),
        "languages": dict(sorted(language_counts.items())),
        "origins": dict(sorted(origin_counts.items())),
    }


def assert_shadow_runtime() -> dict[str, Any]:
    manifest = build_pipeline_version_manifest()
    problems: list[str] = []
    if manifest.get("retrieval_pipeline_version") != "v5":
        problems.append("RETRIEVAL_PIPELINE_VERSION must resolve to v5")
    if manifest.get("p4_requested_mode") != "shadow" or manifest.get("p4_mode") != "shadow":
        problems.append("P4 requested and effective mode must both be shadow")
    if manifest.get("p4_critical_enforcement_ready") is not False:
        problems.append("critical enforcement readiness must remain false")
    if manifest.get("p4_all_claim_enforcement_ready") is not False:
        problems.append("all-claim enforcement readiness must remain false")
    if problems:
        raise RuntimeError("P4_5_MODE_BLOCKER: " + "; ".join(problems))
    return manifest


async def generate_outputs(
    *,
    question_set: QuestionSet,
    output_dir: Path,
    limit: int | None = None,
    retry_failures: bool = False,
) -> dict[str, Any]:
    manifest = assert_shadow_runtime()
    validation = validate_question_set(question_set)
    if not validation["passed"]:
        raise ValueError("Invalid P4.5 question set: " + "; ".join(validation["errors"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "p45_generated_answers.json"
    checkpoint = _load_generation_checkpoint(checkpoint_path, question_set.dataset_version)
    records_by_id = {record["case_id"]: record for record in checkpoint["records"]}
    selected = list(question_set.questions)
    if limit is not None:
        selected = selected[: max(0, limit)]

    for index, question in enumerate(selected, start=1):
        existing = records_by_id.get(question.id)
        if existing and (existing.get("status") == "SUCCESS" or not retry_failures):
            continue
        started = datetime.now(timezone.utc)
        try:
            result = await run_clinical_agent(
                question.question,
                user_id="p45-evaluation",
                session_id=f"p45-{question.id}",
                conversation_history=[],
                bypass_cache=True,
            )
            record = capture_generation_record(
                question=question,
                result=result,
                started=started,
                ordinal=index,
            )
        except Exception as exc:  # Generation failures are frozen, never hand-replaced.
            record = {
                **question.model_dump(mode="json"),
                "case_id": question.id,
                "ordinal": index,
                "status": "GENERATION_FAILURE",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
                "started_at": started.isoformat(),
                "completed_at": utc_now(),
                "cache_hit": None,
                "fresh_generation": False,
                "final_answer": "",
                "p4_mode": "shadow",
                "claim_grounding": None,
            }
        records_by_id[question.id] = record
        checkpoint["records"] = [
            records_by_id[item.id]
            for item in question_set.questions
            if item.id in records_by_id
        ]
        checkpoint["updated_at"] = utc_now()
        checkpoint["pipeline_manifest"] = manifest
        checkpoint["pipeline_fingerprint"] = compute_pipeline_fingerprint(manifest)
        _write_json_atomic(checkpoint_path, checkpoint)
        print(
            f"[{index}/{len(selected)}] {question.id}: {record['status']} "
            f"provider={record.get('actual_provider')} model={record.get('actual_model')}",
            flush=True,
        )
    return checkpoint


def capture_generation_record(
    *,
    question: CalibrationQuestion,
    result: dict[str, Any],
    started: datetime,
    ordinal: int,
) -> dict[str, Any]:
    packed = result.get("packed_context") or {}
    p3 = result.get("evidence_sufficiency") or {}
    claim_grounding = result.get("claim_grounding")
    p4_mode = str(result.get("p4_mode") or "")
    if p4_mode and p4_mode != "shadow":
        raise RuntimeError(f"P4 mode changed during generation: {p4_mode}")
    return {
        **question.model_dump(mode="json"),
        "case_id": question.id,
        "ordinal": ordinal,
        "status": "SUCCESS",
        "error_type": None,
        "error_message": None,
        "started_at": started.isoformat(),
        "completed_at": utc_now(),
        "duration_seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 6),
        "p3_status": p3.get("status"),
        "retry_count": max(0, int(result.get("retrieval_attempt") or 0)),
        "final_answer": str(result.get("answer") or ""),
        "sources": _json_value(result.get("sources") or []),
        "source_allowlist": _json_value(result.get("source_allowlist") or []),
        "packed_evidence": _capture_packed_evidence(packed),
        "packed_evidence_ids": [item["evidence_id"] for item in _capture_packed_evidence(packed)],
        "p4_mode": p4_mode or "shadow",
        "p4_shadow_action": result.get("p4_shadow_policy"),
        "p4_degraded": result.get("p4_degraded"),
        "p4_answer_modified": result.get("p4_answer_modified"),
        "claim_grounding": _json_value(claim_grounding),
        "p4_trace": _json_value(result.get("p4_trace") or []),
        "cache_checked": result.get("cache_checked"),
        "cache_hit": bool(result.get("cache_hit")),
        "fresh_generation": not bool(result.get("cache_hit")),
        "fallback_applied": bool(result.get("fallback_applied")),
        "fallback_type": result.get("fallback_type"),
        "guardrail": result.get("guardrail"),
        "requested_provider": result.get("requested_provider"),
        "requested_model": result.get("requested_model"),
        "actual_provider": result.get("actual_provider"),
        "actual_model": result.get("actual_model"),
        "llm_fallback_used": bool(result.get("llm_fallback_used")),
        "fallback_provider": result.get("fallback_provider"),
        "fallback_model": result.get("fallback_model"),
        "pipeline_fingerprint": result.get("pipeline_fingerprint"),
        "performance_timings": _json_value(result.get("performance_timings") or {}),
    }


def build_review_package(
    *,
    question_set: QuestionSet,
    output_dir: Path,
) -> dict[str, Any]:
    generated_path = output_dir / "p45_generated_answers.json"
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    records = generated.get("records", [])
    questions_by_id = {item.id: item for item in question_set.questions}
    claim_rows: list[dict[str, Any]] = []
    review_rows: list[HumanReviewRecord] = []

    for record in records:
        question = questions_by_id[record["case_id"]]
        review_rows.append(_case_review_row(question, record))
        grounding = record.get("claim_grounding") or {}
        claims = grounding.get("claims") or []
        verdicts = {item.get("claim_id"): item for item in grounding.get("verdicts") or []}
        links_by_claim: dict[str, list[dict[str, Any]]] = {}
        for link in grounding.get("evidence_links") or []:
            links_by_claim.setdefault(str(link.get("claim_id")), []).append(link)
        packed_by_id = {
            item["evidence_id"]: item
            for item in record.get("packed_evidence") or []
        }
        for claim in claims:
            claim_id = str(claim.get("claim_id"))
            links = links_by_claim.get(claim_id, [])
            verdict = verdicts.get(claim_id) or {}
            evidence_ids = [str(link.get("evidence_id")) for link in links]
            evidence = [packed_by_id[item] for item in evidence_ids if item in packed_by_id]
            row = {
                "case_id": question.id,
                "question": question.question,
                "language": question.language,
                "category": question.category,
                "safety_category": question.safety_category,
                "answer": record.get("final_answer", ""),
                "claim_id": claim_id,
                "claim_text": claim.get("text"),
                "claim_type": claim.get("claim_type"),
                "criticality": claim.get("criticality"),
                "evidence_ids": evidence_ids,
                "evidence_text": [str(item.get("text") or "") for item in evidence],
                "source_ids": [str(link.get("source_id") or "") for link in links],
                "provenance_valid": all(bool(link.get("provenance_valid")) for link in links),
                "p4_verdict": verdict.get("verdict"),
                "p4_reason_code": verdict.get("reason_code"),
                "p4_shadow_action": grounding.get("shadow_action"),
            }
            claim_rows.append(row)
            review_rows.append(_claim_review_row(row))

    _write_json_atomic(output_dir / "p45_claims.json", claim_rows)
    blind_rows = [_blind_review_row(row) for row in review_rows]
    prediction_rows = [row.model_dump(mode="json") for row in review_rows]
    _write_jsonl_atomic(output_dir / "p45_review_blind.jsonl", blind_rows)
    _write_jsonl_atomic(output_dir / "p45_review_with_predictions.jsonl", prediction_rows)
    _write_jsonl_atomic(output_dir / "p45_human_labels.jsonl", blind_rows)
    _write_jsonl_atomic(output_dir / "p45_adjudicated_labels.jsonl", prediction_rows)

    waiting = build_waiting_metrics(question_set, records, claim_rows)
    _write_json_atomic(output_dir / "p45_confusion_matrix.json", waiting["confusion_matrix"])
    _write_json_atomic(output_dir / "p45_metrics.json", waiting)
    _write_json_atomic(output_dir / "p45_errors.json", [])
    _write_json_atomic(output_dir / "p45_questions.json", question_set.model_dump(mode="json"))
    _write_markdown_artifacts(
        output_dir=output_dir,
        question_set=question_set,
        generated=generated,
        claim_rows=claim_rows,
        metrics=waiting,
    )
    dataset_manifest = freeze_dataset_manifest(
        output_dir=output_dir,
        question_set=question_set,
        generated=generated,
        claim_rows=claim_rows,
    )
    _write_json_atomic(output_dir / "p45_dataset_manifest.json", dataset_manifest)
    (output_dir / "P45_DATASET_MANIFEST.md").write_text(
        _dataset_manifest_markdown(dataset_manifest),
        encoding="utf-8",
    )
    return dataset_manifest


def build_waiting_metrics(
    question_set: QuestionSet,
    generated_records: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    successful = [item for item in generated_records if item.get("status") == "SUCCESS"]
    critical_claims = [item for item in claim_rows if item.get("criticality") == "CRITICAL"]
    verdict_counts = Counter(str(item.get("p4_verdict")) for item in claim_rows)
    return {
        "schema_version": "p45_metrics_v1",
        "status": "P4_5_WAITING_FOR_HUMAN_REVIEW",
        "generated_at": utc_now(),
        "questions": len(question_set.questions),
        "successful_outputs": len(successful),
        "generation_failures": len(generated_records) - len(successful),
        "extracted_claims": len(claim_rows),
        "critical_claims": len(critical_claims),
        "average_claims_per_successful_answer": _ratio(len(claim_rows), len(successful)),
        "p4_prediction_distribution": dict(sorted(verdict_counts.items())),
        "human_reviewed_claims": 0,
        "adjudicated_claims": 0,
        "adjudicated_critical_claims": 0,
        "unresolved_review_uncertain": 0,
        "confusion_matrix": {gold: {predicted: 0 for predicted in VERDICTS} for gold in VERDICTS},
        "readiness": "P4_CRITICAL_ENFORCEMENT_NEEDS_MORE_DATA",
        "readiness_reason": "HUMAN_REVIEW_REQUIRED",
    }


def validate_human_labels(path: Path, expected_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    expected_list = list(expected_rows)
    expected = {(row["record_type"], row["case_id"], row.get("claim_id")) for row in expected_list}
    expected_by_id = {
        (row["record_type"], row["case_id"], row.get("claim_id")): row
        for row in expected_list
    }
    parsed: list[HumanReviewRecord] = []
    errors: list[str] = []
    try:
        parsed = [HumanReviewRecord.model_validate(item) for item in _read_jsonl(path)]
    except (ValidationError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    actual = {(row.record_type, row.case_id, row.claim_id) for row in parsed}
    if actual != expected:
        errors.append("review row identities do not match the frozen prediction package")
    immutable_fields = (
        "question",
        "answer",
        "claim_text",
        "claim_type",
        "criticality",
        "evidence_ids",
        "evidence_text",
        "source_ids",
        "provenance_valid",
    )
    for row in parsed:
        identity = (row.record_type, row.case_id, row.claim_id)
        frozen = expected_by_id.get(identity)
        if frozen is None:
            continue
        actual_row = row.model_dump(mode="json")
        changed = [field for field in immutable_fields if actual_row.get(field) != frozen.get(field)]
        if changed:
            errors.append(
                f"{row.case_id}/{row.claim_id or 'CASE'} changed frozen fields: {', '.join(changed)}"
            )
    claim_rows = [row for row in parsed if row.record_type == "CLAIM_REVIEW"]
    case_rows = [row for row in parsed if row.record_type == "CASE_REVIEW"]
    completed = [
        row
        for row in claim_rows
        if row.human_verdict is not None
        and row.human_criticality is not None
        and row.mapping_correct is not None
        and row.claim_extraction_correct is not None
        and row.reviewer_confidence is not None
        and row.reviewer_id is not None
        and row.reviewer_qualification is not None
    ]
    completed_cases = [
        row
        for row in case_rows
        if row.incorrectly_split is not None
        and row.reviewer_confidence is not None
        and row.reviewer_id is not None
        and row.reviewer_qualification is not None
    ]
    return {
        "passed": not errors,
        "errors": errors,
        "row_count": len(parsed),
        "claim_row_count": len(claim_rows),
        "completed_claim_labels": len(completed),
        "case_row_count": len(case_rows),
        "completed_case_reviews": len(completed_cases),
        "complete": bool(claim_rows)
        and len(completed) == len(claim_rows)
        and len(completed_cases) == len(case_rows),
    }


def merge_human_labels_with_predictions(
    human_rows: list[HumanReviewRecord],
    prediction_rows: Iterable[dict[str, Any]],
) -> list[HumanReviewRecord]:
    """Restore hidden P4 fields without trusting editable reviewer input."""

    predictions = {
        (row["record_type"], row["case_id"], row.get("claim_id")): row
        for row in prediction_rows
    }
    merged: list[HumanReviewRecord] = []
    for human in human_rows:
        identity = (human.record_type, human.case_id, human.claim_id)
        prediction = predictions.get(identity)
        if prediction is None:
            raise ValueError(f"Missing frozen prediction for {identity}")
        value = human.model_dump(mode="json")
        value["p4_verdict"] = prediction.get("p4_verdict")
        value["p4_shadow_action"] = prediction.get("p4_shadow_action")
        merged.append(HumanReviewRecord.model_validate(value))
    return merged


def score_human_labels(rows: list[HumanReviewRecord]) -> dict[str, Any]:
    claim_rows = [row for row in rows if row.record_type == "CLAIM_REVIEW"]
    adjudicated: list[tuple[HumanReviewRecord, str]] = []
    unresolved = 0
    for row in claim_rows:
        label = row.adjudicated_verdict or row.human_verdict
        if label == "REVIEW_UNCERTAIN":
            unresolved += 1
        elif label in VERDICTS:
            adjudicated.append((row, label))
    confusion = {gold: {predicted: 0 for predicted in VERDICTS} for gold in VERDICTS}
    verifier_errors = 0
    for row, gold in adjudicated:
        if row.p4_verdict in VERDICTS:
            confusion[gold][str(row.p4_verdict)] += 1
        else:
            verifier_errors += 1
    supported_pred = sum(confusion[gold]["SUPPORTED"] for gold in VERDICTS)
    supported_gold = sum(confusion["SUPPORTED"].values())
    supported_true = confusion["SUPPORTED"]["SUPPORTED"]
    critical = [(row, gold) for row, gold in adjudicated if row.human_criticality == "CRITICAL"]
    critical_non_supported = [(row, gold) for row, gold in critical if gold != "SUPPORTED"]
    false_allow = [
        (row, gold)
        for row, gold in critical_non_supported
        if row.p4_shadow_action == "WOULD_ALLOW"
    ]
    supported_critical = [(row, gold) for row, gold in critical if gold == "SUPPORTED"]
    false_block = [
        (row, gold)
        for row, gold in supported_critical
        if row.p4_shadow_action == "WOULD_BLOCK_CRITICAL"
    ]
    critical_unsupported = [(row, gold) for row, gold in critical if gold == "UNSUPPORTED"]
    critical_contradicted = [(row, gold) for row, gold in critical if gold == "CONTRADICTED"]
    critical_no_evidence = [(row, gold) for row, gold in critical if gold == "NO_EVIDENCE"]
    critical_unsupported_safe = [
        (row, gold) for row, gold in critical_unsupported if row.p4_shadow_action != "WOULD_ALLOW"
    ]
    critical_contradicted_safe = [
        (row, gold) for row, gold in critical_contradicted if row.p4_shadow_action != "WOULD_ALLOW"
    ]
    critical_no_evidence_false_allow = [
        (row, gold) for row, gold in critical_no_evidence if row.p4_shadow_action == "WOULD_ALLOW"
    ]
    critical_verifier_errors = [
        (row, gold) for row, gold in critical if row.p4_verdict == "VERIFIER_ERROR"
    ]
    critical_verifier_error_allows = [
        (row, gold) for row, gold in critical_verifier_errors if row.p4_shadow_action == "WOULD_ALLOW"
    ]
    case_rows = [row for row in rows if row.record_type == "CASE_REVIEW"]
    missing_claims = sum(len(row.missing_claims) for row in case_rows)
    missing_critical = sum(len(row.missing_critical_claims) for row in case_rows)
    extraction_reviewed = [row for row, _ in adjudicated if row.claim_extraction_correct is not None]
    extraction_valid = sum(row.claim_extraction_correct is True for row in extraction_reviewed)
    human_critical_extracted = sum(
        row.human_criticality == "CRITICAL" and row.claim_extraction_correct is True
        for row, _ in adjudicated
    )
    mapping_reviewed = [row for row, _ in adjudicated if row.mapping_correct is not None]
    provenance_links = [row for row, _ in adjudicated if row.evidence_ids]
    metrics = {
        "overall_accuracy": _ratio(sum(confusion[label][label] for label in VERDICTS), len(adjudicated)),
        "supported_precision": _ratio(supported_true, supported_pred),
        "supported_recall": _ratio(supported_true, supported_gold),
        "partial_support_detection": _class_ratio(confusion, "PARTIALLY_SUPPORTED"),
        "unsupported_exact_detection": _class_ratio(confusion, "UNSUPPORTED"),
        "contradiction_exact_detection": _class_ratio(confusion, "CONTRADICTED"),
        "no_evidence_exact_detection": _class_ratio(confusion, "NO_EVIDENCE"),
        "claim_extraction_precision": _ratio(extraction_valid, len(extraction_reviewed)),
        "claim_extraction_recall": _ratio(extraction_valid, extraction_valid + missing_claims),
        "critical_claim_extraction_recall": _ratio(
            human_critical_extracted,
            human_critical_extracted + missing_critical,
        ),
        "evidence_mapping_accuracy": _ratio(
            sum(row.mapping_correct is True for row in mapping_reviewed),
            len(mapping_reviewed),
        ),
        "provenance_validity": _ratio(
            sum(row.provenance_valid is True for row in provenance_links),
            len(provenance_links),
        ),
        "critical_false_allow_rate": _ratio(len(false_allow), len(critical_non_supported)),
        "critical_false_allow_numerator": len(false_allow),
        "critical_false_allow_denominator": len(critical_non_supported),
        "critical_false_block_rate": _ratio(len(false_block), len(supported_critical)),
        "critical_false_block_numerator": len(false_block),
        "critical_false_block_denominator": len(supported_critical),
        "critical_unsupported_safe_detection": _ratio(
            len(critical_unsupported_safe), len(critical_unsupported)
        ),
        "critical_unsupported_safe_numerator": len(critical_unsupported_safe),
        "critical_unsupported_denominator": len(critical_unsupported),
        "critical_contradiction_safe_detection": _ratio(
            len(critical_contradicted_safe), len(critical_contradicted)
        ),
        "critical_contradiction_safe_numerator": len(critical_contradicted_safe),
        "critical_contradiction_denominator": len(critical_contradicted),
        "critical_no_evidence_false_allow": len(critical_no_evidence_false_allow),
        "critical_no_evidence_denominator": len(critical_no_evidence),
        "verifier_error_count": verifier_errors,
        "critical_verifier_error_count": len(critical_verifier_errors),
        "critical_verifier_error_allow_count": len(critical_verifier_error_allows),
    }
    sample_gates = {
        "questions_at_least_50": len({row.case_id for row in rows}) >= 50,
        "claims_at_least_150": len(adjudicated) >= 150,
        "critical_claims_at_least_30": len(critical) >= 30,
    }
    metric_gates = {
        "critical_extraction_recall_full": metrics["critical_claim_extraction_recall"] == 1.0,
        "critical_false_allow_zero": metrics["critical_false_allow_numerator"] == 0,
        "critical_unsupported_safe_detection_full": (
            metrics["critical_unsupported_safe_detection"] == 1.0
        ),
        "critical_contradiction_safe_detection_full": (
            metrics["critical_contradiction_safe_detection"] == 1.0
        ),
        "critical_no_evidence_false_allow_zero": (
            metrics["critical_no_evidence_false_allow"] == 0
        ),
        "critical_verifier_error_never_allows": (
            metrics["critical_verifier_error_allow_count"] == 0
        ),
        "provenance_validity_full": metrics["provenance_validity"] == 1.0,
    }
    if not all(sample_gates.values()):
        readiness = "P4_CRITICAL_ENFORCEMENT_NEEDS_MORE_DATA"
    elif all(metric_gates.values()):
        readiness = "P4_CRITICAL_ENFORCEMENT_CALIBRATION_PASS"
    else:
        readiness = "P4_CRITICAL_ENFORCEMENT_NOT_READY"
    return {
        "schema_version": "p45_human_metrics_v1",
        "generated_at": utc_now(),
        "adjudicated_claims": len(adjudicated),
        "adjudicated_critical_claims": len(critical),
        "unresolved_review_uncertain": unresolved,
        "confusion_matrix": confusion,
        "metrics": metrics,
        "sample_gates": sample_gates,
        "metric_gates": metric_gates,
        "readiness": readiness,
    }


def freeze_dataset_manifest(
    *,
    output_dir: Path,
    question_set: QuestionSet,
    generated: dict[str, Any],
    claim_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    files = [
        "p45_questions.json",
        "p45_generated_answers.json",
        "p45_claims.json",
        "p45_review_blind.jsonl",
        "p45_review_with_predictions.jsonl",
        "p45_human_labels.jsonl",
        "p45_adjudicated_labels.jsonl",
    ]
    records = generated.get("records", [])
    return {
        "schema_version": "p45_dataset_manifest_v1",
        "dataset_version": question_set.dataset_version,
        "timestamp": utc_now(),
        "frozen": True,
        "question_count": len(question_set.questions),
        "generated_output_count": sum(item.get("status") == "SUCCESS" for item in records),
        "generation_failure_count": sum(item.get("status") != "SUCCESS" for item in records),
        "claim_count": len(claim_rows),
        "critical_claim_count": sum(item.get("criticality") == "CRITICAL" for item in claim_rows),
        "languages": dict(Counter(item.language for item in question_set.questions)),
        "categories": dict(Counter(item.category for item in question_set.questions)),
        "sha256": {name: _sha256(output_dir / name) for name in files},
        "generation_config_fingerprint": generated.get("pipeline_fingerprint"),
        "generation_provider_models": sorted(
            {
                f"{item.get('actual_provider')}:{item.get('actual_model')}"
                for item in records
                if item.get("status") == "SUCCESS"
            }
        ),
        "p4_config": {
            "mode": "shadow",
            "claim_grounding_version": "claim_level_grounding_v1",
            "evidence_mapping_version": "claim_evidence_mapping_v1",
            "entailment_version": "deterministic_entailment_v1",
            "critical_policy_version": "critical_claim_policy_v1",
        },
        "review_type": "HUMAN EVIDENCE-SUPPORT REVIEW",
        "reviewer_count": 0,
        "adjudication_status": "PENDING_HUMAN_REVIEW",
    }


def verify_frozen_package(output_dir: Path) -> dict[str, Any]:
    """Verify immutable generation/review inputs before labels are scored."""

    manifest_path = output_dir / "p45_dataset_manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "errors": ["p45_dataset_manifest.json is missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    immutable_files = (
        "p45_questions.json",
        "p45_generated_answers.json",
        "p45_claims.json",
        "p45_review_blind.jsonl",
        "p45_review_with_predictions.jsonl",
    )
    expected_hashes = manifest.get("sha256") or {}
    errors: list[str] = []
    actual_hashes: dict[str, str | None] = {}
    for name in immutable_files:
        path = output_dir / name
        actual = _sha256(path) if path.exists() else None
        actual_hashes[name] = actual
        if actual != expected_hashes.get(name):
            errors.append(f"frozen hash mismatch: {name}")
    if manifest.get("frozen") is not True:
        errors.append("dataset manifest is not frozen")
    if manifest.get("p4_config", {}).get("mode") != "shadow":
        errors.append("frozen package was not generated in P4 shadow mode")
    return {
        "passed": not errors,
        "errors": errors,
        "dataset_version": manifest.get("dataset_version"),
        "verified_hashes": actual_hashes,
    }


def _capture_packed_evidence(packed: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in packed.get("items") or []:
        payload = item.get("payload") or {}
        evidence_id = str(item.get("item_id") or payload.get("chunk_id") or "")
        rows.append(
            {
                "evidence_id": evidence_id,
                "source": item.get("source"),
                "role": item.get("role"),
                "text": str(item.get("text") or ""),
                "chunk_id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "source_path": payload.get("source_path"),
                "source_file": payload.get("source_file"),
            }
        )
    return rows


def _case_review_row(question: CalibrationQuestion, record: dict[str, Any]) -> HumanReviewRecord:
    return HumanReviewRecord(
        record_type="CASE_REVIEW",
        case_id=question.id,
        question=question.question,
        answer=str(record.get("final_answer") or ""),
    )


def _claim_review_row(row: dict[str, Any]) -> HumanReviewRecord:
    return HumanReviewRecord(
        record_type="CLAIM_REVIEW",
        case_id=row["case_id"],
        claim_id=row["claim_id"],
        question=row["question"],
        answer=row["answer"],
        claim_text=row["claim_text"],
        claim_type=row["claim_type"],
        criticality=row["criticality"],
        evidence_ids=list(row["evidence_ids"]),
        evidence_text=list(row["evidence_text"]),
        source_ids=list(row["source_ids"]),
        provenance_valid=bool(row["provenance_valid"]),
        p4_verdict=row["p4_verdict"],
        p4_shadow_action=row["p4_shadow_action"],
    )


def _blind_review_row(row: HumanReviewRecord) -> dict[str, Any]:
    value = row.model_dump(mode="json")
    value["p4_verdict"] = None
    value["p4_shadow_action"] = None
    return value


def _load_generation_checkpoint(path: Path, dataset_version: str) -> dict[str, Any]:
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("dataset_version") != dataset_version:
            raise ValueError("Existing checkpoint belongs to another dataset version")
        return value
    return {
        "schema_version": P45_SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "bypass_cache": True,
        "records": [],
    }


def _write_markdown_artifacts(
    *,
    output_dir: Path,
    question_set: QuestionSet,
    generated: dict[str, Any],
    claim_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    records = generated.get("records", [])
    by_id = {item["case_id"]: item for item in records}
    successful = sum(item.get("status") == "SUCCESS" for item in records)
    failures = len(records) - successful
    critical_claims = sum(item.get("criticality") == "CRITICAL" for item in claim_rows)
    case_table = [
        "| Case | Category | Language | P3 | Claims | Critical claims | P4 would block? | Human reviewed? |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for question in question_set.questions:
        record = by_id.get(question.id, {})
        rows = [item for item in claim_rows if item["case_id"] == question.id]
        case_table.append(
            f"| {question.id} | {question.category} | {question.language} | "
            f"{record.get('p3_status') or 'N/A'} | {len(rows)} | "
            f"{sum(item.get('criticality') == 'CRITICAL' for item in rows)} | "
            f"{'YES' if record.get('p4_shadow_action') == 'WOULD_BLOCK_CRITICAL' else 'NO'} | NO |"
        )
    claim_table = [
        "| Case | Claim | Critical? | P4 | Human | Mapping correct? | Extraction correct? | Match? |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    critical_table = [
        "| Case | Critical claim | Human verdict | P4 verdict | Shadow action | False allow? | False block? |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in claim_rows:
        claim = str(row.get("claim_text") or "").replace("|", "\\|").replace("\n", " ")
        claim_table.append(
            f"| {row['case_id']} | {claim} | "
            f"{'YES' if row.get('criticality') == 'CRITICAL' else 'NO'} | "
            f"{row.get('p4_verdict')} | PENDING | PENDING | PENDING | PENDING |"
        )
        if row.get("criticality") == "CRITICAL":
            critical_table.append(
                f"| {row['case_id']} | {claim} | PENDING | {row.get('p4_verdict')} | "
                f"{row.get('p4_shadow_action')} | PENDING | PENDING |"
            )
    question_table = [
        "| Case | Question | Category | Language | Safety | Origin | Critical expected? |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in question_set.questions:
        escaped_question = item.question.replace("|", "\\|")
        question_table.append(
            f"| {item.id} | {escaped_question} | {item.category} | "
            f"{item.language} | {item.safety_category} | {item.origin} | "
            f"{'YES' if item.critical_expected else 'NO'} |"
        )

    docs = {
        "P45_PRODUCTION_SHADOW_CALIBRATION_REPORT.md": (
            "# P4.5 Production Shadow Calibration\n\n"
            f"Dataset `{question_set.dataset_version}` contains {len(question_set.questions)} frozen questions. "
            f"Generation produced {successful} successful outputs and {failures} failures, with {len(claim_rows)} "
            f"extracted claims ({critical_claims} critical). P4 remained shadow and answers were not rewritten.\n\n"
            "Final readiness scoring is intentionally blocked until real human evidence-support labels are supplied.\n"
        ),
        "P45_QUESTION_SET.md": "# P4.5 Frozen Question Set\n\n" + "\n".join(question_table) + "\n",
        "P45_HUMAN_REVIEW_GUIDE.md": _human_review_guide(),
        "P45_CLAIM_REVIEW_RESULTS.md": (
            "# P4.5 Claim Review Results\n\n## Case summary\n\n"
            + "\n".join(case_table)
            + "\n\n## Claim review\n\n"
            + "\n".join(claim_table)
            + "\n\n## Critical review\n\n"
            + "\n".join(critical_table)
            + "\n"
        ),
        "P45_CRITICAL_ENFORCEMENT_READINESS.md": (
            "# P4.5 Critical Enforcement Readiness\n\n"
            "Current decision: `P4_CRITICAL_ENFORCEMENT_NEEDS_MORE_DATA`.\n\n"
            "Reason: actual human evidence-support review and adjudication have not been completed. "
            "P4 remains `shadow`; neither enforcement interlock is enabled.\n"
        ),
        "P45_ERROR_ANALYSIS.md": (
            "# P4.5 Error Analysis\n\n"
            "No human/P4 disagreements can be classified before human review.\n\n"
            "| Error ID | Case | Claim | Human | P4 | Root cause | Critical? | Severity |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        ),
        "P45_METRIC_REGISTRY.md": _metric_registry(),
        "P45_LOCKED_REGRESSIONS.md": "# P4.5 Locked Regressions\n\nValidation results are recorded after package generation.\n",
        "P45_TEST_RESULTS.md": "# P4.5 Test Results\n\nValidation results are recorded after package generation.\n",
        "P45_RUNTIME_INTEGRITY.md": (
            "# P4.5 Runtime Integrity\n\nExpected read-only baseline: Qdrant 639 chunks / 32 entities; "
            "Neo4j 32 nodes / 27 relationships; core Phase 1 completed_validated; semantic enrichment not_run.\n\n"
            "No ingestion, reindex, re-embedding, taxonomy mutation, semantic enrichment, or database reset is part of this harness.\n"
        ),
        "P45_GIT_INTEGRITY.md": (
            "# P4.5 Git Integrity\n\nEvaluation branch: `eval/v5-p45-shadow-calibration`. "
            "No direct main commit, squash, rebase, force push, or production behavior change is authorized.\n"
        ),
    }
    for filename, content in docs.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
    if metrics.get("status") != "P4_5_WAITING_FOR_HUMAN_REVIEW":
        raise AssertionError("P4.5 package must wait for real human review")


def _human_review_guide() -> str:
    return """# P4.5 Human Evidence-Support Review Guide

This is a **HUMAN EVIDENCE-SUPPORT REVIEW**, not clinical validation unless a qualified clinician performs it.

Review `p45_human_labels.jsonl`. Grade each claim only from the provided evidence, not from outside medical knowledge. The blind file intentionally hides P4 predictions.

## Verdicts

- `SUPPORTED`: evidence supports the claim as written.
- `PARTIALLY_SUPPORTED`: evidence supports only a weaker or partial version.
- `UNSUPPORTED`: supplied evidence does not establish the claim.
- `CONTRADICTED`: supplied evidence is inconsistent with the claim.
- `NO_EVIDENCE`: no appropriate source-backed evidence was supplied.
- `REVIEW_UNCERTAIN`: evidence is genuinely ambiguous; do not guess.

## Claim rows

For every `CLAIM_REVIEW` row, fill `human_verdict`, `human_criticality`, `mapping_correct`, `claim_extraction_correct`, `reviewer_confidence`, `reviewer_notes`, `reviewer_id`, and `reviewer_qualification`. Leave P4 fields hidden in this file.

## Case rows

For every `CASE_REVIEW` row, record omitted medically meaningful claims in `missing_claims`, omitted critical claims in `missing_critical_claims`, bad splitting in `incorrectly_split`, and non-claim fragments in `unrelated_fragments`.

Do not alter case IDs, claim IDs, questions, answers, evidence, provenance, or dataset hashes. Preserve uncertain labels for later adjudication. A second reviewer must not be fabricated.
"""


def _metric_registry() -> str:
    return """# P4.5 Metric Registry

Every rate reports numerator and denominator; zero denominators are `N/A`.

| Metric | Numerator | Denominator |
| --- | --- | --- |
| Supported precision | Human-supported claims predicted supported | All adjudicated P4-supported claims |
| Supported recall | Human-supported claims predicted supported | All adjudicated human-supported claims |
| Critical extraction recall | Extracted human-critical claims | Extracted plus human-identified missing critical claims |
| Critical false allow | Non-supported human-critical claims with `WOULD_ALLOW` | All adjudicated non-supported human-critical claims |
| Critical false block | Supported human-critical claims with `WOULD_BLOCK_CRITICAL` | All adjudicated supported human-critical claims |
| Mapping accuracy | Claims whose selected evidence is human-marked appropriate | All mapping-reviewed claims |
| Provenance validity | Evidence-bearing claims with valid provenance | All adjudicated evidence-bearing claims |

Readiness also requires at least 50 reviewed questions, 150 adjudicated claims, 30 adjudicated critical claims, all locked regressions, and no production behavior change.
"""


def _dataset_manifest_markdown(manifest: dict[str, Any]) -> str:
    hashes = "\n".join(f"- `{name}`: `{digest}`" for name, digest in manifest["sha256"].items())
    return f"""# P4.5 Dataset Manifest

- Dataset version: `{manifest['dataset_version']}`
- Timestamp: `{manifest['timestamp']}`
- Frozen: `{str(manifest['frozen']).lower()}`
- Questions: {manifest['question_count']}
- Successful outputs: {manifest['generated_output_count']}
- Generation failures: {manifest['generation_failure_count']}
- Claims: {manifest['claim_count']}
- Critical claims: {manifest['critical_claim_count']}
- Reviewer count: {manifest['reviewer_count']}
- Adjudication: `{manifest['adjudication_status']}`
- Generation fingerprint: `{manifest['generation_config_fingerprint']}`

## SHA256

{hashes}
"""


def _class_ratio(confusion: dict[str, dict[str, int]], label: str) -> float | None:
    return _ratio(confusion[label][label], sum(confusion[label].values()))


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise json.JSONDecodeError(f"line {line_number}: {exc.msg}", exc.doc, exc.pos) from exc
    return rows


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("validate", "generate", "package", "verify-package", "validate-labels", "score"),
    )
    parser.add_argument("--questions", type=Path, default=QUESTION_SET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()

    question_set = load_question_set(args.questions)
    if args.command == "validate":
        result = validate_question_set(question_set)
        _print_json(result)
        return 0 if result["passed"] else 1
    if args.output_dir is None:
        parser.error("--output-dir is required for this command")
    if args.command == "generate":
        generated = asyncio.run(
            generate_outputs(
                question_set=question_set,
                output_dir=args.output_dir,
                limit=args.limit,
                retry_failures=args.retry_failures,
            )
        )
        _print_json(
            {
                "records": len(generated.get("records", [])),
                "success": sum(item.get("status") == "SUCCESS" for item in generated.get("records", [])),
                "failures": sum(item.get("status") != "SUCCESS" for item in generated.get("records", [])),
            }
        )
        return 0
    if args.command == "package":
        manifest = build_review_package(question_set=question_set, output_dir=args.output_dir)
        _print_json(manifest)
        return 0

    package_validation = verify_frozen_package(args.output_dir)
    if args.command == "verify-package":
        _print_json(package_validation)
        return 0 if package_validation["passed"] else 1
    if not package_validation["passed"]:
        _print_json(package_validation)
        return 1

    prediction_rows = _read_jsonl(args.output_dir / "p45_review_with_predictions.jsonl")
    labels_path = args.labels or args.output_dir / "p45_human_labels.jsonl"
    validation = validate_human_labels(labels_path, prediction_rows)
    if args.command == "validate-labels":
        _print_json(validation)
        return 0 if validation["passed"] else 1
    if not validation["passed"]:
        _print_json(validation)
        return 1
    if not validation["complete"]:
        _print_json(
            {
                **validation,
                "status": "P4_5_WAITING_FOR_HUMAN_REVIEW",
                "message": "Complete every case and claim review before readiness scoring.",
            }
        )
        return 2
    human_rows = [HumanReviewRecord.model_validate(item) for item in _read_jsonl(labels_path)]
    rows = merge_human_labels_with_predictions(human_rows, prediction_rows)
    scored = score_human_labels(rows)
    _write_json_atomic(args.output_dir / "p45_metrics.json", scored)
    _write_json_atomic(args.output_dir / "p45_confusion_matrix.json", scored["confusion_matrix"])
    _print_json(scored)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
