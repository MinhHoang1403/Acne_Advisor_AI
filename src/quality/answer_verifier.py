"""Deterministic structural and provenance answer verification."""

from __future__ import annotations

from typing import Any

from src.agent.answer_formatting import assess_structural_quality, infer_response_profile
from src.quality.contracts import AnswerGuardResult, AnswerQualityIssue, AnswerVerificationReport
from src.retrieval.contracts import NormalizedQuery, PackedContext


ERROR = "error"
WARNING = "warning"


def verify_answer_quality(
    query: str,
    answer: str,
    normalized_query: NormalizedQuery | None = None,
    packed_context: PackedContext | None = None,
    retrieval_trace: Any | None = None,
) -> AnswerVerificationReport:
    """Check answer shape and evidence identity without judging medical truth."""

    issues = [
        _issue(
            str(item.get("code") or "presentation_contract_violation"),
            ERROR if item.get("severity") == "error" else WARNING,
            str(item.get("message") or "Answer violates a presentation contract."),
            item.get("evidence") if isinstance(item.get("evidence"), dict) else None,
        )
        for item in assess_structural_quality(
            answer or "",
            user_question=query,
            response_profile=infer_response_profile(query),
        )
    ]
    provenance_errors = _packed_context_provenance_errors(packed_context)
    issues.extend(provenance_errors)
    has_error = any(issue.severity == ERROR for issue in issues)
    return AnswerVerificationReport(
        passed=not has_error,
        original_query=query,
        intent=normalized_query.intent if normalized_query else None,
        checked_answer=answer or "",
        issues=issues,
        metadata={
            "verification_scope": ["presentation", "structural_contract", "provenance_identity"],
            "medical_semantic_verification": False,
            "packed_context_items": len(packed_context.items) if packed_context else 0,
            "retrieval_trace_available": retrieval_trace is not None,
            "answer_verifier": {"version": "answer_verifier_v3"},
        },
    )


def apply_answer_guard(
    query: str,
    answer: str,
    normalized_query: NormalizedQuery | None = None,
    packed_context: PackedContext | None = None,
    retrieval_trace: Any | None = None,
    mode: str = "metadata_only",
) -> AnswerGuardResult:
    """Return verifier metadata without replacing ordinary medical content."""

    report = verify_answer_quality(
        query=query,
        answer=answer,
        normalized_query=normalized_query,
        packed_context=packed_context,
        retrieval_trace=retrieval_trace,
    )
    normalized_mode = (mode or "metadata_only").strip().lower()
    return AnswerGuardResult(
        answer=answer,
        original_answer=answer,
        report=report,
        modified=False,
        modification_reason=(
            "unsupported_guard_mode_preserved_answer"
            if normalized_mode not in {"metadata_only", "append_warnings", "strict_safe"}
            else None
        ),
    )


def _packed_context_provenance_errors(packed_context: PackedContext | None) -> list[AnswerQualityIssue]:
    if packed_context is None:
        return []
    issues: list[AnswerQualityIssue] = []
    for item in packed_context.items:
        source_id = str(
            item.payload.get("source_id")
            or item.payload.get("source_path")
            or item.payload.get("source_file")
            or item.payload.get("document_id")
            or ""
        ).strip()
        if not item.item_id or not item.text.strip() or not source_id:
            issues.append(
                _issue(
                    "packed_evidence_missing_identity",
                    ERROR,
                    "Packed evidence is missing text, item identity, or source identity.",
                    {"item_id": item.item_id, "has_text": bool(item.text.strip()), "has_source_id": bool(source_id)},
                )
            )
    return issues


def _issue(
    code: str,
    severity: str,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> AnswerQualityIssue:
    return AnswerQualityIssue(
        code=code,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        evidence=evidence or {},
        suggested_fix=None,
    )


__all__ = ["apply_answer_guard", "verify_answer_quality"]
