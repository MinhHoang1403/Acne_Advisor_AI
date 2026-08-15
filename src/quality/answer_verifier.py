"""Kiểm tra deterministic về presentation, cấu trúc và provenance identity.

Verifier phát hiện lỗi format và packed evidence thiếu ``item_id``/text/source.
Nó không thực hiện entailment, không so claim với nguồn và không xác minh clinical
truth. Vì vậy ``passed=True`` chỉ nói answer đạt các contract kỹ thuật được liệt
kê trong report, không phải chứng nhận nội dung đúng y khoa.
"""

from __future__ import annotations

from typing import Any

from src.agent.answer_formatting import assess_structural_quality, infer_response_profile
from src.quality.contracts import AnswerQualityIssue, AnswerVerificationReport
from src.retrieval.contracts import PackedContext


ERROR = "error"
WARNING = "warning"


def verify_answer_quality(
    query: str,
    answer: str,
    packed_context: PackedContext | None = None,
    retrieval_trace: Any | None = None,
) -> AnswerVerificationReport:
    """Kiểm tra shape và evidence identity mà không phán xét medical truth."""

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
        checked_answer=answer or "",
        issues=issues,
        metadata={
            "verification_scope": ["presentation", "structural_contract", "provenance_identity"],
            "medical_semantic_verification": False,
            "packed_context_items": len(packed_context.items) if packed_context else 0,
            "retrieval_trace_available": retrieval_trace is not None,
            "answer_validation": {"version": "structural_provenance_validation_v1"},
        },
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


__all__ = ["verify_answer_quality"]
