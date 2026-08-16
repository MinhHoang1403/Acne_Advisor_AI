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
ANSWER_VALIDATION_VERSION = "structural_provenance_locality_validation_v2"


def verify_answer_quality(
    query: str,
    answer: str,
    packed_context: PackedContext | None = None,
    retrieval_trace: Any | None = None,
    final_source_ids: list[str] | None = None,
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
            "answer_validation": {"version": ANSWER_VALIDATION_VERSION},
            "evidence_locality": _evidence_locality_diagnostics(
                packed_context,
                retrieval_trace,
                final_source_ids or [],
            ),
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


def _evidence_locality_diagnostics(
    packed_context: PackedContext | None,
    retrieval_trace: Any | None,
    final_source_ids: list[str],
) -> dict[str, Any]:
    """Expose scope labels per item without inferring claim entailment."""

    trace = retrieval_trace if isinstance(retrieval_trace, dict) else {}
    items: list[dict[str, Any]] = []
    for item in packed_context.items if packed_context else []:
        payload = item.payload
        section_path = payload.get("section_path")
        section = payload.get("header")
        if not section and isinstance(section_path, list) and section_path:
            section = section_path[-1]
        source_id = str(
            payload.get("source_id")
            or payload.get("source_path")
            or payload.get("source_file")
            or payload.get("document_id")
            or ""
        ).strip()
        scope_parts = [
            str(value).strip()
            for value in (
                payload.get("drug_product"),
                payload.get("active_ingredient"),
                payload.get("drug_class"),
                section,
            )
            if str(value or "").strip()
        ]
        items.append(
            {
                "item_id": item.item_id,
                "source_id": source_id,
                "publisher": payload.get("publisher") or payload.get("authority"),
                "section": section,
                "drug_product": payload.get("drug_product"),
                "active_ingredient": payload.get("active_ingredient"),
                "drug_class": payload.get("drug_class"),
                "scope": " | ".join(scope_parts) or "unscoped",
            }
        )
    return {
        "diagnostic_only": True,
        "semantic_entailment_checked": False,
        "retrieval_attempt": trace.get("attempt") or trace.get("retrieval_attempt"),
        "selected_ids": list(trace.get("selected_ids") or []),
        "final_cited_source_ids": list(dict.fromkeys(final_source_ids)),
        "items": items,
    }


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


__all__ = ["ANSWER_VALIDATION_VERSION", "verify_answer_quality"]
