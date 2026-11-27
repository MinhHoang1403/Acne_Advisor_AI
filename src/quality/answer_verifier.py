"""Kiểm tra deterministic về presentation, cấu trúc và provenance identity.

Verifier phát hiện lỗi format và packed evidence thiếu ``item_id``/text/source.
Nó không thực hiện entailment, không so claim với nguồn và không xác minh clinical
truth. Vì vậy ``passed=True`` chỉ nói answer đạt các contract kỹ thuật được liệt
kê trong report, không phải chứng nhận nội dung đúng y khoa.
"""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Any

from src.agent.answer_formatting import assess_structural_quality, infer_response_profile
from src.agent.requested_structure import RequestShape
from src.quality.contracts import AnswerQualityIssue, AnswerVerificationReport
from src.knowledge.normalizer import DrugEntityNormalizer
from src.retrieval.contracts import PackedContext


ERROR = "error"
WARNING = "warning"
ANSWER_VALIDATION_VERSION = "structural_provenance_requested_entity_scope"


def verify_answer_quality(
    query: str,
    answer: str,
    packed_context: PackedContext | None = None,
    retrieval_trace: Any | None = None,
    final_source_ids: list[str] | None = None,
    request_shape: RequestShape | dict[str, Any] | None = None,
    *,
    response_profile: str | None = None,
    fallback_applied: bool = False,
) -> AnswerVerificationReport:
    """Kiểm tra shape và evidence identity mà không phán xét medical truth."""

    effective_profile = response_profile or infer_response_profile(
        query,
        fallback_type="no_retrieval_evidence" if fallback_applied else None,
        request_shape=request_shape,
    )
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
            response_profile=effective_profile,
            request_shape=None if fallback_applied else request_shape,
        )
    ]
    if not fallback_applied:
        provenance_errors = _packed_context_provenance_errors(packed_context)
        issues.extend(provenance_errors)
        issues.extend(_requested_entity_scope_errors(query, answer or ""))
    has_error = any(issue.severity == ERROR for issue in issues)
    return AnswerVerificationReport(
        passed=not has_error,
        original_query=query,
        checked_answer=answer or "",
        issues=issues,
        metadata={
            "verification_scope": [
                "presentation",
                "structural_contract",
                "provenance_identity",
                "requested_entity_scope",
            ],
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


_TOPIC_RESET_RE = re.compile(
    r"^\s*(?:bỏ\s+qua|bo\s+qua|ignore)\b[^.!?;\n]*[.!?;\n]+\s*(?P<question>.+)$",
    flags=re.IGNORECASE | re.DOTALL,
)


def _effective_scope_query(query: str) -> str:
    """Bỏ tiền tố topic-reset để không phạt nhầm thực thể người dùng yêu cầu bỏ qua."""

    match = _TOPIC_RESET_RE.match(query or "")
    if not match:
        return query
    remainder = " ".join(match.group("question").split())
    return remainder or query


def _requested_entity_scope_errors(query: str, answer: str) -> list[AnswerQualityIssue]:
    """Catch explicit entity omissions without claiming semantic entailment."""

    effective_query = _effective_scope_query(query)
    requested = _matched_entity_names(effective_query)
    addressed = _matched_entity_names(answer)
    satisfied_keys = set(addressed)
    # Nếu câu hỏi nêu tên biệt dược (DrugProduct) còn câu trả lời thảo luận trực tiếp
    # toàn bộ hoạt chất chuẩn (active_ingredients) cấu thành biệt dược đó, coi thực thể đã được đáp ứng.
    for card in _scope_normalizer().match_alias(effective_query):
        if card.entity_type == "drug_product" and card.active_ingredients:
            ing_keys = {
                f"active_ingredient:{ing.casefold()}"
                for ing in card.active_ingredients
                if ing
            }
            if ing_keys and ing_keys.issubset(satisfied_keys):
                satisfied_keys.add(f"{card.entity_type}:{card.canonical_name.casefold()}")

    missing = sorted(set(requested) - satisfied_keys)
    if len(requested) >= 2 and missing:
        return [
            _issue(
                "requested_entity_scope_incomplete",
                ERROR,
                "Answer omits one or more explicitly requested entities.",
                {"requested_entities": sorted(requested), "missing_entities": missing},
            )
        ]
    if len(requested) == 1 and missing and addressed:
        return [
            _issue(
                "answer_entity_off_scope",
                ERROR,
                "Answer discusses a different known entity without addressing the requested one.",
                {
                    "requested_entities": sorted(requested),
                    "answer_entities": sorted(addressed),
                },
            )
        ]
    return []


def _matched_entity_names(text: str) -> dict[str, str]:
    matches: dict[str, str] = {}
    for card in _scope_normalizer().match_alias(text):
        # Loại trừ thực thể gốc bao trùm toàn miền (acne_vulgaris) khỏi phép kiểm lệch phạm vi
        # thực thể cụ thể để tránh phạt sai khi câu trả lời mô tả nhân mụn / biệt dược chuẩn xác.
        if card.entity_type == "condition" and card.canonical_name.casefold() == "acne_vulgaris":
            continue
        key = f"{card.entity_type}:{card.canonical_name.casefold()}"
        matches[key] = card.canonical_name
    return matches


@lru_cache(maxsize=1)
def _scope_normalizer() -> DrugEntityNormalizer:
    return DrugEntityNormalizer()


def _packed_context_provenance_errors(
    packed_context: PackedContext | None,
) -> list[AnswerQualityIssue]:
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
                    {
                        "item_id": item.item_id,
                        "has_text": bool(item.text.strip()),
                        "has_source_id": bool(source_id),
                    },
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
        "retrieval_attempt": (
            trace.get("attempt")
            or trace.get("retrieval_attempt")
            or (
                trace.get("retry_evidence", {}).get("retrieval_attempt")
                if isinstance(trace.get("retry_evidence"), dict)
                else None
            )
        ),
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
