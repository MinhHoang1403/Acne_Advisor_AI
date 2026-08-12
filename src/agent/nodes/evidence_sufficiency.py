"""LangGraph orchestration nodes for P3 evidence sufficiency."""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any

from src.agent.state import ClinicalState
from src.quality.safe_fallback import build_safe_fallback_answer
from src.retrieval.evidence_sufficiency import (
    EvidenceAbstentionType,
    EvidenceSufficiencyAssessment,
    EvidenceSufficiencyStatus,
    P3TraceEvent,
    P3TraceEventType,
    RetryEligibility,
    append_p3_event,
    assess_evidence_sufficiency,
    build_evidence_abstention,
    build_retry_plan,
    p3_enabled_from_env,
    p3_max_attempts_from_env,
)


async def assess_evidence_sufficiency_node(state: ClinicalState) -> dict[str, Any]:
    """Evaluate V5 selector coverage and evidence surviving the prompt packer."""

    if not _p3_active(state):
        return {
            "p3_active": False,
            "p3_trace": _append_event(
                state,
                P3TraceEventType.RETRY_SKIPPED,
                status="DISABLED_OR_V4_ROLLBACK",
                reason_code="P3_NOT_ACTIVE",
            ),
        }

    started = time.perf_counter()
    attempt = min(1, max(0, int(state.get("retrieval_attempt") or 0)))
    trace_id = _trace_id(state)
    result = assess_evidence_sufficiency(
        evidence_selector=state.get("evidence_selector"),
        evidence_packer=state.get("evidence_packer"),
        retrieval_status=state.get("retrieval_status"),
        is_in_domain=state.get("is_in_domain"),
        attempt_index=attempt,
        trace_id=trace_id,
    )
    final = result.final
    trace = state.get("p3_trace")
    if attempt == 1:
        trace = append_p3_event(
            trace,
            P3TraceEvent(
                event_type=P3TraceEventType.RETRY_COMPLETED,
                attempt_index=attempt,
                status=final.status.value,
                reason_code="RETRY_RETRIEVAL_COMPLETED",
                missing_roles=final.missing_roles,
                query_hash=_attempt_query_hash(state),
            ),
            trace_id=trace_id,
        )
    trace = append_p3_event(
        trace,
        P3TraceEvent(
            event_type=P3TraceEventType.SUFFICIENCY_ASSESSED,
            attempt_index=attempt,
            status=final.status.value,
            reason_code=final.reasons[0] if final.reasons else None,
            missing_roles=final.missing_roles,
            query_hash=_attempt_query_hash(state),
            elapsed_ms=final.elapsed_ms,
        ),
        trace_id=trace_id,
    )
    if (
        final.status == EvidenceSufficiencyStatus.SUFFICIENT
        or final.retry_eligibility == RetryEligibility.NON_RETRYABLE
    ):
        trace = append_p3_event(
            trace,
            P3TraceEvent(
                event_type=P3TraceEventType.RETRY_SKIPPED,
                attempt_index=attempt,
                status=final.status.value,
                reason_code=(
                    "EVIDENCE_SUFFICIENT"
                    if final.status == EvidenceSufficiencyStatus.SUFFICIENT
                    else "NON_RETRYABLE_OR_MAX_ATTEMPTS"
                ),
                missing_roles=final.missing_roles,
                query_hash=_attempt_query_hash(state),
            ),
            trace_id=trace_id,
        )
    history = [*(state.get("retry_history") or [])]
    history.append(
        {
            "attempt_index": attempt,
            "query_hash": _attempt_query_hash(state),
            "pre_pack_status": result.pre_pack.status.value,
            "post_pack_status": result.post_pack.status.value,
            "final_status": final.status.value,
            "missing_roles": list(final.missing_roles),
            "critical_missing_roles": list(final.critical_missing_roles),
            "selected_evidence_ids": list(result.pre_pack.evidence_ids),
            "packed_evidence_ids": list(result.post_pack.evidence_ids),
            "source_ids": list(final.source_ids),
        }
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "p3_active": True,
        "evidence_sufficiency_pre_pack": result.pre_pack.model_dump(mode="json"),
        "evidence_sufficiency_post_pack": result.post_pack.model_dump(mode="json"),
        "evidence_sufficiency": final.model_dump(mode="json"),
        "retry_history": history,
        "p3_trace": trace,
        "performance_timings": {
            **(state.get("performance_timings") or {}),
            f"p3_assessment_attempt_{attempt}": round(elapsed_ms, 3),
        },
    }


async def build_evidence_retry_plan_node(state: ClinicalState) -> dict[str, Any]:
    """Build and record the only P3 retry representation."""

    started = time.perf_counter()
    assessment = EvidenceSufficiencyAssessment.model_validate(state.get("evidence_sufficiency"))
    query = str(state.get("standalone_question") or state.get("normalized_question") or "")
    plan = build_retry_plan(
        original_query=query,
        assessment=assessment,
        retrieval_trace_v5=state.get("retrieval_trace_v5"),
    )
    trace_id = assessment.trace_id
    trace = append_p3_event(
        state.get("p3_trace"),
        P3TraceEvent(
            event_type=P3TraceEventType.RETRY_PLANNED,
            attempt_index=1,
            status=assessment.status.value,
            reason_code="QUERY_REPRESENTATION_REFINEMENT",
            missing_roles=assessment.missing_roles,
            query_hash=plan.retry_query_hash,
        ),
        trace_id=trace_id,
    )
    trace = append_p3_event(
        trace,
        P3TraceEvent(
            event_type=P3TraceEventType.RETRY_STARTED,
            attempt_index=1,
            status="STARTED",
            reason_code="BOUNDED_RETRY",
            missing_roles=assessment.missing_roles,
            query_hash=plan.retry_query_hash,
        ),
        trace_id=trace_id,
    )
    return {
        "retrieval_attempt": 1,
        "retry_plan": plan.model_dump(mode="json"),
        "p3_trace": trace,
        "performance_timings": {
            **(state.get("performance_timings") or {}),
            "p3_retry_planning": round((time.perf_counter() - started) * 1000, 3),
        },
    }


async def evidence_abstention_node(state: ClinicalState) -> dict[str, Any]:
    """Produce structured abstention state and a deterministic safe answer."""

    assessment = EvidenceSufficiencyAssessment.model_validate(state.get("evidence_sufficiency"))
    abstention = build_evidence_abstention(assessment)
    fallback_type = (
        "retrieval_error"
        if abstention.abstention_type == EvidenceAbstentionType.RETRIEVAL_PROVIDER_FAILURE
        else "insufficient_context"
    )
    answer = _safe_abstention_answer(abstention.abstention_type, fallback_type)
    trace = append_p3_event(
        state.get("p3_trace"),
        P3TraceEvent(
            event_type=P3TraceEventType.ABSTENTION_TRIGGERED,
            attempt_index=assessment.attempt_index,
            status=assessment.status.value,
            reason_code=abstention.abstention_type.value,
            missing_roles=assessment.missing_roles,
            query_hash=_attempt_query_hash(state),
        ),
        trace_id=assessment.trace_id,
    )
    return {
        "abstention": abstention.model_dump(mode="json"),
        "p3_trace": trace,
        "fallback_applied": True,
        "fallback_type": fallback_type,
        "fallback_reason": abstention.reason,
        "fallback_answer": answer,
        "fallback_cache_eligible": False,
    }


def route_after_evidence_sufficiency(state: ClinicalState) -> str:
    """Choose normal generation, the one retry, or deterministic abstention."""

    if not state.get("p3_active"):
        return "fallback_decision"
    value = state.get("evidence_sufficiency")
    if not isinstance(value, dict):
        return "evidence_abstention"
    assessment = EvidenceSufficiencyAssessment.model_validate(value)
    if assessment.status == EvidenceSufficiencyStatus.SUFFICIENT:
        return "fallback_decision"
    attempts_remaining = assessment.attempt_index + 1 < _p3_max_attempts(state)
    if assessment.retry_eligibility == RetryEligibility.RETRYABLE and attempts_remaining:
        return "build_retry_plan"
    return "evidence_abstention"


def _p3_active(state: ClinicalState) -> bool:
    manifest = state.get("pipeline_manifest") or {}
    pipeline = str(manifest.get("retrieval_pipeline_version") or "v5").strip().lower()
    configured = manifest.get("p3_evidence_sufficiency_enabled")
    if configured is None:
        enabled = p3_enabled_from_env()
    elif isinstance(configured, bool):
        enabled = configured
    else:
        enabled = str(configured).strip().lower() in {"1", "true", "yes", "on"}
    return enabled and pipeline == "v5"


def _p3_max_attempts(state: ClinicalState) -> int:
    manifest = state.get("pipeline_manifest") or {}
    try:
        configured = int(manifest.get("p3_max_retrieval_attempts"))
    except (TypeError, ValueError):
        configured = p3_max_attempts_from_env()
    return min(2, max(1, configured))


def _trace_id(state: ClinicalState) -> str:
    trace = state.get("retrieval_trace_v5") or {}
    existing = str(trace.get("trace_id") or "").strip() if isinstance(trace, dict) else ""
    return existing or str(uuid.uuid4())


def _attempt_query_hash(state: ClinicalState) -> str:
    if int(state.get("retrieval_attempt") or 0) == 1:
        plan = state.get("retry_plan") or {}
        known = str(plan.get("retry_query_hash") or "").strip() if isinstance(plan, dict) else ""
        if known:
            return known
    query = str(state.get("standalone_question") or state.get("normalized_question") or "")
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def _append_event(
    state: ClinicalState,
    event_type: P3TraceEventType,
    *,
    status: str,
    reason_code: str,
) -> dict[str, Any]:
    trace_id = _trace_id(state)
    return append_p3_event(
        state.get("p3_trace"),
        P3TraceEvent(
            event_type=event_type,
            attempt_index=min(1, max(0, int(state.get("retrieval_attempt") or 0))),
            status=status,
            reason_code=reason_code,
            query_hash=_attempt_query_hash(state),
        ),
        trace_id=trace_id,
    )


def _safe_abstention_answer(
    abstention_type: EvidenceAbstentionType,
    fallback_type: str,
) -> str:
    if abstention_type == EvidenceAbstentionType.CRITICAL_EVIDENCE_MISSING:
        return (
            "**Tóm tắt ngắn**\n"
            "Nguồn hiện được truy xuất chưa đủ bằng chứng an toàn thiết yếu để đưa ra "
            "khuyến nghị y khoa cho câu hỏi này.\n\n"
            "**Hướng xử lý an toàn**\n"
            "Mình sẽ không suy đoán khi thiếu bằng chứng về chống chỉ định hoặc nguy cơ "
            "quan trọng. Bạn nên trao đổi với bác sĩ hoặc dược sĩ, đặc biệt nếu câu hỏi "
            "liên quan mang thai, thuốc kê đơn hoặc triệu chứng nặng.\n\n"
            "**Lưu ý**\n"
            "Nếu có khó thở, sưng môi hoặc mặt, đau dữ dội, sốt cao hay triệu chứng "
            "tiến triển nhanh, hãy đi khám cấp cứu."
        )
    if abstention_type == EvidenceAbstentionType.SOURCE_PROVENANCE_FAILURE:
        return (
            "**Tóm tắt ngắn**\n"
            "Bằng chứng truy xuất được không có thông tin nguồn đủ tin cậy để mình trả lời "
            "câu hỏi này.\n\n"
            "**Bạn có thể làm gì tiếp theo**\n"
            "Vui lòng thử lại hoặc cung cấp rõ tên thuốc, hoạt chất và bối cảnh cần hỏi. "
            "Mình sẽ không dùng dữ liệu không truy vết được để đưa ra kết luận y khoa."
        )
    return build_safe_fallback_answer(fallback_type)


__all__ = [
    "assess_evidence_sufficiency_node",
    "build_evidence_retry_plan_node",
    "evidence_abstention_node",
    "route_after_evidence_sufficiency",
]
