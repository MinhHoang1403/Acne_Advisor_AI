"""Tạo và tùy chọn ghi trace đã giới hạn, loại bỏ dữ liệu giống secret.

Raw query không được đưa vào summary; event chỉ giữ số ký tự và SHA-256 prefix để
correlate cùng input trong phạm vi quan sát. Export là best-effort JSONL và tắt
mặc định, nên lỗi ghi log không được làm hỏng response path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.observability.contracts import (
    AgentDecisionTelemetry,
    ObservabilityEvent,
    PipelineTraceSummary,
    StageTelemetry,
)
from src.observability.error_taxonomy import classify_error
from src.observability.langfuse_sink import LangfuseRequestHandle, export_event_to_langfuse
from src.observability.versioning import (
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
    pipeline_manifest_summary,
)
from src.quality.safe_fallback import sanitize_fallback_reason

logger = logging.getLogger(__name__)

SECRET_KEY_MARKERS = (
    "api_key",
    "token",
    "password",
    "secret",
    "authorization",
    "bearer",
    "cookie",
)

# Operational payload bound only; this is not a retrieval relevance threshold.
MAX_OBSERVABILITY_CANDIDATE_IDS = 20


def sanitize_for_observability(data: Any, max_text_chars: int = 500) -> Any:
    """Redact key giống secret và giới hạn text trước khi đưa vào telemetry."""

    if isinstance(data, dict):
        output: dict[str, Any] = {}
        for key, value in data.items():
            key_text = str(key)
            if any(marker in key_text.casefold() for marker in SECRET_KEY_MARKERS):
                output[key_text] = "[REDACTED]"
            else:
                output[key_text] = sanitize_for_observability(value, max_text_chars)
        return output
    if isinstance(data, (list, tuple)):
        return [sanitize_for_observability(item, max_text_chars) for item in data]
    if isinstance(data, str):
        safe = sanitize_fallback_reason(data, max_chars=max(len(data), max_text_chars))
        if len(safe) <= max_text_chars:
            return safe
        omitted = len(safe) - max_text_chars
        return f"{safe[:max_text_chars]}...[truncated {omitted} chars]"
    if isinstance(data, (int, float, bool)) or data is None:
        return data
    return sanitize_for_observability(str(data), max_text_chars)


def build_observability_event(
    *,
    query: str,
    state: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    event_type: str = "phase2_chat_trace",
    pipeline_manifest: dict[str, Any] | None = None,
    pipeline_fingerprint: str | None = None,
    safe_payload: dict[str, Any] | None = None,
    max_text_chars: int | None = None,
    complete: bool = False,
    error: BaseException | str | None = None,
    error_stage: str | None = None,
) -> ObservabilityEvent:
    """Build one safe event for a request identity created by the caller."""

    state = state or {}
    result = result or {}
    max_chars = max_text_chars or int(os.getenv("OBSERVABILITY_MAX_TEXT_CHARS", "500") or 500)
    manifest = (
        pipeline_manifest
        or _as_dict(result.get("pipeline_manifest") or state.get("pipeline_manifest"))
        or build_pipeline_version_manifest()
    )
    fingerprint_value = (
        pipeline_fingerprint
        or result.get("pipeline_fingerprint")
        or state.get("pipeline_fingerprint")
    )
    fingerprint = str(fingerprint_value) if fingerprint_value else compute_pipeline_fingerprint(manifest)

    canonical_id = str(request_id or result.get("request_id") or state.get("request_id") or "")
    if not canonical_id:
        raise ValueError("request_id must be created before building observability telemetry")

    retrieval = _as_dict(result.get("retrieval_trace") or state.get("retrieval_trace"))
    packed = _as_dict(result.get("packed_context") or state.get("packed_context"))
    assessment = _as_dict(result.get("evidence_assessment") or state.get("evidence_assessment"))
    quality = _as_dict(result.get("answer_quality_report") or state.get("answer_quality_report"))
    issues = quality.get("issues", []) if isinstance(quality.get("issues"), list) else []
    channels = _as_dict(retrieval.get("channels"))
    selected_ids = retrieval.get("selected_ids", [])
    selected_count = len(selected_ids) if isinstance(selected_ids, list) else 0
    warnings = retrieval.get("warnings", [])
    warnings_count = len(warnings) if isinstance(warnings, list) else 0
    warnings_count += sum(
        1 for issue in issues if isinstance(issue, dict) and issue.get("severity") == "warning"
    )

    timings = _float_dict(result.get("performance_timings") or state.get("performance_timings"))
    if retrieval.get("elapsed_ms") is not None:
        timings.setdefault("retrieval_total", float(retrieval["elapsed_ms"]))

    stages = _build_stage_telemetry(state, result, retrieval, quality, timings)
    err_family, err_owner = (None, None)
    if error:
        err_family, err_owner = classify_error(error, error_stage)
    request_status = _request_status(stages, fatal_error=error is not None)
    stages.append(
        StageTelemetry(
            component="api",
            stage="request",
            status=request_status,
            duration_ms=timings.get("total_request"),
            error_family=err_family,
            error_owner=err_owner,
            error_type=_safe_error_type(error),
        )
    )

    decision = _as_dict(result.get("agent_decision") or state.get("agent_decision"))

    summary = PipelineTraceSummary(
        query=_safe_query_summary(query),
        action=(
            result.get("next_action")
            or state.get("next_action")
            or decision.get("action")
        ),
        retrieval_candidates_count=sum(
            int(_as_dict(channels.get(name)).get("count") or 0) for name in ("dense", "bm25")
        ),
        packed_context_items_count=selected_count or len(packed.get("items", []) or []),
        retrieval_attempts=int(result.get("retrieval_attempt", state.get("retrieval_attempt", 0)) or 0),
        evidence_usable=assessment.get("usable") if assessment else None,
        answer_quality_passed=quality.get("passed") if quality else None,
        critical_issues_count=sum(
            1 for issue in issues if isinstance(issue, dict) and issue.get("severity") == "critical"
        ),
        warnings_count=warnings_count,
        cache_hit=result.get("cache_hit", state.get("cache_hit")),
        pipeline_fingerprint=fingerprint,
        knowledge_build_id=str(manifest.get("kb_version") or "") or None,
        timings_ms=timings,
        stages=stages,
        decisions=_safe_agent_decisions(
            result.get("agent_decision_history", state.get("agent_decision_history", [])),
            decision,
        ),
        status=request_status,
        error_family=err_family,
        error_owner=err_owner,
        complete=complete,
        metadata=sanitize_for_observability(
            {
                "retrieval_status": result.get("retrieval_status", state.get("retrieval_status")),
                "reranker": _as_dict(retrieval.get("reranker")),
                "evidence_assessment": assessment,
                "fallback_applied": result.get("fallback_applied", state.get("fallback_applied")),
                "fallback_type": result.get("fallback_type", state.get("fallback_type")),
                "fallback_reason_code": result.get(
                    "fallback_reason_code", state.get("fallback_reason_code")
                ),
                "safety_severity": result.get("safety_severity", state.get("safety_severity")),
                "pipeline_manifest": pipeline_manifest_summary(manifest),
                "runtime_resilience": result.get("runtime_resilience", state.get("runtime_resilience")),
            },
            max_chars,
        ),
    )
    payload = safe_payload if safe_payload is not None else {
        "sources": result.get("sources", state.get("sources", [])),
        "retrieval_status": result.get("retrieval_status", state.get("retrieval_status")),
        "retrieval_attempts": _safe_retry_attempts(
            result.get("retry_history", state.get("retry_history", []))
        ),
        "fallback_applied": result.get("fallback_applied", state.get("fallback_applied")),
        "fallback_type": result.get("fallback_type", state.get("fallback_type")),
        "fallback_reason_code": result.get(
            "fallback_reason_code", state.get("fallback_reason_code")
        ),
        "evidence_locality": _as_dict(
            quality.get("metadata") if isinstance(quality, dict) else {}
        ).get("evidence_locality"),
        "quality_issues": _safe_quality_issues(issues),
    }
    return ObservabilityEvent(
        event_type=event_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
        request_id=canonical_id,
        session_id=session_id,
        query_hash=hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:16],
        summary=summary,
        safe_payload=sanitize_for_observability(payload, max_chars),
    )


def export_observability_event(
    event: ObservabilityEvent,
    output_dir: str | Path = "logs/phase2_traces",
    *,
    enabled: bool | None = None,
) -> bool:
    """Append một event JSONL chỉ khi observability được bật rõ ràng."""

    if enabled is None:
        enabled = os.getenv("OBSERVABILITY_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return False
    try:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"phase2_traces-{datetime.now(timezone.utc):%Y%m%d}.jsonl"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
        return True
    except Exception as exc:  # pragma: no cover - fail-open observability
        logger.warning(
            "Observability sink failed safely: error_family=observability "
            "owner=observability error_type=%s",
            exc.__class__.__name__,
        )
        return False


def emit_request_completion(
    *,
    query: str,
    request_id: str,
    result: dict[str, Any] | None = None,
    session_id: str | None = None,
    error: BaseException | str | None = None,
    error_stage: str | None = None,
    enabled: bool | None = None,
    langfuse_handle: LangfuseRequestHandle | None = None,
) -> bool:
    """Build and export the request-complete event without affecting chat."""

    if enabled is None:
        enabled = os.getenv("OBSERVABILITY_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if not enabled:
        if langfuse_handle is not None:
            langfuse_handle.close()
        return False
    try:
        event = build_observability_event(
            query=query,
            result=result,
            session_id=session_id,
            request_id=request_id,
            event_type="chat_request_completed",
            complete=True,
            error=error,
            error_stage=error_stage,
        )
        file_exported = export_observability_event(
            event,
            output_dir=os.getenv("OBSERVABILITY_TRACE_DIR", "logs/phase2_traces"),
            enabled=True,
        )
        langfuse_exported = export_event_to_langfuse(event, handle=langfuse_handle)
        return file_exported or langfuse_exported
    except Exception as exc:  # fail-open includes serialization/contract failures
        if langfuse_handle is not None:
            langfuse_handle.close()
        logger.warning(
            "Observability event construction failed safely: error_family=observability "
            "owner=observability error_type=%s",
            exc.__class__.__name__,
        )
        return False


def _build_stage_telemetry(
    state: dict[str, Any],
    result: dict[str, Any],
    retrieval: dict[str, Any],
    quality: dict[str, Any],
    timings: dict[str, float],
) -> list[StageTelemetry]:
    """Normalize existing diagnostics without duplicating their raw payloads."""

    stages: list[StageTelemetry] = []
    if "prepare" in timings:
        stages.append(_stage("agent/preparation", "prepare", "success", timings["prepare"]))
    if "guard" in timings or result.get("safety_decision") or state.get("safety_decision"):
        stages.append(_stage("agent/safety_cache", "guard", "success", timings.get("guard")))

    cache_checked = result.get("cache_checked", state.get("cache_checked"))
    cache_reason = result.get("cache_reason", state.get("cache_reason"))
    if cache_checked is not None or cache_reason:
        skipped_reasons = {
            "bypassed",
            "history_present",
            "input_not_cacheable",
            "out_of_domain",
            "safe_fallback_no_retrieval_evidence",
            "safety_override",
            "skipped",
        }
        stages.append(
            _stage(
                "cache/exact_cache",
                "cache",
                "skipped" if cache_checked is False or cache_reason in skipped_reasons else "success",
            )
        )

    decision = _as_dict(result.get("agent_decision") or state.get("agent_decision"))
    decision_durations = [
        value for key, value in timings.items() if key.startswith("agent_decision_")
    ]
    if decision or decision_durations:
        stages.append(
            _stage(
                "agent/decision",
                "decide",
                "success",
                sum(decision_durations) if decision_durations else None,
                attempt=len(decision_durations) or None,
                fallback=bool(decision.get("fallback_used")),
                provider=decision.get("provider"),
                model=decision.get("model"),
            )
        )

    r_status = result.get("retrieval_status") or state.get("retrieval_status")
    channels = _as_dict(retrieval.get("channels"))
    if channels:
        for channel_name in ("dense", "bm25"):
            channel = _as_dict(channels.get(channel_name))
            error_type = _safe_error_type(channel.get("error"))
            family, owner = (None, None)
            if error_type:
                family, owner = classify_error(error_type, channel_name)
            stages.append(
                _stage(
                    "retrieval",
                    channel_name,
                    "failed" if error_type else "success",
                    error_family=family,
                    error_owner=owner,
                    error_type=error_type,
                    candidate_count=int(channel.get("count") or 0),
                    candidate_ids=_candidate_ids(retrieval, channel_name),
                )
            )
        both_failed = all(stage.status == "failed" for stage in stages[-2:])
        stages.append(
            _stage(
                "retrieval/fusion",
                "rrf",
                "skipped" if both_failed else "success",
                candidate_count=int(retrieval.get("fused_candidate_count") or 0),
                candidate_ids=_candidate_ids(retrieval, "fused"),
            )
        )
        reranker_info = _as_dict(retrieval.get("reranker"))
        if reranker_info.get("enabled"):
            rerank_fallback = bool(reranker_info.get("fallback_used"))
            family, owner = (None, None)
            if rerank_fallback:
                family, owner = classify_error("RerankerFallback", "rerank")
            stages.append(
                _stage(
                    "retrieval/reranker",
                    "rerank",
                    "degraded" if rerank_fallback else "success",
                    _optional_float(reranker_info.get("elapsed_ms")),
                    error_family=family,
                    error_owner=owner,
                    error_type=(
                        _safe_error_type(reranker_info.get("fallback_reason"))
                        if rerank_fallback
                        else None
                    ),
                    fallback=rerank_fallback,
                    model=reranker_info.get("model"),
                    candidate_count=int(retrieval.get("eligible_candidate_count") or 0),
                    candidate_ids=_candidate_ids(retrieval, "fused"),
                )
            )
        else:
            stages.append(_stage("retrieval/reranker", "rerank", "skipped"))
        stages.append(
            _stage(
                "retrieval/context_packer",
                "pack",
                "skipped" if both_failed else "success",
                candidate_count=len(retrieval.get("selected_ids") or []),
                candidate_ids=[
                    str(item)
                    for item in (retrieval.get("selected_ids") or [])[
                        :MAX_OBSERVABILITY_CANDIDATE_IDS
                    ]
                ],
            )
        )

    if r_status:
        retrieval_status = _retrieval_stage_status(str(r_status))
        retry_evidence = _as_dict(retrieval.get("retry_evidence"))
        retained_ids = retry_evidence.get("retained_candidate_ids")
        duplicate_ids = retry_evidence.get("duplicate_candidate_ids")
        error_type = _safe_error_type(
            result.get("retrieval_error") or state.get("retrieval_error")
        )
        family, owner = (None, None)
        if retrieval_status == "failed":
            family, owner = classify_error(error_type or "RetrievalError", "retrieve")
        stages.append(
            _stage(
                "retrieval",
                "retrieve",
                retrieval_status,
                timings.get("retrieval_total"),
                attempt=int(result.get("retrieval_attempt", state.get("retrieval_attempt", 0)) or 0)
                or None,
                error_family=family,
                error_owner=owner,
                error_type=error_type,
                candidate_count=len(retrieval.get("selected_ids") or []),
                retained_candidate_count=(
                    len(retained_ids) if isinstance(retained_ids, list) else None
                ),
                duplicate_candidate_count=(
                    len(duplicate_ids) if isinstance(duplicate_ids, list) else None
                ),
                candidate_ids=[
                    str(item)
                    for item in (retrieval.get("selected_ids") or [])[
                        :MAX_OBSERVABILITY_CANDIDATE_IDS
                    ]
                ],
            )
        )

    assessment = _as_dict(result.get("evidence_assessment") or state.get("evidence_assessment"))
    if assessment:
        stages.append(_stage("agent/evidence", "assess", "success"))

    action = result.get("next_action") or state.get("next_action") or decision.get("action")
    if action == "abstain":
        stages.append(_stage("agent", "abstain", "success"))
        stages.append(_stage("generation", "generate", "skipped"))
    elif result.get("generation_invoked") or "llm_generation" in timings:
        fallback_used = bool(result.get("llm_fallback_used", state.get("llm_fallback_used")))
        stages.append(
            _stage(
                "generation",
                "generate",
                "degraded" if fallback_used else "success",
                timings.get("llm_generation"),
                fallback=fallback_used,
                provider=result.get("generation_provider") or result.get("actual_provider"),
                model=result.get("generation_model") or result.get("actual_model"),
            )
        )
        stages.append(_stage("agent", "abstain", "skipped"))

    if "finalize" in timings or result.get("answer"):
        source_validation = _as_dict(
            result.get("source_validation") or state.get("source_validation")
        )
        if source_validation:
            invalid_count = int(source_validation.get("invalid_source_name_count") or 0)
            family, owner = (None, None)
            if invalid_count:
                family, owner = classify_error("SourceValidationIssue", "source_validation")
            stages.append(
                _stage(
                    "agent/source_presentation",
                    "source_validation",
                    "degraded" if invalid_count else "success",
                    timings.get("source_validation"),
                    error_family=family,
                    error_owner=owner,
                )
            )
        if quality:
            quality_passed = quality.get("passed") is not False
            family, owner = (None, None)
            error_type = None
            if not quality_passed:
                family, owner = classify_error("QualityValidationIssue", "quality")
                issues = quality.get("issues") if isinstance(quality.get("issues"), list) else []
                if issues and isinstance(issues[0], dict):
                    error_type = _safe_error_type(issues[0].get("code"))
            stages.append(
                _stage(
                    "quality/answer_verifier",
                    "quality",
                    "success" if quality_passed else "degraded",
                    error_family=family,
                    error_owner=owner,
                    error_type=error_type,
                )
            )
        stages.append(_stage("agent", "finalize", "success", timings.get("finalize")))

    if "persistence" in timings:
        persistence_error = _safe_error_type(result.get("persistence_error_type"))
        family, owner = (None, None)
        if persistence_error:
            family, owner = classify_error(persistence_error, "persistence")
        stages.append(
            _stage(
                "database/api",
                "persistence",
                (
                    "skipped"
                    if result.get("persistence_skipped")
                    else "failed" if persistence_error else "success"
                ),
                timings["persistence"],
                error_family=family,
                error_owner=owner,
                error_type=persistence_error,
            )
        )

    return stages


def _stage(
    component: str,
    stage: str,
    status: str,
    duration_ms: float | None = None,
    **kwargs: Any,
) -> StageTelemetry:
    return StageTelemetry(
        component=component,
        stage=stage,
        status=status,
        duration_ms=duration_ms,
        **kwargs,
    )


def _retrieval_stage_status(status: str) -> str:
    if status == "not_started":
        return "skipped"
    if status.startswith("degraded"):
        return "degraded"
    if status in {"failed", "recoverable_error"}:
        return "failed"
    return "success"


def _request_status(stages: list[StageTelemetry], *, fatal_error: bool) -> str:
    if fatal_error:
        return "failed"
    if any(stage.status in {"degraded", "failed"} for stage in stages):
        return "degraded"
    return "success"


def _candidate_ids(retrieval: dict[str, Any], channel: str) -> list[str]:
    trace = _as_dict(retrieval.get("candidate_trace"))
    values = trace.get(channel)
    if not isinstance(values, list):
        return []
    return [
        str(item.get("candidate_id"))
        for item in values[:MAX_OBSERVABILITY_CANDIDATE_IDS]
        if isinstance(item, dict) and item.get("candidate_id")
    ]


def _safe_retry_attempts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        {
            "attempt": item.get("attempt"),
            "status": item.get("status"),
            "selected_ids": [
                str(candidate_id)
                for candidate_id in (item.get("selected_ids") or [])[
                    :MAX_OBSERVABILITY_CANDIDATE_IDS
                ]
            ],
        }
        for item in value
        if isinstance(item, dict)
    ]


def _safe_quality_issues(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        {"code": item.get("code"), "severity": item.get("severity")}
        for item in value
        if isinstance(item, dict)
    ]


def _safe_agent_decisions(
    value: Any,
    final_decision: dict[str, Any],
) -> list[AgentDecisionTelemetry]:
    """Project only the validated action schema; never copy reasoning/query fields."""

    source = value if isinstance(value, list) and value else [final_decision]
    decisions: list[AgentDecisionTelemetry] = []
    allowed_actions = {"retrieve", "retry", "generate", "abstain", "finalize"}
    for index, item in enumerate(source, start=1):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "")
        if action not in allowed_actions:
            continue
        reason_code = item.get("reason_code")
        decisions.append(
            AgentDecisionTelemetry(
                attempt=index,
                action=action,
                reason_code=_safe_categorical(reason_code),
            )
        )
    return decisions


def _safe_categorical(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text and len(text) <= 80 and all(char.isalnum() or char in "._-" for char in text):
        return text
    return None


def _safe_error_type(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, BaseException):
        return value.__class__.__name__
    text = str(value).strip()
    if text and len(text) <= 80 and all(char.isalnum() or char in "._-" for char in text):
        return text
    return "OperationalError"


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {}


def _float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, float] = {}
    for key, item in value.items():
        try:
            output[str(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return output


def _safe_query_summary(query: str) -> str:
    clean = " ".join((query or "").split())
    return f"[REDACTED_QUERY chars={len(clean)}]"


__all__ = [
    "MAX_OBSERVABILITY_CANDIDATE_IDS",
    "build_observability_event",
    "emit_request_completion",
    "export_observability_event",
    "sanitize_for_observability",
]
