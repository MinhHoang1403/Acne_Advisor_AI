"""Safe Phase 2 observability trace sanitization and JSONL export."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.observability.contracts import ObservabilityEvent, PipelineTraceSummary
from src.observability.versioning import (
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
    pipeline_manifest_summary,
)
from src.quality.safe_fallback import sanitize_fallback_reason
from src.quality.claim_grounding import compact_claim_grounding

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


def sanitize_for_observability(data: Any, max_text_chars: int = 500) -> Any:
    """Redact secrets and truncate long text while preserving JSON shape."""

    if isinstance(data, dict):
        sanitized: dict[str, Any] = {}
        for key, value in data.items():
            key_text = str(key)
            key_lower = key_text.lower()
            if any(marker in key_lower for marker in SECRET_KEY_MARKERS):
                sanitized[key_text] = "[REDACTED]"
                continue
            sanitized[key_text] = sanitize_for_observability(value, max_text_chars=max_text_chars)
        return sanitized

    if isinstance(data, list):
        return [sanitize_for_observability(item, max_text_chars=max_text_chars) for item in data]

    if isinstance(data, tuple):
        return [sanitize_for_observability(item, max_text_chars=max_text_chars) for item in data]

    if isinstance(data, str):
        data = sanitize_fallback_reason(data, max_chars=max(len(data), max_text_chars))
        if len(data) > max_text_chars:
            return data[:max_text_chars] + f"...[truncated {len(data) - max_text_chars} chars]"
        return data

    if isinstance(data, (int, float, bool)) or data is None:
        return data

    return sanitize_for_observability(str(data), max_text_chars=max_text_chars)


def build_observability_event(
    *,
    query: str,
    state: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    session_id: str | None = None,
    event_type: str = "phase2_chat_trace",
    pipeline_manifest: dict[str, Any] | None = None,
    pipeline_fingerprint: str | None = None,
    safe_payload: dict[str, Any] | None = None,
    max_text_chars: int | None = None,
) -> ObservabilityEvent:
    """Build a safe observability event from runtime state/result dictionaries."""

    state = state or {}
    result = result or {}
    max_text_chars = max_text_chars or int(os.getenv("OBSERVABILITY_MAX_TEXT_CHARS", "500") or 500)
    pipeline_manifest = pipeline_manifest or build_pipeline_version_manifest()
    pipeline_fingerprint = pipeline_fingerprint or compute_pipeline_fingerprint(pipeline_manifest)

    retrieval_trace = _as_dict(result.get("retrieval_trace") or state.get("retrieval_trace"))
    packed_context = _as_dict(result.get("packed_context") or state.get("packed_context"))
    quality_report = _as_dict(result.get("answer_quality_report") or state.get("answer_quality_report"))
    normalized_query = _as_dict(retrieval_trace.get("normalized_query", {}))
    rerank_trace = _as_dict(retrieval_trace.get("rerank_trace", {}))
    issues = quality_report.get("issues", []) if isinstance(quality_report.get("issues"), list) else []

    warnings_count = len(retrieval_trace.get("warnings", []) or [])
    warnings_count += len(packed_context.get("warnings", []) or [])
    warnings_count += sum(1 for issue in issues if isinstance(issue, dict) and issue.get("severity") == "warning")

    summary_metadata = {
        "answer_guard_modified": result.get("answer_guard_modified", state.get("answer_guard_modified")),
        "answer_guard_mode": result.get("answer_guard_mode", state.get("answer_guard_mode")),
        "retrieval_status": result.get("retrieval_status", state.get("retrieval_status")),
        "fallback_applied": result.get("fallback_applied", state.get("fallback_applied")),
        "fallback_type": result.get("fallback_type", state.get("fallback_type")),
        "fallback_reason": result.get("fallback_reason", state.get("fallback_reason")),
        "fallback_cache_eligible": result.get(
            "fallback_cache_eligible",
            state.get("fallback_cache_eligible"),
        ),
        "medical_severity": result.get("medical_severity", state.get("medical_severity")),
        "severity_guard_modified": result.get(
            "severity_guard_modified",
            state.get("severity_guard_modified"),
        ),
        "cache_reason": result.get("cache_reason", state.get("cache_reason")),
        "pipeline_manifest": pipeline_manifest_summary(pipeline_manifest),
        "runtime_resilience": result.get("runtime_resilience", state.get("runtime_resilience")),
        "evidence_sufficiency": _compact_p3_summary(
            result.get("evidence_sufficiency", state.get("evidence_sufficiency")),
            result.get("abstention", state.get("abstention")),
            result.get("retry_history", state.get("retry_history")),
        ),
        "claim_grounding": _compact_p4_summary(
            result.get("claim_grounding", state.get("claim_grounding"))
        ),
    }

    summary = PipelineTraceSummary(
        query=_safe_query_summary(query),
        intent=normalized_query.get("intent") or quality_report.get("intent"),
        normalized_entities={
            "drug_product": normalized_query.get("drug_product", []),
            "active_ingredient": normalized_query.get("active_ingredient", []),
            "drug_class": normalized_query.get("drug_class", []),
            "condition": normalized_query.get("condition", []),
            "safety_context": normalized_query.get("safety_context", []),
        },
        retrieval_candidates_count=len(retrieval_trace.get("merged_candidates", []) or []),
        reranked_candidates_count=int(rerank_trace.get("output_count") or 0),
        packed_context_items_count=len(packed_context.get("items", []) or []),
        answer_quality_passed=quality_report.get("passed") if quality_report else None,
        critical_issues_count=sum(
            1 for issue in issues if isinstance(issue, dict) and issue.get("severity") == "critical"
        ),
        warnings_count=warnings_count,
        cache_hit=result.get("cache_hit", state.get("cache_hit")),
        pipeline_fingerprint=pipeline_fingerprint,
        timings_ms=_float_dict(retrieval_trace.get("timings_ms", {})),
        metadata=sanitize_for_observability(summary_metadata, max_text_chars=max_text_chars),
    )

    payload = safe_payload or {
        "sources": result.get("sources", state.get("sources", [])),
        "cache_reason": result.get("cache_reason", state.get("cache_reason")),
        "retrieval_status": result.get("retrieval_status", state.get("retrieval_status")),
        "fallback_applied": result.get("fallback_applied", state.get("fallback_applied")),
        "fallback_type": result.get("fallback_type", state.get("fallback_type")),
        "fallback_reason": result.get("fallback_reason", state.get("fallback_reason")),
        "fallback_cache_eligible": result.get("fallback_cache_eligible", state.get("fallback_cache_eligible")),
        "quality_issues": issues,
        "p3_trace": result.get("p3_trace", state.get("p3_trace")),
        "p4_trace": result.get("p4_trace", state.get("p4_trace")),
        "claim_grounding": _compact_p4_summary(
            result.get("claim_grounding", state.get("claim_grounding"))
        ),
    }

    return ObservabilityEvent(
        event_type=event_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
        trace_id=str(uuid.uuid4()),
        session_id=session_id,
        query_hash=hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:16],
        summary=summary,
        safe_payload=sanitize_for_observability(payload, max_text_chars=max_text_chars),
    )


def export_observability_event(
    event: ObservabilityEvent,
    output_dir: str | Path = "logs/phase2_traces",
    *,
    enabled: bool | None = None,
) -> bool:
    """Export an event as JSONL when enabled; fail open on filesystem errors."""

    if enabled is None:
        enabled = os.getenv("OBSERVABILITY_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if not enabled:
        return False

    try:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        target = target_dir / f"phase2_traces-{day}.jsonl"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
        return True
    except Exception as exc:  # pragma: no cover - deliberately fail-open
        logger.warning("Failed to export observability event: %s", sanitize_fallback_reason(exc))
        return False


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {}


def _float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        try:
            result[str(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return result


def _safe_query_summary(query: str) -> str:
    text = query or ""
    return f"[REDACTED_QUERY chars={len(text)}]"


def _compact_p3_summary(
    assessment: Any,
    abstention: Any,
    history: Any,
) -> dict[str, Any] | None:
    if not isinstance(assessment, dict):
        return None
    abstention = abstention if isinstance(abstention, dict) else {}
    history = history if isinstance(history, list) else []
    return {
        "status": assessment.get("status"),
        "attempts": min(2, max(1, len(history))),
        "missing_roles": list(assessment.get("missing_roles") or [])[:12],
        "critical_missing_roles": list(assessment.get("critical_missing_roles") or [])[:12],
        "retry_eligibility": assessment.get("retry_eligibility"),
        "abstention_type": abstention.get("abstention_type"),
        "abstention_reason": abstention.get("reason"),
    }


def _compact_p4_summary(value: Any) -> dict[str, Any] | None:
    try:
        return compact_claim_grounding(value)
    except Exception:
        return {
            "status": "degraded",
            "verifier_degraded": True,
            "production_answer_modified": False,
        }


__all__ = [
    "build_observability_event",
    "export_observability_event",
    "sanitize_for_observability",
]
