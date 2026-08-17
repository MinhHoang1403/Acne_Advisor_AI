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
    event_type: str = "phase2_chat_trace",
    pipeline_manifest: dict[str, Any] | None = None,
    pipeline_fingerprint: str | None = None,
    safe_payload: dict[str, Any] | None = None,
    max_text_chars: int | None = None,
) -> ObservabilityEvent:
    """Tạo event gọn từ state/result và runtime contract hiện tại."""

    state = state or {}
    result = result or {}
    max_chars = max_text_chars or int(os.getenv("OBSERVABILITY_MAX_TEXT_CHARS", "500") or 500)
    manifest = pipeline_manifest or build_pipeline_version_manifest()
    fingerprint = pipeline_fingerprint or compute_pipeline_fingerprint(manifest)
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

    summary = PipelineTraceSummary(
        query=_safe_query_summary(query),
        action=result.get("next_action", state.get("next_action")),
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
        timings_ms=timings,
        metadata=sanitize_for_observability(
            {
                "retrieval_status": result.get("retrieval_status", state.get("retrieval_status")),
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
    payload = safe_payload or {
        "sources": result.get("sources", state.get("sources", [])),
        "retrieval_status": result.get("retrieval_status", state.get("retrieval_status")),
        "retry_history": result.get("retry_history", state.get("retry_history", [])),
        "fallback_applied": result.get("fallback_applied", state.get("fallback_applied")),
        "fallback_type": result.get("fallback_type", state.get("fallback_type")),
        "fallback_reason": result.get("fallback_reason", state.get("fallback_reason")),
        "fallback_reason_code": result.get(
            "fallback_reason_code", state.get("fallback_reason_code")
        ),
        "evidence_locality": _as_dict(
            quality.get("metadata") if isinstance(quality, dict) else {}
        ).get("evidence_locality"),
        "quality_issues": issues,
    }
    return ObservabilityEvent(
        event_type=event_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
        trace_id=str(uuid.uuid4()),
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
        logger.warning("Failed to export observability event: %s", sanitize_fallback_reason(exc))
        return False


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


__all__ = ["build_observability_event", "export_observability_event", "sanitize_for_observability"]
