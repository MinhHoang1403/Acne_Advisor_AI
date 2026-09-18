"""Privacy-preserving Langfuse backend for finalized application telemetry.

The application observability contract remains authoritative. This module is the
only place that imports or calls Langfuse, and it reconstructs an explicit
allowlist instead of forwarding arbitrary event metadata.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from src.observability.contracts import ObservabilityEvent, StageTelemetry

logger = logging.getLogger(__name__)

SUPPORTED_EVALUATION_SCORES = frozenset(
    {
        "Claim Recall",
        "Context Precision",
        "Faithfulness",
        "Claim F1",
        "Negative Rejection Rate",
    }
)

_PROHIBITED_ATTRIBUTE_MARKERS = (
    "authorization",
    "answer",
    "completion",
    "conversation",
    "cookie",
    "email",
    "history",
    "input",
    "message",
    "output",
    "password",
    "prompt",
    "question",
    "reasoning",
    "secret",
    "token",
)


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LangfuseSettings:
    """Minimal configuration; credentials are deliberately excluded from repr."""

    enabled: bool
    configured: bool
    base_url: str
    public_key: str = field(repr=False)
    secret_key: str = field(repr=False)
    environment: str = "local"

    @classmethod
    def from_environment(cls) -> "LangfuseSettings":
        master_enabled = _env_enabled("OBSERVABILITY_ENABLED")
        sink_enabled = _env_enabled("LANGFUSE_ENABLED")
        base_url = os.getenv("LANGFUSE_BASE_URL", "").strip().rstrip("/")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        return cls(
            enabled=master_enabled and sink_enabled,
            configured=bool(base_url and public_key and secret_key),
            base_url=base_url,
            public_key=public_key,
            secret_key=secret_key,
            environment=(
                os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "local").strip() or "local"
            ),
        )


@dataclass
class LangfuseRequestHandle:
    """Opaque request-scope handle used by the API without exposing SDK types."""

    sink: "LangfuseSink"
    request_id: str
    trace_id: str
    root_observation: Any
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root_observation.end()
        except Exception as exc:  # pragma: no cover - defensive SDK boundary
            self.sink.record_error(exc)


class LangfuseSink:
    """Maps allowlisted telemetry to one Langfuse observation tree."""

    def __init__(self, settings: LangfuseSettings, client: Any) -> None:
        self.settings = settings
        self.client = client
        self.last_error_category: str | None = None
        self.last_error_type: str | None = None
        self.last_enqueue_at: str | None = None
        self.reachable: bool | None = None

    def record_error(self, exc: BaseException) -> None:
        self.last_error_category = _safe_error_category(exc)
        self.last_error_type = exc.__class__.__name__
        self.reachable = False
        logger.warning(
            "Langfuse backend failed safely: error_family=observability "
            "owner=observability error_type=%s",
            exc.__class__.__name__,
        )

    def begin_request(self, request_id: str) -> LangfuseRequestHandle | None:
        try:
            trace_id = self.client.create_trace_id(seed=request_id)
            root = self.client.start_observation(
                trace_context={"trace_id": trace_id},
                name="chat-request",
                as_type="span",
                metadata={"request_id": request_id, "status": "in_progress"},
            )
            return LangfuseRequestHandle(
                sink=self,
                request_id=request_id,
                trace_id=trace_id,
                root_observation=root,
            )
        except Exception as exc:
            self.record_error(exc)
            return None

    def export(
        self,
        event: ObservabilityEvent,
        handle: LangfuseRequestHandle | None = None,
    ) -> bool:
        active_handle = handle or self.begin_request(event.request_id)
        if active_handle is None:
            return False
        if active_handle.request_id != event.request_id:
            self.record_error(ValueError("request identity mismatch"))
            active_handle.close()
            return False

        success = True
        try:
            self._add_observation_tree(active_handle.root_observation, event)
            active_handle.root_observation.update(
                metadata=_root_metadata(event),
                level=_level_for_status(event.summary.status),
                status_message=event.summary.status,
            )
            self.last_enqueue_at = datetime.now(timezone.utc).isoformat()
            self.last_error_category = None
            self.last_error_type = None
        except Exception as exc:
            success = False
            self.record_error(exc)
        finally:
            active_handle.close()
        return success

    def _add_observation_tree(self, root: Any, event: ObservabilityEvent) -> None:
        for decision in event.summary.decisions:
            observation = root.start_observation(
                name="agent-decision",
                as_type="agent",
                metadata=_without_none(
                    {
                        "request_id": event.request_id,
                        "attempt": decision.attempt,
                        "action": decision.action,
                        "reason_code": decision.reason_code,
                    }
                ),
            )
            observation.end()

        stages = [stage for stage in event.summary.stages if stage.stage != "request"]
        retrieval_stage = next((stage for stage in stages if stage.stage == "retrieve"), None)
        retrieval_parent = None
        if retrieval_stage is not None:
            retrieval_parent = _start_stage_observation(root, retrieval_stage, event.request_id)

        for stage in stages:
            if stage is retrieval_stage:
                continue
            parent = (
                retrieval_parent
                if retrieval_parent is not None and stage.component.startswith("retrieval")
                else root
            )
            observation = _start_stage_observation(parent, stage, event.request_id)
            observation.end()

        if retrieval_parent is not None:
            retrieval_parent.end()

    def check_connectivity(self) -> bool:
        """Explicit operator/test probe; this is never called by the request path."""

        try:
            self.reachable = bool(self.client.auth_check())
            if not self.reachable:
                self.last_error_category = "authentication"
                self.last_error_type = None
            return self.reachable
        except Exception as exc:
            self.record_error(exc)
            return False

    def flush(self) -> bool:
        try:
            self.client.flush()
            return True
        except Exception as exc:
            self.record_error(exc)
            return False

    def shutdown(self) -> None:
        try:
            self.client.shutdown()
        except Exception as exc:  # pragma: no cover - process shutdown safeguard
            self.record_error(exc)

    def submit_score(self, request_id: str, name: str, value: float) -> bool:
        try:
            trace_id = self.client.create_trace_id(seed=request_id)
            self.client.create_score(
                name=name,
                value=value,
                trace_id=trace_id,
                data_type="NUMERIC",
            )
            self.last_enqueue_at = datetime.now(timezone.utc).isoformat()
            return True
        except Exception as exc:
            self.record_error(exc)
            return False


def mask_langfuse_otel_spans(*, params: Any) -> Any:
    """Delete content-bearing OTel attributes before the exporter sends a batch."""

    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches: dict[Any, Any] = {}
    for identifier, span in params.spans.items():
        delete_attributes = tuple(
            key
            for key in span.attributes
            if any(marker in key.casefold() for marker in _PROHIBITED_ATTRIBUTE_MARKERS)
        )
        if delete_attributes:
            patches[identifier] = OtelSpanPatch(delete_attributes=delete_attributes)
    if not patches:
        return None
    return MaskOtelSpansResult(span_patches=patches)


def begin_langfuse_request(request_id: str) -> LangfuseRequestHandle | None:
    settings = LangfuseSettings.from_environment()
    if not settings.enabled or not settings.configured:
        return None
    sink = _get_or_create_sink(settings)
    return sink.begin_request(request_id) if sink is not None else None


def export_event_to_langfuse(
    event: ObservabilityEvent,
    *,
    handle: LangfuseRequestHandle | None = None,
) -> bool:
    settings = LangfuseSettings.from_environment()
    if not settings.enabled or not settings.configured:
        if handle is not None:
            handle.close()
        return False
    sink = handle.sink if handle is not None else _get_or_create_sink(settings)
    return sink.export(event, handle) if sink is not None else False


def submit_evaluation_score(request_id: str, name: str, value: float) -> bool:
    """Attach one real project metric without changing its name, value, or unit."""

    if not request_id.strip():
        raise ValueError("request_id is required")
    if name not in SUPPORTED_EVALUATION_SCORES:
        raise ValueError(f"unsupported evaluation score: {name}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("evaluation score value must be numeric")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError("evaluation score value must be finite")

    settings = LangfuseSettings.from_environment()
    if not settings.enabled or not settings.configured:
        return False
    sink = _get_or_create_sink(settings)
    return sink.submit_score(request_id, name, numeric_value) if sink is not None else False


def langfuse_backend_status() -> dict[str, Any]:
    """Return optional-backend diagnostics without probing the network."""

    settings = LangfuseSettings.from_environment()
    with _SINK_LOCK:
        sink = _SINK
        global_error = _GLOBAL_ERROR

    if not settings.enabled:
        state = "disabled"
    elif not settings.configured:
        state = "not_configured"
    elif sink is None and global_error is None:
        state = "configured"
    elif global_error is not None or (sink and sink.last_error_category):
        state = "degraded"
    else:
        state = "accepting_events"

    return _without_none(
        {
            "status": state,
            "enabled": settings.enabled,
            "configured": settings.configured,
            "optional": True,
            "required": False,
            "reachable": sink.reachable if sink is not None else None,
            "last_error_category": (
                sink.last_error_category if sink is not None else global_error
            ),
            "last_error_type": sink.last_error_type if sink is not None else None,
            "last_enqueue_at": sink.last_enqueue_at if sink is not None else None,
        }
    )


def check_langfuse_connectivity() -> bool:
    settings = LangfuseSettings.from_environment()
    if not settings.enabled or not settings.configured:
        return False
    sink = _get_or_create_sink(settings)
    return sink.check_connectivity() if sink is not None else False


def flush_langfuse() -> bool:
    with _SINK_LOCK:
        sink = _SINK
    return sink.flush() if sink is not None else False


def shutdown_langfuse() -> None:
    global _SINK
    with _SINK_LOCK:
        sink = _SINK
        _SINK = None
    if sink is not None:
        sink.shutdown()


def _start_stage_observation(parent: Any, stage: StageTelemetry, request_id: str) -> Any:
    kwargs: dict[str, Any] = {
        "name": stage.stage,
        "as_type": _observation_type(stage),
        "metadata": _stage_metadata(stage, request_id),
        "level": _level_for_status(stage.status),
        "status_message": stage.status,
    }
    if stage.stage == "generate" and stage.model:
        kwargs["model"] = stage.model
    return parent.start_observation(**kwargs)


def _root_metadata(event: ObservabilityEvent) -> dict[str, Any]:
    summary = event.summary
    metadata = summary.metadata
    request_stage = next(
        (stage for stage in summary.stages if stage.stage == "request"),
        None,
    )
    return _without_none(
        {
            "request_id": event.request_id,
            "status": summary.status,
            "complete": summary.complete,
            "duration_ms": summary.timings_ms.get("total_request"),
            "action": summary.action,
            "retrieval_attempts": summary.retrieval_attempts,
            "retrieval_candidates_count": summary.retrieval_candidates_count,
            "packed_context_items_count": summary.packed_context_items_count,
            "evidence_usable": summary.evidence_usable,
            "answer_quality_passed": summary.answer_quality_passed,
            "critical_issues_count": summary.critical_issues_count,
            "warnings_count": summary.warnings_count,
            "cache_hit": summary.cache_hit,
            "pipeline_fingerprint": summary.pipeline_fingerprint,
            "knowledge_build_id": summary.knowledge_build_id,
            "error_family": summary.error_family,
            "error_owner": summary.error_owner,
            "error_type": request_stage.error_type if request_stage is not None else None,
            "retrieval_status": _safe_category(metadata.get("retrieval_status")),
            "fallback_applied": metadata.get("fallback_applied"),
            "fallback_type": _safe_category(metadata.get("fallback_type")),
            "fallback_reason_code": _safe_category(metadata.get("fallback_reason_code")),
            "safety_severity": _safe_category(metadata.get("safety_severity")),
        }
    )


def _stage_metadata(stage: StageTelemetry, request_id: str) -> dict[str, Any]:
    return _without_none(
        {
            "request_id": request_id,
            "component": stage.component,
            "stage": stage.stage,
            "status": stage.status,
            "duration_ms": stage.duration_ms,
            "attempt": stage.attempt,
            "error_family": stage.error_family,
            "error_owner": stage.error_owner,
            "error_type": stage.error_type,
            "fallback": stage.fallback,
            "provider": stage.provider,
            "model": stage.model,
            "candidate_count": stage.candidate_count,
            "retained_candidate_count": stage.retained_candidate_count,
            "duplicate_candidate_count": stage.duplicate_candidate_count,
            "candidate_ids": stage.candidate_ids,
        }
    )


def _observation_type(stage: StageTelemetry) -> str:
    if stage.stage == "generate":
        return "generation"
    if stage.stage == "decide":
        return "agent"
    if stage.stage == "guard":
        return "guardrail"
    if stage.component.startswith("retrieval"):
        return "retriever" if stage.stage in {"retrieve", "dense", "bm25"} else "span"
    return "span"


def _level_for_status(status: str) -> str:
    if status == "failed":
        return "ERROR"
    if status == "degraded":
        return "WARNING"
    return "DEFAULT"


def _safe_category(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text and len(text) <= 80 and all(char.isalnum() or char in "._-" for char in text):
        return text
    return None


def _safe_error_category(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, (ConnectionError, OSError)):
        return "unavailable"
    if isinstance(exc, (TypeError, ValueError)):
        return "configuration_or_serialization"
    return "client_error"


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


_SINK_LOCK = threading.Lock()
_SINK: LangfuseSink | None = None
_SINK_SIGNATURE: tuple[str, str, str, str] | None = None
_GLOBAL_ERROR: str | None = None


def _get_or_create_sink(
    settings: LangfuseSettings,
    *,
    client_factory: Callable[..., Any] | None = None,
) -> LangfuseSink | None:
    global _GLOBAL_ERROR, _SINK, _SINK_SIGNATURE

    signature = (
        settings.base_url,
        settings.public_key,
        settings.secret_key,
        settings.environment,
    )
    with _SINK_LOCK:
        if _SINK is not None and _SINK_SIGNATURE == signature:
            return _SINK
        try:
            if client_factory is None:
                from langfuse import Langfuse

                client_factory = Langfuse
            client = client_factory(
                public_key=settings.public_key,
                secret_key=settings.secret_key,
                base_url=settings.base_url,
                tracing_enabled=True,
                environment=settings.environment,
                mask_otel_spans=mask_langfuse_otel_spans,
            )
            _SINK = LangfuseSink(settings, client)
            _SINK_SIGNATURE = signature
            _GLOBAL_ERROR = None
            return _SINK
        except Exception as exc:
            _GLOBAL_ERROR = _safe_error_category(exc)
            logger.warning(
                "Langfuse client initialization failed safely: error_type=%s",
                exc.__class__.__name__,
            )
            return None


def _reset_langfuse_sink_for_tests() -> None:
    global _GLOBAL_ERROR, _SINK, _SINK_SIGNATURE
    with _SINK_LOCK:
        _SINK = None
        _SINK_SIGNATURE = None
        _GLOBAL_ERROR = None


__all__ = [
    "LangfuseRequestHandle",
    "LangfuseSettings",
    "LangfuseSink",
    "SUPPORTED_EVALUATION_SCORES",
    "begin_langfuse_request",
    "check_langfuse_connectivity",
    "export_event_to_langfuse",
    "flush_langfuse",
    "langfuse_backend_status",
    "mask_langfuse_otel_spans",
    "shutdown_langfuse",
    "submit_evaluation_score",
]
