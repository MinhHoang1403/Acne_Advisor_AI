"""Runtime observability helpers."""

from src.observability.contracts import (
    AgentDecisionTelemetry,
    ObservabilityEvent,
    PipelineTraceSummary,
    StageStatus,
    StageTelemetry,
)
from src.observability.langfuse_sink import (
    SUPPORTED_EVALUATION_SCORES,
    begin_langfuse_request,
    check_langfuse_connectivity,
    flush_langfuse,
    langfuse_backend_status,
    shutdown_langfuse,
    submit_evaluation_score,
)
from src.observability.error_taxonomy import ERROR_FAMILY_OWNERS, ErrorFamily, classify_error
from src.observability.trace_exporter import (
    build_observability_event,
    emit_request_completion,
    export_observability_event,
    sanitize_for_observability,
)
from src.observability.versioning import (
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
    current_pipeline_fingerprint,
    get_answer_cache_version,
    pipeline_manifest_summary,
)

__all__ = [
    "ERROR_FAMILY_OWNERS",
    "ErrorFamily",
    "AgentDecisionTelemetry",
    "ObservabilityEvent",
    "PipelineTraceSummary",
    "StageStatus",
    "StageTelemetry",
    "build_observability_event",
    "build_pipeline_version_manifest",
    "classify_error",
    "compute_pipeline_fingerprint",
    "current_pipeline_fingerprint",
    "export_observability_event",
    "emit_request_completion",
    "get_answer_cache_version",
    "pipeline_manifest_summary",
    "sanitize_for_observability",
    "SUPPORTED_EVALUATION_SCORES",
    "begin_langfuse_request",
    "check_langfuse_connectivity",
    "flush_langfuse",
    "langfuse_backend_status",
    "shutdown_langfuse",
    "submit_evaluation_score",
]
