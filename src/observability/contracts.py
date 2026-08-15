"""Pydantic contracts for runtime observability events."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PipelineTraceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    action: str | None = None
    retrieval_candidates_count: int = 0
    packed_context_items_count: int = 0
    retrieval_attempts: int = 0
    evidence_usable: bool | None = None
    answer_quality_passed: bool | None = None
    critical_issues_count: int = 0
    warnings_count: int = 0
    cache_hit: bool | None = None
    pipeline_fingerprint: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservabilityEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: str
    timestamp: str
    trace_id: str
    session_id: str | None = None
    query_hash: str
    summary: PipelineTraceSummary
    safe_payload: dict[str, Any] = Field(default_factory=dict)


__all__ = ["ObservabilityEvent", "PipelineTraceSummary"]
