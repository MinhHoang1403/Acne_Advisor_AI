"""Pydantic contracts for runtime observability events."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

StageStatus = Literal["success", "degraded", "failed", "skipped"]
AgentAction = Literal["retrieve", "retry", "generate", "abstain", "finalize"]


class StageTelemetry(BaseModel):
    """Safe operational facts for one observed runtime stage."""

    model_config = ConfigDict(extra="forbid")

    component: str
    stage: str
    status: StageStatus
    duration_ms: float | None = Field(default=None, ge=0)
    attempt: int | None = Field(default=None, ge=1)
    error_family: str | None = None
    error_owner: str | None = None
    error_type: str | None = None
    fallback: bool | None = None
    provider: str | None = None
    model: str | None = None
    candidate_count: int | None = Field(default=None, ge=0)
    retained_candidate_count: int | None = Field(default=None, ge=0)
    duplicate_candidate_count: int | None = Field(default=None, ge=0)
    candidate_ids: list[str] = Field(default_factory=list)


class AgentDecisionTelemetry(BaseModel):
    """Allowlisted decision facts; free-form model reasoning is intentionally absent."""

    model_config = ConfigDict(extra="forbid")

    attempt: int = Field(ge=1)
    action: AgentAction
    reason_code: str | None = None


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
    knowledge_build_id: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    stages: list[StageTelemetry] = Field(default_factory=list)
    decisions: list[AgentDecisionTelemetry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: StageStatus = "success"
    error_family: str | None = None
    error_owner: str | None = None
    complete: bool = False


class ObservabilityEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: str
    timestamp: str
    request_id: str
    session_id: str | None = None
    query_hash: str
    summary: PipelineTraceSummary
    safe_payload: dict[str, Any] = Field(default_factory=dict)

__all__ = [
    "AgentDecisionTelemetry",
    "ObservabilityEvent",
    "PipelineTraceSummary",
    "StageStatus",
    "StageTelemetry",
]
