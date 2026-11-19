"""Pydantic contracts for runtime observability events."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_TOKEN_USAGE_COLLECTOR: ContextVar[list[dict[str, int]] | None] = ContextVar(
    "acne_llm_token_usage_collector",
    default=None,
)


def begin_token_usage_scope() -> tuple[list[dict[str, int]], Token[list[dict[str, int]] | None]]:
    """Mở vùng thu thập token theo từng lần gọi LLM, an toàn với async và worker thread."""
    collector: list[dict[str, int]] = []
    token = _TOKEN_USAGE_COLLECTOR.set(collector)
    return collector, token


def reset_token_usage_scope(token: Token[list[dict[str, int]] | None]) -> None:
    """Khôi phục ContextVar sau khi kết thúc lời gọi LLM."""
    _TOKEN_USAGE_COLLECTOR.reset(token)


def record_llm_token_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int | None = None,
) -> None:
    """Ghi nhận bộ đếm token số học từ Gemini/Ollama vào collector hiện hành."""
    collector = _TOKEN_USAGE_COLLECTOR.get()
    if collector is None:
        return
    if input_tokens < 0 or output_tokens < 0:
        return
    total = total_tokens if (total_tokens is not None and total_tokens >= input_tokens + output_tokens) else input_tokens + output_tokens
    collector.append(
        {
            "input": int(input_tokens),
            "output": int(output_tokens),
            "total": int(total),
        }
    )


def latest_token_usage(collector: list[dict[str, int]]) -> dict[str, int] | None:
    """Lấy bản ghi token mới nhất của lần sinh thành công."""
    return dict(collector[-1]) if collector else None

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
    usage_details: dict[str, int] | None = None


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
    "begin_token_usage_scope",
    "latest_token_usage",
    "record_llm_token_usage",
    "reset_token_usage_scope",
]
