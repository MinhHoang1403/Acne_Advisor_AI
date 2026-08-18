"""Contract state dùng chung giữa các node của LangGraph Agent.

Mỗi node chỉ đọc/ghi các field cần truyền sang node kế tiếp; dữ liệu tạm nên ở
local variable hoặc trace nhỏ. File này định nghĩa hình dạng dữ liệu, không chứa
routing, retrieval, safety hay generation logic.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


AgentAction = Literal["retrieve", "retry", "generate", "abstain", "finalize"]


class ClinicalState(TypedDict, total=False):
    """Các field có thể đi qua ranh giới giữa các graph node."""

    # Input request và conversation context đã được giới hạn kích thước.
    user_question: str
    user_id: str | None
    session_id: str | None
    conversation_history: list[dict[str, str]]
    standalone_question: str | None
    retrieval_query: str | None
    normalized_question: str
    conversation_context: dict[str, Any] | None

    # Kết quả safety deterministic và action đã qua Python validation.
    is_in_domain: bool | None
    safety_severity: str | None
    safety_decision: dict[str, Any] | None
    next_action: AgentAction
    agent_decision: dict[str, Any] | None
    safety_override: bool

    # Evidence, provenance và trace từ retrieval; không đồng nghĩa clinical truth.
    vector_contexts: list[dict[str, Any]]
    sources: list[str]
    source_allowlist: list[dict[str, Any]]
    source_validation: dict[str, Any] | None
    retrieval_status: str | None
    retrieval_error: str | None
    retrieval_trace: dict[str, Any] | None
    packed_context: dict[str, Any] | None
    evidence_assessment: dict[str, Any] | None
    retrieval_attempt: int
    retry_history: list[dict[str, Any]]
    agent_decision_history: list[dict[str, Any]]
    retrieval_attempt_traces: list[dict[str, Any]]

    # Bản nháp, câu trả lời trình bày cuối cùng và trạng thái fallback.
    draft_answer: str
    final_answer: str
    answer_quality_report: dict[str, Any] | None
    fallback_applied: bool
    fallback_type: str | None
    fallback_reason: str | None
    fallback_reason_code: str | None
    fallback_answer: str | None
    fallback_cache_eligible: bool | None

    # Exact cache identity và provider/model thực tế đã phục vụ request.
    cache_checked: bool | None
    cache_hit: bool | None
    cache_reason: str | None
    cached_answer: str | None
    cache_metadata: dict[str, Any] | None
    bypass_cache: bool
    llm_provider: str | None
    llm_model: str | None
    allow_model_fallback: bool
    requested_provider: str | None
    requested_model: str | None
    actual_provider: str | None
    actual_model: str | None
    llm_fallback_used: bool
    fallback_provider: str | None
    fallback_model: str | None
    fallback_chain: list[dict[str, Any]] | None

    # Deadline dùng chung, version identity và telemetry có giới hạn.
    pipeline_manifest: dict[str, Any]
    pipeline_fingerprint: str
    runtime_budget: Any
    runtime_resilience_settings: dict[str, Any]
    runtime_resilience: dict[str, Any] | None
    performance_timings: dict[str, float]
    prompt_budget: dict[str, Any] | None
    generation_evidence_trace: dict[str, Any] | None
    response_profile: str | None
    observability_exported: bool | None

__all__ = ["AgentAction", "ClinicalState"]
