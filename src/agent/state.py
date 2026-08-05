"""
src/agent/state.py – LangGraph Agent State
==========================================
Defines the `ClinicalState` representing the execution state of the
Acne Advisor AI agent across graph nodes.
"""

from typing import Any, TypedDict


class ClinicalState(TypedDict):
    """The state payload passed between nodes in the LangGraph agent."""

    # Input
    user_question: str
    user_id: str | None
    session_id: str | None
    conversation_history: list[dict[str, str]]
    
    # Processed Input
    standalone_question: str | None
    use_history_context: bool | None
    normalized_question: str
    patient_profile: dict[str, Any]
    symptoms: list[str]
    
    # Guardrails
    is_in_domain: bool | None
    guardrail: str | None
    ignored_out_of_domain_part: bool | None
    domain_reason: str | None
    refusal_message: str | None
    
    # Retrieval
    vector_contexts: list[dict[str, Any]]
    graph_facts: list[dict[str, Any]]
    sources: list[str]
    retrieval_status: str | None
    retrieval_error: str | None
    retrieval_trace: dict[str, Any] | None
    packed_context: dict[str, Any] | None
    pipeline_manifest: dict[str, Any] | None
    pipeline_fingerprint: str | None
    observability_exported: bool | None
    runtime_budget: Any
    runtime_resilience_settings: dict[str, Any] | None
    runtime_resilience: dict[str, Any] | None
    
    # Reasoning & Generation
    safety_flags: list[str]
    draft_answer: str
    final_answer: str
    answer_quality_report: dict[str, Any] | None
    answer_guard_modified: bool | None
    answer_guard_mode: str | None
    medical_severity: str | None
    severity_guard: dict[str, Any] | None
    severity_guard_modified: bool | None
    severity_guard_cache_eligible: bool | None
    fallback_applied: bool
    fallback_type: str | None
    fallback_reason: str | None
    fallback_answer: str | None
    fallback_cache_eligible: bool | None
    
    # Error handling
    errors: list[str]
    llm_fallback: bool | None
    
    # Cache
    cache_enabled: bool | None
    cache_checked: bool | None
    cache_hit: bool | None
    cache_key: str | None
    cache_similarity: float | None
    cache_reason: str | None
    cached_answer: str | None
    cached_sources: list[str] | None
    cache_metadata: dict[str, Any] | None
    
    # Multi-Model Support
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
    
    # Cache bypass (for test scripts)
    bypass_cache: bool

    # Internal-only execution mode used by the local evaluation runner. It is
    # intentionally not exposed by the public API contract.
    evaluation_mode: bool
