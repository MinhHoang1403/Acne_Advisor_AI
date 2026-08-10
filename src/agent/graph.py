"""
src/agent/graph.py
==================
Main definition of the LangGraph workflow for Acne Advisor AI.
"""

import asyncio
import logging
from typing import Any

from langgraph.graph import StateGraph, START, END  # type: ignore[import]

from src.agent.state import ClinicalState
from src.agent.nodes.retrieve import (
    normalize_question_node,
    rewrite_question_node,
    extract_symptoms_node,
    retrieve_context_node
)
from src.agent.nodes.reason import (
    safety_check_node,
    generate_answer_node
)
from src.agent.nodes.respond import finalize_response_node
from src.agent.nodes.fallback import (
    fallback_decision_node,
    generation_fallback_decision_node,
    safe_fallback_node,
)
from src.agent.nodes.quality import answer_quality_node
from src.agent.nodes.guardrails import domain_guard_node
from src.agent.nodes.cache import cache_lookup_node, cache_store_node
from src.agent.nodes.severity import severity_classification_node
from src.agent.nodes.observability import observability_export_node
from src.observability.versioning import (
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
)
from src.resilience.budget import DeadlineBudget
from src.resilience.contracts import runtime_resilience_settings_from_env
from src.resilience.exceptions import AgentTimeoutError

logger = logging.getLogger(__name__)

def route_after_guard(state: ClinicalState):
    """Route to cache lookup or finalize based on guardrail check."""
    if state.get("is_in_domain"):
        return "cache_lookup"
    return "finalize"

def route_after_cache(state: ClinicalState):
    """Route to finalize (if hit) or extract (if miss)."""
    if state.get("cache_hit"):
        return "finalize"
    return "extract"


def route_after_fallback_decision(state: ClinicalState):
    """Route to deterministic fallback when evidence or generation is unusable."""
    if state.get("fallback_applied"):
        return "safe_fallback"
    return "safety"


def route_after_generation_fallback_decision(state: ClinicalState):
    """Route post-generation validation to fallback or finalization."""
    if state.get("fallback_applied"):
        return "safe_fallback"
    return "finalize"

def build_clinical_graph():
    """Builds and compiles the StateGraph for the clinical agent."""
    
    # Initialize the graph with our state schema
    builder = StateGraph(ClinicalState)
    
    # Add all nodes
    builder.add_node("normalize", normalize_question_node)
    builder.add_node("rewrite", rewrite_question_node)
    builder.add_node("severity", severity_classification_node)
    builder.add_node("guard", domain_guard_node)
    builder.add_node("cache_lookup", cache_lookup_node)
    builder.add_node("extract", extract_symptoms_node)
    builder.add_node("retrieve", retrieve_context_node)
    builder.add_node("fallback_decision", fallback_decision_node)
    builder.add_node("safety", safety_check_node)
    builder.add_node("generate", generate_answer_node)
    builder.add_node("generation_fallback_decision", generation_fallback_decision_node)
    builder.add_node("safe_fallback", safe_fallback_node)
    builder.add_node("cache_store", cache_store_node)
    builder.add_node("finalize", finalize_response_node)
    builder.add_node("answer_quality", answer_quality_node)
    builder.add_node("observability_export", observability_export_node)
    
    # Define the edges (the flow)
    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "rewrite")
    builder.add_edge("rewrite", "severity")
    builder.add_edge("severity", "guard")
    
    # Conditional routing after guard
    builder.add_conditional_edges("guard", route_after_guard)
    
    # Conditional routing after cache lookup
    builder.add_conditional_edges("cache_lookup", route_after_cache)
    
    builder.add_edge("extract", "retrieve")
    builder.add_edge("retrieve", "fallback_decision")
    builder.add_conditional_edges("fallback_decision", route_after_fallback_decision)
    builder.add_edge("safety", "generate")
    builder.add_edge("generate", "generation_fallback_decision")
    builder.add_conditional_edges(
        "generation_fallback_decision",
        route_after_generation_fallback_decision,
    )
    builder.add_edge("safe_fallback", "finalize")
    builder.add_edge("finalize", "answer_quality")
    builder.add_edge("answer_quality", "cache_store")
    builder.add_edge("cache_store", "observability_export")
    builder.add_edge("observability_export", END)
    
    # Compile the graph
    graph = builder.compile()
    return graph


# Create a singleton instance of the graph
clinical_graph = build_clinical_graph()


async def run_clinical_agent(
    message: str,
    user_id: str | None = None,
    session_id: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    allow_model_fallback: bool = True,
    bypass_cache: bool = False,
    evaluation_mode: bool = False,
    evaluation_expected_concepts: list[str] | None = None,
    evaluation_critical_case: bool = False,
    evaluation_concept_matcher: Any = None,
) -> dict[str, Any]:
    """
    Entrypoint to run the agent with a user message.
    
    Args:
        message (str): The user's question or statement.
        user_id (str, optional): The ID of the user.
        session_id (str, optional): The ID of the chat session.
        conversation_history: List of history messages.
        
    Returns:
        dict: A dictionary containing the final answer, symptoms, safety flags, sources, etc.
    """
    
    logger.info(f"Running clinical agent for user={user_id}, session={session_id}")
    
    if conversation_history is None:
        conversation_history = []
    
    pipeline_manifest = build_pipeline_version_manifest()
    pipeline_fingerprint = compute_pipeline_fingerprint(pipeline_manifest)
    resilience_settings = runtime_resilience_settings_from_env()
    runtime_budget = DeadlineBudget.from_timeout(resilience_settings.agent_total_timeout_seconds)

    # Initialize state
    initial_state = {
        "user_question": message,
        "user_id": user_id,
        "session_id": session_id,
        "conversation_history": conversation_history,
        "standalone_question": None,
        "use_history_context": False,
        "conversation_context": None,
        "normalized_question": "",
        "patient_profile": {},
        "symptoms": [],
        "vector_contexts": [],
        "graph_facts": [],
        "graph_relation_found": False,
        "sources": [],
        "source_allowlist": [],
        "source_validation": None,
        "retrieval_status": "not_started",
        "retrieval_error": None,
        "retrieval_trace": None,
        "packed_context": None,
        "retrieval_diagnostics": None,
        "prompt_evidence_trace": None,
        "pipeline_manifest": pipeline_manifest,
        "pipeline_fingerprint": pipeline_fingerprint,
        "observability_exported": None,
        "runtime_budget": runtime_budget,
        "runtime_resilience_settings": resilience_settings.model_dump(mode="json"),
        "runtime_resilience": {
            "runtime_resilience_version": pipeline_manifest.get("runtime_resilience_version"),
            "agent_total_timeout_seconds": resilience_settings.agent_total_timeout_seconds,
            "deadline_started": True,
        },
        "performance_timings": {},
        "safety_flags": [],
        "draft_answer": "",
        "final_answer": "",
        "answer_quality_report": None,
        "answer_guard_modified": None,
        "answer_guard_mode": None,
        "medical_severity": None,
        "severity_guard": None,
        "severity_guard_modified": None,
        "severity_guard_cache_eligible": None,
        "fallback_applied": False,
        "fallback_type": "none",
        "fallback_reason": None,
        "fallback_answer": None,
        "fallback_cache_eligible": True,
        "answerability": None,
        "errors": [],
        "cache_enabled": None,
        "cache_checked": None,
        "cache_hit": None,
        "cache_key": None,
        "cache_similarity": None,
        "cache_reason": None,
        "cached_answer": None,
        "cached_sources": None,
        "cache_metadata": None,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "allow_model_fallback": allow_model_fallback,
        "requested_provider": None,
        "requested_model": None,
        "actual_provider": None,
        "actual_model": None,
        "llm_fallback_used": False,
        "fallback_provider": None,
        "fallback_model": None,
        "fallback_chain": None,
        "bypass_cache": bypass_cache,
        "evaluation_mode": evaluation_mode,
        "evaluation_expected_concepts": list(evaluation_expected_concepts or []),
        "evaluation_critical_case": evaluation_critical_case,
        "evaluation_concept_matcher": evaluation_concept_matcher,
    }
    
    # Invoke the graph asynchronously
    try:
        async with asyncio.timeout(runtime_budget.remaining_seconds()):
            final_state = await clinical_graph.ainvoke(initial_state)
    except asyncio.CancelledError:
        raise
    except TimeoutError as exc:
        raise AgentTimeoutError(
            f"Agent exceeded total timeout of {resilience_settings.agent_total_timeout_seconds:.1f}s."
        ) from exc
    
    # Format and return the output
    return {
        "answer": final_state.get("final_answer", ""),
        "user_id": final_state.get("user_id"),
        "session_id": final_state.get("session_id"),
        "standalone_question": final_state.get("standalone_question"),
        "symptoms": final_state.get("symptoms", []),
        "safety_flags": final_state.get("safety_flags", []),
        "vector_contexts": final_state.get("vector_contexts", []),
        "sources": final_state.get("sources", []),
        "graph_facts": final_state.get("graph_facts", []),
        "graph_relation_found": final_state.get("graph_relation_found", False),
        "source_allowlist": final_state.get("source_allowlist", []),
        "source_validation": final_state.get("source_validation"),
        "retrieval_status": final_state.get("retrieval_status"),
        "retrieval_error": final_state.get("retrieval_error"),
        "retrieval_trace": final_state.get("retrieval_trace"),
        "packed_context": final_state.get("packed_context"),
        "retrieval_diagnostics": final_state.get("retrieval_diagnostics"),
        "prompt_evidence_trace": final_state.get("prompt_evidence_trace"),
        "pipeline_manifest": final_state.get("pipeline_manifest"),
        "pipeline_fingerprint": final_state.get("pipeline_fingerprint"),
        "observability_exported": final_state.get("observability_exported"),
        "runtime_resilience": final_state.get("runtime_resilience"),
        "performance_timings": final_state.get("performance_timings", {}),
        "answer_quality_report": final_state.get("answer_quality_report"),
        "answer_guard_modified": final_state.get("answer_guard_modified"),
        "answer_guard_mode": final_state.get("answer_guard_mode"),
        "medical_severity": final_state.get("medical_severity"),
        "severity_guard": final_state.get("severity_guard"),
        "severity_guard_modified": final_state.get("severity_guard_modified"),
        "severity_guard_cache_eligible": final_state.get("severity_guard_cache_eligible"),
        "fallback_applied": final_state.get("fallback_applied"),
        "fallback_type": final_state.get("fallback_type"),
        "fallback_reason": final_state.get("fallback_reason"),
        "fallback_cache_eligible": final_state.get("fallback_cache_eligible"),
        "answerability": final_state.get("answerability"),
        "errors": final_state.get("errors", []),
        "is_in_domain": final_state.get("is_in_domain"),
        "guardrail": final_state.get("guardrail"),
        "ignored_out_of_domain_part": final_state.get("ignored_out_of_domain_part"),
        "domain_reason": final_state.get("domain_reason"),
        "cache_checked": final_state.get("cache_checked"),
        "cache_hit": final_state.get("cache_hit"),
        "cache_reason": final_state.get("cache_reason"),
        "cache_metadata": final_state.get("cache_metadata"),
        "llm_fallback": final_state.get("llm_fallback"),
        "fallback_reason": final_state.get("fallback_reason"),
        "requested_provider": final_state.get("requested_provider"),
        "requested_model": final_state.get("requested_model"),
        "actual_provider": final_state.get("actual_provider"),
        "actual_model": final_state.get("actual_model"),
        "llm_fallback_used": final_state.get("llm_fallback_used"),
        "fallback_provider": final_state.get("fallback_provider"),
        "fallback_model": final_state.get("fallback_model"),
        "fallback_chain": final_state.get("fallback_chain")
    }
