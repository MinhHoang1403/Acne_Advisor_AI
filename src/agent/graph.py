"""Ghép các node thành workflow LangGraph có retrieval loop hữu hạn.

Luồng chính là ``prepare -> guard -> decide``. Action ``retrieve`` và ``retry``
đều chạy node ``retrieve`` rồi ``assess`` và quay lại ``decide``; action
``generate`` hoặc ``abstain`` đi tới ``finalize``. Model chọn action trong
``action_decision.py``, còn file này sở hữu topology và thực thi transition.

Workflow không trực tiếp search Qdrant hoặc gọi LLM sinh câu trả lời; các side
effect đó thuộc retrieval service và generation node tương ứng.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langgraph.graph import END, START, StateGraph  # type: ignore[import]

from src.agent.nodes.workflow import (
    abstain_node,
    assess_evidence_node,
    decide_node,
    finalize_node,
    generate_node,
    guard_node,
    prepare_node,
    retrieve_node,
)
from src.agent.state import AgentAction, ClinicalState
from src.observability.versioning import build_pipeline_version_manifest, compute_pipeline_fingerprint
from src.resilience.budget import DeadlineBudget
from src.resilience.contracts import runtime_resilience_settings_from_env
from src.resilience.exceptions import AgentTimeoutError

logger = logging.getLogger(__name__)


def route_agent_action(state: ClinicalState) -> AgentAction:
    """Trả action đã được decision node chọn và Python validation."""

    return state.get("next_action", "abstain")


def build_clinical_graph():
    """Tạo workflow tám node; ``retry`` tái sử dụng đúng node ``retrieve``."""

    builder = StateGraph(ClinicalState)
    builder.add_node("prepare", prepare_node)
    builder.add_node("guard", guard_node)
    builder.add_node("decide", decide_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("assess", assess_evidence_node)
    builder.add_node("generate", generate_node)
    builder.add_node("abstain", abstain_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "guard")
    builder.add_edge("guard", "decide")
    builder.add_conditional_edges(
        "decide",
        route_agent_action,
        {
            "retrieve": "retrieve",
            "retry": "retrieve",
            "generate": "generate",
            "abstain": "abstain",
            "finalize": "finalize",
        },
    )
    builder.add_edge("retrieve", "assess")
    builder.add_edge("assess", "decide")
    builder.add_edge("generate", "finalize")
    builder.add_edge("abstain", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


clinical_graph = build_clinical_graph()


async def run_clinical_agent(
    message: str,
    user_id: str | None = None,
    session_id: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    allow_model_fallback: bool = False,
    bypass_cache: bool = False,
    include_generation_diagnostics: bool = False,
) -> dict[str, Any]:
    """Chạy một request trong deadline chung và trả contract ổn định cho API.

    State khởi tạo ở đây là dữ liệu trao đổi giữa các node. ``DeadlineBudget``
    dùng cùng một mốc thời gian cho toàn request, nên timeout con không làm mới
    tổng ngân sách. Hàm chỉ chọn các field public sau khi graph hoàn tất; raw
    generation data chỉ được trả khi diagnostic caller chủ động yêu cầu.
    """

    started = time.perf_counter()
    manifest = build_pipeline_version_manifest()
    fingerprint = compute_pipeline_fingerprint(manifest)
    settings = runtime_resilience_settings_from_env()
    effective_model_fallback = bool(
        allow_model_fallback and settings.llm_provider_fallback_enabled
    )
    budget = DeadlineBudget.from_timeout(settings.agent_total_timeout_seconds)
    initial_state: ClinicalState = {
        "user_question": message,
        "user_id": user_id,
        "session_id": session_id,
        "conversation_history": list(conversation_history or []),
        "standalone_question": None,
        "retrieval_query": None,
        "normalized_question": "",
        "conversation_context": None,
        "is_in_domain": None,
        "agent_decision": None,
        "safety_override": False,
        "vector_contexts": [],
        "sources": [],
        "source_allowlist": [],
        "retrieval_status": "not_started",
        "retrieval_attempt": 0,
        "retry_history": [],
        "agent_decision_history": [],
        "agent_decision_evidence_traces": [],
        "retrieval_attempt_traces": [],
        "evidence_assessment": None,
        "safety_severity": None,
        "safety_decision": None,
        "draft_answer": "",
        "final_answer": "",
        "fallback_applied": False,
        "fallback_type": "none",
        "fallback_reason_code": None,
        "fallback_cache_eligible": True,
        "cache_hit": False,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "allow_model_fallback": effective_model_fallback,
        "llm_fallback_used": False,
        "bypass_cache": bypass_cache,
        "pipeline_manifest": manifest,
        "pipeline_fingerprint": fingerprint,
        "runtime_budget": budget,
        "runtime_resilience_settings": settings.model_dump(mode="json"),
        "runtime_resilience": {
            "runtime_resilience_version": manifest.get("runtime_resilience_version"),
            "agent_total_timeout_seconds": settings.agent_total_timeout_seconds,
            "deadline_started": True,
        },
        "performance_timings": {},
    }

    try:
        async with asyncio.timeout(budget.remaining_seconds()):
            final = await clinical_graph.ainvoke(initial_state)
    except asyncio.CancelledError:
        raise
    except TimeoutError as exc:
        raise AgentTimeoutError(
            f"Agent exceeded total timeout of {settings.agent_total_timeout_seconds:.1f}s."
        ) from exc

    performance_timings = {
        **(final.get("performance_timings") or {}),
        "agent_total": round((time.perf_counter() - started) * 1000, 3),
    }
    result = {
        "answer": final.get("final_answer", ""),
        "user_id": final.get("user_id"),
        "session_id": final.get("session_id"),
        "standalone_question": final.get("standalone_question"),
        "retrieval_query": final.get("retrieval_query"),
        "vector_contexts": final.get("vector_contexts", []),
        "sources": final.get("sources", []),
        "source_allowlist": final.get("source_allowlist", []),
        "source_validation": final.get("source_validation"),
        "retrieval_status": final.get("retrieval_status"),
        "retrieval_error": final.get("retrieval_error"),
        "retrieval_trace": final.get("retrieval_trace"),
        "packed_context": final.get("packed_context"),
        "evidence_assessment": final.get("evidence_assessment"),
        "agent_decision": final.get("agent_decision"),
        "agent_decision_history": final.get("agent_decision_history", []),
        "agent_decision_evidence_traces": final.get("agent_decision_evidence_traces", []),
        "safety_decision": final.get("safety_decision"),
        "retrieval_attempt": final.get("retrieval_attempt", 0),
        "retry_history": final.get("retry_history", []),
        "retrieval_attempt_traces": final.get("retrieval_attempt_traces", []),
        "pipeline_manifest": final.get("pipeline_manifest"),
        "pipeline_fingerprint": final.get("pipeline_fingerprint"),
        "observability_exported": final.get("observability_exported"),
        "runtime_resilience": final.get("runtime_resilience"),
        "performance_timings": performance_timings,
        "prompt_budget": final.get("prompt_budget"),
        "generation_evidence_trace": final.get("generation_evidence_trace"),
        "answer_quality_report": final.get("answer_quality_report"),
        "safety_severity": final.get("safety_severity"),
        "fallback_applied": final.get("fallback_applied", False),
        "fallback_type": final.get("fallback_type"),
        "fallback_reason": final.get("fallback_reason"),
        "fallback_reason_code": final.get("fallback_reason_code"),
        "fallback_cache_eligible": final.get("fallback_cache_eligible"),
        "is_in_domain": final.get("is_in_domain"),
        "cache_checked": final.get("cache_checked"),
        "cache_hit": final.get("cache_hit"),
        "cache_reason": final.get("cache_reason"),
        "cache_metadata": final.get("cache_metadata"),
        "requested_provider": final.get("requested_provider"),
        "requested_model": final.get("requested_model"),
        "actual_provider": final.get("actual_provider"),
        "actual_model": final.get("actual_model"),
        "llm_fallback_used": final.get("llm_fallback_used", False),
        "fallback_provider": final.get("fallback_provider"),
        "fallback_model": final.get("fallback_model"),
        "fallback_chain": final.get("fallback_chain"),
    }
    if include_generation_diagnostics:
        quality_report = final.get("answer_quality_report")
        result["generation_diagnostics"] = {
            "raw_generated_answer": final.get("draft_answer", ""),
            "pre_verifier_answer": (
                quality_report.get("checked_answer") or final.get("final_answer", "")
                if isinstance(quality_report, dict)
                else final.get("final_answer", "")
            ),
        }
    return result


__all__ = ["build_clinical_graph", "clinical_graph", "route_agent_action", "run_clinical_agent"]
