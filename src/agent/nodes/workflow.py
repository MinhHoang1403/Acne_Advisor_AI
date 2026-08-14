"""Nodes for the final bounded LangGraph agent workflow."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.agent.nodes.cache import cache_lookup_node, cache_store_node
from src.agent.nodes.fallback import generation_fallback_decision_node, safe_fallback_node
from src.agent.nodes.guardrails import domain_guard_node
from src.agent.nodes.observability import observability_export_node
from src.agent.nodes.quality import answer_quality_node
from src.agent.nodes.reason import generate_answer_node
from src.agent.nodes.respond import finalize_response_node
from src.agent.nodes.preparation import extract_symptoms_node, normalize_question_node, rewrite_question_node
from src.agent.nodes.severity import severity_classification_node
from src.agent.source_presentation import build_source_allowlist
from src.agent.state import AgentAction, ClinicalState
from src.quality.safe_fallback import sanitize_fallback_reason
from src.retrieval.service import retrieve_evidence

MAX_RETRIEVAL_ATTEMPTS = 2


async def prepare_node(state: ClinicalState) -> dict[str, Any]:
    """Normalize the request, resolve conversation context, and classify severity."""

    updates: dict[str, Any] = {}
    for node in (normalize_question_node, rewrite_question_node, extract_symptoms_node, severity_classification_node):
        updates.update(await node({**state, **updates}))
    return updates


async def guard_node(state: ClinicalState) -> dict[str, Any]:
    """Apply deterministic domain/safety policy before consulting the cache."""

    guard = await domain_guard_node(state)
    updates: dict[str, Any] = dict(guard)
    if guard.get("is_in_domain"):
        updates.update(await cache_lookup_node({**state, **updates}))
    return updates


async def decide_node(state: ClinicalState) -> dict[str, AgentAction]:
    """Choose the next meaningful bounded agent action."""

    if state.get("is_in_domain") is False or state.get("cache_hit"):
        action: AgentAction = "finalize"
    else:
        assessment = state.get("evidence_assessment") or {}
        if assessment.get("usable"):
            action = "generate"
        elif state.get("retrieval_attempt", 0) == 0:
            action = "retrieve"
        elif _next_retrieval_query(state) is None:
            action = "abstain"
        else:
            action = "retrieve"
    return {"next_action": action}


async def retrieve_node(state: ClinicalState) -> dict[str, Any]:
    """Invoke the one source-evidence tool and preserve its typed trace."""

    attempt = state.get("retrieval_attempt", 0) + 1
    retry_plan = _next_retrieval_query(state)
    if retry_plan is None:
        question = state.get("standalone_question") or state.get("user_question", "")
        retry_reason = "initial_retrieval"
    else:
        question, retry_reason = retry_plan

    started = time.perf_counter()
    try:
        payload = await retrieve_evidence.ainvoke({"query": question, "top_k": 8})
        metadata = payload.get("metadata") or {}
        trace = metadata.get("retrieval_trace") or {}
        status = metadata.get("retrieval_status") or ("ok" if payload.get("vector_contexts") else "no_evidence")
        history_entry = {
            "attempt": attempt,
            "query": question,
            "status": status,
            "reason": retry_reason,
            "selected_ids": trace.get("selected_ids", []),
        }
        return {
            "retrieval_attempt": attempt,
            "vector_contexts": payload.get("vector_contexts", []),
            "sources": payload.get("sources", []),
            "source_allowlist": build_source_allowlist(
                payload.get("sources", []),
                payload.get("vector_contexts", []),
            ),
            "retrieval_status": status,
            "retrieval_error": None,
            "retrieval_trace": trace,
            "packed_context": metadata.get("packed_context"),
            "retry_history": [*(state.get("retry_history") or []), history_entry],
            "performance_timings": {
                **(state.get("performance_timings") or {}),
                f"retrieval_attempt_{attempt}": round((time.perf_counter() - started) * 1000, 3),
            },
        }
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error = sanitize_fallback_reason(exc)
        return {
            "retrieval_attempt": attempt,
            "vector_contexts": [],
            "sources": [],
            "retrieval_status": "failed",
            "retrieval_error": error,
            "retrieval_trace": {"architecture": "dense_bm25_rrf", "error": error},
            "packed_context": None,
            "retry_history": [
                *(state.get("retry_history") or []),
                {
                    "attempt": attempt,
                    "query": question,
                    "status": "failed",
                    "reason": retry_reason,
                    "error": error,
                },
            ],
        }


async def assess_evidence_node(state: ClinicalState) -> dict[str, Any]:
    """Apply provenance and presence contracts; do not invent semantic scores."""

    usable: list[dict[str, Any]] = []
    source_ids: list[str] = []
    for context in state.get("vector_contexts") or []:
        text = str(context.get("text") or context.get("content") or "").strip()
        source_id = str(
            context.get("source_id")
            or context.get("source_path")
            or context.get("source_file")
            or context.get("document_id")
            or ""
        ).strip()
        if text and source_id:
            usable.append(context)
            source_ids.append(source_id)

    evidence_usable = bool(usable)
    reason = "provenance_complete_evidence_available" if evidence_usable else "no_provenance_complete_evidence"
    return {
        "evidence_assessment": {
            "usable": evidence_usable,
            "assessment_kind": "provenance_complete_evidence_presence",
            "reason": reason,
            "usable_items": len(usable),
            "source_ids": list(dict.fromkeys(source_ids)),
            "attempt": state.get("retrieval_attempt", 0),
            "max_attempts": MAX_RETRIEVAL_ATTEMPTS,
        }
    }


def _next_retrieval_query(state: ClinicalState) -> tuple[str, str] | None:
    """Return a justified next query; never spend an attempt without a reason."""

    attempt = int(state.get("retrieval_attempt", 0) or 0)
    if attempt >= MAX_RETRIEVAL_ATTEMPTS:
        return None
    standalone = str(state.get("standalone_question") or state.get("user_question") or "").strip()
    if attempt == 0:
        return (standalone, "initial_retrieval") if standalone else None

    status = str(state.get("retrieval_status") or "")
    if status in {"failed", "recoverable_error"}:
        return (standalone, "retry_transient_retrieval_failure") if standalone else None

    original = str(state.get("user_question") or "").strip()
    previous_queries = {
        str(entry.get("query") or "").strip()
        for entry in state.get("retry_history") or []
        if isinstance(entry, dict)
    }
    if status == "no_evidence" and original and original != standalone and original not in previous_queries:
        return original, "retry_with_materially_distinct_original_query"
    return None


async def generate_node(state: ClinicalState) -> dict[str, Any]:
    """Generate once from packed source evidence, then validate provider output."""

    generated = await generate_answer_node(state)
    merged = {**state, **generated}
    decision = await generation_fallback_decision_node(merged)
    merged.update(decision)
    if decision.get("fallback_applied"):
        return {**generated, **decision, **(await safe_fallback_node(merged))}
    return {**generated, **decision}


async def abstain_node(state: ClinicalState) -> dict[str, Any]:
    """Produce an explicit safe abstention after the bounded retrieval budget."""

    retrieval_error = state.get("retrieval_error")
    reason = retrieval_error or "No provenance-complete source evidence after bounded retrieval."
    fallback_state = {
        **state,
        "fallback_applied": True,
        "fallback_type": "retrieval_error" if retrieval_error else "no_retrieval_evidence",
        "fallback_reason": reason,
        "fallback_cache_eligible": False,
    }
    return await safe_fallback_node(fallback_state)


async def finalize_node(state: ClinicalState) -> dict[str, Any]:
    """Apply presentation, final safety/quality, cache, and observability contracts."""

    updates = await finalize_response_node(state)
    quality = await answer_quality_node({**state, **updates})
    updates.update(quality)
    cache = await cache_store_node({**state, **updates})
    updates.update(cache)
    updates.update(await observability_export_node({**state, **updates}))
    return updates


__all__ = [
    "MAX_RETRIEVAL_ATTEMPTS",
    "abstain_node",
    "assess_evidence_node",
    "decide_node",
    "finalize_node",
    "generate_node",
    "guard_node",
    "prepare_node",
    "retrieve_node",
]
