"""Các node điều phối workflow Agent từ request đến response.

Module nối các owner chuyên biệt: preparation, safety/exact cache, action model,
retrieval, evidence presence, generation, fallback, presentation, verifier và
observability. Nó không tự triển khai Dense/BM25 hay nội dung prompt y khoa.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.agent.nodes.cache import cache_lookup_node, cache_store_node
from src.agent.nodes.fallback import generation_fallback_decision_node, safe_fallback_node
from src.agent.nodes.observability import observability_export_node
from src.agent.nodes.quality import answer_quality_node
from src.agent.nodes.reason import generate_answer_node
from src.agent.nodes.respond import finalize_response_node
from src.agent.action_decision import MAX_RETRIEVAL_ATTEMPTS, select_agent_action
from src.agent.nodes.preparation import prepare_request_node
from src.agent.source_presentation import build_source_allowlist
from src.agent.safety_policy import evaluate_safety
from src.agent.state import ClinicalState
from src.quality.safe_fallback import (
    fallback_reason_code_from_agent_reason,
    fallback_type_for_reason,
    sanitize_fallback_reason,
)
from src.retrieval.service import retrieve_evidence

async def prepare_node(state: ClinicalState) -> dict[str, Any]:
    """Chuẩn hóa request và chỉ đưa conversation context hữu hạn vào state."""

    started = time.perf_counter()
    updates = await prepare_request_node(state)
    return {
        **updates,
        "performance_timings": {
            **(state.get("performance_timings") or {}),
            "prepare": round((time.perf_counter() - started) * 1000, 3),
        },
    }


async def guard_node(state: ClinicalState) -> dict[str, Any]:
    """Áp dụng safety override hẹp; nếu không khớp mới đọc exact cache."""

    started = time.perf_counter()
    question = state.get("normalized_question") or state.get("user_question") or ""
    safety = evaluate_safety(question)
    if safety is not None:
        updates = {
            "safety_override": True,
            "safety_decision": {
                "rule_id": safety.rule_id,
                "severity": safety.severity,
                "action": safety.action,
                "source_ids": list(safety.source_ids),
                "source_urls": list(safety.source_urls),
            },
            "safety_severity": safety.severity,
            "draft_answer": safety.response,
            "sources": [],
            "source_allowlist": [],
            "actual_provider": "system",
            "actual_model": None,
            "fallback_cache_eligible": False,
            "cache_reason": "deterministic_safety_override",
        }
    else:
        updates = await cache_lookup_node(state)
    return {
        **updates,
        "performance_timings": {
            **(state.get("performance_timings") or {}),
            "guard": round((time.perf_counter() - started) * 1000, 3),
        },
    }


async def decide_node(state: ClinicalState) -> dict[str, Any]:
    """Dùng một semantic decision owner cho mọi transition có lựa chọn."""

    if state.get("safety_override") or state.get("cache_hit"):
        return {"next_action": "finalize"}
    updates = await select_agent_action(state)
    decision = updates.get("agent_decision") or {}
    attempts_used = int(state.get("retrieval_attempt", 0) or 0)
    action = str(decision.get("action") or updates.get("next_action") or "abstain")
    decision_history = list(state.get("agent_decision_history") or [])
    evidence_traces = list(state.get("agent_decision_evidence_traces") or [])
    evidence_trace = decision.get("evidence_trace")
    if attempts_used > 0 and isinstance(evidence_trace, dict):
        evidence_traces.append(
            {
                "decision_index": len(decision_history) + 1,
                "retrieval_attempts_used": attempts_used,
                **evidence_trace,
                "action": action,
                "reason_code": decision.get("reason_code"),
                "retrieval_query": decision.get("retrieval_query"),
                "provider": decision.get("provider"),
                "model": decision.get("model"),
                "requested_provider": decision.get("requested_provider"),
                "requested_model": decision.get("requested_model"),
                "provider_fallback_attempted": bool(decision.get("fallback_chain")),
                "provider_fallback_used": bool(decision.get("fallback_used")),
                "fallback_reason_code": updates.get("fallback_reason_code"),
            }
        )
    return {
        **updates,
        "agent_decision_history": [
            *decision_history,
            {
                "action": action,
                "reason_code": decision.get("reason_code"),
                "retrieval_query": decision.get("retrieval_query"),
                "retrieval_executions_used": attempts_used,
                "remaining_retrieval_budget": max(0, MAX_RETRIEVAL_ATTEMPTS - attempts_used),
                "retry_requested": action == "retry",
                "abstain": action == "abstain",
            },
        ],
        "agent_decision_evidence_traces": evidence_traces,
    }


async def retrieve_node(state: ClinicalState) -> dict[str, Any]:
    """Gọi evidence tool duy nhất và bảo toàn typed trace của lần retrieval.

    Cả ``retrieve`` và ``retry`` đi qua hàm này. Số lần thử được tăng trước khi
    gọi service và lưu cùng query/status/selected IDs để decision node kiểm soát
    vòng lặp, kể cả khi provider hoặc Qdrant trả lỗi.
    """

    attempt = state.get("retrieval_attempt", 0) + 1
    decision = state.get("agent_decision") or {}
    question = str(
        decision.get("retrieval_query")
        or state.get("retrieval_query")
        or state.get("standalone_question")
        or state.get("normalized_question")
        or ""
    ).strip()
    retry_reason = str(decision.get("reason_code") or "needs_evidence")

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
        attempt_trace = _retrieval_attempt_trace(
            attempt=attempt,
            decision=decision,
            current_question=state.get("normalized_question") or state.get("user_question") or "",
            retrieval_query=question,
            status=status,
            trace=trace,
            packed_context=metadata.get("packed_context"),
        )
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
            "retrieval_attempt_traces": [
                *(state.get("retrieval_attempt_traces") or []),
                attempt_trace,
            ],
            "performance_timings": {
                **(state.get("performance_timings") or {}),
                f"retrieval_attempt_{attempt}": round((time.perf_counter() - started) * 1000, 3),
            },
        }
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error = sanitize_fallback_reason(exc)
        attempt_trace = _retrieval_attempt_trace(
            attempt=attempt,
            decision=decision,
            current_question=state.get("normalized_question") or state.get("user_question") or "",
            retrieval_query=question,
            status="failed",
            trace={"architecture": "dense_bm25_rrf", "error": error},
            packed_context=None,
        )
        return {
            "retrieval_attempt": attempt,
            "vector_contexts": [],
            "sources": [],
            "retrieval_status": "failed",
            "retrieval_error": error,
            "fallback_reason_code": "retrieval_unavailable",
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
            "retrieval_attempt_traces": [
                *(state.get("retrieval_attempt_traces") or []),
                attempt_trace,
            ],
        }


def _retrieval_attempt_trace(
    *,
    attempt: int,
    decision: dict[str, Any],
    current_question: str,
    retrieval_query: str,
    status: str,
    trace: dict[str, Any],
    packed_context: Any,
) -> dict[str, Any]:
    """Preserve one bounded retrieval execution for internal diagnostics."""

    packed = packed_context if isinstance(packed_context, dict) else {}
    return {
        "attempt_index": attempt,
        "initiating_action": decision.get("action"),
        "initiating_reason": decision.get("reason_code"),
        "current_question": str(current_question),
        "retrieval_query": retrieval_query,
        "normalized_retrieval_query": trace.get("query") or retrieval_query,
        "status": status,
        "channels": dict(trace.get("channels") or {}),
        "candidate_trace": dict(trace.get("candidate_trace") or {}),
        "packed_evidence": _evidence_identity_trace(packed.get("items") or []),
        "packer": dict(trace.get("packer") or {}),
        "warnings": list(trace.get("warnings") or []),
        "error": trace.get("error"),
    }


def _evidence_identity_trace(items: list[Any]) -> list[dict[str, Any]]:
    """Return only provenance identifiers used to build a packed evidence block."""

    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else item
        section_path = payload.get("section_path")
        section = payload.get("header") or (
            section_path[-1] if isinstance(section_path, list) and section_path else None
        )
        result.append(
            {
                "item_id": item.get("item_id") or item.get("id") or payload.get("chunk_id"),
                "source_id": payload.get("source_id")
                or payload.get("source_path")
                or payload.get("source_file")
                or payload.get("document_id"),
                "section": section,
            }
        )
    return result


async def assess_evidence_node(state: ClinicalState) -> dict[str, Any]:
    """Kiểm tra evidence có text + source identity, không chấm semantic quality.

    ``usable=True`` chỉ có nghĩa item có nội dung và provenance tối thiểu. Nó
    không chứng minh source trả lời đủ câu hỏi, không xác nhận claim và không
    đánh giá medical truth; model action selector xử lý mức liên quan ngữ nghĩa.
    """

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


async def generate_node(state: ClinicalState) -> dict[str, Any]:
    """Sinh một lần từ packed evidence rồi quyết định fallback cho provider output."""

    generated = await generate_answer_node(state)
    generation_metadata = {
        "generation_invoked": True,
        "generation_provider": generated.get("actual_provider"),
        "generation_model": generated.get("actual_model"),
    }
    merged = {**state, **generated}
    decision = await generation_fallback_decision_node(merged)
    merged.update(decision)
    if decision.get("fallback_applied"):
        return {
            **generated,
            **decision,
            **(await safe_fallback_node(merged)),
            **generation_metadata,
        }
    return {**generated, **decision, **generation_metadata}


async def abstain_node(state: ClinicalState) -> dict[str, Any]:
    """Tạo safe abstention rõ ràng khi retrieval budget không cho đủ evidence."""

    retrieval_error = state.get("retrieval_error")
    decision_reason = str((state.get("agent_decision") or {}).get("reason_code") or "")
    reason_code = state.get("fallback_reason_code") or (
        "retrieval_unavailable"
        if retrieval_error
        else fallback_reason_code_from_agent_reason(decision_reason)
    )
    fallback_type = fallback_type_for_reason(reason_code)  # type: ignore[arg-type]
    reason = retrieval_error or {
        "provider_unavailable": "The configured action provider was unavailable.",
        "out_of_scope": "The request is outside the supported acne-information scope.",
        "insufficient_evidence": "Bounded retrieval did not establish sufficient evidence.",
        "cannot_safely_proceed": "The bounded Agent could not establish a legal safe transition.",
        "retrieval_unavailable": "The retrieval service was unavailable.",
        "generation_unavailable": "The generation provider was unavailable.",
    }[reason_code]
    fallback_state = {
        **state,
        "fallback_applied": True,
        "fallback_type": fallback_type,
        "fallback_reason": reason,
        "fallback_reason_code": reason_code,
        "fallback_cache_eligible": False,
    }
    return await safe_fallback_node(fallback_state)


async def finalize_node(state: ClinicalState) -> dict[str, Any]:
    """Áp dụng presentation, verifier, exact cache và observability theo thứ tự."""

    started = time.perf_counter()
    updates = await finalize_response_node(state)
    quality = await answer_quality_node({**state, **updates})
    updates.update(quality)
    cache = await cache_store_node({**state, **updates})
    updates.update(cache)
    updates.update(await observability_export_node({**state, **updates}))
    updates["performance_timings"] = {
        **(state.get("performance_timings") or {}),
        **(updates.get("performance_timings") or {}),
        "finalize": round((time.perf_counter() - started) * 1000, 3),
    }
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
