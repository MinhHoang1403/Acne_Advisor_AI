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
from src.quality.safe_fallback import sanitize_fallback_reason
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
    return await select_agent_action(state)


async def retrieve_node(state: ClinicalState) -> dict[str, Any]:
    """Gọi evidence tool duy nhất và bảo toàn typed trace của lần retrieval.

    Cả ``retrieve`` và ``retry`` đi qua hàm này. Số lần thử được tăng trước khi
    gọi service và lưu cùng query/status/selected IDs để decision node kiểm soát
    vòng lặp, kể cả khi provider hoặc Qdrant trả lỗi.
    """

    attempt = state.get("retrieval_attempt", 0) + 1
    decision = state.get("agent_decision") or {}
    question = str(decision.get("retrieval_query") or state.get("standalone_question") or "").strip()
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
    merged = {**state, **generated}
    decision = await generation_fallback_decision_node(merged)
    merged.update(decision)
    if decision.get("fallback_applied"):
        return {**generated, **decision, **(await safe_fallback_node(merged))}
    return {**generated, **decision}


async def abstain_node(state: ClinicalState) -> dict[str, Any]:
    """Tạo safe abstention rõ ràng khi retrieval budget không cho đủ evidence."""

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
