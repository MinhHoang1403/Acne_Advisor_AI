"""LangGraph nodes for exact normalized answer caching."""

from __future__ import annotations

import os
from typing import Any

from src.agent.state import ClinicalState
from src.agent.text_encoding import repair_mojibake
from src.cache.exact_cache import get_exact_cache, normalize_question, set_answer_cache
from src.observability.versioning import (
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
    get_answer_cache_version,
    pipeline_manifest_summary,
)


def _resolve_cache_model_key(state: ClinicalState) -> tuple[str, str]:
    provider = (state.get("llm_provider") or os.getenv("LLM_PROVIDER", "gemini")).lower()
    model = state.get("llm_model")
    if provider == "gemini":
        return provider, model or os.getenv("GOOGLE_MODEL", "gemini-3.5-flash")
    if provider in {"ollama", "local"}:
        resolved = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        return "ollama", resolved if ":" in resolved else f"{resolved}:latest"
    return provider, model or "unknown"


async def cache_lookup_node(state: ClinicalState) -> dict[str, Any]:
    question = state.get("normalized_question") or state.get("user_question") or ""
    normalized = normalize_question(question)
    base = {"normalized_question": normalized, "cache_hit": False}
    if state.get("bypass_cache"):
        return {**base, "cache_checked": False, "cache_reason": "bypassed"}
    if state.get("safety_override"):
        return {**base, "cache_checked": True, "cache_reason": "safety_override"}
    if (state.get("conversation_context") or {}).get("message_count"):
        return {**base, "cache_checked": True, "cache_reason": "history_present"}
    max_chars = max(1, int(os.getenv("CACHE_MAX_QUESTION_CHARS", "600")))
    if not normalized or len(question) > max_chars:
        return {**base, "cache_checked": True, "cache_reason": "input_not_cacheable"}

    provider, model = _resolve_cache_model_key(state)
    manifest = state.get("pipeline_manifest") or build_pipeline_version_manifest()
    fingerprint = state.get("pipeline_fingerprint") or compute_pipeline_fingerprint(manifest)
    cached = await get_exact_cache(
        normalized,
        provider=provider,
        model=model,
        pipeline_fingerprint=fingerprint,
    )
    if not cached:
        return {**base, "cache_checked": True, "cache_reason": "miss"}

    metadata = cached.get("metadata") or {}
    if (
        metadata.get("answer_version") != get_answer_cache_version()
        or metadata.get("pipeline_fingerprint") != fingerprint
        or not metadata.get("selected_evidence_ids")
    ):
        return {**base, "cache_checked": True, "cache_reason": "invalid_cache_metadata"}

    answer = repair_mojibake(cached.get("answer", ""))
    return {
        **base,
        "cache_checked": True,
        "cache_hit": True,
        "cache_reason": "hit",
        "cached_answer": answer,
        "cache_metadata": metadata,
        "final_answer": answer,
        "sources": cached.get("sources", []),
        "source_allowlist": metadata.get("source_allowlist", []),
        "actual_provider": "cache",
        "actual_model": cached.get("model_name"),
    }


async def cache_store_node(state: ClinicalState) -> dict[str, Any]:
    if state.get("cache_hit") or state.get("bypass_cache") or state.get("safety_override"):
        return {}
    if (state.get("conversation_context") or {}).get("message_count"):
        return {}
    if state.get("cache_reason") not in {None, "miss"}:
        return {}
    if state.get("fallback_applied") or state.get("fallback_cache_eligible") is False:
        return {}
    if state.get("llm_fallback_used"):
        return {}
    if state.get("retrieval_status") not in {"ok", "degraded_dense", "degraded_bm25"}:
        return {}

    answer = repair_mojibake(state.get("final_answer", ""))
    sources = [str(value) for value in state.get("sources") or [] if value]
    if not answer or not sources:
        return {}
    report = state.get("answer_quality_report") or {}
    if isinstance(report, dict) and report.get("passed") is False:
        return {}

    packed = state.get("packed_context") or {}
    selected_evidence = []
    for item in packed.get("items") or []:
        if not isinstance(item, dict) or not item.get("item_id"):
            continue
        payload = item.get("payload") or {}
        selected_evidence.append(
            {
                "item_id": item["item_id"],
                "source_id": payload.get("source_id")
                or payload.get("source_path")
                or payload.get("source_file")
                or payload.get("document_id"),
            }
        )
    if not selected_evidence:
        return {}

    normalized = normalize_question(state.get("normalized_question") or state.get("user_question") or "")
    manifest = state.get("pipeline_manifest") or build_pipeline_version_manifest()
    fingerprint = state.get("pipeline_fingerprint") or compute_pipeline_fingerprint(manifest)
    provider, model = _resolve_cache_model_key(state)
    version = get_answer_cache_version()
    metadata = {
        "provider": state.get("actual_provider"),
        "model": state.get("actual_model"),
        "answer_version": version,
        "pipeline_fingerprint": fingerprint,
        "pipeline_manifest": pipeline_manifest_summary(manifest),
        "source_ids": sources,
        "source_allowlist": list(state.get("source_allowlist") or []),
        "selected_evidence": selected_evidence,
        "selected_evidence_ids": [item["item_id"] for item in selected_evidence],
    }
    await set_answer_cache(
        normalized,
        answer,
        sources,
        metadata,
        provider=provider,
        model=model,
        pipeline_fingerprint=fingerprint,
    )
    return {}


__all__ = ["cache_lookup_node", "cache_store_node"]
