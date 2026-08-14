"""
src/agent/nodes/cache.py
========================
LangGraph nodes for Exact Normalized Answer Caching.
"""

import logging
import os
from typing import Any

from src.agent.state import ClinicalState
from src.agent.text_encoding import repair_mojibake
from src.cache.semantic_cache import (
    normalize_question,
    is_cacheable_question,
    infer_cache_intent,
    get_exact_cache,
    set_answer_cache
)
from src.observability.versioning import (
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
    get_answer_cache_version,
    pipeline_manifest_summary,
)

logger = logging.getLogger(__name__)


def _resolve_cache_model_key(state: ClinicalState) -> tuple[str, str]:
    """Resolve the provider/model pair used in Redis cache keys."""
    provider = (state.get("llm_provider") or "gemini").lower()
    model = state.get("llm_model")

    if provider == "gemini":
        resolved = model or os.getenv("GOOGLE_MODEL", "gemini-3.5-flash")
        if resolved == "gemini-1.5-flash":
            resolved = "gemini-3.5-flash"
        return provider, resolved

    if provider in {"ollama", "local"}:
        resolved = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        if ":" not in resolved:
            resolved = f"{resolved}:latest"
        return "ollama", resolved

    return provider, model or "unknown"

async def cache_lookup_node(state: ClinicalState) -> dict[str, Any]:
    """
    Checks if the question has a cached answer.
    Must run AFTER rewrite_question and domain_guard.
    """
    logger.info("Executing cache_lookup_node...")
    
    # Decide which question string to use for cache key
    # Use standalone_question ONLY for multi-turn follow-ups with vague references
    # Use user_question for clear, direct questions to ensure cache key consistency
    user_question = state.get("user_question", "")
    standalone_question = state.get("standalone_question")
    history = state.get("conversation_history", [])
    
    # Detect if the question is a vague follow-up (uses pronouns/references)
    vague_markers = ["loại đó", "nó ", "cái đó", "hoạt chất đó", "thuốc đó", "sản phẩm đó", "vậy ", "thì sao"]
    is_vague_followup = (
        history 
        and len(history) > 0
        and any(marker in user_question.lower() for marker in vague_markers)
    )
    
    if is_vague_followup and standalone_question and standalone_question.strip():
        target_question = standalone_question
        cache_key_source = "standalone_question"
    else:
        target_question = user_question
        cache_key_source = "user_question"
    
    logger.info(
        "Cache key source=%s, question_chars=%d",
        cache_key_source,
        len(target_question),
    )
    normalized = normalize_question(target_question)
    guard_status = "in_domain" if state.get("is_in_domain") else "out_of_domain"
    intent = infer_cache_intent(target_question, guard_status)
    
    # Check if we should bypass cache completely
    if state.get("bypass_cache"):
        logger.info("Cache BYPASSED due to request flag.")
        return {
            "cache_checked": False,
            "cache_hit": False,
            "cache_reason": "bypassed",
            "normalized_question": normalized,
            "cache_intent": intent,
        }

    severity = (state.get("medical_severity") or "routine").lower()
    if severity in {"urgent", "emergency"}:
        logger.info("Cache SKIPPED: medical_severity=%s", severity)
        return {
            "cache_checked": True,
            "cache_hit": False,
            "cache_reason": f"severity_{severity}",
            "normalized_question": normalized,
            "cache_intent": intent,
        }

    if history:
        logger.info("Cache SKIPPED: conversation history present.")
        return {
            "cache_checked": True,
            "cache_hit": False,
            "cache_reason": "history_present",
            "normalized_question": normalized,
            "cache_intent": intent,
        }
        
    # Check if cacheable
    history = state.get("conversation_history", [])
    
    is_cacheable, reason = is_cacheable_question(target_question, history, guard_status)
    
    if not is_cacheable:
        logger.info(f"Question not cacheable. Reason: {reason}")
        return {
            "cache_checked": True,
            "cache_hit": False,
            "cache_reason": reason,
            "normalized_question": normalized,
            "cache_intent": intent,
        }
        
    # Attempt to retrieve from Redis
    cache_provider, cache_model = _resolve_cache_model_key(state)
    pipeline_manifest = state.get("pipeline_manifest") or build_pipeline_version_manifest()
    pipeline_fingerprint = state.get("pipeline_fingerprint") or compute_pipeline_fingerprint(pipeline_manifest)
    cached_data = await get_exact_cache(
        normalized,
        intent=intent,
        provider=cache_provider,
        model=cache_model,
        pipeline_fingerprint=pipeline_fingerprint,
    )
    
    if cached_data:
        meta = cached_data.get("metadata", {})
        expected_version = get_answer_cache_version()
        cached_fingerprint = meta.get("pipeline_fingerprint") or cached_data.get("pipeline_fingerprint")
        
        if (
            not meta.get("quality_passed")
            or meta.get("answer_version") != expected_version
            or cached_fingerprint != pipeline_fingerprint
        ):
            logger.info("Cache entry invalid or missing quality metadata. Treating as MISS.")
            return {
                "cache_checked": True,
                "cache_hit": False,
                "cache_reason": "invalid_cache_metadata",
                "normalized_question": normalized,
                "cache_intent": intent,
            }
            
        logger.info("Cache HIT with valid quality metadata!")
        
        # We need to construct metadata properly
        
        # Let's populate the state
        cached_answer = repair_mojibake(cached_data.get("answer", ""))
        return {
            "cache_checked": True,
            "cache_hit": True,
            "cache_reason": "hit",
            "cache_intent": intent,
            "cached_answer": cached_answer,
            "cache_metadata": cached_data.get("metadata", {}),
            "normalized_question": normalized,
            "pipeline_manifest": pipeline_manifest,
            "pipeline_fingerprint": pipeline_fingerprint,
            
            # Since it's a hit, we bypass LLM generation
            "final_answer": cached_answer,
            "sources": cached_data.get("sources", []),
            "source_allowlist": meta.get("source_allowlist", []),
            "actual_provider": "cache",
            "actual_model": cached_data.get("model_name", "unknown")
        }
    else:
        logger.info("Cache MISS!")
        return {
            "cache_checked": True,
            "cache_hit": False,
            "cache_reason": "miss",
            "normalized_question": normalized,
            "cache_intent": intent,
        }


async def cache_store_node(state: ClinicalState) -> dict[str, Any]:
    """
    Stores the generated answer into the cache.
    Must run AFTER generate_answer_node.
    """
    logger.info("Executing cache_store_node...")
    
    # Only store if cache was missed and we generated a valid answer
    if state.get("cache_hit"):
        logger.info("Cache store SKIPPED: cache_hit is True")
        return {}
    
    # Skip cache store if bypass requested
    if state.get("bypass_cache"):
        logger.info("Cache store BYPASSED due to request flag.")
        return {}

    if state.get("conversation_history"):
        logger.info("Cache store SKIPPED: conversation history present.")
        return {}
        
    # Check if it was deemed cacheable in the lookup phase
    if state.get("cache_reason") not in ["miss", None]:
        logger.info(f"Cache store SKIPPED: cache_reason={state.get('cache_reason')}")
        return {}

    severity = (state.get("medical_severity") or "routine").lower()
    if severity in {"urgent", "emergency"}:
        logger.info("Cache store SKIPPED: medical_severity=%s", severity)
        return {}

    if state.get("severity_guard_modified") or state.get("severity_guard_cache_eligible") is False:
        logger.info(
            "Cache store SKIPPED: severity guard modified=%s cache_eligible=%s",
            state.get("severity_guard_modified"),
            state.get("severity_guard_cache_eligible"),
        )
        return {}

    fallback_type = state.get("fallback_type")
    retrieval_status = state.get("retrieval_status")
    if (
        state.get("fallback_applied")
        or (fallback_type and fallback_type != "none")
        or state.get("fallback_cache_eligible") is False
        or retrieval_status in {"empty_query", "no_evidence", "recoverable_error"}
    ):
        reason = f"safe_fallback_{fallback_type or retrieval_status or 'unknown'}"
        logger.info("Cache store SKIPPED: %s", reason)
        return {"cache_reason": reason}
        
    if not state.get("is_in_domain"):
        logger.info("Cache store SKIPPED: not in domain")
        return {}

    if state.get("use_history_context"):
        logger.info("Cache store SKIPPED: history context enabled")
        return {}

    if any(
        marker in (state.get("user_question") or "").lower()
        for marker in [
            "giả vờ bạn là bác sĩ",
            "giả vờ là bác sĩ",
            "pretend you are a doctor",
            "kê đơn",
            "kê thuốc",
            "toa thuốc",
        ]
    ):
        logger.info("Cache store SKIPPED: unsafe prompt/prescription request")
        return {}
        
    if state.get("llm_fallback"):
        logger.info("Cache store SKIPPED: llm_fallback=%s", state.get("llm_fallback"))
        return {}
        
    # Use final_answer (post-processed by finalize_response_node) instead of draft_answer
    answer = repair_mojibake(state.get("final_answer", ""))
    sources = state.get("sources", [])
    
    # --- QUALITY GATE ---
    if not answer:
        logger.info("Quality Gate Failed: Empty answer")
        return {}
        
    min_chars = int(os.getenv("CACHE_MIN_ANSWER_CHARS", "350"))
    if len(answer) < min_chars:
        logger.info(f"Quality Gate Failed: Answer too short ({len(answer)} < {min_chars})")
        return {}
        
    if state.get("llm_fallback") or state.get("llm_fallback_used"):
        logger.info(f"Quality Gate Failed: Answer is a fallback (llm_fallback={state.get('llm_fallback')}, llm_fallback_used={state.get('llm_fallback_used')})")
        return {}
        
    if state.get("fallback_provider") == "rule_based" or state.get("guardrail") == "in_domain_fallback":
        logger.info(f"Quality Gate Failed: rule_based/in_domain_fallback (fallback_provider={state.get('fallback_provider')}, guardrail={state.get('guardrail')})")
        return {}
        
    # Check if sources are empty but in-domain
    if not sources and state.get("is_in_domain"):
        logger.info(f"Quality Gate Failed: No sources for in-domain question. sources={sources}")
        return {}
    
    # Semantic medical validity is not inferred by cache heuristics. The cache
    # only stores responses that passed runtime structure/provenance contracts.
    
    # Use same cache key logic as cache_lookup_node
    raw_question = state.get("user_question", "")
    standalone_q = state.get("standalone_question")
    store_history = state.get("conversation_history", [])
    
    vague_markers = ["loại đó", "nó ", "cái đó", "hoạt chất đó", "thuốc đó", "sản phẩm đó", "vậy ", "thì sao"]
    is_vague_followup = (
        store_history 
        and len(store_history) > 0
        and any(marker in raw_question.lower() for marker in vague_markers)
    )
    
    if is_vague_followup and standalone_q and standalone_q.strip():
        target_question = standalone_q
    else:
        target_question = raw_question
        
    normalized = normalize_question(target_question)
    guard_status = "in_domain" if state.get("is_in_domain") else "out_of_domain"
    intent = infer_cache_intent(target_question, guard_status)
    
    # Build metadata to store
    answer_quality_report = state.get("answer_quality_report") or {}
    quality_passed = answer_quality_report.get("passed") if isinstance(answer_quality_report, dict) else None
    pipeline_manifest = state.get("pipeline_manifest") or build_pipeline_version_manifest()
    pipeline_fingerprint = state.get("pipeline_fingerprint") or compute_pipeline_fingerprint(pipeline_manifest)
    answer_cache_version = get_answer_cache_version()
    quality_issues = (
        answer_quality_report.get("issues", [])
        if isinstance(answer_quality_report, dict)
        else []
    )
    packed_context = state.get("packed_context") or {}
    packed_items = packed_context.get("items") or []
    selected_evidence = [
        {
            "item_id": item.get("item_id"),
            "source_id": (
                (item.get("payload") or {}).get("source_id")
                or (item.get("payload") or {}).get("source_path")
                or (item.get("payload") or {}).get("source_file")
                or (item.get("payload") or {}).get("document_id")
            ),
        }
        for item in packed_items
        if isinstance(item, dict) and item.get("item_id")
    ]
    metadata = {
        "provider": state.get("actual_provider"),
        "model": state.get("actual_model"),
        "retrieval": "dense_bm25_rrf" if state.get("sources") else "skipped",
        "quality_checked": True,
        "quality_passed": bool(quality_passed) if quality_passed is not None else True,
        "quality_reason": "passed_answer_verifier" if quality_passed is not False else "answer_verifier_failed",
        "answer_quality_issues": quality_issues,
        "answer_quality_summary": {
            "passed": quality_passed,
            "issue_count": len(quality_issues),
            "critical_count": sum(
                1
                for issue in quality_issues
                if isinstance(issue, dict) and issue.get("severity") == "critical"
            ),
        },
        "medical_severity": state.get("medical_severity") or "routine",
        "severity_guard": state.get("severity_guard") or {},
        "severity_guard_modified": bool(state.get("severity_guard_modified")),
        "retrieval_status": state.get("retrieval_status"),
        "fallback_applied": bool(state.get("fallback_applied")),
        "fallback_type": state.get("fallback_type") or "none",
        "fallback_reason": state.get("fallback_reason"),
        "fallback_cache_eligible": state.get("fallback_cache_eligible"),
        "answer_version": answer_cache_version,
        "answer_cache_version": answer_cache_version,
        "pipeline_fingerprint": pipeline_fingerprint,
        "pipeline_manifest": pipeline_manifest_summary(pipeline_manifest),
        "source_ids": list(dict.fromkeys(str(source) for source in sources if source)),
        "source_allowlist": list(state.get("source_allowlist") or []),
        "selected_evidence": selected_evidence,
        "selected_evidence_ids": [item["item_id"] for item in selected_evidence],
        "raw_question": raw_question,
        "standalone_question": standalone_q or "",
        "cache_key_question": target_question,
        "normalized_question": normalized,
        "cache_intent": intent,
    }
    
    # Store it
    cache_provider, cache_model = _resolve_cache_model_key(state)
    logger.info(
        "Storing answer cache: key_source=%s, normalized_chars=%d",
        "standalone" if is_vague_followup else "user",
        len(normalized),
    )
    await set_answer_cache(
        normalized_question=normalized,
        standalone_question=target_question,
        answer=answer,
        sources=sources,
        metadata=metadata,
        intent=intent,
        provider=cache_provider,
        model=cache_model,
        pipeline_fingerprint=pipeline_fingerprint,
    )
    
    return {}
