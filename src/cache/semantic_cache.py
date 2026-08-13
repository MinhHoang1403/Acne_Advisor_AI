"""
src/cache/semantic_cache.py
===========================
Logic for Exact Normalized Cache and safety validations.
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

from src.agent.text_encoding import repair_mojibake
from src.cache.redis_cache import get_redis
from src.observability.versioning import (
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
    get_answer_cache_version,
)
from src.quality.safe_fallback import sanitize_fallback_reason

CACHE_SCHEMA_VERSION = os.getenv("CACHE_SCHEMA_VERSION", "v3")
CACHE_PROMPT_VERSION = os.getenv("CACHE_PROMPT_VERSION", "medical_prompt_v2")
logger = logging.getLogger(__name__)


def infer_cache_intent(question: str, guardrail_status: str) -> str:
    """Infer a coarse intent label for cache partitioning."""
    q = (question or "").lower()
    if guardrail_status != "in_domain":
        return guardrail_status or "out_of_domain"
    if any(k in q for k in ["mang thai", "có thai", "có bầu", "cho con bú"]):
        return "pregnancy_lactation"
    if any(k in q for k in ["isotretinoin", "retinoid", "adapalene", "adapalen", "tretinoin", "kháng sinh", "antibiotic"]):
        return "medication_safety"
    if any(k in q for k in ["uống nước chanh", "chanh", "chocolate", "kem đánh răng"]):
        return "folk_remedy"
    if any(k in q for k in ["bác sĩ", "khám", "cấp cứu", "nguy hiểm", "sưng", "khó thở"]):
        return "urgent_or_referral"
    return "general_acne"

def normalize_question(text: str) -> str:
    """Normalize the question for exact match hashing."""
    if not text:
        return ""
    # lowercase
    text = text.lower()
    # basic punctuation removal (keep vietnamese characters)
    # Just remove ? ! . , and excessive spaces
    text = re.sub(r'[?!.,:;]+', ' ', text)
    # strip and collapse whitespaces
    text = " ".join(text.split())
    return text

def is_cacheable_question(
    question: str, 
    conversation_history: list | None, 
    guardrail_status: str
) -> Tuple[bool, str]:
    """
    Determine if a question should be cached.
    Returns (is_cacheable, reason)
    """
    # 1. Must be in domain
    if guardrail_status != "in_domain":
        return False, "out_of_domain"
        
    question_lower = question.lower()
    
    # 2. Prevent emergency/urgent
    urgent_keywords = ["cấp cứu", "khó thở", "sưng phù", "ngất", "nhập viện", "sốc phản vệ"]
    if any(k in question_lower for k in urgent_keywords):
        return False, "possible_emergency"

    high_risk_med_keywords = [
        "isotretinoin",
        "retinoid",
        "tretinoin",
        "adapalene",
        "adapalen",
        "kháng sinh",
        "antibiotic",
        "uống thuốc gì",
        "dùng thuốc gì",
        "kê thuốc",
        "kê đơn",
        "liều cao",
    ]
    if any(k in question_lower for k in high_risk_med_keywords):
        return False, "high_risk_medication_context"
        
    # 3. Prevent highly personal/specific contexts
    personal_keywords = [
        "mang thai", "có thai", "bầu", "cho con bú", "trẻ em", "em bé", "con tôi", "bé nhà",
        "dị ứng", "dị ứng với", "bệnh nền", "tiểu đường", "huyết áp", "gan", "thận",
        "đang dùng thuốc", "liều dùng", "toa thuốc", "đơn thuốc", "của tôi", "tôi đang"
    ]
    if any(k in question_lower for k in personal_keywords):
        return False, "contains_personal_context"

    # 3b. Do not cache prompt-injection / prescription-style queries.
    unsafe_prompt_keywords = [
        "giả vờ bạn là bác sĩ",
        "giả vờ là bác sĩ",
        "pretend you are a doctor",
        "kê đơn",
        "kê thuốc",
        "toa thuốc",
        "cho tôi đơn",
    ]
    if any(k in question_lower for k in unsafe_prompt_keywords):
        return False, "unsafe_prompt_or_prescription_request"
        
    # 4. Prevent queries with PII-like patterns (email, phone, etc - basic check)
    if re.search(r'\d{8,}', question):  # numbers with 8+ digits could be phone/id
        return False, "too_specific"
        
    # 5. Check history for strong personal context
    if conversation_history:
        for msg in conversation_history:
            msg_lower = msg.get("content", "").lower()
            if any(k in msg_lower for k in ["mang thai", "đang dùng thuốc", "dị ứng"]):
                return False, "history_contains_personal_context"
                
    # 6. Question length limits
    max_chars = int(os.getenv("CACHE_MAX_QUESTION_CHARS", "600"))
    if len(question) > max_chars:
        return False, "too_long"
        
    return True, "cacheable"

def make_cache_key(
    normalized_question: str,
    *,
    intent: str = "general_acne",
    provider: str = "unknown",
    model: str = "unknown",
    pipeline_fingerprint: str | None = None,
) -> str:
    """
    Generate a Redis cache key based on the normalized question and versions.
    Format: cache:answer:{schema_version}:{answer_version}:{pipeline_fingerprint}:{hash}
    """
    answer_version = get_answer_cache_version()
    prompt_version = os.getenv("PROMPT_VERSION", CACHE_PROMPT_VERSION)
    kb_version = os.getenv("KB_VERSION", "frozen_phase1_build")
    pipeline_fingerprint = pipeline_fingerprint or compute_pipeline_fingerprint(
        build_pipeline_version_manifest()
    )
    
    # Create deterministic payload
    payload = "|".join([
        CACHE_SCHEMA_VERSION,
        answer_version,
        pipeline_fingerprint,
        normalized_question,
        intent,
        provider,
        model,
        prompt_version,
        kb_version,
    ])
    hash_obj = hashlib.sha256(payload.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    
    return f"cache:answer:{CACHE_SCHEMA_VERSION}:{answer_version}:{pipeline_fingerprint}:{hash_hex}"

async def get_exact_cache(
    normalized_question: str,
    *,
    intent: str = "general_acne",
    provider: str = "unknown",
    model: str = "unknown",
    pipeline_fingerprint: str | None = None,
) -> Optional[dict[str, Any]]:
    """Retrieve exact match cache from Redis."""
    redis = await get_redis()
    if not redis:
        return None

    pipeline_fingerprint = pipeline_fingerprint or compute_pipeline_fingerprint(
        build_pipeline_version_manifest()
    )
    cache_key = make_cache_key(
        normalized_question,
        intent=intent,
        provider=provider,
        model=model,
        pipeline_fingerprint=pipeline_fingerprint,
    )
    try:
        cached_data = await redis.get(cache_key)
        if cached_data:
            parsed = json.loads(cached_data)
            if not isinstance(parsed, dict):
                return None
            if isinstance(parsed.get("answer"), str):
                parsed["answer"] = repair_mojibake(parsed["answer"])
            return parsed
    except Exception as exc:
        logger.debug("Cache lookup skipped after Redis decode/read error: %s", sanitize_fallback_reason(exc))
    return None

async def set_answer_cache(
    normalized_question: str,
    standalone_question: str,
    answer: str,
    sources: list[str],
    metadata: dict[str, Any],
    *,
    intent: str = "general_acne",
    provider: str = "unknown",
    model: str = "unknown",
    pipeline_fingerprint: str | None = None,
):
    """Store the answer in Redis."""
    redis = await get_redis()
    if not redis:
        return

    pipeline_fingerprint = pipeline_fingerprint or compute_pipeline_fingerprint(
        build_pipeline_version_manifest()
    )
    cache_key = make_cache_key(
        normalized_question,
        intent=intent,
        provider=provider,
        model=model,
        pipeline_fingerprint=pipeline_fingerprint,
    )
    ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
    
    data = {
        "normalized_question": normalized_question,
        "standalone_question": standalone_question,
        "answer": repair_mojibake(answer),
        "sources": sources,
        "metadata": metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "answer_version": get_answer_cache_version(),
        "answer_cache_version": get_answer_cache_version(),
        "pipeline_fingerprint": pipeline_fingerprint,
        "model_provider": metadata.get("provider"),
        "model_name": metadata.get("model"),
        "retrieval_method": metadata.get("retrieval"),
        "cache_intent": intent,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "safety_disclaimer": True
    }
    
    try:
        await redis.setex(cache_key, ttl_seconds, json.dumps(data, ensure_ascii=False))
        logger.info("Successfully cached answer for key=%s", cache_key)
    except Exception as e:
        logger.error("Failed to cache answer: %s", sanitize_fallback_reason(e))
        pass
