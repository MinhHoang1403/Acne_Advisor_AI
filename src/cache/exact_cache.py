"""Exact normalized Redis answer cache with versioned provenance identity."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import re
from typing import Any

from src.agent.text_encoding import repair_mojibake
from src.cache.redis_cache import get_redis
from src.observability.versioning import (
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
    get_answer_cache_version,
)
from src.quality.safe_fallback import sanitize_fallback_reason

logger = logging.getLogger(__name__)
CACHE_SCHEMA_VERSION = os.getenv("CACHE_SCHEMA_VERSION", "v3")


def normalize_question(text: str) -> str:
    """Normalize case, punctuation, and whitespace for exact matching."""

    normalized = re.sub(r"[?!.,:;]+", " ", str(text or "").casefold())
    return " ".join(normalized.split())


def make_cache_key(
    normalized_question: str,
    *,
    provider: str,
    model: str,
    pipeline_fingerprint: str | None = None,
) -> str:
    """Build a key partitioned by exact question and runtime identity."""

    fingerprint = pipeline_fingerprint or compute_pipeline_fingerprint(
        build_pipeline_version_manifest()
    )
    payload = "|".join(
        (
            CACHE_SCHEMA_VERSION,
            get_answer_cache_version(),
            fingerprint,
            normalized_question,
            provider,
            model,
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"cache:answer:{CACHE_SCHEMA_VERSION}:{get_answer_cache_version()}:{fingerprint}:{digest}"


async def get_exact_cache(
    normalized_question: str,
    *,
    provider: str,
    model: str,
    pipeline_fingerprint: str | None = None,
) -> dict[str, Any] | None:
    redis = await get_redis()
    if not redis:
        return None
    key = make_cache_key(
        normalized_question,
        provider=provider,
        model=model,
        pipeline_fingerprint=pipeline_fingerprint,
    )
    try:
        value = await redis.get(key)
        if not value:
            return None
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            return None
        if isinstance(parsed.get("answer"), str):
            parsed["answer"] = repair_mojibake(parsed["answer"])
        return parsed
    except Exception as exc:
        logger.debug("Exact cache read skipped: %s", sanitize_fallback_reason(exc))
        return None


async def set_answer_cache(
    normalized_question: str,
    answer: str,
    sources: list[str],
    metadata: dict[str, Any],
    *,
    provider: str,
    model: str,
    pipeline_fingerprint: str | None = None,
) -> None:
    redis = await get_redis()
    if not redis:
        return
    fingerprint = pipeline_fingerprint or compute_pipeline_fingerprint(
        build_pipeline_version_manifest()
    )
    key = make_cache_key(
        normalized_question,
        provider=provider,
        model=model,
        pipeline_fingerprint=fingerprint,
    )
    data = {
        "normalized_question": normalized_question,
        "answer": repair_mojibake(answer),
        "sources": sources,
        "metadata": metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "answer_version": get_answer_cache_version(),
        "pipeline_fingerprint": fingerprint,
        "model_provider": metadata.get("provider"),
        "model_name": metadata.get("model"),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
    }
    try:
        ttl = max(1, int(os.getenv("CACHE_TTL_SECONDS", "86400")))
        await redis.setex(key, ttl, json.dumps(data, ensure_ascii=False))
    except Exception as exc:
        logger.warning("Exact cache write skipped: %s", sanitize_fallback_reason(exc))


__all__ = ["get_exact_cache", "make_cache_key", "normalize_question", "set_answer_cache"]
