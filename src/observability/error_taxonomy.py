"""Error taxonomy và component ownership cho observability.

Phân loại lỗi runtime thành taxonomy nhỏ, có subsystem owner cụ thể.
Không dùng tên cá nhân hay incident framework; owner là subsystem/component kỹ thuật.
"""

from __future__ import annotations

from typing import Literal

ErrorFamily = Literal[
    "api_validation",
    "api_internal",
    "agent_timeout",
    "decision_provider",
    "decision_validation",
    "dense_channel",
    "dense_embedding",
    "dense_store",
    "bm25_store",
    "retrieval",
    "reranker",
    "context_packing",
    "generation_provider",
    "generation_timeout",
    "source_validation",
    "quality_validation",
    "cache",
    "persistence",
    "observability",
    "unknown",
]

ERROR_FAMILY_OWNERS: dict[str, str] = {
    "api_validation": "api",
    "api_internal": "api",
    "agent_timeout": "agent",
    "decision_provider": "agent/decision",
    "decision_validation": "agent/decision",
    "dense_channel": "retrieval",
    "dense_embedding": "retrieval/integrations",
    "dense_store": "retrieval/qdrant",
    "bm25_store": "retrieval/bm25",
    "retrieval": "retrieval",
    "reranker": "retrieval/reranker",
    "context_packing": "retrieval/context_packer",
    "generation_provider": "generation",
    "generation_timeout": "generation",
    "source_validation": "agent/source_presentation",
    "quality_validation": "quality/answer_verifier",
    "cache": "cache",
    "persistence": "database",
    "observability": "observability",
    "unknown": "unknown",
}


def classify_error(
    exc: BaseException | str | None,
    stage: str | None = None,
) -> tuple[str, str]:
    """Classify from an existing error code and an explicit component boundary.

    Error message content is deliberately not parsed. It may contain private
    data and is not a stable operational contract.
    """

    if exc is None:
        return "unknown", "unknown"

    exc_type = type(exc).__name__ if isinstance(exc, BaseException) else str(exc)
    normalized_type = exc_type.casefold()
    error_code = str(getattr(exc, "error_code", "") or "").casefold()
    normalized_stage = str(stage or "").casefold()

    # Agent timeout is a request-wide condition regardless of the last stage.
    if error_code == "agent_timeout" or "agenttimeout" in normalized_type:
        return "agent_timeout", ERROR_FAMILY_OWNERS["agent_timeout"]

    stage_families: dict[str, str] = {
        "api_validation": "api_validation",
        "api": "api_internal",
        "api_internal": "api_internal",
        "decide": "decision_provider",
        "agent_decision": "decision_provider",
        "decision_provider": "decision_provider",
        "decision_validation": "decision_validation",
        "dense_embedding": "dense_embedding",
        "dense": "dense_channel",
        "dense_channel": "dense_channel",
        "dense_store": "dense_store",
        "bm25": "bm25_store",
        "bm25_store": "bm25_store",
        "retrieve": "retrieval",
        "retrieval": "retrieval",
        "rerank": "reranker",
        "reranker": "reranker",
        "pack": "context_packing",
        "context_packing": "context_packing",
        "generate": "generation_provider",
        "llm_generation": "generation_provider",
        "generation_provider": "generation_provider",
        "generation_timeout": "generation_timeout",
        "source_validation": "source_validation",
        "quality": "quality_validation",
        "quality_validation": "quality_validation",
        "cache": "cache",
        "cache_store": "cache",
        "cache_read": "cache",
        "persist": "persistence",
        "persistence": "persistence",
        "db": "persistence",
        "observe": "observability",
        "observability": "observability",
        "telemetry": "observability",
    }
    family = stage_families.get(normalized_stage)
    if family:
        if family == "generation_provider" and (
            error_code in {"provider_timeout", "stage_timeout"}
            or "timeout" in normalized_type
        ):
            family = "generation_timeout"
        return family, ERROR_FAMILY_OWNERS[family]

    # Provider errors without a more precise stage retain a useful broad family.
    if error_code == "provider_timeout" or "providertimeout" in normalized_type:
        return "generation_timeout", ERROR_FAMILY_OWNERS["generation_timeout"]
    if error_code in {
        "provider_unavailable",
        "permanent_provider_error",
        "provider_quota_exhausted",
    }:
        return "generation_provider", ERROR_FAMILY_OWNERS["generation_provider"]

    return "unknown", "unknown"


__all__ = ["ErrorFamily", "ERROR_FAMILY_OWNERS", "classify_error"]
