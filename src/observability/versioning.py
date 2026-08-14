"""Deterministic, secret-free manifest for the frozen runtime."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping

DEFAULT_ANSWER_CACHE_VERSION = "v6"
DEFAULT_ANSWER_FORMATTING_CONTRACT_VERSION = "answer_formatting_contract_v11"
LEGACY_ANSWER_CACHE_VERSIONS = {"v1", "v2", "v3", "v4", "v5"}
LEGACY_ANSWER_FORMATTING_CONTRACT_VERSIONS = {
    f"answer_formatting_contract_v{version}" for version in range(1, 11)
}
ARCHITECTURE_VERSION = "s4b_final_agentic_rag_v1"
ARCHITECTURE_FROZEN = True

_SECRET_KEY_MARKERS = ("api_key", "token", "password", "secret", "authorization", "bearer", "cookie")


def build_pipeline_version_manifest(settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the stable production contract that participates in cache identity."""

    settings = settings or {}

    def value(name: str, default: Any = "") -> Any:
        return settings[name] if name in settings else os.getenv(name, default)

    manifest = {
        "phase": "s4b",
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_frozen": ARCHITECTURE_FROZEN,
        "orchestrator": "langgraph",
        "retrieval_architecture": "dense_bm25_rrf",
        "dense_vector_name": "dense",
        "bm25_vector_name": "bm25",
        "rrf_k": 60,
        "rrf_dense_weight": 1.0,
        "rrf_bm25_weight": 1.0,
        "context_packer_version": "bounded_provenance_packer_v1",
        "retrieval_candidate_limit": _env_int(value("RETRIEVAL_CANDIDATE_LIMIT", "16"), 16),
        "retrieval_context_max_items": _env_int(value("RETRIEVAL_CONTEXT_MAX_ITEMS", "8"), 8),
        "retrieval_context_max_chars": _env_int(value("RETRIEVAL_CONTEXT_MAX_CHARS", "6000"), 6000),
        "max_retrieval_attempts": 2,
        "evidence_contract_version": "source_presence_provenance_v1",
        "answer_verifier_version": value("ANSWER_VERIFIER_VERSION", "answer_verifier_v2"),
        "answer_formatting_contract_version": _effective_answer_formatting_contract_version(
            value(
                "ANSWER_FORMATTING_CONTRACT_VERSION",
                DEFAULT_ANSWER_FORMATTING_CONTRACT_VERSION,
            )
        ),
        "severity_guard_version": value("SEVERITY_GUARD_VERSION", "severity_aware_answer_guard_v1"),
        "safe_fallback_flow_version": value("SAFE_FALLBACK_FLOW_VERSION", "safe_fallback_flow_v1"),
        "runtime_resilience_version": value("RUNTIME_RESILIENCE_VERSION", "runtime_resilience_v1"),
        "llm_fallback_policy_version": value("LLM_FALLBACK_POLICY_VERSION", "llm_fallback_policy_v2"),
        "google_genai_sdk_version": value("GOOGLE_GENAI_SDK_VERSION", "google_genai_sdk_v1"),
        "google_model": value("GOOGLE_MODEL", "gemini-3.5-flash") or "gemini-3.5-flash",
        "google_fallback_models": _csv_list(value("GOOGLE_FALLBACK_MODELS", "gemini-3.1-flash-lite")),
        "ollama_model": value("OLLAMA_MODEL", "qwen3:8b") or "qwen3:8b",
        "answer_guard_mode": value("ANSWER_GUARD_MODE", "metadata_only") or "metadata_only",
        "answer_cache_version": _effective_answer_cache_version(value("CACHE_ANSWER_VERSION", None)),
        "cache_schema_version": value("CACHE_SCHEMA_VERSION", "v3"),
        "embedding_model": value("EMBEDDING_MODEL", "models/gemini-embedding-2"),
        "embedding_dimensions": _env_int(value("EMBEDDING_DIMENSIONS", "3072"), 3072),
        "qdrant_collection_name": _runtime_chunk_collection_name(settings),
        "kb_version": value("KB_VERSION", "frozen_phase1_build"),
        "prompt_version": value("PROMPT_VERSION", "medical_prompt_v2"),
        "taxonomy_version": value("TAXONOMY_VERSION", "acne_taxonomy_2026_08"),
        "neo4j_schema_version": value("NEO4J_SCHEMA_VERSION", "neo4j_schema_v1"),
        "source_normalization_version": value("SOURCE_NORMALIZATION_VERSION", "source_normalization_v1"),
        "conversation_context_version": value("CONVERSATION_CONTEXT_VERSION", "conversation_context_v1"),
        "performance_instrumentation_version": value(
            "PERFORMANCE_INSTRUMENTATION_VERSION", "performance_instrumentation_v1"
        ),
        "reproducible_environment_version": value(
            "REPRODUCIBLE_ENVIRONMENT_VERSION", "reproducible_environment_v1"
        ),
        "end_to_end_release_readiness_version": value(
            "END_TO_END_RELEASE_READINESS_VERSION", "end_to_end_release_readiness_v1"
        ),
    }
    return _strip_secret_keys(manifest)


def compute_pipeline_fingerprint(manifest: dict[str, Any]) -> str:
    safe = _strip_secret_keys(manifest)
    payload = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def current_pipeline_fingerprint() -> str:
    return compute_pipeline_fingerprint(build_pipeline_version_manifest())


def get_answer_cache_version(settings: Mapping[str, Any] | None = None) -> str:
    settings = settings or {}
    configured = settings.get("CACHE_ANSWER_VERSION")
    if configured is None:
        configured = os.getenv("CACHE_ANSWER_VERSION")
    return _effective_answer_cache_version(configured)


def pipeline_manifest_summary(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = manifest or build_pipeline_version_manifest()
    keys = (
        "phase",
        "architecture_version",
        "architecture_frozen",
        "orchestrator",
        "retrieval_architecture",
        "rrf_k",
        "context_packer_version",
        "max_retrieval_attempts",
        "evidence_contract_version",
        "answer_verifier_version",
        "answer_formatting_contract_version",
        "severity_guard_version",
        "safe_fallback_flow_version",
        "runtime_resilience_version",
        "end_to_end_release_readiness_version",
        "answer_cache_version",
        "embedding_model",
        "embedding_dimensions",
        "qdrant_collection_name",
        "kb_version",
        "prompt_version",
    )
    return {key: manifest.get(key) for key in keys}


def _effective_answer_cache_version(configured: Any) -> str:
    text = str(configured or "").strip()
    if not text or text.lower() in LEGACY_ANSWER_CACHE_VERSIONS:
        return DEFAULT_ANSWER_CACHE_VERSION
    return text


def _effective_answer_formatting_contract_version(configured: Any) -> str:
    text = str(configured or "").strip()
    if not text or text.lower() in LEGACY_ANSWER_FORMATTING_CONTRACT_VERSIONS:
        return DEFAULT_ANSWER_FORMATTING_CONTRACT_VERSION
    return text


def _env_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _csv_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value or "").split(",")
    return [str(part).strip() for part in parts if str(part).strip()]


def _strip_secret_keys(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if not any(marker in key.casefold() for marker in _SECRET_KEY_MARKERS)
    }


def _runtime_chunk_collection_name(settings: Mapping[str, Any]) -> str:
    return str(
        settings.get("QDRANT_COLLECTION_NAME")
        or os.getenv("QDRANT_COLLECTION_NAME")
        or "acne_knowledge"
    )


__all__ = [
    "DEFAULT_ANSWER_CACHE_VERSION",
    "ARCHITECTURE_FROZEN",
    "ARCHITECTURE_VERSION",
    "build_pipeline_version_manifest",
    "compute_pipeline_fingerprint",
    "current_pipeline_fingerprint",
    "get_answer_cache_version",
    "pipeline_manifest_summary",
]
