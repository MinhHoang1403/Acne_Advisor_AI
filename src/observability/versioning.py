"""Deterministic Phase 2 pipeline version manifest and fingerprint."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping

from src.retrieval.v5_contracts import RETRIEVAL_V5_CONFIG_VERSION


DEFAULT_ANSWER_CACHE_VERSION = "v5"
DEFAULT_RERANKER_VERSION = "reranker_pipeline_v2"
DEFAULT_GOOGLE_GENAI_SDK_VERSION = "google_genai_sdk_v1"
DEFAULT_REPRODUCIBLE_ENVIRONMENT_VERSION = "reproducible_environment_v1"
DEFAULT_END_TO_END_RELEASE_READINESS_VERSION = "end_to_end_release_readiness_v1"
DEFAULT_ANSWER_FORMATTING_CONTRACT_VERSION = "answer_formatting_contract_v11"
DEFAULT_LLM_FALLBACK_POLICY_VERSION = "llm_fallback_policy_v2"
DEFAULT_ENTITY_FOUNDATION_VERSION = "entity_foundation_v2"
DEFAULT_SOURCE_NORMALIZATION_VERSION = "source_normalization_v1"
DEFAULT_ANSWERABILITY_POLICY_VERSION = "answerability_policy_v1"
DEFAULT_CONVERSATION_CONTEXT_VERSION = "conversation_context_v1"
DEFAULT_GRAPH_ENTITY_ANSWER_VERSION = "graph_entity_answer_v1"
DEFAULT_PERFORMANCE_INSTRUMENTATION_VERSION = "performance_instrumentation_v1"
LEGACY_ANSWER_CACHE_VERSIONS = {"v1", "v2", "v3", "v4"}
LEGACY_ANSWER_FORMATTING_CONTRACT_VERSIONS = {
    "answer_formatting_contract_v1",
    "answer_formatting_contract_v2",
    "answer_formatting_contract_v3",
    "answer_formatting_contract_v4",
    "answer_formatting_contract_v5",
    "answer_formatting_contract_v6",
    "answer_formatting_contract_v7",
    "answer_formatting_contract_v8",
    "answer_formatting_contract_v9",
    "answer_formatting_contract_v10",
}

_SECRET_KEY_MARKERS = (
    "api_key",
    "token",
    "password",
    "secret",
    "authorization",
    "bearer",
    "cookie",
)


def build_pipeline_version_manifest(settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a stable, secret-free Phase 2 pipeline version manifest."""

    settings = settings or {}

    def value(name: str, default: Any = "") -> Any:
        if name in settings:
            return settings[name]
        return os.getenv(name, default)

    answer_cache_version = _effective_answer_cache_version(value("CACHE_ANSWER_VERSION", None))

    manifest = {
        "phase": "phase2e",
        "retrieval_pipeline_version": _effective_retrieval_pipeline_version(
            value("RETRIEVAL_PIPELINE_VERSION", "v4")
        ),
        "retrieval_v5_config_version": _effective_retrieval_v5_config_version(
            value("RETRIEVAL_V5_CONFIG_VERSION", RETRIEVAL_V5_CONFIG_VERSION)
        ),
        "context_packer_version": value("CONTEXT_PACKER_VERSION", "context_packer_v3"),
        "reranker_version": value("RERANKER_VERSION", DEFAULT_RERANKER_VERSION),
        "answer_verifier_version": value("ANSWER_VERIFIER_VERSION", "answer_verifier_v2"),
        "answer_formatting_contract_version": _effective_answer_formatting_contract_version(
            value(
                "ANSWER_FORMATTING_CONTRACT_VERSION",
                DEFAULT_ANSWER_FORMATTING_CONTRACT_VERSION,
            )
        ),
        "severity_guard_version": value("SEVERITY_GUARD_VERSION", "severity_aware_answer_guard_v1"),
        "safe_fallback_flow_version": value("SAFE_FALLBACK_FLOW_VERSION", "safe_fallback_flow_v1"),
        "google_genai_sdk_version": value("GOOGLE_GENAI_SDK_VERSION", DEFAULT_GOOGLE_GENAI_SDK_VERSION),
        "reproducible_environment_version": value(
            "REPRODUCIBLE_ENVIRONMENT_VERSION",
            DEFAULT_REPRODUCIBLE_ENVIRONMENT_VERSION,
        ),
        "end_to_end_release_readiness_version": value(
            "END_TO_END_RELEASE_READINESS_VERSION",
            DEFAULT_END_TO_END_RELEASE_READINESS_VERSION,
        ),
        "runtime_resilience_version": value("RUNTIME_RESILIENCE_VERSION", "runtime_resilience_v1"),
        "llm_fallback_policy_version": value(
            "LLM_FALLBACK_POLICY_VERSION",
            DEFAULT_LLM_FALLBACK_POLICY_VERSION,
        ),
        "google_model": value("GOOGLE_MODEL", "gemini-3.5-flash") or "gemini-3.5-flash",
        "google_fallback_models": _csv_list(
            value("GOOGLE_FALLBACK_MODELS", "gemini-3.1-flash-lite")
        ),
        "ollama_model": value("OLLAMA_MODEL", "qwen3:8b") or "qwen3:8b",
        "neo4j_schema_version": value("NEO4J_SCHEMA_VERSION", "neo4j_schema_v1"),
        "taxonomy_version": value("TAXONOMY_VERSION", "drug_taxonomy_v1"),
        "entity_foundation_version": value(
            "ENTITY_FOUNDATION_VERSION",
            DEFAULT_ENTITY_FOUNDATION_VERSION,
        ),
        "cache_schema_version": value("CACHE_SCHEMA_VERSION", "v3"),
        "answer_cache_version": answer_cache_version,
        "embedding_model": value("EMBEDDING_MODEL", "models/gemini-embedding-2"),
        "embedding_dimensions": _env_int(value("EMBEDDING_DIMENSIONS", "3072"), default=3072),
        "qdrant_collection_name": _runtime_chunk_collection_name(settings),
        "entity_collection_name": _runtime_entity_collection_name(settings),
        "kb_version": value("KB_VERSION", "acne_kb_v1"),
        "prompt_version": value("PROMPT_VERSION", "medical_prompt_v2"),
        "rerank_enabled": _env_bool(value("RERANK_ENABLED", "true"), default=True),
        "rerank_provider": value("RERANK_PROVIDER", "local_rules") or "local_rules",
        "rerank_top_n": _env_int(value("RERANK_TOP_N", "8"), default=8),
        "retrieval_context_max_items": _env_int(value("RETRIEVAL_CONTEXT_MAX_ITEMS", "5"), default=5),
        "retrieval_context_max_chars": _env_int(value("RETRIEVAL_CONTEXT_MAX_CHARS", "4200"), default=4200),
        "semantic_rerank_model_identifier": _semantic_model_identifier(
            value("SEMANTIC_RERANK_MODEL_PATH", "")
        ),
        "semantic_rerank_max_candidates": _env_int(
            value("SEMANTIC_RERANK_MAX_CANDIDATES", "32"),
            default=32,
        ),
        "semantic_rerank_max_query_chars": _env_int(
            value("SEMANTIC_RERANK_MAX_QUERY_CHARS", "1000"),
            default=1000,
        ),
        "semantic_rerank_max_document_chars": _env_int(
            value("SEMANTIC_RERANK_MAX_DOCUMENT_CHARS", "4000"),
            default=4000,
        ),
        "semantic_rerank_allow_fallback": _env_bool(
            value("SEMANTIC_RERANK_ALLOW_FALLBACK", "true"),
            default=True,
        ),
        "semantic_rerank_weight": _env_float(value("SEMANTIC_RERANK_WEIGHT", "0.70"), default=0.70),
        "rule_rerank_weight": _env_float(value("RULE_RERANK_WEIGHT", "0.20"), default=0.20),
        "retrieval_rerank_weight": _env_float(value("RETRIEVAL_RERANK_WEIGHT", "0.10"), default=0.10),
        "answer_verifier_enabled": _env_bool(value("ANSWER_VERIFIER_ENABLED", "true"), default=True),
        "answer_guard_mode": value("ANSWER_GUARD_MODE", "metadata_only") or "metadata_only",
        "source_normalization_version": value(
            "SOURCE_NORMALIZATION_VERSION",
            DEFAULT_SOURCE_NORMALIZATION_VERSION,
        ),
        "answerability_policy_version": value(
            "ANSWERABILITY_POLICY_VERSION",
            DEFAULT_ANSWERABILITY_POLICY_VERSION,
        ),
        "conversation_context_version": value(
            "CONVERSATION_CONTEXT_VERSION",
            DEFAULT_CONVERSATION_CONTEXT_VERSION,
        ),
        "graph_entity_answer_version": value(
            "GRAPH_ENTITY_ANSWER_VERSION",
            DEFAULT_GRAPH_ENTITY_ANSWER_VERSION,
        ),
        "performance_instrumentation_version": value(
            "PERFORMANCE_INSTRUMENTATION_VERSION",
            DEFAULT_PERFORMANCE_INSTRUMENTATION_VERSION,
        ),
    }
    return _strip_secret_keys(manifest)


def compute_pipeline_fingerprint(manifest: dict[str, Any]) -> str:
    """Compute a deterministic short SHA256 fingerprint for a manifest."""

    safe_manifest = _strip_secret_keys(manifest)
    payload = json.dumps(safe_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def current_pipeline_fingerprint() -> str:
    """Return the fingerprint for the current environment manifest."""

    return compute_pipeline_fingerprint(build_pipeline_version_manifest())


def get_answer_cache_version(settings: Mapping[str, Any] | None = None) -> str:
    """Return the effective Phase 2E cache answer version.

    Legacy values v1-v4 are promoted to v5 so stale `.env` files do not reuse
    pre-Phase-2E answer-cache namespaces. Newer explicit values are respected.
    """

    settings = settings or {}
    configured = settings.get("CACHE_ANSWER_VERSION")
    if configured is None:
        configured = os.getenv("CACHE_ANSWER_VERSION")
    return _effective_answer_cache_version(configured)


def pipeline_manifest_summary(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a compact manifest summary suitable for cache/debug metadata."""

    manifest = manifest or build_pipeline_version_manifest()
    summary_keys = [
        "phase",
        "retrieval_pipeline_version",
        "retrieval_v5_config_version",
        "context_packer_version",
        "reranker_version",
        "answer_verifier_version",
        "answer_formatting_contract_version",
        "severity_guard_version",
        "safe_fallback_flow_version",
        "google_genai_sdk_version",
        "llm_fallback_policy_version",
        "google_model",
        "google_fallback_models",
        "ollama_model",
        "reproducible_environment_version",
        "end_to_end_release_readiness_version",
        "runtime_resilience_version",
        "neo4j_schema_version",
        "taxonomy_version",
        "entity_foundation_version",
        "cache_schema_version",
        "answer_cache_version",
        "embedding_model",
        "qdrant_collection_name",
        "entity_collection_name",
        "rerank_enabled",
        "rerank_provider",
        "rerank_top_n",
        "retrieval_context_max_items",
        "retrieval_context_max_chars",
        "semantic_rerank_model_identifier",
        "semantic_rerank_max_candidates",
        "answer_verifier_enabled",
        "answer_guard_mode",
        "source_normalization_version",
        "answerability_policy_version",
        "conversation_context_version",
        "graph_entity_answer_version",
        "performance_instrumentation_version",
    ]
    return {key: manifest.get(key) for key in summary_keys if key in manifest}


def _env_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _effective_answer_cache_version(configured: Any) -> str:
    text = str(configured or "").strip()
    if not text:
        return DEFAULT_ANSWER_CACHE_VERSION
    if text.lower() in LEGACY_ANSWER_CACHE_VERSIONS:
        return DEFAULT_ANSWER_CACHE_VERSION
    return text


def _effective_retrieval_pipeline_version(configured: Any) -> str:
    """Normalize cache identity to the staged retrieval execution selector."""

    text = str(configured or "").strip().lower()
    if text in {"v4", "v5_shadow", "v5"}:
        return text
    return "v4"


def _effective_retrieval_v5_config_version(configured: Any) -> str:
    """Keep the V5 cache namespace versioned when semantic contracts change."""

    return str(configured or "").strip() or RETRIEVAL_V5_CONFIG_VERSION


def _effective_answer_formatting_contract_version(configured: Any) -> str:
    text = str(configured or "").strip()
    if not text:
        return DEFAULT_ANSWER_FORMATTING_CONTRACT_VERSION
    if text.lower() in LEGACY_ANSWER_FORMATTING_CONTRACT_VERSIONS:
        return DEFAULT_ANSWER_FORMATTING_CONTRACT_VERSION
    return text


def _env_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _csv_list(value: Any) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in str(value or "").split(","):
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _semantic_model_identifier(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = text.replace("\\", "/").rstrip("/")
    return normalized.split("/")[-1] or "local_model"


def _strip_secret_keys(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if not any(marker in key.lower() for marker in _SECRET_KEY_MARKERS)
    }


def _runtime_chunk_collection_name(settings: Mapping[str, Any]) -> str:
    if "CHUNK_QDRANT_COLLECTION_NAME" in settings or "QDRANT_COLLECTION_NAME" in settings:
        configured = str(settings.get("CHUNK_QDRANT_COLLECTION_NAME") or "").strip()
        base = str(settings.get("QDRANT_COLLECTION_NAME") or os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge")).strip()
        if not configured or configured == "acne_chunks_v1":
            return base or "acne_knowledge"
        return configured
    try:
        from src.knowledge.entity_index import get_chunk_collection_name

        return get_chunk_collection_name()
    except Exception:
        configured = os.getenv("CHUNK_QDRANT_COLLECTION_NAME", "").strip()
        base = os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge").strip() or "acne_knowledge"
        if not configured or configured == "acne_chunks_v1":
            return base
        return configured


def _runtime_entity_collection_name(settings: Mapping[str, Any]) -> str:
    if "ENTITY_QDRANT_COLLECTION_NAME" in settings:
        return str(settings["ENTITY_QDRANT_COLLECTION_NAME"] or "acne_entities_v1")
    try:
        from src.knowledge.entity_index import get_entity_collection_name

        return get_entity_collection_name()
    except Exception:
        return os.getenv("ENTITY_QDRANT_COLLECTION_NAME", "acne_entities_v1")


__all__ = [
    "DEFAULT_ANSWER_CACHE_VERSION",
    "DEFAULT_ANSWER_FORMATTING_CONTRACT_VERSION",
    "DEFAULT_ENTITY_FOUNDATION_VERSION",
    "DEFAULT_END_TO_END_RELEASE_READINESS_VERSION",
    "DEFAULT_GOOGLE_GENAI_SDK_VERSION",
    "DEFAULT_LLM_FALLBACK_POLICY_VERSION",
    "DEFAULT_REPRODUCIBLE_ENVIRONMENT_VERSION",
    "DEFAULT_RERANKER_VERSION",
    "build_pipeline_version_manifest",
    "compute_pipeline_fingerprint",
    "current_pipeline_fingerprint",
    "get_answer_cache_version",
    "pipeline_manifest_summary",
]
