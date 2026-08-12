from __future__ import annotations

import pytest

from src.agent.nodes import cache as cache_node
from src.observability.versioning import (
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
    current_pipeline_fingerprint,
    get_answer_cache_version,
    pipeline_manifest_summary,
)


def test_pipeline_fingerprint_is_deterministic_and_changes():
    manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "RERANKER_VERSION": "local_reranker_v1",
        }
    )
    fingerprint_a = compute_pipeline_fingerprint(manifest)
    fingerprint_b = compute_pipeline_fingerprint(dict(reversed(list(manifest.items()))))

    changed = dict(manifest)
    changed["reranker_version"] = "local_reranker_v2"

    assert fingerprint_a == fingerprint_b
    assert fingerprint_a != compute_pipeline_fingerprint(changed)
    assert len(fingerprint_a) == 24


def test_retrieval_pipeline_version_partitions_cache_identity():
    v4_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "RETRIEVAL_PIPELINE_VERSION": "v4",
        }
    )
    v5_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "RETRIEVAL_PIPELINE_VERSION": "v5",
        }
    )

    assert v4_manifest["retrieval_pipeline_version"] == "v4"
    assert v5_manifest["retrieval_pipeline_version"] == "v5"
    assert compute_pipeline_fingerprint(v4_manifest) != compute_pipeline_fingerprint(v5_manifest)


def test_retrieval_pipeline_manifest_defaults_to_v4():
    manifest = build_pipeline_version_manifest({"RETRIEVAL_PIPELINE_VERSION": ""})

    assert manifest["retrieval_pipeline_version"] == "v4"


def test_retrieval_v5_config_version_partitions_cache_identity():
    previous = build_pipeline_version_manifest(
        {
            "RETRIEVAL_PIPELINE_VERSION": "v5",
            "RETRIEVAL_V5_CONFIG_VERSION": "retrieval_v5_entity_graph_signals_v1",
        }
    )
    current = build_pipeline_version_manifest(
        {
            "RETRIEVAL_PIPELINE_VERSION": "v5",
            "RETRIEVAL_V5_CONFIG_VERSION": "retrieval_v5_candidate_policy_v1",
        }
    )

    assert current["retrieval_v5_config_version"] == "retrieval_v5_candidate_policy_v1"
    assert compute_pipeline_fingerprint(previous) != compute_pipeline_fingerprint(current)


def test_answer_verifier_version_is_in_manifest_and_changes_fingerprint():
    old_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "ANSWER_VERIFIER_VERSION": "answer_verifier_v1",
        }
    )
    new_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "ANSWER_VERIFIER_VERSION": "answer_verifier_v2",
        }
    )

    assert new_manifest["answer_verifier_version"] == "answer_verifier_v2"
    assert compute_pipeline_fingerprint(old_manifest) != compute_pipeline_fingerprint(new_manifest)


def test_answer_formatting_contract_version_is_in_manifest_and_changes_fingerprint():
    old_manifest = build_pipeline_version_manifest({"CACHE_ANSWER_VERSION": "v5"})
    old_manifest["answer_formatting_contract_version"] = "answer_formatting_contract_v6"
    new_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "ANSWER_FORMATTING_CONTRACT_VERSION": "answer_formatting_contract_v11",
        }
    )

    assert new_manifest["answer_formatting_contract_version"] == "answer_formatting_contract_v11"
    assert compute_pipeline_fingerprint(old_manifest) != compute_pipeline_fingerprint(new_manifest)


def test_legacy_answer_formatting_contract_is_promoted_to_v11():
    manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "ANSWER_FORMATTING_CONTRACT_VERSION": "answer_formatting_contract_v8",
        }
    )

    assert manifest["answer_formatting_contract_version"] == "answer_formatting_contract_v11"


def test_llm_fallback_policy_and_google_fallback_models_change_fingerprint():
    old_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "LLM_FALLBACK_POLICY_VERSION": "llm_fallback_policy_v1",
            "GOOGLE_MODEL": "gemini-3.5-flash",
            "GOOGLE_FALLBACK_MODELS": "",
        }
    )
    new_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "LLM_FALLBACK_POLICY_VERSION": "llm_fallback_policy_v2",
            "GOOGLE_MODEL": "gemini-3.5-flash",
            "GOOGLE_FALLBACK_MODELS": "gemini-3.1-flash-lite",
        }
    )

    assert new_manifest["answer_cache_version"] == "v5"
    assert new_manifest["llm_fallback_policy_version"] == "llm_fallback_policy_v2"
    assert new_manifest["google_fallback_models"] == ["gemini-3.1-flash-lite"]
    assert compute_pipeline_fingerprint(old_manifest) != compute_pipeline_fingerprint(new_manifest)


def test_severity_guard_version_is_in_manifest_and_changes_fingerprint():
    old_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "SEVERITY_GUARD_VERSION": "severity_aware_answer_guard_v0",
        }
    )
    new_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "SEVERITY_GUARD_VERSION": "severity_aware_answer_guard_v1",
        }
    )

    assert new_manifest["severity_guard_version"] == "severity_aware_answer_guard_v1"
    assert compute_pipeline_fingerprint(old_manifest) != compute_pipeline_fingerprint(new_manifest)


def test_entity_foundation_version_is_in_manifest_and_changes_fingerprint():
    old_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "ENTITY_FOUNDATION_VERSION": "entity_foundation_v1",
        }
    )
    new_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "ENTITY_FOUNDATION_VERSION": "entity_foundation_v2",
        }
    )

    assert new_manifest["entity_foundation_version"] == "entity_foundation_v2"
    assert new_manifest["answer_cache_version"] == "v5"
    assert compute_pipeline_fingerprint(old_manifest) != compute_pipeline_fingerprint(new_manifest)


def test_grounding_context_and_performance_versions_change_fingerprint():
    old_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "SOURCE_NORMALIZATION_VERSION": "source_normalization_v0",
            "ANSWERABILITY_POLICY_VERSION": "answerability_policy_v0",
            "CONVERSATION_CONTEXT_VERSION": "conversation_context_v0",
            "GRAPH_ENTITY_ANSWER_VERSION": "graph_entity_answer_v0",
            "PERFORMANCE_INSTRUMENTATION_VERSION": "performance_instrumentation_v0",
        }
    )
    new_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "SOURCE_NORMALIZATION_VERSION": "source_normalization_v1",
            "ANSWERABILITY_POLICY_VERSION": "answerability_policy_v1",
            "CONVERSATION_CONTEXT_VERSION": "conversation_context_v1",
            "GRAPH_ENTITY_ANSWER_VERSION": "graph_entity_answer_v1",
            "PERFORMANCE_INSTRUMENTATION_VERSION": "performance_instrumentation_v1",
        }
    )

    assert new_manifest["source_normalization_version"] == "source_normalization_v1"
    assert new_manifest["answerability_policy_version"] == "answerability_policy_v1"
    assert new_manifest["conversation_context_version"] == "conversation_context_v1"
    assert new_manifest["graph_entity_answer_version"] == "graph_entity_answer_v1"
    assert new_manifest["performance_instrumentation_version"] == "performance_instrumentation_v1"
    assert compute_pipeline_fingerprint(old_manifest) != compute_pipeline_fingerprint(new_manifest)


def test_google_genai_sdk_version_is_in_manifest_and_changes_fingerprint():
    old_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "GOOGLE_GENAI_SDK_VERSION": "legacy_google_sdk",
        }
    )
    new_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "GOOGLE_GENAI_SDK_VERSION": "google_genai_sdk_v1",
        }
    )

    assert new_manifest["google_genai_sdk_version"] == "google_genai_sdk_v1"
    assert compute_pipeline_fingerprint(old_manifest) != compute_pipeline_fingerprint(new_manifest)


def test_pipeline_manifest_does_not_include_secret_keys():
    manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "QDRANT_API_KEY": "secret",
            "GOOGLE_API_KEY": "secret",
        }
    )
    serialized = str(manifest).lower()

    assert "api_key" not in serialized
    assert "secret" not in serialized


def test_manifest_summary_includes_severity_and_runtime_fields():
    manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "SEVERITY_GUARD_VERSION": "severity_aware_answer_guard_v1",
            "CHUNK_QDRANT_COLLECTION_NAME": "acne_chunks_v2",
            "QDRANT_COLLECTION_NAME": "acne_knowledge",
            "ENTITY_QDRANT_COLLECTION_NAME": "acne_entities_v2",
            "EMBEDDING_DIMENSIONS": "not-an-int",
            "RERANK_ENABLED": "off",
            "RERANK_TOP_N": "bad",
            "SEMANTIC_RERANK_MODEL_PATH": "C:/Models/acne-reranker/bge-reranker-v2-m3",
            "SEMANTIC_RERANK_MAX_CANDIDATES": "bad",
            "SEMANTIC_RERANK_WEIGHT": "bad",
            "RULE_RERANK_WEIGHT": "0.25",
            "RETRIEVAL_RERANK_WEIGHT": "bad",
        }
    )
    summary = pipeline_manifest_summary(manifest)

    assert manifest["severity_guard_version"] == "severity_aware_answer_guard_v1"
    assert manifest["entity_foundation_version"] == "entity_foundation_v2"
    assert manifest["qdrant_collection_name"] == "acne_chunks_v2"
    assert manifest["entity_collection_name"] == "acne_entities_v2"
    assert manifest["embedding_dimensions"] == 3072
    assert manifest["rerank_enabled"] is False
    assert manifest["rerank_top_n"] == 8
    assert manifest["semantic_rerank_model_identifier"] == "bge-reranker-v2-m3"
    assert manifest["semantic_rerank_max_candidates"] == 32
    assert manifest["semantic_rerank_weight"] == 0.70
    assert manifest["rule_rerank_weight"] == 0.25
    assert manifest["retrieval_rerank_weight"] == 0.10
    assert summary["severity_guard_version"] == manifest["severity_guard_version"]
    assert summary["entity_foundation_version"] == manifest["entity_foundation_version"]


def test_manifest_promotes_legacy_chunk_collection_to_base_collection():
    manifest = build_pipeline_version_manifest(
        {
            "CHUNK_QDRANT_COLLECTION_NAME": "acne_chunks_v1",
            "QDRANT_COLLECTION_NAME": "acne_knowledge",
            "ENTITY_QDRANT_COLLECTION_NAME": "",
            "CACHE_ANSWER_VERSION": "",
            "RERANK_ENABLED": "maybe",
        }
    )

    assert manifest["qdrant_collection_name"] == "acne_knowledge"
    assert manifest["entity_collection_name"] == "acne_entities_v1"
    assert manifest["answer_cache_version"] == "v5"
    assert manifest["rerank_enabled"] is True


def test_current_pipeline_fingerprint_uses_environment(monkeypatch):
    monkeypatch.setenv("CACHE_ANSWER_VERSION", "v5")
    monkeypatch.setenv("SEVERITY_GUARD_VERSION", "severity_aware_answer_guard_test")

    assert current_pipeline_fingerprint() == compute_pipeline_fingerprint(
        build_pipeline_version_manifest()
    )


def test_legacy_answer_cache_version_is_promoted_to_phase2e_default():
    assert get_answer_cache_version({"CACHE_ANSWER_VERSION": "v4"}) == "v5"
    assert get_answer_cache_version({"CACHE_ANSWER_VERSION": "v6"}) == "v6"


@pytest.mark.asyncio
async def test_cache_store_metadata_has_pipeline_fingerprint(monkeypatch):
    captured: dict = {}

    async def fake_set_answer_cache(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cache_node, "set_answer_cache", fake_set_answer_cache)
    monkeypatch.setenv("CACHE_MIN_ANSWER_CHARS", "10")
    monkeypatch.setenv("CACHE_REQUIRED_ENTITY_CHECK", "false")
    monkeypatch.setenv("CACHE_ANSWER_VERSION", "v5")

    result = await cache_node.cache_store_node(
        {
            "cache_hit": False,
            "bypass_cache": False,
            "conversation_history": [],
            "cache_reason": "miss",
            "is_in_domain": True,
            "use_history_context": False,
            "errors": [],
            "llm_fallback": False,
            "llm_fallback_used": False,
            "guardrail": "in_domain",
            "fallback_provider": None,
            "user_question": "Mụn đầu đen là gì?",
            "standalone_question": None,
            "final_answer": "Mụn đầu đen là dạng nhân mụn mở liên quan bít tắc nang lông.",
            "sources": ["source.pdf"],
            "actual_provider": "gemini",
            "actual_model": "gemini-2.5-flash",
            "answer_quality_report": {"passed": True, "issues": []},
            "pipeline_manifest": {"phase": "phase2e", "cache_schema_version": "v3"},
            "pipeline_fingerprint": "abc123fingerprint",
        }
    )

    assert result == {}
    assert captured["pipeline_fingerprint"] == "abc123fingerprint"
    assert captured["metadata"]["pipeline_fingerprint"] == "abc123fingerprint"
    assert captured["metadata"]["answer_cache_version"] == "v5"
