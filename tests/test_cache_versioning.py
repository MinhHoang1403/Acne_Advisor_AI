from __future__ import annotations

import pytest

from src.agent.nodes import cache as cache_node
from src.observability.versioning import (
    ARCHITECTURE_FROZEN,
    ARCHITECTURE_VERSION,
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
    current_pipeline_fingerprint,
    get_answer_cache_version,
    pipeline_manifest_summary,
)


def test_pipeline_fingerprint_is_deterministic_and_sensitive() -> None:
    manifest = build_pipeline_version_manifest({"CACHE_ANSWER_VERSION": "v10"})
    reversed_manifest = dict(reversed(list(manifest.items())))
    changed = {**manifest, "context_packer_version": "bounded_provenance_packer_v3"}

    assert compute_pipeline_fingerprint(manifest) == compute_pipeline_fingerprint(reversed_manifest)
    assert compute_pipeline_fingerprint(manifest) != compute_pipeline_fingerprint(changed)
    assert len(compute_pipeline_fingerprint(manifest)) == 24


def test_manifest_describes_only_the_frozen_runtime_architecture() -> None:
    manifest = build_pipeline_version_manifest()
    serialized = str(manifest).casefold()

    assert manifest["architecture_version"] == ARCHITECTURE_VERSION
    assert manifest["architecture_frozen"] is ARCHITECTURE_FROZEN
    assert manifest["orchestrator"] == "langgraph"
    assert manifest["retrieval_architecture"] == "dense_bm25_rrf"
    assert manifest["rrf_k"] == 60
    assert manifest["rrf_dense_weight"] == 1.0
    assert manifest["rrf_bm25_weight"] == 1.0
    assert manifest["max_retrieval_attempts"] == 2
    assert "rerank" not in serialized
    assert "candidate_policy" not in serialized
    assert "claim_shadow" not in serialized


def test_retrieval_limits_partition_cache_identity() -> None:
    baseline = build_pipeline_version_manifest(
        {"RETRIEVAL_CONTEXT_MAX_ITEMS": "8", "RETRIEVAL_CONTEXT_MAX_CHARS": "6000"}
    )
    changed = build_pipeline_version_manifest(
        {"RETRIEVAL_CONTEXT_MAX_ITEMS": "6", "RETRIEVAL_CONTEXT_MAX_CHARS": "5000"}
    )

    assert baseline["retrieval_context_max_items"] == 8
    assert changed["retrieval_context_max_items"] == 6
    assert compute_pipeline_fingerprint(baseline) != compute_pipeline_fingerprint(changed)


def test_answer_and_fallback_versions_partition_cache_identity() -> None:
    baseline = build_pipeline_version_manifest(
        {
            "PROMPT_VERSION": "medical_prompt_v4",
            "SAFE_FALLBACK_FLOW_VERSION": "safe_fallback_v1",
        }
    )
    changed = build_pipeline_version_manifest(
        {
            "PROMPT_VERSION": "medical_prompt_v5",
            "SAFE_FALLBACK_FLOW_VERSION": "safe_fallback_v2",
        }
    )

    assert compute_pipeline_fingerprint(baseline) != compute_pipeline_fingerprint(changed)


def test_provider_fallback_contract_partitions_cache_identity() -> None:
    baseline = build_pipeline_version_manifest(
        {"LLM_FALLBACK_POLICY_VERSION": "fallback_v1", "GOOGLE_FALLBACK_MODELS": ""}
    )
    changed = build_pipeline_version_manifest(
        {
            "LLM_FALLBACK_POLICY_VERSION": "fallback_v2",
            "GOOGLE_FALLBACK_MODELS": "gemini-3.1-flash-lite",
        }
    )

    assert changed["google_fallback_models"] == ["gemini-3.1-flash-lite"]
    assert compute_pipeline_fingerprint(baseline) != compute_pipeline_fingerprint(changed)


def test_legacy_cache_versions_are_promoted_to_v10() -> None:
    for version in ("", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"):
        assert get_answer_cache_version({"CACHE_ANSWER_VERSION": version}) == "v10"
    assert get_answer_cache_version({"CACHE_ANSWER_VERSION": "v10"}) == "v10"


def test_legacy_answer_formatting_contract_is_promoted_to_v16() -> None:
    manifest = build_pipeline_version_manifest(
        {"ANSWER_FORMATTING_CONTRACT_VERSION": "answer_formatting_contract_v8"}
    )

    assert manifest["answer_formatting_contract_version"] == "answer_formatting_contract_v16"


def test_changed_semantic_contract_versions_partition_cache_without_v11() -> None:
    current = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v10",
            "ANSWER_FORMATTING_CONTRACT_VERSION": "answer_formatting_contract_v15",
            "SAFE_FALLBACK_FLOW_VERSION": "safe_fallback_flow_v1",
        }
    )

    assert current["answer_cache_version"] == "v10"
    assert current["answer_formatting_contract_version"] == "answer_formatting_contract_v16"
    assert current["safe_fallback_flow_version"] == "safe_fallback_flow_v2"
    assert current["agent_decision_version"] == "minimal_agent_decision_v3"
    assert current["safety_policy_version"] == "source_mapped_safety_policy_v6"
    assert current["answer_validation_version"] == "structural_provenance_locality_validation_v2"


def test_legacy_prompt_version_is_promoted_to_v6() -> None:
    manifest = build_pipeline_version_manifest({"PROMPT_VERSION": "medical_prompt_v3"})

    assert manifest["prompt_version"] == "medical_prompt_v6"


def test_changed_runtime_contract_owners_partition_cache_without_answer_version_bump() -> None:
    current = build_pipeline_version_manifest({"CACHE_ANSWER_VERSION": "v10"})
    previous = {
        **current,
        "agent_decision_version": "minimal_agent_decision_v1",
        "safety_policy_version": "source_mapped_safety_policy_v1",
        "prompt_version": "medical_prompt_v4",
        "answer_formatting_contract_version": "answer_formatting_contract_v13",
    }

    assert current["answer_cache_version"] == previous["answer_cache_version"] == "v10"
    assert compute_pipeline_fingerprint(current) != compute_pipeline_fingerprint(previous)


def test_pipeline_manifest_does_not_include_secrets() -> None:
    manifest = build_pipeline_version_manifest(
        {"QDRANT_API_KEY": "secret", "GOOGLE_API_KEY": "secret", "PASSWORD": "secret"}
    )
    serialized = str(manifest).casefold()

    assert "api_key" not in serialized
    assert "password" not in serialized
    assert "secret" not in serialized


def test_manifest_summary_is_a_compact_production_contract() -> None:
    manifest = build_pipeline_version_manifest(
        {"QDRANT_COLLECTION_NAME": "acne_knowledge", "EMBEDDING_DIMENSIONS": "bad"}
    )
    summary = pipeline_manifest_summary(manifest)

    assert summary["phase"] == "production"
    assert summary["architecture_version"] == ARCHITECTURE_VERSION
    assert summary["retrieval_architecture"] == "dense_bm25_rrf"
    assert summary["answer_cache_version"] == "v10"
    assert summary["evidence_grounding_version"] == "evidence_grounded_runtime_v2"
    assert summary["embedding_dimensions"] == 3072
    assert summary["qdrant_collection_name"] == "acne_knowledge"


def test_current_pipeline_fingerprint_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CACHE_ANSWER_VERSION", "v10")

    assert current_pipeline_fingerprint() == compute_pipeline_fingerprint(
        build_pipeline_version_manifest()
    )


@pytest.mark.asyncio
async def test_cache_store_metadata_has_production_fingerprint_and_v10(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    async def fake_set_answer_cache(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)

    monkeypatch.setattr(cache_node, "set_answer_cache", fake_set_answer_cache)
    monkeypatch.setenv("CACHE_ANSWER_VERSION", "v10")
    manifest = build_pipeline_version_manifest({"CACHE_ANSWER_VERSION": "v10"})

    result = await cache_node.cache_store_node(
        {
            "cache_hit": False,
            "bypass_cache": False,
            "conversation_context": {"messages": [], "message_count": 0},
            "cache_reason": "miss",
            "llm_fallback_used": False,
            "fallback_applied": False,
            "fallback_cache_eligible": True,
            "retrieval_status": "ok",
            "user_question": "Mụn đầu đen là gì?",
            "standalone_question": None,
            "final_answer": "Mụn đầu đen là dạng nhân mụn mở liên quan bít tắc nang lông.",
            "sources": ["source.pdf"],
            "actual_provider": "gemini",
            "actual_model": "gemini-3.5-flash",
            "answer_quality_report": {"passed": True, "issues": []},
            "pipeline_manifest": manifest,
            "pipeline_fingerprint": "abc123fingerprint",
            "source_allowlist": [{"source_id": "source.pdf", "display_name": "Source"}],
            "packed_context": {
                "items": [{"item_id": "chunk-1", "payload": {"source_id": "source.pdf"}}]
            },
        }
    )

    assert result == {}
    assert captured["pipeline_fingerprint"] == "abc123fingerprint"
    assert captured["args"][3]["pipeline_fingerprint"] == "abc123fingerprint"
    assert captured["args"][3]["answer_version"] == "v10"
    assert captured["args"][3]["selected_evidence_ids"] == ["chunk-1"]
