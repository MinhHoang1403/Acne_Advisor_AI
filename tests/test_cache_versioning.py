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
    manifest = build_pipeline_version_manifest({"CACHE_ANSWER_VERSION": "v7"})
    reversed_manifest = dict(reversed(list(manifest.items())))
    changed = {**manifest, "context_packer_version": "bounded_provenance_packer_v2"}

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


def test_answer_and_safety_versions_partition_cache_identity() -> None:
    baseline = build_pipeline_version_manifest(
        {
            "ANSWER_VERIFIER_VERSION": "answer_verifier_v1",
            "SEVERITY_GUARD_VERSION": "severity_guard_v1",
        }
    )
    changed = build_pipeline_version_manifest(
        {
            "ANSWER_VERIFIER_VERSION": "answer_verifier_v2",
            "SEVERITY_GUARD_VERSION": "severity_guard_v2",
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


def test_legacy_cache_versions_are_promoted_to_v7() -> None:
    for version in ("", "v1", "v2", "v3", "v4", "v5", "v6"):
        assert get_answer_cache_version({"CACHE_ANSWER_VERSION": version}) == "v7"
    assert get_answer_cache_version({"CACHE_ANSWER_VERSION": "v7"}) == "v7"


def test_legacy_answer_formatting_contract_is_promoted_to_v12() -> None:
    manifest = build_pipeline_version_manifest(
        {"ANSWER_FORMATTING_CONTRACT_VERSION": "answer_formatting_contract_v8"}
    )

    assert manifest["answer_formatting_contract_version"] == "answer_formatting_contract_v12"


def test_pipeline_manifest_does_not_include_secrets() -> None:
    manifest = build_pipeline_version_manifest(
        {"QDRANT_API_KEY": "secret", "GOOGLE_API_KEY": "secret", "PASSWORD": "secret"}
    )
    serialized = str(manifest).casefold()

    assert "api_key" not in serialized
    assert "password" not in serialized
    assert "secret" not in serialized


def test_manifest_summary_is_a_compact_s4b_contract() -> None:
    manifest = build_pipeline_version_manifest(
        {"QDRANT_COLLECTION_NAME": "acne_knowledge", "EMBEDDING_DIMENSIONS": "bad"}
    )
    summary = pipeline_manifest_summary(manifest)

    assert summary["phase"] == "s4b"
    assert summary["architecture_version"] == ARCHITECTURE_VERSION
    assert summary["retrieval_architecture"] == "dense_bm25_rrf"
    assert summary["answer_cache_version"] == "v7"
    assert summary["evidence_grounding_version"] == "evidence_grounded_runtime_v1"
    assert summary["embedding_dimensions"] == 3072
    assert summary["qdrant_collection_name"] == "acne_knowledge"


def test_current_pipeline_fingerprint_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CACHE_ANSWER_VERSION", "v7")
    monkeypatch.setenv("SEVERITY_GUARD_VERSION", "severity_guard_test")

    assert current_pipeline_fingerprint() == compute_pipeline_fingerprint(
        build_pipeline_version_manifest()
    )


@pytest.mark.asyncio
async def test_cache_store_metadata_has_s4b_fingerprint_and_v7(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    async def fake_set_answer_cache(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cache_node, "set_answer_cache", fake_set_answer_cache)
    monkeypatch.setenv("CACHE_MIN_ANSWER_CHARS", "10")
    monkeypatch.setenv("CACHE_ANSWER_VERSION", "v7")
    manifest = build_pipeline_version_manifest({"CACHE_ANSWER_VERSION": "v7"})

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
            "fallback_applied": False,
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
    assert captured["metadata"]["pipeline_fingerprint"] == "abc123fingerprint"
    assert captured["metadata"]["answer_cache_version"] == "v7"
    assert captured["metadata"]["retrieval"] == "dense_bm25_rrf"
    assert captured["metadata"]["selected_evidence_ids"] == ["chunk-1"]
