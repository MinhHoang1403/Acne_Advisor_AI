#!/usr/bin/env python3
"""Read-only Phase 2 readiness inspection.

This script validates runtime compatibility with the hardened Phase 1 outputs.
It does not run ingestion, build indexes, call LLM/embedding providers, or write
to Qdrant/Neo4j/PostgreSQL/Redis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from scripts.validate_kb_collections import (  # noqa: E402
    inspect_qdrant_schema,
    qdrant_addressable_names,
)
from src.database.vector_store import qdrant_client_kwargs  # noqa: E402
from src.knowledge.entity_index import (  # noqa: E402
    get_chunk_collection_name,
    get_entity_collection_name,
)
from src.knowledge.versioning import get_embedding_metadata  # noqa: E402
from src.observability.versioning import get_answer_cache_version  # noqa: E402
from src.retrieval.reranker import rerank_provider_from_env  # noqa: E402
from src.retrieval.reranking.providers import semantic_config_from_env  # noqa: E402

logging.getLogger("httpx").setLevel(logging.WARNING)

EXPECTED_NEO4J_LABELS = {
    "ActiveIngredient": 7,
    "Condition": 1,
    "DrugClass": 6,
    "DrugProduct": 3,
    "SafetyContext": 4,
}
REBUILT_NEO4J_LABELS = {
    "ActiveIngredient": 8,
    "Condition": 1,
    "DrugClass": 6,
    "DrugProduct": 4,
    "SafetyContext": 4,
}
EXPECTED_NEO4J_RELATIONSHIPS = {
    "BELONGS_TO_CLASS": 11,
    "CONTRAINDICATED_IN": 0,
    "HAS_ACTIVE_INGREDIENT": 4,
}
REBUILT_NEO4J_RELATIONSHIPS = {
    "BELONGS_TO_CLASS": 13,
    "CONTRAINDICATED_IN": 0,
    "HAS_ACTIVE_INGREDIENT": 5,
}
EXPANDED_NEO4J_LABELS = {
    "ActiveIngredient": 15,
    "Condition": 1,
    "DrugClass": 8,
    "DrugProduct": 4,
    "SafetyContext": 4,
}
EXPANDED_NEO4J_RELATIONSHIPS = {
    "BELONGS_TO_CLASS": 20,
    "CONTRAINDICATED_IN": 2,
    "HAS_ACTIVE_INGREDIENT": 5,
}
ACCEPTABLE_ENTITY_POINTS = {20, 22, 32}


def _check(
    name: str,
    passed: bool,
    details: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "passed": bool(passed),
        "details": details or {},
    }
    if error:
        record["error"] = error
    return record


async def inspect_qdrant() -> dict[str, Any]:
    from qdrant_client import AsyncQdrantClient  # type: ignore[import]

    checks: list[dict[str, Any]] = []
    chunk_collection = get_chunk_collection_name()
    entity_collection = get_entity_collection_name()
    expected_dimensions = int(get_embedding_metadata()["embedding_dimensions"])
    client = AsyncQdrantClient(**qdrant_client_kwargs())

    try:
        collections = await client.get_collections()
        aliases = await client.get_aliases()
        existing = qdrant_addressable_names(collections, aliases)
        checks.append(_check("qdrant_reachable", True, {"collections_and_aliases": sorted(existing)}))

        for role, collection_name, expected_points in (
            ("chunk", chunk_collection, None),
            ("entity", entity_collection, ACCEPTABLE_ENTITY_POINTS),
        ):
            if collection_name not in existing:
                checks.append(
                    _check(
                        f"qdrant_{role}_collection_exists",
                        False,
                        {"collection": collection_name},
                        f"{role} collection {collection_name!r} missing",
                    )
                )
                continue

            info = await client.get_collection(collection_name=collection_name)
            schema = inspect_qdrant_schema(info.config.params)
            points_count = int(getattr(info, "points_count", 0) or 0)
            details = {
                "collection": collection_name,
                "points_count": points_count,
                **schema,
            }
            schema_ok = (
                schema["has_dense"]
                and schema["dense_vector_name"] == "dense"
                and schema["dense_size"] == expected_dimensions
                and schema["has_bm25"]
                and schema["sparse_vector_name"] == "bm25"
            )
            points_ok = points_count > 0 if expected_points is None else points_count in expected_points
            checks.append(
                _check(
                    f"qdrant_{role}_schema_and_points",
                    schema_ok and points_ok,
                    details,
                    f"{role} collection schema/points mismatch" if not (schema_ok and points_ok) else None,
                )
            )
    except Exception as exc:
        checks.append(_check("qdrant_reachable", False, error=str(exc)))
    finally:
        await client.close()

    return {"checks": checks}


async def inspect_neo4j() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        from neo4j import AsyncGraphDatabase  # type: ignore[import]
    except ImportError as exc:
        return {"checks": [_check("neo4j_import", False, error=str(exc))]}

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    driver = AsyncGraphDatabase.driver(
        uri,
        auth=(
            os.getenv("NEO4J_USERNAME", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password"),
        ),
    )
    try:
        async with driver.session() as session:
            label_counts: dict[str, int] = {}
            for label in EXPECTED_NEO4J_LABELS:
                result = await session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
                record = await result.single()
                label_counts[label] = int(record["count"]) if record else 0

            rel_counts: dict[str, int] = {}
            for rel_type in EXPECTED_NEO4J_RELATIONSHIPS:
                result = await session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count")
                record = await result.single()
                rel_counts[rel_type] = int(record["count"]) if record else 0

        node_total = sum(label_counts.values())
        rel_total = sum(rel_counts.values())
        details = {
            "uri": uri,
            "nodes": node_total,
            "relationships": rel_total,
            "labels": label_counts,
            "relationship_types": rel_counts,
        }
        baseline_ok = (
            node_total == 21
            and rel_total == 15
            and label_counts == EXPECTED_NEO4J_LABELS
            and rel_counts == EXPECTED_NEO4J_RELATIONSHIPS
        )
        rebuilt_ok = (
            node_total == 23
            and rel_total == 18
            and label_counts == REBUILT_NEO4J_LABELS
            and rel_counts == REBUILT_NEO4J_RELATIONSHIPS
        )
        expanded_ok = (
            node_total == 32
            and rel_total == 27
            and label_counts == EXPANDED_NEO4J_LABELS
            and rel_counts == EXPANDED_NEO4J_RELATIONSHIPS
        )
        passed = baseline_ok or rebuilt_ok or expanded_ok
        checks.append(
            _check(
                "neo4j_deterministic_graph",
                passed,
                details,
                "Neo4j deterministic graph counts differ from Phase 1 baseline" if not passed else None,
            )
        )
    except Exception as exc:
        checks.append(_check("neo4j_reachable", False, {"uri": uri}, str(exc)))
    finally:
        await driver.close()

    return {"checks": checks}


def inspect_runtime_code() -> dict[str, Any]:
    retriever_source = (PROJECT_ROOT / "src" / "database" / "retriever.py").read_text(encoding="utf-8")
    graph_store_source = (PROJECT_ROOT / "src" / "database" / "graph_store.py").read_text(encoding="utf-8")
    entity_retriever_source = (PROJECT_ROOT / "src" / "retrieval" / "entity_retriever.py").read_text(encoding="utf-8")
    query_normalization_source = (PROJECT_ROOT / "src" / "retrieval" / "query_normalization.py").read_text(encoding="utf-8")
    context_packer_source = (PROJECT_ROOT / "src" / "retrieval" / "context_packer.py").read_text(encoding="utf-8")
    reranker_source = (PROJECT_ROOT / "src" / "retrieval" / "reranker.py").read_text(encoding="utf-8")
    reranking_contracts_source = (PROJECT_ROOT / "src" / "retrieval" / "reranking" / "contracts.py").read_text(encoding="utf-8")
    reranking_providers_source = (PROJECT_ROOT / "src" / "retrieval" / "reranking" / "providers.py").read_text(encoding="utf-8")
    reranking_metrics_source = (PROJECT_ROOT / "src" / "retrieval" / "reranking" / "metrics.py").read_text(encoding="utf-8")
    agent_graph_source = (PROJECT_ROOT / "src" / "agent" / "graph.py").read_text(encoding="utf-8")
    llm_provider_source = (PROJECT_ROOT / "src" / "agent" / "llm" / "provider.py").read_text(encoding="utf-8")
    resilience_provider_path = PROJECT_ROOT / "src" / "resilience" / "provider.py"
    resilience_circuit_path = PROJECT_ROOT / "src" / "resilience" / "circuit_breaker.py"
    resilience_retry_path = PROJECT_ROOT / "src" / "resilience" / "retry.py"
    answer_verifier_path = PROJECT_ROOT / "src" / "quality" / "answer_verifier.py"
    severity_guard_path = PROJECT_ROOT / "src" / "quality" / "severity_guard.py"
    safe_fallback_path = PROJECT_ROOT / "src" / "quality" / "safe_fallback.py"
    fallback_node_path = PROJECT_ROOT / "src" / "agent" / "nodes" / "fallback.py"
    google_genai_adapter_path = PROJECT_ROOT / "src" / "integrations" / "google_genai.py"
    reproducible_checker_path = PROJECT_ROOT / "scripts" / "check_reproducible_environment.py"
    release_checker_path = PROJECT_ROOT / "scripts" / "check_release_readiness.py"
    answer_quality_eval_path = PROJECT_ROOT / "scripts" / "eval_phase2_answer_quality.py"
    safe_fallback_eval_path = PROJECT_ROOT / "scripts" / "eval_safe_fallback_flow.py"
    runtime_smoke_path = PROJECT_ROOT / "scripts" / "smoke_phase2_runtime.py"
    cache_versioning_path = PROJECT_ROOT / "src" / "observability" / "versioning.py"
    observability_exporter_path = PROJECT_ROOT / "src" / "observability" / "trace_exporter.py"
    neo4j_queries_path = PROJECT_ROOT / "src" / "database" / "neo4j_queries.py"
    neo4j_validator_path = PROJECT_ROOT / "scripts" / "validate_neo4j_schema.py"
    taxonomy_models_path = PROJECT_ROOT / "src" / "knowledge" / "taxonomy_models.py"
    taxonomy_validator_path = PROJECT_ROOT / "scripts" / "validate_taxonomy.py"
    taxonomy_qdrant_planner_path = PROJECT_ROOT / "scripts" / "plan_entity_index_update.py"
    taxonomy_graph_planner_path = PROJECT_ROOT / "scripts" / "plan_taxonomy_graph_update.py"
    all_eval_path = PROJECT_ROOT / "scripts" / "eval_phase2_all.py"
    cache_inspect_path = PROJECT_ROOT / "scripts" / "inspect_cache_versions.py"
    answer_verifier_source = answer_verifier_path.read_text(encoding="utf-8") if answer_verifier_path.exists() else ""
    severity_guard_source = severity_guard_path.read_text(encoding="utf-8") if severity_guard_path.exists() else ""
    safe_fallback_source = safe_fallback_path.read_text(encoding="utf-8") if safe_fallback_path.exists() else ""
    fallback_node_source = fallback_node_path.read_text(encoding="utf-8") if fallback_node_path.exists() else ""
    google_genai_adapter_source = google_genai_adapter_path.read_text(encoding="utf-8") if google_genai_adapter_path.exists() else ""
    runtime_smoke_source = runtime_smoke_path.read_text(encoding="utf-8") if runtime_smoke_path.exists() else ""
    cache_versioning_source = cache_versioning_path.read_text(encoding="utf-8") if cache_versioning_path.exists() else ""
    observability_exporter_source = observability_exporter_path.read_text(encoding="utf-8") if observability_exporter_path.exists() else ""
    neo4j_queries_source = neo4j_queries_path.read_text(encoding="utf-8") if neo4j_queries_path.exists() else ""
    taxonomy_models_source = taxonomy_models_path.read_text(encoding="utf-8") if taxonomy_models_path.exists() else ""

    return {
        "current_capabilities": {
            "entity_aware_retrieval": "EntityRetriever" in retriever_source
            and "normalize_query" in retriever_source
            and "expand_normalized_query" in retriever_source,
            "qdrant_dense_search": "dense_results = await self._vector_store.search(" in retriever_source,
            "qdrant_sparse_bm25_search": "search_sparse" in retriever_source,
            "rrf_fusion": "rrf_fusion" in retriever_source,
            "legacy_query_metadata_boost": "extract_dermatology_metadata" in retriever_source,
            "neo4j_graph_context": "Neo4jGraphStore" in retriever_source,
            "neo4j_canonical_name_compatible": "canonical_name" in graph_store_source,
            "entity_collection_runtime_retrieval": "EntityRetriever" in retriever_source
            and "get_entity_collection_name" in entity_retriever_source,
            "drug_entity_normalizer_runtime": "normalize_query" in retriever_source
            and "DrugEntityNormalizer" in query_normalization_source,
            "taxonomy_query_expansion": "expand_normalized_query" in retriever_source,
            "metadata_aware_chunk_boost": "boost_chunk_results" in retriever_source,
            "candidate_merge_trace": "RetrievalTrace" in retriever_source,
            "entity_aware_context_packing": "pack_context" in retriever_source
            and "PackedContext" in context_packer_source,
            "local_reranker_available": "def rerank_candidates" in reranker_source
            and "local_rules" in reranker_source,
            "reranker_provider_contract_present": "class RerankCandidate" in reranking_contracts_source
            and "class RerankScore" in reranking_contracts_source,
            "semantic_reranker_adapter_present": "local_files_only=True" in reranking_providers_source
            and "LocalSemanticReranker" in reranking_providers_source,
            "reranker_fallback_present": "falling back to local_rules" in reranker_source,
            "reranker_metrics_present": "def ranking_metrics" in reranking_metrics_source,
            "reranking_integrated": "rerank_candidates" in retriever_source
            and "rerank_trace" in retriever_source,
            "answer_quality_verifier_available": "def verify_answer_quality" in answer_verifier_source
            and "def apply_answer_guard" in answer_verifier_source,
            "vietnamese_answer_verifier_hardened": "answer_verifier_v2" in cache_versioning_source
            and "extract_domain_propositions" in answer_verifier_source,
            "runtime_timeout_enabled": "AgentTimeoutError" in agent_graph_source
            and "asyncio.timeout" in agent_graph_source,
            "runtime_retry_policy_enabled": resilience_retry_path.exists()
            and "RetryPolicy" in resilience_retry_path.read_text(encoding="utf-8")
            and "call_provider_with_resilience" in llm_provider_source,
            "runtime_circuit_breaker_enabled": resilience_circuit_path.exists()
            and "CircuitBreaker" in llm_provider_source,
            "runtime_resilience_versioned": "runtime_resilience_version" in cache_versioning_source,
            "neo4j_schema_contract_present": "CANONICAL_NODE_SCHEMAS" in (
                PROJECT_ROOT / "src" / "knowledge" / "graph_schema.py"
            ).read_text(encoding="utf-8"),
            "neo4j_schema_validator_present": neo4j_validator_path.exists(),
            "neo4j_legacy_property_warnings_resolved": ".name" not in neo4j_queries_source
            and ".description" not in neo4j_queries_source
            and ".evidence" not in neo4j_queries_source,
            "neo4j_schema_versioned": "neo4j_schema_version" in cache_versioning_source,
            "taxonomy_schema_v2_present": "TaxonomyCatalog" in taxonomy_models_source
            and "taxonomy_schema_v2" in taxonomy_models_source,
            "taxonomy_validator_present": taxonomy_validator_path.exists(),
            "taxonomy_alias_collision_check_present": "alias_collision" in taxonomy_models_source,
            "taxonomy_incremental_planner_present": taxonomy_qdrant_planner_path.exists()
            and taxonomy_graph_planner_path.exists(),
            "taxonomy_versioned": "taxonomy_version" in cache_versioning_source,
            "answer_guard_integrated": "answer_quality" in agent_graph_source
            and "cache_store" in agent_graph_source,
            "severity_aware_answer_guard_available": "def classify_medical_severity" in severity_guard_source
            and "def apply_severity_aware_answer_guard" in severity_guard_source
            and "severity_aware_answer_guard_v1" in cache_versioning_source,
            "safe_fallback_flow_available": "def decide_retrieval_fallback" in safe_fallback_source
            and "def build_safe_fallback_answer" in safe_fallback_source
            and "fallback_decision" in agent_graph_source
            and "safe_fallback" in agent_graph_source,
            "safe_fallback_eval_available": safe_fallback_eval_path.exists(),
            "safe_fallback_flow_versioned": "safe_fallback_flow_version" in cache_versioning_source
            and "safe_fallback_flow_v1" in cache_versioning_source
            and "safe_fallback_node" in fallback_node_source,
            "google_genai_sdk_migration_available": "build_google_genai_client" in google_genai_adapter_source
            and "generate_text_async" in google_genai_adapter_source
            and "embed_texts_sync" in google_genai_adapter_source,
            "google_genai_sdk_versioned": "google_genai_sdk_version" in cache_versioning_source
            and "google_genai_sdk_v1" in cache_versioning_source,
            "reproducible_environment_checker_available": reproducible_checker_path.exists(),
            "reproducible_environment_versioned": "reproducible_environment_version" in cache_versioning_source
            and "reproducible_environment_v1" in cache_versioning_source,
            "end_to_end_release_readiness_checker_available": release_checker_path.exists(),
            "end_to_end_release_readiness_versioned": "end_to_end_release_readiness_version" in cache_versioning_source
            and "end_to_end_release_readiness_v1" in cache_versioning_source,
            "answer_quality_eval_available": answer_quality_eval_path.exists(),
            "runtime_smoke_available": runtime_smoke_path.exists(),
            "offline_smoke_available": "def run_offline_smoke" in runtime_smoke_source
            and "--live-chat" in runtime_smoke_source,
            "cache_versioning_available": "build_pipeline_version_manifest" in cache_versioning_source
            and "compute_pipeline_fingerprint" in cache_versioning_source,
            "pipeline_fingerprint_available": "pipeline_fingerprint" in cache_versioning_source,
            "observability_available": "sanitize_for_observability" in observability_exporter_source
            and "export_observability_event" in observability_exporter_source,
            "cache_inspection_available": cache_inspect_path.exists(),
            "phase2_all_eval_available": all_eval_path.exists(),
        },
        "rerank_provider_default": _display_rerank_provider(),
        "reranker_pipeline_version": cache_versioning_source and "reranker_pipeline_v2" in cache_versioning_source,
        "semantic_model_available": _semantic_model_available(),
        "deferred_phase2_features": [
            "Add external rerank provider or installed local model reranker.",
            "Add optional LLM-backed medical answer reviewer for complex answers.",
            "Use deterministic Neo4j graph for deeper structured expansion beyond 1-hop supplemental facts.",
            "Web fallback is intentionally not implemented.",
            "Full clinical safety engine is intentionally out of scope for this deterministic guard.",
        ],
    }


def _display_rerank_provider() -> str:
    provider = rerank_provider_from_env().strip().lower()
    if provider in {"", "local", "local_rules"}:
        return "local_rules"
    if provider in {"local_model", "local_semantic", "local_cross_encoder", "semantic"}:
        suffix = "available" if _semantic_model_available() else "not provisioned; falls back to local_rules"
        return f"{provider} ({suffix})"
    if provider == "hybrid":
        suffix = "semantic model available" if _semantic_model_available() else "semantic model missing; falls back to local_rules"
        return f"hybrid ({suffix})"
    return f"{provider} (unknown; runtime falls back to local_rules)"


def _semantic_model_available() -> bool:
    config = semantic_config_from_env()
    return bool(config.model_path and Path(config.model_path).exists())


async def main() -> int:
    runtime_config = {
        "chunk_collection": get_chunk_collection_name(),
        "entity_collection": get_entity_collection_name(),
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "embedding": get_embedding_metadata(),
        "kb_version": os.getenv("KB_VERSION", "acne_kb_v1"),
        "prompt_version": os.getenv("PROMPT_VERSION", "medical_prompt_v2"),
        "cache_answer_version": get_answer_cache_version(),
    }

    qdrant, neo4j = await asyncio.gather(inspect_qdrant(), inspect_neo4j())
    checks = qdrant["checks"] + neo4j["checks"]
    runtime_code = inspect_runtime_code()

    report = {
        "passed": all(check["passed"] for check in checks),
        "runtime_config": runtime_config,
        "phase1_state_checks": checks,
        **runtime_code,
        "recommended_next_step": "Run Phase 2D offline answer quality and runtime smoke evals before any live chat smoke.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
