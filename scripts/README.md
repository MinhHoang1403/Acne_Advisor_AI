# Supported Commands

This directory contains operator entrypoints and provider-free regression gates. Application logic belongs under `src/`; scripts should remain thin where practical. Historical forensic and obsolete benchmark runners are intentionally absent and remain available through Git history.

## Phase 1 and Knowledge Foundation

- `run_full_phase1.py`: authoritative complete core Phase 1 coordinator.
- `ingest_knowledge.py`: lower-level incremental ingestion and compatibility CLI.
- `run_semantic_enrichment.py`: optional semantic graph enrichment, outside the canonical core build.
- `build_entity_index.py`, `build_entity_graph.py`: deterministic entity index and graph builders.
- `inspect_ingestion_manifest.py`, `validate_phase1_complete.py`, `eval_phase1_readiness.py`: manifest and build validation.
- `validate_kb_collections.py`, `validate_neo4j_schema.py`, `validate_taxonomy.py`: datastore and taxonomy contracts.
- `inspect_taxonomy_candidates.py`, `plan_entity_index_update.py`, `plan_taxonomy_graph_update.py`: read-only or dry-run maintenance planning.

The current Qdrant sparse storage key remains `bm25` for compatibility, but the implementation is a custom hashed normalized log-TF vector, not true BM25. Its migration is reserved for S4A.

## Datastore Administration

- `init_schema.py`, `init_chat_schema.py`: initialize supported schemas without resetting existing data.
- `clear_redis_cache.py`: explicit operator command for answer-cache cleanup.

## Runtime Readiness

- `check_release_readiness.py`, `check_reproducible_environment.py`: release and environment contracts.
- `inspect_phase2_readiness.py`, `pre_ui_runtime_check.py`, `smoke_phase2_runtime.py`: Phase 2 readiness and bounded smoke checks.
- `inspect_cache_versions.py`: cache and pipeline fingerprint inspection.

## Locked Regression Gates

- `eval_retrieval_v5_release.py`: provider-free Retrieval V5 contract.
- `eval_p3_evidence_sufficiency.py`: provider-free evidence sufficiency and bounded-retry contract.
- `eval_phase2_all.py`: aggregate supported Phase 2 regression checks.
- `eval_phase2_retrieval.py`, `eval_phase2_context_packing.py`, `eval_phase2_reranking.py`, `eval_phase2_answer_quality.py`: focused Phase 2 contracts.
- `eval_runtime_resilience.py`, `eval_safe_fallback_flow.py`, `eval_semantic_reranker.py`: resilience, fallback and reranker contracts.

These are regression gates, not clinical validation or benchmark gold. The retired synthetic 300-question evaluator is not part of the active architecture.
