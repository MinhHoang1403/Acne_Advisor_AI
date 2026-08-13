# Architecture

## Current Runtime

Acne Advisor AI has two maintained paths:

1. Phase 1 converts controlled source documents into Qdrant chunks, EntityCards,
   and a deterministic Neo4j graph.
2. Phase 2 serves a FastAPI chat API whose production orchestrator is the
   21-node LangGraph `StateGraph` in `src/agent/graph.py`.

The active responsibilities are:

| Responsibility | Canonical location |
|---|---|
| API and preflight | `src/api/` |
| Phase 1 primitives | `src/ingestion/` |
| Phase 1 operator commands | `scripts/run_full_phase1.py`, `scripts/ingest_knowledge.py` |
| Vector/graph/database adapters | `src/database/` |
| Retrieval policy and formulas | `src/retrieval/` |
| Taxonomy, entities and graph schema | `src/knowledge/` |
| LangGraph workflow and generation | `src/agent/` |
| Safety and answer verification | `src/quality/` |
| Cache | `src/cache/` |
| Provider resilience | `src/resilience/` |
| Frontend | `src/frontend/` |

## Compatibility Boundaries

- Qdrant uses named dense vector `dense` with 3072 dimensions.
- Qdrant uses sparse storage key `bm25`, but its current formula is custom
  MD5-indexed normalized log-TF, not canonical BM25.
- P3 permits at most two retrieval attempts.
- P4 claim grounding remains shadow behavior.
- Entity and graph signals supplement retrieval; they are not substitutes for
  source-backed chunk evidence.

## Planned Work

S4A owns Phase 1 methodology and true-BM25 decisions. S4B owns any major agent,
candidate-policy, reranker, graph/tool, or ClinicalState redesign. These are not
current runtime claims.
