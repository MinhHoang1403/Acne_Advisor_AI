# Architecture

Acne Advisor AI separates a frozen knowledge foundation from a small runtime:

1. Phase 1 owns parsing, chunking, provenance, embedding, native BM25,
   EntityCards, and the Neo4j taxonomy graph.
2. Phase 2 is an eight-node LangGraph agent over the frozen Qdrant knowledge
   collection. It does not write to Phase 1 stores.

```text
User -> FastAPI -> LangGraph
                     |
                     +-> deterministic domain/severity guard
                     +-> decide action
                     +-> retrieve_evidence
                     |      +-> Dense search
                     |      +-> native BM25 search
                     |      +-> equal-weight RRF
                     |      +-> bounded provenance packer
                     +-> assess evidence
                     +-> retrieve again (maximum two attempts)
                     +-> generate or abstain
                     +-> final safety, presentation, cache, observability
```

| Responsibility | Canonical location |
|---|---|
| Frozen Phase 1 methods | `src/ingestion/`, `scripts/phase1.py` |
| Dense + BM25 + RRF retrieval | `src/retrieval/service.py`, `src/retrieval/rrf.py` |
| One bounded context packer | `src/retrieval/context_packer.py` |
| LangGraph orchestration and decisions | `src/agent/graph.py`, `src/agent/nodes/workflow.py` |
| Generation and final presentation | `src/agent/nodes/reason.py`, `src/agent/nodes/respond.py` |
| Deterministic medical safety | `src/quality/`, `src/agent/nodes/severity.py` |
| API, cache, persistence | `src/api/`, `src/cache/`, `src/database/` |

The runtime has one typed evidence tool: `retrieve_evidence`. EntityCards and
Neo4j remain frozen Phase 1 assets for offline inspection and future
evidence-driven proposals; they are intentionally absent from the production
answer path and never count as medical evidence.

The active architecture marker and pipeline fingerprint are computed in
`src/observability/versioning.py`. Its serialized architecture value remains an
external cache/diagnostic compatibility contract; it does not select an older
runtime implementation.

## Package Boundaries

| Package | Current responsibility |
|---|---|
| `src/ingestion` | immutable Phase 1 compilation and activation contracts |
| `src/knowledge` | taxonomy identities, EntityCards and deterministic graph build |
| `src/retrieval` | one source-evidence retrieval service |
| `src/agent` | state graph, generation and presentation |
| `src/quality` | deterministic safety and answer-quality gates |
| `src/integrations` | external generation and embedding provider adapters |
| `src/api` | HTTP boundary and dependency preflight |
| `src/cache`, `src/database` | answer cache and storage adapters |

PostgreSQL stores chat history, Redis stores versioned eligible answers,
Qdrant serves read-only runtime evidence, and Neo4j stores the frozen structural
graph. Only Qdrant knowledge chunks participate in online medical grounding.
