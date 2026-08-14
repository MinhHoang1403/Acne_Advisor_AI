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

`S4B_ARCHITECTURE_FROZEN` in `src/observability/versioning.py` freezes this
production architecture for E0/E1 evaluation-methodology work.
