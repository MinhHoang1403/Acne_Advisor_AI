# Architecture

Acne Advisor AI separates a frozen knowledge foundation from a small runtime:

1. Phase 1 owns parsing, chunking, provenance, embedding, native BM25,
   EntityCards, and the Neo4j taxonomy graph.
2. Phase 2 is an eight-node LangGraph agent over the frozen Qdrant knowledge
   collection. It does not write to Phase 1 stores.

```text
User -> FastAPI -> LangGraph
                     |
                     +-> narrow source-mapped safety policy / exact cache
                     +-> model-selected typed action
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
| Narrow deterministic safety overrides | `src/agent/safety_policy.py` |
| Structural/provenance verification | `src/quality/answer_verifier.py` |
| Exact normalized answer cache | `src/cache/exact_cache.py` |
| API and persistence | `src/api/`, `src/database/` |

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
| `src/quality` | structural/provenance verification and safe fallback contracts |
| `src/integrations` | external generation and embedding provider adapters |
| `src/api` | HTTP boundary and dependency preflight |
| `src/cache`, `src/database` | answer cache and storage adapters |

PostgreSQL stores chat history, Redis stores versioned eligible answers,
Qdrant serves read-only runtime evidence, and Neo4j stores the frozen structural
graph. Only Qdrant knowledge chunks participate in online medical grounding.
The Redis cache uses exact normalized question identity plus provider, model,
pipeline fingerprint, and cache version; it performs no semantic lookup.

## Application Boundary

`src/api/app.py::chat_endpoint` is the primary HTTP entrypoint and invokes
`src/agent/graph.py::run_clinical_agent` directly. Retrieval, context packing,
prompt assembly, and provider dispatch remain internal Python calls; there is
no separate project-internal context HTTP service.

The React client starts at `src/frontend/src/main.jsx`. `App.jsx::handleSubmit`
calls `chatApi.js::sendChatMessage`, then updates session state from
`ChatResponse`. `ChatWindow.jsx` and `ChatMessage.jsx` render the answer and
source labels. Provider credentials remain backend-only.

Generation passes policy through the provider's real system-instruction field.
The question, bounded history, source allowlist, and exact packed evidence stay
inside delimited user data. Presentation code does not contain a normal medical
answer table, and the verifier does not judge ordinary medical truth.

The supported deployment boundary is local, single-user development. The
chat-history routes do not implement end-user authentication or tenant
authorization, so a public deployment requires an authenticated TLS boundary,
authorization, restrictive CORS, rate limiting, and tenant isolation.
