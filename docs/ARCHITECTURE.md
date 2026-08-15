# Architecture

Acne Advisor AI separates knowledge preparation from online answer generation.

1. **Knowledge preparation and indexing** owns parsing, normalization, chunking,
   provenance, Dense embeddings, native BM25 documents, EntityCards, and the
   Neo4j taxonomy graph.
2. **The Agentic RAG runtime** reads the validated Qdrant knowledge index through
   an eight-node LangGraph workflow. It does not write to the indexed medical
   knowledge during normal requests.

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
                     +-> assess evidence presence and source identity
                     +-> retrieve again when requested (maximum two executions)
                     +-> generate or abstain
                     +-> presentation, provenance, cache, and observability
```

| Responsibility | Canonical location |
|---|---|
| Knowledge compilation and activation | `src/ingestion/`, `scripts/knowledge_build.py` |
| Dense + BM25 + RRF retrieval | `src/retrieval/service.py`, `src/retrieval/rrf.py` |
| Bounded context packing | `src/retrieval/context_packer.py` |
| LangGraph orchestration and decisions | `src/agent/graph.py`, `src/agent/nodes/workflow.py` |
| Generation and presentation | `src/agent/nodes/reason.py`, `src/agent/nodes/respond.py` |
| Narrow deterministic safety boundaries | `src/agent/safety_policy.py` |
| Structural/provenance verification | `src/quality/answer_verifier.py` |
| Exact normalized answer cache | `src/cache/exact_cache.py` |
| API and persistence | `src/api/`, `src/database/` |

The runtime has one typed evidence tool: `retrieve_evidence`. EntityCards and
Neo4j remain structural knowledge assets for build validation, inspection, and
research. They are intentionally absent from the normal answer path and do not
count as runtime medical evidence.

The architecture marker and pipeline fingerprint are computed in
`src/observability/versioning.py`. Their serialized values are cache and
diagnostic compatibility contracts; they do not select alternate runtime
implementations.

## Package Boundaries

| Package | Current responsibility |
|---|---|
| `src/ingestion` | content-addressed knowledge compilation and controlled activation |
| `src/knowledge` | taxonomy identities, EntityCards, and deterministic graph build |
| `src/retrieval` | source-evidence retrieval and context packing |
| `src/agent` | state graph, action decisions, generation, safety, and presentation |
| `src/quality` | structural/provenance verification and safe fallback contracts |
| `src/integrations` | external generation and embedding provider adapters |
| `src/api` | HTTP boundary and dependency preflight |
| `src/cache`, `src/database` | answer cache and application storage adapters |

PostgreSQL stores chat history, Redis stores versioned eligible answers,
Qdrant serves read-only runtime evidence, and Neo4j stores the structural graph.
Only Qdrant knowledge chunks participate in online medical grounding. Redis
uses exact normalized question identity plus provider, model, pipeline
fingerprint, and cache version; it performs no semantic lookup.

## Application Boundary

`src/api/app.py::chat_endpoint` is the primary HTTP entrypoint and calls
`src/agent/graph.py::run_clinical_agent` directly. Retrieval, context packing,
prompt assembly, and provider dispatch remain internal Python calls; there is
no separate project-internal context service.

The React client starts at `src/frontend/src/main.jsx`. `App.jsx::handleSubmit`
calls `chatApi.js::sendChatMessage`, then updates session state from
`ChatResponse`. `ChatWindow.jsx` and `ChatMessage.jsx` render answers and source
labels. Provider credentials remain backend-only.

Generation sends policy through the provider's system-instruction field. The
question, bounded history, source allowlist, and exact packed evidence remain
inside delimited user data. Presentation code does not contain a normal medical
answer table, and the verifier does not judge ordinary medical truth.

The supported deployment boundary is local, single-user development. Chat
history routes do not implement end-user authentication or tenant authorization.
A public deployment requires authentication, TLS, authorization, restrictive
CORS, rate limiting, tenant isolation, and operational hardening.
