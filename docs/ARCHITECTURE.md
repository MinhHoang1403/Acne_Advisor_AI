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
                     |      +-> local BGE cross-encoder reranking
                     |      +-> whole-chunk provenance packing
                     +-> assess evidence presence and source identity
                     +-> retrieve again when requested (maximum two executions)
                     +-> generate or abstain
                     +-> presentation, provenance, and cache
       -> best-effort PostgreSQL persistence
       -> request-complete observability event
```

The compiled graph topology is:

```text
START -> prepare -> guard -> decide
                           | retrieve -> assess -> decide
                           | generate -> finalize -> END
                           | abstain  -> finalize -> END
                           | finalize -> END
```

`guard` always evaluates deterministic safety before exact-cache lookup. A safety
override or eligible cache hit routes directly to `finalize` and does not call the
decision model, retrieval, reranking, or generation. Every other graph cycle is
bounded by `ActionPolicy`; retrieval can execute at most twice, and all stages
share a finite request deadline.

| Responsibility | Canonical location |
|---|---|
| Knowledge compilation and activation | `src/ingestion/`, `scripts/knowledge_build.py` |
| Dense + BM25 + RRF retrieval | `src/retrieval/service.py`, `src/retrieval/rrf.py` |
| Local candidate reranking | `src/retrieval/reranker.py` |
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

## Online Runtime Responsibilities

The first `ActionDecision` owns semantic routing and standalone acquisition-query
rewrite. This model output is a strict action object, not an answer or free-form
reasoning channel. Keeping the rewrite is an empirical project decision: a
fixed-input study found that using the normalized user question directly reduced
RRF acquisition recall from `1.00` to `0.95` and removed all required evidence
for one reviewed case. Python `ActionPolicy` owns legal transitions, query novelty,
evidence-ID membership, and attempt limits; invalid proposals fail closed to
abstention.

Each retrieval executes Dense and native BM25 search, equal-weight RRF, local BGE
cross-encoder reranking, and whole-chunk context packing. The first acquisition
query is also its rerank query. A retry uses the targeted evidence-gap query for
both new acquisition and reranking. Prior and newly acquired candidates are
merged, stable-deduplicated, and the complete union is reranked before packing.
The full-union invariant is retained because an evaluated depth cap discarded a
reviewed required candidate before the reranker could score it.

The BGE owner is one process-wide `CandidateReranker`. The requested production
configuration is CUDA BF16, batch 4, with current CrossEncoder max-length
behavior. Unsupported BF16 or CPU execution uses FP32 and records requested and
effective settings. Load, inference, timeout, or runtime failure preserves the
deterministic RRF order; it does not create a second reranking implementation.

The packer consumes the supplied order, keeps chunks whole, deduplicates by stable
identity, preserves provenance, and enforces the configured item and character
budgets. Dropped and not-examined candidates retain explicit reasons. These
limits are resource policies, not medical confidence or relevance thresholds.

`assess` checks only that packed evidence contains non-empty text and a source
identity. The post-retrieval `ActionDecision` separately owns semantic sufficiency,
direct-support evidence IDs, a targeted retry gap, or abstention. Generation sees
the exact bounded packed context through delimited user data and receives its
behavioral policy as a system instruction.

Final verification is deliberately structural. It checks requested output shape,
requested-entity scope, citation identifiers, source allowlisting, and provenance.
It does not prove semantic entailment, clinical truth, or general evidence
completeness. Those semantic quality questions belong to the offline evaluation
boundary, not to a hidden runtime verifier model.

## Failure and Fallback Contract

| Failure point | Bounded behavior |
|---|---|
| Safety match | Produce the source-mapped system response and bypass cache/model/retrieval/generation. |
| Exact cache unavailable or miss | Continue through the normal graph; cache failure is non-fatal. |
| Invalid or unavailable ActionDecision | Fail closed to abstention or the owned infrastructure fallback within the request budget. |
| One retrieval channel fails | Continue with the other channel and record degraded status. |
| Both channels fail with retained retry candidates | Rerank and repack retained candidates. |
| Both channels fail without retained candidates | Return a retrieval-owned error/fallback; never invent evidence. |
| BGE load/inference/timeout failure | Preserve deterministic pre-reranker RRF order and record the fallback. |
| No provenance-complete packed evidence | Retry when legally proposed and budget remains; otherwise abstain. |
| Generation provider failure | Use only the configured bounded provider fallback, then a safe owned fallback. |
| Invalid citation/source/critical response contract | Remove invalid source presentation or use a safe fallback according to severity. |
| Cache write or telemetry sink failure | Keep the completed response and report the non-semantic operational failure. |

No fallback changes Qdrant, Neo4j, the active knowledge build, or benchmark truth.

The architecture marker and pipeline fingerprint are computed in
`src/observability/versioning.py`. Their serialized values are cache and
diagnostic compatibility contracts; they do not select alternate runtime
implementations.

The fingerprint includes knowledge, embedding, retrieval, reranker, retry,
packing, decision, generation, safety, verification, and resilience contracts.
It records the configured reranker precision; host-specific effective precision
and fallback reason are runtime telemetry. Exact answer-cache keys additionally
include cache schema/version, exact normalized question, provider, and model.

The FastAPI boundary creates one opaque UUID `request_id` before invoking the
Agent. The same identity is retained in Agent state, retrieval/generation
diagnostics, safe database metadata, the HTTP response metadata, and the final
observability event. Request-complete telemetry is emitted only after Agent and
best-effort persistence timings are available. Telemetry construction and sink
failures remain non-fatal to the chat response.

When a client supplies that UUID again for the same canonical request, database
persistence derives stable user/assistant message IDs and replays the existing
logical turn instead of duplicating it. Different request IDs keep identical
text as distinct legitimate turns. The frontend reconciles server and local
history by message ID, preserving an unsynchronized local tail across refreshes
without merging repeated text or leaking state across a new chat/session switch.

## Package Boundaries

| Package | Current responsibility |
|---|---|
| `src/ingestion` | content-addressed knowledge compilation and controlled activation |
| `src/knowledge` | taxonomy identities, EntityCards, and deterministic graph build |
| `src/retrieval` | source-evidence retrieval, local reranking, and context packing |
| `src/agent` | state graph, action decisions, generation, safety, and presentation |
| `src/quality` | structural/provenance verification and safe fallback contracts |
| `src/integrations` | external generation and embedding provider adapters |
| `src/api` | HTTP boundary and dependency preflight |
| `src/cache`, `src/database` | answer cache and application storage adapters |

## Lifecycle and Evaluation Boundary

FastAPI startup preloads the process-wide reranker. Shutdown closes its executor
and the optional Langfuse sink. Qdrant, embedding, and the configured generation
provider are core preflight dependencies. PostgreSQL, Redis, and Neo4j are
reported separately and are optional for the online grounding core; Ollama is
required only when configured as primary or required fallback.

Formal evaluation runs outside the request graph. It may score captured system
outputs, retrieved evidence, citations, latency, fallback accounting, and semantic
quality, but it cannot change runtime state, source labels, or benchmark truth.
Deterministic checks remain authoritative for safety outcomes, evidence-gap
classification, citation IDs, provenance, pipeline identity, retry accounting,
provider/fallback identity, latency, and runtime errors. RAGChecker remains the
current semantic evaluator unless a separate calibrated migration is approved.

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

## Accepted Limitations

- BF16 quality and latency evidence applies to the measured GPU/library/model-cache
  environment; truthful FP32/RRF fallbacks remain required.
- The validated retry improvement is narrow: one of six reviewed evidence items
  became packed on the fixed retry study. It is not a universal quality claim.
- The decision model sees fewer bounded evidence excerpts than generation; this
  is an evaluation question, not an unbounded prompt path.
- Context budgets are versioned engineering policies, not proven optimal values.
- Neo4j is not an online evidence source, and its unavailability does not remove
  already activated Qdrant evidence.
- The supported deployment is local and single-user. The current HTTP history
  routes are not a public multi-tenant authorization boundary.

The supported deployment boundary is local, single-user development. Chat
history routes do not implement end-user authentication or tenant authorization.
A public deployment requires authentication, TLS, authorization, restrictive
CORS, rate limiting, tenant isolation, and operational hardening.
