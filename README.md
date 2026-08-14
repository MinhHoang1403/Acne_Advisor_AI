# Acne Advisor AI

Acne Advisor AI is a Vietnamese medical-information assistant built on a
frozen, provenance-preserving knowledge foundation and a bounded LangGraph RAG
runtime. It provides educational information about acne and related skin care.
It does not diagnose, prescribe, or replace a qualified clinician.

## Design Principles

- No medical knowledge without provenance.
- No method without a source and an obvious code owner.
- Retrieval supplies evidence; the agent decides whether to retrieve, retry,
  generate, abstain, or finalize.
- Deterministic logic owns safety, provenance, bounded limits, contracts, and
  fail-safe behavior. It is not the primary semantic medical reasoner.
- Phase 1 knowledge stores are read-only during Phase 2 runtime.
- Tests protect implementation contracts; they are not final AI-quality proof.

## Architecture

The repository has two frozen system boundaries:

1. **Phase 1, knowledge foundation.** Four canonical sources are parsed,
   normalized, structurally chunked, embedded, indexed, and validated. The
   active build contains 512 knowledge chunks, 32 EntityCards, and a
   deterministic Neo4j graph with 32 nodes and 27 relationships.
2. **Phase 2, Agentic RAG runtime.** An eight-node LangGraph workflow performs
   read-only source retrieval with Dense + native BM25, equal-weight RRF, and a
   bounded provenance-preserving context packer. It then generates or abstains
   under deterministic safety and evidence contracts.

Actual request direction:

```text
React ChatInput
  -> App.handleSubmit / App.sendQuestion
  -> POST /chat through sendChatMessage
  -> FastAPI chat_endpoint
  -> run_clinical_agent
  -> LangGraph: prepare -> guard -> decide
       -> retrieve -> assess -> decide (maximum two attempts)
       -> generate or abstain
       -> finalize
  -> ChatResponse
  -> React session state
  -> ChatWindow -> ChatMessage -> answer and citations
```

Runtime generation does not query EntityCards or Neo4j. Those stores remain
frozen Phase 1 structural assets. There is no project-internal HTTP context API:
retrieval, packing, prompt assembly, and provider dispatch are Python calls.

## Technology

- Python 3.11.9, FastAPI, Pydantic, LangGraph
- Google Gemini and local Ollama generation providers
- Gemini Embedding 2, 3072 dimensions, cosine distance
- Qdrant named Dense vectors and Qdrant-native BM25
- PostgreSQL for chat history, Redis for versioned answer cache
- Neo4j for the frozen deterministic entity graph
- React, Vite, Vitest, ESLint
- Docker Compose for local infrastructure

## Phase 1

The only supported operator interface is [scripts/phase1.py](scripts/phase1.py).
The active build is `ec0a6de32d58ac181af6` and is frozen.

```text
4 canonical sources
  -> source parsing and normalization
  -> proof-only artifact filtering
  -> deterministic structure-first chunks (2400 Unicode chars, overlap 0)
  -> portable content-bound provenance identities
  -> Gemini Embedding 2 Dense vectors + Qdrant-native BM25
  -> immutable Qdrant knowledge/entity build candidates
  -> deterministic Neo4j entity graph
  -> validation and controlled activation
```

Key owners:

| Concern | Canonical location |
|---|---|
| Operator interface | `scripts/phase1.py` |
| Parsing and compilation | `src/ingestion/pipeline.py` |
| Structural chunking | `src/ingestion/chunking.py::structural_chunks` |
| Provenance identities | `src/ingestion/provenance.py` |
| Native BM25 contract | `src/ingestion/bm25.py` |
| Qdrant build indexing | `src/ingestion/index.py` |
| Source manifest | `data/sources/manifest.yaml` |
| Method/source mapping | `data/phase1_method_sources.json` |

Read-only integrity checks:

```powershell
.\venv\Scripts\python.exe scripts\phase1.py status
.\venv\Scripts\python.exe scripts\phase1.py validate
```

`build` and `--activate` are controlled migration operations. Do not rebuild,
re-embed, reindex, or activate Phase 1 for ordinary API/frontend development.
See [Data Pipeline](docs/DATA_PIPELINE.md) before any approved source change.

## Phase 2

[src/agent/graph.py](src/agent/graph.py) compiles the production LangGraph with
exactly eight semantic nodes:

```text
START -> prepare -> guard -> decide
                          |-> retrieve -> assess -> decide
                          |-> generate -> finalize -> END
                          |-> abstain  -> finalize -> END
                          `-> finalize -> END
```

`decide_node` selects a typed action from `retrieve`, `generate`, `abstain`, or
`finalize`. `MAX_RETRIEVAL_ATTEMPTS = 2`; the loop is bounded. Cache hits and
out-of-domain responses finalize without retrieval. Provenance-complete source
evidence routes to generation. This check establishes presence and identity
only, not semantic sufficiency. Exhausted evidence routes to safe abstention.

The runtime has no active reranker, Candidate Policy, selector, metadata score
boost, EntityCard retrieval, or Graph retrieval. Historical P3/P4 subsystems are
not active production stages.

## Retrieval

[src/retrieval/service.py](src/retrieval/service.py) is the sole production
evidence service:

```text
query
  -> embed_query -> Qdrant named vector "dense"
  -> Qdrant-native sparse vector "bm25"
  -> reciprocal_rank_fusion
  -> pack_context
  -> provenance-complete source contexts
```

Active defaults are `RETRIEVAL_CANDIDATE_LIMIT=16`,
`RETRIEVAL_CONTEXT_MAX_ITEMS=8`, `RETRIEVAL_CONTEXT_MAX_CHARS=6000`, and
`RETRIEVAL_TIMEOUT_SECONDS=20`. RRF uses `k=60`, Dense weight `1.0`, and BM25
weight `1.0`. Definitions and formulas are in
[Methods and Formulas](docs/METHODS_AND_FORMULAS.md).

## End-to-End Request Lifecycle

1. `src/frontend/src/components/ChatInput.jsx` captures the question.
2. `src/frontend/src/App.jsx::handleSubmit` delegates to `sendQuestion`.
3. `src/frontend/src/api/chatApi.js::sendChatMessage` sends `POST /chat` with a
   finite 225-second browser timeout.
4. `src/api/app.py::ChatRequest` validates the request, including the supported
   `gemini`, `ollama`, and `local` provider values.
5. `src/api/app.py::chat_endpoint` loads bounded history and invokes
   `src/agent/graph.py::run_clinical_agent`.
6. LangGraph prepares, guards, decides, retrieves, assesses, and then generates
   or abstains. The retrieval loop can execute at most twice.
7. `src/agent/nodes/reason.py::generate_answer_node` calls
   `src/agent/prompts/medical_answer.py::build_medical_prompt` with the user
   question and packed source evidence.
8. `src/agent/llm/provider.py::generate_llm_response` dispatches to the selected
   provider. Gemini reaches `src/integrations/google_genai.py`; Ollama reaches
   `src/agent/llm/ollama_client.py`.
9. Finalization validates source mentions, applies safety/presentation rules,
   writes only eligible answers to the versioned Redis cache, and exports
   bounded observability metadata.
10. `ChatResponse` returns the answer, display sources, source metadata, safety
    fields, compatibility `graph_facts`, and runtime metadata.
11. `App.sendQuestion` updates React session state. `ChatWindow.jsx` selects the
    conversation and `ChatMessage.jsx` renders Markdown, badges, and citations.

## FastAPI Backend

Application entrypoint: `src.api.app:app`. The current route inventory is:

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | bounded dependency and generation-runtime health |
| GET | `/retrieve?q=...` | operator diagnostic retrieval; disabled by default |
| GET | `/models` | generation model catalog |
| POST | `/chat` | complete guarded Agentic RAG request |
| GET | `/chat/sessions` | list persisted sessions |
| DELETE | `/chat/sessions` | delete app-owned chat history and answer-cache keys |
| GET | `/chat/sessions/{session_id}/messages` | list session messages |
| PATCH | `/chat/sessions/{session_id}/rename` | rename a session |
| PATCH | `/chat/sessions/{session_id}/hide` | mark a session hidden |
| POST | `/chat/sessions/sync` | external/backward-compatible history import |

`/retrieve` is diagnostics, not an alternate evidence pipeline. It returns 404
unless `ENABLE_DIAGNOSTIC_RETRIEVE=true` is explicitly set by a trusted local
operator. It uses the same Dense + BM25 + RRF service as the agent. OpenAPI/Swagger is available at
`http://127.0.0.1:8000/docs`.

## React Frontend

The frontend root is `src/frontend`; `src/main.jsx` mounts `App.jsx`.
`App.jsx` owns sessions, selected session, connectivity, model selection, and
submit state. `ChatInput.jsx` owns composer interaction, `chatApi.js` owns HTTP,
`ChatWindow.jsx` arranges messages, and `ChatMessage.jsx` renders answers and
citations. `presentationMetadata.js::sourceDisplayLabels` converts trusted
source metadata into friendly labels while raw source IDs remain in backend
metadata.

The only browser environment variable is `VITE_API_URL`. Vite exposes `VITE_*`
values to client code, so never place API keys, passwords, or private URLs in a
`VITE_*` variable.

## Repository Map

| Location | Responsibility |
|---|---|
| `scripts/phase1.py` | supported Phase 1 build/validate/status interface |
| `src/ingestion/` | frozen parsing, chunking, filtering, provenance, indexing |
| `src/knowledge/` | taxonomy, source-backed EntityCards, deterministic graph |
| `src/retrieval/` | Dense + BM25 + RRF and bounded context packing |
| `src/agent/` | LangGraph state, decisions, generation, presentation |
| `src/quality/` | deterministic safety, verification, fallback contracts |
| `src/api/` | FastAPI routes and runtime preflight |
| `src/cache/`, `src/database/` | answer cache and application persistence |
| `src/integrations/` | external provider SDK adapters |
| `src/frontend/` | React/Vite client |
| `docs/` | eight canonical project documents |
| `tests/` | implementation and regression contracts |

## Prerequisites

- Python `3.11.9`
- pip `26.1.2`
- Node.js/npm compatible with `src/frontend/package-lock.json`
- Docker Desktop with Compose
- Ollama with `qwen3:8b` only when using the local provider/fallback
- A Gemini key only for live Gemini embedding/generation operations

## Installation

```powershell
git clone https://github.com/MinhHoang1403/RAG-system-for-acne-diagnose.git
Set-Location RAG-system-for-acne-diagnose

py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip==26.1.2
python -m pip install -r requirements.lock.txt
Copy-Item .env.example .env
```

Fill only the secrets required by the chosen provider in local `.env`. Never
commit that file. Local Docker Qdrant works with an empty `QDRANT_API_KEY`;
secured Qdrant uses the same key through the runtime and preflight clients.

Start infrastructure and initialize application schemas:

```powershell
docker compose up -d --pull never --no-build
docker compose ps
.\venv\Scripts\python.exe scripts\init_schema.py
.\venv\Scripts\python.exe scripts\init_chat_schema.py
```

Do not run a Phase 1 build merely to start the API or UI. The frozen knowledge
foundation must already exist in the local stores.

## Environment and Security

`.env.example` documents non-secret defaults and blank secret slots. Important
groups are provider selection, embedding identity, frozen version markers,
PostgreSQL/Neo4j/Qdrant/Redis connections, retrieval limits, answer safety,
runtime resilience, and observability. `CACHE_ANSWER_VERSION=v7` is the current
cache namespace.

Provider fallback is disabled by default. It is effective only when both
`LLM_PROVIDER_FALLBACK_ENABLED=true` and request field
`allow_model_fallback=true`; the frontend request opt-in also defaults to false.

This repository currently provides **no end-user authentication or tenant
authorization**. Chat-history endpoints can enumerate or change local history.
The supported trust boundary is therefore single-user development on loopback
addresses (`127.0.0.1`/`localhost`). Do not expose the API or databases to an
untrusted network. A public deployment requires authentication, authorization,
TLS, restrictive CORS, rate limiting, secret management, and tenant isolation.

Docker Compose binds project services to localhost. FastAPI error responses and
preflight details do not return raw datastore exception text. The frontend
receives no provider secret; browser configuration contains only `VITE_API_URL`.

## Local Development

The bounded launcher starts Docker, reuses an already healthy backend, and does
not kill unknown processes:

```powershell
.\scripts\start_local_dev.ps1
```

Manual backend:

```powershell
.\venv\Scripts\python.exe -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

Manual frontend:

```powershell
Set-Location src\frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

## Testing

Backend and contract checks:

```powershell
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m compileall -q src scripts tests
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe scripts\inspect_phase2_readiness.py
.\venv\Scripts\python.exe scripts\pre_ui_runtime_check.py
.\venv\Scripts\python.exe scripts\check_reproducible_environment.py
.\venv\Scripts\python.exe scripts\check_release_readiness.py --mode offline
.\venv\Scripts\python.exe scripts\check_phase2_contracts.py
```

Frontend:

```powershell
Set-Location src\frontend
npm ci
npm test
npm run lint
npm run build
npm audit
```

Controlled smoke tests prove integration wiring only. They do not establish
medical correctness, retrieval quality, or final AI quality. Final evaluation
methodology is a separate E0 research task.

## Provenance, Citations, and Cache

Phase 1 source records carry stable source/document/record/chunk identities,
content hashes, build/version markers, and ingestion timestamps. Retrieval keeps
those payloads through RRF and context packing. `build_source_allowlist` and
`validate_answer_source_mentions` prevent the final answer from citing sources
outside the request evidence. FastAPI returns both friendly `sources` and raw
traceable `source_metadata`; React renders the friendly labels.

Source allowlisting validates source identity and display attribution. It does
not prove that every answer claim is entailed by a cited chunk.

`cache_lookup_node` and `cache_store_node` use the same resolved model identity,
pipeline fingerprint, prompt/version manifest, and answer-cache namespace.
Out-of-domain, failed, fallback, unsafe, or quality-rejected answers are not
stored as ordinary reusable answers.

## Safety

- The assistant does not diagnose or prescribe.
- Emergency, self-harm, acne-fulminans, high-risk pregnancy, prescription
  refusal, and abstention contracts are deterministic and regression-tested.
- Insufficient provenance-complete evidence triggers bounded retry and then a
  safe abstention rather than an unsupported answer.
- Provider/runtime failure produces structured 503/504 responses or a defined
  safe fallback; frontend network and request timeouts remain finite.
- Do not put identifying patient information in committed files or reports.

See [Safety](docs/SAFETY.md) for the medical boundary and authoritative safety
references.

## Methods and References

The project distinguishes scientific sources, provider contracts, framework
documentation, and clinical references:

- [Methods and Formulas](docs/METHODS_AND_FORMULAS.md)
- [References](docs/REFERENCES.md)
- [Phase 1 method/source map](data/phase1_method_sources.json)
- [Canonical source manifest](data/sources/manifest.yaml)

Official runtime contracts include the
[LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api),
[FastAPI response models](https://fastapi.tiangolo.com/tutorial/response-model/),
[FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/),
[Qdrant hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/),
[Qdrant BM25](https://qdrant.tech/documentation/inference/inference-bm25/),
[Gemini generateContent](https://ai.google.dev/api/generate-content),
[React state](https://react.dev/learn/adding-interactivity), and
[Vite environment variables](https://vite.dev/guide/env-and-mode).

## Canonical Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data Pipeline](docs/DATA_PIPELINE.md)
- [Agent Workflow](docs/AGENT_WORKFLOW.md)
- [Methods and Formulas](docs/METHODS_AND_FORMULAS.md)
- [References](docs/REFERENCES.md)
- [Safety](docs/SAFETY.md)
- [Operations](docs/OPERATIONS.md)

These files describe the current frozen architecture. Git history and merged
pull requests retain development history; active docs do not serve as a changelog.

## Frozen Architecture and Evaluation Status

Phase 1 build `ec0a6de32d58ac181af6` and the S4B eight-node Phase 2 architecture
are frozen. S4D verifies implementation, contracts, security configuration, and
end-to-end traceability without changing either semantic architecture. No final
AI-quality claim is made here. E0 must first research and approve evaluation
methodology through the stated research cutoff.

## License

Project metadata in `pyproject.toml` declares MIT. No standalone `LICENSE` file
is currently present. Confirm institutional, source-corpus, and redistribution
requirements before publishing medical documents or generated artifacts.
