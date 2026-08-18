# Acne Advisor AI

Acne Advisor AI is a Vietnamese, evidence-grounded acne information assistant
built with bounded Agentic RAG. It retrieves medical evidence, lets a language
model decide whether to retrieve again, generate, or abstain, and applies
deterministic boundaries for a small set of safety-critical situations.

The system is not a diagnostic model, an autonomous physician, a prescription
service, an image-analysis system, or a clinically validated medical product.

## Overview

The repository contains two cooperating parts:

1. **Knowledge preparation and indexing** converts curated medical source
   snapshots into versioned, provenance-preserving Qdrant indexes and structural
   knowledge assets.
2. **The Agentic RAG runtime** accepts questions through FastAPI, uses LangGraph
   to select bounded actions, retrieves evidence with Dense and BM25 search,
   generates or abstains, and returns an answer to the React interface.

Responsibilities are intentionally separate:

- retrieval supplies bounded source evidence;
- the language model selects semantic actions and synthesizes normal answers;
- Python validates transitions and enforces safety, provenance, finite budgets,
  cache eligibility, and fail-safe behavior;
- FastAPI transports the result and React presents it.

## Architecture

```text
KNOWLEDGE PREPARATION

curated source snapshots
  -> deterministic parsing and normalization
  -> structure-aware chunking and provenance identities
  -> Gemini Embedding 2 + Qdrant-native BM25 documents
  -> versioned Qdrant knowledge/entity indexes
  -> EntityCards and a deterministic Neo4j structural graph

AGENTIC RAG RUNTIME

React -> FastAPI -> START -> prepare -> guard -> decide
                                             +-- retrieve/retry
                                             |      -> retrieve -> assess -> decide
                                             +-- generate -> generate -> finalize -> END
                                             +-- abstain  -> abstain  -> finalize -> END
                                             `-- cache/safety -> finalize -> END
```

The eight LangGraph nodes are `prepare`, `guard`, `decide`, `retrieve`,
`assess`, `generate`, `abstain`, and `finalize`. `retry` is an Agent action that
routes through the existing `retrieve` node, not a separate graph node.

Normal runtime requests read the existing indexed medical knowledge. They do
not rebuild, re-embed, reindex, activate, or write to the knowledge indexes.

## Knowledge Preparation and Indexing

The current validated build is `94d613bc9b33628de3ef`:

| Artifact | Count or contract |
|---|---|
| Curated source snapshots | 4 |
| Knowledge chunks | 512 |
| EntityCards | 32 |
| Neo4j graph | 32 nodes / 27 relationships |
| Qdrant knowledge index | 512 points |
| Qdrant entity index | 32 points |
| Dense vectors | `models/gemini-embedding-2`, 3072 dimensions, cosine distance |
| Sparse vectors | Qdrant-native BM25 |

Source, document, record, and chunk identities are content-bound. The pipeline
uses structure-aware chunks capped at 2400 Unicode characters with zero overlap,
proof-based artifact filtering, exact deduplication, and complete provenance.
The deterministic graph is built from source-backed taxonomy data; no LLM graph
extraction runs in the current pipeline.

The supported operator interface is
[`scripts/knowledge_build.py`](scripts/knowledge_build.py):

```powershell
.\venv\Scripts\python.exe scripts\knowledge_build.py status
.\venv\Scripts\python.exe scripts\knowledge_build.py validate
```

Normal application development uses the existing indexed build. A controlled
rebuild is needed only when source data, indexing configuration, provider
compatibility, or the stored indexes materially change. Build and activation
procedures are documented in [Data Pipeline](docs/DATA_PIPELINE.md) and
[Operations](docs/OPERATIONS.md).

EntityCards, the entity Qdrant index, and the Neo4j graph are maintained as
structural knowledge assets for validation, inspection, and research. They are
not queried when generating normal medical answers.

## Agentic RAG Runtime

The language model selects one of four typed actions:

| Action | Meaning |
|---|---|
| `retrieve` | acquire evidence for the first time |
| `retry` | make a later evidence request after a previous retrieval |
| `generate` | answer from provenance-complete evidence |
| `abstain` | stop without making an unsupported medical claim |

Python validates whether each selected action is legal. Invalid model output,
impossible transitions, repeated non-recoverable queries, and exhausted limits
become abstention. The retrieval tool can execute at most twice.

The normal request lifecycle is:

```text
ChatInput -> App -> POST /chat -> run_clinical_agent
  -> prepare -> guard -> decide -> retrieve -> assess -> decide
  -> generate / retry / abstain -> finalize -> ChatResponse
  -> ChatWindow -> ChatMessage and citations
```

Cache hits and narrow safety overrides take a deterministic path to `finalize`.
For normal generation, the model receives the current question, bounded
conversation context when applicable, the exact packed evidence, and a source
allowlist. Retrieval, context packing, prompt assembly, and provider dispatch
are internal Python calls rather than separate HTTP services.

The answer path does not contain a reranker, Candidate Policy, metadata score
boost, evidence selector, EntityCard retrieval, graph retrieval, medical
proposition engine, semantic cache, or deterministic normal-answer engine.

## Retrieval

[`src/retrieval/service.py`](src/retrieval/service.py) owns runtime evidence
retrieval:

```text
query
  +-> Gemini query embedding -> Qdrant "dense"
  +-> query text             -> Qdrant "bm25"
                                  |
                                  v
                    Reciprocal Rank Fusion (RRF)
                                  |
                                  v
                    bounded provenance context
```

| Default | Value |
|---|---:|
| Candidate limit per channel | 16 |
| Selected context items | 8 |
| Packed context characters | 6000 |
| Timeout per channel | 20 seconds |
| RRF `k` | 60 |
| Dense / BM25 weights | 1.0 / 1.0 |

Dense and BM25 execute concurrently with independent timeouts. If Dense fails
but BM25 returns evidence, the result is preserved as `degraded_dense`. If BM25
fails but Dense succeeds, it is preserved as `degraded_bm25`. If neither
channel yields usable evidence, the Agent can request one legal retry and then
abstains when the retrieval budget is exhausted.

The formulas, provider contracts, and parameter classifications are documented
in [Methods and Formulas](docs/METHODS_AND_FORMULAS.md).

## Evidence and Citations

The deterministic `assess` node confirms only that packed evidence contains
text and a source identifier. It does not determine medical relevance,
completeness, truth, or semantic sufficiency. The model's `decide` action makes
the semantic sufficiency decision.

Source allowlisting prevents an answer from displaying a source outside the
request's retrieved evidence. This validates source identity, but it does not
prove that every sentence or medical claim is entailed by its cited chunk.

[`src/quality/answer_verifier.py`](src/quality/answer_verifier.py) checks
presentation, structural contracts, and provenance identity. Claim-level
groundedness and medical correctness are evaluated separately; they are not
deterministically established by the current verifier.

## Safety

[`src/agent/safety_policy.py`](src/agent/safety_policy.py) defines nine narrow,
source-mapped deterministic boundaries:

- anaphylaxis-like breathing difficulty with swelling or hives;
- active breathing difficulty after medication use;
- significant bleeding after acne manipulation;
- chest pain or tightness with breathlessness;
- explicit personal and current self-harm or suicide intent;
- acne fulminans-like severe lesions with fever or joint pain;
- isotretinoin during pregnancy or pregnancy planning;
- isotretinoin with severe headache plus visual or gastrointestinal symptoms;
- explicit requests to prescribe, select a drug, or select a dose.

These rules are action-oriented safeguards, not a general medical classifier.
The self-harm detector is intentionally narrow and is not a validated
suicide-risk classifier. Ordinary medical questions continue through retrieved
evidence and model synthesis. See [Safety](docs/SAFETY.md) for the exact trigger
boundaries and sources.

## Cache and Reliability

Redis stores exact normalized answers, not semantic-similarity matches. Cache
identity includes the schema and answer versions, pipeline fingerprint,
normalized question, provider, and model. Stored metadata retains selected
evidence identifiers and the source allowlist. Answers involving history,
safety overrides, fallback, failed retrieval, or failed quality checks are not
reused as ordinary cache entries.

The effective answer-cache namespace is `v10`. The pipeline fingerprint is
computed from a secret-free runtime manifest. Provider calls, retrieval
channels, the overall Agent request, and frontend requests use finite timeouts.
Retries are bounded, and provider fallback requires both server configuration
and request-level opt-in.

## Data Stores and Providers

| Component | Responsibility | Runtime role |
|---|---|---|
| Qdrant knowledge index | Dense and BM25 medical evidence | required for normal RAG |
| Google embedding API | 3072-dimensional Dense query embeddings | required by current preflight; BM25 can preserve evidence during an individual Dense-channel failure |
| Gemini or Ollama | action selection and answer generation | one configured generation path is required |
| PostgreSQL | chat/session persistence and history endpoints | optional for a single chat response; history features depend on it |
| Redis | exact answer cache | optional; cache failure becomes a miss |
| Neo4j | structural graph and integrity inspection | not used for runtime medical grounding |
| Qdrant entity index | EntityCard lookup asset | not used by the normal answer path |

Ollama can run answer generation locally, but Dense retrieval still uses Gemini
Embedding 2 under the current embedding configuration. Selecting Ollama does
not make the complete RAG pipeline fully local or offline.

## API

The FastAPI entrypoint is `src.api.app:app`.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | bounded dependency and provider readiness |
| `GET` | `/retrieve?q=...` | trusted-operator retrieval diagnostics, disabled by default |
| `GET` | `/models` | available generation models |
| `POST` | `/chat` | guarded Agentic RAG request |
| `GET` | `/chat/sessions` | list persisted sessions |
| `DELETE` | `/chat/sessions` | delete application chat history and answer-cache keys |
| `GET` | `/chat/sessions/{session_id}/messages` | list session messages |
| `PATCH` | `/chat/sessions/{session_id}/rename` | rename a session |
| `PATCH` | `/chat/sessions/{session_id}/hide` | hide a session |
| `POST` | `/chat/sessions/sync` | import compatible history records |

`/retrieve` returns 404 unless a trusted local operator enables
`ENABLE_DIAGNOSTIC_RETRIEVE`. It uses the same retrieval service as the Agent
and is not a separate production answer API.

`ChatResponse` includes friendly citations and structured source/runtime
metadata for local research, debugging, and provenance inspection. The current
API is not hardened as an untrusted multi-tenant public interface. Local OpenAPI
documentation is available at `http://127.0.0.1:8000/docs`.

## Frontend

The React request and render path is:

```text
ChatInput -> App.sendQuestion -> chatApi.sendChatMessage -> FastAPI
  -> ChatResponse -> session state -> ChatWindow -> ChatMessage/citations
```

Source metadata is converted to friendly labels for display while the backend
retains raw source identity. `VITE_API_URL` is the only browser configuration.
All `VITE_*` values are visible to browser users, so provider keys and database
credentials belong only in backend configuration.

## Deployment Boundary

The current deployment configuration targets trusted, local, single-user
research and development. Docker Compose binds project services to
`127.0.0.1`. Chat/session routes do not provide end-user authentication,
tenant-level authorization, multi-tenant isolation, production rate limiting,
or distributed session locking.

Public deployment requires authentication, authorization, TLS, restrictive
CORS, rate limiting, tenant isolation, secret management, and operational
hardening.

## Repository Structure

| Location | Responsibility |
|---|---|
| `scripts/knowledge_build.py` | knowledge build, validation, and status interface |
| `src/ingestion/` | parsing, chunking, provenance, validation, and indexing |
| `src/knowledge/` | taxonomy, EntityCards, and deterministic graph assets |
| `src/retrieval/` | Dense + native BM25 + RRF evidence retrieval |
| `src/agent/` | LangGraph state, decisions, generation, safety, and presentation |
| `src/quality/` | structural/provenance verification and safe fallback contracts |
| `src/cache/` | exact Redis answer cache |
| `src/resilience/` | deadlines, retries, and provider-failure contracts |
| `src/api/` | FastAPI routes and dependency preflight |
| `src/database/` | Qdrant adapter and PostgreSQL persistence |
| `src/integrations/` | external provider SDK adapters |
| `src/frontend/` | React/Vite interface |
| `docs/` | architecture, methods, safety, data, and operations documentation |
| `tests/` | implementation and regression contracts |

## Local Setup

### Prerequisites

- Python 3.11 (`3.11.9` in CI) and pip `26.1.2`
- Node.js 24 and npm
- Docker Desktop with Compose
- a Gemini API key for live Gemini generation and Dense query embeddings
- Ollama with `qwen3:8b` when Ollama generation or fallback is enabled

### Installation

```powershell
git clone https://github.com/MinhHoang1403/Acne_Advisor_AI.git
Set-Location Acne_Advisor_AI

py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip==26.1.2
python -m pip install -r requirements.lock.txt
Copy-Item .env.example .env
```

Fill only the secrets required by the selected providers, and keep `.env` out
of version control. Local Docker Qdrant works with an empty `QDRANT_API_KEY`;
secured Qdrant passes a configured key through runtime and preflight clients.

Start the backing services and initialize application schemas:

```powershell
docker compose pull
docker compose up -d --pull never --no-build
docker compose ps
.\venv\Scripts\python.exe scripts\init_schema.py
.\venv\Scripts\python.exe scripts\init_chat_schema.py
```

These commands initialize application infrastructure but do not create the
indexed medical knowledge. An existing development environment can verify its
current build with `scripts/knowledge_build.py status`. A new environment must provision
the validated indexes through the controlled knowledge-build procedure in
[Operations](docs/OPERATIONS.md).

Start the application with the bounded local launcher:

```powershell
.\scripts\start_local_dev.ps1
```

Or start each layer manually:

```powershell
.\venv\Scripts\python.exe -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000

Set-Location src\frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

Environment settings are grouped in [`.env.example`](.env.example) by provider,
embedding, retrieval, datastore, reliability, versioning, and observability.

## Testing

Backend and runtime contracts:

```powershell
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m compileall -q src scripts tests
.\venv\Scripts\python.exe -m ruff check src scripts tests
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe scripts\check_runtime_contracts.py
.\venv\Scripts\python.exe scripts\pre_ui_runtime_check.py
.\venv\Scripts\python.exe scripts\check_reproducible_environment.py
.\venv\Scripts\python.exe scripts\check_release_readiness.py --mode offline
```

Knowledge build validation:

```powershell
.\venv\Scripts\python.exe scripts\knowledge_build.py status
.\venv\Scripts\python.exe scripts\knowledge_build.py validate
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

Unit and integration tests verify software contracts, graph bounds, regression
behavior, cache and retrieval wiring, provenance handling, and environment
readiness. They do not establish representative retrieval quality, medical
correctness, claim-level faithfulness, clinical effectiveness, or deployment
readiness. Those qualities require separate system evaluation and clinical
review.

## Limitations

- The indexed corpus contains four curated acne source snapshots and is not
  comprehensive dermatology coverage.
- The NICE-derived snapshot has a provenance discrepancy: official NICE
  metadata reports 30 April 2026, while the local snapshot represents 3 August
  2026. Its exact current-version provenance has not been independently
  reconciled.
- Source allowlisting does not prove sentence-level or claim-level entailment.
- End-to-end medical quality and clinical safety effectiveness have not been
  clinically validated.
- Dense query embedding uses the external Gemini Embedding 2 provider under the
  current configuration. The current query/document contract does not prepend
  Google's retrieval-specific task instructions; their effect on this corpus
  remains an evaluation question that would require controlled re-embedding.
- Deployment assumptions target trusted, local, single-user use.

## References and Technical Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Agent Workflow](docs/AGENT_WORKFLOW.md)
- [Data Pipeline](docs/DATA_PIPELINE.md)
- [Methods and Formulas](docs/METHODS_AND_FORMULAS.md)
- [Method Traceability](docs/METHOD_TRACEABILITY.md)
- [Vietnamese Source Code Guide](docs/CODE_GUIDE_VI.md)
- [Safety](docs/SAFETY.md)
- [Operations](docs/OPERATIONS.md)
- [References](docs/REFERENCES.md)
- [Method and source registry](data/method_sources.json)
- [Source manifest](data/sources/manifest.yaml)

Project metadata declares the MIT license in `pyproject.toml`. No standalone
`LICENSE` file is currently present. Source-corpus, institutional, and
redistribution requirements should be reviewed before publishing medical
documents or generated artifacts.
