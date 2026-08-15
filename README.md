# Acne Advisor AI

Acne Advisor AI is a bounded, evidence-grounded research system for Vietnamese
acne information and skin-care advice. It combines a frozen medical knowledge
foundation with a LangGraph Agentic RAG runtime. It is not an autonomous doctor,
a diagnosis or prescription system, an image-based acne model, or a clinically
validated decision-support product.

> The repository slug `RAG-system-for-acne-diagnose` is historical. The current
> product provides acne information and advisory support; it does not diagnose.

## What This Project Is

The system answers text questions by retrieving bounded source excerpts and
asking a generation model to synthesize an answer from that evidence. It keeps
source identity through retrieval and exposes citations and research metadata.
For a small, explicit set of high-risk situations, deterministic policy can
replace the normal RAG answer with urgent or policy guidance.

The design separates responsibilities deliberately:

- **Phase 1 prepares evidence.** It compiles canonical source snapshots into
  immutable, versioned knowledge assets.
- **Retrieval finds evidence.** Dense and native BM25 channels search the frozen
  Qdrant knowledge collection.
- **The Agent decides.** A generation model selects one bounded semantic action:
  `retrieve`, `retry`, `generate`, or `abstain`.
- **The answer model synthesizes.** Normal medical meaning comes from retrieved
  evidence and the configured Gemini or Ollama generation provider.
- **Python enforces contracts.** It owns narrow safety overrides, legal state
  transitions, finite budgets, provenance, cache eligibility, and fail-safe
  behavior. It is not a general medical reasoner.
- **FastAPI transports and React displays.** Neither layer supplies medical
  knowledge.

## System at a Glance

| Area | Current verified state |
|---|---|
| Product boundary | acne medical-information/advisory research system |
| **ACTIVE PHASE 1** | frozen build `ec0a6de32d58ac181af6` |
| Frozen corpus | 4 source snapshots, 512 knowledge chunks |
| Structural assets | 32 EntityCards; Neo4j 32 nodes / 27 relationships |
| Qdrant | 512 knowledge points; 32 entity points |
| Dense embedding | `models/gemini-embedding-2`, 3072 dimensions, cosine distance |
| Sparse retrieval | Qdrant-native BM25 |
| **ACTIVE RUNTIME** | 8-node LangGraph with 4 model-selectable actions |
| Runtime contract | `minimal_agentic_rag_v1`; effective answer-cache namespace `v8` |
| Retrieval budget | at most 2 actual retrieval executions |
| Generation | Gemini or Ollama/local; provider fallback is opt-in |
| Cache | exact normalized, versioned Redis answer cache; not semantic cache |
| Deployment boundary | local, single-user research/development |
| Research status | runtime architecture frozen; final E0 AI-quality evaluation not started |

These counts are recorded in
[`data/phase1_build_manifest.json`](data/phase1_build_manifest.json) and checked
by `scripts/phase1.py status` and `scripts/phase1.py validate`.

## Architecture

The two phases have different lifecycles and ownership:

```text
PHASE 1 - FROZEN KNOWLEDGE FOUNDATION

canonical source snapshots
  -> deterministic parsing and normalization
  -> structure-first chunking and provenance identities
  -> Gemini Embedding 2 + Qdrant-native BM25 documents
  -> immutable Qdrant knowledge/entity collections
  -> deterministic EntityCards and Neo4j structural graph

PHASE 2 - BOUNDED AGENTIC RAG RUNTIME

React -> FastAPI -> START -> prepare -> guard -> decide
                                             +-- AI: retrieve/retry
                                             |      -> retrieve -> assess -> decide
                                             +-- AI: generate -> generate -> finalize -> END
                                             +-- AI: abstain  -> abstain  -> finalize -> END
                                             `-- Python: cache/safety -> finalize -> END
```

`retry` is a semantic action routed to the existing `retrieve` node; it is not a
ninth graph node. Phase 2 reads the frozen Phase 1 knowledge collection and does
not rebuild, re-embed, reindex, activate, or write to Phase 1 stores.

The normal request path is:

```text
ChatInput -> App -> chat API client -> POST /chat -> run_clinical_agent
  -> prepare -> guard -> decide -> retrieve -> assess -> decide
  -> generate / retry / abstain -> finalize -> ChatResponse
  -> ChatWindow -> ChatMessage and citations
```

See [Architecture](docs/ARCHITECTURE.md) and
[Agent Workflow](docs/AGENT_WORKFLOW.md) for the maintained technical view.

## Why It Is Agentic RAG

LangGraph topology alone is not the agency claim. At the `decide` boundary, the
configured model must return strict `AgentDecision` JSON with one of four
semantic actions:

| Action | Meaning |
|---|---|
| `retrieve` | first evidence acquisition, legal only before retrieval has run |
| `retry` | later evidence acquisition after a previous attempt |
| `generate` | answer from provenance-complete evidence |
| `abstain` | stop without an unsupported medical answer |

The model owns semantic action selection, including whether evidence addresses
the question. Python validates the schema and whether the transition is legal.
Invalid output, impossible transitions, repeated non-recoverable queries, or an
exhausted budget fail closed to abstention. The retrieval tool can execute at
most twice; a third execution is rejected by both the transition validator and
graph regression tests.

Cache hits and narrow safety overrides bypass model action selection and route
deterministically to `finalize`. This is **bounded Agentic RAG**, not unrestricted
autonomy and not deterministic rules presented as agent reasoning.

## Phase 1 - Frozen Knowledge Foundation

The supported operator interface is [`scripts/phase1.py`](scripts/phase1.py).
Build `ec0a6de32d58ac181af6` is frozen and currently validates as:

```text
4 canonical source snapshots
  -> 512 provenance-complete knowledge chunks
  -> 32 source-backed EntityCards
  -> deterministic graph: 32 nodes / 27 relationships
  -> Qdrant: 512 knowledge points / 32 entity points
```

The pipeline uses deterministic, structure-first chunks capped at 2400 Unicode
characters with zero overlap. Source, document, record, and chunk identities
are content-bound. Dense vectors use Gemini Embedding 2 at 3072 dimensions with
cosine distance; sparse documents use Qdrant-native BM25. The build has no LLM
semantic graph-extraction stage.

| Concern | Owner |
|---|---|
| operator interface | `scripts/phase1.py` |
| parsing and orchestration | `src/ingestion/` |
| structure-first chunking | `src/ingestion/chunking.py` |
| provenance identities | `src/ingestion/provenance.py` |
| Dense and native BM25 indexing | `src/ingestion/embedding.py`, `src/ingestion/bm25.py`, `src/ingestion/index.py` |
| taxonomy, EntityCards, graph | `src/knowledge/` |
| source registry | `data/sources/manifest.yaml` |
| build machine record | `data/phase1_build_manifest.json` |

EntityCards, the entity Qdrant collection, and Neo4j are **FROZEN STRUCTURAL
ASSETS**. They remain valid Phase 1 outputs for build validation, structural
inspection, and future evidence-driven research. They are not dead code, but
they are absent from normal runtime medical grounding.

Use only read-only checks during ordinary development:

```powershell
.\venv\Scripts\python.exe scripts\phase1.py status
.\venv\Scripts\python.exe scripts\phase1.py validate
```

`build` and `--activate` are controlled migration operations. Do not rebuild
Phase 1 to start the API, tune a runtime result, or prepare E0. See
[Data Pipeline](docs/DATA_PIPELINE.md) before any approved source migration.

## Phase 2 - Runtime

[`src/agent/graph.py`](src/agent/graph.py) compiles these eight semantic nodes:

`prepare`, `guard`, `decide`, `retrieve`, `assess`, `generate`, `abstain`, and
`finalize`.

Important paths are:

- **Normal evidence path:** prepare, narrow guard/cache check, model decision,
  retrieval, deterministic evidence-presence assessment, another model decision,
  generation or abstention, then finalization.
- **Exact-cache path:** an eligible cache hit routes from `decide` directly to
  `finalize`; no retrieval or generation is repeated.
- **Safety-override path:** a matching narrow safety rule supplies a deterministic
  answer, clears normal evidence attribution, bypasses cache reuse, and finalizes.
- **No-evidence/failure path:** the model may request one legal retry; after the
  finite budget, the system abstains rather than fabricating evidence.

Normal generation receives the current question, bounded conversational context
when applicable, the exact packed evidence, and a source allowlist. The prompt
requires evidence-grounded synthesis, but this instruction does not guarantee
perfect medical correctness or claim entailment.

Retrieval, context packing, prompt assembly, and provider dispatch are internal
Python calls. There is no separate project-internal context HTTP service.

The **ACTIVE RUNTIME** has no reranker, Candidate Policy, post-RRF metadata
boost, selector, EntityCard retrieval, Neo4j/graph retrieval, medical proposition
engine, semantic cache, or deterministic normal-medical-answer engine. Historical
experiments and frozen structural assets do not silently participate in answers.

## Retrieval

[`src/retrieval/service.py`](src/retrieval/service.py) is the sole production
evidence service:

```text
query
  +-> Gemini query embedding -> Qdrant named vector "dense"
  +-> query text             -> Qdrant native sparse vector "bm25"
                                  |
                                  v
                    equal-weight Reciprocal Rank Fusion
                                  |
                                  v
                   bounded provenance context packer
                                  |
                                  v
                               evidence
```

| Runtime default | Value |
|---|---:|
| candidate limit per channel | 16 |
| selected context items | at most 8 |
| packed context characters | at most 6000 |
| timeout per retrieval channel | 20 seconds |
| RRF `k` | 60 |
| Dense / BM25 weights | 1.0 / 1.0 |

Dense and BM25 run concurrently with independent finite outcomes:

| Channel outcome | Runtime behavior |
|---|---|
| selected evidence and neither channel raised | normal RRF, status `ok` |
| Dense timeout/failure, BM25 succeeds | preserve BM25 evidence, status `degraded_dense` |
| BM25 timeout/failure, Dense succeeds | preserve Dense evidence, status `degraded_bm25` |
| both return no evidence | status `no_evidence` |
| both fail, or failure leaves no usable channel | recoverable failure for bounded retry/fail-safe handling |

One channel failure does not discard the other channel's evidence. Degraded
answers remain subject to the same provenance, evidence, generation, and cache
eligibility contracts. Definitions, formulas, parameter classifications, and
method sources are in [Methods and Formulas](docs/METHODS_AND_FORMULAS.md).

## Evidence, Grounding, and Limitations

The deterministic `assess` node proves only **provenance-complete evidence
presence**: at least one packed item has non-empty text and a source identifier.
It does not prove relevance, completeness, medical truth, or semantic
sufficiency. The model-selected `decide` action evaluates semantic sufficiency.

Final source allowlisting ensures that a displayed source belongs to the
request's retrieved evidence. This is **source validity**, not proof of **claim
faithfulness**. The current runtime does not deterministically verify that every
sentence or atomic medical claim is entailed by its cited chunk.

[`src/quality/answer_verifier.py`](src/quality/answer_verifier.py) checks answer
presentation, structural contracts, and provenance identity. It is not a
clinical-truth verifier, a semantic entailment model, or a complete claim-level
grounding verifier. Claim-level groundedness is an E0 measurement target; this
README task does not add a mitigation subsystem before that evidence exists.

## Safety

[`src/agent/safety_policy.py`](src/agent/safety_policy.py) is the one deterministic
owner for seven narrow, source-mapped overrides:

- anaphylaxis-like breathing difficulty with swelling/hives;
- chest pain or tightness with breathlessness;
- explicit personal/current self-harm or suicide intent;
- acne fulminans-like severe lesions with fever or joint pain;
- isotretinoin in pregnancy or pregnancy planning;
- isotretinoin with severe headache plus visual or gastrointestinal symptoms;
- explicit requests to prescribe, choose a drug, or choose a dose.

These are high-precision action boundaries, not a comprehensive medical
classifier. The self-harm phrase detector is intentionally narrow and is not a
validated suicide-risk classifier. Ordinary medical semantics continue through
retrieved evidence and model synthesis. Tests protect known rule behavior; they
do not establish clinical sensitivity, specificity, or patient safety.

See [Safety](docs/SAFETY.md) for trigger boundaries and authoritative sources.

## Cache and Resilience

Redis stores an **exact normalized answer cache**, not a semantic cache. Its key
identity contains cache schema/version, pipeline fingerprint, exact normalized
question, provider, and model. Stored metadata retains the selected evidence
IDs and source allowlist. History-bearing, safety, fallback, failed, or
quality-rejected paths are not reused as ordinary cache answers.

`CACHE_ANSWER_VERSION=v8` is the effective namespace. The pipeline fingerprint
is computed from the current secret-free runtime manifest; it is not hardcoded.
Provider calls, retrieval channels, frontend requests, and the total agent run
have finite timeouts. Provider retries are bounded and classified as transient
or permanent. Provider fallback is disabled by default and requires both server
configuration and per-request opt-in.

## Data Stores and External Providers

| System | Current role | Runtime requirement |
|---|---|---|
| Qdrant knowledge collection | read-only Dense + BM25 medical evidence | core for normal RAG |
| Google embedding API | 3072-dimensional Dense query embedding | core preflight dependency; BM25 can preserve evidence during an individual Dense-channel failure |
| Gemini or Ollama | action selection and answer generation | one configured generation path is required |
| PostgreSQL | chat/session persistence and history endpoints | optional/degradable for one chat response; history features depend on it |
| Redis | exact eligible-answer cache | optional/degradable; a miss continues normally |
| Neo4j | frozen Phase 1 structural graph and integrity tooling | optional for Phase 2; not runtime medical grounding |
| Qdrant entity collection | frozen EntityCard index | not queried by the normal answer path |

Choosing Ollama changes answer generation only. Dense query embedding still
uses the configured Google Gemini Embedding 2 provider, so Ollama generation
does not make the entire RAG pipeline fully local or offline.

## API

The application entrypoint is `src.api.app:app`.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | bounded dependency and provider readiness |
| `GET` | `/retrieve?q=...` | trusted-operator retrieval diagnostics; disabled by default |
| `GET` | `/models` | available generation-model catalog |
| `POST` | `/chat` | complete guarded Agentic RAG request |
| `GET` | `/chat/sessions` | list persisted chat sessions |
| `DELETE` | `/chat/sessions` | delete app-owned chat history and answer-cache keys |
| `GET` | `/chat/sessions/{session_id}/messages` | list persisted session messages |
| `PATCH` | `/chat/sessions/{session_id}/rename` | rename a session |
| `PATCH` | `/chat/sessions/{session_id}/hide` | hide a session |
| `POST` | `/chat/sessions/sync` | import external/backward-compatible history |

`/retrieve` returns 404 unless a trusted local operator sets
`ENABLE_DIAGNOSTIC_RETRIEVE=true`. It uses the same retrieval service as the
agent and is not a parallel production RAG API.

`ChatResponse` exposes friendly source labels plus structured source/runtime
metadata useful for local research, debugging, evaluation, and provenance
inspection. The response contract has not been sanitized as an untrusted,
multi-tenant public API. Swagger is available locally at
`http://127.0.0.1:8000/docs`.

## Frontend

The React request/render path is intentionally small:

```text
ChatInput -> App.sendQuestion -> chatApi.sendChatMessage -> FastAPI
  -> ChatResponse -> React session state -> ChatWindow -> ChatMessage/citations
```

`src/frontend/src/utils/presentationMetadata.js` maps trusted source metadata to
friendly display labels while the backend retains raw source identity. The only
browser configuration is `VITE_API_URL`. Every `VITE_*` value is browser-visible;
never put provider keys, database credentials, or private secrets there.

## Deployment Boundary

The supported deployment is **local, single-user research and development** on
loopback addresses. Docker Compose binds the project services to `127.0.0.1`.
The application has an in-process session lock, but it does not provide public
end-user authentication, tenant authorization, multi-tenant isolation,
production rate limiting, distributed session locking, or a hardened public
security boundary.

A public multi-user deployment requires additional authentication,
authorization, TLS, restrictive CORS, rate limiting, tenant isolation, secret
management, and operational hardening. This is a declared thesis-prototype scope
boundary, not a claim of production deployment readiness.

## Repository Map

| Location | Responsibility |
|---|---|
| `scripts/phase1.py` | supported Phase 1 build/validate/status interface |
| `src/ingestion/` | Phase 1 parsing, chunking, provenance, validation, and indexing |
| `src/knowledge/` | Phase 1 taxonomy, EntityCards, and deterministic graph assets |
| `src/retrieval/` | active Dense + native BM25 + RRF evidence retrieval |
| `src/agent/` | LangGraph state, model decisions, generation, safety, and presentation |
| `src/quality/` | structural/provenance verification and safe fallback contracts |
| `src/cache/` | exact Redis answer cache |
| `src/resilience/` | finite deadlines, retries, and provider-failure contracts |
| `src/api/` | FastAPI routes and dependency preflight |
| `src/database/` | Qdrant adapter and PostgreSQL persistence |
| `src/integrations/` | external provider SDK adapters |
| `src/frontend/` | React/Vite user interface |
| `docs/` | canonical architecture, methods, safety, data, and operations docs |
| `tests/` | implementation and regression contracts |

## Local Development

### Prerequisites

- Python 3.11 (`3.11.9` in CI) and pip `26.1.2`
- Node.js 24 and npm for the frontend
- Docker Desktop with Compose
- a Gemini API key for live Gemini generation and Dense query embeddings
- Ollama with `qwen3:8b` only when Ollama generation or fallback is enabled

### Install

```powershell
git clone https://github.com/MinhHoang1403/RAG-system-for-acne-diagnose.git
Set-Location RAG-system-for-acne-diagnose

py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip==26.1.2
python -m pip install -r requirements.lock.txt
Copy-Item .env.example .env
```

Fill only the secrets required by the selected providers. Never commit `.env`.
Local Docker Qdrant works with an empty `QDRANT_API_KEY`; secured Qdrant passes a
non-empty key through runtime and preflight clients.

Start pinned backing services and initialize application schemas:

```powershell
docker compose pull
docker compose up -d --pull never --no-build
docker compose ps
.\venv\Scripts\python.exe scripts\init_schema.py
.\venv\Scripts\python.exe scripts\init_chat_schema.py
```

These commands do not create or mutate the frozen Phase 1 knowledge foundation.
Normal development assumes its Qdrant/Neo4j bind-mounted data has been
provisioned. Confirm it with the read-only Phase 1 checks above. A fresh machine
without those stores needs an explicitly approved provisioning/build workflow;
Phase 1 is not an automatic API-startup step.

Start the bounded local launcher:

```powershell
.\scripts\start_local_dev.ps1
```

Or start each application manually:

```powershell
.\venv\Scripts\python.exe -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000

Set-Location src\frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

Environment variables are grouped in [`.env.example`](.env.example): generation,
embedding, retrieval budgets, Qdrant, PostgreSQL, Redis, Neo4j, resilience,
versioning, and observability. Do not expose backend secrets to the frontend.

## Validation and Testing

Backend and repository contracts:

```powershell
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m compileall -q src scripts tests
.\venv\Scripts\python.exe -m ruff check src scripts tests
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe scripts\check_phase2_contracts.py
.\venv\Scripts\python.exe scripts\inspect_phase2_readiness.py
.\venv\Scripts\python.exe scripts\pre_ui_runtime_check.py
.\venv\Scripts\python.exe scripts\check_reproducible_environment.py
.\venv\Scripts\python.exe scripts\check_release_readiness.py --mode offline
```

Frozen Phase 1, read-only:

```powershell
.\venv\Scripts\python.exe scripts\phase1.py status
.\venv\Scripts\python.exe scripts\phase1.py validate
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

These checks establish software contracts, graph bounds, regression behavior,
cache and retrieval wiring, provenance handling, and environment readiness. They
do not prove representative retrieval relevance, model action accuracy, medical
factual correctness, claim-level faithfulness, human preference, clinical safety
effectiveness, or deployment readiness.

## Research and Evaluation Status

The runtime architecture is frozen for evaluation. **E0 has not started.** The
next research stage is intended to measure separate dimensions rather than infer
AI quality from implementation tests:

- Agent decision quality;
- retrieval quality;
- answer quality and groundedness;
- abstention behavior;
- narrow safety behavior;
- latency and cost;
- controlled ablations.

No final AI-quality or clinical claim is made from the current test suite.

## Known Limitations

- **Small frozen corpus:** the knowledge foundation contains four source
  snapshots. It is not comprehensive dermatology coverage.
- **NICE provenance:** NICE NG198 exists as a NICE-derived project snapshot.
  Official metadata reports an update date of 2026-04-30, while the local frozen
  snapshot represents 2026-08-03. Full current official-version provenance has
  not been independently reconciled, so the project does not claim a fully
  verified current official NICE snapshot. It also does not claim the snapshot's
  medical content is wrong without evidence.
- **Claim-level verification:** source allowlisting does not prove every answer
  claim is entailed by its cited evidence.
- **Evaluation pending:** final E0 retrieval, decision, answer, safety, and human
  quality evaluation has not started.
- **No clinical validation:** the system has no established clinical accuracy,
  efficacy, sensitivity, or safety performance.
- **Local trust boundary:** authentication, authorization, tenant isolation, and
  public-production hardening are outside the current implementation.
- **External Dense embedding:** runtime Dense queries use Gemini Embedding 2;
  Ollama answer generation alone is not a fully local/offline system.

The accepted NICE limitation is detailed in
[Data Pipeline](docs/DATA_PIPELINE.md), [References](docs/REFERENCES.md), and the
[method/source registry](data/phase1_method_sources.json).

## Methods, Sources, and Documentation

The project separates scientific methods, provider/framework contracts,
clinical safety sources, engineering policies, and empirical project decisions.
Runtime constants are bounded project contracts, not claims of scientific or
clinical optimality.

- [Architecture](docs/ARCHITECTURE.md)
- [Agent Workflow](docs/AGENT_WORKFLOW.md)
- [Data Pipeline](docs/DATA_PIPELINE.md)
- [Methods and Formulas](docs/METHODS_AND_FORMULAS.md)
- [Safety](docs/SAFETY.md)
- [Operations](docs/OPERATIONS.md)
- [References](docs/REFERENCES.md)
- [Phase 1 method/source registry](data/phase1_method_sources.json)
- [Canonical source manifest](data/sources/manifest.yaml)

Code is the source of truth, tests confirm contracts, canonical docs explain
methods, and this README summarizes the merged pre-E0 system.

## Scope and Disclaimer

Acne Advisor AI provides research and educational information only. It does not
diagnose disease, select personal treatment, prescribe medication, or replace a
doctor, dermatologist, pharmacist, emergency service, or other qualified health
professional. Do not use it as the sole basis for urgent or high-risk medical
decisions.

Project metadata declares the MIT license in `pyproject.toml`. No standalone
`LICENSE` file is currently present. Confirm source-corpus, institutional, and
redistribution requirements before publishing medical documents or generated
artifacts.
