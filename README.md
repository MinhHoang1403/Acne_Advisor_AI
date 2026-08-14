# Acne Advisor AI

Acne Advisor AI is a Vietnamese medical-information assistant built with a
frozen, provenance-preserving knowledge foundation and a bounded LangGraph RAG
runtime. It provides educational information about acne and skin care; it does
not diagnose, prescribe, or replace a qualified clinician.

## Current System

The repository has two explicit boundaries:

1. **Phase 1: frozen knowledge foundation.** Four canonical sources are parsed,
   normalized, structurally chunked, embedded, indexed, and validated. The
   activated build contains 512 knowledge chunks, 32 EntityCards, and a
   deterministic Neo4j graph with 32 nodes and 27 relationships.
2. **Phase 2: read-only Agentic RAG.** An eight-node LangGraph workflow retrieves
   source chunks with Dense + native BM25, fuses ranks with equal-weight RRF,
   packs bounded provenance, and then generates or abstains under deterministic
   safety and evidence gates.

```text
User -> FastAPI -> LangGraph
                     -> guard and decide
                     -> Dense + BM25 -> RRF -> context packer
                     -> assess evidence (maximum two retrieval attempts)
                     -> generate or abstain
                     -> safety, presentation, cache, observability
```

Runtime answer generation does not query EntityCards or Neo4j. Those stores are
frozen Phase 1 structural assets and do not independently ground medical claims.

## Technology

- Python 3.11.9, FastAPI, Pydantic, LangGraph
- Google Gemini and local Ollama generation providers
- Gemini Embedding 2, 3072 dimensions, cosine distance
- Qdrant Dense vectors and Qdrant-native BM25
- PostgreSQL, Redis, Neo4j, Docker Compose
- React, Vite, Vitest, ESLint

## Repository Map

| Location | Responsibility |
|---|---|
| `scripts/phase1.py` | only supported Phase 1 build/validate/status interface |
| `src/ingestion/` | frozen parsing, chunking, filtering, provenance and indexing |
| `src/knowledge/` | source-backed taxonomy, EntityCards and deterministic graph |
| `src/retrieval/` | Dense + BM25 + RRF and bounded context packing |
| `src/agent/` | LangGraph state, decisions, generation and presentation |
| `src/quality/` | deterministic safety, verification and fallback contracts |
| `src/api/` | FastAPI routes and runtime preflight |
| `src/cache/`, `src/database/` | cache and persistence adapters |
| `src/frontend/` | React/Vite chat client |
| `tests/` | retained behavior and contract regressions |

## Prerequisites

- Python `3.11.9`
- pip `26.1.2`
- Node.js/npm compatible with `src/frontend/package-lock.json`
- Docker Desktop with Compose
- Ollama with `qwen3:8b` for local generation
- Provider keys only for the live operations that use them

## Backend Setup

```powershell
git clone https://github.com/MinhHoang1403/RAG-system-for-acne-diagnose.git
Set-Location RAG-system-for-acne-diagnose

py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip==26.1.2
python -m pip install -r requirements.lock.txt
Copy-Item .env.example .env
```

Fill only the secrets needed by your chosen providers in `.env`. Never commit
that file. The public defaults are documented in `.env.example`; local Docker
Qdrant works with an empty `QDRANT_API_KEY`.

Start backing services and initialize relational schemas:

```powershell
docker compose up -d --pull never --no-build
docker compose ps
.\venv\Scripts\python.exe scripts\init_schema.py
.\venv\Scripts\python.exe scripts\init_chat_schema.py
```

Do not run a Phase 1 build merely to start the API or UI. The current activated
knowledge foundation is expected to exist in the local data stores.

## Run Locally

The bounded local launcher starts Docker, reuses a healthy backend when one is
already running, and does not kill unknown processes:

```powershell
.\scripts\start_local_dev.ps1
```

Manual backend start:

```powershell
.\venv\Scripts\python.exe -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

Manual frontend start:

```powershell
Set-Location src\frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

The frontend reads `VITE_API_URL` and otherwise uses
`http://127.0.0.1:8000`. Swagger is available at
`http://127.0.0.1:8000/docs`.

## Validate

Read-only Phase 1 integrity:

```powershell
.\venv\Scripts\python.exe scripts\phase1.py status
.\venv\Scripts\python.exe scripts\phase1.py validate
```

Runtime and release checks:

```powershell
.\venv\Scripts\python.exe scripts\inspect_phase2_readiness.py
.\venv\Scripts\python.exe scripts\pre_ui_runtime_check.py
.\venv\Scripts\python.exe scripts\check_reproducible_environment.py
.\venv\Scripts\python.exe scripts\check_release_readiness.py --mode offline
.\venv\Scripts\python.exe -m pytest -q
```

Frontend checks:

```powershell
Set-Location src\frontend
npm ci
npm test
npm run lint
npm run build
npm audit
```

Phase 1 `build` and `--activate` are controlled migration operations. Use them
only with an explicit source change or repair plan and the rollback safeguards
described in [Data Pipeline](docs/DATA_PIPELINE.md).

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | dependency and runtime health |
| `GET /models` | available generation models |
| `GET /retrieve?q=...` | debug the canonical source retrieval path |
| `POST /chat` | run the complete guarded RAG workflow |
| `/sessions`, `/sessions/{id}/messages` | chat-history persistence |

`/retrieve` is a diagnostics endpoint, not an alternate evidence pipeline. It
uses the same Dense + BM25 + RRF retrieval service as the agent.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data Pipeline](docs/DATA_PIPELINE.md)
- [Agent Workflow](docs/AGENT_WORKFLOW.md)
- [Methods and Formulas](docs/METHODS_AND_FORMULAS.md)
- [References](docs/REFERENCES.md)
- [Safety](docs/SAFETY.md)
- [Operations](docs/OPERATIONS.md)

Each topic has one current source of truth. Git and merged pull requests retain
development history; active documentation describes only the present system.

## Safety and Privacy

- Do not enter secrets or identifying patient information in committed files.
- Do not present responses as diagnosis or prescriptions.
- Emergency, pregnancy, severe-acne, medication and abstention contracts are
  deterministic and regression-tested.
- `.env`, databases, caches, source documents, snapshots and generated reports
  are local-only unless deliberately approved for publication.

See [Safety](docs/SAFETY.md) for the medical-response boundary and
[Operations](docs/OPERATIONS.md) for safe startup, shutdown and rollback.

## License

Project metadata declares the MIT license. Confirm institutional and source
licensing requirements before redistributing medical corpus files or generated
artifacts.
