# Operations

## Install

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip==26.1.2
python -m pip install -r requirements.lock.txt
Copy-Item .env.example .env
```

Keep secrets only in `.env`. Start pinned local services and initialize SQL:

```powershell
docker compose up -d --pull never --no-build
docker compose ps
.\venv\Scripts\python.exe scripts\init_schema.py
.\venv\Scripts\python.exe scripts\init_chat_schema.py
```

## Phase 1

```powershell
.\venv\Scripts\python.exe scripts\phase1.py build --source sample_data
.\venv\Scripts\python.exe scripts\phase1.py validate --offline
.\venv\Scripts\python.exe scripts\phase1.py validate
.\venv\Scripts\python.exe scripts\phase1.py status
```

Activation is intentionally guarded and requires verified native Qdrant
snapshots plus a Neo4j cold backup:

```powershell
.\venv\Scripts\python.exe scripts\phase1.py build --activate --rollback-root data\backups\<snapshot>
```

Do not run Phase 1 merely to start the UI. `scripts/init_schema.py` owns SQL
schema initialization and does not create, recreate or delete the frozen
Qdrant knowledge store.

## Runtime Checks

```powershell
.\venv\Scripts\python.exe scripts\inspect_phase2_readiness.py
.\venv\Scripts\python.exe scripts\pre_ui_runtime_check.py
.\venv\Scripts\python.exe scripts\check_release_readiness.py --mode offline
.\venv\Scripts\python.exe -m pytest -q
```

The production runtime uses answer cache `v6`. Runtime retrieval is read-only
Dense + native BM25 + RRF over
`acne_knowledge`; it does not query EntityCards or Neo4j. Do not run Phase 1
build, activation, reindexing, or embedding as part of Phase 2 startup.

## Supported Commands

Application logic lives under `src/`; scripts are thin operators or bounded
checks.

| Script | Responsibility |
|---|---|
| `phase1.py` | controlled build, validate and status |
| `init_schema.py`, `init_chat_schema.py` | relational schema initialization |
| `inspect_phase2_readiness.py`, `pre_ui_runtime_check.py` | local readiness |
| `check_reproducible_environment.py`, `check_release_readiness.py` | release gates |
| `smoke_phase2_runtime.py` | provider-free structural agent smoke |
| `eval_phase2_answer_quality.py`, `eval_phase2_all.py` | offline answer contracts |
| `eval_safe_fallback_flow.py`, `eval_runtime_resilience.py` | fallback/resilience gates |
| `inspect_cache_versions.py` | cache and fingerprint inspection |
| `clear_redis_cache.py` | explicit, developer-triggered answer-cache cleanup |

## Backend and Frontend

```powershell
.\scripts\start_local_dev.ps1
```

Or start components manually:

```powershell
.\venv\Scripts\python.exe -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
Set-Location src\frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

The UI uses `VITE_API_URL`, defaulting to `http://127.0.0.1:8000`. It treats
HTTP 400/429/503/504 as reachable backend/provider errors rather than network
disconnects.

## Validation Modes

```powershell
.\venv\Scripts\python.exe scripts\check_release_readiness.py --mode offline
.\venv\Scripts\python.exe scripts\check_release_readiness.py --mode local-services
```

`--mode live` performs bounded provider smoke calls; it does not run ingestion,
embedding, model download or database rebuild.

## Shutdown and Rollback

Use `docker compose stop` for normal shutdown. Do not use global prune commands,
`docker compose down -v`, Qdrant collection deletion, Neo4j destructive deletes,
or Redis `FLUSHALL` as a startup/shutdown strategy.

For runtime rollback, check out the previous integration commit, preserve the
existing bind-mounted data, start pinned services with `--pull never --no-build`,
then rerun readiness and Phase 1 status checks. A Phase 1 datastore rollback
must use its verified snapshots and cold backup; it is a separate controlled
migration operation.

`.env`, provider secrets, caches, databases, snapshots and generated reports
remain local-only. Never print secrets or use destructive global Docker cleanup.

## Local Trust Boundary

Keep FastAPI, Qdrant, PostgreSQL, Neo4j, Redis, and the frontend bound to
`127.0.0.1`/`localhost`. The current application has no end-user authentication
or tenant authorization, and chat-history endpoints can read or mutate local
history. Do not expose this development stack directly to an untrusted network.
Production deployment requires TLS, authentication, authorization, rate
limiting, restrictive CORS, tenant isolation, and managed secret storage.
