# Operations

## Install

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip==26.1.2
python -m pip install -r requirements.lock.txt
Copy-Item .env.example .env
```

Keep secrets only in `.env`. Start pinned local services and initialize SQL:

```powershell
docker compose up -d --pull never --no-build
docker compose ps
.\.venv\Scripts\python.exe scripts\init_schema.py
.\.venv\Scripts\python.exe scripts\init_chat_schema.py
```

## Knowledge Build

```powershell
.\.venv\Scripts\python.exe scripts\knowledge_build.py build --source sample_data
.\.venv\Scripts\python.exe scripts\knowledge_build.py validate --offline
.\.venv\Scripts\python.exe scripts\knowledge_build.py validate
.\.venv\Scripts\python.exe scripts\knowledge_build.py status
```

Activation is intentionally guarded and requires verified native Qdrant
snapshots plus a Neo4j cold backup:

```powershell
.\.venv\Scripts\python.exe scripts\knowledge_build.py build --activate --rollback-root data\backups\<snapshot>
```

Starting the UI or API normally reuses the existing indexed knowledge.
`scripts/init_schema.py` owns SQL schema initialization and does not create,
recreate, or delete the Qdrant knowledge index.

## Runtime Checks

```powershell
.\.venv\Scripts\python.exe scripts\inspect_runtime_readiness.py
.\.venv\Scripts\python.exe scripts\pre_ui_runtime_check.py
.\.venv\Scripts\python.exe scripts\check_release_readiness.py --mode offline
.\.venv\Scripts\python.exe -m pytest -q
```

The runtime uses exact normalized answer cache `v10`. Cache identity
includes provider, model, pipeline fingerprint, and the normalized question; no
semantic-similarity lookup is performed. Runtime retrieval is read-only
Dense + native BM25 + RRF, local BGE reranking, and whole-chunk packing over
`acne_knowledge`; it does not query EntityCards or Neo4j. Knowledge compilation,
activation, reindexing, and embedding are separate maintenance operations rather
than application-startup steps.

## Request Observability

Runtime observability is disabled by default. When explicitly enabled, the
JSONL sink receives one `chat_request_completed` event after Agent execution
and best-effort chat persistence. Its canonical correlation key is the opaque
API-generated `request_id`; no separate trace/correlation identifier is
created for the same request.

The event contains statuses, durations, safe error type/family/owner,
provider/model identity, pipeline and knowledge identities, candidate IDs and
counts, and bounded fallback metadata. It does not persist the raw question,
conversation history, prompt, model reasoning, credentials, or raw exception
messages. Query text is represented only by a redacted length summary and the
pre-existing short hash. Export, serialization, and sink failures fail open and
must not change the business response.

The current effective environment computes fingerprint
`0a8d129a215c884f2d9654e9` for active knowledge build
`d4a1819fe7fb77fe1f40`. The immutable historical formal evaluation artifact
records fingerprint `f93ad3e8dfb2c39f403b0794` and build
`94d613bc9b33628de3ef`; it must not be presented as the current system. Never
hard-code either digest or remove manifest fields to force equality.

Readiness reports provider/model configuration separately from an actual
generation probe. `generation_probed=false` is truthful evidence that no probe
ran, not evidence that generation succeeded or failed. Chat persistence uses an
optional UUID request id as an idempotency key and remains fail-open for the
current response.

## Supported Commands

Application logic lives under `src/`; scripts are thin operators or bounded
checks.

| Script | Responsibility |
|---|---|
| `knowledge_build.py` | controlled build, validate and status |
| `init_schema.py`, `init_chat_schema.py` | relational schema initialization |
| `inspect_runtime_readiness.py`, `pre_ui_runtime_check.py` | local readiness |
| `check_reproducible_environment.py`, `check_release_readiness.py` | environment and release-readiness checks |
| `smoke_runtime.py` | provider-free structural agent smoke |
| `check_answer_contracts.py`, `check_runtime_contracts.py` | implementation contracts; not clinical evaluation |
| `check_safe_fallback_flow.py`, `check_runtime_resilience.py` | fallback and resilience checks |
| `inspect_cache_versions.py` | cache and fingerprint inspection |
| `clear_redis_cache.py` | explicit, developer-triggered answer-cache cleanup |

## Backend and Frontend

```powershell
.\scripts\start_local_dev.ps1
```

Or start components manually:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
Set-Location src\frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

The UI uses `VITE_API_URL`, defaulting to `http://127.0.0.1:8000`. It treats
HTTP 400/429/503/504 as reachable backend/provider errors rather than network
disconnects.

## Validation Modes

```powershell
.\.venv\Scripts\python.exe scripts\check_release_readiness.py --mode offline
.\.venv\Scripts\python.exe scripts\check_release_readiness.py --mode local-services
```

`--mode live` performs bounded provider smoke calls; it does not run ingestion,
embedding, model download or database rebuild.

## Shutdown and Rollback

Use `docker compose stop` for normal shutdown. Do not use global prune commands,
`docker compose down -v`, Qdrant collection deletion, Neo4j destructive deletes,
or Redis `FLUSHALL` as a startup/shutdown strategy.

For runtime rollback, check out the previous integration commit, preserve the
existing bind-mounted data, start pinned services with `--pull never --no-build`,
then rerun readiness and knowledge-build status checks. A knowledge datastore rollback
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
