# Operations

## Install

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip==26.1.2
python -m pip install -r requirements.lock.txt
Copy-Item .env.example .env
```

Keep secrets only in `.env`. For host development, start just the pinned datastores:

```powershell
docker compose up -d --pull never --no-build postgres redis qdrant neo4j
docker compose ps
```

Only on a genuinely new, empty PostgreSQL installation, initialize SQL explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\init_schema.py
.\.venv\Scripts\python.exe scripts\init_chat_schema.py
```

## Docker Full Stack

The canonical Compose project remains `acne-agent-system`. It manages six services
without changing any existing database image digest or bind mount. The backend
image contains the API and immutable knowledge metadata, never the corpus,
datastores, credentials, model weights or Git history. `requirements.runtime.lock.txt`
is the Linux API dependency closure constrained by the existing development lock;
it excludes ingestion/evaluator/test tools. Torch's pinned Linux wheels supply CUDA
13.0 libraries. Regenerate this lock only deliberately with the command in its header.

The existing complete BGE Hugging Face hub cache is mounted read-only at
`/models/hub`. Set `HF_MODEL_CACHE_PATH` to an existing absolute directory on a
different machine; missing paths fail rather than creating an empty cache. No
startup model download, schema migration, parsing, embedding or activation runs.

```powershell
docker compose config --quiet
docker compose build backend frontend
# This NVIDIA/WSL 2 machine: one API worker, one BGE model, CUDA/BF16 when supported.
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
docker compose ps
docker compose logs --tail 100 backend frontend
docker stats --no-stream
Invoke-RestMethod http://127.0.0.1:8088/api/health
Invoke-RestMethod http://127.0.0.1:8088/api/ready
# Graceful stop, preserving all bind-mounted database data:
docker compose stop
```

CPU-only hosts use `docker compose up -d` without the GPU override. The configured
CUDA scorer reports its CPU/FP32 fallback when CUDA is unavailable; explicitly set
`RERANKER_DEVICE=cpu` if CPU operation is intended. Do not raise API worker count:
each additional process would duplicate model memory and bypass process-local locks.

| Service | Host loopback port | Internal address |
|---|---|---|
| UI / Nginx | 8088 (`FRONTEND_PORT`) | frontend:8080 |
| FastAPI | 8000 (`BACKEND_PORT`) | backend:8000 |
| PostgreSQL | existing 5433 | postgres:5432 |
| Redis | existing 6379 | redis:6379 |
| Qdrant | existing 6333 / 6334 | qdrant:6333 |
| Neo4j | existing 7474 / 7687 | neo4j:7687 |

The browser calls relative `/api/...`; Nginx strips `/api` and resolves `backend`
through Docker DNS, including after backend recreation. Proxy read timeout is
220 seconds between the 210-second Agent budget and 225-second browser deadline.
It forwards error statuses and never automatically retries POSTs. Host Vite/API
development retains `VITE_API_URL=http://127.0.0.1:8000` and
`scripts/start_local_dev.ps1` starts datastores only. Stop Docker FE/BE first when
using the same host API port, or choose distinct ports.

`/live` proves only API process liveness. `/health` reports detailed dependency and
configuration state; `/ready` returns 503 when required core checks fail. No health
probe calls paid providers. A configured Gemini key does not prove live generation.
Redis, PostgreSQL and Neo4j retain their optional runtime role. Only healthy Qdrant
gates backend startup. Qdrant uses the pinned image's bash HTTP `/readyz` probe;
neither curl nor wget is assumed. Nginx `/healthz` proves static serving only; use
`/api/ready` to inspect the application behind it. Unhealthy status alone does not
make Compose restart a container, and provider faults must not trigger restart loops.

Gemini credentials are injected from ignored `.env` at container creation. Parser
and evaluator keys are blanked in the API environment. Local optional Langfuse is
disabled in this stack; bounded JSONL diagnostics use ephemeral `/tmp/acne-traces`
when enabled. `docker compose logs` captures application stdout/stderr with rotation.
API access logs are disabled to avoid recording session paths. Docker access itself
is privileged access to runtime environment secrets; do not print resolved config.

Ollama is not started or downloaded by Compose. Use
`DOCKER_OLLAMA_BASE_URL=http://host.docker.internal:11434` for an existing host
service. Its absence does not veto primary Gemini readiness. To explicitly require
that fallback, set `DOCKER_LLM_PROVIDER_FALLBACK_ENABLED=true`; then its unavailability
is correctly reported as degraded. `DOCKER_DATABASE_URL` accepts a fully encoded
internal PostgreSQL URL when credentials need URL escaping.

When startup fails, inspect `docker compose ps`, `/health` and service logs. Check
the cache mount and `docker compose exec backend python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
Never delete data or rebuild the knowledge index to fix DNS, GPU or credential
errors. Retry just FE/BE with `docker compose ... up -d --no-deps backend frontend`;
use the same GPU override as the original start. Check the active identity through
`/api/health` -> `checks.qdrant.knowledge_build_id`, and verify counts 536/32 through
Qdrant's API. No new database mounts or named data volumes are needed.

### Runtime audit and operational limits

The 25 October 2026 audit reproduced resource leaks on provider cancellation and
Redis initialization, invalid NullPool options, a zero-history slicing error,
response-body timeouts, request replay divergence, incomplete history pagination,
malformed cache payloads and duplicate-channel RRF inflation. Regression tests live
under `tests/test_runtime_resource_lifecycle.py`, `tests/test_api_idempotency.py`,
`tests/test_storage_safety.py`, `tests/test_retrieval_invariants.py` and frontend
HTTP/turn contract tests. These are engineering correctness checks; they do not
certify clinical safety or semantic answer faithfulness.

Frontend logical retries now keep a UUID `request_id`, the original input and
pre-submit history. PostgreSQL retains deterministic UUID5 user/assistant message
IDs. A successfully persisted turn replays its original answer with
`metadata.replayed=true`; reusing its request ID for different input or session
returns 409. Missing/corrupt historical replay metadata also fails closed with 409.
The API locks both session and request IDs within its single process. Persistence
remains best-effort: when PostgreSQL is unavailable, replay cannot promise
exactly-once generation. No distributed idempotency claim is made. Requests without
client IDs preserve the existing behavior. History has bounded pages (1–200,
nonnegative offset); the UI loads subsequent pages in order. Avoid editing a
session concurrently from several independent clients.

Optional chat SQL work uses `CHAT_DB_TIMEOUT_SECONDS=2` per replay/history/save
operation. A stalled transaction is cancelled and rolled back before the Agent
answer is returned; this preserves time for the proxy/browser response after the
210-second Agent budget. Pool acquisition's former 30-second default could exceed
that remaining envelope. Provider and Redis clients now close on failure/shutdown;
caller-supplied Google clients remain owned by their caller. Cache normalization
uses NFC without stripping Vietnamese diacritics or negation; invalid cache JSON
shapes are misses. Cache remains exact and keeps its namespace and eligibility rules.

Package auditing found vulnerable versions in ten runtime packages. The two lock
files now select the minimum identified fixes: AnyIO 4.14.2, certifi 2026.10.19,
cryptography 50.0.0, filelock 4.2.1, h2 4.4.1, langgraph-sdk 0.4.4,
opentelemetry-api 1.45.1 (with matching SDK/exporter closure),
sentence-transformers 5.6.0, setuptools 84.1.0 and urllib3 2.8.0.
The Nginx runtime uses 1.30.6, selected against its upstream security advisory.
Affected-version presence alone does not prove exploitability in this application:
several issues require custom model code, process pools, custom SDK auth or local
filesystem attackers. BGE artifacts remain local/read-only and no custom model
Python code is provisioned. Primary records include
[AnyIO](https://github.com/agronholm/anyio/security/advisories/GHSA-82r6-8w77-94w6),
[certifi](https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2),
[sentence-transformers](https://github.com/huggingface/sentence-transformers/pull/3807),
[urllib3](https://github.com/urllib3/urllib3/security/advisories/GHSA-vxq7-64xx-v4gw)
and [Nginx](https://nginx.org/en/security_advisories.html). Vulnerability databases
change; a clean version scan is not a guarantee against all vulnerabilities.
The separate development/ingestion lock also received verified fixes in aiohttp
3.14.4, banks 2.5.1, multidict 6.9.1, NLTK 3.10.4, Pillow 13.0.0 and tinytag 2.3.0
(plus the required yarl 1.25.1). These packages are absent from the API image's
minimal dependency closure. OSV returned no affected packages for the resulting
148-package development lock and 123-package runtime lock on this audit date.
No source parsing or model/corpus download was used to test these changes.

On this host, Docker Desktop 4.89.0 / Engine 29.7.2 / Compose 5.5.0 uses WSL 2
with 12 logical CPUs and 7.42 GiB Linux memory, sharing a 15.31 GiB Windows machine.
RTX 3050 Laptop has 6 GiB VRAM, driver 617.42; Torch 2.13.0 CUDA 13.0 and BF16
were verified. Initial API warmup took about 150 seconds under concurrent host
work; a separate warm-cache model load took about 70 seconds. An unhealthy startup
probe does not restart the container. GPU model inference with 16 synthetic pairs
measured 177 ms warm at concurrency 1, 300 ms maximum at concurrency 2 and 591 ms
maximum at concurrency 4, using one serialized model executor. These tiny probes
are not representative p95/SLO measurements. CUDA-hidden CPU/FP32 fallback produced
finite scores but took about 61 seconds for two pairs while the host was under
load; a 20-second runtime reranker timeout can therefore preserve RRF order.
Normal GPU backend residency was approximately 2 GiB RAM / 1.16 GiB VRAM;
an additional diagnostic CPU model raised cgroup peak memory to about 4.26 GiB.
No global Docker Desktop settings or arbitrary hard memory limits were imposed on
the shared daemon. Keep one API worker, monitor `docker stats`, and avoid parallel
CPU model diagnostics on this 16 GiB laptop.

Browser E2E was blocked by the Windows Computer Use tool's unsupported browser URL
policy enforcement. HTTP/proxy tests do not substitute for real UI interaction.
Existing frozen research/evaluation artifacts remain immutable. Semantic evaluator
credit/calibration, independent holdout and clinical review remain separate gates.

### Bounded live verification on 25 October 2026

Five HTTP chat scenarios plus one idempotent replay were exercised through
`http://127.0.0.1:8088/api/chat`, within the approved five-question / USD 1 limit.
No further paid probe is needed to start this stack.

| Scenario | Observed result |
|---|---|
| Benzoyl peroxide classification, initial implementation | HTTP 200, 18.15 s, two retrievals; exposed answer generation after `abstain/evidence_gap` |
| Identical request-id replay | Same answer, about 22 ms, no additional generation or history rows |
| Benzoyl peroxide treatment information | HTTP 200, 5.99 s, one retrieval and source-backed generation |
| Same treatment question in a fresh session | Exact-cache hit, 0.232 s, no retrieval or generation calls |
| Classification after non-generating abstention fix | HTTP 200, 8.61 s; semantic model selected sufficient evidence at the first retrieval |
| Synthetic unsupported genotype / exact-effect claim | HTTP 200, 4.48 s; `abstain/evidence_gap`, `generation_invoked=false`, non-cacheable deterministic fallback |

The five scenarios created five clearly named `docker-audit-20261025-*` sessions
and ten messages; replay created none. The original 15 sessions / 98 messages
retained their pre-audit row checksums after excluding those new diagnostic rows.
These rows are retained for review. The unsupported-claim probe additionally
exposed retrieved source labels on a non-generated fallback. API attribution and
source-request presentation now suppress those labels while preserving retrieval
diagnostics. Regression tests and a provider-free ASGI check in the final Linux
image verify this final correction; no sixth paid scenario was run. Historical
stored answers/metadata are not rewritten retroactively.

Twelve Gemini 3.5 Flash-Lite calls across these scenarios consumed 26,729 input
and 1,075 output tokens (including thinking): estimated USD 0.0107062 at the
[published pricing](https://ai.google.dev/gemini-api/docs/pricing), excluding the
small query-embedding charge. This is a token-based estimate, not a billing
invoice. Corpus embeddings were never regenerated. The cache-hit measurement was
before the abstention policy's deliberate fingerprint change; it is not a claim
that old cache entries remain eligible in the new partition.

One repeated question followed different retrieval/decision paths in the two
live executions. This is evidence of semantic/provider variability, not proof
that sufficiency improved. The bounded probes do not establish clinical safety,
faithfulness, independent-holdout scores, sustained throughput or a latency SLO.
The NICE source's represented revision date still warrants provenance review;
the frozen source snapshot and active build were preserved rather than silently
relabelled or reingested. Optional Ollama generation was not verified live.

### Generated files and shared Docker storage

The current `.venv` (about 3.48 GB), frontend `node_modules` (78 MB), read-only
Hugging Face model cache and four datastore directories are active dependencies
or data. Keep them for continued local development. Frontend `dist` (263 kB),
`.pytest_cache`, `.ruff_cache`, Python `__pycache__`/`.pyc` and editable-install
`acne_advisor_ai.egg-info` are regenerable outputs; the latter is useful for the
current installed development package. Local logs may be retained for diagnosis.
All remain ignored and outside production build contexts as appropriate.

Only the two current tagged application images were found with this Compose
project's ownership label; no dangling image was listed. The daemon-wide storage
snapshot showed 68.69 GB of images and 70.95 GB of build-cache logical usage,
including other projects and shared layers. These figures overlap and must not
be added or treated as Acne-only disk consumption. The current backend's CUDA
dependency build layer is approximately 8.59 GB; its compressed image is about
3.02 GB and the frontend about 25.7 MB. Build cache can accelerate later rebuilds.
No global prune, volume deletion, model deletion or cleanup of another project's
resources was performed. Classify ownership and container references before any
future deletion; a daemon-wide reclaimable total is insufficient authorization.

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

Compute the effective fingerprint with `scripts/inspect_cache_versions.py`; it
depends on the active knowledge build and runtime configuration. Do not copy a
digest from an older report into configuration. The immutable historical formal evaluation artifact
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

## Shutdown and Knowledge Maintenance

Use `docker compose stop` for normal shutdown. Do not use global prune commands,
`docker compose down -v`, Qdrant collection deletion, Neo4j destructive deletes,
or Redis `FLUSHALL` as a startup/shutdown strategy.

Develop and operate from the canonical `main` checkout. Preserve the existing
bind-mounted data and reuse the activated build. Historical local backup folders,
old Qdrant collections and research Docker stacks are not runtime dependencies.
The manifest's `activation.rollback_root` records the backup used at the historical
cutover; it does not promise that this local folder is still retained. Its value
must not be rewritten to imply a new backup or a different build identity.

A future knowledge migration remains a separate operation: activation still
requires verified rollback artifacts supplied explicitly to that operation.
Deleting historical backups does not bypass or disable this guard.

The optional Langfuse sink is disabled in the normal local configuration; JSONL
telemetry remains available. Its Compose recipe is retained for integration
contracts, not as a backing-service requirement. Do not start an observability
stack as part of normal application startup.

Windows Application Control previously blocked the Python `select` extension.
On 25 October 2026 the existing environment could import `select` and Torch CUDA
again; no security policy or system DLL was changed by this task. Keep checking
this separately from the Linux API image and GitHub Windows CI. Docker operation
does not repair or prove the underlying Windows security policy. The isolated
thesis evaluator recipe remains `evaluation/Dockerfile.thesis` and is not needed
to start the application.

`.env`, provider secrets, caches, databases, snapshots and generated reports
remain local-only. Never print secrets or use destructive global Docker cleanup.

## Local Trust Boundary

Keep FastAPI, Qdrant, PostgreSQL, Neo4j, Redis, and the frontend bound to
`127.0.0.1`/`localhost`. The current application has no end-user authentication
or tenant authorization, and chat-history endpoints can read or mutate local
history. Do not expose this development stack directly to an untrusted network.
Production deployment requires TLS, authentication, authorization, rate
limiting, restrictive CORS, tenant isolation, and managed secret storage.
