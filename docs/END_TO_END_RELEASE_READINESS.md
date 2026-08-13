# End-to-End Release Readiness

## Prerequisites

- Python `3.11.9`
- pip `26.1.2`
- Node/npm theo frontend lock file hiện tại
- Docker Compose
- Ollama chạy local với model `qwen3:8b`
- Qdrant, Neo4j, PostgreSQL và Redis theo `docker-compose.yml`
- `.env` local có các biến runtime cần thiết, nhưng không commit secret
- `GOOGLE_API_KEY` chỉ cần cho live Gemini smoke

## Safe Startup

```powershell
docker compose up -d --pull never --no-build
docker compose ps
```

Nếu containers chỉ đang stopped:

```powershell
docker compose start
```

## Health Validation

```powershell
.\venv\Scripts\python.exe scripts\check_reproducible_environment.py
.\venv\Scripts\python.exe scripts\check_release_readiness.py --mode offline
.\venv\Scripts\python.exe scripts\check_release_readiness.py --mode local-services
.\venv\Scripts\python.exe scripts\pre_ui_runtime_check.py
```

Kiểm tra thêm Ollama:

```powershell
ollama list
```

## Offline Validation

```powershell
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe scripts\check_reproducible_environment.py
.\venv\Scripts\python.exe scripts\check_release_readiness.py --mode offline
.\venv\Scripts\python.exe -m pytest -q
```

Frontend:

```powershell
Set-Location src\frontend
npm ci
npm run build
npm run lint
```

## Live Validation

Live mode uses at most a small Gemini generation smoke and a small Ollama generation smoke. It does not run ingestion, LlamaParse, live embedding, model download, or database rebuild.

```powershell
.\venv\Scripts\python.exe scripts\check_release_readiness.py --mode live
```

## Data Integrity

Expected Phase 1 data:

```text
Qdrant acne_knowledge = 639
Qdrant acne_entities_v1 = 32
Neo4j nodes = 32
Neo4j relationships = 27
```

Validate:

```powershell
.\venv\Scripts\python.exe scripts\inspect_phase2_readiness.py
```

PostgreSQL and Redis connectivity are checked by `/health`, `pre_ui_runtime_check.py`, and `check_release_readiness.py --mode local-services`.

## Safe Shutdown

```powershell
docker compose stop
```

## Forbidden Commands

Do not run these during release readiness unless there is a separate, explicit backup and rebuild plan:

```text
docker compose down -v
docker volume prune
docker system prune
docker image prune
ingestion/rebuild without backup
qdrant collection delete
Neo4j destructive deletes
Redis FLUSHALL or FLUSHDB
```

## Rollback

1. Checkout the previous integration tag.
2. Keep existing volumes and bind-mounted data directories.
3. Do not downgrade database images unless compatibility is verified.
4. Start with pinned local images:

```powershell
docker compose up -d --pull never --no-build
```

5. Run health and data checks:

```powershell
.\venv\Scripts\python.exe scripts\inspect_phase2_readiness.py
.\venv\Scripts\python.exe scripts\pre_ui_runtime_check.py
```

Do not use reset, force push, volume deletion, or database truncation as a runtime rollback strategy.
