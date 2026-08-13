# Operations

## Supported Commands

See `scripts/README.md` for the maintained command inventory. Common commands:

```powershell
.\venv\Scripts\python.exe scripts\inspect_phase2_readiness.py
.\venv\Scripts\python.exe scripts\pre_ui_runtime_check.py
.\venv\Scripts\python.exe scripts\check_release_readiness.py --mode offline
.\venv\Scripts\python.exe -m pytest -q
```

Full Phase 1 is an explicit operator action:

```powershell
.\venv\Scripts\python.exe scripts\run_full_phase1.py --source sample_data
```

Do not run ingestion merely to start the UI. Do not use destructive Docker,
Qdrant, Neo4j, PostgreSQL or Redis cleanup as a readiness step.

## Local-Only State

`.env`, `venv/`, runtime databases under `data/`, caches, build output and
generated reports remain ignored. Secrets must never enter reports or Git.

## Current Data Contract

The S3B starting baseline is Qdrant knowledge 639, entities 32, Neo4j 32 nodes / 27
relationships, dense dimension 3072, and semantic enrichment `not_run`. Verify
with `scripts/inspect_phase2_readiness.py`; do not mutate stores during checks.
