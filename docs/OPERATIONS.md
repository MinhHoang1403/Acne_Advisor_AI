# Operations

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

`.env`, provider secrets, caches, databases, snapshots and generated reports
remain local-only. Never print secrets or use destructive global Docker cleanup.
