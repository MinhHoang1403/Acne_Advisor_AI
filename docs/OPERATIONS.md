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

The production runtime is frozen as `s4b_final_agentic_rag_v1` with answer
cache `v6`. Runtime retrieval is read-only Dense + native BM25 + RRF over
`acne_knowledge`; it does not query EntityCards or Neo4j. Do not run Phase 1
build, activation, reindexing, or embedding as part of Phase 2 startup.

`.env`, provider secrets, caches, databases, snapshots and generated reports
remain local-only. Never print secrets or use destructive global Docker cleanup.
