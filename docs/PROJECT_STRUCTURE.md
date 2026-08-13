# Project Structure

This document defines the maintained repository paths for Acne Advisor AI.
Start with `docs/ARCHITECTURE.md`; use `docs/DATA_PIPELINE.md` and
`docs/AGENT_WORKFLOW.md` for the two primary runtime paths.

## Application

- `src/api/app.py`: FastAPI application entrypoint. Run with `uvicorn src.api.app:app --reload --port 8000`.
- `src/agent/`: LangGraph clinical-answer pipeline, prompts, guardrails, fallback and response presentation.
- `src/database/`: Qdrant, Neo4j and PostgreSQL access layers.
- `src/retrieval/`: query normalization, sparse/dense fusion, entity retrieval, context packing and reranking.
- `src/knowledge/`: taxonomy models, entity cards and deterministic graph indexing.
- `src/frontend/`: the only maintained web client. Run its Vite commands from this directory.

## Data and Ingestion

- `data/taxonomy/`: tracked taxonomy and alias source data.
- `scripts/run_full_phase1.py`: authoritative complete Phase 1 build command.
- `scripts/ingest_knowledge.py`: lower-level incremental ingestion command retained for supported operator workflows.
- `scripts/run_semantic_enrichment.py`: optional semantic graph enrichment; it is not part of the canonical core build.
- `scripts/init_schema.py` and `scripts/init_chat_schema.py`: supported schema initialization commands.
- Runtime data, cache, ingestion manifest, local reports and logs remain untracked.

## Regression Validation

- `tests/`: production behavior, safety, retrieval, Phase 1, API and integration regression contracts.
- `tests/fixtures/`: bounded regression data. Fixtures are not clinical gold and do not establish scientific quality claims.
- `scripts/eval_retrieval_v5_release.py`: locked provider-free Retrieval V5 regression gate.
- `scripts/eval_p3_evidence_sufficiency.py`: locked P3 sufficiency/retry regression gate.
- `scripts/eval_phase2_all.py`: aggregate offline readiness checks retained by the current operator workflow.
- The former top-level Evaluation V3 framework and synthetic 300-question dataset are not part of the active architecture. Git history preserves them for provenance; a future evaluation methodology will be designed separately.

## Validation and Operations

- `tests/`: maintained Python regression suite.
- `scripts/check_release_readiness.py`, `scripts/check_reproducible_environment.py`, `scripts/inspect_phase2_readiness.py` and `scripts/pre_ui_runtime_check.py`: supported readiness checks.
- `scripts/README.md`: maintained command classification and canonical entrypoints.
- `docs/END_TO_END_RELEASE_READINESS.md`: release validation sequence.

## Excluded Local Paths

The following are intentionally local-only: `.env`, `venv/`, caches, Docker bind-mounted data, `node_modules/`, built frontend assets, reports, logs, backups, notebooks and sample data. Do not commit secrets, generated raw responses, Gemini judge artifacts or runtime databases.
