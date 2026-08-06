# Project Structure

This document defines the maintained repository paths for Acne Advisor AI.

## Application

- `src/api/app.py`: FastAPI application entrypoint. Run with `uvicorn src.api.app:app --reload --port 8000`.
- `src/agent/`: LangGraph clinical-answer pipeline, prompts, guardrails, fallback and response presentation.
- `src/database/`: Qdrant, Neo4j and PostgreSQL access layers.
- `src/retrieval/`: query normalization, entity retrieval, context packing and reranking.
- `src/knowledge/`: taxonomy models, entity cards and deterministic graph indexing.
- `src/frontend/`: the only maintained web client. Run its Vite commands from this directory.

## Data and Ingestion

- `data/taxonomy/`: tracked taxonomy and alias source data.
- `scripts/ingest_knowledge.py`: supported Phase 1 ingestion command.
- `scripts/init_schema.py` and `scripts/init_chat_schema.py`: supported schema initialization commands.
- Runtime data, cache, ingestion manifest, local reports and logs remain untracked.

## Evaluation

- `evaluation/data/acne_system_eval_v3.jsonl`: canonical 300-case Evaluation V3 dataset.
- `scripts/validate_final_evaluation_v3.py`: validates the canonical dataset and evaluation configuration.
- `scripts/run_final_evaluation_v3.py`: the only canonical final-evaluation runner. It owns live, judge and finalize stages.
- `evaluation/README.md`: the authoritative Evaluation V3 workflow.
- Older V2/V13 answer-quality runners and historical tracked reports were removed. Git history preserves their provenance; new run artifacts belong in ignored `reports/evaluation/` paths.

## Validation and Operations

- `tests/`: maintained Python regression suite.
- `scripts/check_release_readiness.py`, `scripts/check_reproducible_environment.py` and `scripts/pre_ui_runtime_check.py`: supported readiness checks.
- `docs/END_TO_END_RELEASE_READINESS.md`: release validation sequence.
- `scripts/diagnostics/`: manual diagnostic tools. They are not canonical evaluation paths and must not replace Evaluation V3 evidence.

## Excluded Local Paths

The following are intentionally local-only: `.env`, `venv/`, caches, Docker bind-mounted data, `node_modules/`, built frontend assets, reports, logs, backups, notebooks and sample data. Do not commit secrets, generated raw responses, Gemini judge artifacts or runtime databases.
