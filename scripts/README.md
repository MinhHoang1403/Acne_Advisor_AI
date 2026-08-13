# Supported Commands

Application logic belongs under `src/`; scripts are thin operators or bounded
read-only regression gates.

## Phase 1

- `phase1.py`: the only supported Phase 1 interface (`build`, `validate`,
  `status`). It owns candidate creation and rollback-guarded activation.

## Infrastructure and Runtime

- `init_schema.py`, `init_chat_schema.py`: SQL schema initialization.
- `inspect_phase2_readiness.py`, `pre_ui_runtime_check.py`,
  `check_release_readiness.py`, `check_reproducible_environment.py`: readiness.
- `clear_redis_cache.py`: explicit answer-cache cleanup.

## Regression Gates

- `eval_retrieval_v5_release.py`: provider-free Retrieval V5 contract.
- `eval_p3_evidence_sufficiency.py`: P3 bounded retry/abstention contract.
- `eval_phase2_all.py`: aggregate supported offline checks.
- Other `eval_phase2_*`, resilience, fallback and reranker scripts are focused
  Phase 2 regression gates, not Phase 1 build paths or clinical validation.

Historical incremental ingestion, the retired sparse implementation, and LLM
semantic graph scripts were removed. Git history and rollback artifacts preserve
auditability.
