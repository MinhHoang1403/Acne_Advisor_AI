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

- `smoke_phase2_runtime.py`: provider-free eight-case structural agent smoke.
- `eval_phase2_answer_quality.py`: retained answer-policy regression.
- `eval_safe_fallback_flow.py`: deterministic fallback regression.
- `eval_runtime_resilience.py`: timeout/retry/circuit-breaker regression.
- `eval_phase2_all.py`: aggregate supported offline checks.
- `inspect_cache_versions.py`: cache/fingerprint contract inspection.

Historical incremental ingestion, the retired sparse implementation, and LLM
semantic graph scripts were removed. Git history and rollback artifacts preserve
auditability.
