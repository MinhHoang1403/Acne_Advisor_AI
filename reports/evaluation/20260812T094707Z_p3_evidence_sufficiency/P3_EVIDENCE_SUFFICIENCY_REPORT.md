# P3 Evidence Sufficiency, Bounded Retry, and Safe Abstention

## Decision

`P3_READY_TO_MERGE`

## Runtime flow

```text
Retrieval V5 (attempt 0)
  -> PRE_PACK assessment
  -> POST_PACK assessment
     -> SUFFICIENT: existing fallback decision -> safety -> generation
     -> RETRYABLE and attempt 0: deterministic retry plan
        -> Retrieval V5 (attempt 1, fresh result)
        -> PRE_PACK + POST_PACK reassessment
           -> SUFFICIENT: normal generation
           -> otherwise: structured abstention -> safe fallback
     -> NON_RETRYABLE: structured abstention -> safe fallback
```

## Contract summary

- Typed status separates sufficient, ordinary insufficient, and critical evidence missing.
- Source-backed chunk provenance is mandatory.
- Entity and graph remain structural side channels only.
- PRE_PACK and POST_PACK states prevent false sufficiency when evidence is dropped by budget.
- Retry is deterministic, meaningfully different, and hard-bounded to one additional retrieval.
- Attempt scores are never mixed.
- Abstention is typed, severity-guarded, and non-cacheable.
- P3 version/config participates in the existing pipeline fingerprint; `CACHE_ANSWER_VERSION` remains `v5`.
- Explicit `RETRIEVAL_PIPELINE_VERSION=v4` bypasses P3 for rollback.

## Validation summary

- Dedicated fixture: 17/17 PASS.
- Locked R8: 18/18 PASS; critical/source coverage 100%; answer quality 55/55.
- Backend: 818 PASS, 83.81% coverage.
- Frontend: 42 PASS; lint/build/audit PASS.
- Runtime: 639 chunk points, 32 entity points, Neo4j 32/27.
- Data mutation: none.

## Remaining limitations

- Sufficiency is intentionally deterministic; it measures role/provenance coverage, not semantic entailment.
- One retry can add provider latency in live traffic; the fixture timing only measures local policy overhead.
- Corpus absence and a query-representation miss cannot always be distinguished after the single retry; both safely abstain.
- Partial supported facts remain in structured evidence state, but P3 does not add a new generative partial-answer provider.

## Artifacts

- `P3_SUFFICIENCY_CONTRACT.md`
- `P3_RETRY_POLICY.md`
- `P3_ABSTENTION_CONTRACT.md`
- `P3_EVALUATION_RESULTS.md`
- `P3_LOCKED_R8_REGRESSION.md`
- `P3_TEST_RESULTS.md`
- `P3_RUNTIME_INTEGRITY.md`
- `P3_GIT_INTEGRITY.md`
- `p3_cases.json`, `p3_metrics.json`, `p3_retry_traces.json`, `p3_abstentions.json`
- `locked_r8_results.json`
