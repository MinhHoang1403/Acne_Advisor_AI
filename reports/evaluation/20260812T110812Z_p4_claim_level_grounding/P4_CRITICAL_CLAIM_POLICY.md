# P4 Critical Claim Policy

Future enforcement may allow a critical claim only when its verdict is exactly `SUPPORTED`. Partial, unsupported, contradicted, no-evidence, and verifier-error critical claims all fail the gate.

Shadow actions are diagnostic only:

- all supported: `WOULD_ALLOW`;
- non-critical partial: `WOULD_REWRITE_PARTIAL`;
- non-critical unsupported/contradicted/no-evidence: `WOULD_DROP_NONCRITICAL` or `WOULD_ABSTAIN` when no claim remains;
- any critical failure: `WOULD_BLOCK_CRITICAL`;
- verifier failure without a critical failure: `VERIFIER_UNAVAILABLE`.

P3 has precedence. `INSUFFICIENT` or `CRITICAL_EVIDENCE_MISSING` causes P4 to record `skipped_p3_precedence`; P4 cannot override P3 abstention or request another retrieval retry.

Release defaults:

```text
P4_MODE=shadow
P4_CRITICAL_ENFORCEMENT_READY=false
P4_ALL_CLAIM_ENFORCEMENT_READY=false
```

An enforcement mode request degrades to effective shadow unless its separate readiness interlock is true. This release does not authorize either interlock.
