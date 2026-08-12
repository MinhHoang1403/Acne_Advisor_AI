# P4 Claim-Level Grounding Report

Decision before PR: `P4_SHADOW_READY_TO_MERGE`.

## Runtime position

```text
Retrieval V5
  -> P3 evidence sufficiency / bounded retry / abstention
  -> draft generation
  -> generation fallback validation
  -> P4 claim extraction
  -> source-backed packed-evidence mapping
  -> deterministic entailment verdicts
  -> critical shadow policy
  -> existing finalizer / answer verifier / cache
```

P4 defaults to `shadow`; the draft passed to the existing finalizer is byte-for-byte unchanged. `P4_MODE=disabled` is the immediate rollback. Enforcement requests are additionally interlocked by explicit readiness flags, both false in this release.

## Result

- Frozen fixture: 32 claim-evidence pairs, 14 critical.
- Verdict distribution: 12 supported, 5 partial, 5 unsupported, 5 contradicted, 5 no-evidence.
- Verification accuracy: 32/32.
- Evidence mapping accuracy: 32/32.
- Critical false allow: 0/10 non-supported critical claims.
- External verifier calls: 0.
- Mean provider-free P4 time: 73.058 ms; this is not live-provider latency.
- Production answer change rate in shadow: 0/32.

## Release interpretation

The deterministic fixture proves plumbing, typed contracts, provenance rejection, verdict separation, and safe shadow behavior. It does not establish broad clinical entailment calibration on naturally generated answers. Therefore critical and all-claim enforcement remain disabled.

Machine-readable results are in `p4_entailment_results.json`, `p4_claims.json`, `p4_confusion_matrix.json`, `p4_shadow_decisions.json`, and `p4_metrics.json`.
