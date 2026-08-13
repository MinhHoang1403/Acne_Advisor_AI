# P4.5 Production Shadow Calibration

## Scope

This evaluation asks how P4 classifies claims in naturally generated, production-like answers. It does not change Retrieval V5, P3, P4 semantics, prompts, models, indexes, or enforcement state.

## Frozen Dataset

- Version: `p45_production_shadow_v1`
- Questions: 75 across 23 categories
- Language: 48 Vietnamese, 12 English, 15 mixed
- Critical-question seed set: 18/75 (24%)
- Natural outputs: 75 successful, 0 generation failures
- Cache: 75 fresh generations, 0 cache hits
- Extracted claims: 522 total, 126 P4-critical
- Mean claims per answer: 6.96

The generation run used the configured provider chain unchanged. The captured outputs comprise 23 `gemini-3.5-flash`, 39 `gemini-3.1-flash-lite`, 10 deterministic system responses, and 3 rule-based guardrail responses. Exact reproduction can vary with provider behavior; the actual reviewed answers are frozen by SHA256.

## Shadow Observations

P4 prediction distribution before human review:

| Verdict | Count |
| --- | ---: |
| SUPPORTED | 20 |
| PARTIALLY_SUPPORTED | 142 |
| UNSUPPORTED | 193 |
| CONTRADICTED | 114 |
| NO_EVIDENCE | 53 |
| VERIFIER_ERROR | 0 |

These are model predictions, not gold labels. The high non-supported/contradicted count is a calibration signal requiring human evidence-support review; it must not be interpreted as measured accuracy.

P3 returned 63 `SUFFICIENT`, 3 `INSUFFICIENT`, 5 `CRITICAL_EVIDENCE_MISSING`, and 4 non-P3 guardrail paths. P4 remained `shadow` for every case and did not rewrite user-facing answers.

## Operational Snapshot

- Mean generation latency: 12.8093 seconds
- Median: 12.1430 seconds
- P95: 24.4628 seconds
- Maximum: 30.3378 seconds
- Pipeline fingerprint: `af95aafe84645cf9a7987b91`
- Verifier: `deterministic_entailment_v1`, no external verifier calls

## Decision

Final readiness scoring is intentionally blocked until real human evidence-support labels are supplied. Current state: `P4_5_WAITING_FOR_HUMAN_REVIEW`; enforcement remains disabled.
