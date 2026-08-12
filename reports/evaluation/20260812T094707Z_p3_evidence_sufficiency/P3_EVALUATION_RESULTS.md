# P3 Evaluation Results

Provider-free fixture: `tests/golden/p3_evidence_sufficiency_cases.json`

Cases: 17/17 PASS

| Case | Initial status | Retry eligible? | Retry strategy | Retry status | Final action | Correct? |
| --- | --- | --- | --- | --- | --- | --- |
| sufficient_first_pass | SUFFICIENT | No | None | N/A | ANSWER | Yes |
| retry_recovers | INSUFFICIENT | Yes | Query representation refinement | SUFFICIENT | ANSWER | Yes |
| retry_no_recovery | INSUFFICIENT | Yes | Query representation refinement | INSUFFICIENT | ABSTAIN | Yes |
| critical_missing_pregnancy | CRITICAL_EVIDENCE_MISSING | Yes | Safety-role refinement | CRITICAL_EVIDENCE_MISSING | ABSTAIN | Yes |
| out_of_scope | INSUFFICIENT | No | None | N/A | ABSTAIN | Yes |
| source_provenance_failure | INSUFFICIENT | No | None | N/A | ABSTAIN | Yes |
| retrieval_provider_failure | INSUFFICIENT | No | None | N/A | ABSTAIN | Yes |
| packer_critical_overflow | CRITICAL_EVIDENCE_MISSING | No | None | N/A | ABSTAIN | Yes |
| sentinel_ba_nhon | SUFFICIENT | No | None | N/A | ANSWER | Yes |
| sentinel_mo_hoi | SUFFICIENT | No | None | N/A | ANSWER | Yes |
| sentinel_tazorac | SUFFICIENT | No | None | N/A | ANSWER | Yes |
| sentinel_pregnancy | SUFFICIENT | No | None | N/A | ANSWER | Yes |
| sentinel_danh_gia | SUFFICIENT | No | None | N/A | ANSWER | Yes |
| sentinel_nguon | SUFFICIENT | No | None | N/A | ANSWER | Yes |
| sentinel_bit_tac | SUFFICIENT | No | None | N/A | ANSWER | Yes |
| sentinel_lo_chan_long | SUFFICIENT | No | None | N/A | ANSWER | Yes |
| sentinel_oxy_hoa | SUFFICIENT | No | None | N/A | ANSWER | Yes |

## Metrics

| Metric | Numerator | Denominator | Scope | Gate? | Observed |
| --- | --- | --- | --- | --- | --- |
| Sufficiency rate | First-pass sufficient | All 17 cases | P3 fixture | Diagnostic | 58.82% |
| Insufficiency precision | Correct insufficient predictions | All insufficient predictions | Labeled fixture | Yes | 100% |
| Insufficiency recall | Correct insufficient predictions | All labeled initial-insufficient cases | Labeled fixture | Yes | 100% |
| Retry trigger rate | Triggered retries | 3 labeled retry-eligible cases | P3 fixture | Yes | 100% |
| Retry success rate | Recovered sufficient cases | 3 retries | P3 fixture | Diagnostic | 33.33% |
| Unnecessary retry rate | Retries on first-pass-sufficient cases | First-pass-sufficient cases | P3 fixture | Yes | 0% |
| Abstention rate | Final abstentions | All 17 cases | P3 fixture | Diagnostic | 35.29% |
| Correct abstention rate | Correct abstentions | Expected abstentions | P3 fixture | Yes | 100% |
| Critical detection | Detected critical misses | 2 critical-missing cases | P3 fixture | Yes | 100% |
| Provenance detection | Detected provenance failures | 1 provenance case | P3 fixture | Yes | 100% |
| Unsafe answer after insufficiency | Unsafe answers | Critical insufficient cases | P3 fixture | Yes | 0 |
| Average attempts | Sum of attempts | All 17 cases | P3 fixture | Yes | 1.1765 |

Provider-free timing: mean no-retry evaluator path 0.0445 ms; mean retry planning plus attempt-1 evaluator overhead 0.0808 ms. These values exclude retrieval and external provider latency.

The A/B result is stored in `p3_cases.json`: A0 lacks bounded retry/structured abstention; A1 passes 17/17 with zero unsafe answer after insufficient evidence.
