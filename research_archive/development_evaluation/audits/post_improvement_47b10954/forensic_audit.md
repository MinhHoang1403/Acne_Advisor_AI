# Post-Improvement Formal Evaluation — Forensic Audit

## 1. Executive Summary

The saved run is complete and internally consistent: 100/100 cases, zero infrastructure failures, and all identity fields match the frozen system. Retrieval-facing metrics improved materially, while 14 evidence-gap false-generates reduced NRR to 16/30. The dominant remaining high-severity defect is not hallucinated natural-language assertion: all 14 failed gap answers reject the unsupported premise, but the evidence assessment/structured action still emits `generate/evidence_sufficient`.

The audit assigns exactly one primary classification to every case. It recommends a narrow evidence-gap action-contract correction before final freeze; no production change is made here.

## 2. Run Integrity

- Records: 100/100; answerable: 70; evidence-gap: 30.
- Unique IDs: 100; missing/duplicate IDs: 0/0.
- Infrastructure failures: 0.
- Benchmark SHA: `f61d6807c0ce39f902936844d810562c486f1bcadaa57a8a2da0e460ad7e534b`.
- System SHA: `47b10954d217de91cf8919650b31dc7569ec1f0e`.
- KB build: `94d613bc9b33628de3ef`.
- Pipeline fingerprint: `5991f3c8effb67091fd6274c` on all 100 records.
- Repository head recorded at run: `046634fba47efa858f45ae588302d6524213d375`.
- Evaluator / RAGChecker: `gpt-5.4-mini-2026-03-17` / `0.1.9`.

### Source artifact hashes

| Artifact | Pre-audit SHA | Post-audit SHA | Unchanged |
|---|---|---|---|
| evaluation/results/post_improvement_47b10954/raw_results.json | 20bf0af0e7b697c5b3a3bcc78c5c5fe4db6d9b9985022c8655cb94bb0e794427 | 20bf0af0e7b697c5b3a3bcc78c5c5fe4db6d9b9985022c8655cb94bb0e794427 | True |
| evaluation/results/post_improvement_47b10954/case_metrics.csv | d603903ce692905a3ec4e03f4c03a10c19c64a602f8d802620baca9a74f33359 | d603903ce692905a3ec4e03f4c03a10c19c64a602f8d802620baca9a74f33359 | True |
| evaluation/results/post_improvement_47b10954/metrics_summary.csv | a81752fadf2111f952af31fa2d8fbec546c1847f163e67c3cad529867d56406e | a81752fadf2111f952af31fa2d8fbec546c1847f163e67c3cad529867d56406e | True |
| evaluation/results/post_improvement_47b10954/ragchecker_checkpoint.json | 74c787a37853e16c3875c3243a367489587a085aa51ef81a0e2531a7d9613ea7 | 74c787a37853e16c3875c3243a367489587a085aa51ef81a0e2531a7d9613ea7 | True |
| evaluation/results/post_improvement_47b10954/evaluator_calibration_results.json | b5a50e23b03fadf6161a21dee7c3462c85828b5988a1b945a9a6f1cd7d8f92e8 | b5a50e23b03fadf6161a21dee7c3462c85828b5988a1b945a9a6f1cd7d8f92e8 | True |
| evaluation/results/post_improvement_47b10954/calibration_adjudication.json | 8b659baa918a2378b76afba101b5bc088fa097d7e4b00d6617cb3ced50234552 | 8b659baa918a2378b76afba101b5bc088fa097d7e4b00d6617cb3ced50234552 | True |
| evaluation/formal_evaluation.ipynb | 14deaa1cf16babf18af7272cb0166d50603eec31b01913e2979c0c6b11d1abd2 | 14deaa1cf16babf18af7272cb0166d50603eec31b01913e2979c0c6b11d1abd2 | True |
| evaluation/benchmark_100.json | ad42fbc333adcd70c265b8873d0e241e5401224d1cf9f3e7faa08adf1001773f | ad42fbc333adcd70c265b8873d0e241e5401224d1cf9f3e7faa08adf1001773f | True |
| evaluation/benchmark_manifest.json | d67fb1419d3237643816be7ca342efef6ba6a33833f6ea0a8db4e0f558a69a98 | d67fb1419d3237643816be7ca342efef6ba6a33833f6ea0a8db4e0f558a69a98 | True |
| evaluation/evaluator_calibration.json | 1eb50921c25512404451ed97bad1cf31a151d19a84d946895fa7bd5dc67773dd | 1eb50921c25512404451ed97bad1cf31a151d19a84d946895fa7bd5dc67773dd | True |
| evaluation/results/formal_run_baseline/raw_results.json | 08ff7c624ec633abcce1e877dc55c26cc502bd828134472cb1e9436ece88d16a | 08ff7c624ec633abcce1e877dc55c26cc502bd828134472cb1e9436ece88d16a | True |
| evaluation/results/formal_run_baseline/case_metrics.csv | 6a6b07db0cb227c43917aef5df15f0d09ada7e58599d91ff4ac61538fd6050c4 | 6a6b07db0cb227c43917aef5df15f0d09ada7e58599d91ff4ac61538fd6050c4 | True |
| evaluation/results/formal_run_baseline/metrics_summary.csv | 5f33ae584c79d16dfcba061bc02ee9ae1c1beadc7c3017ef6e5dbda2f35c21a1 | 5f33ae584c79d16dfcba061bc02ee9ae1c1beadc7c3017ef6e5dbda2f35c21a1 | True |
| evaluation/results/formal_run_baseline/ragchecker_checkpoint.json | 4b51bc1563ea3c9e4ecdeec7171e89a8a067de4769fee8b48518757c481da187 | 4b51bc1563ea3c9e4ecdeec7171e89a8a067de4769fee8b48518757c481da187 | True |
| evaluation/results/formal_run_baseline/evaluator_calibration_results.json | 4000ad61b71d6be0cb981ff83faf1292f037dc1b05e75b09edaa0e615298cc85 | 4000ad61b71d6be0cb981ff83faf1292f037dc1b05e75b09edaa0e615298cc85 | True |

## 3. Metric Comparison

| Metric | Baseline (%) | Post-improvement (%) | Delta pp |
|---|---|---|---|
| Claim Recall | 60.5000 | 71.0000 | +10.5000 |
| Context Precision | 26.3000 | 39.2000 | +12.9000 |
| Faithfulness | 74.3000 | 85.6000 | +11.3000 |
| Claim F1 | 37.3000 | 42.0000 | +4.7000 |
| Negative Rejection Rate | 70.0000 | 53.3333 | -16.6667 |

Retrieval/reranking/packing improvements worked in aggregate: Claim Recall rose 10.5 pp and Context Precision 12.9 pp, with Faithfulness up 11.3 pp. Case traces nevertheless show residual exact-provenance candidate misses and candidate-to-pack loss; the gain is material, not complete.

## 4. Calibration Audit

Automatic calibration remained 7/8 extraction, 12/12 checking, and 19/20 repeat consistency with `CALIBRATION_REVIEW_REQUIRED` on `CAL-EXT-03`. The separate adjudication records `APPROVED`, producing effective `CALIBRATION_READY_FOR_FORMAL_RUN`. This is classified as `RESEARCHER_ADJUDICATED_SEMANTIC_VARIATION`; automatic counts were not rewritten.

## 5. Overall Case Classification

| Primary classification | Count |
|---|---|
| EXPECTED_METRIC_BEHAVIOR | 2 |
| PASS | 42 |
| SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP | 14 |
| SYSTEM_BUG_GENERATION_OMISSION | 4 |
| SYSTEM_BUG_GENERATION_OVER_ANSWERING | 7 |
| SYSTEM_BUG_MULTI_TURN_CONTEXT | 5 |
| SYSTEM_BUG_PACKER_SELECTION | 12 |
| SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | 12 |
| SYSTEM_BUG_SAFETY_SHORT_CIRCUIT | 1 |
| SYSTEM_BUG_UNSUPPORTED_GENERATION | 1 |

The CSV contains exactly 100 exclusive primary classifications.

## 6. Root-Cause Distribution

| Dominant layer | Count |
|---|---|
| evidence_assessment | 14 |
| generation | 12 |
| metric_behavior | 2 |
| multi_turn | 5 |
| none | 42 |
| packing | 12 |
| retrieval | 12 |
| safety | 1 |

`none` denotes PASS cases. `metric_behavior` is reserved for low-score cases where supported detail or claim segmentation explains the score without a material system defect.

## 7. Answerable Cases

False abstentions: 6. Claim Recall zero cases: 11. Faithfulness below 75: 10.

### False abstention list

| Case | Category | Attempts | Candidate hits | Pack hits | Primary cause |
|---|---|---|---|---|---|
| ANS-DEF-004 | definition_classification | 2 | 0 | 0 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-DEF-010 | definition_classification | 2 | 0 | 0 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-TRT-001 | treatment_use_combination | 1 | 1 | 0 | SYSTEM_BUG_PACKER_SELECTION |
| ANS-REF-003 | referral_care_seeking | 1 | 0 | 0 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-MUL-009 | follow_up_continuity | 2 | 0 | 0 | SYSTEM_BUG_MULTI_TURN_CONTEXT |
| ANS-MUL-013 | repeated_question_history_isolation | 0 | 0 | 0 | SYSTEM_BUG_MULTI_TURN_CONTEXT |

### Claim Recall == 0

| Case | Action | Faithfulness | Primary cause |
|---|---|---|---|
| ANS-DEF-004 | abstain | 0.0 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-DEF-010 | abstain | 0.0 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-TRT-011 | generate | 100.0 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-CMP-010 | generate | 0.0 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-REF-003 | abstain | 0.0 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-REF-004 | generate | 100.0 | SYSTEM_BUG_PACKER_SELECTION |
| ANS-MUL-006 | generate | 100.0 | SYSTEM_BUG_MULTI_TURN_CONTEXT |
| ANS-MUL-009 | abstain | 0.0 | SYSTEM_BUG_MULTI_TURN_CONTEXT |
| ANS-MUL-010 | null | 0.0 | SYSTEM_BUG_SAFETY_SHORT_CIRCUIT |
| ANS-MUL-013 | abstain | 0.0 | SYSTEM_BUG_MULTI_TURN_CONTEXT |
| ANS-MUL-015 | generate | 75.0 | SYSTEM_BUG_MULTI_TURN_CONTEXT |

The other recall buckets are: 0<CR<50 = 6 cases; 50<=CR<100 = 14 cases. Their IDs are retained in `forensic_summary.json`; each row's candidate/pack evidence distinguishes retrieval undercoverage from packer loss.

### Faithfulness < 75

| Case | Action | CR | Faithfulness | Primary cause |
|---|---|---|---|---|
| ANS-DEF-004 | abstain | 0.0 | 0.0 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-DEF-010 | abstain | 0.0 | 0.0 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-TRT-001 | abstain | 50.0 | 0.0 | SYSTEM_BUG_PACKER_SELECTION |
| ANS-ADV-003 | generate | 50.0 | 54.54545454545454 | SYSTEM_BUG_UNSUPPORTED_GENERATION |
| ANS-CMP-010 | generate | 0.0 | 0.0 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-REF-002 | generate | 100.0 | 50.0 | EXPECTED_METRIC_BEHAVIOR |
| ANS-REF-003 | abstain | 0.0 | 0.0 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-MUL-009 | abstain | 0.0 | 0.0 | SYSTEM_BUG_MULTI_TURN_CONTEXT |
| ANS-MUL-010 | null | 0.0 | 0.0 | SYSTEM_BUG_SAFETY_SHORT_CIRCUIT |
| ANS-MUL-013 | abstain | 0.0 | 0.0 | SYSTEM_BUG_MULTI_TURN_CONTEXT |

## 8. Evidence-Gap Cases and NRR Regression

NRR is 16/30 (53.3333%), down from 21/30 (70.0%). All 14 failures are structured false-generates. Ten absolute-guarantee answers explicitly say “Không” but still emit `generate/evidence_sufficient` (type B). Four exact/relationship cases reject the unsupported precision and then provide nearby supported facts (type C). No saved answer positively asserts the requested unsupported absolute/quantity as fact.

### NRR failures by category

| Category | Failures |
|---|---|
| unsupported_absolute_certainty_guarantee | 10 |
| unsupported_comparison_relationship_specificity | 3 |
| unsupported_exact_quantity_time | 1 |

### All 30 gap cases

| Case | Category | Action | Reason | NRR correct | Primary classification |
|---|---|---|---|---|---|
| GAP-ABS-001 | unsupported_absolute_certainty_guarantee | generate | evidence_sufficient | false | SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP |
| GAP-ABS-002 | unsupported_absolute_certainty_guarantee | generate | evidence_sufficient | false | SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP |
| GAP-ABS-003 | unsupported_absolute_certainty_guarantee | generate | evidence_sufficient | false | SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP |
| GAP-ABS-004 | unsupported_absolute_certainty_guarantee | generate | evidence_sufficient | false | SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP |
| GAP-ABS-005 | unsupported_absolute_certainty_guarantee | generate | evidence_sufficient | false | SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP |
| GAP-ABS-006 | unsupported_absolute_certainty_guarantee | generate | evidence_sufficient | false | SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP |
| GAP-ABS-007 | unsupported_absolute_certainty_guarantee | generate | evidence_sufficient | false | SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP |
| GAP-ABS-008 | unsupported_absolute_certainty_guarantee | generate | evidence_sufficient | false | SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP |
| GAP-ABS-009 | unsupported_absolute_certainty_guarantee | generate | evidence_sufficient | false | SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP |
| GAP-ABS-010 | unsupported_absolute_certainty_guarantee | generate | evidence_sufficient | false | SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP |
| GAP-EXA-001 | unsupported_exact_quantity_time | abstain | evidence_gap | true | PASS |
| GAP-EXA-002 | unsupported_exact_quantity_time | abstain | evidence_gap | true | PASS |
| GAP-EXA-003 | unsupported_exact_quantity_time | abstain | evidence_gap | true | PASS |
| GAP-EXA-004 | unsupported_exact_quantity_time | abstain | evidence_gap | true | PASS |
| GAP-EXA-005 | unsupported_exact_quantity_time | generate | evidence_sufficient | false | SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP |
| GAP-EXA-006 | unsupported_exact_quantity_time | abstain | evidence_gap | true | PASS |
| GAP-EXA-007 | unsupported_exact_quantity_time | abstain | evidence_gap | true | PASS |
| GAP-EXA-008 | unsupported_exact_quantity_time | abstain | evidence_gap | true | PASS |
| GAP-EXA-009 | unsupported_exact_quantity_time | abstain | evidence_gap | true | PASS |
| GAP-EXA-010 | unsupported_exact_quantity_time | abstain | evidence_gap | true | PASS |
| GAP-REL-001 | unsupported_comparison_relationship_specificity | abstain | evidence_gap | true | PASS |
| GAP-REL-002 | unsupported_comparison_relationship_specificity | generate | evidence_sufficient | false | SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP |
| GAP-REL-003 | unsupported_comparison_relationship_specificity | abstain | evidence_gap | true | PASS |
| GAP-REL-004 | unsupported_comparison_relationship_specificity | abstain | evidence_gap | true | PASS |
| GAP-REL-005 | unsupported_comparison_relationship_specificity | generate | evidence_sufficient | false | SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP |
| GAP-REL-006 | unsupported_comparison_relationship_specificity | abstain | evidence_gap | true | PASS |
| GAP-REL-007 | unsupported_comparison_relationship_specificity | abstain | evidence_gap | true | PASS |
| GAP-REL-008 | unsupported_comparison_relationship_specificity | abstain | evidence_gap | true | PASS |
| GAP-REL-009 | unsupported_comparison_relationship_specificity | abstain | evidence_gap | true | PASS |
| GAP-REL-010 | unsupported_comparison_relationship_specificity | generate | evidence_sufficient | false | SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP |

### Detailed 14 failures

| Case | Category | Pack | Evidence assessment outcome | Behavior type |
|---|---|---|---|---|
| GAP-ABS-001 | unsupported_absolute_certainty_guarantee | 3 | generate/evidence_sufficient | B |
| GAP-ABS-002 | unsupported_absolute_certainty_guarantee | 5 | generate/evidence_sufficient | B |
| GAP-ABS-003 | unsupported_absolute_certainty_guarantee | 3 | generate/evidence_sufficient | B |
| GAP-ABS-004 | unsupported_absolute_certainty_guarantee | 5 | generate/evidence_sufficient | B |
| GAP-ABS-005 | unsupported_absolute_certainty_guarantee | 5 | generate/evidence_sufficient | B |
| GAP-ABS-006 | unsupported_absolute_certainty_guarantee | 4 | generate/evidence_sufficient | B |
| GAP-ABS-007 | unsupported_absolute_certainty_guarantee | 3 | generate/evidence_sufficient | B |
| GAP-ABS-008 | unsupported_absolute_certainty_guarantee | 4 | generate/evidence_sufficient | B |
| GAP-ABS-009 | unsupported_absolute_certainty_guarantee | 3 | generate/evidence_sufficient | B |
| GAP-ABS-010 | unsupported_absolute_certainty_guarantee | 5 | generate/evidence_sufficient | B |
| GAP-EXA-005 | unsupported_exact_quantity_time | 8 | generate/evidence_sufficient | C |
| GAP-REL-002 | unsupported_comparison_relationship_specificity | 3 | generate/evidence_sufficient | C |
| GAP-REL-005 | unsupported_comparison_relationship_specificity | 4 | generate/evidence_sufficient | C |
| GAP-REL-010 | unsupported_comparison_relationship_specificity | 3 | generate/evidence_sufficient | C |

`GAP-ABS-001` rejects a permanent cure guarantee in text but generates structurally. `GAP-EXA-005` says no exact temperature is available, then gives general washing advice while generating structurally. This is predominantly an evidence-assessment/action-policy mismatch, not 14 hallucinations.

## 9. Multi-Turn Audit

| Cohort | N | CR | CP | Faithfulness | F1 |
|---|---|---|---|---|---|
| single_turn | 55 | 77.4156 | 41.6840 | 87.8606 | 45.1587 |
| multi_turn | 15 | 47.6508 | 30.2302 | 77.2222 | 30.4447 |

| Case | Category | Action | CR | F1 | Primary classification |
|---|---|---|---|---|---|
| ANS-MUL-001 | pronoun_coreference | generate | 100.0 | 33.33333333333333 | SYSTEM_BUG_GENERATION_OMISSION |
| ANS-MUL-002 | pronoun_coreference | generate | 100.0 | 57.14285714285715 | PASS |
| ANS-MUL-003 | pronoun_coreference | generate | 71.42857142857143 | 58.823529411764696 | SYSTEM_BUG_PACKER_SELECTION |
| ANS-MUL-004 | pronoun_coreference | generate | 66.66666666666666 | 57.14285714285715 | SYSTEM_BUG_PACKER_SELECTION |
| ANS-MUL-005 | pronoun_coreference | generate | 25.0 | 36.36363636363637 | SYSTEM_BUG_PACKER_SELECTION |
| ANS-MUL-006 | follow_up_continuity | generate | 0.0 | 0.0 | SYSTEM_BUG_MULTI_TURN_CONTEXT |
| ANS-MUL-007 | follow_up_continuity | generate | 100.0 | 66.66666666666666 | PASS |
| ANS-MUL-008 | follow_up_continuity | generate | 66.66666666666666 | 26.666666666666668 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-MUL-009 | follow_up_continuity | abstain | 0.0 | 0.0 | SYSTEM_BUG_MULTI_TURN_CONTEXT |
| ANS-MUL-010 | explicit_topic_switch | null | 0.0 | 0.0 | SYSTEM_BUG_SAFETY_SHORT_CIRCUIT |
| ANS-MUL-011 | explicit_topic_switch | generate | 60.0 | 26.08695652173913 | SYSTEM_BUG_PACKER_SELECTION |
| ANS-MUL-012 | explicit_topic_switch | generate | 25.0 | 11.111111111111112 | SYSTEM_BUG_MULTI_TURN_CONTEXT |
| ANS-MUL-013 | repeated_question_history_isolation | abstain | 0.0 | 0.0 | SYSTEM_BUG_MULTI_TURN_CONTEXT |
| ANS-MUL-014 | repeated_question_history_isolation | generate | 100.0 | 50.0 | PASS |
| ANS-MUL-015 | repeated_question_history_isolation | generate | 0.0 | 33.33333333333333 | SYSTEM_BUG_MULTI_TURN_CONTEXT |

Multi-turn remains operationally weaker: it contains 2 false abstentions, 1 missing structured action, and 5 primary multi-turn defects. These are descriptive benchmark results, not a statistical-significance claim.

Additional multi-turn counts: retrieval/packing failures = 5; generation omissions = 1; over-answering = 0.

## 10. Retrieval / Reranker / Packer Audit

- Answerable pack-size distribution: `{'0': 2, '3': 14, '4': 17, '5': 8, '6': 2, '7': 3, '8': 24}`.
- Mean/median pack size: 5.3571 / 5.0000.
- Mean/median packed chars: 5366.3857 / 5826.5000.
- CR>=75 with CP<=25: 11; CR=100 with CP<=25: 10.
- Saved final reranker traces: 96; succeeded: 96; fallback/failure: 0/0.
- Reranker mean/median latency: 2953.6026/3056.7505 ms.
- Mean fused/eligible/selected candidates in saved final traces: 26.2396/26.4896/5.1771.

### Context Precision by pack size

| Packed items | Cases | Mean CP |
|---|---|---|
| 0 | 2 | 0.0 |
| 3 | 14 | 73.8095 |
| 4 | 17 | 48.5294 |
| 5 | 8 | 35.0 |
| 6 | 2 | 58.3333 |
| 7 | 3 | 9.5238 |
| 8 | 24 | 19.2708 |

Low CP is mixed: some full packs contain useful but non-gold context and incur expected precision penalties; other rows show exact gold candidates displaced before packing. No reranker runtime failure explains the failed cases.

### Provider fallback distribution

| Requested provider | Requested model | Actual provider | Actual model | Fallback | Cases |
|---|---|---|---|---|---|
| gemini | gemini-3.5-flash-lite | gemini | gemini-3.1-flash-lite | true | 5 |
| gemini | gemini-3.5-flash-lite | gemini | gemini-3.5-flash-lite | false | 72 |
| none | none | system | none | false | 23 |

Fallback changed the model for exactly 5 cases: `ANS-TRT-005, ANS-TRT-011, ANS-CMP-004, ANS-MUL-012, GAP-ABS-008`. All five primary rate-limit failures recovered on Gemini 3.1 Flash-Lite; Ollama uses and infrastructure failures were both zero. This limited same-provider fallback can change individual generation wording, but it does not explain retrieval metrics or the systematic 14-case structured NRR pattern.

## 11. Retry Audit

| Measure | Value |
|---|---|
| attempt_distribution | {0: 4, 1: 90, 2: 6} |
| retry_count | 6 |
| retry_rate_pct | 6.0 |
| answerable_retry_count | 3 |
| gap_retry_count | 3 |
| retry_to_generate | 1 |
| retry_to_abstain | 5 |
| retry_final_correct | 2 |
| retry_final_incorrect | 4 |
| recovered_exact_gold_evidence | 0 |
| changed_final_action_correctly | 2 |
| added_noise_full_pack_cp_below_25 | 3 |
| failed_to_help | 4 |
| baseline_attempt_distribution | {0: 1, 1: 77, 2: 22} |

| Case | Family | Final action | Outcome | Primary cause |
|---|---|---|---|---|
| ANS-DEF-004 | answerable_single_turn | abstain | final_action_incorrect | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-DEF-010 | answerable_single_turn | abstain | final_action_incorrect | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-MUL-009 | answerable_multi_turn | abstain | final_action_incorrect | SYSTEM_BUG_MULTI_TURN_CONTEXT |
| GAP-ABS-003 | evidence_gap | generate | final_action_incorrect | SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP |
| GAP-EXA-001 | evidence_gap | abstain | final_action_correct | PASS |
| GAP-EXA-006 | evidence_gap | abstain | final_action_correct | PASS |

Retries were rare (6%). They did not systematically recover correctness: the final action remained incorrect in the majority of retried cases. Saved traces support this descriptive result but do not expose a counterfactual final answer without retry.

## 12. Generation Audit

Claim F1 rose only 4.7 pp because retrieval gains did not remove response-side scope mismatch. Among F1<40 rows, causes include false abstention/missing claims, substantial extra supported claims, upstream evidence undercoverage, multi-turn loss, and a small number of unsupported response claims.

| Primary cause among F1<40 | Cases |
|---|---|
| EXPECTED_METRIC_BEHAVIOR | 1 |
| SYSTEM_BUG_GENERATION_OMISSION | 3 |
| SYSTEM_BUG_GENERATION_OVER_ANSWERING | 7 |
| SYSTEM_BUG_MULTI_TURN_CONTEXT | 5 |
| SYSTEM_BUG_PACKER_SELECTION | 9 |
| SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | 9 |
| SYSTEM_BUG_SAFETY_SHORT_CIRCUIT | 1 |
| SYSTEM_BUG_UNSUPPORTED_GENERATION | 1 |

### CR=100, Faithfulness=100, F1<40

| Case | F1 | Response/gold claims | Primary classification |
|---|---|---|---|
| ANS-DEF-001 | 30.76923076923077 | 11/1 | SYSTEM_BUG_GENERATION_OVER_ANSWERING |
| ANS-DEF-007 | 25.0 | 14/4 | SYSTEM_BUG_GENERATION_OVER_ANSWERING |
| ANS-TRT-005 | 25.0 | 7/3 | SYSTEM_BUG_GENERATION_OVER_ANSWERING |
| ANS-MEC-002 | 33.333333333333336 | 10/2 | SYSTEM_BUG_GENERATION_OVER_ANSWERING |
| ANS-MEC-004 | 33.333333333333336 | 5/3 | EXPECTED_METRIC_BEHAVIOR |
| ANS-MEC-005 | 20.0 | 9/2 | SYSTEM_BUG_GENERATION_OVER_ANSWERING |
| ANS-REF-006 | 25.0 | 7/1 | SYSTEM_BUG_GENERATION_OVER_ANSWERING |
| ANS-MUL-001 | 33.33333333333333 | 4/2 | SYSTEM_BUG_GENERATION_OMISSION |

The listed cases are not retrieval failures. Their packed evidence covers gold and generated claims are grounded; low F1 primarily reflects over-answering or claim-granularity/gold-scope penalties.

## 13. Safety Short-Circuit Audit

`ANS-MUL-010` produced a deterministic pregnancy warning with no retrieval, no packed context, and null structured action/reason. The safety content is appropriate, but the missing structured benchmark contract is a high-severity consistency defect. Faithfulness zero here is empty-context metric behavior, not evidence of a hallucinated unsafe answer.

## 14. Evaluator and Metric Behavior

No confirmed case-level evaluator anomaly was proven. `ANS-MUL-006` and `ANS-MUL-015` have strict-looking scores, but both also omit part of the exact gold scope. The calibration wording issue remains separately adjudicated and is not counted as a production defect.

Expected metric behavior is visible where useful supported details are outside narrow gold scope, Context Precision penalizes non-gold but relevant chunks, or claim extraction granularity expands the response claim count.

## 15. Formal Run #1 Comparison

| Case-level comparison | Count |
|---|---|
| IMPROVED | 34 |
| MIXED | 12 |
| REGRESSED | 21 |
| UNCHANGED | 33 |

Criteria were fixed before application: action correctness transitions dominate; otherwise >=10 pp gains/losses across individual metrics yield improved/regressed, opposing material movements yield mixed, and no material movement yields unchanged. No composite score was used.

The historical audit reported 57 system bugs, 3 evaluator anomalies, 19 expected-metric cases, and 21 passes. Direct count comparison is approximate because this audit separates upstream retrieval, packer loss, structured gap actions, and metric behavior more strictly. The post-improvement run has fewer false abstentions and stronger retrieval metrics, but five additional gap action failures versus baseline explain the NRR regression.

## 16. Remaining Bottlenecks

1. **Evidence-gap structured action (14 HIGH):** unsupported premises are rejected in prose but accepted structurally.
2. **Residual retrieval/packing undercoverage:** exact gold provenance still misses candidate or packed stages in multiple answerable cases.
3. **Multi-turn contract (6 primary defects):** includes false abstention, topic-switch safety bypass, and claim loss.
4. **Generation scope control:** supported over-answering depresses F1 despite high recall/faithfulness.
5. **Safety structured-action consistency (1 HIGH):** appropriate deterministic warning lacks action/reason trace.

## 17. Recommendation

**TARGETED_PRODUCTION_FIX_JUSTIFIED_BEFORE_FINAL_FREEZE**

The justification is narrow and evidence-based: 14 HIGH failures share one localized evidence-gap action-contract pattern, and one safety short-circuit lacks structured action. A separate approved task should address only these contracts and justify any new formal run. This audit makes no production change.
