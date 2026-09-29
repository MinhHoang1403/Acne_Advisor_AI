# Benchmark Current-KB Adjudication

Trạng thái cuối: **STEP3_ARCHITECTURE_BLOCKER**.

Phạm vi review này chỉ xác nhận quan hệ kỹ thuật giữa câu claim và văn bản nguồn của dự án. `TECHNICALLY_CLEAR` không phải chứng nhận độc lập về chân lý lâm sàng. Không sửa production runtime, benchmark lịch sử, KB, Qdrant, cache hay Step 3.

## 1. Identity

- Branch: `research/benchmark-current-kb-adjudication`
- Production base SHA: `059144f343a762f309ae96c2b75b44a27a3b401d`
- Pipeline fingerprint: `0a8d129a215c884f2d9654e9`
- Benchmark checksum: `f61d6807c0ce39f902936844d810562c486f1bcadaa57a8a2da0e460ad7e534b`
- Old benchmark KB: `94d613bc9b33628de3ef`
- Current KB: `d4a1819fe7fb77fe1f40`
- Collection: `acne_knowledge__d4a1819fe7fb77fe1f40` (536 points)

## 2. Benchmark inventory

- 100 cases: 55 answerable single-turn, 15 answerable multi-turn, 30 evidence-gap.
- 104 gold claims; 37 unique old evidence IDs.
- 30 negative labels: 10 absolute guarantee, 10 exact quantity/time, 10 relationship-specificity.
- JSON adjudication records all required fields for all 100 cases; the historical benchmark remains unchanged.

## 3. Current-KB mapping

| Status | Count |
|---|---:|
| `EXACT_TEXT_LINEAGE` | 11 |
| `STRONG_TEXTUAL_EQUIVALENT` | 26 |
| `PARTIAL_TEXTUAL_MATCH` / `AMBIGUOUS` / `NO_CURRENT_MATCH` | 0 |

The 26 changed IDs are explained by current rechunking. Matching used stable source identity, document, section and normalized source text. Dense similarity was used only for search assistance, never for approval. The former broad diet chunk maps claim-specifically to current AAD Diet chunk `8050fed1-...`; the exact diet language remains present.

## 4. Gold claim adjudication

| Dimension | Result |
|---|---:|
| `DIRECT_SUPPORT` | 104 |
| Other technical statuses | 0 |
| `TECHNICALLY_CLEAR` | 104 |
| Claim-level `OWNER_REVIEW_REQUIRED` | 0 |

Every claim retains its route/population/qualifier scope. The detailed JSON has one record per claim, old IDs, proposed current IDs and the two independent adjudication dimensions.

## 5. Negative/absence adjudication

The review scanned all 536 current chunks directly using bilingual subject and requested qualifier/relation co-occurrence. It additionally inspected current native-BM25 top 20 and Gemini-Embedding-2/Qdrant-dense top 5 for every negative query. All 30 full requested absolute/exact relations had zero direct corpus matches. One shared batch embedding call produced the 30 query vectors; Qdrant remained read-only.

| Case | Label status | Subject candidates in full scan | Full requirement matches |
|---|---|---:|---:|
| GAP-ABS-001 | POLICY_AMBIGUOUS | 87 | 0 |
| GAP-ABS-002 | POLICY_AMBIGUOUS | 50 | 0 |
| GAP-ABS-003 | CONFIRMED_ABSENCE | 33 | 0 |
| GAP-ABS-004 | CONFIRMED_ABSENCE | 95 | 0 |
| GAP-ABS-005 | CONFIRMED_ABSENCE | 47 | 0 |
| GAP-ABS-006 | POLICY_AMBIGUOUS | 8 | 0 |
| GAP-ABS-007 | POLICY_AMBIGUOUS | 20 | 0 |
| GAP-ABS-008 | CONFIRMED_ABSENCE | 18 | 0 |
| GAP-ABS-009 | POLICY_AMBIGUOUS | 28 | 0 |
| GAP-ABS-010 | POLICY_AMBIGUOUS | 10 | 0 |
| GAP-EXA-001 | CONFIRMED_ABSENCE | 50 | 0 |
| GAP-EXA-002 | CONFIRMED_ABSENCE | 87 | 0 |
| GAP-EXA-003 | CONFIRMED_ABSENCE | 33 | 0 |
| GAP-EXA-004 | CONFIRMED_ABSENCE | 31 | 0 |
| GAP-EXA-005 | CONFIRMED_ABSENCE | 0 | 0 |
| GAP-EXA-006 | CONFIRMED_ABSENCE | 18 | 0 |
| GAP-EXA-007 | CONFIRMED_ABSENCE | 87 | 0 |
| GAP-EXA-008 | CONFIRMED_ABSENCE | 50 | 0 |
| GAP-EXA-009 | CONFIRMED_ABSENCE | 143 | 0 |
| GAP-EXA-010 | CONFIRMED_ABSENCE | 6 | 0 |
| GAP-REL-001 | CONFIRMED_ABSENCE | 50 | 0 |
| GAP-REL-002 | CONFIRMED_ABSENCE | 87 | 0 |
| GAP-REL-003 | CONFIRMED_ABSENCE | 33 | 0 |
| GAP-REL-004 | CONFIRMED_ABSENCE | 50 | 0 |
| GAP-REL-005 | CONFIRMED_ABSENCE | 6 | 0 |
| GAP-REL-006 | CONFIRMED_ABSENCE | 18 | 0 |
| GAP-REL-007 | CONFIRMED_ABSENCE | 0 | 0 |
| GAP-REL-008 | CONFIRMED_ABSENCE | 1 | 0 |
| GAP-REL-009 | CONFIRMED_ABSENCE | 6 | 0 |
| GAP-REL-010 | CONFIRMED_ABSENCE | 7 | 0 |

Counts: 24 `CONFIRMED_ABSENCE`, 6 `POLICY_AMBIGUOUS`, 0 `EVIDENCE_PRESENT`, 0 `PARTIAL_OR_SCOPE_DEPENDENT`, 0 `SEARCH_INCONCLUSIVE`.

## 6. Six policy cases

| Case | Interpretation A | Interpretation B | Decision |
|---|---|---|---|
| GAP-ABS-001 | strict abstention | qualified rejection accepted | owner |
| GAP-ABS-002 | strict abstention | qualified rejection accepted | owner |
| GAP-ABS-006 | strict abstention | qualified rejection accepted | owner |
| GAP-ABS-007 | strict abstention | qualified rejection accepted | owner |
| GAP-ABS-009 | strict abstention | qualified rejection accepted | owner |
| GAP-ABS-010 | strict abstention | qualified rejection accepted | owner |

For all six, the corpus does not support the absolute promise. The unresolved issue is normative: whether a bounded evidence-backed rejection counts as correct or whether only strict abstention counts. Technical recommendation: score `qualified_rejection` separately from `unsupported_generation` and `strict_abstention`; exclude these cases from gates until the benchmark owner chooses accepted outcomes.

Owner-review queue size: **6**. Exact IDs: GAP-ABS-001, GAP-ABS-002, GAP-ABS-006, GAP-ABS-007, GAP-ABS-009, GAP-ABS-010.

## 7. Frozen filtering rules and set

Filtering-rule checksum: `7ef29dd021ebd22ff1a676f5c2107592f2d63ca42a7b9a14f2075bd36bb3f3c4`.

Rules were fixed before viewing any live legacy answer. They include only technically resolved single-turn/evidence-gap labels, and exclude multi-turn, comparison/referral relationship classes, multi-claim cases, explicit presentation constraints, pregnancy/childbearing scope, and unresolved policy labels.

- Frozen benchmark subset: 34 cases = 20 positive + 14 confirmed corpus-evidence-gap negative.
- Safety controls: 20.
- Frozen-set checksum: `fb360a6c5f65d186bc1ff6db582dea585cfadc6f1f5b87496e2bbb8f96b346fe`.
- This set is valid for future retrieval/packing/citation layer work, but **not sufficient for a Step-3 routing release baseline**.

## 8. Safety precheck

Direct execution of the current deterministic `evaluate_safety` contract matched **4/20** existing emergency fixtures. Unmatched fixture IDs:

`v3_urgent_emergency_002, v3_urgent_emergency_003, v3_urgent_emergency_004, v3_urgent_emergency_006, v3_urgent_emergency_007, v3_urgent_emergency_008, v3_urgent_emergency_009, v3_urgent_emergency_010, v3_urgent_emergency_013, v3_urgent_emergency_014, v3_urgent_emergency_015, v3_urgent_emergency_016, v3_urgent_emergency_017, v3_urgent_emergency_018, v3_urgent_emergency_019, v3_urgent_emergency_020`

Matched cases used zero decision and zero generation calls in the direct probe. Unmatched cases were not sent through the graph, so no provider-call claim is made for them. Because the requested 20/20 control condition fails on current main, a zero-regression safety gate cannot be frozen as “20/20 remains valid.”

## 9. Eligibility architecture blocker

Current deterministic pre-decision state owns history count, limited request shape, safety and exact cache. It does **not** own domain/out-of-scope or semantic answerability. Static probes show an out-of-domain weather question, a product-choice question and an acne-definition question all produce the same `routine` RequestShape with no explicit constraints.

The reviewed benchmark contains `CORPUS_EVIDENCE_GAP`, but no `DOMAIN_OUT_OF_SCOPE` control. Therefore an arbitrary simple-looking no-history query could enter a decision-free route, and there is no existing deterministic signal that can conservatively send that uncertainty to legacy. Adding a keyword domain classifier is outside scope and not justified.

## 10. Baseline and gates

- Legacy live baseline: **not run**.
- Generation calls: 0.
- ActionDecision calls: 0.
- Negative-search embedding calls: 1 batch / 30 embeddings.
- Retrieval/packing/quality/citation/latency baseline: not measured.
- Release gates: not frozen.

Reason: the Step-3 eligibility contract and safety prerequisite failed before generation execution. Provider output cannot repair either prerequisite, so running the live answer baseline would spend calls without creating a valid Step-3 release package.

## 11. Final decision

**STEP3_ARCHITECTURE_BLOCKER**

Current-KB revalidation succeeded and the original 104-claim technical label blocker is resolved. Six action-policy labels remain for the owner but are excluded from the frozen subset. The blocking condition for Step 3 is architectural: no safe deterministic eligibility boundary exists before the current scope-owning ActionDecision; additionally, the current deterministic safety policy matches only 4/20 frozen emergency controls.
