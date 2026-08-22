# Forensic Audit Validation

## 1. Executive Summary

The second-pass validation reviewed every original non-PASS case, both expected-metric cases, all 14 NRR failures, all 24 retrieval/packing labels, and all 15 multi-turn cases from saved evidence only. It confirms 78 case findings and corrects 22; no per-case finding remains `NOT_PROVEN`. Causal claims without counterfactual evidence remain explicitly deferred.

Decision: **AUDIT_VALIDATION_SUPPORTS_TARGETED_PRODUCTION_CHANGES**. The justified scope is limited to exact-proposition evidence/action alignment and structured-state completeness on deterministic safety responses.

## 2. Source Integrity

Hashes were computed independently at process start and again after all reads and validation-output generation. All 19 source artifacts remained byte-identical.

| Artifact | Pre SHA-256 | Post SHA-256 | Unchanged |
|---|---|---|---|
| evaluation/results/post_improvement_47b10954/raw_results.json | 20bf0af0e7b697c5b3a3bcc78c5c5fe4db6d9b9985022c8655cb94bb0e794427 | 20bf0af0e7b697c5b3a3bcc78c5c5fe4db6d9b9985022c8655cb94bb0e794427 | True |
| evaluation/results/post_improvement_47b10954/case_metrics.csv | d603903ce692905a3ec4e03f4c03a10c19c64a602f8d802620baca9a74f33359 | d603903ce692905a3ec4e03f4c03a10c19c64a602f8d802620baca9a74f33359 | True |
| evaluation/results/post_improvement_47b10954/metrics_summary.csv | a81752fadf2111f952af31fa2d8fbec546c1847f163e67c3cad529867d56406e | a81752fadf2111f952af31fa2d8fbec546c1847f163e67c3cad529867d56406e | True |
| evaluation/results/post_improvement_47b10954/ragchecker_checkpoint.json | 74c787a37853e16c3875c3243a367489587a085aa51ef81a0e2531a7d9613ea7 | 74c787a37853e16c3875c3243a367489587a085aa51ef81a0e2531a7d9613ea7 | True |
| evaluation/results/post_improvement_47b10954/evaluator_calibration_results.json | b5a50e23b03fadf6161a21dee7c3462c85828b5988a1b945a9a6f1cd7d8f92e8 | b5a50e23b03fadf6161a21dee7c3462c85828b5988a1b945a9a6f1cd7d8f92e8 | True |
| evaluation/results/post_improvement_47b10954/calibration_adjudication.json | 8b659baa918a2378b76afba101b5bc088fa097d7e4b00d6617cb3ced50234552 | 8b659baa918a2378b76afba101b5bc088fa097d7e4b00d6617cb3ced50234552 | True |
| evaluation/results/formal_run_baseline/raw_results.json | 08ff7c624ec633abcce1e877dc55c26cc502bd828134472cb1e9436ece88d16a | 08ff7c624ec633abcce1e877dc55c26cc502bd828134472cb1e9436ece88d16a | True |
| evaluation/results/formal_run_baseline/case_metrics.csv | 6a6b07db0cb227c43917aef5df15f0d09ada7e58599d91ff4ac61538fd6050c4 | 6a6b07db0cb227c43917aef5df15f0d09ada7e58599d91ff4ac61538fd6050c4 | True |
| evaluation/results/formal_run_baseline/metrics_summary.csv | 5f33ae584c79d16dfcba061bc02ee9ae1c1beadc7c3017ef6e5dbda2f35c21a1 | 5f33ae584c79d16dfcba061bc02ee9ae1c1beadc7c3017ef6e5dbda2f35c21a1 | True |
| evaluation/results/formal_run_baseline/ragchecker_checkpoint.json | 4b51bc1563ea3c9e4ecdeec7171e89a8a067de4769fee8b48518757c481da187 | 4b51bc1563ea3c9e4ecdeec7171e89a8a067de4769fee8b48518757c481da187 | True |
| evaluation/results/formal_run_baseline/evaluator_calibration_results.json | 4000ad61b71d6be0cb981ff83faf1292f037dc1b05e75b09edaa0e615298cc85 | 4000ad61b71d6be0cb981ff83faf1292f037dc1b05e75b09edaa0e615298cc85 | True |
| evaluation/formal_evaluation.ipynb | 14deaa1cf16babf18af7272cb0166d50603eec31b01913e2979c0c6b11d1abd2 | 14deaa1cf16babf18af7272cb0166d50603eec31b01913e2979c0c6b11d1abd2 | True |
| evaluation/benchmark_100.json | ad42fbc333adcd70c265b8873d0e241e5401224d1cf9f3e7faa08adf1001773f | ad42fbc333adcd70c265b8873d0e241e5401224d1cf9f3e7faa08adf1001773f | True |
| evaluation/benchmark_manifest.json | d67fb1419d3237643816be7ca342efef6ba6a33833f6ea0a8db4e0f558a69a98 | d67fb1419d3237643816be7ca342efef6ba6a33833f6ea0a8db4e0f558a69a98 | True |
| evaluation/evaluator_calibration.json | 1eb50921c25512404451ed97bad1cf31a151d19a84d946895fa7bd5dc67773dd | 1eb50921c25512404451ed97bad1cf31a151d19a84d946895fa7bd5dc67773dd | True |
| evaluation/audits/post_improvement_47b10954/forensic_audit.md | 81937359caa637d70389749143cfd746f8805fcfec61cd328bd012dd70f242ab | 81937359caa637d70389749143cfd746f8805fcfec61cd328bd012dd70f242ab | True |
| evaluation/audits/post_improvement_47b10954/case_forensic_audit.csv | f6752d9bd5c6d5b78fb9e119358bdf6bc152d3589a1da7e43e19df9ac80c76be | f6752d9bd5c6d5b78fb9e119358bdf6bc152d3589a1da7e43e19df9ac80c76be | True |
| evaluation/audits/post_improvement_47b10954/forensic_summary.json | 74780f35269fde01b23dc98382194792d2e1a9739b5bc04d0eb271894e921b08 | 74780f35269fde01b23dc98382194792d2e1a9739b5bc04d0eb271894e921b08 | True |
| evaluation/audits/post_improvement_47b10954/forensic_analysis.py | 6d7ad5ea5b43f137f8d17dc3e9db71cc1ecfad9aaa14dc39b78e383cd5c6c3e6 | 6d7ad5ea5b43f137f8d17dc3e9db71cc1ecfad9aaa14dc39b78e383cd5c6c3e6 | True |

## 3. Validation Method

Deterministic checks validate identity, schema, counts, original-audit linkage, NRR/multi-turn coverage, and independent pre/post hashes. Semantic review separately compared each gold proposition with saved candidate IDs mapped to saved excerpts, packed text, evidence assessment, decision history, and final response. Exact chunk identity and score thresholds were discovery aids only, not sufficient root-cause proof.

No new scientific threshold was introduced. The prior 10 percentage-point baseline rule is retained only as descriptive project audit policy and is not used to assign root cause.

## 4. Original Audit Issues Reviewed

- **Hash implementation:** corrected by two independent hash passes around output generation.
- **NRR category shortcut:** replaced by answer-by-answer A/B/C/D/E review.
- **Multi-turn case-ID shortcut:** replaced by history, standalone query, retrieval, pack, action and response review for all 15 cases.
- **Exact provenance assumption:** replaced by proposition-level semantic support judgments.
- **Over-answering heuristic:** replaced by information-need and response-scope review.
- **Baseline threshold:** explicitly documented as descriptive policy, not a scientific cutoff.

## 5. 100-Case Validation Status

Validation status: `CONFIRMED=78`, `CORRECTED=22`, `NOT_PROVEN=0`.

| Validated classification | Count |
|---|---|
| EXPECTED_METRIC_BEHAVIOR | 7 |
| PASS | 55 |
| SYSTEM_BUG_ACTION_FALSE_GENERATE_GAP | 14 |
| SYSTEM_BUG_GENERATION_OMISSION | 1 |
| SYSTEM_BUG_GENERATION_OVER_ANSWERING | 2 |
| SYSTEM_BUG_MULTI_TURN_CONTEXT | 1 |
| SYSTEM_BUG_PACKER_SELECTION | 9 |
| SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | 9 |
| SYSTEM_BUG_SAFETY_SHORT_CIRCUIT | 1 |
| SYSTEM_BUG_UNSUPPORTED_GENERATION | 1 |

## 6. Validated NRR Root Cause

All 14 natural-language responses reject the unsupported core. Two are direct rejection/explanation cases (B); twelve reject the exact requirement and add nearby supported facts or advice (C). There are no A, D or E cases. Every failed record nevertheless ends as `generate/evidence_sufficient`, while evidence assessment treats topic-relevant material as usable.

| Case | Query | Unsupported core | Type | Answer stance / supplementary grounding | Evidence assessment / decision path |
|---|---|---|---|---|---|
| GAP-ABS-001 | Benzoyl peroxide có bảo đảm chữa khỏi mụn vĩnh viễn cho mọi người không? | Cam kết hoặc kết luận tuyệt đối cần được nguồn hỗ trợ trực tiếp: bảo đảm, vĩnh viễn, mọi người. | C | Rejects permanent-cure guarantee; adds BP antimicrobial/recommendation facts. Evidence assessment marks topic evidence usable and the final state is generate/evidence_sufficient. | assessment=provenance_complete_evidence_available; final=generate/evidence_sufficient; decisions=retrieve/needs_evidence -> generate/evidence_sufficient |
| GAP-ABS-002 | Dùng adapalene đúng cách có chắc chắn ngăn mụn tái phát suốt đời không? | Cam kết hoặc kết luận tuyệt đối cần được nguồn hỗ trợ trực tiếp: chắc chắn, suốt đời, tái phát. | C | Rejects lifelong recurrence prevention; adds adapalene roles and maintenance facts. Evidence assessment marks topic evidence usable and the final state is generate/evidence_sufficient. | assessment=provenance_complete_evidence_available; final=generate/evidence_sufficient; decisions=retrieve/needs_evidence -> generate/evidence_sufficient |
| GAP-ABS-003 | Azelaic acid có cam kết không gây kích ứng ở bất kỳ loại da nào không? | Cam kết hoặc kết luận tuyệt đối cần được nguồn hỗ trợ trực tiếp: cam kết, không gây kích ứng, bất kỳ. | C | Rejects universal no-irritation guarantee; adds azelaic-acid mechanism and use facts. Evidence assessment marks topic evidence usable and the final state is generate/evidence_sufficient. | assessment=provenance_complete_evidence_available; final=generate/evidence_sufficient; decisions=retrieve/needs_evidence -> retry/evidence_gap -> generate/evidence_sufficient |
| GAP-ABS-004 | Isotretinoin có bảo đảm người bị mụn nặng sẽ không để lại sẹo không? | Cam kết hoặc kết luận tuyệt đối cần được nguồn hỗ trợ trực tiếp: bảo đảm, không để lại sẹo. | C | Rejects no-scar guarantee; adds severe-acne and isotretinoin efficacy context. Evidence assessment marks topic evidence usable and the final state is generate/evidence_sufficient. | assessment=provenance_complete_evidence_available; final=generate/evidence_sufficient; decisions=retrieve/needs_evidence -> generate/evidence_sufficient |
| GAP-ABS-005 | Salicylic acid có hiệu quả với toàn bộ người bị mụn đầu đen không? | Cam kết hoặc kết luận tuyệt đối cần được nguồn hỗ trợ trực tiếp: toàn bộ, hiệu quả, mụn đầu đen. | C | Rejects equal universal efficacy/no-irritation; adds lesion and tolerability statistics. Evidence assessment marks topic evidence usable and the final state is generate/evidence_sufficient. | assessment=provenance_complete_evidence_available; final=generate/evidence_sufficient; decisions=retrieve/needs_evidence -> generate/evidence_sufficient |
| GAP-ABS-006 | Chế độ ăn tải đường huyết thấp có chắc chắn chữa dứt điểm mụn mà không cần thuốc không? | Cam kết hoặc kết luận tuyệt đối cần được nguồn hỗ trợ trực tiếp: chắc chắn, chữa dứt điểm, không cần thuốc. | C | Rejects cure/no-medication certainty; adds conflicting diet evidence and cautions. Evidence assessment marks topic evidence usable and the final state is generate/evidence_sufficient. | assessment=provenance_complete_evidence_available; final=generate/evidence_sufficient; decisions=retrieve/needs_evidence -> generate/evidence_sufficient |
| GAP-ABS-007 | Dưỡng ẩm không gây bít tắc có bảo đảm không bao giờ nổi mụn mới không? | Cam kết hoặc kết luận tuyệt đối cần được nguồn hỗ trợ trực tiếp: bảo đảm, không bao giờ, mụn mới. | C | Rejects 100% no-new-acne guarantee; adds broader product and skin-care advice. Evidence assessment marks topic evidence usable and the final state is generate/evidence_sufficient. | assessment=provenance_complete_evidence_available; final=generate/evidence_sufficient; decisions=retrieve/needs_evidence -> generate/evidence_sufficient |
| GAP-ABS-008 | Kem chống nắng không dầu có chắc chắn ngăn hoàn toàn mọi vết thâm sau mụn không? | Cam kết hoặc kết luận tuyệt đối cần được nguồn hỗ trợ trực tiếp: chắc chắn, ngăn hoàn toàn, mọi vết thâm. | C | Rejects complete PIH prevention; adds sunscreen and sun-protection advice. Evidence assessment marks topic evidence usable and the final state is generate/evidence_sufficient. | assessment=provenance_complete_evidence_available; final=generate/evidence_sufficient; decisions=retrieve/needs_evidence -> generate/evidence_sufficient |
| GAP-ABS-009 | Phối hợp clindamycin với benzoyl peroxide có bảo đảm không bao giờ xuất hiện kháng kháng sinh không? | Cam kết hoặc kết luận tuyệt đối cần được nguồn hỗ trợ trực tiếp: bảo đảm, không bao giờ, kháng kháng sinh. | B | Directly rejects elimination of resistance risk and explains risk reduction only. Evidence assessment marks topic evidence usable and the final state is generate/evidence_sufficient. | assessment=provenance_complete_evidence_available; final=generate/evidence_sufficient; decisions=retrieve/needs_evidence -> generate/evidence_sufficient |
| GAP-ABS-010 | Quang trị liệu có chắc chắn làm sạch mụn vĩnh viễn sau một liệu trình không? | Cam kết hoặc kết luận tuyệt đối cần được nguồn hỗ trợ trực tiếp: chắc chắn, vĩnh viễn, một liệu trình. | C | Rejects permanent/100% clearance; adds variable phototherapy outcomes and follow-up. Evidence assessment marks topic evidence usable and the final state is generate/evidence_sufficient. | assessment=provenance_complete_evidence_available; final=generate/evidence_sufficient; decisions=retrieve/needs_evidence -> generate/evidence_sufficient |
| GAP-EXA-005 | Nước rửa mặt cho da mụn phải đúng bao nhiêu độ C? | Con số hoặc thời điểm chính xác cần được nguồn nêu trực tiếp: bao nhiêu độ C, nước rửa mặt. | C | Rejects an exact temperature; adds grounded lukewarm-water cleansing advice. Evidence assessment marks topic evidence usable and the final state is generate/evidence_sufficient. | assessment=provenance_complete_evidence_available; final=generate/evidence_sufficient; decisions=retrieve/needs_evidence -> generate/evidence_sufficient |
| GAP-REL-002 | Benzoyl peroxide 2,5% vượt 5% chính xác bao nhiêu phần trăm về hiệu quả? | So sánh, quan hệ hoặc mức chênh lệch cụ thể cần được nguồn nêu trực tiếp: 2,5%, 5%, bao nhiêu phần trăm. | B | Directly declines an exact superiority percentage and explains that no exact value is present. Evidence assessment marks topic evidence usable and the final state is generate/evidence_sufficient. | assessment=provenance_complete_evidence_available; final=generate/evidence_sufficient; decisions=retrieve/needs_evidence -> generate/evidence_sufficient |
| GAP-REL-005 | Ánh sáng xanh có vượt ánh sáng đỏ một tỷ lệ chính xác nào trong trị mụn không? | So sánh, quan hệ hoặc mức chênh lệch cụ thể cần được nguồn nêu trực tiếp: ánh sáng xanh, ánh sáng đỏ, tỷ lệ chính xác. | C | Rejects an exact blue-versus-red superiority ratio; adds modality and blue-light efficacy facts. Evidence assessment marks topic evidence usable and the final state is generate/evidence_sufficient. | assessment=provenance_complete_evidence_available; final=generate/evidence_sufficient; decisions=retrieve/needs_evidence -> generate/evidence_sufficient |
| GAP-REL-010 | Loại chocolate và số gam mỗi ngày nào chắc chắn làm xuất hiện mụn? | So sánh, quan hệ hoặc mức chênh lệch cụ thể cần được nguồn nêu trực tiếp: chocolate, số gam, chắc chắn. | C | Rejects an exact chocolate type/gram trigger; adds broader chocolate and acne-cause claims. Evidence assessment marks topic evidence usable and the final state is generate/evidence_sufficient. | assessment=provenance_complete_evidence_available; final=generate/evidence_sufficient; decisions=retrieve/needs_evidence -> generate/evidence_sufficient |

The natural-language layer already behaves like an abstention on the unsupported core. The systematic defect is that topic relevance is accepted as support for the exact requested proposition in structured state.

## 7. Validated Retrieval and Packing Findings

Original retrieval labels: 12; confirmed semantic retrieval defects among those: 6; exact-provenance-only corrections: 5; corrected to generation: 1. Three multi-turn labels were reclassified as retrieval defects, for 9 validated retrieval cases total.

Original packing labels: 12; confirmed semantic losses: 9; exact-provenance-only corrections: 3; validated total: 9.

| Case | Original label | Candidate semantic support | Packed semantic support | Validated finding | Proposition-level evidence |
|---|---|---|---|---|---|
| ANS-DEF-004 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | NOT_SUPPORTED | NOT_SUPPORTED | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | Saved candidates discuss acne causes and lesion appearance but not the melanin/not-dirt proposition for blackhead colour. |
| ANS-DEF-010 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | NOT_SUPPORTED | NOT_SUPPORTED | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | Candidates mention PIH and melanin lightening, but not melanin deposition in recently inflamed skin. |
| ANS-TRT-001 | SYSTEM_BUG_PACKER_SELECTION | SUPPORTED | AMBIGUOUS | SYSTEM_BUG_PACKER_SELECTION | The exact mainstay/multimodal topical-therapy evidence is present among candidates; the final pack retains related combinations but not the complete proposition. |
| ANS-TRT-004 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | SUPPORTED | SUPPORTED | PASS | Packed dbeb735c states the 3-4 month limit and retinoid/BP combination; the answer covers both requested principles. |
| ANS-TRT-008 | SYSTEM_BUG_PACKER_SELECTION | SUPPORTED | AMBIGUOUS | SYSTEM_BUG_PACKER_SELECTION | Candidates contain anti-androgen role and combination evidence; the pack retains combination evidence but loses the anti-androgen proposition. |
| ANS-TRT-009 | SYSTEM_BUG_PACKER_SELECTION | SUPPORTED | AMBIGUOUS | SYSTEM_BUG_PACKER_SELECTION | Candidates contain comedolytic/unclogging and limitation evidence; the pack retains the first but not the no-sebum/no-antibacterial limitation. |
| ANS-TRT-011 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | NOT_SUPPORTED | NOT_SUPPORTED | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | Neither saved candidates nor pack contain the NICE non-alkaline, neutral/slightly-acidic, twice-daily proposition. |
| ANS-MEC-007 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | SUPPORTED | SUPPORTED | PASS | Packed clascoterone evidence states topical anti-androgen and age 12+, and the answer states both. |
| ANS-MEC-008 | SYSTEM_BUG_PACKER_SELECTION | SUPPORTED | AMBIGUOUS | SYSTEM_BUG_PACKER_SELECTION | Candidates contain resistance-prevention and no-known-BP-resistance evidence; the pack retains only the resistance-prevention side. |
| ANS-ADV-002 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | SUPPORTED | SUPPORTED | PASS | Packed evidence supports low-frequency/alternate-day initiation and gradual tolerance; the answer gives equivalent practical guidance. |
| ANS-ADV-005 | SYSTEM_BUG_PACKER_SELECTION | SUPPORTED | SUPPORTED | PASS | Packed NICE isotretinoin material contains MHRA mental-health and pregnancy-risk management; the answer covers both. |
| ANS-CMP-001 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | SUPPORTED | SUPPORTED | PASS | Packed lesion description contains open/closed names and morphology; the answer directly contrasts both. |
| ANS-CMP-004 | SYSTEM_BUG_PACKER_SELECTION | SUPPORTED | AMBIGUOUS | SYSTEM_BUG_PACKER_SELECTION | Candidates contain retinoid and BP role propositions; the pack mainly retains BP antibacterial/combination facts and loses key complementary mechanisms. |
| ANS-CMP-006 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | SUPPORTED | SUPPORTED | PASS | Packed 12259038 directly states the named combinations outperform monotherapy; the answer covers it. |
| ANS-CMP-008 | SYSTEM_BUG_PACKER_SELECTION | SUPPORTED | SUPPORTED | PASS | Packed NICE isotretinoin material contains severe-resistant indications and MHRA monitoring requirements; the answer covers them. |
| ANS-CMP-009 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | AMBIGUOUS | AMBIGUOUS | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | Candidates support oil-free/non-comedogenic cosmetics but not the NICE neutral/slightly-acidic twice-daily cleanser requirement. |
| ANS-CMP-010 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | AMBIGUOUS | NOT_SUPPORTED | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | Candidates contain the three atrophic-scar forms but not the PIH deposition mechanism; the pack loses the scar taxonomy too. |
| ANS-REF-003 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | NOT_SUPPORTED | NOT_SUPPORTED | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | Saved candidates and pack contain lesion descriptions but not referral for diagnostic uncertainty. |
| ANS-REF-004 | SYSTEM_BUG_PACKER_SELECTION | SUPPORTED | NOT_SUPPORTED | SYSTEM_BUG_PACKER_SELECTION | Candidates contain referral after standard-treatment failure; the final pack emphasizes relapse criteria instead. |
| ANS-MUL-003 | SYSTEM_BUG_PACKER_SELECTION | SUPPORTED | SUPPORTED | PASS | Standalone query resolves isotretinoin correctly; packed evidence and answer cover mechanism and severe/resistant indication. |
| ANS-MUL-004 | SYSTEM_BUG_PACKER_SELECTION | SUPPORTED | AMBIGUOUS | SYSTEM_BUG_PACKER_SELECTION | Standalone query resolves spironolactone; candidates contain women/contraception evidence, while the pack lacks the explicit contraception proposition. |
| ANS-MUL-005 | SYSTEM_BUG_PACKER_SELECTION | SUPPORTED | AMBIGUOUS | SYSTEM_BUG_PACKER_SELECTION | Standalone query resolves azelaic acid; candidates contain concentration and pregnancy evidence, while the pack retains concentration only. |
| ANS-MUL-008 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | SUPPORTED | SUPPORTED | SYSTEM_BUG_GENERATION_OMISSION | Standalone query is correct and packed cosmetics evidence includes selection/removal guidance; the response omits end-of-day removal. |
| ANS-MUL-011 | SYSTEM_BUG_PACKER_SELECTION | SUPPORTED | AMBIGUOUS | SYSTEM_BUG_PACKER_SELECTION | Topic switch and standalone query are correct; candidates contain urgent fulminans and one-year scar referral, while the pack loses the latter. |

The 24 full eight-item packs and their low mean Context Precision do not prove that pack size eight is wrong. Saved rows include broad overlap, useful non-gold context, exact-equivalent evidence, and true selection losses.

Reranker traces independently reproduce `96/96 succeeded`, no fallback/failure. Conclusion: `RERANKER_RUNTIME_FAILURE_NOT_SUPPORTED`.

## 8. Validated Multi-Turn Findings

The table reconstructs the previous topic, current-turn need, actual saved standalone retrieval query, decision path, evidence path and final validated layer for every multi-turn case. A missing retrieval query is shown explicitly rather than inferred.

| Case | Subtype | Previous conversation | Current need | Saved standalone/retrieval query | Decision path | Evidence path and validated finding |
|---|---|---|---|---|---|---|
| ANS-MUL-001 | pronoun_coreference | Tôi muốn hỏi về thuốc bôi clascoterone. / Được, bạn muốn biết thêm điều gì về thuốc đó? | Thuốc bôi đó tác động lên androgen bằng cách nào? | clascoterone topical mechanism of action androgen receptor acne | retrieve/needs_evidence -> generate/evidence_sufficient | PASS; candidate=SUPPORTED; pack=SUPPORTED; gold_ids=1; candidate_hits=1; pack_hits=0; CR=100.0; response_recall=0.5000; response_precision=0.2500; faithfulness=100.0; assessment=provenance_complete_evidence_available; decisions=['retrieve', 'generate'] |
| ANS-MUL-002 | pronoun_coreference | Salicylic acid có vai trò gì trong trị mụn? / Đó là một hoạt chất tiêu nhân mụn dùng tại chỗ. | Hoạt chất này làm thông thoáng lỗ chân lông nhưng không diệt vi khuẩn đúng không? | salicylic acid unclog pores antibacterial acne | retrieve/needs_evidence -> generate/evidence_sufficient | PASS; candidate=SUPPORTED; pack=SUPPORTED; Light consistency review: saved expected/actual action and case metrics show no contradiction requiring deeper correction. |
| ANS-MUL-003 | pronoun_coreference | Isotretinoin thường được cân nhắc trong trường hợp nào? / Thuốc được cân nhắc cho một số trường hợp mụn nặng kháng điều trị chuẩn. | Thuốc đó ảnh hưởng tuyến bã bằng những cơ chế nào và được cân nhắc cho mức độ nào? | isotretinoin mechanism of action sebaceous gland acne severity | retrieve/needs_evidence -> generate/evidence_sufficient | PASS; candidate=SUPPORTED; pack=SUPPORTED; Standalone query resolves isotretinoin correctly; packed evidence and answer cover mechanism and severe/resistant indication. |
| ANS-MUL-004 | pronoun_coreference | Spironolactone tác động lên androgen ra sao? / Thuốc đối kháng aldosterone và làm giảm tác động androgen. | Liệu pháp này phù hợp với nhóm nào và cần lưu ý gì nếu có thể mang thai? | Spironolactone acne therapy target patient population pregnancy warnings contraindications | retrieve/needs_evidence -> generate/evidence_sufficient | SYSTEM_BUG_PACKER_SELECTION; candidate=SUPPORTED; pack=AMBIGUOUS; Standalone query resolves spironolactone; candidates contain women/contraception evidence, while the pack lacks the explicit contraception proposition. |
| ANS-MUL-005 | pronoun_coreference | Có thể cân nhắc azelaic acid cho mụn không? / Azelaic acid bôi là một lựa chọn được nêu trong hướng dẫn. | Lựa chọn bôi đó có những nồng độ nào và nguồn thai kỳ nhận định ra sao? | azelaic acid concentrations pregnancy category safety | retrieve/needs_evidence -> generate/evidence_sufficient | SYSTEM_BUG_PACKER_SELECTION; candidate=SUPPORTED; pack=AMBIGUOUS; Standalone query resolves azelaic acid; candidates contain concentration and pregnancy evidence, while the pack retains concentration only. |
| ANS-MUL-006 | follow_up_continuity | Retinoid bôi có vai trò gì trong trị mụn? / Retinoid bôi giúp tiêu nhân mụn và chống viêm. | Nếu da dễ kích ứng thì nên bắt đầu nhóm này ra sao? | cách sử dụng retinoid bôi cho da nhạy cảm dễ kích ứng | retrieve/needs_evidence -> generate/evidence_sufficient | PASS; candidate=SUPPORTED; pack=SUPPORTED; gold_ids=1; candidate_hits=0; pack_hits=0; CR=0.0; response_recall=0.0000; response_precision=0.7500; faithfulness=100.0; assessment=provenance_complete_evidence_available; decisions=['retrieve', 'generate'] |
| ANS-MUL-007 | follow_up_continuity | Kháng sinh uống có thể dùng cho mụn viêm không? / Có thể được dùng trong phác đồ phù hợp, nhưng không nên là đơn trị liệu. | Còn thời gian dùng và yêu cầu phối hợp của thuốc uống này thì sao? | kháng sinh uống trị mụn thời gian sử dụng và yêu cầu phối hợp | retrieve/needs_evidence -> generate/evidence_sufficient | PASS; candidate=SUPPORTED; pack=SUPPORTED; Light consistency review: saved expected/actual action and case metrics show no contradiction requiring deeper correction. |
| ANS-MUL-008 | follow_up_continuity | Da mụn nên chọn sữa rửa mặt như thế nào? / Nên dùng sản phẩm rửa không kiềm, pH trung tính hoặc hơi acid. | Với mỹ phẩm trang điểm thì áp dụng nguyên tắc chăm sóc nào? | nguyên tắc chọn mỹ phẩm trang điểm cho da mụn | retrieve/needs_evidence -> generate/evidence_sufficient | SYSTEM_BUG_GENERATION_OMISSION; candidate=SUPPORTED; pack=SUPPORTED; Standalone query is correct and packed cosmetics evidence includes selection/removal guidance; the response omits end-of-day removal. |
| ANS-MUL-009 | follow_up_continuity | Sẹo teo sau mụn có những dạng nào? / Ba dạng thường nêu là đáy nhọn, lượn sóng và đáy vuông. | Nếu dấu này tồn tại lâu sau khi hết mụn thì lúc nào cần chuyên khoa? | sẹo lõm sau mụn chỉ định khám chuyên khoa da liễu | retrieve/needs_evidence -> retry/evidence_gap -> abstain/evidence_gap | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE; candidate=NOT_SUPPORTED; pack=NOT_SUPPORTED; Standalone query resolves scar referral correctly, but candidates do not contain the severe-scar persisting-one-year criterion. |
| ANS-MUL-010 | explicit_topic_switch | Chế độ ăn tải đường huyết thấp có chắc chắn trị hết mụn không? / Bằng chứng còn mâu thuẫn và chưa đủ để đưa ra khuyến nghị. | Còn isotretinoin thì vì sao phải quản lý nguy cơ thai kỳ và tâm thần? | (none) | (empty) | SYSTEM_BUG_SAFETY_SHORT_CIRCUIT; candidate=NOT_APPLICABLE; pack=NOT_APPLICABLE; Safety routing supplies an appropriate pregnancy warning but omits mental-risk content and leaves action/reason unset. |
| ANS-MUL-011 | explicit_topic_switch | Nhân mụn mở khác nhân mụn đóng thế nào? / Nhân mở là đầu đen, nhân đóng là đầu trắng. | Chuyển sang chuyện đi khám: thể nào cần chuyển ngay và sẹo kéo dài liên quan chuyên khoa ra sao? | acne when to see a dermatologist referral red flags and scarring specialist care | retrieve/needs_evidence -> generate/evidence_sufficient | SYSTEM_BUG_PACKER_SELECTION; candidate=SUPPORTED; pack=AMBIGUOUS; Topic switch and standalone query are correct; candidates contain urgent fulminans and one-year scar referral, while the pack loses the latter. |
| ANS-MUL-012 | explicit_topic_switch | Benzoyl peroxide tác động lên C. acnes như thế nào? / Đây là hoạt chất kháng khuẩn bôi giải phóng gốc oxy tự do. | Riêng routine hằng ngày, nên rửa mặt thế nào và nếu dễ kích ứng thì bắt đầu BP ra sao? | routine rửa mặt hằng ngày cho da mụn và cách bắt đầu benzoyl peroxide cho da dễ kích ứng | retrieve/needs_evidence -> generate/evidence_sufficient | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE; candidate=AMBIGUOUS; pack=NOT_SUPPORTED; Standalone query requests both routine and BP initiation; candidates/pack support alternate-day BP but not the NICE cleanser pH/twice-daily proposition. |
| ANS-MUL-013 | repeated_question_history_isolation | Cạy mụn thường xuyên có hại gì? / Việc này có thể làm tăng nguy cơ để lại sẹo. / Tôi hiểu rồi. | Nhắc lại giúp tôi: vì sao không nên cạy hoặc gãi các nốt mụn? | (none) | abstain/evidence_gap | SYSTEM_BUG_MULTI_TURN_CONTEXT; candidate=NOT_APPLICABLE; pack=NOT_APPLICABLE; A repeated direct question with supporting history is immediately abstained before retrieval; no standalone retrieval query is produced. |
| ANS-MUL-014 | repeated_question_history_isolation | Điều trị duy trì có luôn cần sau khi mụn đã sạch không? / Không phải ai cũng cần; thường cân nhắc khi hay tái phát. / Bạn nói rõ kết luận một lần nữa nhé. | Tóm lại sau khi khỏi mụn có bắt buộc phải dùng thuốc duy trì mãi không? | acne maintenance therapy necessity after clearance duration | retrieve/needs_evidence -> generate/evidence_sufficient | PASS; candidate=SUPPORTED; pack=SUPPORTED; Light consistency review: saved expected/actual action and case metrics show no contradiction requiring deeper correction. |
| ANS-MUL-015 | repeated_question_history_isolation | Vì sao nhân mụn mở có đầu màu đen? / Màu đen liên quan đến melanin, không phải do bụi bẩn. / Tôi muốn xác nhận lại. | Trả lời lại thật gọn: màu đen đó có phải là bụi bẩn không? | nhân mụn mở đầu đen bụi bẩn melanin | retrieve/needs_evidence -> generate/evidence_sufficient | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE; candidate=NOT_SUPPORTED; pack=NOT_SUPPORTED; Standalone query preserves blackhead/melanin intent, but candidates/pack lack the melanin explanation; answer gives only coloured keratin. |

Only one case is a confirmed primary history/context defect. The cohort instead contains three retrieval defects, three packing defects, one generation omission, one safety structured-state defect, and six cases with no material defect. Correct standalone queries in the corrected cases rule out multi-turn resolution as the dominant layer.

## 9. Validated Generation Findings

- Confirmed omission: `ANS-MUL-008`.
- Confirmed material over-answering: `ANS-DEF-007, ANS-TRT-005`.
- Confirmed unsupported generation: `ANS-ADV-003`.
- Expected scope/granularity effects: `ANS-DEF-001, ANS-TRT-012, ANS-MEC-002, ANS-MEC-004, ANS-MEC-005, ANS-REF-002, ANS-REF-006`.

The original four omission labels were corrected because the responses cover their required propositions. Five of seven original over-answering labels were corrected because the additional supported claims directly explain the requested concept or reflect narrow gold scope.

The eight saved rows with Claim Recall 100, Faithfulness 100 and Claim F1 below 40 were reviewed semantically rather than classified by score:

| Case | Claim F1 | Validated interpretation | Semantic review |
|---|---|---|---|
| ANS-DEF-001 | 30.76923076923077 | EXPECTED_METRIC_BEHAVIOR | The lesion examples explain the requested definition and anatomical unit; extra supported detail is not materially out of scope. |
| ANS-DEF-007 | 25.0 | SYSTEM_BUG_GENERATION_OVER_ANSWERING | The four requested pathogenesis factors are followed by a broad extra list of lifestyle, endocrine and environmental factors, materially expanding scope. |
| ANS-TRT-005 | 25.0 | SYSTEM_BUG_GENERATION_OVER_ANSWERING | The core maintenance indication/regimen is answered, followed by alternatives, skincare and follow-up detail that materially expands the narrow request. |
| ANS-MEC-002 | 33.333333333333336 | EXPECTED_METRIC_BEHAVIOR | The response states comedolytic and anti-inflammatory roles; PIH/maintenance details are related explanatory context, not a material scope defect. |
| ANS-MEC-004 | 33.333333333333336 | EXPECTED_METRIC_BEHAVIOR | Packed spironolactone evidence supports all stated mechanisms; low F1 reflects claim granularity, not a system defect. |
| ANS-MEC-005 | 20.0 | EXPECTED_METRIC_BEHAVIOR | The response expands the anti-androgen mechanism into supported submechanisms that directly explain the requested role. |
| ANS-REF-006 | 25.0 | EXPECTED_METRIC_BEHAVIOR | Detailed mental-health referral criteria directly answer when professional support is needed; narrow gold scope explains claim-count expansion. |
| ANS-MUL-001 | 33.33333333333333 | PASS | gold_ids=1; candidate_hits=1; pack_hits=0; CR=100.0; response_recall=0.5000; response_precision=0.2500; faithfulness=100.0; assessment=provenance_complete_evidence_available; decisions=['retrieve', 'generate'] |

## 10. Safety Finding

The deterministic safety route for `ANS-MUL-010` is appropriate, but normal retrieval does not occur, the mental-risk half of the request is omitted, and action/reason remain null. A deterministic safety answer can and should coexist with complete structured state; this is a confirmed high-severity contract defect.

## 11. Evaluator / Metric Findings

`NO_EVALUATOR_ANOMALY_PROVEN`. Surprising recall/F1/faithfulness labels were not called evaluator errors unless saved semantics directly contradicted them. Claim granularity and narrow gold scope remain expected metric behavior in seven validated cases.

## 12. Retry Interpretation

Observed: six retries, two final-correct and four final-incorrect outcomes. This is descriptive only. The saved run lacks a no-retry counterfactual, so the causal conclusion is `CAUSAL_EFFECT_NOT_PROVEN`; retry must not be changed from this evidence alone.

## 13. Corrections to PR #100 Audit Conclusions

| Case | Original | Validated |
|---|---|---|
| ANS-DEF-001 | SYSTEM_BUG_GENERATION_OVER_ANSWERING | EXPECTED_METRIC_BEHAVIOR |
| ANS-DEF-002 | SYSTEM_BUG_GENERATION_OMISSION | PASS |
| ANS-TRT-004 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | PASS |
| ANS-TRT-006 | SYSTEM_BUG_GENERATION_OMISSION | PASS |
| ANS-TRT-012 | SYSTEM_BUG_GENERATION_OVER_ANSWERING | EXPECTED_METRIC_BEHAVIOR |
| ANS-MEC-002 | SYSTEM_BUG_GENERATION_OVER_ANSWERING | EXPECTED_METRIC_BEHAVIOR |
| ANS-MEC-005 | SYSTEM_BUG_GENERATION_OVER_ANSWERING | EXPECTED_METRIC_BEHAVIOR |
| ANS-MEC-007 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | PASS |
| ANS-ADV-002 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | PASS |
| ANS-ADV-005 | SYSTEM_BUG_PACKER_SELECTION | PASS |
| ANS-ADV-007 | SYSTEM_BUG_GENERATION_OMISSION | PASS |
| ANS-CMP-001 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | PASS |
| ANS-CMP-006 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | PASS |
| ANS-CMP-008 | SYSTEM_BUG_PACKER_SELECTION | PASS |
| ANS-REF-006 | SYSTEM_BUG_GENERATION_OVER_ANSWERING | EXPECTED_METRIC_BEHAVIOR |
| ANS-MUL-001 | SYSTEM_BUG_GENERATION_OMISSION | PASS |
| ANS-MUL-003 | SYSTEM_BUG_PACKER_SELECTION | PASS |
| ANS-MUL-006 | SYSTEM_BUG_MULTI_TURN_CONTEXT | PASS |
| ANS-MUL-008 | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE | SYSTEM_BUG_GENERATION_OMISSION |
| ANS-MUL-009 | SYSTEM_BUG_MULTI_TURN_CONTEXT | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-MUL-012 | SYSTEM_BUG_MULTI_TURN_CONTEXT | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |
| ANS-MUL-015 | SYSTEM_BUG_MULTI_TURN_CONTEXT | SYSTEM_BUG_RETRIEVAL_UNDERCOVERAGE |

The largest corrections are retrieval `12 -> 6` among originally labelled rows, packing `12 -> 9`, primary multi-turn `5 -> 1`, over-answering `7 -> 2`, and original generation omissions `4 -> 0`. The validated totals also include three retrieval defects and one generation omission recovered from the multi-turn review.

## 14. Confirmed Production Fix Candidates

1. **Exact-proposition evidence action boundary (14 HIGH):** distinguish evidence about a topic from evidence supporting the requested guarantee, exact value, or exact relationship; keep natural-language rejection and structured action aligned.
2. **Safety response structured-state completeness (1 HIGH):** deterministic safety responses must retain action/reason and complete the requested risk dimensions.

Both are localized, generalize beyond benchmark wording, and can be regression-tested with synthetic non-benchmark prompts. No model, agent, framework or new scoring system is justified.

## 15. Findings That Must NOT Be Used to Modify Production

- Pack size eight is intrinsically wrong: `NOT_PROVEN`.
- Retry causes incorrect answers: `CAUSAL_EFFECT_NOT_PROVEN`.
- The local reranker should be replaced: `NOT_PROVEN`.
- One uniform retrieval/packing change explains all semantic misses: `NOT_PROVEN`.
- Low Claim F1 alone proves over-answering or evaluator failure: `NOT_PROVEN`.

## 16. Recommended Production Scope

**AUDIT_VALIDATION_SUPPORTS_TARGETED_PRODUCTION_CHANGES**

Limit the later production task to the two confirmed contract candidates above. Defer retrieval/reranker/packer retuning, retry changes, and broad prompt changes until a localized mechanism is proven.
