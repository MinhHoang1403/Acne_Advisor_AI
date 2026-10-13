# Nghiên cứu có đối chứng: Offline Budget-Aware Context Packing

## Tóm tắt điều hành

Baseline production đã được replay chính xác trên toàn bộ 100 cases và 104 retrieval
attempts. Candidate text/payload thiếu trong frozen trace được tái dựng bằng exact-ID lookup
read-only từ đúng Qdrant build, sau đó actual `pack_context()` tái tạo 104/104 selected ID/order,
104/104 context length, 104/104 dropped/not-examined/provenance và 100/100 final context hash;
không có mismatch. Vì vậy trạng thái baseline là `EXACT_BASELINE_REPLAY_CONFIRMED`, không phải
approximation.

Treatment duy nhất `RANK_WEIGHTED_BUDGET_AWARE_WHOLE_CHUNK_SELECTION_V1` tăng packed
required-evidence coverage trên development đã lộ từ 58/88 (65,91%) lên 65/88 (73,86%):
12 requirements được thêm và 5 requirements vốn đã packed bị mất, tức +7 hit ròng và
+7,95 điểm phần trăm. Complete-case coverage tăng từ 46/70 lên 49/70. Tuy nhiên treatment
có 5 evidence regressions, không khôi phục requirement nào trong 7 item-budget losses, có
runtime cao, và repository không có independent holdout hợp lệ. Do đó đây là
`EXPERIMENT_DEVELOPMENT_COMPLETE`, đồng thời outcome là `EXPERIMENT_FAILED` theo regression
criterion và `EXPLORATORY_ONLY_NO_INDEPENDENT_VALIDATION`. Production adoption là
`NOT_AUTHORIZED`.

Nghiên cứu này không chạy generation, LLM, embedding, vector/BM25 search, BGE reranking,
RAGChecker, benchmark mới hoặc ghi Qdrant. Không có production file nào bị sửa.

## Frozen identity và nguồn dữ liệu

| Thành phần | Identity |
|---|---|
| Forensic base | `2d32317d6abd335aedd69b7b066fca53c13b0bb8` |
| Evaluated production | `366f3ad836c36be36565caad4a94ad17a7a6278f` |
| Frozen run | `thesis_evaluation_366f3ad8` |
| Pipeline fingerprint | `e4c420923b79b37c992cf36d` |
| KB build | `d4a1819fe7fb77fe1f40` |
| Raw byte SHA-256 | `b078858d61ced4ed072339fd9cb576517a2ce585724e425c6fc0a2f772f57f16` |
| Raw canonical SHA-256 | `1f525efb44091c2444d298d45de4f31637b9be4d1d512f88db38ae9738125feb` |
| Benchmark canonical SHA-256 | `f61d6807c0ce39f902936844d810562c486f1bcadaa57a8a2da0e460ad7e534b` |
| Locked protocol SHA-256 | `bc73ccc69ccf5785ec46d6c65a34d39f5626f2b1f64d962f34a552e671fd9cd2` |

Các frozen artifact cụ thể, owner commit và Git blob OID được khóa trong
`experiment_identity.json`. Chúng gồm benchmark và manifest, locked current-KB gold và
manifest, raw results, deterministic metrics, run manifest và system-output manifest.

Frozen trace có đủ ID/order/rank/score nhưng không chứa text/payload của mọi non-packed
candidate. Source A vì vậy không đủ; Source B không có exact candidate-object arrays gắn với
từng frozen attempt. Source C được dùng một lần để lấy đúng 430/430 ID bằng Qdrant point
lookup, không search. Alias `acne_knowledge` trỏ tới
`acne_knowledge__d4a1819fe7fb77fe1f40`; collection green, 536 points. Tất cả payload có đúng
build ID, chunk ID, text và provenance. Cache đầy đủ nằm dưới ignored
`data/cache/research_budget_aware_packing/`, không được commit. Manifest tracked lưu audit
cho 2.768 candidate occurrences nhưng không lưu full medical text.

## Prior selector study

Đã đọc `EVIDENCE_SELECTOR_FOCUSED_REPORT.md`, `CONTROLLED_EVIDENCE_REVIEW.md`, code và kết
quả trên các nhánh prior. Prototype cũ tách facet từ query rồi gọi thêm BGE cho từng facet;
7/8 development cases fallback, coverage không tăng, có thể chọn nhầm section `References`,
và mất trung bình khoảng 7,7 giây cho case kích hoạt. Năm regression-only cases đều tie và
không phải final holdout; kết luận cũ là `REJECT`/`NO_DEFENSIBLE_FINAL_HOLDOUT_YET`.

V1 hiện tại khác về bản chất: không facet extraction, không BGE/inference, không score mới,
không semantic heuristic. Nó giải exact 0/1 subset trên chính frozen final order bằng utility
rank cố định và exact rendered character cost. Vì vậy đây không phải việc chạy lại prototype
đã thất bại.

## Baseline replay

| Metric | Frozen reference | Replayed | Match |
|---|---:|---:|---|
| Cases | 100 | 100 | yes |
| Retrieval attempts | 104 | 104 | yes |
| Required evidence | 88 | 88 | yes |
| Acquired | 85 | 85 | yes |
| Packed | 58 | 58 | yes |
| Character losses | 20 | 20 | yes |
| Item losses | 7 | 7 | yes |
| Complete answerable cases | 46 | 46 | yes |
| Selected IDs/order | 104 | 104 | yes |
| Context lengths | 104 | 104 | yes |
| Dropped/not-examined/provenance | 104 | 104 | yes |
| Final context hashes | 100 verifiable | 100 | yes |

Mismatch count là 0. Four production blob IDs của context packer, contracts, retrieval
service và reranker đều khớp giữa evaluated production SHA và forensic base. Vì replay dùng
actual renderer/packer và tái tạo toàn bộ frozen observable output, Source C được nâng từ
reconstructed input lên exact replay-verified reconstruction.

## Protocol và selector cố định

Selector chỉ nhận normalized query identity, ordered candidates và budget. Case ID/gold chỉ
được outer evaluator join sau khi selection hoàn tất. Với vị trí one-based `r`:

```text
utility(r) = round(1_000_000 / log2(r + 1))
maximize sum(utility)
subject to selected_items <= 9 and exact_rendered_chars <= 7000
```

Solver là deterministic exact bounded DP, giữ nguyên whole chunks và original relevance
order. Cost dùng production renderer thật, gồm header, provenance scope fields, newline và
separator. Tie-break khóa theo: utility lớn hơn; tuple original positions sớm hơn theo
lexicographic order; ít rendered chars hơn; tuple stable IDs sớm hơn. Không tune threshold,
không có thuật toán thứ hai và protocol không đổi sau khi xem development result.

## Controlled development comparison

Tất cả 100 frozen cases đều đã bị historical forensic analysis phơi lộ, nên đây chỉ là
`DEVELOPMENT_RESULT`. Exact case IDs và hash nằm trong `experiment_identity.json` (các nhóm
`ANS-DEF-*`, `ANS-TRT-*`, `ANS-MEC-*`, `ANS-ADV-*`, `ANS-CMP-*`, `ANS-REF-*`, `ANS-MUL-*`,
`GAP-ABS-*`, `GAP-EXA-*`, `GAP-REL-*`).

| Metric | Baseline | Treatment | Delta |
|---|---:|---:|---:|
| Packed evidence recall | 58/88 (65,91%) | 65/88 (73,86%) | +7 hits; +7,95 pp |
| Complete answerable cases | 46/70 | 49/70 | +3 |
| Historical character losses remaining | 20 | 8 | -12 recovered |
| Historical item losses remaining | 7 | 7 | 0 recovered |
| Previously packed requirements lost | 0 | 5 | +5 regressions |
| Selected chunks, all 104 attempts | 607 | 843 | +236 |
| Context chars, all 104 attempts | 691.665 | 687.889 | -3.776 |
| Character utilization | 95,01% | 94,49% | -0,52 pp |
| Selection latency | not separately measured | p50 5404,032 ms (~5,40 s); p95 18660,599 ms (~18,66 s) | substantial |
| Mean/max selection latency | not separately measured | 7446,204 / 25176,651 ms | substantial |
| Python `tracemalloc` peak | not measured | 19.664.072 bytes | measurement-wide peak |
| Provenance/whole-chunk/budget violations | 0 | 0 | 0 |

Runtime numbers include Python `tracemalloc` instrumentation and are not deterministic output.
Compute proxy trên 104 attempts: state count p50 17.274, p95 29.031, max 30.833; transitions
p50 81.816, p95 266.197, max 355.690. Exact DP vì vậy quá đắt để coi là production-ready.

## Case-level gain và regression audit

Mỗi row dưới đây dùng candidate identity đông cứng. `Cost` là rendered cost của whole chunk
với production header (display index một chữ số); `rank` là frozen final candidate order.
Full baseline/treatment selected IDs, added/removed IDs, candidate-pool SHA-256, requirement
IDs và limitations cho từng case nằm trong `regression_audit.json`; bảng không rút gọn identity
của evidence được gain/loss.

| Case | Outcome | Required evidence gain/loss (chunk; rank; cost; source) | Context chars B→T | Added/removed chunks | Qualifier/unlabeled risk |
|---|---|---|---:|---:|---|
| ANS-DEF-003 | loss | `ac93995d-741e-53f0-8b8c-72252cf81119`; r6; 2.081; `web_dermnetnz_cddf43ad3f7f941d` | 6.754→6.545 | +2/-1 | unassessed |
| ANS-DEF-006 | gain | `9d116ea2-912f-5c45-bb76-243f6bcf6bdb`; r6; 336; `nice_ng198_2026_08` | 6.902→6.949 | +7/-2 | unassessed |
| ANS-MEC-008 | gain ×2 | `0a117c04-a7d2-5491-86af-f5d3e8f0b493`; r8; 602; `aad_acne_guideline_2024`; `e4e54d11-c8bf-56eb-8e41-b7be9ecea11e`; r14; 675; same source | 6.741→6.774 | +7/-2 | unassessed |
| ANS-ADV-002 | gain | `14857147-5eff-5e98-a6ce-c2256fdc0a05`; r10; 1.219; `nice_ng198_2026_08` | 6.766→5.795 | +2/-2 | unassessed |
| ANS-ADV-003 | gain | `14857147-5eff-5e98-a6ce-c2256fdc0a05`; r5; 1.219; `nice_ng198_2026_08` | 6.884→6.961 | +5/-2 | unassessed |
| ANS-ADV-006 | loss | `acd75b36-19cc-5a24-8630-caece16315d0`; r3; 2.326; `nice_ng198_2026_08` | 6.589→6.732 | +4/-1 | unassessed |
| ANS-CMP-002 | gain ×2 | `9d116ea2-912f-5c45-bb76-243f6bcf6bdb`; r7; 336; `nice_ng198_2026_08`; `42d9c3a7-1569-54da-93a4-fd8c048d96a1`; r8; 378; same source | 6.781→6.840 | +6/-2 | unassessed |
| ANS-CMP-003 | gain | `e4e54d11-c8bf-56eb-8e41-b7be9ecea11e`; r15; 675; `aad_acne_guideline_2024` | 6.924→6.935 | +3/-1 | unassessed |
| ANS-CMP-004 | gain | `0a117c04-a7d2-5491-86af-f5d3e8f0b493`; r13; 602; `aad_acne_guideline_2024` | 6.978→6.882 | +7/-2 | unassessed |
| ANS-REF-004 | loss | `cb9725c0-9c87-518f-a7e4-7ef767bfea4e`; r3; 1.807; `nice_ng198_2026_08` | 6.730→6.850 | +6/-2 | unassessed |
| ANS-REF-006 | loss | `cb9725c0-9c87-518f-a7e4-7ef767bfea4e`; r4; 1.807; `nice_ng198_2026_08` | 6.700→6.870 | +3/-1 | unassessed |
| ANS-MUL-001 | gain | `8f7614fe-0375-5441-9822-3773a0cc4e07`; r9; 515; `vn_moh_dermatology_4416_2023_acne` | 6.950→6.709 | +6/-2 | unassessed |
| ANS-MUL-007 | loss | `601b9c19-52c6-56cd-9965-0f3e0b1fa08e`; r6; 1.512; `aad_acne_guideline_2024` | 6.954→6.952 | +3/-1 | unassessed |
| ANS-MUL-011 | gain | `78e9df3c-560c-5c35-b149-385bdb884f11`; r7; 603; `nice_ng198_2026_08` | 6.797→6.889 | +4/-1 | unassessed |
| ANS-MUL-012 | gain | `4a9b8aee-59c0-5724-a214-d93d0d110696`; r11; 697; `nice_ng198_2026_08` | 6.604→6.619 | +4/-1 | unassessed |

Tổng cộng có 15 changed cases: 10 gain-only cases và 5 regression-only cases. Không có
case vừa gain vừa loss trong reviewed requirements. Tất cả 12 gains là historical
`CHARACTER_BUDGET_LOSS`; không có item-budget recovery. Năm loss là evidence baseline đã
packed. Mỗi changed case đều có nguy cơ treatment loại useful unlabeled support hoặc qualifier;
không có generation/clinical review mới để bác bỏ nguy cơ đó.

## Cơ chế và giới hạn diễn giải

Kết quả cho thấy small-chunk admission sau long-chunk displacement. Treatment chọn thêm 332
candidate occurrences và bỏ 103 occurrences so với baseline, tăng tổng selected chunks thêm
236 trong khi dùng ít hơn 3.776 chars. Ví dụ ANS-CMP-002 thay hai chunk tổng cost 2.832 bằng
sáu chunk tổng cost 2.883 và thu hồi hai gold chunks chỉ 336/378 chars. Chiều ngược lại,
ANS-ADV-006 bỏ một gold chunk 2.326 chars để lấy bốn chunk tổng 2.463 chars; ANS-REF-004 bỏ
gold chunk 1.807 chars trong một thay đổi +6/-2. Objective rank-weighted không biết evidence
truth, nên cả gain lẫn regression đều là hệ quả dự kiến của tối ưu utility/character.

Packed identity recall không đo context coherence, semantic sufficiency, qualifier coverage,
answer correctness hoặc clinical safety. Chọn nhiều chunk ngắn hơn có thể tăng labeled recall
nhưng giảm quality do gold không liệt kê mọi supporting evidence. Không được gắn answer cũ
với context mới để suy diễn chất lượng.

## Independent validation

Repository refs đã được audit. Không có reviewer-approved evidence-requirement holdout nằm
ngoài 100 exposed cases và có frozen replayable candidates theo cùng contract. Các artifact có
tên “holdout” trên nhánh local-RAGChecker liên quan claim-extraction contract, không phải
packing. Random split từ 100 exposed cases không tạo independent holdout. Vì vậy:

```text
INDEPENDENT_HOLDOUT_UNAVAILABLE
HOLDOUT_RESULT=NOT_RUN
EXPLORATORY_ONLY_NO_INDEPENDENT_VALIDATION
```

Không có holdout score giả được tạo.

## Verification

- Focused research tests: `40 passed`.
- Tests bao phủ 18 baseline reconstruction conditions, 20 selector conditions và một
  brute-force reference optimum; parametrization tạo 40 executed tests.
- Ruff lint: pass cho research code và hai test files.
- Ruff format check: pass.
- Python compile/import/schema path: pass.
- JSON/CSV parse: pass.
- Reproducibility: analyzer chạy lần hai từ exact ignored cache, không network; cả 9
  deterministic artifacts khớp SHA-256 và byte size. Runtime measurement bị loại khỏi hash
  vì wall-clock không deterministic.
- Git diff scope: chỉ `research/` và hai focused test files; không production diff.

## Trả lời 35 câu hỏi handoff

1. Dùng đúng tám frozen artifact được liệt kê bằng path/owner commit/blob OID trong
   `experiment_identity.json`.
2. Có; raw byte, raw canonical, benchmark canonical, Git blob và production blob hashes đều
   được xác minh lại.
3. Có; đã đọc report, controlled review, code và results của prior selector studies.
4. V1 dùng exact rank/cost DP thuần Python; prior dùng facet extraction và extra BGE scoring.
5. Có; đủ 430 unique candidates cho 2.768 occurrences/104 attempts.
6. Text đến từ exact-ID read-only payload của active Qdrant build, không từ Internet.
7. Có, vì frozen trace không giữ text/payload của mọi non-packed candidate; chỉ exact lookup,
   không retrieval search.
8. Có; build/chunk/source/section identity khớp và replay provenance 104/104.
9. Có; baseline replay đạt 100% trên 100 cases/104 attempts.
10. Có; selected IDs và order khớp 104/104.
11. Có; length khớp 104/104 và final context hash khớp 100/100.
12. Không; mismatch count bằng 0.
13. Không áp dụng vì không có mismatch.
14. Có; chỉ V1 được triển khai/chạy.
15. Utility `round(1_000_000/log2(rank+1))`; exact maximum dưới 9/7.000; tie theo utility,
    positions, chars, IDs.
16. Không; gold không thuộc selector signature/input và chỉ join sau selection.
17. Development gồm toàn bộ 100 frozen exposed cases; exact IDs/hash nằm trong identity.
18. Không; `INDEPENDENT_HOLDOUT_UNAVAILABLE`.
19. Baseline development coverage 58/88 = 65,91%.
20. Treatment development coverage 65/88 = 73,86%.
21. +12 requirements, -5 requirements, net +7.
22. Khôi phục 12/20 character-budget losses; còn 8.
23. Khôi phục 0/7 item-budget losses; còn 7.
24. Có; treatment làm mất 5 previously packed requirements.
25. Không có machine-detectable provenance/whole-chunk/budget violation; qualifier/unlabeled
    displacement vẫn là rủi ro chưa đánh giá.
26. p50/p95/mean/max là 5404,032/18660,599/7446,204/25176,651 ms với `tracemalloc`;
    peak 19.664.072 bytes; overhead đáng kể so với greedy packer.
27. Holdout `NOT_RUN` vì không có independent eligible artifact.
28. Không; 5 regressions, runtime cao và thiếu independent validation.
29. Không có production file nào thay đổi.
30. Không chạy LLM, embedding, retrieval search, reranker, generation, RAGChecker hoặc
    benchmark mới; chỉ một exact-ID reconstruction read và offline replay/analysis.
31. Focused tests 40 passed; Ruff/format/compile/parse pass.
32. Có; 9/9 deterministic outputs byte/hash-identical.
33. Branch `research/budget-aware-packing-controlled`, base `2d32317...`; final research
    commit được ghi ở Git handoff sau khi report được tạo.
34. Không push/merge/PR và không đổi KB/alias/points; các stash/nhánh cũ được giữ nguyên.
35. `EXACT_BASELINE_REPLAY_CONFIRMED`; `EXPERIMENT_FAILED` trên exposed development;
    `EXPLORATORY_ONLY_NO_INDEPENDENT_VALIDATION`; `PRODUCTION_CHANGE_NOT_AUTHORIZED`.

## Quyết định cuối

```text
RESEARCH_TASK:
Offline Budget-Aware Context Packing

BASELINE_REPLAY_STATUS:
EXACT_BASELINE_REPLAY_CONFIRMED

EXPERIMENT_STATUS:
EXPERIMENT_FAILED (EXPERIMENT_DEVELOPMENT_COMPLETE; +12/-5 requirements)

INDEPENDENT_VALIDATION_STATUS:
EXPLORATORY_ONLY_NO_INDEPENDENT_VALIDATION

BEST_SUPPORTED_CONCLUSION:
V1 bảo toàn thêm 7 required-evidence identities ròng trên 100 frozen exposed cases với cùng
9-item/7.000-character budget, nhưng làm mất 5 evidence identities vốn đã packed, tốn runtime
đáng kể và chưa có bằng chứng generalization hoặc answer/clinical quality.

PRODUCTION_ADOPTION:
NOT_AUTHORIZED

NEXT_ACTION:
Giữ production nguyên trạng. Nếu tiếp tục nghiên cứu, tạo reviewer-approved independent
holdout có frozen replayable candidates; preregister validation riêng. Bất kỳ loss guard hoặc
selector redesign nào phải là thí nghiệm mới, không sửa ngược protocol V1 này.
```
