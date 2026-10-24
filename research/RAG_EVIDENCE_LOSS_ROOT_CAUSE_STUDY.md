# RAG Evidence Loss Root-Cause Study

## 1. Kết luận điều hành

Phạm vi nghiên cứu là 100 frozen outputs được tạo bởi production SHA
`366f3ad836c36be36565caad4a94ad17a7a6278f`; SHA cleanup hiện tại
`6eca9b58df3bb5a277612ab40917a90aac9f0595` chỉ là nền Git để chứa nghiên cứu. Hai SHA
không được dùng thay thế cho nhau.

Kết quả cuối:

- `PARTIAL_ROOT_CAUSE_CONFIRMED`
- `NO_CONFIRMED_IMPLEMENTATION_BUG`

Các kết luận chính:

1. [OBSERVED] Đơn vị metric gốc là một cặp duy nhất
   `case_id + required_chunk_id`; claim IDs chỉ gắn kèm và không làm tăng mẫu số. Có đúng
   88 requirements trên 70 answerable cases.
2. [OBSERVED] Acquisition tìm thấy `85/88`; final generation context chứa `58/88`.
   Packed set là tập con của acquisition union. Vì vậy có 30 requirements không nằm trong
   final pack, gồm ba acquisition misses và đúng 27 post-acquisition losses.
3. [OBSERVED] 27 post-acquisition losses được reconcile không trùng primary category:
   20 `CHARACTER_BUDGET_LOSS` trên 16 cases và bảy `ITEM_BUDGET_LOSS` trên sáu cases.
   Không có primary loss do RRF/fusion, identity/provenance, contract rejection, retry state
   hoặc trace không xác định.
4. [OBSERVED] Nguyên nhân có ảnh hưởng lớn nhất là whole-chunk character admission:
   20/27, tương đương 74,1% post-acquisition losses. Trace ghi trực tiếp
   `drop_reason=character_limit`.
5. [DETERMINISTICALLY_REPRODUCED] Cả bảy item losses đứng sau prefix chín item theo thứ
   tự rerank và được ghi `not_examined_reason=item_limit_reached`. Tất cả bảy requirements
   đều bị đẩy xuống so với RRF rank; đây là contributing factor của rerank order, không phải
   bằng chứng rằng BGE hiểu sai ngữ nghĩa.
6. [OBSERVED] Không có retry-state loss. Hai required IDs thuộc answerable retry cases vẫn
   nằm trong retained/eligible union ở attempt 2; chúng mất tại final packing vì item limit.
7. [SUPPORTED_INFERENCE] Decision và generation dùng cùng `PackedContext` theo frozen code,
   nhưng serializer không lưu `agent_decision_evidence_traces`, nên không thể gọi equality
   này là OBSERVED từ run trace.
8. [OBSERVED] Sáu confirmed-absence cases bị metric action-contract đánh trượt là
   `GAP-ABS-003`, `GAP-ABS-004`, `GAP-EXA-001`, `GAP-EXA-005`, `GAP-REL-002` và
   `GAP-REL-010`; cả sáu có final action `generate/evidence_sufficient`.
9. [SUPPORTED_INFERENCE] Sáu trường hợp trên thuộc nhóm
   `G1_INSUFFICIENT_EVIDENCE_MISJUDGED`. Direct-support IDs và policy-validation state không
   được lưu, nên không thể xác nhận sâu hơn cơ chế quyết định. Nhiều final answers vẫn nói
   rõ giới hạn, do đó không được biến action-contract failure thành kết luận “câu trả lời sai
   y khoa”.
10. Không có production/KB change, model inference, API call, live Qdrant query, benchmark
    rerun hoặc gold mutation trong nghiên cứu này.

## 2. Danh tính và tính toàn vẹn dữ liệu

### 2.1 Nguồn đã dùng

Tất cả artifact được đọc từ local branch `evaluation/thesis-final-evaluation` tại HEAD
`5a19e3c079ef82006fbb42cba44e92560544b7da`; không checkout hoặc merge branch đó.

| Artifact | Path | Commit sở hữu nội dung |
|---|---|---|
| Benchmark | `evaluation/benchmark_100.json` | `bd3db1a87fc1e6491c11cab983dfa0b62112db00` |
| Benchmark manifest | `evaluation/benchmark_manifest.json` | `88b28c7e74ce1f110339ccd75e526bf96046ffcc` |
| Locked gold | `evaluation/locked_gold/current_kb_adjudication.json` | `afc60883e9060c1dd041b2105c3e9a69a8313bf1` |
| Gold manifest | `evaluation/locked_gold/current_kb_adjudication_manifest.json` | `afc60883e9060c1dd041b2105c3e9a69a8313bf1` |
| Frozen raw results | `evaluation/results/thesis_evaluation_366f3ad8/raw_results.json` | `c679994efc4463e96f7154dfb80296378c32ff02` |
| System-output manifest | `evaluation/results/thesis_evaluation_366f3ad8/system_output_manifest.json` | `c679994efc4463e96f7154dfb80296378c32ff02` |
| Run manifest | `evaluation/results/thesis_evaluation_366f3ad8/run_manifest.json` | `5a19e3c079ef82006fbb42cba44e92560544b7da` |
| Reference deterministic metrics | `evaluation/results/thesis_evaluation_366f3ad8/deterministic_metrics.json` | `5a19e3c079ef82006fbb42cba44e92560544b7da` |

`formal_run_2d5f0124` trên `main` không được dùng.

### 2.2 Hash và run identity

| Trường | Giá trị quan sát | Kết quả |
|---|---|---|
| Production SHA | `366f3ad836c36be36565caad4a94ad17a7a6278f` | khớp raw + run manifest |
| Run ID | `thesis_evaluation_366f3ad8` | khớp |
| Raw canonical SHA-256 | `1f525efb44091c2444d298d45de4f31637b9be4d1d512f88db38ae9738125feb` | khớp expected |
| Raw byte SHA-256 | `b078858d61ced4ed072339fd9cb576517a2ce585724e425c6fc0a2f772f57f16` | khớp Git blob + manifest |
| Benchmark canonical SHA-256 | `f61d6807c0ce39f902936844d810562c486f1bcadaa57a8a2da0e460ad7e534b` | khớp raw + manifests |
| Gold canonical SHA-256 | `e0597523954311d954266adae59f4bb86530e6bce9b5ca6b59b0148ef7d3801e` | khớp raw + run manifest |
| Gold raw SHA-256 | `68e025affb608d220eebd5988e76d19a1baebc2f60d35a986a87b2d93db6a02e` | khớp manifest |
| KB build | `d4a1819fe7fb77fe1f40` | khớp raw + run + locked gold |
| Pipeline fingerprint | `e4c420923b79b37c992cf36d` | khớp raw + 100/100 records |
| Case count/order | 100 unique, đúng benchmark order | khớp |

Git lưu raw results bằng LF và đúng byte hash `b078...`. `git archive` trên máy Windows
chuyển nó sang CRLF, tạo hash `41d176...`; analyzer vì vậy dùng exact Git blob cho raw
results. Ngược lại, gold raw hash trong run manifest được khóa từ biểu diễn CRLF; bản CRLF
đó được dùng để xác minh raw hash, trong khi canonical hash vẫn giống Git blob. Không file
nguồn nào bị ghi đè.

`benchmark_manifest.json` ghi build lịch sử `94d613bc9b33628de3ef`; locked gold ghi rõ đây
là `old_benchmark_kb_build_id` và map sang current build `d4a181...`. Vì benchmark content
hash, current-gold hash và frozen-run current KB đều khớp, đây là lineage metadata đã được
reconcile, không phải trộn run.

Run có 100 locked system outputs. RAGChecker semantic evaluator bị chặn sau khi output đã
đóng băng; nghiên cứu này chỉ dùng deterministic traces và không dùng partial semantic
evaluation.

## 3. Audit implementation tạo frozen outputs

`git diff` trên 13 file được chỉ định giữa `366f3ad...` và `6eca9b5...` không có thay đổi.
Do đó line references dưới đây áp dụng đồng thời cho frozen SHA và cleanup main.

| Thành phần | Code location | Contract xác minh | Quan sát từ frozen run |
|---|---|---|---|
| Dense/BM25 acquisition | `src/retrieval/service.py:128-153` | hai channel độc lập, mỗi channel tối đa 16 candidates | 104 attempts có bounded channel traces |
| RRF | `src/retrieval/rrf.py:20-55` | tính `sum(weight/(k+rank))`, dedup bằng `str(document["id"])` | `rank`/`rrf_rank` lưu trong fused trace |
| Candidate conversion | `src/retrieval/service.py:423-441` | `id` → `chunk_id` → synthetic fallback; text giữ nguyên | không có required loss do conversion |
| Retry merge | `src/retrieval/service.py:444-556` | dedup stable candidate ID, giữ `seen_in_attempts` và `rank_by_attempt` | retained IDs còn trong final union |
| Reranker input | `src/retrieval/service.py:187-201`; `src/retrieval/reranker.py:181-183,286-341` | merged union + effective rerank query; fallback giữ deterministic input order | 102 succeeded events, hai fallback events |
| Packer | `src/retrieval/context_packer.py:19-119` | scan supplied order, whole-chunk admission, max 9 items/7.000 chars | 20 `character_limit`, bảy `item_limit_reached` |
| Header/cost | `src/retrieval/context_packer.py:146-180` | rendered header và delimiter đều tính vào character budget | exact nonpacked block cost không được trace lưu |
| Retrieval workflow | `src/agent/nodes/workflow.py:168-307` | tối đa hai attempts, truyền old candidates sang retry | bốn cases có hai attempts |
| Attempt trace | `src/agent/nodes/workflow.py:310-361` | lưu query, channel ranks, union, pack outcome | đủ phân loại stage; thiếu candidate text nonpacked |
| Action policy | `src/agent/action_policy.py:83-160` | generate cần direct-support IDs thuộc visible evidence; retry bounded | frozen serializer không lưu IDs/policy-validation |
| Decision evidence | `src/agent/action_decision.py:276-311` | normal runtime dùng toàn bộ generator `PackedContext` | trực tiếp trace bị serializer bỏ |
| Generation evidence | `src/agent/nodes/reason.py:70-95`; `src/agent/prompts/medical_answer.py:67-104` | prompt nhận nguyên `packed_context_text` | final generation contexts + hashes được lưu |
| Structural verifier | `src/quality/answer_verifier.py:26-72` | presentation/provenance only, `medical_semantic_verification=False` | không được dùng pass này như semantic truth |

Các câu hỏi code-path được trả lời cụ thể:

- Dense và BM25 đều lấy tối đa 16 candidates. Candidate có thể không vào RRF vì nằm ngoài
  top 16 hoặc thiếu `id`; không có pre-rerank relevance cutoff khác trong path này.
- RRF dedup theo stable point/candidate ID. Trace lưu RRF position trong cả `rank` và
  `rrf_rank`; `rank` của candidate vẫn là pre-rerank RRF rank sau khi rerank.
- Reranker nhận toàn bộ acquired/retained merged union và `effective_rerank_query`.
  Workflow đặt acquisition query và rerank query cùng giá trị; 104/104 frozen attempts xác
  nhận hai trường này bằng nhau.
- Hai fallback BGE là `ANS-DEF-001` và `ANS-DEF-002`, lý do `timeout`. Không post-acquisition
  required loss nào liên quan fallback. Required ID của `ANS-DEF-002` đã vắng ở cả hai
  channel trước reranking.
- Packer không truncate chunk. Nó tính full rendered block, gồm header và separator; chunk
  không vừa bị ghi `dropped/character_limit` rồi scan tiếp. Khi đã chọn đủ chín item, phần
  còn lại được ghi `not_examined/item_limit_reached`.
- Retry giữ candidates cũ, merge/dedup với acquisition mới rồi rerank/repack. Rank thay đổi
  sau retry không tự động là state-loss bug.
- Normal runtime đưa cùng packed evidence vào action decision và generation. Tuy nhiên exact
  decision trace không nằm trong artifact, nên kết luận run-level chỉ là SUPPORTED_INFERENCE.
- Structural provenance đạt 581/581 và citation identity 267/267 trong reference metrics.
  Không có candidate đủ điều kiện bị mất do serialization, provenance hoặc unexpected dedup.

## 4. Evidence accounting

### 4.1 Định nghĩa denominator

Implementation metric gốc trong `evaluation/thesis_evaluation_support.py` tạo ordered unique
list từ mọi `current_evidence_ids` trong một case rồi kiểm membership trong acquisition union
và generation contexts. Vì vậy identity đúng là:

```text
case_id + required_chunk_id
```

Một chunk dùng trong hai cases là hai requirements. Một chunk hỗ trợ nhiều claims trong cùng
case chỉ là một requirement và giữ danh sách claim IDs. Không tự gộp alternative IDs; gold
contract hiện tại diễn giải mỗi `current_evidence_id` là bắt buộc.

### 4.2 Reconciliation

| Outcome/stage | Requirements | Cases | Trace đủ | Unknown |
|---|---:|---:|---:|---:|
| Packed | 58 | 46 complete cases; có thể xuất hiện ở các incomplete cases | 58 | 0 |
| Acquisition miss | 3 | 3 | 3 cho bounded top-16 outcome | 0 primary; nguyên nhân ngoài top-16 unknown |
| Fusion/identity | 0 | 0 | 0 | 0 |
| Character budget | 20 | 16 | 20 drop reasons | exact arithmetic replay thiếu text/cost |
| Item budget | 7 | 6 | 7 | 0 |
| Contract/provenance | 0 | 0 | 0 | 0 |
| Retry state | 0 | 0 | 0 | 0 |
| Gold mismatch | 0 | 0 | 0 | 0 |
| Trace-insufficient primary | 0 | 0 | 0 | 0 |

Phép bảo toàn:

```text
88 = 58 packed + 3 acquisition miss + 20 character loss + 7 item loss
85 acquired = 58 packed + 20 character loss + 7 item loss
27 post-acquisition losses = 20 + 7
```

30 nonpacked requirements ảnh hưởng 24 cases. Riêng 27 post-acquisition losses ảnh hưởng
22 cases.

## 5. Per-requirement root causes

Full 88-row journey, queries và trace references nằm trong
`research/results/rag_evidence_journey.csv`. Mỗi requirement chỉ có một primary category.

### 5.1 Acquisition misses

| Case | Claim | Required chunk | Attempt | Dense/BM25 | Primary evidence |
|---|---|---|---:|---|---|
| `ANS-DEF-002` | `C01` | `904d057a-aa4d-5d6a-bcb7-7379c5137b60` | 1 | absent/absent | cả hai channel trả đủ 16, không channel error |
| `ANS-ADV-009` | `C01` | `4a9b8aee-59c0-5724-a214-d93d0d110696` | 1 | absent/absent | cả hai channel trả đủ 16, không channel error |
| `ANS-CMP-005` | `C01` | `3725e30a-9a79-5f53-87b9-8fb9165b9451` | 1 | absent/absent | cả hai channel trả đủ 16, không channel error |

[UNKNOWN] Trace không chứa rank ngoài top 16, nên không thể tách “candidate limit” khỏi
ranking/corpus miss. Không truy vấn Qdrant hoặc tăng top-k để bổ sung dữ liệu.

### 5.2 Character-budget losses

`Final chars` là total cuối của packed context, không phải used chars tại đúng thời điểm
candidate được xét. Trace không lưu candidate text/cost cho nonpacked items nên các trường
đó để `null` trong CSV.

| Case | Claim | Required chunk | Attempt | RRF → rerank | Final chars | Trace reason |
|---|---|---|---:|---:|---:|---|
| `ANS-DEF-006` | `C01` | `9d116ea2-912f-5c45-bb76-243f6bcf6bdb` | 1 | 4 → 6 | 6902 | `character_limit` |
| `ANS-TRT-003` | `C01` | `3725e30a-9a79-5f53-87b9-8fb9165b9451` | 1 | 18 → 21 | 6986 | `character_limit` |
| `ANS-TRT-009` | `C01,C02` | `6d6de22f-f537-5336-9812-1133ba9fbc7b` | 1 | 3 → 5 | 6970 | `character_limit` |
| `ANS-MEC-008` | `C01` | `e4e54d11-c8bf-56eb-8e41-b7be9ecea11e` | 1 | 5 → 14 | 6741 | `character_limit` |
| `ANS-MEC-008` | `C02` | `0a117c04-a7d2-5491-86af-f5d3e8f0b493` | 1 | 10 → 8 | 6741 | `character_limit` |
| `ANS-ADV-002` | `C01` | `14857147-5eff-5e98-a6ce-c2256fdc0a05` | 1 | 19 → 10 | 6766 | `character_limit` |
| `ANS-ADV-003` | `C01` | `14857147-5eff-5e98-a6ce-c2256fdc0a05` | 1 | 5 → 5 | 6884 | `character_limit` |
| `ANS-ADV-005` | `C02` | `41a81bea-8def-5d49-b03e-6cc8a0d3db10` | 1 | 13 → 8 | 6998 | `character_limit` |
| `ANS-CMP-002` | `C01` | `42d9c3a7-1569-54da-93a4-fd8c048d96a1` | 1 | 6 → 8 | 6781 | `character_limit` |
| `ANS-CMP-002` | `C02` | `9d116ea2-912f-5c45-bb76-243f6bcf6bdb` | 1 | 7 → 7 | 6781 | `character_limit` |
| `ANS-CMP-003` | `C03` | `e4e54d11-c8bf-56eb-8e41-b7be9ecea11e` | 1 | 7 → 15 | 6924 | `character_limit` |
| `ANS-CMP-004` | `C01` | `624729be-fe2e-53e0-9738-134a08f481dc` | 1 | 11 → 21 | 6978 | `character_limit` |
| `ANS-CMP-004` | `C02,C03` | `0a117c04-a7d2-5491-86af-f5d3e8f0b493` | 1 | 19 → 13 | 6978 | `character_limit` |
| `ANS-CMP-008` | `C03` | `41a81bea-8def-5d49-b03e-6cc8a0d3db10` | 1 | 13 → 7 | 6919 | `character_limit` |
| `ANS-MUL-001` | `C01` | `8f7614fe-0375-5441-9822-3773a0cc4e07` | 1 | 5 → 9 | 6950 | `character_limit` |
| `ANS-MUL-003` | `C02` | `ec49f82f-1b66-57ff-973a-caaea5959cf1` | 1 | 29 → 9 | 6950 | `character_limit` |
| `ANS-MUL-010` | `C01` | `ec49f82f-1b66-57ff-973a-caaea5959cf1` | 1 | 20 → 14 | 6829 | `character_limit` |
| `ANS-MUL-011` | `C01` | `78e9df3c-560c-5c35-b149-385bdb884f11` | 1 | 11 → 7 | 6797 | `character_limit` |
| `ANS-MUL-012` | `C01` | `4a9b8aee-59c0-5724-a214-d93d0d110696` | 1 | 1 → 11 | 6604 | `character_limit` |
| `ANS-MUL-012` | `C02` | `14857147-5eff-5e98-a6ce-c2256fdc0a05` | 1 | 25 → 18 | 6604 | `character_limit` |

Primary reason là observed packer outcome tại `context_packer.py:68-76`. Rerank demotion
được ghi là contributing factor khi có, nhưng không đủ counterfactual để nói BGE gây ra
toàn bộ 20 character losses.

### 5.3 Item-budget losses và liên hệ reranking

| Case | Claim | Required chunk | Attempt(s) | RRF rank(s) → rerank rank(s) | Trace reason |
|---|---|---|---|---|---|
| `ANS-DEF-009` | `C01` | `c5a99f3c-abf4-5ed3-be4a-6acdf0b82d3a` | 1 | 9 → 16 | `item_limit_reached` |
| `ANS-DEF-010` | `C01` | `df9c7438-a085-5063-9824-a96830e2224a` | 1,2 | 8,8 → 21,27 | `item_limit_reached` |
| `ANS-TRT-001` | `C01,C02` | `06a0e33d-b9f6-53c0-b5ed-a163ec1e0d5b` | 1,2 | 9,9 → 23,27 | `item_limit_reached` |
| `ANS-TRT-008` | `C02` | `43191f74-eff1-5540-bd20-b512c055fccd` | 1 | 7 → 20 | `item_limit_reached` |
| `ANS-CMP-005` | `C02` | `601b9c19-52c6-56cd-9965-0f3e0b1fa08e` | 1 | 6 → 20 | `item_limit_reached` |
| `ANS-CMP-010` | `C01` | `df9c7438-a085-5063-9824-a96830e2224a` | 1 | 13 → 18 | `item_limit_reached` |
| `ANS-CMP-010` | `C02` | `c5a99f3c-abf4-5ed3-be4a-6acdf0b82d3a` | 1 | 9 → 19 | `item_limit_reached` |

[OBSERVED] Cả bảy IDs có rerank rank thấp hơn RRF rank. [DETERMINISTICALLY_REPRODUCED]
Packer đã đủ chín selected items trước khi xét chúng. Đây là bằng chứng mạnh cho quan hệ
`rerank order → position after packed prefix → item_limit_reached`; không chứng minh lỗi
semantic bên trong BGE.

## 6. Retry, identity và provenance

Frozen run có bốn retry cases: `ANS-DEF-010`, `ANS-TRT-001`, `GAP-EXA-002` và
`GAP-REL-002`. Chỉ hai answerable cases có required-evidence rows.

- `ANS-DEF-010` required ID được acquired ở attempt 1 và 2, retained ở attempt 2, vẫn
  eligible; final rerank rank 27 và `item_limit_reached`.
- `ANS-TRT-001` có cùng pattern, final rerank rank 27 và `item_limit_reached`.
- Không required evidence nào được packed ở attempt 1 rồi biến mất khỏi state.
- Không required evidence nào biến mất khỏi eligible union do retry merge/dedup.

Vì vậy retry có thể thay đổi final rank/pack outcome, nhưng `RETRY_STATE_LOSS=0`.

Không có `FUSION_OR_IDENTITY_LOSS`, unexpected duplicate rejection hoặc provenance
rejection. RRF và retry dedup đều dùng stable candidate ID; generation context giữ đúng
chunk/source identity. Structural provenance metrics là kỹ thuật, không phải semantic
entailment.

## 7. Evidence-gap failures

30 negative cases gồm 24 `CONFIRMED_ABSENCE` và sáu `POLICY_AMBIGUOUS`. Ambiguous cases
không được tính pass/fail. Kết quả confirmed rejection tái hiện đúng `18/24`.

| Case | Category | Attempts | Actual | Packed items | Generation | Primary diagnosis |
|---|---|---:|---|---:|---|---|
| `GAP-ABS-003` | absolute certainty | 1 | `generate/evidence_sufficient` | 5 | Gemini 3.1 fallback model | G1, SUPPORTED_INFERENCE |
| `GAP-ABS-004` | absolute certainty | 1 | `generate/evidence_sufficient` | 5 | Gemini 3.5 | G1, SUPPORTED_INFERENCE |
| `GAP-EXA-001` | exact quantity/time | 1 | `generate/evidence_sufficient` | 3 | Gemini 3.5 | G1, SUPPORTED_INFERENCE |
| `GAP-EXA-005` | exact quantity/time | 1 | `generate/evidence_sufficient` | 9 | Gemini 3.1 fallback model | G1, SUPPORTED_INFERENCE |
| `GAP-REL-002` | relationship specificity | 2 | `generate/evidence_sufficient` | 4 | Gemini 3.1 fallback model | G1, SUPPORTED_INFERENCE |
| `GAP-REL-010` | relationship specificity | 1 | `generate/evidence_sufficient` | 4 | Gemini 3.5 | G1, SUPPORTED_INFERENCE |

“Fallback model” trong bảng là generation-provider model selection, không phải
`fallback_applied=true`; persisted records ghi `fallback_applied=false`.

[OBSERVED] Gold xác nhận core factual requirement không có trong corpus, nhưng action cuối
là `generate/evidence_sufficient`. [SUPPORTED_INFERENCE] Semantic decision đã xem related
evidence là đủ. Không thể nâng lên OBSERVED vì frozen serializer bỏ:

- `agent_decision_history`;
- `agent_decision_evidence_traces`;
- `direct_supporting_evidence_ids`;
- `policy_validation`;
- `missing_evidence` do model nêu.

Sáu responses đều pass structural verifier, nhưng verifier khai báo
`medical_semantic_verification=False`. Một số response vẫn nói rõ không có con số/cam kết
chính xác. Do đó kết luận chắc chắn chỉ là action-contract failure, không phải sáu medical
answer failures. Không có G3 retrieval-context deficiency, G4 policy-code gap hoặc G5
response-contract failure nào được xác nhận từ trace hiện có.

`generation_invoked=true` ở 100/100 records, kể cả abstention path; nó không được dùng làm
failure criterion.

## 8. Bug, design limitation và evaluation limitation

### 8.1 Không có implementation bug được xác nhận

Không invariant nào bị vi phạm trong frozen data:

- acquisition union khớp union của per-attempt acquired IDs;
- packed IDs là subset của acquisition;
- RRF/fused/retry identities không mất;
- `dropped` và `not_examined` giải thích đủ 27 post-acquisition losses;
- retry retained union được bảo toàn;
- provenance/citation identity hợp lệ;
- frozen code và cleanup main giống nhau ở toàn bộ 13 file audit.

Vì vậy không có production patch trong task này.

### 8.2 Design limitations được quan sát

1. Whole-chunk, rank-ordered packing bỏ nguyên chunk khi không vừa 7.000 ký tự. Đây là
   20/27 post-acquisition losses.
2. Packer dừng sau chín selected items. Rerank order đặt bảy required IDs sau prefix đó.
3. ActionDecision có thể trả `evidence_sufficient` cho confirmed-absence query khi related
   evidence tồn tại; action policy chỉ kiểm tra ID thuộc visible set, không tự đánh giá
   entailment.

### 8.3 Data/evaluation limitations

- Ba acquisition misses không có rank ngoài top 16.
- Nonpacked candidate text, rendered cost, used/remaining chars tại inspection không được
  lưu; character drop reason quan sát được nhưng exact pure-function replay không thể làm.
- Decision evidence traces/direct-support IDs không được lưu.
- Negative output semantics chưa được independent reviewer chấm; structural verifier không
  phải semantic verifier.
- Sáu policy-ambiguous cases vẫn cần owner review và không được đưa vào failure count.

Không tính oracle upper bound vì artifact thiếu candidate text/rendered costs cho toàn bộ
nonpacked pool. Tạo lại chúng bằng Qdrant hoặc model inference bị cấm và sẽ phá frozen-data
boundary.

## 9. Root-cause ranking

| Rank | Finding | Requirements/cases | Confidence | Khả năng xử lý | Regression risk |
|---:|---|---:|---|---|---|
| 1 | Whole-chunk character budget | 20 / 16 | OBSERVED stage | trung bình | qualifier/provenance displacement |
| 2 | Item boundary sau rerank order | 7 / 6 | OBSERVED + deterministic link | trung bình | useful unlabeled evidence displacement |
| 3 | Evidence-sufficiency decision | 6 confirmed negative cases | SUPPORTED_INFERENCE | trung bình | over-abstention |
| 4 | Acquisition miss trong bounded top 16 | 3 / 3 | OBSERVED outcome, UNKNOWN deeper cause | chưa đủ căn cứ | tăng noise/latency nếu tăng top-k |

Nếu chỉ sửa một điểm để nghiên cứu trước, bằng chứng mạnh nhất hướng tới **budget-aware
whole-chunk selection mà không tăng budget**: nó nhắm đúng 20/27 tổn thất lớn nhất. Đây mới
là giả thuyết thí nghiệm, không phải production recommendation đã được chứng minh.

## 10. Follow-up experiments

Ba thí nghiệm tối đa được preregister trong
`research/results/rag_followup_experiments.md`:

1. Budget-aware whole-chunk selection, ưu tiên cao nhất.
2. Rerank-order diagnosis tại item boundary.
3. Evidence-sufficiency decision calibration.

Không chọn tăng top-k hoặc thay model trước: acquisition đã đạt `85/88`, còn 27 losses xảy
ra sau acquisition. Trace chứng minh pack/order outcome nhưng chưa chứng minh model khác là
biện pháp cần thiết.

## 11. Analyzer, kiểm thử và khả năng tái lập

Analyzer: `research/rag_evidence_loss_forensics.py`.

Source ownership metadata truyền cho analyzer được khóa tại
`research/rag_evidence_loss_source_inventory.json`.

Thuộc tính:

- chỉ đọc frozen JSON;
- schema/hash/identity fail-closed;
- không import production retrieval/model code;
- không gọi model, provider, database hoặc network;
- không hardcode 27 loss IDs hoặc sáu failure IDs;
- output không chứa current timestamp và deterministic trên cùng input.

Focused test: `tests/test_rag_evidence_loss_forensics.py`, gồm đúng 20 fixtures cho Dense,
BM25, rerank demotion, character/item limit, duplicate/missing identity, retry, shared chunk,
multiple evidence IDs, missing/invalid trace, pending/ambiguous gold, abstain with generation,
hash mismatch, baseline/KB mismatch và repeated deterministic execution.

Các artifact machine-readable:

- `research/results/rag_evidence_loss_identity.json`
- `research/results/rag_evidence_journey.csv`
- `research/results/rag_evidence_loss_by_case.json`
- `research/results/rag_evidence_gap_failures.json`
- `research/results/rag_root_cause_summary.json`
- `research/results/rag_missing_trace_fields.json`
- `research/results/rag_followup_experiments.md`

## 12. Trạng thái cuối

- Frozen data identity: `VERIFIED`.
- Metrics `85/88`, `58/88`, `18/24`: tái hiện đúng.
- Evidence-loss stages: reconcile đầy đủ ở primary-category level.
- Decision internals của sáu negative failures: thiếu trace, chỉ đạt SUPPORTED_INFERENCE.
- Production/KB modification: không.
- Push/PR/main merge: không.
- Final decision: `PARTIAL_ROOT_CAUSE_CONFIRMED` cùng
  `NO_CONFIRMED_IMPLEMENTATION_BUG`.
