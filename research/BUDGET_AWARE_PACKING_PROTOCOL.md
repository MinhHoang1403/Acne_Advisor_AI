# Offline Budget-Aware Context Packing Protocol

## Trạng thái preregistration

Protocol này khóa trước khi chạy treatment. Nghiên cứu chỉ được phép tiếp tục qua
baseline gate nếu toàn bộ 100 cases/104 retrieval attempts được replay chính xác.
Mọi thay đổi algorithmic sau khi xem development outcome sẽ tạo thành một nghiên cứu
mới, không được hợp thức hóa như protocol này.

## Research question

Với cùng frozen candidate pool, cùng final relevance order, cùng whole chunks, tối đa
9 items và 7.000 ký tự, liệu một exact rank-weighted budget-aware subset selector có giữ
được nhiều reviewed required evidence hơn greedy rank-order production packer không?

## Frozen baseline và identity

- Forensic commit: `2d32317d6abd335aedd69b7b066fca53c13b0bb8`.
- Evaluated production SHA: `366f3ad836c36be36565caad4a94ad17a7a6278f`.
- Frozen run: `thesis_evaluation_366f3ad8`.
- Pipeline fingerprint: `e4c420923b79b37c992cf36d`.
- KB build: `d4a1819fe7fb77fe1f40`.
- Frozen raw canonical SHA-256:
  `1f525efb44091c2444d298d45de4f31637b9be4d1d512f88db38ae9738125feb`.
- Frozen raw byte SHA-256:
  `b078858d61ced4ed072339fd9cb576517a2ce585724e425c6fc0a2f772f57f16`.
- Benchmark canonical SHA-256:
  `f61d6807c0ce39f902936844d810562c486f1bcadaa57a8a2da0e460ad7e534b`.
- Historical reference: 100 cases, 104 attempts, 88 requirements, 85 acquired,
  58 packed, 20 character losses và 7 item losses.

Raw byte hash và canonical JSON hash là hai identity riêng biệt. Identity gate fail
closed nếu bất kỳ giá trị khóa nào không khớp.

## Reconstruction contract

Nguồn được xét theo thứ tự:

1. Exact persisted frozen candidate objects.
2. Immutable artifact thuộc đúng KB build.
3. Read-only exact-ID lookup từ active Qdrant chỉ khi alias, build và schema khớp.

Không được vector search, BM25 search, embedding, rerank, query rewrite, update alias,
modify point hoặc dùng gold text để bù dữ liệu thiếu. Candidate lấy từ Source C vẫn là
reconstructed input cho tới khi production packer replay khớp frozen trace.

Mỗi reconstructed candidate phải giữ frozen candidate ID/order/ranks/scores và payload
text/provenance của đúng build. Full candidate cache nằm trong ignored research cache;
Git chỉ lưu checksum/manifest không chứa corpus duplicate.

## Exact baseline gate

Control là actual `pack_context()` tại evaluated production SHA; implementation hiện tại
chỉ được import nếu Git xác nhận không khác SHA đó. Giữ `max_items=9`, `max_chars=7000`.

Để qua gate, bắt buộc:

- 100/100 cases và 104/104 attempts có đủ candidate inputs.
- Selected IDs và order khớp 100%.
- Context length khớp 100%.
- Final context hashes khớp khi frozen hash tồn tại.
- Dropped/not-examined IDs và reasons khớp 100%.
- Packed provenance khớp.
- Coverage reconcile đúng 58/88 và acquisition đúng 85/88.
- Không có mismatch chưa giải thích.

Nếu không đạt: `BASELINE_REPLAY_BLOCKED`; không triển khai/chạy treatment.

## Treatment selector

Tên cố định:
`RANK_WEIGHTED_BUDGET_AWARE_WHOLE_CHUNK_SELECTION_V1`.

Input chỉ gồm ordered frozen candidates và current fixed budgets. Query/case identity chỉ
được giữ cho trace/harness, không phải feature. Selector không nhận gold IDs, claims,
answers, labels, evaluation scores hoặc annotations.

Với vị trí một-based trong final frozen order:

```text
utility_i = round(1_000_000 / log2(rank_i + 1))
```

Chọn exact 0/1 subset để tối đa tổng utility dưới các ràng buộc:

```text
selected_items <= 9
exact_rendered_context_chars <= 7000
whole_chunks_only = true
candidate IDs unique
```

Chi phí dùng actual production renderer, gồm header, source/chunk ID, bounded local scope
metadata, newline và hai ký tự separator giữa blocks. Candidate được giữ nguyên text và
payload. Selected subset được render lại theo original relevance order với display index
1..N. Bất kỳ feasible-state/rendered-output disagreement nào là correctness failure.

Solver là deterministic exact bounded dynamic programming; không được thay bằng greedy
heuristic vì outcome. Không có tham số tune.

## Tie-break contract

So sánh solution theo thứ tự cố định:

1. Tổng utility lớn hơn.
2. Lexicographically earlier tuple của original one-based candidate positions.
3. Tổng rendered characters nhỏ hơn.
4. Lexicographically earlier tuple của stable candidate IDs.

Selected output luôn giữ original relevance order.

## Experimental arms

- Control: actual production greedy whole-chunk `pack_context()`.
- Treatment: selector V1 ở trên.

Hai arms nhận cùng IDs, texts, payload, ranks/scores, attempts, query identity, 9-item
limit, 7.000-character limit và evaluation labels. Khác biệt duy nhất là subset policy.

## Development và holdout

Toàn bộ 100 frozen cases là `historically_exposed` vì đã được forensic analysis; chúng
chỉ tạo `DEVELOPMENT_RESULT`. Danh sách và hash được khóa trong experiment identity trước
khi comparison.

Independent holdout chỉ hợp lệ nếu nằm ngoài 100 cases, ngoài tám controlled mappings và
năm regression-only cases của prior selector study; đồng thời có reviewer-approved gold,
frozen replayable candidates, provenance và baseline cùng contract. Không chia ngẫu nhiên
100 cases cũ để gọi là holdout. Nếu không tìm thấy artifact hợp lệ:
`INDEPENDENT_HOLDOUT_UNAVAILABLE` và
`EXPLORATORY_ONLY_NO_INDEPENDENT_VALIDATION`.

## Metrics

Primary metric là packed required-evidence coverage với identity
`case_id + required_chunk_id` trên 88 reviewed requirements.

Paired output bắt buộc gồm baseline/treatment hits, denominator, absolute hit và
percentage-point delta, gained/lost/unchanged requirements. Secondary metrics gồm complete
cases, character/item-loss outcomes, item/chars utilization, selected-rank distribution,
candidate churn, gain/regression case IDs, selector runtime p50/p95 và resource proxy.

Không tính generation/RAGChecker/model-based metrics và không gán answer cũ cho treatment
context mới.

## Safety invariants

- Không đổi source/chunk IDs, medical text hoặc provenance.
- Không truncate/rewrite/summarize chunk, qualifier, negation, tên thuốc hay liều lượng.
- Không thêm text/candidate ngoài frozen eligible pool.
- Không duplicate ID, vượt 9 items hoặc vượt 7.000 ký tự.
- Không đổi citation/source allowlist contract.
- Gold chỉ xuất hiện sau selection trong evaluation join.

Packed recall không chứng minh medical correctness. Treatment có thể loại useful unlabeled
context, giảm coherence hoặc bỏ supporting qualifier ở chunk khác; nghiên cứu không chạy
generation nên không thể xác nhận các rủi ro ngữ nghĩa đó.

## Success, failure và stop rules

Sau exact replay, chạy đúng một paired development comparison, không tune và không thử
algorithm thứ hai. No improvement hoặc bất kỳ contract/provenance regression nào phải được
báo nguyên trạng.

Không có independent holdout thì dừng sau development. Có holdout hợp lệ thì khóa
protocol/code/IDs trước, inspect gold bị cấm trong development, và chỉ chạy đúng một paired
evaluation. Production adoption luôn `NOT_AUTHORIZED` trong task này.

## Reproducibility

- Stable ordering/tie-break; không RNG, current timestamp, GPU, provider hoặc network trong
  algorithm execution.
- Machine-readable deterministic outputs dùng sorted JSON keys, fixed CSV columns, UTF-8/LF.
- Chạy analyzer lần hai trên exact cached inputs và xác minh các deterministic output byte
  hashes không đổi.
- Runtime timing được đo riêng vì bản chất không byte-deterministic và không tham gia
  decision/tie-break/result hashes.

## Known limitations

- Frozen trace không chứa text/payload cho nonpacked candidates; exact source reconstruction
  phải được chứng minh bằng replay.
- Development set đã được nghiên cứu trước và không phải independent validation.
- Gold identity chỉ đánh giá labeled required evidence, không đo mọi useful context.
- Không có generation mới nên không suy ra answer quality hoặc clinical safety.
