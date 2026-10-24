# Giao thức đánh giá luận văn cuối cùng

Trạng thái tài liệu: đăng ký trước phương pháp, phải được commit trước request đầu tiên của lần chạy chính thức.

## 1. Đối tượng đánh giá và ranh giới

- Repository: `MinhHoang1403/Acne_Advisor_AI`.
- System-under-test (SUT, hệ thống được đánh giá): `366f3ad836c36be36565caad4a94ad17a7a6278f`.
- Pipeline fingerprint (dấu vân tay cấu hình đường chạy): `e4c420923b79b37c992cf36d`.
- Answer-cache namespace: `v10`; cache production phải bật.
- KB build: `d4a1819fe7fb77fe1f40`.
- Qdrant target: `acne_knowledge__d4a1819fe7fb77fe1f40`, đúng 536 điểm.
- Benchmark: `evaluation/benchmark_100.json`.
- Canonical benchmark SHA-256: `f61d6807c0ce39f902936844d810562c486f1bcadaa57a8a2da0e460ad7e534b`.
- Gold/adjudication nguồn: commit `6da9b67634aa231a3d9cd51891af6d1685ab25cc`.
- Gold gồm 104 claim đã review, 24 ca xác nhận thiếu bằng chứng và 6 ca mơ hồ về chính sách.

Mọi commit của lần đánh giá chỉ được thay đổi `evaluation/` và test dành riêng cho evaluation. Không thay đổi production, prompt, model sinh, embedding, Dense, BM25, RRF, reranker, retry, packer, safety, cache semantics, KB, Qdrant hoặc nhãn benchmark. Failure được ghi nhận, không sửa production trong cùng task.

## 2. Thiết kế bộ ca

Bộ benchmark có đúng 100 ca và chạy theo thứ tự đang lưu:

- 70 ca answerable (có thể trả lời từ KB): 55 single-turn và 15 multi-turn;
- 30 ca evidence-gap candidate: 24 ca `CONFIRMED_ABSENCE`, 6 ca `POLICY_AMBIGUOUS`.

Mỗi ca chỉ được thực thi một lần. Không loại ca khỏi mẫu số và không chạy lại vì output thấp. Retry/fallback nội tại của production vẫn hoạt động. Chỉ được resume từ checkpoint khi có hỏng hạ tầng bên ngoài SUT; các ca đã hoàn tất không chạy lại. Provider failure được production chuyển thành output hợp lệ vẫn là kết quả của ca đó.

## 3. Điều kiện cache và datastore

Một Redis dùng riêng cho evaluation, không gắn volume dữ liệu production, được dùng với `CACHE_ENABLED=true`. Trước mỗi ca, chỉ database của Redis evaluation được làm rỗng và kiểm tra có 0 key. Điều kiện được ghi là `isolated_cold_per_case`. Không flush Redis thông thường của người dùng.

Qdrant và KB chỉ đọc. Không ingestion, rebuild, upsert, delete hoặc đổi alias; không mutation Neo4j. Điểm và alias Qdrant được đọc trước và sau lần chạy. Write audit kỳ vọng: Qdrant 0, KB 0, production Redis flush 0.

Một cache smoke test tách biệt sẽ dùng chính Redis evaluation để xác nhận lookup/store production còn hoạt động; dữ liệu smoke không tham gia 100 ca và Redis được làm rỗng sau smoke.

## 4. Output hệ thống được khóa trước evaluator

Thứ tự bắt buộc:

1. chạy đủ 100 ca production;
2. lưu final answer sau finalize/verifier cùng action và reason;
3. lưu full acquisition candidate union, merged retry candidates và exact packed generation context;
4. xác minh packed IDs trùng `generation_evidence_trace`, tối đa 9 chunk/7000 ký tự;
5. lưu raw output, tính canonical SHA-256 và raw-byte SHA-256;
6. commit raw output;
7. chỉ sau commit này mới tính metric và chạy semantic evaluator.

Draft, intermediate answer hoặc debug context chỉ có 5 item không được dùng. RAGChecker và metric packed evidence dùng toàn bộ các block đúng như context đã đưa vào prompt sinh. Nếu exact generation context không tái dựng được, dừng với `FINAL_EVALUATION_CONTEXT_BLOCKER`.

## 5. Metric xác định không dùng LLM judge

Các metric dưới đây được tính từ raw output đã khóa và gold hiện hành.

### 5.1. Evidence trên 70 ca answerable

- Acquisition required-evidence recall = số reviewed required evidence ID xuất hiện trong hợp của toàn bộ candidate được acquisition / tổng reviewed required evidence ID.
- Packed required-evidence recall = số reviewed required evidence ID xuất hiện trong exact generation context / tổng reviewed required evidence ID.
- Complete-claim packing = claim có đủ mọi evidence ID bắt buộc; báo cáo số claim hoàn tất / 104.
- Complete-case packing = ca có đủ mọi reviewed required evidence ID; báo cáo số ca hoàn tất / 70.
- Strict reviewed-ID context precision = số packed ID thuộc reviewed gold của chính ca / tổng packed ID.

Metric precision cuối được ghi rõ là **STRICT REVIEWED-ID PRECISION**. Reviewed ID có thể không bao quát mọi chunk hữu ích về ngữ nghĩa, nên metric không phải chân lý relevance phổ quát.

### 5.2. Evidence-gap

Metric primary là **Confirmed evidence-gap rejection rate**, N=24. Một ca đạt khi và chỉ khi structured `action=abstain` và `reason=evidence_gap`.

Sáu ca `POLICY_AMBIGUOUS` chỉ báo cáo phân bố hành vi `generate`, `retry`, `abstain`, `other/failure`, không tạo aggregate pass/fail. Nếu báo cáo legacy 30-ca thì phải ghi secondary và không dùng làm kết luận primary.

### 5.3. Completion, provenance và cấu trúc

- Phân bố action trên 70 ca answerable: generate, abstain, other/error.
- Answerable completion rate dùng `final_action=generate`; generation không được diễn giải thành factual correctness.
- Citation ID validity: source ID trả về thuộc source allowlist của cùng ca.
- Source allowlist validity: không còn invalid source mention sau production validation.
- Provenance validity: packed item có chunk ID, nguyên văn và source ID.
- Structural verifier pass rate: `structural_verification_report.passed=true`.

Các metric provenance/cấu trúc chỉ kiểm tra contract kỹ thuật, không khẳng định entailment hoặc đúng lâm sàng.

### 5.4. Operational

Trên 100 ca, báo cáo N/min/median/p95/max theo mili giây cho:

- end-to-end;
- pre-decision = prepare + guard + action decision đầu tiên;
- tổng retrieval của mọi attempt;
- tổng reranker của mọi attempt;
- generation;
- finalization.

P95 dùng nearest-rank: phần tử thứ `ceil(0.95*N)` sau khi sắp tăng. Ngoài ra báo cáo retrieval-attempt distribution, retry rate, provider fallback rate, reranker fallback rate, timeout, error, cache hit và generation provider/model distribution.

### 5.5. Subgroup và category

Giữ nguyên metric đã khóa để báo cáo riêng 55 single-turn, 15 multi-turn và từng category sẵn có trong benchmark. Không tạo metric hay nhóm category mới sau khi nhìn score.

## 6. RAGChecker cho 70 ca answerable

RAGChecker (bộ đo claim-level dùng LLM evaluator) được khóa ở `0.1.9`; evaluator là `gpt-5.4-2026-03-05`. Adapter dùng OpenAI Responses API, `store=False`, extraction reasoning `medium`, checking reasoning `low`, batch size 8, `joint_check=true`, `joint_check_num=5`. Không đổi prompt hoặc model.

Bốn headline metric dùng đúng định nghĩa package:

1. Claim Recall;
2. Context Precision;
3. Faithfulness;
4. Claim F1.

Per-case ratio 0..1 được nhân 100 để báo cáo phần trăm; aggregate package được kiểm tra là 0..100. Chỉ 70 ca answerable được đưa vào headline RAGChecker. Không đưa negative abstention vào các metric này.

Calibration hiện có chỉ được tái sử dụng nếu đồng thời khớp canonical calibration input SHA, RAGChecker version, evaluator model và AST hash của adapter contract tại commit đã được researcher phê duyệt. Nếu không chứng minh được exact match thì phải calibration lại. Nếu có item cần phán xét người nghiên cứu, Codex không tự duyệt và dừng `RAGCHECKER_CALIBRATION_REVIEW_REQUIRED`. Nếu gate thất bại, dừng `RAGCHECKER_CALIBRATION_BLOCKER`. Nếu model exact không dùng được, dừng `RAGCHECKER_EVALUATOR_BLOCKER`.

Evaluator failure không bao giờ làm chạy lại output SUT. Resume evaluator chỉ dùng cùng raw-output hash.

## 7. RAGAS bị loại khỏi formal evaluation

Không cài, không chạy, không thêm dependency và không báo cáo RAGAS như formal metric. Kết quả nghiên cứu trước `RAGAS_CALIBRATION_BLOCKER` chỉ có thể xuất hiện trong limitations/methodology, không trộn với score chính thức.

## 8. Preflight và validation cuối

Trước request live đầu tiên phải đạt:

- schema/count/hash benchmark;
- gold identity/count/hash và byte identity với commit nguồn;
- toàn bộ reviewed required evidence ID tồn tại trong Qdrant 536 điểm;
- SUT, pipeline fingerprint, KB, alias, cache namespace;
- production-sensitive diff rỗng;
- dependency consistency;
- Ruff và format check cho evaluation code;
- unit test công thức, JSON, exact context và checkpoint/resume;
- no-secret scan.

Sau kết quả phải parse mọi JSON; kiểm tra row count CSV, 100 case ID duy nhất, denominator, artifact hash; chạy Ruff/format, `git diff --check`, secret scan, safety regression suite và kiểm tra lại production-sensitive diff/Qdrant/KB write audit.

## 9. Artifact và thống kê

Thư mục duy nhất là `evaluation/results/thesis_evaluation_366f3ad8/`. Artifact bắt buộc: run manifest, system output manifest, raw results, deterministic metrics, case/category metrics, operational metrics, RAGChecker checkpoint, metrics summary, báo cáo Markdown và artifact manifest.

Proportion luôn có count/denominator/percentage, làm tròn một chữ số thập phân. Latency có N/min/median/p95/max với độ chính xác hợp lý. Không confidence interval, significance test hoặc historical causal claim nào được thêm sau khi nhìn kết quả.

## 10. Kết luận và giới hạn diễn giải

Kết quả mô tả hệ thống đã khóa trên benchmark acne-domain 100 ca và KB đã khóa. Nó không chứng minh hiệu quả lâm sàng hoặc clinical safety accuracy. Safety fixture chỉ được gọi là **deterministic safety regression coverage**. Báo cáo phải công khai kết quả âm, stochasticity của model sinh, phụ thuộc evaluator LLM, giới hạn reviewed-ID precision, resource policy 9/7000, giới hạn retry/BF16 và sáu nhãn policy-ambiguous.
