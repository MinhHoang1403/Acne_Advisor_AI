# Method Traceability

Tài liệu này nối từng phương pháp hoặc công thức với code owner, nguồn hỗ trợ và
phần thích nghi riêng của Acne Advisor AI. Việc implementation có test không đồng
nghĩa phương pháp đã được chứng minh tối ưu, cải thiện chất lượng trên corpus này,
hay được xác nhận hiệu quả lâm sàng.

## Classification

- `IMPLEMENTED_RESEARCH_METHOD`: phương pháp có nguồn nghiên cứu và có mặt trong code.
- `OFFICIAL_PROVIDER_CONTRACT`: hành vi do tài liệu chính thức của provider định nghĩa.
- `OFFICIAL_FRAMEWORK_CONTRACT`: hành vi do framework chính thức định nghĩa.
- `CLINICAL_SAFETY_SOURCE`: nguồn y khoa/y tế công cộng hỗ trợ một safety action hẹp.
- `RELATED_LITERATURE`: ý tưởng liên quan nhưng project không tuyên bố tái hiện paper.
- `ENGINEERING_POLICY`: quyết định hữu hạn của project, không phải hằng số khoa học.
- `EMPIRICAL_PROJECT_DECISION`: quyết định dựa trên đo lường nội bộ và chỉ có phạm vi project.

## Canonical Matrix

| Method / Formula | Code owner | Purpose | Source | Classification | Project adaptation | Source does NOT validate |
|---|---|---|---|---|---|---|
| Dense retrieval | `src/database/vector_store.py`, `src/retrieval/service.py` | Tìm chunk gần query theo biểu diễn dense | `karpukhin_dpr_2020` | `IMPLEMENTED_RESEARCH_METHOD` | Dùng Gemini Embedding 2 và Qdrant thay cho DPR encoder/index | Chất lượng Gemini, tiếng Việt hoặc corpus mụn |
| Gemini Embedding 2 provider contract | `src/integrations/google_genai.py`, `src/ingestion/embedding.py` | Gửi text, nhận và kiểm vector 3072 chiều | `google_gemini_embedding2_2026` | `OFFICIAL_PROVIDER_CONTRACT` | Project kiểm batch count/dimension; model không nhận `task_type` | Chất lượng cross-lingual hoặc format text hiện tại là tối ưu |
| Cosine vector search | `src/ingestion/index.py`, `src/database/vector_store.py` | Cấu hình và gọi Dense similarity search | `qdrant_cosine_search_2026` | `OFFICIAL_PROVIDER_CONTRACT` | Qdrant sở hữu normalization/search; project gửi query vector | Relevance của vector hoặc ngưỡng chất lượng project |
| BM25 | `src/ingestion/bm25.py` | Biểu diễn sparse với IDF, TF saturation và length normalization | `robertson_zaragoza_bm25_2009` | `IMPLEMENTED_RESEARCH_METHOD` | Python formula chỉ là reference test; runtime do Qdrant thực thi | Qdrant details hoặc `k1=1.2`, `b=0.75`, `avg_len=256` là tối ưu |
| Qdrant native BM25 | `src/ingestion/bm25.py`, `src/database/vector_store.py` | Giữ document/query preprocessing parity và sparse search | `qdrant_bm25_2026` | `OFFICIAL_PROVIDER_CONTRACT` | Tokenizer `word`, lowercase, language `none`, IDF collection-side | Chất lượng retrieval trên corpus mụn hoặc parameter optimality |
| Reciprocal Rank Fusion | `src/retrieval/rrf.py` | Hợp nhất Dense/BM25 bằng rank, không cộng raw score | `cormack_clarke_buettcher_rrf_2009` | `IMPLEMENTED_RESEARCH_METHOD` | `k=60`, hai weight bằng `1.0` | Các giá trị project là tối ưu hoặc có medical-confidence semantics |
| Agent action selection | `src/agent/action_decision.py`, `src/agent/graph.py` | Model chọn một action semantic; Python kiểm schema/transition | LangGraph Graph API; LangChain Tools | `OFFICIAL_FRAMEWORK_CONTRACT` | Bốn action `retrieve/retry/generate/abstain`, không cho model tự chạy transition | Chất lượng quyết định hoặc clinical truth |
| ReAct relationship | `src/agent/action_decision.py` | Nêu quan hệ với reasoning/action interleaving | `yao_react_2023` | `RELATED_LITERATURE` | Chỉ nhận strict JSON decision, không thu/lưu free-form chain of thought | Action schema, topology hoặc safety policy của project |
| Active RAG relationship | `src/agent/action_decision.py`, `src/agent/nodes/workflow.py` | Cho model quyết định nhu cầu retrieve/retry | `jiang_active_rag_2023` | `RELATED_LITERATURE` | Retrieval theo request nhưng hữu hạn, không dùng FLARE token-level trigger | Giới hạn hai retrieval hoặc cơ chế generation của project |
| Adaptive-RAG relationship | `src/agent/action_decision.py` | Thích nghi action theo state/request | `jeong_adaptive_rag_2024` | `RELATED_LITERATURE` | Không có complexity classifier; model chọn trong schema đóng | Strategy set hoặc classifier của project |
| Context packing | `src/retrieval/context_packer.py` | Giữ fused order, dedupe identity, bảo toàn provenance | Không có claim nghiên cứu riêng | `ENGINEERING_POLICY` | Tối đa 8 item và 6000 ký tự theo mặc định | Semantic sufficiency, entailment hoặc budget tối ưu |
| Maximum retrieval attempts | `src/agent/action_decision.py`, `src/agent/graph.py` | Chặn loop retrieval vô hạn | Không có source định lượng | `ENGINEERING_POLICY` | Tối đa 2 retrieval executions | Hai lần là tối ưu cho latency/quality |
| Per-channel retrieval timeout | `src/retrieval/service.py` | Cô lập Dense/BM25 failure và giữ evidence channel còn lại | `aws_timeouts_retries_backoff_jitter_2019` | `ENGINEERING_POLICY` | Mặc định 20 giây cho mỗi channel | Timeout 20 giây là tối ưu cho hạ tầng hiện tại |
| Exact cache normalization | `src/cache/exact_cache.py` | Case-fold, thay dấu câu và collapse whitespace trước exact match | Không có semantic-cache claim | `ENGINEERING_POLICY` | Không dùng embedding hoặc near-match | Hai câu chuẩn hóa giống nhau luôn đồng nghĩa về y khoa |
| SHA-256 cache identity | `src/cache/exact_cache.py` | Tạo key cố định từ version, fingerprint, question, provider/model | `nist_fips_180_4_sha256` | `ENGINEERING_POLICY` | Hash canonical payload bằng SHA-256 theo technical standard | Field selection, normalization, authenticity hoặc cache correctness |
| Pipeline fingerprint | `src/observability/versioning.py` | Phân vùng cache theo canonical secret-free manifest | `nist_fips_180_4_sha256` | `ENGINEERING_POLICY` | Lấy 24 hex đầu của SHA-256 trên canonical JSON | Digest truncation là security signature hoặc version manifest đầy đủ |
| UUIDv5 EntityCard identity | `src/knowledge/entity_identity.py` | Tạo Qdrant point ID ổn định từ entity identity | `ietf_rfc9562_uuidv5` | `ENGINEERING_POLICY` | Namespace project cố định + canonical entity name theo technical standard | Canonical-name policy, provenance hoặc data authenticity |
| Exponential backoff | `src/resilience/retry.py` | Giãn retry cho transient provider failure | `aws_timeouts_retries_backoff_jitter_2019` | `ENGINEERING_POLICY` | Một retry, base 1 giây, cap 4 giây | Số retry và delay project là tối ưu |
| Positive jitter | `src/resilience/retry.py` | Giảm retry đồng pha | `aws_timeouts_retries_backoff_jitter_2019` | `ENGINEERING_POLICY` | Cộng `U(0, capped * 0.1)` rồi giới hạn bởi cap/deadline | Tỷ lệ `0.1` là tối ưu hoặc phù hợp mọi workload |
| Deadline budgeting | `src/resilience/budget.py`, `src/agent/graph.py` | Chia sẻ finite request deadline giữa stage/retry/fallback | `aws_timeouts_retries_backoff_jitter_2019` | `ENGINEERING_POLICY` | `effective_timeout=min(configured_timeout, remaining_deadline)` | Tổng 210 giây hay stage deadlines là tối ưu |
| Structure-aware chunking | `src/ingestion/chunking.py` | Ưu tiên heading/paragraph/sentence boundary | `wang_segmentation_2025` | `RELATED_LITERATURE` | Heuristic deterministic của project, không triển khai PIC | Heuristic này cải thiện retrieval hoặc generation trên corpus hiện tại |
| 2400-character chunk cap | `src/ingestion/chunking.py`, `src/knowledge/versioning.py` | Giới hạn kích thước chunk build | Không có nguồn chứng minh optimum | `ENGINEERING_POLICY` | Cap theo Unicode characters, nằm trong build identity | 2400 là độ dài tối ưu hoặc generalizable |
| Zero chunk overlap | `src/ingestion/chunking.py`, `src/knowledge/versioning.py` | Tránh lặp text giữa chunks trong build hiện tại | Không có nguồn chứng minh optimum | `ENGINEERING_POLICY` | Overlap bằng 0 và được version hóa | Zero overlap không làm mất context ở mọi source |
| Metadata coverage heuristic | `src/ingestion/domain_metadata.py` | Biểu diễn mức phủ của entity fields đã match | Không có source nghiên cứu | `ENGINEERING_POLICY` | `0` khi không match; ngược lại `min(0.3 + 0.1*n, 1.0)` | Xác suất đúng, confidence y khoa hoặc calibration |
| Deterministic safety rules | `src/agent/safety_policy.py` | Override hẹp cho emergency/pregnancy/isotretinoin và no-prescription | Các nguồn tại `docs/REFERENCES.md` | `CLINICAL_SAFETY_SOURCE` | Bảy rule source-mapped; Python quyết định trigger/action | General medical reasoning, diagnosis hoặc độ bao phủ mọi tình huống |
| Provenance identity | `src/ingestion/provenance.py` | Nối source, document, record, chunk và point identity | NIST SHA-256 qua `nist_fips_180_4_sha256` | `ENGINEERING_POLICY` | Hash canonical inputs để tái tạo identity và phát hiện thay đổi | Nội dung nguồn đúng, đầy đủ hoặc đáng tin cậy |
| Answer verification boundary | `src/quality/answer_verifier.py` | Kiểm structure, source allowlist và provenance-related contract | Không có clinical validation claim | `ENGINEERING_POLICY` | Technical verifier sau generation; fail closed theo contract | Medical truth, claim-level entailment hoặc evidence completeness |

## Gemini Retrieval-Instruction Evaluation Question

Google ghi nhận `gemini-embedding-2` không hỗ trợ trường `task_type` và khuyến
nghị prefix instruction trực tiếp vào text cho text-only retrieval, ví dụ phân
biệt query với document. Build và query hiện tại của Acne Advisor AI dùng text
không có prefix instruction. Chưa có A/B test trên corpus hiện tại chứng minh
instruction prefix cải thiện hay làm giảm chất lượng tiếng Việt. Vì thay format
document sẽ yêu cầu re-embed/reindex, đây chỉ là câu hỏi đánh giá sau này; tài
liệu này không thay đổi embedding request, stored vector hoặc index.

## Cách đọc kết quả kiểm thử

- Unit/formula test chứng minh implementation tuân theo contract đã viết.
- Provider-contract test chứng minh assumption tích hợp trong test environment.
- Các test đó không chứng minh retrieval quality, answer quality, clinical
  effectiveness hoặc parameter optimality.

Chi tiết công thức nằm tại [Methods and Formulas](METHODS_AND_FORMULAS.md); metadata
nguồn nằm tại `data/phase1_method_sources.json` và [References](REFERENCES.md).
