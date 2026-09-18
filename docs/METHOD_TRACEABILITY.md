# Method Traceability

Tài liệu này nối từng phương pháp với code owner, nguồn hỗ trợ, phần thích nghi
của Acne Advisor AI và giới hạn diễn giải. Một implementation có test không đồng
nghĩa phương pháp đã tối ưu hoặc được xác nhận hiệu quả lâm sàng.

## Phân loại

- `IMPLEMENTED_RESEARCH_METHOD`: phương pháp nghiên cứu có mặt trong code.
- `OFFICIAL_PROVIDER_CONTRACT`: hành vi do tài liệu chính thức của provider mô tả.
- `RELATED_LITERATURE`: nghiên cứu liên quan nhưng không được tuyên bố là đã tái hiện.
- `ENGINEERING_POLICY`: quyết định hữu hạn của project.
- `EMPIRICAL_PROJECT_DECISION`: quyết định từ đo lường nội bộ, chỉ có phạm vi project.
- `CLINICAL_SAFETY_SOURCE`: nguồn hỗ trợ một safety action hẹp.

## Phương pháp đang hoạt động

| Phương pháp | Code owner | Nguồn | Phân loại | Thích nghi và giới hạn |
|---|---|---|---|---|
| Dense retrieval | `src/database/vector_store.py`, `src/retrieval/service.py` | `karpukhin_dpr_2020`, `qdrant_cosine_search_2026` | `IMPLEMENTED_RESEARCH_METHOD` + `OFFICIAL_PROVIDER_CONTRACT` | Gemini Embedding 2 và Qdrant thay DPR encoder/index; không chứng minh chất lượng tiếng Việt hoặc y khoa. |
| Gemini Embedding 2 | `src/integrations/google_genai.py`, `src/ingestion/embedding.py` | `google_gemini_embedding2_2026` | `OFFICIAL_PROVIDER_CONTRACT` | 3072 chiều, cosine, không dùng `task_type`; document là `title: {title} \| text: {content}`, query là `task: question answering \| query: {content}`. Provider docs không chứng minh format này tối ưu. |
| Native BM25 | `src/ingestion/bm25.py`, `src/database/vector_store.py` | `robertson_zaragoza_bm25_2009`, `qdrant_bm25_2026` | `IMPLEMENTED_RESEARCH_METHOD` + `OFFICIAL_PROVIDER_CONTRACT` | Qdrant thực thi; tokenizer `word`, lowercase, ASCII folding, language `none`, IDF collection-side. `k1=1.2`, `b=0.75`, `avg_len=256` không được tuyên bố tối ưu. |
| Reciprocal Rank Fusion | `src/retrieval/rrf.py` | `cormack_clarke_buettcher_rrf_2009` | `IMPLEMENTED_RESEARCH_METHOD` | `k=60`, hai weight `1.0`; source không chứng minh các giá trị project là tối ưu hoặc mang nghĩa medical confidence. |
| Local cross-encoder | `src/retrieval/reranker.py`, `src/retrieval/service.py` | `bge_reranker_v2_m3_model_card` | `OFFICIAL_PROVIDER_CONTRACT` + `ENGINEERING_POLICY` | `BAAI/bge-reranker-v2-m3`, local-files-only, timeout hữu hạn; lỗi giữ thứ tự deterministic trước rerank. Raw score không phải xác suất. |
| Whole-chunk packing | `src/retrieval/context_packer.py` | `lost_in_the_middle_2024` chỉ là related context | `EMPIRICAL_PROJECT_DECISION` | Giữ nguyên chunk và provenance, tối đa 9 item/7000 ký tự; không tuyên bố semantic sufficiency hoặc optimality. |
| Bounded Agent actions | `src/agent/action_decision.py`, `src/agent/nodes/workflow.py` | `yao_react_2023`, `jiang_active_rag_2023`, `jeong_adaptive_rag_2024` | `ENGINEERING_POLICY` | Bốn action `retrieve/retry/generate/abstain`, tối đa hai lần retrieval. Project không tái hiện prompt, classifier hoặc strategy set của các paper. |
| Purposeful evidence retry | `src/agent/action_decision.py`, `src/agent/nodes/workflow.py`, `src/retrieval/service.py` | `rewrite_retrieve_read_2023`, `conqrr_2022`, `itercqr_2024` là related literature | `ENGINEERING_POLICY` | Một retry khi không có evidence; giữ stable overall query cho rerank, dedupe/rerank/repack bounded. Không có dedicated query rewriter. |
| Structure-aware chunking | `src/ingestion/chunking.py`, `src/ingestion/parser.py` | `wang_segmentation_2025` | `RELATED_LITERATURE` + `ENGINEERING_POLICY` | Block, list lead-in và table row; cap 2400 Unicode chars, overlap 0. Không triển khai PIC và không tuyên bố cap tối ưu. |
| Exact cache/fingerprint | `src/cache/exact_cache.py`, `src/observability/versioning.py` | `nist_fips_180_4_sha256` | `ENGINEERING_POLICY` | SHA-256 trên payload canonical secret-free; fingerprint 24 hex là compatibility partition, không phải authentication signature. |
| Bounded resilience | `src/resilience/`, `src/api/preflight.py` | `aws_timeouts_retries_backoff_jitter_2019` | `ENGINEERING_POLICY` | Deadline, retry/backoff và jitter đều hữu hạn theo từng provider; tài liệu AWS là nguồn thiết kế liên quan, không chứng minh các ngưỡng của project là tối ưu. |
| Stable request persistence | `src/api/app.py`, `src/database/repositories/chat_history.py`, `src/frontend/src/utils/sessionMerge.js` | `ietf_rfc9562_uuidv5` cho UUID construction | `ENGINEERING_POLICY` | UUID request id tạo stable message IDs; replay không nhân đôi turn, nhưng request khác có cùng text vẫn tách biệt. |
| Deterministic safety boundaries | `src/agent/safety_policy.py` | các `CLINICAL_SAFETY_SOURCE` trong registry | `CLINICAL_SAFETY_SOURCE` + `ENGINEERING_POLICY` | Chín rule hẹp, current-question-first, không trộn third-person/resolved/topic-reset; không phải general medical classifier. |
| Answer verification | `src/quality/answer_verifier.py` | không có clinical validation claim | `ENGINEERING_POLICY` | Kiểm structure, provenance/source allowlist, explicit requested-entity omission và off-scope substitution; không kiểm medical truth hoặc entailment tổng quát. |
| RAG evaluation | `evaluation/` | `ragchecker_2024` | `IMPLEMENTED_RESEARCH_METHOD` | Historical package dùng Claim Recall, Context Precision, Faithfulness, F1 và NRR; không phải clinical validation và không tự giải quyết benchmark-label ambiguity. |
| Request observability | `src/observability/` | Langfuse/OpenTelemetry entries trong registry | `OFFICIAL_PROVIDER_CONTRACT` + `ENGINEERING_POLICY` | Một identity từ `request_id`, không export raw content, fail-open; không chứng minh privacy tuyệt đối hoặc quality improvement. |

## Quyết định thực nghiệm hiện tại

- Embedding instruction representation được giữ: trên 14 gold identities,
  recall@8/16/32 giữ 14/14, MRR tăng từ 0.6444 lên 0.7556 và source diversity
  top-16 tăng từ 5.467 lên 7.467.
- BM25 giữ tokenizer `word` và bật ASCII folding. Trên 15 cặp, tiếng Việt có
  dấu giữ 8/15 ở top-16 và 9/15 ở top-32; query không dấu đạt cùng mức đó.
- Candidate depth giữ 16. Depth 24/32 chỉ cứu thêm một case/nhóm trong phép thử
  nhỏ và tăng cost/noise; RRF vẫn `k=60`, weight 1:1.
- Packer tăng từ 8/6000 lên 9/7000 vì cứu hai evidence ở rank 9 mà không cắt
  chunk; candidate thật pack đầy đủ 13/15 case source-grounded.
- Query prompt B bị loại và revert vì completeness top-9 giảm từ 5/9 xuống 3/9,
  dù top-16 giữ 5/9. Không thêm dedicated rewriter.

Các số trên là development diagnostics của corpus này, không phải benchmark
chính thức hoặc tuyên bố tối ưu phổ quát.

## Nghiên cứu liên quan nhưng không triển khai

| Nguồn | Vai trò | Trạng thái project |
|---|---|---|
| Rewrite-Retrieve-Read, CONQRR, IterCQR | query/conversational reformulation | Đã dùng để định hướng thí nghiệm; prompt B bị loại, không thêm rewriter. |
| Adaptive-RAG | request-dependent retrieval strategy | Chỉ là context; không có complexity classifier. |
| `question_decomposition_2025`, `subquestion_coverage_2025` | multi-part retrieval | Đã xem xét, hoãn; không có decomposition agent/framework. |
| Lost in the Middle | context-position sensitivity | Context cho packing; không chứng minh budget 9/7000 tối ưu. |
| RAGChecker | fine-grained RAG diagnosis | Dùng ở package đánh giá lịch sử; sáu GAP-ABS vẫn chờ researcher adjudication. |

MMR, GraphRAG, CRAG, Self-RAG, HyDE, extra reranker, decomposition agent và
observability queue không được thêm. Chi tiết công thức nằm tại
[Methods and Formulas](METHODS_AND_FORMULAS.md); metadata nguồn nằm tại
`data/method_sources.json` và [References](REFERENCES.md).
