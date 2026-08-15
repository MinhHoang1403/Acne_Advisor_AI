# Hướng dẫn đọc mã nguồn Acne Advisor AI

Tài liệu này giúp người bảo trì tìm đúng module owner trước khi thay đổi hệ thống.
Nó mô tả source code hiện tại; các identifier, API và environment variable vẫn
dùng tên tiếng Anh như trong code.

## Luồng request chính

1. Frontend gửi `POST /chat` qua
   [`src/frontend/src/api/chatApi.js`](../src/frontend/src/api/chatApi.js).
2. [`src/api/app.py`](../src/api/app.py) kiểm input, gọi LangGraph Agent và map lỗi
   runtime sang HTTP response.
3. [`src/agent/graph.py`](../src/agent/graph.py) chạy workflow tám node với một
   deadline dùng chung.
4. [`src/agent/action_decision.py`](../src/agent/action_decision.py) yêu cầu model
   chọn action; Python kiểm schema, state và giới hạn retrieval.
5. [`src/retrieval/service.py`](../src/retrieval/service.py) chạy Dense và BM25
   song song, fuse rank bằng RRF rồi đóng gói context.
6. [`src/agent/nodes/reason.py`](../src/agent/nodes/reason.py) xây prompt từ packed
   evidence và gọi LLM provider để sinh draft.
7. Response đi qua presentation, source validation, verifier kỹ thuật, exact cache
   và observability trước khi API trả về frontend.

## Bản đồ source

| Khu vực | Module chính | Trách nhiệm | Không chịu trách nhiệm |
|---|---|---|---|
| Knowledge preparation | `src/ingestion/parser.py`, `chunking.py`, `filtering.py`, `provenance.py` | Parse source, chia chunk, lọc proof và tạo identity | Không search khi user hỏi |
| Embedding/indexing | `src/ingestion/embedding.py`, `index.py`, `bm25.py` | Gọi embedding provider, cấu hình/tạo Qdrant candidates | Không tự chạy cosine/BM25 search engine |
| Build orchestration | `src/ingestion/pipeline.py`, `scripts/knowledge_build.py` | Prepare, validate, build và activate có kiểm soát | Không thuộc normal API request path |
| Retrieval | `src/database/vector_store.py`, `src/retrieval/service.py` | Gọi Dense/BM25 search, quản lý timeout/degraded status | Không sinh answer hoặc xác minh y khoa |
| Fusion/context | `src/retrieval/rrf.py`, `context_packer.py` | Fuse rank, dedupe identity và áp resource budget | Không rerank theo truth/confidence |
| Agent | `src/agent/graph.py`, `action_decision.py`, `state.py` | Topology, semantic action và state contract | Model không tự thực thi transition |
| Safety | `src/agent/safety_policy.py` | Bảy override deterministic, hẹp, source-mapped | Không phải classifier/chẩn đoán tổng quát |
| Generation | `src/agent/nodes/reason.py`, `agent/prompts/medical_answer.py` | Xây prompt và gọi LLM từ evidence | Không search/rerank lại context |
| Presentation | `src/agent/nodes/respond.py`, `answer_formatting.py`, `source_presentation.py` | Format answer và giới hạn source mention | Không chứng minh claim đúng |
| Verification | `src/quality/answer_verifier.py` | Kiểm cấu trúc và provenance identity | Không kiểm clinical truth/entailment |
| Cache | `src/agent/nodes/cache.py`, `src/cache/exact_cache.py` | Exact normalized Redis cache có version identity | Không tìm câu hỏi gần nghĩa |
| Resilience | `src/resilience/` | Deadline, timeout, retry và error classification | Không quyết định provider content |
| Provider adapters | `src/integrations/google_genai.py`, `src/agent/llm/` | Đóng gói SDK/HTTP contract | Provider mới thực thi model inference |
| Persistence | `src/database/connection.py`, `repositories/chat_history.py` | PostgreSQL session/transaction và chat SQL | Không lưu knowledge vectors |
| Structural knowledge | `src/knowledge/`, Neo4j | EntityCards và deterministic entity graph | Không grounding normal answer runtime |
| API | `src/api/app.py` | HTTP validation, orchestration boundary, response metadata | Không sở hữu retrieval/generation algorithms |
| Frontend | `src/frontend/src/App.jsx`, `api/chatApi.js`, `components/` | Session UI, request lifecycle và rendering | Không đánh giá safety/evidence |
| Operator checks | `scripts/check_*.py`, `inspect_*.py`, `pre_ui_runtime_check.py` | Readiness/contract reports có timeout | Không thay đổi dữ liệu trừ script init/build được gọi rõ ràng |

## Knowledge preparation và indexing

### Parse, chunk và provenance

`parser.py` đọc Markdown/JSON trực tiếp và dùng LlamaParse cho PDF. Parsed cache
được nhận diện bằng content/contract identity. `chunking.py` ưu tiên ranh giới cấu
trúc và giới hạn 2400 Unicode characters, overlap bằng 0. Đây là engineering
contract của build, không phải kích thước tối ưu cho mọi corpus.

`provenance.py` nối các identity:

```text
source file -> document_id -> record_id -> chunk_id -> Qdrant point UUID
```

Hash/fingerprint làm output deterministic và phát hiện input thay đổi. Provenance
chỉ trả lời "dữ liệu đến từ đâu", không chứng minh nội dung nguồn đúng hay đủ.

### Dense embedding

Project gọi `models/gemini-embedding-2` qua Google GenAI adapter. Google provider
tạo vector; project kiểm số vector và dimension. Khi query runtime:

```text
query text -> Gemini embedding provider -> 3072-D vector
           -> Qdrant cosine search -> ranked Dense candidates
```

Qdrant mới thực thi cosine search trên vectors đã index. Muốn đổi model/dimension
phải xem đồng thời embedding/index compatibility; không chỉ sửa một constant.
Gemini Embedding 2 không nhận `task_type`; tài liệu provider khuyến nghị prefix
instruction vào text cho text-only retrieval. Build hiện tại dùng text không có
prefix. Đây là câu hỏi cần A/B test trước khi thay đổi vì format document mới sẽ
yêu cầu re-embed/reindex; không được suy ra chất lượng chỉ từ provider contract.

### BM25

[`src/ingestion/bm25.py`](../src/ingestion/bm25.py) có ba lớp trách nhiệm:

- Project khai báo `k1`, `b`, `avg_len`, tokenizer/language và document/query
  contract gửi cho Qdrant.
- Qdrant tạo sparse representation, lưu sparse vector, tính collection-side IDF
  và thực thi BM25 search.
- `reference_bm25_score()` là phép tính Python cho unit test/reference; user
  retrieval không gọi hàm này.

Với term `t`, document `D` và corpus có `N` documents:

```text
IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

score(t,D) = IDF(t) * tf(t,D) * (k1 + 1)
             / (tf(t,D) + k1 * (1 - b + b * |D| / avgdl))
```

- `df(t)`: số document chứa `t`.
- `tf(t,D)`: số lần `t` xuất hiện trong `D`.
- `|D|`, `avgdl`: độ dài document và độ dài trung bình.
- `k1`: điều khiển TF saturation.
- `b`: điều khiển document-length normalization.

Trong build hiện tại, `avg_len=256` là configured provider/index baseline, không
phải phép đo được tuyên bố là true corpus average hay giá trị đã chứng minh tối ưu.

## Retrieval runtime

Dense và BM25 chạy đồng thời với timeout độc lập. Một channel lỗi không xóa
evidence đã nhận từ channel còn lại; status trở thành `degraded_dense` hoặc
`degraded_bm25`. Hai raw score có thang đo khác nhau nên không được cộng trực tiếp.

### Reciprocal Rank Fusion

RRF dùng one-indexed rank:

```text
RRF(d) = sum_r weight_r / (k + rank_r(d))
```

Trong code, `rank_r(d)` là vị trí của candidate `d` trong channel `r`, `weight_r`
là channel weight và `k=60` làm giảm ảnh hưởng của chênh lệch rank tuyệt đối.
Candidate xuất hiện ở cả Dense và BM25 nhận hai contribution. `k` là engineering
parameter, không phải relevance hay medical-confidence threshold.

`context_packer.py` giữ nguyên thứ tự RRF, dedupe bằng stable item identity và áp
`max_items`/`max_chars`. Nó không chạy reranker và không quyết định item nào là
"sự thật". Packed context chính là bounded evidence được gửi tới generation.

## Agent, safety và generation

Model trong `action_decision.py` chọn semantics của một action:

- `retrieve`: lấy evidence lần đầu.
- `retry`: lấy evidence lần sau bằng query khác.
- `generate`: sinh answer khi evidence có thể dùng.
- `abstain`: không tiếp tục khi thiếu evidence hoặc không thể xử lý an toàn.

Python parse strict JSON rồi kiểm action có hợp lệ với state hay không. Chỉ có tối
đa hai retrieval executions; action sai schema/state fail closed thành `abstain`.
Model không thể tự thêm node, chạy tool hoặc vượt budget.

`assess_evidence_node` chỉ kiểm item có text và source identity. Presence và
provenance không đồng nghĩa semantic sufficiency, entailment hay medical truth.
Safety policy cũng cố ý hẹp: ngoài inventory deterministic, Agent xử lý semantics
từ source evidence và model synthesis.

## Cache, version và resilience

Exact cache key dựa trên:

```text
SHA256(cache schema | answer version | pipeline fingerprint |
       normalized question | provider | model)
```

Question chỉ được chuẩn hóa case/dấu câu/whitespace rồi exact-match. Không có
embedding similarity. Pipeline fingerprint là 24 hex characters đầu của SHA-256
trên canonical, secret-free JSON manifest; nó phân vùng cache khi contract đổi,
không phải security signature.

Retry delay dùng exponential backoff có cap và jitter:

```text
base(i)  = base_delay * 2^(max(0, i - 1))
capped   = min(base(i), max_delay)
delay(i) = min(capped + U(0, capped * jitter_ratio), max_delay)
```

Mỗi stage dùng `effective_timeout = min(configured_timeout, remaining_deadline)`.
Retry và provider fallback tiếp tục dùng phần deadline còn lại, không khởi tạo lại
tổng thời gian request. Đây là resource policy, không phải confidence formula.

Metadata ingestion có field compatibility tên `confidence`, nhưng giá trị chỉ là
coverage heuristic: bằng 0 khi không có nhóm metadata nào, nếu không thì bằng
`min(0.3 + 0.1*n, 1.0)` với `n` là số nhóm có giá trị. Nó không tham gia retrieval
score và không phải xác suất, source confidence hay medical confidence.

## Structural knowledge và verifier

Taxonomy normalizer, EntityCards và Neo4j graph giúp build/validate các quan hệ
cấu trúc. Normal chat runtime không truy vấn Neo4j/EntityCards để grounding; do đó
không được mô tả graph records như answer evidence.

Answer verifier kiểm Markdown/presentation contract và packed evidence identity.
Nó không so từng claim với source, không làm clinical review và không thể chứng
nhận answer đúng y khoa.

## Frontend và persistence

`App.jsx` sở hữu session state, health loop và request-in-flight guard. `chatApi.js`
sở hữu HTTP timeout/error mapping. `ChatMessage.jsx` chỉ render answer, display
source metadata và badge; raw source IDs vẫn nằm trong response data.

Frontend giữ localStorage để hoạt động khi backend chưa reachable, sau đó đồng bộ
session summaries/messages với PostgreSQL. Repository dùng bind parameters và
message ID để tránh insert lặp; transaction commit/rollback thuộc session owner.

## Muốn thay đổi chức năng thì đọc ở đâu

| Nhu cầu | Bắt đầu tại | Cần kiểm tra cùng |
|---|---|---|
| Đổi BM25 config | `src/ingestion/bm25.py` | index rebuild contract, Qdrant collection |
| Đổi Dense model/dimension | `src/ingestion/embedding.py`, `src/database/vector_store.py` | Google adapter, Qdrant vector config |
| Đổi RRF `k`/weights | `src/retrieval/rrf.py` | version manifest, retrieval tests |
| Đổi candidate/context limits | `src/retrieval/service.py`, `context_packer.py` | prompt budget, version manifest |
| Đổi Agent actions | `src/agent/action_decision.py` | state contract, graph routes, tests |
| Đổi graph topology | `src/agent/graph.py` | workflow nodes, bounded loop invariants |
| Thêm safety override | `src/agent/safety_policy.py` | source mapping, positive/negative tests |
| Đổi generation instructions | `src/agent/prompts/medical_answer.py` | prompt budget, answer contract tests |
| Đổi presentation/source rules | `src/agent/answer_formatting.py`, `source_presentation.py` | frontend rendering, verifier tests |
| Đổi exact cache identity | `src/cache/exact_cache.py` | versioning, cache nodes, Redis inspection |
| Đổi retry/timeout | `src/resilience/` | provider adapter, total deadline tests |
| Đổi chunking | `src/ingestion/chunking.py` | build identity, provenance, reindex plan |
| Đổi API contract | `src/api/app.py` | Pydantic models, frontend `chatApi.js` |
| Đổi chat persistence | `src/database/repositories/chat_history.py` | schema initializer, API endpoints |
| Đổi frontend request lifecycle | `src/frontend/src/App.jsx` | connectivity, chat API tests |

## Runtime-consumed strings

Không dịch hoặc sửa tùy tiện các nội dung sau vì framework/model/API có thể dùng
trực tiếp:

- Docstring của LangChain `retrieve_evidence` tool.
- System/user prompts và action-selection instruction.
- FastAPI route docstrings/OpenAPI descriptions.
- Pydantic field descriptions và model-facing schemas.
- Safety/fallback response text, API error/status strings.
- Cache/version/build/source/model identifiers và persisted metadata keys.

Tài liệu phương pháp và parameter đầy đủ nằm tại
[`METHODS_AND_FORMULAS.md`](METHODS_AND_FORMULAS.md). Mapping từ phương pháp tới
code owner, nguồn, classification và limitation nằm tại
[`METHOD_TRACEABILITY.md`](METHOD_TRACEABILITY.md). Quy trình vận hành nằm tại
[`OPERATIONS.md`](OPERATIONS.md).
