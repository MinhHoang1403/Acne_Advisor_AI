# AI Observability

Tài liệu này mô tả backend Langfuse tùy chọn cho Acne Advisor AI. Hệ thống vẫn chạy
được khi Langfuse tắt, cấu hình sai hoặc không truy cập được. JSONL telemetry hiện có
vẫn là contract gốc; `src/observability/langfuse_sink.py` là biên tích hợp Langfuse duy
nhất và chỉ dựng lại dữ liệu từ một allowlist đóng.

## Kiến trúc và phạm vi

```text
FastAPI request
  -> LangGraph pipeline (8 node hiện có, không đổi topology)
  -> ObservabilityEvent hoàn tất
       |-> JSONL sink hiện có
       `-> LangfuseSink (tùy chọn, fail-open, async batch)
             `-> Langfuse self-host v4
                 web + worker + PostgreSQL + ClickHouse + Redis + MinIO
```

`request_id` do API tạo là định danh chuẩn xuyên suốt request. Langfuse SDK dẫn xuất
trace ID OpenTelemetry xác định từ chính giá trị này; `request_id` vẫn được giữ trong
metadata để điều tra chéo với log. Hệ thống không xuất `session_id`, vì session ID do
client gửi hiện chưa có contract đảm bảo đây là định danh opaque và không chứa PII.

## Mô hình trace

| Observation | Langfuse type | Parent | Metadata được phép |
|---|---|---|---|
| `chat-request` | `span` | root | request ID, trạng thái, tổng latency, action, count, version/fingerprint, lỗi đã phân loại, fallback/safety category |
| `retrieve` | `retriever` | request | trạng thái, latency, attempt, selected/retained/duplicate count, tối đa 20 opaque candidate IDs |
| `dense`, `bm25` | `retriever` | retrieve | trạng thái từng channel, latency, count, opaque IDs, loại lỗi |
| `rrf`, `rerank`, `pack` | `span` | retrieve | trạng thái, latency khi có, fused/eligible/packed count, loại lỗi |
| `agent-decision` | `agent` | request | attempt, action đóng, reason code đã chuẩn hóa |
| `generate` | `generation` | request | provider/model thực tế, latency, fallback, trạng thái |
| `guard` | `guardrail` | request | trạng thái và lỗi đã phân loại |

Root span bắt đầu cùng request nên có duration thực. Các child observation được dựng
từ telemetry đã hoàn tất để tránh SDK lan vào business code; vì vậy duration gốc của
child observation trong UI phản ánh thời gian enqueue, còn `metadata.duration_ms` mới
là số đo pipeline có thẩm quyền.

## Quyền riêng tư

Mặc định bảo vệ theo hai lớp:

1. `LangfuseSink` dựng metadata mới từ allowlist; không chuyển tiếp `safe_payload`,
   metadata tùy ý, input hoặc output.
2. `mask_otel_spans` xóa mọi attribute có tên gợi ý nội dung/credential trước khi
   exporter gửi batch. Đây là lớp phòng thủ bổ sung, không thay thế allowlist.

Không bao giờ xuất: câu hỏi gốc, lịch sử hội thoại, prompt, câu trả lời, chain of
thought/reasoning, email, tên người dùng, cookie/header xác thực, secret/API key,
exception message/stack trace, session ID chưa xác minh, metadata không giới hạn, hoặc
token usage chưa có contract đo thực tế đáng tin cậy.

Được phép xuất: định danh request opaque, timestamp, status, latency, bounded count và
opaque candidate ID, action/reason code đóng, provider/model, fallback, pipeline
fingerprint, knowledge build ID, tên lớp exception và taxonomy lỗi. Candidate IDs bị
giới hạn tối đa 20 phần tử cho mỗi observation.

## Cấu hình ứng dụng

```dotenv
OBSERVABILITY_ENABLED=true
LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=http://localhost:3001
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_TRACING_ENVIRONMENT=local
```

Cả `OBSERVABILITY_ENABLED` và `LANGFUSE_ENABLED` phải là `true`. Nếu thiếu URL hoặc
key, backend ở trạng thái `not_configured` và request vẫn chạy bình thường. Secret
không được commit; `.env*` cục bộ đã được ignore, trừ file example.

SDK chỉ khởi tạo lazy khi có request cần trace. Request path không gọi `auth_check`,
`flush` hay network probe đồng bộ. SDK dùng batch bất đồng bộ; `flush` chỉ dùng ở smoke
test hoặc bước operator, còn `shutdown` chạy ở lifecycle shutdown của ứng dụng.

## Khởi động Langfuse self-host cục bộ

Stack tùy chọn nằm ở `docker-compose.observability.yml`, tách volume khỏi PostgreSQL,
Redis và Qdrant của ứng dụng. Mọi image đều pin cả version và digest; không dùng
`latest`.

Tạo file `.env.langfuse.local` đã được ignore. Sinh từng secret riêng với PowerShell:

```powershell
[Convert]::ToHexString(
  [Security.Cryptography.RandomNumberGenerator]::GetBytes(32)
).ToLower()
```

Điền các biến sau bằng giá trị mạnh, khác nhau. Đây chỉ là tên biến, không phải giá trị
mẫu để dùng thật:

- Secret bắt buộc: `LANGFUSE_POSTGRES_PASSWORD`,
  `LANGFUSE_CLICKHOUSE_PASSWORD`, `LANGFUSE_REDIS_PASSWORD`,
  `LANGFUSE_MINIO_PASSWORD`, `LANGFUSE_SALT`, `LANGFUSE_ENCRYPTION_KEY` và
  `LANGFUSE_NEXTAUTH_SECRET`.
- Headless bootstrap tùy chọn: `LANGFUSE_INIT_ORG_ID`,
  `LANGFUSE_INIT_ORG_NAME`, `LANGFUSE_INIT_PROJECT_ID`,
  `LANGFUSE_INIT_PROJECT_NAME`, `LANGFUSE_INIT_PROJECT_PUBLIC_KEY`,
  `LANGFUSE_INIT_PROJECT_SECRET_KEY`, `LANGFUSE_INIT_USER_EMAIL`,
  `LANGFUSE_INIT_USER_NAME` và `LANGFUSE_INIT_USER_PASSWORD`.

`LANGFUSE_ENCRYPTION_KEY` phải là khóa hex 64 ký tự. Các credential khởi tạo chỉ phục
vụ bootstrap lần đầu; sau khi volume đã có dữ liệu, thay đổi chúng không tái tạo project.

```powershell
docker compose --env-file .env.langfuse.local `
  -f docker-compose.observability.yml config --quiet
docker compose --env-file .env.langfuse.local `
  -f docker-compose.observability.yml up -d
docker compose --env-file .env.langfuse.local `
  -f docker-compose.observability.yml ps
```

UI: `http://localhost:3001`. Readiness có kiểm tra database:
`http://localhost:3001/api/public/health?failIfDatabaseUnavailable=true`.

Để dừng mà vẫn giữ dữ liệu:

```powershell
docker compose --env-file .env.langfuse.local `
  -f docker-compose.observability.yml down
```

Không thêm `-v` trừ khi chủ động muốn xóa toàn bộ dữ liệu observability cục bộ.

Smoke test tích hợp dùng một Compose project riêng, secret ngẫu nhiên chỉ trong bộ nhớ,
fixture tổng hợp và volume tạm. Nó không đụng stack/volume mặc định và tự dừng, xóa đúng
volume smoke sau khi kiểm tra:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_langfuse_observability.py
```

## Health và chế độ lỗi

`GET /health` có check `langfuse` với `optional=true`, `required=false`. Endpoint không
probe mạng và không làm readiness của ứng dụng thất bại. Các trạng thái:

- `disabled`: một trong hai toggle tắt;
- `not_configured`: đã bật nhưng thiếu URL/key;
- `configured`: hợp lệ nhưng chưa tạo client;
- `accepting_events`: SDK đã nhận event vào batch;
- `degraded`: khởi tạo/enqueue/probe gặp lỗi.

`check_langfuse_connectivity()` là probe operator chủ động. Lỗi SDK, serialization,
timeout hay backend outage chỉ được log bằng loại lỗi an toàn; business response và
JSONL sink không bị phụ thuộc vào Langfuse.

## Evaluation scores

`submit_evaluation_score(request_id, name, value)` chỉ nhận số hữu hạn và đúng một
trong năm metric chính thức của project:

- `Claim Recall`
- `Context Precision`
- `Faithfulness`
- `Claim F1`
- `Negative Rejection Rate`

Giá trị và đơn vị được giữ nguyên; integration không tự scale, normalize, invent score
hay backfill run cũ. Hàm dùng cùng deterministic trace ID từ `request_id`, nên score
gắn vào trace tương ứng. Đây là đường cho evaluation pipeline/operator, không chạy tự
động trên request production.

## Dashboard tích hợp sẵn

Không cần thêm React, Prometheus hoặc Grafana. Trong Langfuse, tạo một Custom Dashboard
và dùng các widget/filter sau:

| Câu hỏi vận hành | Measure / filter |
|---|---|
| Traffic, success, degraded, failed | count root `chat-request`, group/filter `metadata.status` |
| Latency request | percentile/average latency của root; đối chiếu `metadata.duration_ms` |
| Retrieval health | `retrieve`, `dense`, `bm25`; group `metadata.status`, `metadata.error_type` |
| Retrieval volume | average/sum `metadata.candidate_count`, filter theo stage |
| Agent behavior | count `agent-decision`, group `metadata.action` và `metadata.reason_code` |
| Generation/fallback | `generation`, group `metadata.provider`, `metadata.model`, `metadata.fallback` |
| Version correlation | filter `metadata.pipeline_fingerprint`, `metadata.knowledge_build_id` |
| Evaluation quality | score name/value cho năm metric được hỗ trợ |

Golden signals được ánh xạ thành latency, traffic, errors và saturation. Hiện chưa có
queue-depth hay resource saturation có độ tin cậy ở application contract, nên dashboard
ghi rõ khoảng trống này thay vì dùng candidate count như một proxy sai nghĩa.

## Alert policy

Langfuse v4 có alerting, nhưng repository không kích hoạt alert với ngưỡng tùy ý.
Chính sách hiện tại: **threshold-based alert activation deferred pending baseline**.

Sau khi có workload đại diện, owner phải xác định SLO và cửa sổ đo; chỉ tạo alert cho
triệu chứng có hành động rõ ràng, ví dụ tỷ lệ `failed`, latency request hoặc backend
ingestion health. Không dùng ngưỡng lấy từ tài liệu nhà cung cấp làm SLO của Acne
Advisor AI. Lưu baseline, lý do chọn threshold, owner và runbook trước khi bật alert.

## Chẩn đoán

- Không thấy trace: kiểm tra hai toggle, URL/keys, `GET /health`, rồi chạy probe operator.
- `not_configured`: không điền key vào code; sửa file env cục bộ hoặc secret store.
- `degraded`: xem loại lỗi an toàn trong health/log; kiểm tra container health và clock.
- Trace có root nhưng thiếu child: kiểm tra `ObservabilityEvent.complete` và JSONL event.
- Duration child rất nhỏ: dùng `metadata.duration_ms`, theo giới hạn post-hoc đã nêu.
- Sai project sau khi đổi init key: headless init không ghi đè volume đã bootstrap; tạo
  project/key trong UI hoặc chủ động dùng một stack/volume mới.

## Nguồn và quyết định thiết kế

| Nguồn chính thức | Quyết định được hỗ trợ |
|---|---|
| [Langfuse Python SDK 4.15.4](https://pypi.org/project/langfuse/4.15.4/) | Pin SDK v4 tương thích Python 3.11 |
| [SDK overview](https://langfuse.com/docs/observability/sdk/overview) | OpenTelemetry, batching, fail-safe SDK boundary |
| [Trace IDs](https://langfuse.com/docs/observability/features/trace-ids-and-distributed-tracing) | Dẫn xuất trace ID xác định từ request ID |
| [Data model](https://langfuse.com/docs/observability/data-model) và [observation types](https://langfuse.com/docs/observability/features/observation-types) | Cây request/retriever/agent/generation/guardrail |
| [Masking](https://langfuse.com/docs/observability/features/masking) | Lớp `mask_otel_spans` phòng thủ bổ sung |
| [Queuing and batching](https://langfuse.com/docs/observability/features/queuing-batching) | Không flush đồng bộ trên từng request |
| [Scores via SDK](https://langfuse.com/docs/evaluation/evaluation-methods/scores-via-sdk) | Gắn metric evaluation theo trace |
| [Custom dashboards](https://langfuse.com/docs/metrics/features/custom-dashboards) | Dùng dashboard tích hợp sẵn |
| [Alerts](https://langfuse.com/docs/observability/features/alerts) | Xác nhận capability; chưa tự đặt threshold |
| [Self-host Docker Compose](https://langfuse.com/self-hosting/deployment/docker-compose) | Topology stack cục bộ hiện hành |
| [Health/readiness](https://langfuse.com/self-hosting/configuration/health-readiness-endpoints) | Healthcheck có database readiness |
| [OpenTelemetry Trace API](https://opentelemetry.io/docs/specs/otel/trace/api/) | Trace identity theo chuẩn OTel |
| [Google SRE monitoring](https://sre.google/sre-book/monitoring-distributed-systems/) và [practical alerting](https://sre.google/sre-book/practical-alerting/) | Golden signals và alert dựa trên symptom/action |

Registry máy đọc được nằm ở `data/method_sources.json`. Các nguồn trên chỉ chứng minh
contract/capability kỹ thuật; không chứng minh chất lượng retrieval, độ đúng y khoa,
clinical effectiveness, capacity hoặc SLO của deployment này.
