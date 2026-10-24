# Nghiên cứu contract claim extraction cho local RAGChecker

Trạng thái tài liệu: **development đã hoàn tất; dừng chờ developer adjudication
và independent holdout gold review**.

Nghiên cứu này chỉ đánh giá contract đầu ra cho `qwen3.5:9b-q4_K_M` trên máy
local. Không sửa production, không dùng model lớn hơn, không gọi OpenAI, không
chạy lại 100 câu formal evaluation và không thay đổi ý nghĩa metric RAGChecker.

## Identity đã khóa

- RAGChecker `0.1.9`, RefChecker `0.2.17`, Ollama `0.40.2`.
- Model `qwen3.5:9b-q4_K_M`, digest
  `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`.
- Development set: 8 extraction + 12 checking, canonical SHA-256
  `69b837a5571614b24c7705b362af1cb451b6111fddd60c9e471ffbd49fd04644`.
- Candidate A tái sử dụng artifact ba lần chạy có SHA-256
  `6b620873c2ffe008afbb98ff5ba96129674ac580e0abdb9c503c0bba9e53df71`.
- Cấu hình A/B/C, prompt C, JSON Schema, gates và stop rules nằm trong
  `research/results/local_extraction_contract_protocol.json`.

## Forensic audit RAGChecker 0.1.9

Audit được thực hiện trực tiếp trên package cài trong image
`acne-thesis-evaluation:366f3ad8`; hash từng file nguồn đã được khóa trong
protocol JSON.

### Parser chính xác

`LLMExtractor.extract_claim_triplets` dùng prompt stock, gọi callback với
`temperature=1e-5`, rồi gọi `parse_claims(res, "###")`. Với claim format
`triplet`, parser:

1. dùng tám regex cho mọi tổ hợp dấu nháy đơn/kép của tuple ba phần tử;
2. gọi `eval` trên chuỗi match, có fallback tách bằng `", "`;
3. bỏ record nếu độ dài khác 3, có phần tử không phải string, hoặc bất kỳ string
   nào có độ dài bằng 0;
4. loại tuple header `("subject", "predicate", "object")` và deduplicate bằng
   exact tuple equality.

Hệ quả: empty object không được giữ dưới dạng claim thiếu trường. Stock parser
âm thầm bỏ chính record đó và tiếp tục giữ record hợp lệ khác. Strict local
adapter của nghiên cứu 9B chủ động fail cả response để lỗi không bị che giấu;
hai hành vi này phải được phân biệt khi đọc artifact cũ.

### Checking parser chính xác

Joint checker dùng regex không phân biệt hoa thường để lấy mọi token
`Entailment|Neutral|Contradiction`. Parser package không tự xác nhận số label tại
bước regex; local adapter cũ bổ sung kiểm tra fail-closed rằng số label phải đúng
bằng số claim và không cho phép text khác.

### Claim đi vào metric nào

`RAGChecker.check_claims` định tuyến claim như sau:

| Intermediate | Claim được extract | Reference |
|---|---|---|
| `answer2response` | response | ground-truth answer |
| `response2answer` | ground-truth answer | response |
| `retrieved2answer` | ground-truth answer | từng retrieved passage |
| `retrieved2response` | response | từng retrieved passage |

Sau đó `ragchecker.computation` tính:

- precision = mean của `answer2response`;
- recall = mean của `response2answer`;
- F1 từ precision và recall;
- claim recall = mean theo các ground-truth claim sau khi lấy max qua passages;
- context precision = mean theo passages sau khi lấy max qua ground-truth claims;
- faithfulness = mean theo response claims sau khi lấy max qua passages.

Do đó mất một ground-truth claim làm mất một hàng của `retrieved2answer` và một
phần tử của `response2answer`: claim recall/recall có thể bị tăng giả nếu claim
khó bị bỏ; context precision cũng có thể giảm nếu passage chỉ hỗ trợ claim bị
bỏ. Mất một response claim làm mất một hàng của `retrieved2response` và một phần
tử của `answer2response`: faithfulness/precision có thể bị tăng giả nếu claim
không được support bị bỏ. F1 thay đổi theo hai thành phần này. Đây là thay đổi
ngữ nghĩa đánh giá, không phải lỗi trình bày vô hại.

## Phân loại forensic CAL-EXT-01..08

Phân loại dựa trên input, reference annotation và raw output giống nhau ở cả ba
run. `review_required` được giữ riêng, không tự động biến thành “model sai”.

| Case | Phân loại | Kết luận forensic |
|---|---|---|
| CAL-EXT-01 | `NO_DEFECT_CONFIRMED` | Hai tuple bao phủ bệnh viêm mạn và đơn vị nang lông–tuyến bã; parser nhận được. |
| CAL-EXT-02 | `SEMANTIC_DISTORTION`, `SEMANTIC_OMISSION` | “tác dụng tiêu nhân mụn nhẹ” bị đổi thành “mild acne reduction”: đổi loại hiệu quả dược lý và không giữ qualifier “nhẹ/mildly”. Không phải syntax-only. |
| CAL-EXT-03 | `STRUCTURE_WITH_INFORMATION_LOSS` | Claim “bằng chứng còn mâu thuẫn” bị dồn vào predicate và object rỗng. Stock parser bỏ record này. Điền object đòi hỏi tái phân chia ý nghĩa, nên B phải reject. |
| CAL-EXT-04 | `INCONCLUSIVE_NEEDS_HUMAN_REVIEW` | Output giữ đủ token “cần thiết” và “không phải lúc nào” nhưng phân bố qua predicate/object bất thường. Lexical assessor không xác nhận R01; chưa có independent adjudication để gọi đây là semantic error. |
| CAL-EXT-05 | `SEMANTIC_OMISSION`, `SEMANTIC_DISTORTION` | “kháng sinh bôi” bị tổng quát thành “antibiotics”; chủ thể phối hợp trở thành “Treatment”; “có thể hạn chế” đổi thành “can reduce”. Route và scope quan trọng không được bảo toàn rõ. |
| CAL-EXT-06 | `STRUCTURE_WITH_INFORMATION_LOSS` | “tác dụng chống viêm” bị đặt toàn bộ vào predicate với object rỗng. Stock parser bỏ claim thứ ba; B không được tự sáng tạo object. |
| CAL-EXT-07 | `NO_DEFECT_CONFIRMED` | Phân biệt open/closed comedone đúng, không thấy omission hoặc addition. |
| CAL-EXT-08 | `NO_DEFECT_CONFIRMED` | Giữ đúng phủ định “BPO is not topical antibiotic” và phân biệt clindamycin. |

Không có lỗi nào được xác nhận là `FORMAT_ONLY`: A đã parse được các tuple không
rỗng, còn hai empty-object case chứa lỗi cấu trúc gắn với phân chia thông tin.
Không có `UNSUPPORTED_ADDITION` được xác nhận riêng; các chỗ diễn đạt lệch ở
EXT-02/05 được đánh dấu distortion/omission, không được hợp thức hóa thành claim
mới đúng.

## Ba candidate preregistered

### A — stock baseline

Giữ nguyên prompt/parser RAGChecker. Chỉ tái sử dụng artifact cũ khi toàn bộ
identity khớp. Kết quả đã quan sát trước nghiên cứu contract là extraction 3/8
và checking 12/12 ở mỗi lần; protocol mới không coi đó là quan sát candidate C.

### B — deterministic syntax-only normalization

Chỉ được bóc heading/fence bao ngoài, parse tuple/list/JSON-array ba string, và
serialize lại chính ba string đó. Không trim nội dung field, không sửa empty
field, không split/merge/reorder claim, không suy luận hoặc đổi label. Nếu không
chứng minh được bảo toàn lossless thì reject toàn response.

### C — Ollama native structured extraction

Dùng native `format` JSON Schema, prompt mới đã khóa và deterministic conversion
sang từng dòng tuple chính xác của RAGChecker. Đây được ghi rõ là evaluator config
mới; không gọi nó là stock prompt. Checking vẫn dùng stock checking path.

## Independent holdout boundary

`research/results/local_extraction_contract_holdout_draft.json` chứa 30 case
extraction và 18 case checking mới lấy từ source documents, không lấy từ output
100 câu. Tất cả gold để `null` và trạng thái
`pending_independent_review`; người tạo adapter/model không được tự duyệt gold.

Không được chạy candidate trên holdout trước khi:

1. một reviewer độc lập chốt complete claim set cho 30 extraction case;
2. reviewer độc lập chốt đúng một nhãn ba lớp cho 18 checking case;
3. near-duplicate report với calibration và formal benchmark đạt gate;
4. hash của packet đã adjudicate được commit trước lần chạy mù.

Nếu thiếu bước này, nghiên cứu phải dừng ở
`INDEPENDENT_GOLD_REVIEW_REQUIRED`, không được tuyên bố validated.

## Development result — chờ developer adjudication

Candidate A và B được tính lại từ exact raw artifact cũ, không gọi model mới.
Candidate C chạy đủ 3 × 20 case theo protocol đã commit trước inference.

| Candidate | Parseable/run | Accepted/run | Review/run | Checking/run | Kết quả gate |
|---|---:|---:|---:|---:|---|
| A stock | 8/8 | 3/8 | 5/8 | 12/12 | fail: 2 empty-object + quality |
| B syntax-only | 6/8 | 3/8 | 3/8 | 12/12 | fail: lossless repair không thể chứng minh cho EXT-03/06 |
| C structured | 8/8 | 5/8 | 3/8 | 12/12 | pending/fail automatic gate: EXT-01/03/04 cần review |

C loại bỏ empty field và sửa rõ các lỗi cũ ở EXT-02/05/06, nhưng schema không
bảo đảm ngữ nghĩa. Đặc biệt EXT-03 sinh string literal `"null"` làm object; đây
là red flag ngữ nghĩa dù JSON và tuple đều hợp lệ. EXT-01 tách subject thành
chuỗi hai claim; EXT-04 phân bố phủ định qua predicate/object bất thường.

- 24/24 output extraction C được official RAGChecker 0.1.9 parser đọc lại đúng
  bằng exact tuple equality.
- Raw structured output, converted output và checking labels lặp lại 100% theo
  item qua ba run.
- 36/36 checking labels khớp baseline; không timeout, OOM, truncation hay model
  error.
- 60 request: 24 structured extraction + 36 stock checking; tổng wall telemetry
  981.565 giây, median 4.074 giây, max 51.809 giây; 2,643 output tokens; median
  generation 7.19 token/giây.
- Runtime vẫn hybrid: 6,687,398,293 bytes tổng và 3,520,279,345 bytes VRAM.
- Một giới hạn 10 phút của command orchestrator làm tiến trình ngoài model bị
  ngắt; runner resume từ durable checkpoint ở item 45/60. Không output đã
  checkpoint nào bị chạy lại và sự kiện được ghi trong artifact.

Ba case chưa được tự phán quyết. Packet
`research/results/local_extraction_contract_development_review.json` để trống
verdict/reviewer và yêu cầu developer xác nhận. Theo stop rule, không chọn
candidate cuối, không chạy holdout và không tuyên bố validated trước adjudication
này. Ngoài ra, 48 gold holdout vẫn đang chờ reviewer độc lập.

Trạng thái hiện tại theo tập trạng thái được phép:
**`INDEPENDENT_GOLD_REVIEW_REQUIRED`**.
