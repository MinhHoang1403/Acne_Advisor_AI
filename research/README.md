# Chỉ mục Nghiên cứu & Thực nghiệm Ngoại tuyến (`research/`)

Hệ thống chính thức chỉ có **một** bản triển khai duy nhất trong thư mục `src/`. Toàn bộ các script và báo cáo được giữ lại tại thư mục `research/` là **tài liệu phân tích và đối chứng ngoại tuyến ở trạng thái lưu trữ**, **không** phải một nhánh backend thay thế và **không** được import vào luồng xử lý câu hỏi trực tuyến.

## 1. Các nghiên cứu trọng tâm đã ghi nhận

- **[Phân tích nguyên nhân mất bằng chứng (RAG Evidence-Loss Study)](RAG_EVIDENCE_LOSS_ROOT_CAUSE_STUDY.md):** Phân tích trên 88 vị trí bằng chứng mục tiêu cho thấy hệ thống truy xuất được `85/88` và đóng gói vào ngữ cảnh `58/88`; nguyên nhân chênh lệch nằm ở giới hạn ký tự (`20` ca) và giới hạn số đoạn (`7` ca), không phải lỗi thực thi truy xuất.
- **[Nghiên cứu đóng gói theo ngân sách (Budget-Aware Packing Study)](BUDGET_AWARE_PACKING_CONTROLLED_STUDY.md) & [Giao thức khóa](BUDGET_AWARE_PACKING_PROTOCOL.md):** So sánh có đối chứng nâng tỷ lệ đóng gói từ `58/88` lên `65/88` nhưng phát sinh thêm chi phí lựa chọn và đánh đổi `+12/-5` yêu cầu, không có tập kiểm định độc lập hợp lệ — kết luận `EXPERIMENT_FAILED` (không đưa vào sản phẩm). Cache ứng viên ngoại tuyến tại `data/cache/research_budget_aware_packing/candidates.json` được giữ cho mục đích tái chạy phân tích ngoại tuyến.
- **[Nghiên cứu hợp đồng trích xuất cục bộ (Local Extraction Contract Study)](LOCAL_RAGCHECKER_EXTRACTION_CONTRACT_STUDY.md):** Ghi nhận kết quả thử nghiệm extractor cục bộ (`Candidate C`) cùng dữ liệu hiệu chuẩn 4B/9B trong `results/local_extraction_contract_*` ở trạng thái `INDEPENDENT_GOLD_REVIEW_REQUIRED`.
- **[Rà soát ánh xạ bằng chứng (Controlled Evidence Review)](CONTROLLED_EVIDENCE_REVIEW.md):** Danh sách 8 mục ánh xạ kỹ thuật lưu trữ chờ rà soát bổ sung; không thay thế kiểm định lâm sàng độc lập.

## 2. Bảng tổng hợp các thực nghiệm đã đóng

| Nghiên cứu / Commit nguồn | Kết quả ghi nhận | Quyết định với bản chính (`main`) |
| --- | --- | --- |
| Local RAGChecker 4B (`0a598a8`) | `LOCAL_EVALUATOR_QUALITY_FAILED`; chưa đạt hiệu chuẩn | Không áp dụng |
| Local RAGChecker 9B (`9e52989`) | Cải thiện kiểm tra nhưng cổng trích xuất chưa đạt | Không áp dụng |
| RAGAS shadow (`1c4748c`) | `RAGAS_CALIBRATION_BLOCKER` ở thời điểm thử nghiệm cũ | Đã thay bằng bộ RAGAS 50 ca mới ở thư mục `evaluation/` |
| Focused evidence selector (`104c4fe`) | Thiếu tập kiểm định holdout độc lập; loại prototype facet/BGE | Không áp dụng |
| GPU evidence selection (`ed0b469`) | Chưa đủ cơ sở kết luận; cần rà soát nhãn thêm | Không áp dụng |
| Information-need decomposition (`7806b52`) | Chưa đạt ngưỡng ổn định (`3/13` ca phát triển đạt) | Giữ nguyên routing hiện tại |
| Requested-information representation (`a2fd3f4`) | Không đạt cổng schema và các ca trọng yếu | Không thay thế runtime |
| Retrieval-first boundary (`a410730`) | `RETRIEVAL_FIRST_QUERY_REWRITE_BLOCKER` | Giữ nguyên ranh giới action/query-rewrite |
| Simple-path baseline (`6baa08a`) | Chưa chốt cổng an toàn/nhãn | Không thay thế baseline |
| Retrieval methods (`72046e2`) | So sánh phục vụ nghiên cứu, chưa có cơ sở chuyển sản phẩm | Giữ nguyên cơ chế truy xuất hiện hành |
| Reranker studies (`a374b8e`, `7e02625`, `0d96c9a`) | Tích hợp kết luận BF16; loại bỏ giới hạn depth-24 | Giữ full-union inference (PR #115) |
| Retry evidence loss (`b6e1fd6`) | Hợp nhất kết quả giữ lại và truy xuất mới để rerank lại | Đã tích hợp vào sản phẩm qua PR #116 |
| Evidence-path audit (`a99f259`) | Các bản vá đóng gói/suy luận tuần tự đã có bản triển khai mới hơn trên `main` | Không áp dụng lại bản vá cũ |

## 3. Quan hệ với bộ đánh giá chính thức hiện tại

Bộ đánh giá RAGChecker luận văn trước đây đã được thay thế trên `main` bằng **bộ đánh giá định lượng RAGAS 50 ca** sử dụng Judge độc lập qua NVIDIA NIM (`nvidia/nemotron-3-ultra-550b-a55b`). Xem chi tiết cấu trúc, cách chạy và báo cáo hiện hành tại [evaluation/README.md](../evaluation/README.md) và [evaluation/ragas_evaluation.ipynb](../evaluation/ragas_evaluation.ipynb). Toàn bộ báo cáo và dữ liệu trong `research/` chỉ giữ lại làm tài liệu tham khảo lịch sử.
