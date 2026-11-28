# Đánh giá Hệ thống Acne Advisor AI bằng RAGAS

Thư mục này chứa quy trình đánh giá định lượng toàn diện hệ thống **Acne Advisor AI** theo phương pháp **RAGAS** (*Retrieval Augmented Generation Assessment*, Es et al., 2023, `arXiv:2309.15217`).

## 1. Cấu trúc thư mục

| Thành phần | Chức năng |
| --- | --- |
| `evaluation/dataset.json` | Bộ benchmark chuẩn **100 câu hỏi tiếng Việt** bám sát 53 nguồn y khoa trong kho dữ liệu (`b768fae999391f47760a`), bao phủ **7 nhóm chủ đề lâm sàng** và **5 dạng hội thoại** (`single_turn`, `colloquial_patient`, `brand_bridge`, `multi_turn`, `safety_abstention`). |
| `evaluation/ragas_runner.py` | Module lõi gọi trực tiếp `run_clinical_agent` (hỗ trợ cả câu đơn lượt và hội thoại đa lượt có `conversation_history`), chấm 5 chỉ số RAGAS bằng Judge LLM độc lập qua NVIDIA NIM (judge chính `nvidia/nemotron-3.5-super-vl-120b-a12b`, dự phòng `mistralai/mistral-large-4-1t-a52b` → `nvidia/nemotron-3-ultra-550b-a55b`; ghi lại judge thực tế đã chấm từng metric), hỗ trợ cả `models/gemini-embedding-2` (3.072 chiều) và `nvidia/nemotron-3.5-embed-8b` (4.096 chiều), xuất báo cáo Markdown/JSON. |
| `evaluation/ragas_evaluation.ipynb` | Jupyter Notebook trực quan để chạy toàn bộ quy trình đánh giá 100 ca có lưu checkpoint tự động và xuất kết quả ra file `.md` / `.json`. |
| `evaluation/judge_reliability.py` | Kiểm định độ tin cậy của judge trên bản ghi đã có: `rescore` (chấm lại bằng một judge cố định, tắt chuỗi dự phòng) và `planted` (chèn mệnh đề y khoa sai có chủ đích để đo độ nhạy của Faithfulness). |
| `evaluation/ablation.py` | Thí nghiệm ablation 3 cấu hình lồng nhau (`dense_bm25`, `dense_bm25_rerank`, `full_graphrag`) với một judge cố định, checkpoint riêng cho từng cấu hình. |
| `evaluation/paired_comparison.py` | So sánh ghép cặp hai lần chấm theo từng ca: chênh lệch trung bình, tương quan hạng Spearman, số ca thắng/hòa/thua và sign test chính xác. |
| `evaluation/results/` | Thư mục chứa kết quả xuất ra sau mỗi lần chạy: `ragas_evaluation_report.md`, `ragas_evaluation_raw.json` cùng các file checkpoint tiến trình. |

## 2. Các chỉ số đánh giá

1. **Faithfulness (Độ trung thực với bằng chứng):** Tỷ lệ mệnh đề trong câu trả lời được chứng minh trực tiếp bởi các đoạn chứng cứ truy hồi (`retrieved_contexts`), giúp phát hiện ảo giác (*hallucination*).
2. **Answer Relevancy (Độ liên quan câu trả lời):** Mức độ câu trả lời bám sát đúng trọng tâm câu hỏi của người dùng (mặc định dùng `models/gemini-embedding-2` 3.072 chiều hoặc tùy chọn `nvidia/nemotron-3.5-embed-8b`, tự động bảo toàn ngôn ngữ tiếng Việt và câu hỏi giải chiếu đại từ cho ca đa lượt).
3. **Context Precision (Độ chính xác ngữ cảnh):** Kiểm tra các đoạn chứng cứ hữu ích có được xếp ở thứ hạng đầu hay không.
4. **Context Recall (Độ bao phủ ngữ cảnh):** Kiểm tra ngữ cảnh truy hồi có phủ đủ thông tin trong đáp án chuẩn (`reference`) hay không.
5. **Answer Correctness (Độ chính xác câu trả lời):** Kết hợp độ chính xác sự kiện (75%) và độ tương đồng ngữ nghĩa (25%) giữa câu trả lời của hệ thống và đáp án chuẩn.
6. **Safe Abstention Accuracy (Tỷ lệ từ chối an toàn):** Đánh giá riêng các câu hỏi cấp cứu/cảnh báo đỏ, yêu cầu kê đơn/tính liều cá nhân hóa, ngoài miền da liễu hoặc mẹo truyền miệng thiếu bằng chứng xem hệ thống có xử lý/từ chối an toàn thay vì phỏng đoán hay không.

## 3. Cách chạy đánh giá

1. Đảm bảo các container Docker (`acne_qdrant`, `acne_postgres`, `acne_redis`, `acne_neo4j`) đang hoạt động và `.env` đã có `GOOGLE_API_KEY` (cho hệ thống + embeddings) cùng `NVIDIA_API_KEY` (cho RAGAS Judge).
2. Mở file `evaluation/ragas_evaluation.ipynb` bằng VS Code / Jupyter (chọn kernel `.venv` của dự án) và chạy tuần tự các cell (`Run All`).
3. Xem báo cáo tổng hợp và chi tiết từng ca tại `evaluation/results/ragas_evaluation_report.md`.

## 4. Kiểm định độ tin cậy của judge và ablation

Các lệnh dưới đây đọc `.env` giống notebook và ghi kết quả vào `evaluation/results/` (đã nằm trong `.gitignore`). Mọi lần chấm đều dùng **một judge cố định, tắt chuỗi dự phòng**, để chênh lệch điểm không đến từ việc đổi giám khảo giữa chừng.

| Mục đích | Lệnh | Tốn quota |
| --- | --- | --- |
| Độ ổn định của judge (nhiễu nền) | `python -m evaluation.judge_reliability rescore --input evaluation/results/ragas_evaluation_raw.json --label primary_run_a` rồi chạy lại với `--label primary_run_b` | NVIDIA + embedding Gemini (không sinh câu trả lời) |
| Đồng thuận giữa các judge | `rescore` với `--judge mistralai/mistral-large-4-1t-a52b --label mistral` (hoặc judge khác) | Như trên |
| Độ nhạy với lỗi cài sẵn | `python -m evaluation.judge_reliability planted --input evaluation/results/ragas_evaluation_raw.json --limit 20` | Chỉ NVIDIA (chỉ chấm Faithfulness) |
| Ablation | `python -m evaluation.ablation run --config dense_bm25` (lặp lại với `dense_bm25_rerank`, `full_graphrag`) | Khoảng 250 lượt gọi Gemini mỗi cấu hình |
| So sánh hai lần chấm bất kỳ | `python -m evaluation.paired_comparison --baseline A.json --candidate B.json --baseline-label ... --candidate-label ... --title ... --output report.md` | Không |

**Cách đọc kết quả:**

- **Spearman ρ** đo hai lần chấm có xếp hạng các ca giống nhau không (1 = giống hoàn toàn). Lưu ý chỉ số bị "chạm trần" (vd Context Precision gần như toàn 1.0) sẽ cho ρ thấp hoặc không xác định dù hai lần chấm gần như giống hệt nhau; khi đó hãy đọc thêm *Trung bình |B − A|*.
- **Thắng / Hòa / Thua** và **sign test p** cho biết chênh lệch có nhất quán qua các ca hay chỉ do vài ca kéo trung bình. Mức chênh giữa hai lần chấm **cùng judge** là nhiễu nền: chênh lệch ablation nhỏ hơn mức này không có ý nghĩa.
- **Bài cài lỗi** chỉ chọn ca đang đạt Faithfulness = 1.0 và chèn một trong bốn mệnh đề sai đã được đối chiếu là trái ngược với kho tri thức (`PLANTED_FALSE_CLAIMS` trong `judge_reliability.py`, mỗi mệnh đề ghi kèm bằng chứng). Tỷ lệ phát hiện là tỷ lệ ca có Faithfulness giảm sau khi cài lỗi.
- Checkpoint của bước chạy hệ thống chứa pipeline fingerprint, nên đổi cấu hình (vd tắt GraphRAG) sẽ không bao giờ tái sử dụng câu trả lời của cấu hình khác.
