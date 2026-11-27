# Đánh giá Hệ thống Acne Advisor AI bằng RAGAS

Thư mục này chứa quy trình đánh giá định lượng toàn diện hệ thống **Acne Advisor AI** theo phương pháp **RAGAS** (*Retrieval Augmented Generation Assessment*, Es et al., 2023, `arXiv:2309.15217`).

## 1. Cấu trúc thư mục

| Thành phần | Chức năng |
| --- | --- |
| `evaluation/dataset.json` | Bộ benchmark chuẩn **100 câu hỏi tiếng Việt** bám sát 53 nguồn y khoa trong kho dữ liệu (`b768fae999391f47760a`), bao phủ **7 nhóm chủ đề lâm sàng** và **5 dạng hội thoại** (`single_turn`, `colloquial_patient`, `brand_bridge`, `multi_turn`, `safety_abstention`). |
| `evaluation/ragas_runner.py` | Module lõi gọi trực tiếp `run_clinical_agent` (hỗ trợ cả câu đơn lượt và hội thoại đa lượt có `conversation_history`), chấm 5 chỉ số RAGAS bằng Judge LLM độc lập qua NVIDIA NIM (`nvidia/nemotron-3-ultra-550b-a55b`, dự phòng `nvidia/nemotron-3.5-super-vl-120b-a12b` & `mistralai/mistral-large-4-1t-a52b`), hỗ trợ cả `models/gemini-embedding-2` (3.072 chiều) và `nvidia/nemotron-3.5-embed-8b` (4.096 chiều), xuất báo cáo Markdown/JSON. |
| `evaluation/ragas_evaluation.ipynb` | Jupyter Notebook trực quan để chạy toàn bộ quy trình đánh giá 100 ca có lưu checkpoint tự động và xuất kết quả ra file `.md` / `.json`. |
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
