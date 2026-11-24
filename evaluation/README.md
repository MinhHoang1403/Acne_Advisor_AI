# Đánh giá Hệ thống Acne Advisor AI bằng RAGAS

Thư mục này chứa quy trình đánh giá hệ thống RAG hiện tại bằng **RAGAS** (*Retrieval Augmented Generation Assessment*, Es et al., 2023, `arXiv:2309.15217`).

## 1. Cấu trúc thư mục

| Thành phần | Chức năng |
| --- | --- |
| `evaluation/dataset.json` | Bộ câu hỏi đánh giá chuẩn (50 câu hỏi tiếng Việt thuộc 7 nhóm chủ đề bám sát kho dữ liệu `Acne_data_source.json`, gồm 42 câu trả lời và 8 câu từ chối an toàn). |
| `evaluation/ragas_runner.py` | Module lõi gọi hệ thống `run_clinical_agent`, chấm 5 chỉ số RAGAS bằng Gemini (`google-genai`) và xuất báo cáo Markdown/JSON. |
| `evaluation/ragas_evaluation.ipynb` | Jupyter Notebook trực quan để chạy toàn bộ quy trình đánh giá và xuất kết quả ra file `.md`. |
| `evaluation/results/` | Thư mục chứa kết quả xuất ra: `ragas_evaluation_report.md` và `ragas_evaluation_raw.json`. |

## 2. Các chỉ số đánh giá

1. **Faithfulness (Độ trung thực với bằng chứng):** Tỷ lệ mệnh đề trong câu trả lời được chứng minh trực tiếp bởi các đoạn chứng cứ truy hồi (`retrieved_contexts`), giúp phát hiện ảo giác (hallucination).
2. **Answer Relevancy (Độ liên quan câu trả lời):** Mức độ câu trả lời bám sát đúng trọng tâm câu hỏi của người dùng (sử dụng embeddings `models/gemini-embedding-2`).
3. **Context Precision (Độ chính xác ngữ cảnh):** Kiểm tra các đoạn chứng cứ hữu ích có được xếp ở thứ hạng cao hay không.
4. **Context Recall (Độ bao phủ ngữ cảnh):** Kiểm tra ngữ cảnh truy hồi có phủ đủ thông tin trong đáp án chuẩn (`reference`) hay không.
5. **Answer Correctness (Độ chính xác câu trả lời):** Kết hợp độ chính xác sự kiện (75%) và độ tương đồng ngữ nghĩa (25%) giữa câu trả lời của hệ thống và đáp án chuẩn.
6. **Safe Abstention Accuracy (Tỷ lệ từ chối an toàn):** Đánh giá riêng các câu hỏi ngoài phạm vi hoặc thiếu bằng chứng y khoa xem hệ thống có từ chối đúng thay vì phỏng đoán hay không.

## 3. Cách chạy đánh giá

1. Đảm bảo các container Docker (`acne_qdrant`, `acne_postgres`, `acne_redis`, `acne_neo4j`) đang chạy và `.env` đã có `GOOGLE_API_KEY`.
2. Mở file `evaluation/ragas_evaluation.ipynb` bằng VS Code / Jupyter (chọn kernel `.venv` của dự án) và chạy lần lượt các cell (`Run All`).
3. Xem báo cáo chi tiết tại `evaluation/results/ragas_evaluation_report.md`.
