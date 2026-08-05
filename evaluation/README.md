# Evaluation V3

Đây là framework đánh giá canonical duy nhất của Acne Advisor AI. Nó không dùng
notebook và chạy theo ba CLI stage độc lập:

```powershell
.\venv\Scripts\python.exe scripts\validate_final_evaluation_v3.py
.\venv\Scripts\python.exe scripts\run_final_evaluation_v3.py live --run-dir reports\evaluation\<run-id> --bypass-cache --no-persistence --checkpoint
.\venv\Scripts\python.exe scripts\run_final_evaluation_v3.py judge --run-dir reports\evaluation\<run-id> --provider gemini --model gemini-3.1-flash-lite --judge-limit 3 --checkpoint --retry-transient
.\venv\Scripts\python.exe scripts\run_final_evaluation_v3.py judge --run-dir reports\evaluation\<run-id> --provider gemini --model gemini-3.1-flash-lite --judge-limit 300 --checkpoint --retry-transient
.\venv\Scripts\python.exe scripts\run_final_evaluation_v3.py finalize --run-dir reports\evaluation\<run-id>
```

`live --case-id v3_pregnancy_lactation_015` chạy replay có checkpoint cho một case
cụ thể. Với smoke stratified, dùng `--question-limit 15`, `45` hoặc `75`.

Live evaluation gọi trực tiếp `run_clinical_agent` với `evaluation_mode=True` và
`bypass_cache=True`; vì vậy không dùng API public, không đọc/ghi Redis cache và
không ghi PostgreSQL conversation history. Judge chỉ đọc các response đã checkpoint,
không gọi Ollama hoặc regenerate câu trả lời.

Dataset có đúng 300 case, 15 nhóm, mỗi nhóm 20 case. Mọi artifact run được đặt trong
`reports/evaluation/` và bị Git ignore; chỉ framework, dataset, schema và tests được
commit.

## Checkpoint judge và resume

Official run dùng một `--run-dir` duy nhất cho live, smoke judge, full judge và
finalize. `--resume-latest` vẫn còn để tương thích cho thao tác cũ, nhưng không dùng
cho báo cáo chính thức.

`--judge-limit N` là tổng số case canonical tối đa cần được judge trong lần gọi hiện
tại, không phải số API call bổ sung. Vì vậy smoke `--judge-limit 3` ghi ba case thành
công và giữ trạng thái `judge_in_progress`; lệnh `--judge-limit 300` cùng run dir sẽ
bỏ qua ba case đó và chỉ judge các case còn thiếu.

Semantic resume fingerprint khóa dataset hash/schema, metrics, rubric, live/judge
provider-model, target live cases, cache mode và persistence mode. Thay đổi
`judge_limit`, retry tuning, checkpoint flag hoặc stage command không làm thay đổi
semantic compatibility; các giá trị này vẫn được lưu trong invocation metadata.

`judge_completed` chỉ được ghi khi toàn bộ canonical target có các case ID duy nhất
với kết quả `success` và không có final error. Finalize từ chối judge partial hoặc có
unresolved error. Checkpoint schema hiện là `evaluation_checkpoint_v2`; checkpoint
cũ không tương thích được báo lỗi rõ ràng thay vì được đọc ngầm.
