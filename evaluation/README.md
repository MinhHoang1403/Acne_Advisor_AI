# Evaluation V3

Đây là framework đánh giá canonical duy nhất của Acne Advisor AI. Nó không dùng
notebook và chạy theo ba CLI stage độc lập:

```powershell
.\venv\Scripts\python.exe scripts\validate_final_evaluation_v3.py
.\venv\Scripts\python.exe scripts\run_final_evaluation_v3.py live --bypass-cache --no-persistence --checkpoint
.\venv\Scripts\python.exe scripts\run_final_evaluation_v3.py judge --resume-latest --provider gemini --model gemini-3.1-flash-lite --checkpoint --retry-transient
.\venv\Scripts\python.exe scripts\run_final_evaluation_v3.py finalize --resume-latest
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
