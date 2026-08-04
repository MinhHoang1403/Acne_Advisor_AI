# Evaluation datasets

Canonical final benchmark: `acne_rag_eval_comprehensive_v1.jsonl`.

Dataset canonical co 300 case, 15 category can bang va schema route-aware. Dataset `acne_rag_eval_set.jsonl` va `acne_rag_eval_generation_focused.jsonl` duoc giu de so sanh lich su; chung khong phai benchmark chinh thuc cho bao cao cuoi.

Tao lai va kiem tra dataset canonical:

```powershell
.\venv\Scripts\python.exe notebooks\eval_data\build_comprehensive_eval_set.py
.\venv\Scripts\python.exe scripts\validate_comprehensive_eval_set.py
```
