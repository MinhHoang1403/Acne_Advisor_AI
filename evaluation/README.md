<!-- markdownlint-configure-file
{"MD013":{"tables":false,"code_blocks":false}}
-->

# Môi trường notebook đánh giá

Dùng Python 3.11.9 và hai môi trường riêng. Ứng dụng trong `.venv` khóa
`sentence-transformers==5.6.0` / `transformers==5.15.1`; RAGChecker 0.1.9 phụ
thuộc `refchecker`, cần `transformers<5`. Hai bộ thư viện này không thể cài
chung theo các phiên bản đã khóa.

Chạy từ thư mục gốc dự án trên Windows:

```powershell
py -3.11 -m venv evaluation/.venv
evaluation/.venv/Scripts/python.exe -m pip install --upgrade pip==26.1.2
evaluation/.venv/Scripts/python.exe -m pip install -r evaluation/requirements.lock.txt
evaluation/.venv/Scripts/python.exe -m ipykernel install --user --name acne-advisor-evaluation --display-name "Acne Advisor AI (evaluation)"
evaluation/.venv/Scripts/python.exe -m jupyterlab --notebook-dir=.
```

Trong Antigravity, VS Code hoặc Jupyter, chọn kernel **Acne Advisor AI
(evaluation)** để chạy `evaluation/environment_check.ipynb`, phân tích kết quả
đã lưu và dùng RAGChecker. Notebook kiểm tra môi trường không gọi API, chạy
benchmark hoặc ghi đè kết quả nghiên cứu. `openai==2.38.0` và
`ragchecker==0.1.9` giữ theo `thesis_requirements.txt`. Lock file cũng cài model
spaCy `en_core_web_sm==3.8.0`.

Kernel **Acne Advisor AI (runtime)** dùng `.venv` ở gốc dự án cho các đoạn mã
chạy Agent/reranker. Đăng ký lại nếu cần:

```powershell
.venv/Scripts/python.exe -m pip install ipykernel -c requirements.lock.txt
.venv/Scripts/python.exe -m ipykernel install --user --name acne-advisor-runtime --display-name "Acne Advisor AI (runtime)"
```

Không cài `requirements.lock.txt` của ứng dụng vào `evaluation/.venv`. Thu thập
output bằng môi trường runtime; chấm output đã lưu bằng môi trường evaluation.
Notebook `formal_evaluation.ipynb` và `Dockerfile.thesis` giữ quy trình nghiên
cứu cũ, vốn giả định một môi trường chung. Notebook còn kiểm tra commit,
fingerprint và KB lịch sử; khôi phục thư viện không thay đổi các điều kiện đó.
Không chạy toàn bộ notebook lịch sử để kiểm tra cài đặt; dùng
`environment_check.ipynb`.

Kiểm tra xung đột thư viện:

```powershell
evaluation/.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m pip check
```

Khi chủ động cập nhật dependency đánh giá, cài công cụ `uv` rồi tạo lại lock
bằng:

```powershell
uv pip compile evaluation/requirements.txt --python-version 3.11.9 --python-platform x86_64-pc-windows-msvc --output-file evaluation/requirements.lock.txt
```
