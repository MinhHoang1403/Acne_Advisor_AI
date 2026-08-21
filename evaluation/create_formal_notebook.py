"""Create the researcher-facing evaluation notebook without executing any cell."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "evaluation" / "formal_evaluation.ipynb"


def markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict[str, Any]:
    cells = [
        markdown(
            """# Đánh giá hệ thống Acne Advisor AI

Notebook này tổ chức quy trình đánh giá trên 100 tình huống đã được xây dựng từ kho kiến thức tham chiếu của nghiên cứu. Quy trình sử dụng implementation chính thức của [RAGChecker (NeurIPS 2024, arXiv:2408.08067)](https://arxiv.org/abs/2408.08067) cho bốn chỉ số RAG. **Negative Rejection Rate** là phép đo structured-action lấy cảm hứng từ nghiên cứu [RGB (AAAI 2024)](https://doi.org/10.1609/aaai.v38i16.29728), không phải công thức nguyên bản của RGB.

Kết quả phản ánh hiệu năng của hệ thống trên bộ dữ liệu và kho kiến thức được sử dụng trong nghiên cứu; không được diễn giải như một đánh giá hoặc chứng nhận lâm sàng.
"""
        ),
        markdown(
            """## Thuật ngữ

| Tên kỹ thuật | Giải thích tiếng Việt |
|---|---|
| Benchmark | Bộ 100 tình huống dùng để đánh giá hệ thống. |
| Case | Một tình huống hoặc câu hỏi trong bộ đánh giá. |
| Gold answer | Đáp án tham chiếu. |
| Gold claim | Mệnh đề thông tin tham chiếu có nguồn đối chiếu. |
| Evidence | Bằng chứng từ kho kiến thức. |
| Evidence snippet | Đoạn văn ngắn trực tiếp hỗ trợ mệnh đề tham chiếu. |
| Evidence gap | Trường hợp kho kiến thức chưa đủ bằng chứng cho yêu cầu cụ thể. |
| Provenance | Thông tin nguồn dùng để đối chiếu. |
| Calibration | Bước kiểm tra mô hình chấm điểm trước khi dùng cho kết quả thật. |
| Evaluator | Mô hình hỗ trợ chấm điểm. |
| Raw results | Kết quả thô của từng tình huống trước khi tổng hợp chỉ số. |
| Checkpoint | Điểm lưu tiến trình để có thể tiếp tục nếu lần chạy bị gián đoạn. |
| RAGChecker | Framework dùng để tính bốn chỉ số đánh giá RAG. |
| Negative Rejection Rate | Tỷ lệ hệ thống từ chối đúng khi thiếu bằng chứng. |
"""
        ),
        markdown(
            """## Quy trình

1. Kiểm tra bộ dữ liệu đánh giá.
2. Ghi nhận xác nhận của người nghiên cứu.
3. Kiểm tra mô hình chấm điểm.
4. Cho Acne Advisor AI xử lý 100 tình huống.
5. Tính năm chỉ số và xuất kết quả.

Ở lần kiểm tra đầu tiên, giữ `RESEARCHER_REVIEW_APPROVED = False` và chạy toàn bộ notebook để xem cấu trúc dữ liệu. Chỉ sau khi thực sự hoàn tất việc đối chiếu, người nghiên cứu mới đổi biến này thành `True` rồi chạy lại toàn bộ notebook.
"""
        ),
        markdown("## 1. Cấu hình, môi trường và dữ liệu"),
        code(
            """from pathlib import Path
import importlib.metadata
import os
import sys

from IPython.display import Markdown, display
from dotenv import load_dotenv

RESEARCHER_REVIEW_APPROVED = False
ALLOW_MODEL_FALLBACK = True

ROOT = Path.cwd().resolve()
if ROOT.name == "evaluation":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from evaluation.formal_evaluation_support import (  # noqa: E402
    CALIBRATION_BLOCKED,
    CALIBRATION_READY,
    CALIBRATION_RESULTS_PATH,
    CALIBRATION_REVIEW_REQUIRED,
    CASE_METRICS_PATH,
    EVALUATOR_MODEL,
    METRICS_SUMMARY_PATH,
    RAW_RESULTS_PATH,
    RAGCHECKER_VERSION,
    build_openai_batch_adapter,
    evaluate_calibration_runs,
    export_metrics,
    load_evaluation_artifacts,
    negative_rejection_rate,
    require_complete_formal_run,
    run_calibration_once,
    run_formal_cases,
    save_calibration_results,
    score_ragchecker,
    validate_baseline,
    validate_benchmark,
    vietnamese_analysis,
)

required_packages = {"ragchecker": RAGCHECKER_VERSION, "openai": None, "spacy": None}
missing = []
for package, expected in required_packages.items():
    try:
        actual = importlib.metadata.version(package)
        if expected is not None and actual != expected:
            missing.append(f"{package}=={expected} (hiện có {actual})")
    except importlib.metadata.PackageNotFoundError:
        missing.append(package)

try:
    import spacy
    spacy.load("en_core_web_sm")
except Exception:
    missing.append("en_core_web_sm")

if missing:
    raise RuntimeError(
        "Thiếu thành phần phục vụ đánh giá: " + ", ".join(sorted(set(missing)))
        + ". Cài evaluation/requirements.txt và mô hình spaCy en_core_web_sm trước khi tiếp tục."
    )

benchmark, manifest, calibration = load_evaluation_artifacts()
print("✓ Môi trường đánh giá đã sẵn sàng")
print(f"✓ RAGChecker {RAGCHECKER_VERSION}")
print("✓ spaCy model en_core_web_sm đã sẵn sàng")
print("✓ OpenAI API key đã được cấu hình" if os.getenv("OPENAI_API_KEY", "").strip() else "• OpenAI API key chưa được cấu hình")
print(f"Phiên bản hệ thống: {manifest['evaluation_base_sha']}")
print(f"Kho kiến thức tham chiếu: {manifest['active_kb_build_id']}")
print(f"Benchmark: {manifest['benchmark_counts']['total']} tình huống")
print(f"Evaluator: {EVALUATOR_MODEL}")
"""
        ),
        markdown("## 2. Kiểm tra bộ dữ liệu đánh giá"),
        code(
            """baseline_report = validate_baseline(manifest)
benchmark_report = validate_benchmark(benchmark, manifest, calibration)

print("✓ Baseline và kho kiến thức tham chiếu phù hợp")
print("✓ Cấu trúc benchmark hợp lệ")
print(f"• Tổng số: {benchmark_report['total']}")
print(f"• Có đáp án tham chiếu: {benchmark_report['answerable']}")
print(f"• Thiếu bằng chứng: {benchmark_report['evidence_gap']}")
print(f"• Đơn lượt: {benchmark_report['family_counts']['answerable_single_turn']}")
print(f"• Đa lượt: {benchmark_report['family_counts']['answerable_multi_turn']}")
print(f"• Tính toàn vẹn nguồn: {benchmark_report['provenance_reference_integrity']}")
print(f"• Trạng thái duyệt mệnh đề tham chiếu: {benchmark_report['gold_semantic_source_review']}")
print(f"• Tìm kiếm ứng viên cho evidence gap: {benchmark_report['evidence_gap_candidate_searches']}")
print(f"• Kiểm tra calibration: {calibration['counts']['claim_extraction']} extraction + {calibration['counts']['claim_checking']} checking")

category_lines = ["| Nhóm tình huống | Số lượng |", "|---|---:|"]
category_lines.extend(f"| {name} | {count} |" for name, count in benchmark_report["category_counts"].items())
display(Markdown("### Phân bố tình huống\\n\\n" + "\\n".join(category_lines)))

def _table_text(value):
    return " ".join(str(value).split()).replace("|", "/")

review_lines = [
    "| Case | Câu hỏi | Yêu cầu chưa có đủ bằng chứng | Đoạn ứng viên cần xem | Trạng thái |",
    "|---|---|---|---|---|",
]
for row in benchmark_report["evidence_gap_review_rows"]:
    snippets = " / ".join(_table_text(item) for item in row["top_candidate_snippets"])
    status = "Chờ người nghiên cứu xác nhận" if row["absence_review_status"] == "pending_researcher_review" else row["absence_review_status"]
    review_lines.append(
        f"| {row['case_id']} | {_table_text(row['query'])} | "
        f"{_table_text(row['unsupported_requirement'])} | {snippets} | {status} |"
    )
display(Markdown("### Bảng đối chiếu 30 trường hợp thiếu bằng chứng\\n\\n" + "\\n".join(review_lines)))
"""
        ),
        markdown("## 3. Kiểm tra mô hình chấm điểm"),
        code(
            """calibration_first = calibration_second = calibration_decision = None
evaluator_adapter = None

if not RESEARCHER_REVIEW_APPROVED:
    print("Chưa chạy kiểm tra mô hình chấm điểm vì bộ dữ liệu chưa được người nghiên cứu xác nhận.")
else:
    evaluator_adapter = build_openai_batch_adapter(EVALUATOR_MODEL)
    print("Đang kiểm tra mô hình chấm điểm, lần 1/2...")
    calibration_first = run_calibration_once(calibration, evaluator_adapter)
    print("Đang kiểm tra mô hình chấm điểm, lần 2/2...")
    calibration_second = run_calibration_once(calibration, evaluator_adapter)
    calibration_decision = evaluate_calibration_runs(calibration_first, calibration_second)
    save_calibration_results(calibration, calibration_first, calibration_second, calibration_decision)

    print(f"• Claim extraction phù hợp: {calibration_decision['claim_extraction_acceptable']}/8")
    print(f"• Claim checking thống nhất: {calibration_decision['claim_checking_agreement']}/12")
    print(f"• Kết quả lặp lại nhất quán: {calibration_decision['repeat_consistency']}/20")
    if calibration_decision["disagreements"]:
        print("• Các điểm cần người nghiên cứu xem lại:")
        for disagreement in calibration_decision["disagreements"]:
            print("  -", disagreement)

    decision = calibration_decision["decision"]
    if decision == CALIBRATION_READY:
        print("✓ Mô hình chấm điểm sẵn sàng cho lần chạy đánh giá chính thức")
    elif decision == CALIBRATION_REVIEW_REQUIRED:
        print("Chưa thể tiếp tục: kết quả calibration cần được người nghiên cứu xem lại.")
    elif decision == CALIBRATION_BLOCKED:
        print("Chưa thể tiếp tục: calibration phát hiện lỗi ngữ nghĩa nghiêm trọng được lặp lại.")
"""
        ),
        markdown("## 4. Chạy hệ thống, chấm điểm và xuất kết quả"),
        code(
            """raw_results = rag_results = None
case_metric_rows = metric_summary_rows = None
nrr_score = nrr_correct = None

if not RESEARCHER_REVIEW_APPROVED:
    print("Chưa chạy 100 tình huống vì bộ dữ liệu chưa được người nghiên cứu xác nhận.")
elif calibration_decision is None or calibration_decision["decision"] != CALIBRATION_READY:
    print("Chưa chạy 100 tình huống vì mô hình chấm điểm chưa sẵn sàng.")
else:
    raw_results = await run_formal_cases(
        benchmark,
        manifest["benchmark_sha256"],
        researcher_review_approved=RESEARCHER_REVIEW_APPROVED,
        calibration_decision=calibration_decision,
        allow_model_fallback=ALLOW_MODEL_FALLBACK,
    )
    require_complete_formal_run(raw_results, manifest["benchmark_sha256"])
    print("✓ Hoàn tất 100 tình huống")

    rag_results = score_ragchecker(benchmark, raw_results, evaluator_adapter)
    print("✓ Hoàn tất RAGChecker")

    nrr_score, nrr_correct = negative_rejection_rate(raw_results)
    print(f"✓ Hoàn tất Negative Rejection Rate: {nrr_correct}/30 = {nrr_score:.4f}%")

    case_metric_rows, metric_summary_rows = export_metrics(benchmark, raw_results, rag_results)
    print("✓ Đã xuất các file kết quả")
"""
        ),
        markdown("## 5. Kết quả, phân tích và kết luận"),
        code(
            """if metric_summary_rows is None:
    print("Chưa có kết quả định lượng. Hãy hoàn tất bước xác nhận trước khi chạy đánh giá.")
else:
    explanations = {
        "Claim Recall": "Mức độ hệ thống tìm đủ các bằng chứng cần thiết.",
        "Context Precision": "Mức độ các đoạn được truy hồi thực sự liên quan.",
        "Faithfulness": "Mức độ câu trả lời bám sát bằng chứng.",
        "Claim F1": "Mức độ câu trả lời bao quát và khớp với đáp án tham chiếu.",
        "Negative Rejection Rate": "Tỷ lệ hệ thống từ chối phù hợp khi kho kiến thức chưa đủ bằng chứng.",
    }
    result_lines = ["| Metric | N cases | Score (%) | Diễn giải |", "|---|---:|---:|---|"]
    for row in metric_summary_rows:
        result_lines.append(
            f"| {row['Metric']} | {row['N cases']} | {row['Score']:.4f} | {explanations[row['Metric']]} |"
        )
    display(Markdown("### Năm chỉ số đánh giá\\n\\n" + "\\n".join(result_lines)))

    print("Các file kết quả:")
    for path in (RAW_RESULTS_PATH, CASE_METRICS_PATH, METRICS_SUMMARY_PATH, CALIBRATION_RESULTS_PATH):
        print("•", path)
    print("\\nPhân tích:")
    print(vietnamese_analysis(metric_summary_rows))
    print("\\nKết luận:")
    print(
        "Kết quả trên mô tả chất lượng truy hồi, sinh câu trả lời và khả năng từ chối "
        "khi thiếu bằng chứng trên bộ dữ liệu của nghiên cứu."
    )
    print("\\nHạn chế:")
    print("• Đáp án tham chiếu phản ánh kho kiến thức được sử dụng trong nghiên cứu, không đại diện toàn bộ y văn.")
    print("• Các trường hợp thiếu bằng chứng chỉ được sử dụng sau khi người nghiên cứu xác nhận kết quả đối chiếu.")
    print("• RAGChecker sử dụng evaluator LLM nên calibration không loại bỏ hoàn toàn sai số chấm điểm.")
    print("• NRR là RGB-inspired structured-action adaptation, không phải metric RGB nguyên bản.")
    print("• Kết quả không phải đánh giá hoặc chứng nhận lâm sàng.")
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (acne-agent-system)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
