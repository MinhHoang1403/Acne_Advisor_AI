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

Notebook được commit với `RUN_AUTHORIZED = False` và `CALIBRATION_REVIEW_DECISIONS = {}`. `RUN_AUTHORIZED=True` chỉ cho phép thực hiện lần đánh giá trên máy nghiên cứu; giá trị này không có nghĩa toàn bộ benchmark hoặc calibration đã được người nghiên cứu duyệt thủ công. Nếu calibration yêu cầu review, hãy đọc bằng chứng hiển thị và điền quyết định `"approve"` hoặc `"reject"` cho từng item. Sau khi hoàn tất, có thể lưu notebook đã chạy làm bằng chứng cục bộ cho báo cáo, nhưng không ghi đè thư mục baseline và không commit trạng thái ủy quyền/output nếu chưa có kế hoạch riêng.
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

RUN_AUTHORIZED = False
CALIBRATION_REVIEW_DECISIONS = {}
ALLOW_MODEL_FALLBACK = True

ROOT = Path.cwd().resolve()
if ROOT.name == "evaluation":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from evaluation.formal_evaluation_support import (  # noqa: E402
    BASELINE_RESULTS_DIR,
    CALIBRATION_ADJUDICATION_PATH,
    CALIBRATION_BLOCKED,
    CALIBRATION_READY,
    CALIBRATION_RESULTS_PATH,
    CALIBRATION_REVIEW_REQUIRED,
    CASE_METRICS_PATH,
    EVALUATOR_MODEL,
    EXPECTED_BASE_SHA,
    EXPECTED_KB_BUILD_ID,
    EXPECTED_PIPELINE_FINGERPRINT,
    METRICS_SUMMARY_PATH,
    POST_IMPROVEMENT_PATHS,
    RAW_RESULTS_PATH,
    RAGCHECKER_VERSION,
    SYSTEM_UNDER_TEST_SHA,
    build_baseline_comparison,
    build_openai_batch_adapter,
    calibration_review_items,
    evaluate_calibration_runs,
    export_metrics,
    load_evaluation_artifacts,
    load_saved_calibration_results,
    negative_rejection_rate,
    require_complete_formal_run,
    run_calibration_once,
    run_formal_cases,
    resolve_calibration_review,
    save_calibration_adjudication,
    save_calibration_results,
    score_ragchecker,
    validate_benchmark,
    validate_system_under_test,
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
system_report = validate_system_under_test(manifest)
print("✓ Môi trường đánh giá đã sẵn sàng")
print(f"✓ RAGChecker {RAGCHECKER_VERSION}")
print("✓ spaCy model en_core_web_sm đã sẵn sàng")
print("✓ OpenAI API key đã được cấu hình" if os.getenv("OPENAI_API_KEY", "").strip() else "• OpenAI API key chưa được cấu hình")
print(f"Mốc tham chiếu của bộ đánh giá: {EXPECTED_BASE_SHA}")
print(f"Hệ thống được đánh giá: {SYSTEM_UNDER_TEST_SHA}")
print(f"Git HEAD khi chạy: {system_report['repository_head']}")
print(f"Pipeline fingerprint kỳ vọng: {EXPECTED_PIPELINE_FINGERPRINT}")
print(f"Kho kiến thức tham chiếu: {EXPECTED_KB_BUILD_ID}")
print(f"Benchmark SHA: {manifest['benchmark_sha256']}")
print(f"Evaluator: {EVALUATOR_MODEL}")
print(f"Thư mục kết quả: {POST_IMPROVEMENT_PATHS.directory.relative_to(ROOT)}")
print(f"Cho phép model fallback: {ALLOW_MODEL_FALLBACK}")
print(f"Reranker local được bật theo cấu hình: {os.getenv('RERANKER_ENABLED', 'false')}")
"""
        ),
        markdown("## 2. Kiểm tra bộ dữ liệu đánh giá"),
        code(
            """benchmark_report = validate_benchmark(benchmark, manifest, calibration)

print("✓ Benchmark SHA canonical phù hợp")
print("✓ Kho kiến thức đang kích hoạt phù hợp")
print("✓ Mốc hệ thống được đánh giá là ancestor của Git HEAD hiện tại")
print("✓ Không có thay đổi production-sensitive sau mốc hệ thống được đánh giá")
print("✓ Các commit chuẩn bị chỉ thuộc lớp evaluation/test")
print(f"✓ Baseline chỉ đọc: {BASELINE_RESULTS_DIR.relative_to(ROOT)}")
print(f"✓ Output hiện tại: {POST_IMPROVEMENT_PATHS.directory.relative_to(ROOT)}")
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
"""
        ),
        markdown("## 3. Kiểm tra mô hình chấm điểm"),
        code(
            """calibration_first = calibration_second = None
automatic_calibration_decision = effective_calibration_decision = None
calibration_resolution = None
evaluator_adapter = None

saved_calibration = load_saved_calibration_results(calibration)
if saved_calibration is not None:
    print("✓ Đã tìm thấy kết quả calibration đã lưu của run hiện tại.")
    print("Không gọi lại evaluator; sử dụng kết quả đã lưu để tránh chọn lại kết quả ngẫu nhiên.")
    calibration_first = saved_calibration["payload"]["run_1"]
    calibration_second = saved_calibration["payload"]["run_2"]
    automatic_calibration_decision = saved_calibration["automatic_decision"]
elif not RUN_AUTHORIZED:
    print("Chưa có calibration đã lưu và RUN_AUTHORIZED=False; không gọi evaluator.")
else:
    evaluator_adapter = build_openai_batch_adapter(EVALUATOR_MODEL)
    print("Đang kiểm tra mô hình chấm điểm, lần 1/2...")
    calibration_first = run_calibration_once(calibration, evaluator_adapter)
    print("Đang kiểm tra mô hình chấm điểm, lần 2/2...")
    calibration_second = run_calibration_once(calibration, evaluator_adapter)
    automatic_calibration_decision = evaluate_calibration_runs(
        calibration_first, calibration_second
    )
    save_calibration_results(
        calibration,
        calibration_first,
        calibration_second,
        automatic_calibration_decision,
    )

if automatic_calibration_decision is not None:
    print(
        f"• Claim extraction phù hợp: "
        f"{automatic_calibration_decision['claim_extraction_acceptable']}/8"
    )
    print(
        f"• Claim checking thống nhất: "
        f"{automatic_calibration_decision['claim_checking_agreement']}/12"
    )
    print(
        f"• Kết quả lặp lại nhất quán: "
        f"{automatic_calibration_decision['repeat_consistency']}/20"
    )
    print(f"• Automatic calibration decision: {automatic_calibration_decision['decision']}")

    review_evidence = calibration_review_items(calibration, automatic_calibration_decision)
    if review_evidence:
        print("Calibration cần người nghiên cứu đối chiếu thủ công.")
        for item in review_evidence:
            print(f"\\nItem ID: {item['item_id']}")
            print(f"Type: {item['type']}")
            print(f"Automatic reason: {item['automatic_reasons']}")
            print(f"Run 1 status/output: {item['run_1_result']}")
            print(f"Run 2 status/output: {item['run_2_result']}")
            print(f"Expected/reference information: {item['reference_information']}")
        print(
            "Sau khi đối chiếu, điền từng item vào CALIBRATION_REVIEW_DECISIONS "
            "với giá trị 'approve' hoặc 'reject', rồi Run All lại."
        )

    calibration_resolution = resolve_calibration_review(
        automatic_calibration_decision,
        CALIBRATION_REVIEW_DECISIONS,
    )
    effective_calibration_decision = {
        **automatic_calibration_decision,
        "automatic_decision": automatic_calibration_decision["decision"],
        "decision": calibration_resolution["effective_decision"],
        "blocked": calibration_resolution["effective_decision"] == CALIBRATION_BLOCKED,
        "requires_researcher_review": (
            calibration_resolution["effective_decision"] == CALIBRATION_REVIEW_REQUIRED
        ),
        "formal_run_allowed": calibration_resolution["formal_run_allowed"],
        "researcher_adjudication": calibration_resolution["researcher_adjudication"],
    }
    if (
        automatic_calibration_decision["decision"] == CALIBRATION_REVIEW_REQUIRED
        and CALIBRATION_REVIEW_DECISIONS
    ):
        save_calibration_adjudication(
            calibration,
            automatic_calibration_decision,
            calibration_resolution,
        )
        print(f"✓ Đã lưu adjudication riêng tại: {CALIBRATION_ADJUDICATION_PATH}")

    print(f"Automatic calibration decision: {calibration_resolution['automatic_decision']}")
    print(f"Researcher adjudication: {calibration_resolution['researcher_adjudication']}")
    print(f"Effective calibration decision: {calibration_resolution['effective_decision']}")
    if calibration_resolution["unresolved_item_ids"]:
        print(f"Các item chưa có quyết định: {calibration_resolution['unresolved_item_ids']}")

    if (
        RUN_AUTHORIZED
        and calibration_resolution["formal_run_allowed"]
        and evaluator_adapter is None
    ):
        evaluator_adapter = build_openai_batch_adapter(EVALUATOR_MODEL)
"""
        ),
        markdown("## 4. Chạy hệ thống, chấm điểm và xuất kết quả"),
        code(
            """raw_results = rag_results = None
case_metric_rows = metric_summary_rows = None
nrr_score = nrr_correct = None

if not RUN_AUTHORIZED:
    print("Chưa chạy 100 tình huống vì RUN_AUTHORIZED=False.")
elif (
    effective_calibration_decision is None
    or effective_calibration_decision["decision"] != CALIBRATION_READY
):
    print("Chưa chạy 100 tình huống vì mô hình chấm điểm chưa sẵn sàng.")
else:
    raw_results = await run_formal_cases(
        benchmark,
        manifest["benchmark_sha256"],
        run_authorized=RUN_AUTHORIZED,
        calibration_decision=effective_calibration_decision,
        allow_model_fallback=ALLOW_MODEL_FALLBACK,
    )
    require_complete_formal_run(raw_results, manifest["benchmark_sha256"])
    print("✓ Hoàn tất 100 tình huống")

    if evaluator_adapter is None:
        raise RuntimeError("Evaluator adapter must be available before RAGChecker scoring.")
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
    result_lines = [
        "| Metric | N cases | Post-improvement score (%) | Diễn giải |",
        "|---|---:|---:|---|",
    ]
    for row in metric_summary_rows:
        result_lines.append(
            f"| {row['Metric']} | {row['N cases']} | {row['Score']:.4f} | {explanations[row['Metric']]} |"
        )
    display(Markdown("### Năm chỉ số đánh giá post-improvement\\n\\n" + "\\n".join(result_lines)))

    comparison_rows = build_baseline_comparison(metric_summary_rows)
    if comparison_rows is None:
        print(
            "Không tìm thấy metrics_summary.csv của Formal Run baseline trên máy hiện tại; "
            "bỏ qua bảng so sánh nhưng vẫn hiển thị kết quả post-improvement."
        )
    else:
        comparison_lines = [
            "| Metric | N | Formal Run baseline (%) | Post-improvement (%) | Chênh lệch điểm % |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in comparison_rows:
            comparison_lines.append(
                f"| {row['Metric']} | {row['N']} | {row['Formal Run baseline (%)']:.4f} | "
                f"{row['Post-improvement (%)']:.4f} | {row['Chênh lệch điểm %']:+.4f} |"
            )
        display(Markdown("### So sánh với Formal Run baseline\\n\\n" + "\\n".join(comparison_lines)))

    print("Các file post-improvement:")
    for path in (
        RAW_RESULTS_PATH,
        CASE_METRICS_PATH,
        METRICS_SUMMARY_PATH,
        CALIBRATION_RESULTS_PATH,
        CALIBRATION_ADJUDICATION_PATH,
        POST_IMPROVEMENT_PATHS.ragchecker_checkpoint,
    ):
        print("•", path)
    print("Thư mục Formal Run baseline (chỉ đọc):", BASELINE_RESULTS_DIR)
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
