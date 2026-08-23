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
from collections import Counter
from contextlib import redirect_stdout
import importlib.metadata
import io
import sys

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
    CALIBRATION_BLOCKED,
    CALIBRATION_READY,
    CALIBRATION_REVIEW_REQUIRED,
    EVALUATOR_MODEL,
    EvaluationBlocked,
    EXPECTED_KB_BUILD_ID,
    POST_IMPROVEMENT_PATHS,
    POST_IMPROVEMENT_RUN_ID,
    RAGCHECKER_VERSION,
    SYSTEM_UNDER_TEST_SHA,
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
    validate_final_evaluator_model,
    validate_system_under_test,
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
preflight_benchmark_report = validate_benchmark(benchmark, manifest, calibration)
validated_evaluator = validate_final_evaluator_model(EVALUATOR_MODEL)
if POST_IMPROVEMENT_RUN_ID != "formal_run_2d5f0124":
    raise EvaluationBlocked(
        f"OFFICIAL_RESULT_ID_MISMATCH: {POST_IMPROVEMENT_RUN_ID}"
    )
if POST_IMPROVEMENT_PATHS.run_id != POST_IMPROVEMENT_RUN_ID:
    raise EvaluationBlocked("OFFICIAL_RESULT_PATH_ID_MISMATCH")

blocked_evaluator_aliases = (
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-mini",
    "gpt-5.4",
)
for blocked_model in blocked_evaluator_aliases:
    try:
        validate_final_evaluator_model(blocked_model)
    except EvaluationBlocked:
        continue
    raise EvaluationBlocked(f"EVALUATOR_ALIAS_NOT_BLOCKED: {blocked_model}")

print("OFFICIAL GPT-5.4 CAMPAIGN PREFLIGHT")
print("✓ Môi trường đánh giá đã sẵn sàng")
print("✓ Tính toàn vẹn cấu hình đánh giá đã được xác nhận")
print(f"System: {SYSTEM_UNDER_TEST_SHA}")
print(f"Pipeline: {system_report['reproduced_pipeline_fingerprint']}")
print(f"KB: {EXPECTED_KB_BUILD_ID}")
print(f"Benchmark: {preflight_benchmark_report['total']} cases")
print("Integrity: PASS")
print(f"Evaluator: {validated_evaluator}")
print("Anti-mini guard: PASS")
print(f"RAGChecker: {RAGCHECKER_VERSION}")
print(f"Official result: {POST_IMPROVEMENT_RUN_ID}")
print("Ready for researcher authorization.")
"""
        ),
        markdown("## 2. Kiểm tra bộ dữ liệu đánh giá"),
        code(
            """benchmark_report = validate_benchmark(benchmark, manifest, calibration)

print("✓ Bộ dữ liệu đánh giá hợp lệ")
print(f"Tổng số tình huống: {benchmark_report['total']}")
print(f"Có đáp án tham chiếu: {benchmark_report['answerable']}")
print(f"Thiếu bằng chứng: {benchmark_report['evidence_gap']}")
print(f"Đơn lượt: {benchmark_report['family_counts']['answerable_single_turn']}")
print(f"Đa lượt: {benchmark_report['family_counts']['answerable_multi_turn']}")
print("Tính toàn vẹn benchmark và nguồn tham chiếu: đạt")

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
    print("✓ Sử dụng kết quả calibration đã lưu của lần chạy hiện tại")
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
        print("Các điểm chưa thống nhất cần người nghiên cứu đối chiếu:")
        for item in review_evidence:
            reasons = ", ".join(item["automatic_reasons"])
            print(f"• {item['item_id']} ({item['type']}): {reasons}")
        print(
            "Sau khi đối chiếu, điền từng item vào CALIBRATION_REVIEW_DECISIONS "
            "với giá trị 'approve' hoặc 'reject', rồi Run All lại."
        )

    current_review_item_ids = {
        str(item.get("item_id") or "")
        for item in automatic_calibration_decision.get("disagreements") or []
    }
    applicable_review_decisions = {
        item_id: decision
        for item_id, decision in CALIBRATION_REVIEW_DECISIONS.items()
        if item_id in current_review_item_ids
    }
    calibration_resolution = resolve_calibration_review(
        automatic_calibration_decision,
        applicable_review_decisions,
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
        and applicable_review_decisions
    ):
        save_calibration_adjudication(
            calibration,
            automatic_calibration_decision,
            calibration_resolution,
        )
        print("✓ Đã lưu quyết định đối chiếu của người nghiên cứu")

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
completed_cases = infrastructure_failure_count = fallback_cases = None
provider_summary = None

if not RUN_AUTHORIZED:
    print("Chưa chạy 100 tình huống vì RUN_AUTHORIZED=False.")
elif (
    effective_calibration_decision is None
    or effective_calibration_decision["decision"] != CALIBRATION_READY
):
    print("Chưa chạy 100 tình huống vì mô hình chấm điểm chưa sẵn sàng.")
else:
    execution_log = io.StringIO()
    try:
        with redirect_stdout(execution_log):
            raw_results = await run_formal_cases(
                benchmark,
                manifest["benchmark_sha256"],
                run_authorized=RUN_AUTHORIZED,
                calibration_decision=effective_calibration_decision,
                allow_model_fallback=ALLOW_MODEL_FALLBACK,
            )
    except Exception:
        print(execution_log.getvalue())
        raise

    infrastructure_failures = [
        record for record in raw_results["records"] if record.get("infrastructure_error")
    ]
    if infrastructure_failures:
        print("Chi tiết lỗi hạ tầng:")
        for record in infrastructure_failures:
            print(f"• {record['case_id']}: {record['infrastructure_error']}")
    require_complete_formal_run(raw_results, manifest["benchmark_sha256"])
    fallback_cases = sum(
        bool(record.get("llm_fallback_used")) for record in raw_results["records"]
    )
    provider_counts = Counter(
        f"{record.get('actual_provider') or 'unknown'} / "
        f"{record.get('actual_model') or 'unknown'}"
        for record in raw_results["records"]
    )
    provider_summary = ", ".join(
        f"{provider_model}: {count}"
        for provider_model, count in sorted(provider_counts.items())
    )
    completed_cases = len(raw_results["records"])
    infrastructure_failure_count = len(infrastructure_failures)
    successful_cases = completed_cases - infrastructure_failure_count
    print("Formal evaluation")
    print(f"{completed_cases} / 100 completed")
    print(f"✓ Successful cases: {successful_cases}")
    print(f"✓ Infrastructure failures: {infrastructure_failure_count}")
    print(f"Production provider: {provider_summary}")
    print(f"Recovered fallback usage: {fallback_cases} cases")

    if evaluator_adapter is None:
        raise RuntimeError("Evaluator adapter must be available before RAGChecker scoring.")
    rag_results = score_ragchecker(benchmark, raw_results, evaluator_adapter)
    print("RAGChecker: ✓ completed")

    nrr_score, nrr_correct = negative_rejection_rate(raw_results)
    print(f"Negative Rejection Rate: {nrr_correct}/30 = {nrr_score:.4f}%")

    case_metric_rows, metric_summary_rows = export_metrics(benchmark, raw_results, rag_results)
    print("✓ Đã lưu các artifact kết quả")
"""
        ),
        markdown("## 5. Kết quả, phân tích và kết luận"),
        code(
            """if metric_summary_rows is None:
    print("Chưa có kết quả định lượng. Hãy hoàn tất bước xác nhận trước khi chạy đánh giá.")
else:
    headline_metrics = (
        "Claim Recall",
        "Context Precision",
        "Faithfulness",
        "Claim F1",
        "Negative Rejection Rate",
    )
    scores = {str(row["Metric"]): float(row["Score"]) for row in metric_summary_rows}
    if set(scores) != set(headline_metrics):
        raise RuntimeError(f"OFFICIAL_METRIC_SET_MISMATCH: {sorted(scores)}")

    print("OFFICIAL GPT-5.4 EVALUATION")
    for metric in headline_metrics:
        print(f"{metric}: {scores[metric]:.2f}%")
    print(f"NRR numerator: {nrr_correct} / 30")
    print(f"Completed: {completed_cases} / 100")
    print(f"Infrastructure failures: {infrastructure_failure_count}")
    print(f"Production fallback: {fallback_cases} cases; {provider_summary}")

    highest_metric = max(headline_metrics, key=scores.__getitem__)
    lowest_metric = min(headline_metrics, key=scores.__getitem__)
    recall_precision_gap = scores["Claim Recall"] - scores["Context Precision"]
    print("\\nPhân tích xác định:")
    print(f"Chỉ số cao nhất: {highest_metric} ({scores[highest_metric]:.2f}%).")
    print(f"Chỉ số thấp nhất: {lowest_metric} ({scores[lowest_metric]:.2f}%).")
    print(
        "Chênh lệch Claim Recall - Context Precision: "
        f"{recall_precision_gap:+.2f} điểm phần trăm."
    )
    print(
        "Kết quả chỉ phản ánh benchmark và kho kiến thức tham chiếu; "
        "không phải xác nhận lâm sàng."
    )
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
