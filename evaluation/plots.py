"""Compact, Vietnamese-labelled charts for the final V3 report."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PLOT_FILENAMES = (
    "bang_diem_tong_hop.png",
    "diem_theo_nhom_cau_hoi.png",
    "chat_luong_truy_xuat.png",
    "phan_bo_nguon_phan_hoi.png",
    "chi_so_an_toan.png",
    "diem_gemini_theo_nhom.png",
    "diem_gemini_theo_loai_phan_hoi.png",
    "do_tre_theo_loai_phan_hoi.png",
    "nguyen_nhan_case_chua_dat.png",
)


def _metric_value(metric: dict[str, Any]) -> float:
    value = metric.get("value") if isinstance(metric, dict) else None
    return float(value) if isinstance(value, (int, float)) else 0.0


def create_plots(
    report_dir: Path,
    metrics: dict[str, Any],
    results: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = report_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    def bar(filename: str, labels: list[str], values: list[float], title: str, ylabel: str = "Tỷ lệ / điểm (%)") -> str:
        figure, axis = plt.subplots(figsize=(9, 4.8))
        axis.bar(range(len(labels)), values, color="#147a73")
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=24, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, max(100, max(values, default=0) * 1.15))
        for index, value in enumerate(values):
            axis.text(index, value + 1, f"{value:.1f}", ha="center", fontsize=8)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        return str(path)

    reliability = metrics["reliability"]
    safety = metrics["safety_and_scope"]
    retrieval = metrics["retrieval_and_grounding"]
    result: list[str] = []
    result.append(
        bar(
            "bang_diem_tong_hop.png",
            ["Ổn định", "Truy xuất", "Câu trả lời", "An toàn", "Định dạng", "Tự nhiên"],
            [
                _metric_value(reliability["request_success_rate"]),
                _metric_value(retrieval["source_traceability_validity"]),
                metrics["answer_quality"]["concept_recall"]["value"] or 0,
                _metric_value(safety["pregnancy_safety_pass"]),
                _metric_value(metrics["instruction_format_conversation"]["format_pass_rate"]),
                100 - _metric_value(metrics["naturalness_user_experience"]["internal_error_leakage_rate"]),
            ],
            "Bảng điểm tổng hợp",
        )
    )
    category_scores = metrics.get("category_behavior_match_rate", {})
    result.append(bar("diem_theo_nhom_cau_hoi.png", list(category_scores), [value or 0 for value in category_scores.values()], "Mức khớp hành vi theo nhóm"))
    retrieval_items = [
        ("Hit nguồn", retrieval["source_hit_rate"]),
        ("Entity", retrieval["entity_hit_rate"]),
        ("Alias", retrieval["alias_resolution_accuracy"]),
        ("Graph", retrieval["graph_relation_hit_rate"]),
        ("Trace", retrieval["source_traceability_validity"]),
    ]
    result.append(bar("chat_luong_truy_xuat.png", [label for label, _ in retrieval_items], [_metric_value(item) for _, item in retrieval_items], "Chất lượng truy xuất và bám nguồn"))
    origins = Counter(row.get("actual_origin") for row in results)
    result.append(bar("phan_bo_nguon_phan_hoi.png", list(origins), [float(value) for value in origins.values()], "Phân bố nguồn phản hồi", "Số case"))
    safety_items = [
        ("Phát hiện cấp cứu", safety["emergency_detection_recall"]),
        ("Hành động đầu tiên", safety["emergency_first_action_accuracy"]),
        ("Thai kỳ", safety["pregnancy_safety_pass"]),
        ("Kháng sinh", safety["antibiotic_stewardship_pass"]),
        ("OOD recall", safety["ood_recall"]),
        ("Không false emergency", {"value": 100 - (_metric_value(safety["false_emergency_escalation_rate"]))}),
    ]
    result.append(bar("chi_so_an_toan.png", [label for label, _ in safety_items], [_metric_value(item) for _, item in safety_items], "Chỉ số an toàn và ngoài phạm vi"))
    judge_by_category: dict[str, list[float]] = defaultdict(list)
    judge_by_origin: dict[str, list[float]] = defaultdict(list)
    for row in judge_rows:
        if isinstance(row.get("overall_0_to_100"), (int, float)):
            judge_by_category[str(row.get("category"))].append(float(row["overall_0_to_100"]))
            judge_by_origin[str(row.get("origin"))].append(float(row["overall_0_to_100"]))
    result.append(bar("diem_gemini_theo_nhom.png", list(judge_by_category), [sum(values) / len(values) for values in judge_by_category.values()], "Điểm Gemini theo nhóm câu hỏi"))
    result.append(bar("diem_gemini_theo_loai_phan_hoi.png", list(judge_by_origin), [sum(values) / len(values) for values in judge_by_origin.values()], "Điểm Gemini theo loại phản hồi"))
    latency_by_origin: dict[str, list[float]] = defaultdict(list)
    for row in results:
        if isinstance(row.get("latency_ms"), (int, float)):
            latency_by_origin[str(row.get("actual_origin"))].append(float(row["latency_ms"]))
    result.append(bar("do_tre_theo_loai_phan_hoi.png", list(latency_by_origin), [sum(values) / len(values) for values in latency_by_origin.values()], "Độ trễ trung bình theo loại phản hồi", "ms"))
    failures = Counter(reason for row in results for reason in row.get("failure_reasons") or [])
    result.append(bar("nguyen_nhan_case_chua_dat.png", list(failures) or ["Không có"], [float(value) for value in failures.values()] or [0.0], "Nguyên nhân các case chưa đạt", "Số case"))
    return result


__all__ = ["PLOT_FILENAMES", "create_plots"]
