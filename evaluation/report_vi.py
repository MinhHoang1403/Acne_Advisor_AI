"""One concise Vietnamese final report for Evaluation V3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import FINAL_REPORT_NAME


def _metric(metric: dict[str, Any]) -> str:
    value = metric.get("value") if isinstance(metric, dict) else None
    denominator = metric.get("denominator") if isinstance(metric, dict) else None
    if value is None:
        return "Không áp dụng"
    return f"{value:.2f}%" + (f" ({denominator} case)" if denominator is not None else "")


def render_report(
    report_dir: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    judge_summary: dict[str, Any],
    results: list[dict[str, Any]],
    plots: list[str],
) -> Path:
    reliability = metrics["reliability"]
    retrieval = metrics["retrieval_and_grounding"]
    answer = metrics["answer_quality"]
    grounding = metrics.get("grounding_and_answerability", {})
    safety = metrics["safety_and_scope"]
    instruction = metrics["instruction_format_conversation"]
    naturalness = metrics["naturalness_user_experience"]
    performance = metrics["performance"]
    failures = [row for row in results if row.get("failure_reasons")][:12]
    status = "PASS" if metrics["hard_gates_passed"] and judge_summary.get("judge_final_errors") == 0 else "NOT PASS"
    lines = [
        "# BÁO CÁO ĐÁNH GIÁ TOÀN DIỆN ACNE ADVISOR AI — V3",
        "",
        "## 1. Kết luận chính",
        f"- Trạng thái deterministic: **{'PASS' if metrics['hard_gates_passed'] else 'NOT PASS'}**; trạng thái kết hợp judge: **{status}**.",
        "- Đây là đánh giá kỹ thuật của hệ thống RAG, không phải clinical validation hay thay thế đánh giá của chuyên gia y tế.",
        "",
        "## 2. Phạm vi và cấu hình",
        f"- Dataset SHA-256: `{manifest['dataset_sha256']}`; metrics: `{manifest['metrics_version']}`; rubric: `{manifest['judge_rubric_version']}`.",
        f"- Live: `{manifest['live_provider']}` / `{manifest['live_model']}`. Judge: `{manifest['judge_provider']}` / `{manifest['judge_model']}`.",
        "- Evaluation gọi agent nội bộ với persistence/cache reads/cache writes đều tắt.",
        "",
        "## 3. Mức bao phủ của bộ 300 câu hỏi",
        "- 15 nhóm, mỗi nhóm 20 case: kiến thức, hoạt chất, entity/alias, so sánh, routine, điều trị, nguồn, graph, hội thoại, format, kháng sinh, thai kỳ, kích ứng nhẹ, cấp cứu và ngoài phạm vi.",
        "",
        "## 4. Bảng điểm tổng hợp",
        "| Chỉ số | Kết quả |",
        "|---|---:|",
        f"| Request thành công | {_metric(reliability['request_success_rate'])} |",
        f"| Provenance provider | {_metric(reliability['provider_provenance_rate'])} |",
        f"| Khớp hành vi | {_metric(metrics.get('behavior_match_rate', {}))} |",
        f"| Điểm judge trung bình | {judge_summary.get('average_score_1_to_5')} / 5 |",
        "",
        "## 5. Chất lượng truy xuất và bám nguồn",
        f"- Source hit: {_metric(retrieval['source_hit_rate'])}; traceability: {_metric(retrieval['source_traceability_validity'])}; entity hit: {_metric(retrieval['entity_hit_rate'])}; alias: {_metric(retrieval['alias_resolution_accuracy'])}.",
        "- Không báo MRR/nDCG vì bộ ground truth không có nhãn thứ hạng/độ liên quan phù hợp.",
        "",
        "## 6. Chất lượng câu trả lời",
        f"- Concept recall: {answer['concept_recall']['value']}%; entity preservation: {_metric(answer['entity_preservation'])}; polarity: {_metric(answer['polarity_accuracy'])}; comparison: {_metric(answer['comparison_completeness'])}.",
        f"- Forbidden claims: {answer['forbidden_claim_count']}; source requirement: {_metric(answer['source_requirement_pass'])}.",
        f"- Source name không thuộc allowlist: {grounding.get('invalid_source_name_count', 0)}; fallback không cần thiết: {grounding.get('unnecessary_fallback_count', 0)}; direct answer first: {_metric(grounding.get('direct_answer_first_rate', {}))}.",
        "",
        "## 7. An toàn và xử lý ngoài phạm vi",
        f"- Cấp cứu: {_metric(safety['emergency_first_action_accuracy'])}; thai kỳ: {_metric(safety['pregnancy_safety_pass'])}; kháng sinh: {_metric(safety['antibiotic_stewardship_pass'])}.",
        f"- False emergency: {_metric(safety['false_emergency_escalation_rate'])}; OOD precision/recall: {_metric(safety['ood_precision'])} / {_metric(safety['ood_recall'])}.",
        "",
        "## 8. Định dạng, hội thoại và tính tự nhiên",
        f"- Format: {_metric(instruction['format_pass_rate'])}; exact count: {_metric(instruction['exact_count_pass'])}; multi-turn: {_metric(instruction['multi_turn_context_accuracy'])}.",
        f"- Disclaimer lặp: {_metric(naturalness['repeated_disclaimer_rate'])}; lộ lỗi nội bộ: {_metric(naturalness['internal_error_leakage_rate'])}; văn phong phán xét: {_metric(naturalness['judgmental_wording_rate'])}.",
        "",
        "## 9. Hiệu năng và độ ổn định",
        f"- Latency trung bình/P50/P95/P99: {performance['average_latency_ms']} / {performance['p50_latency_ms']} / {performance['p95_latency_ms']} / {performance['p99_latency_ms']} ms.",
        f"- Judge thành công: {judge_summary.get('judge_success_rate')}%; lỗi cuối: {judge_summary.get('judge_final_errors')}; agreement: {judge_summary.get('agreement_rate')}%.",
        "",
        "## 10. Các trường hợp chưa đạt",
        "| Case | Lý do |",
        "|---|---|",
        *([f"| `{row['case_id']}` | {', '.join(row['failure_reasons'])} |" for row in failures] or ["| Không có | Không có lỗi deterministic nổi bật |"]),
        "",
        "## 11. Kết luận",
        f"- Hard gates: **{'PASS' if metrics['hard_gates_passed'] else 'NOT PASS'}**. Các biểu đồ được tạo trong `plots/` ({len(plots)} biểu đồ).",
    ]
    path = report_dir / FINAL_REPORT_NAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = ["render_report"]
