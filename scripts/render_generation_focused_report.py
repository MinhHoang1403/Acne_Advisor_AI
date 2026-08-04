"""Render the Vietnamese report and profile-specific charts for a completed eval run.

This utility is deliberately post-processing only: it reads the immutable run
artifacts produced by ``rag_llm_judge.ipynb`` and never calls a model, API, or
database.  Keeping it separate makes the presentation reproducible without
changing the deterministic evaluation data.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_REPORT_HEADINGS = (
    "# Báo cáo đánh giá generation-focused",
    "## Tóm tắt lần chạy",
    "## Cấu hình đánh giá",
    "## Bộ câu hỏi generation-focused",
    "## Chỉ số deterministic",
    "## Kết quả LLM-as-Judge",
    "## Phân tích origin câu trả lời",
    "## Biểu đồ kết quả",
    "## Kết quả theo nhóm câu hỏi",
    "## Các trường hợp cần xem xét",
    "## Cách đọc kết quả",
    "## Nhận xét kết quả",
    "## Giới hạn diễn giải",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_number(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _escape_table(value: Any) -> str:
    return " ".join(str(value or "").replace("|", "/").split())


def answer_origin(raw_row: dict[str, Any]) -> str:
    verification = str(raw_row.get("runtime_verification") or "").strip()
    if verification:
        return verification
    response = raw_row.get("raw_response")
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            response = {}
    metadata = response.get("metadata", {}) if isinstance(response, dict) else {}
    provider = str(metadata.get("provider") or "unknown").strip().lower()
    origin = str(metadata.get("response_origin") or "unknown").strip().lower()
    return f"{provider}_{origin}"


def create_generation_focused_charts(
    judge_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    plots_dir: Path,
) -> dict[str, Path]:
    """Create the two generation-focused charts from persisted run artifacts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "judge_pass_rate_by_category": plots_dir / "judge_pass_rate_by_category.png",
        "origin_distribution": plots_dir / "origin_distribution.png",
    }

    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in judge_rows:
        if row.get("judge_status") == "ok":
            grouped[str(row.get("category") or "uncategorized")].append(_as_bool(row.get("judge_pass")))
    labels = sorted(grouped) or ["no_data"]
    values = [round(100 * sum(grouped[label]) / len(grouped[label]), 2) for label in labels] if grouped else [0.0]
    figure, axis = plt.subplots(figsize=(10, 5.4))
    axis.bar(range(len(labels)), values, color="#2f7d6d")
    axis.set_ylim(0, 100)
    axis.set_ylabel("Judge pass rate (%)")
    axis.set_title("Tỷ lệ LLM-as-Judge pass theo nhóm câu hỏi")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    figure.tight_layout()
    figure.savefig(paths["judge_pass_rate_by_category"], dpi=160)
    plt.close(figure)

    origins = Counter(answer_origin(row) for row in raw_rows)
    origin_labels = sorted(origins) or ["no_data"]
    origin_values = [origins[label] for label in origin_labels] if origins else [0]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(range(len(origin_labels)), origin_values, color="#4c78a8")
    axis.set_ylabel("Số câu trả lời")
    axis.set_title("Phân bố origin câu trả lời")
    axis.set_xticks(range(len(origin_labels)))
    axis.set_xticklabels(origin_labels, rotation=25, ha="right")
    figure.tight_layout()
    figure.savefig(paths["origin_distribution"], dpi=160)
    plt.close(figure)
    return paths


def render_generation_focused_report(report_dir: Path) -> Path:
    results = _read_csv(report_dir / "results.csv")
    raw_rows = [json.loads(line) for line in (report_dir / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    judge_rows = _read_csv(report_dir / "judge_results.csv")
    run_config = _read_json(report_dir / "run_config.json")
    metrics_payload = _read_json(report_dir / "summary_metrics.json")
    core = {key: value for key, value in metrics_payload.items() if key != "evaluation_run_config"}
    judge_summary = _read_json(report_dir / "judge_summary.json")

    chart_paths = create_generation_focused_charts(judge_rows, raw_rows, report_dir / "plots")
    category_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in results:
        category_rows[str(row.get("category") or "uncategorized")].append(row)
    judge_by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in judge_rows:
        judge_by_category[str(row.get("category") or "uncategorized")].append(row)
    origins = Counter(answer_origin(row) for row in raw_rows)

    def rate(rows: list[dict[str, str]], key: str) -> float:
        return round(100 * sum(_as_bool(row.get(key)) for row in rows) / len(rows), 2) if rows else 0.0

    def average(rows: list[dict[str, str]], key: str) -> float:
        values = [_as_number(row.get(key)) for row in rows]
        numeric = [value for value in values if value is not None]
        return round(sum(numeric) / len(numeric), 2) if numeric else 0.0

    fallback_count = sum(value for origin, value in origins.items() if "safe_fallback" in origin)
    guardrail_count = sum(value for origin, value in origins.items() if "guardrail" in origin)
    ollama_count = sum(value for origin, value in origins.items() if origin == "ollama_llm")
    total = len(raw_rows)
    judge_successful_cases = run_config.get("judge_successful_cases", judge_summary.get("judge_successful_cases", 0))
    judge_error_cases = run_config.get("judge_error_cases", judge_summary.get("judge_error_cases", 0))
    fallback_ratio = (100 * fallback_count / total) if total else 0.0
    guardrail_ratio = (100 * guardrail_count / total) if total else 0.0
    ollama_ratio = (100 * ollama_count / total) if total else 0.0

    category_lines = [
        "| Nhóm | Số câu | Điểm deterministic TB | Judge TB | Judge pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for category in sorted(category_rows):
        result_group = category_rows[category]
        judged = [row for row in judge_by_category.get(category, []) if row.get("judge_status") == "ok"]
        category_lines.append(
            f"| `{category}` | {len(result_group)} | {average(result_group, 'overall_score'):.2f} | "
            f"{average(judged, 'judge_score_100'):.2f} | {rate(judged, 'judge_pass'):.2f}% |"
        )

    review_rows = [row for row in results if not _as_bool(row.get("overall_pass"))]
    review_lines = [
        "| Case ID | Nhóm | Lý do deterministic | Câu hỏi |",
        "|---|---|---|---|",
    ]
    for row in review_rows[:12]:
        review_lines.append(
            f"| `{_escape_table(row.get('case_id'))}` | `{_escape_table(row.get('category'))}` | "
            f"{_escape_table(row.get('failure_reasons'))} | {_escape_table(row.get('question'))} |"
        )
    if len(review_lines) == 2:
        review_lines.append("| N/A | N/A | Không có case deterministic fail | N/A |")

    report = "\n".join(
        [
            "# Báo cáo đánh giá generation-focused",
            "",
            "## Tóm tắt lần chạy",
            f"- Lần chạy: `{run_config.get('timestamp')}`.",
            f"- Đã chạy live `{len(raw_rows)}`/300 câu và judge thành công `{judge_successful_cases}`/300 câu.",
            f"- Provider/model sinh câu trả lời: `{run_config.get('live_eval_provider')}` / `{run_config.get('live_eval_model')}`.",
            f"- Provider/model judge: `{judge_summary.get('judge_provider')}` / `{judge_summary.get('judge_model')}`.",
            "",
            "## Cấu hình đánh giá",
            f"- Dataset: `acne_rag_eval_generation_focused.jsonl`.",
            "- Evaluation profile: `generation-focused`.",
            f"- Cache bypass: `{run_config.get('cache_bypass_for_live_eval')}`; runtime retry tối đa: `{run_config.get('live_eval_max_runtime_attempts')}`.",
            f"- Judge score threshold: `{run_config.get('judge_score_threshold', 70)}`.",
            "",
            "## Bộ câu hỏi generation-focused",
            "- 300 câu in-domain, thiên về giải thích, thực thể thuốc/sản phẩm, so sánh, kế hoạch điều trị và routine.",
            "- Không cố ý đưa các case ngoài miền hoặc cấp cứu để làm méo tỷ lệ origin của luồng generation thông thường.",
            "- Bộ này không thay thế bộ safety/readiness cũ; hai bộ đo hai khía cạnh khác nhau của hệ thống.",
            "",
            "## Chỉ số deterministic",
            "| Chỉ số | Giá trị |",
            "|---|---:|",
            *[f"| `{key}` | {_as_number(value) if _as_number(value) is not None else value} |" for key, value in core.items()],
            "",
            "## Kết quả LLM-as-Judge",
            f"- Judge average score: `{float(judge_summary.get('judge_avg_score') or 0):.2f}`.",
            f"- Judge pass rate: `{float(judge_summary.get('judge_pass_rate') or 0):.2f}%`.",
            f"- Agreement với deterministic score: `{float(judge_summary.get('judge_agreement_rate') or 0):.2f}%`.",
            f"- Judge error cases: `{judge_error_cases}`.",
            "",
            "## Phân tích origin câu trả lời",
            f"- Ollama LLM: `{ollama_count}` ({ollama_ratio:.2f}%).",
            f"- Safe fallback: `{fallback_count}` ({fallback_ratio:.2f}%).",
            f"- Guardrail: `{guardrail_count}` ({guardrail_ratio:.2f}%).",
            *[f"- `{origin}`: `{count}`." for origin, count in sorted(origins.items())],
            "",
            "## Biểu đồ kết quả",
            "- `plots/overall_metrics_bar.png`",
            "- `plots/category_scores.png`",
            "- `plots/latency_distribution.png`",
            "- `plots/pass_fail_breakdown.png`",
            "- `plots/top_failure_categories.png`",
            "- `plots/judge_score_by_category.png`",
            "- `plots/judge_vs_rule_score.png`",
            f"- `plots/{chart_paths['judge_pass_rate_by_category'].name}`",
            f"- `plots/{chart_paths['origin_distribution'].name}`",
            "",
            "## Kết quả theo nhóm câu hỏi",
            *category_lines,
            "",
            "## Các trường hợp cần xem xét",
            *review_lines,
            "",
            "## Cách đọc kết quả",
            "- Deterministic score kiểm tra các tiêu chí có thể lặp lại như từ khóa, nguồn, an toàn và định dạng.",
            "- LLM-as-Judge bổ sung đánh giá relevance, faithfulness, completeness, safety, instruction following và tiếng Việt rõ ràng.",
            "- Hai lớp điểm cần được đọc cùng các case cụ thể; chúng không thay thế đánh giá của chuyên gia y tế.",
            "",
            "## Nhận xét kết quả",
            "- Mục tiêu của profile là đo chất lượng sinh câu trả lời trong miền mụn, không tối ưu hóa bằng cách nới lỏng guardrail hay fallback.",
            "- Ít fallback hoặc guardrail hơn không có nghĩa là hệ thống bỏ qua an toàn; các policy safety vẫn phải được đánh giá bằng bộ safety/readiness riêng.",
            f"- Mục tiêu tham chiếu là Ollama >=75%, safe fallback <=20%, guardrail <=5%. Lần chạy này lần lượt là {ollama_ratio:.2f}%, {fallback_ratio:.2f}% và {guardrail_ratio:.2f}%; target origin không đạt và cần điều tra coverage/runtime thay vì giảm safety.",
            "- Tỷ lệ origin được báo cáo minh bạch để phát hiện khi runtime chuyển quá nhiều sang safe fallback hoặc guardrail.",
            "",
            "## Giới hạn diễn giải",
            "- Kết quả phụ thuộc snapshot KB, mô hình cục bộ, provider judge và điều kiện runtime tại thời điểm chạy.",
            "- Judge dùng Gemini là phép đo hỗ trợ, không phải chân lý lâm sàng và không được dùng thay cho thẩm định chuyên gia.",
            "- Live /chat tạo session/message audit trong PostgreSQL theo thiết kế runtime; báo cáo này không reset hoặc xóa dữ liệu đó.",
            "",
        ]
    )
    target = report_dir / "summary_report.md"
    target.write_text(report, encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a generation-focused evaluation report.")
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    target = render_generation_focused_report(args.report_dir.resolve())
    print(f"GENERATION_FOCUSED_REPORT_RENDERED: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
