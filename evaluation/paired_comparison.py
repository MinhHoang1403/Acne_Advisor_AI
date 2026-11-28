"""So sánh ghép cặp (paired comparison) hai lần chấm RAGAS trên cùng bộ ca đánh giá.

Module dùng chung cho hai mục đích:
1. Kiểm định độ tin cậy của judge: cùng câu trả lời, chấm bằng hai judge (hoặc hai lần cùng
   judge) để đo mức đồng thuận.
2. Ablation: cùng bộ câu hỏi, hai cấu hình hệ thống khác nhau, cùng một judge, để đo đóng góp
   của từng thành phần.

Phương pháp thống kê (đều là phương pháp kinh điển, không tự chế hệ số):
- Hệ số tương quan hạng Spearman (Spearman, 1904, *The proof and measurement of association
  between two things*), dùng hạng trung bình cho các giá trị trùng nhau; đo hai lần chấm có
  xếp hạng các ca giống nhau hay không.
- Kiểm định dấu (sign test) hai phía chính xác theo phân phối nhị thức (Dixon & Mood, 1946,
  *The statistical sign test*, JASA 41(236)); bỏ qua ca hòa. Kiểm định này không giả định
  phân phối chuẩn, phù hợp với điểm RAGAS dồn nhiều ở 0 và 1.

Chạy CLI:
    python -m evaluation.paired_comparison --baseline A.json --candidate B.json \
        --baseline-label "..." --candidate-label "..." --title "..." --output report.md
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
from pathlib import Path
import statistics
from typing import Any

from evaluation.ragas_runner import RAGAS_METRIC_KEYS

# Điểm RAGAS được làm tròn 4 chữ số trong runner; so sánh ở cùng độ chính xác để chênh lệch
# do làm tròn không bị tính là thắng/thua.
_SCORE_DECIMALS = 4
_TOP_DIVERGENT_CASES = 10


def load_scored_records(path: Path) -> list[dict[str, Any]]:
    """Đọc bản ghi đã chấm từ file raw JSON của runner (dict có ``records``) hoặc danh sách."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return list(data["records"])
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Không nhận diện được định dạng bản ghi chấm điểm: {path}")


def average_ranks(values: Sequence[float]) -> list[float]:
    """Gán hạng (1-indexed), các giá trị bằng nhau nhận hạng trung bình của nhóm."""

    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared_rank = (position + end) / 2.0 + 1.0
        for idx in order[position : end + 1]:
            ranks[idx] = shared_rank
        position = end + 1
    return ranks


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Tương quan Spearman = tương quan Pearson trên hạng; ``None`` nếu không xác định được."""

    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rank_x = average_ranks(xs)
    rank_y = average_ranks(ys)
    mean_x = statistics.fmean(rank_x)
    mean_y = statistics.fmean(rank_y)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y, strict=True))
    var_x = sum((a - mean_x) ** 2 for a in rank_x)
    var_y = sum((b - mean_y) ** 2 for b in rank_y)
    if var_x == 0 or var_y == 0:
        # Một phía có toàn bộ giá trị giống nhau (vd toàn 1.0) -> tương quan không xác định.
        return None
    return cov / math.sqrt(var_x * var_y)


def sign_test_p_value(wins: int, losses: int) -> float | None:
    """P-value hai phía chính xác của sign test với giả thuyết H0: P(thắng) = 0.5."""

    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _answer_records_by_case(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(rec.get("case_id")): rec
        for rec in records
        if rec.get("expected_behavior") == "answer" and rec.get("case_id")
    }


def _score(rec: dict[str, Any], key: str) -> float | None:
    value = (rec.get("ragas_scores") or {}).get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), _SCORE_DECIMALS)


def compare_scored_runs(
    baseline_records: Sequence[dict[str, Any]],
    candidate_records: Sequence[dict[str, Any]],
    *,
    metric_keys: Sequence[str] = RAGAS_METRIC_KEYS,
    group_field: str = "dialogue_type",
) -> dict[str, Any]:
    """So sánh ghép cặp theo ``case_id`` trên các ca ``expected_behavior == "answer"``.

    Chênh lệch luôn tính là ``candidate - baseline``. Ca thiếu điểm ở một phía bị loại khỏi
    cặp của metric đó (số cặp thực tế được báo trong ``n``).
    """

    baseline = _answer_records_by_case(baseline_records)
    candidate = _answer_records_by_case(candidate_records)
    shared_ids = sorted(set(baseline) & set(candidate))

    metrics: dict[str, Any] = {}
    for key in metric_keys:
        pairs = [
            (case_id, _score(baseline[case_id], key), _score(candidate[case_id], key))
            for case_id in shared_ids
        ]
        pairs = [(cid, a, b) for cid, a, b in pairs if a is not None and b is not None]
        if not pairs:
            metrics[key] = {"n": 0}
            continue
        base_vals = [a for _, a, _ in pairs]
        cand_vals = [b for _, _, b in pairs]
        diffs = [b - a for _, a, b in pairs]
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        rho = spearman_rho(base_vals, cand_vals)
        divergent = sorted(pairs, key=lambda item: abs(item[2] - item[1]), reverse=True)
        metrics[key] = {
            "n": len(pairs),
            "baseline_mean": round(statistics.fmean(base_vals), 4),
            "candidate_mean": round(statistics.fmean(cand_vals), 4),
            "mean_diff": round(statistics.fmean(diffs), 4),
            "mean_abs_diff": round(statistics.fmean(abs(d) for d in diffs), 4),
            "spearman_rho": round(rho, 4) if rho is not None else None,
            "wins": wins,
            "ties": len(diffs) - wins - losses,
            "losses": losses,
            "sign_test_p": (
                round(p, 4) if (p := sign_test_p_value(wins, losses)) is not None else None
            ),
            "top_divergent_cases": [
                {"case_id": cid, "baseline": a, "candidate": b}
                for cid, a, b in divergent[:_TOP_DIVERGENT_CASES]
                if a != b
            ],
        }

    groups: dict[str, dict[str, Any]] = {}
    for case_id in shared_ids:
        group = str(candidate[case_id].get(group_field) or baseline[case_id].get(group_field) or "-")
        groups.setdefault(group, {"case_ids": []})["case_ids"].append(case_id)
    by_group: dict[str, Any] = {}
    for group, info in sorted(groups.items()):
        entry: dict[str, Any] = {"n": len(info["case_ids"])}
        for key in metric_keys:
            diffs = [
                b - a
                for cid in info["case_ids"]
                if (a := _score(baseline[cid], key)) is not None
                and (b := _score(candidate[cid], key)) is not None
            ]
            entry[key] = round(statistics.fmean(diffs), 4) if diffs else None
        by_group[group] = entry

    def _unexpected(records: dict[str, dict[str, Any]]) -> list[str]:
        return sorted(cid for cid in shared_ids if records[cid].get("unexpected_abstention"))

    return {
        "shared_answer_cases": len(shared_ids),
        "only_in_baseline": sorted(set(baseline) - set(candidate)),
        "only_in_candidate": sorted(set(candidate) - set(baseline)),
        "metrics": metrics,
        "by_group": by_group,
        "group_field": group_field,
        "unexpected_abstentions": {
            "baseline": _unexpected(baseline),
            "candidate": _unexpected(candidate),
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:+.3f}" if value < 0 or value > 0 else "0.000"
    return str(value)


def render_comparison_markdown(
    result: dict[str, Any],
    *,
    baseline_label: str,
    candidate_label: str,
    title: str,
) -> str:
    """Kết xuất kết quả so sánh ghép cặp thành báo cáo Markdown dễ đọc."""

    lines = [
        f"# {title}",
        "",
        f"- **Baseline (A):** {baseline_label}",
        f"- **Candidate (B):** {candidate_label}",
        f"- **Số ca trả lời ghép cặp được:** {result['shared_answer_cases']}",
        "- **Quy ước:** chênh lệch = B − A; *thắng* = B cao hơn A ở cùng một ca.",
        "- **Spearman ρ:** mức giống nhau về thứ hạng các ca giữa A và B (1 = giống hoàn toàn).",
        "- **Sign test p:** xác suất thấy chênh lệch thắng/thua như vậy nếu A và B thực ra ngang nhau"
        " (p < 0.05 thường được coi là khác biệt có ý nghĩa thống kê).",
        "",
        "## 1. Tổng quan theo chỉ số",
        "",
        "| Chỉ số | n | Mean A | Mean B | B − A | Trung bình \\|B − A\\| | Spearman ρ | Thắng / Hòa / Thua | Sign test p |",
        "| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for key, item in result["metrics"].items():
        if not item.get("n"):
            lines.append(f"| `{key}` | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(
            f"| `{key}` | {item['n']} | {item['baseline_mean']:.3f} | {item['candidate_mean']:.3f} "
            f"| {_fmt(item['mean_diff'])} | {item['mean_abs_diff']:.3f} "
            f"| {item['spearman_rho'] if item['spearman_rho'] is not None else 'N/A'} "
            f"| {item['wins']} / {item['ties']} / {item['losses']} "
            f"| {item['sign_test_p'] if item['sign_test_p'] is not None else 'N/A'} |"
        )

    metric_keys = list(result["metrics"].keys())
    lines += [
        "",
        f"## 2. Chênh lệch trung bình (B − A) theo `{result['group_field']}`",
        "",
        "| Nhóm | n | " + " | ".join(f"`{k}`" for k in metric_keys) + " |",
        "| --- | :---: | " + " | ".join(":---:" for _ in metric_keys) + " |",
    ]
    for group, entry in result["by_group"].items():
        lines.append(
            f"| `{group}` | {entry['n']} | "
            + " | ".join(_fmt(entry.get(k)) for k in metric_keys)
            + " |"
        )

    unexpected = result["unexpected_abstentions"]
    lines += [
        "",
        "## 3. Ca từ chối ngoài dự kiến",
        "",
        f"- **A:** {', '.join(unexpected['baseline']) or 'không có'}",
        f"- **B:** {', '.join(unexpected['candidate']) or 'không có'}",
        "",
        "## 4. Các ca chênh lệch lớn nhất theo từng chỉ số",
        "",
    ]
    for key, item in result["metrics"].items():
        cases = item.get("top_divergent_cases") or []
        if not cases:
            continue
        rendered = ", ".join(
            f"`{c['case_id']}` ({c['baseline']:.3f} → {c['candidate']:.3f})" for c in cases
        )
        lines.append(f"- **`{key}`:** {rendered}")
    if result["only_in_baseline"] or result["only_in_candidate"]:
        lines += [
            "",
            "## 5. Ca không ghép cặp được",
            "",
            f"- Chỉ có ở A: {', '.join(result['only_in_baseline']) or 'không có'}",
            f"- Chỉ có ở B: {', '.join(result['only_in_candidate']) or 'không có'}",
        ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="So sánh ghép cặp hai lần chấm RAGAS.")
    parser.add_argument("--baseline", type=Path, required=True, help="File raw JSON của A.")
    parser.add_argument("--candidate", type=Path, required=True, help="File raw JSON của B.")
    parser.add_argument("--baseline-label", default="A")
    parser.add_argument("--candidate-label", default="B")
    parser.add_argument("--title", default="So sánh ghép cặp kết quả RAGAS")
    parser.add_argument("--output", type=Path, required=True, help="Đường dẫn file báo cáo .md")
    args = parser.parse_args(argv)

    result = compare_scored_runs(
        load_scored_records(args.baseline),
        load_scored_records(args.candidate),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_comparison_markdown(
            result,
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
            title=args.title,
        ),
        encoding="utf-8",
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã ghi báo cáo so sánh: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
