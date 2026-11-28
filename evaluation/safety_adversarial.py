"""Đo tỷ lệ bỏ sót và báo động nhầm của 10 luật an toàn deterministic trên bộ câu đối kháng.

Chạy hoàn toàn offline (chỉ gọi ``evaluate_safety``), không tốn quota LLM:
    python -m evaluation.safety_adversarial

Chỉ số báo cáo:
- **Recall (bắt đúng luật):** tỷ lệ ca ``must_trigger`` kích hoạt đúng luật kỳ vọng.
- **Recall (bắt bất kỳ luật an toàn):** tỷ lệ ca ``must_trigger`` được chặn bởi bất kỳ luật nào;
  với người dùng, bị chặn bởi một luật cấp cứu khác vẫn là phản hồi an toàn.
- **Tỷ lệ báo động nhầm:** tỷ lệ ca đối chứng ``must_not_trigger`` bị chặn nhầm (làm mất câu
  trả lời có bằng chứng cho câu hỏi tìm hiểu thông thường).

Trong y tế, ca bỏ sót (``must_trigger`` không được chặn) là lỗi nghiêm trọng nhất cần ưu tiên.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ragas_runner import DEFAULT_RESULTS_DIR  # noqa: E402

SAFETY_ADVERSARIAL_DATASET = Path("evaluation/safety_adversarial.json")
SAFETY_ADVERSARIAL_RESULTS_DIR = DEFAULT_RESULTS_DIR / "safety_adversarial"
_VALID_KINDS = {"must_trigger", "must_not_trigger"}


def load_safety_cases(path: Path = SAFETY_ADVERSARIAL_DATASET) -> list[dict[str, Any]]:
    """Đọc và kiểm tra cấu trúc bộ ca đối kháng; sai cấu trúc thì dừng ngay."""

    from src.agent.safety_policy import SAFETY_RULES

    data = json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))
    known_rules = {rule.rule_id for rule in SAFETY_RULES}
    cases = list(data.get("cases") or [])
    seen: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id") or "")
        if not case_id or case_id in seen:
            raise ValueError(f"case_id trống hoặc trùng: {case_id!r}")
        seen.add(case_id)
        if case.get("kind") not in _VALID_KINDS:
            raise ValueError(f"{case_id}: kind không hợp lệ")
        expected = case.get("expected_rule_id")
        if case["kind"] == "must_trigger" and expected not in known_rules:
            raise ValueError(f"{case_id}: expected_rule_id không thuộc SAFETY_RULES: {expected}")
        if case["kind"] == "must_not_trigger" and expected is not None:
            raise ValueError(f"{case_id}: ca đối chứng không được có expected_rule_id")
        if not str(case.get("user_input") or "").strip():
            raise ValueError(f"{case_id}: thiếu user_input")
    return cases


def evaluate_safety_cases(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Chạy ``evaluate_safety`` trên từng ca và tổng hợp recall / báo động nhầm."""

    from src.agent.safety_policy import evaluate_safety

    rows: list[dict[str, Any]] = []
    for case in cases:
        decision = evaluate_safety(
            str(case["user_input"]),
            conversation_history=case.get("conversation_history") or None,
        )
        actual = decision.rule_id if decision is not None else None
        rows.append(
            {
                "case_id": case["case_id"],
                "kind": case["kind"],
                "variant": case.get("variant"),
                "expected_rule_id": case.get("expected_rule_id"),
                "actual_rule_id": actual,
                "triggered": actual is not None,
                "exact": actual == case.get("expected_rule_id"),
                "user_input": case["user_input"],
            }
        )

    positives = [row for row in rows if row["kind"] == "must_trigger"]
    controls = [row for row in rows if row["kind"] == "must_not_trigger"]

    def _rate(hits: int, total: int) -> float | None:
        return round(hits / total, 4) if total else None

    by_rule: dict[str, dict[str, int]] = {}
    for row in positives:
        entry = by_rule.setdefault(str(row["expected_rule_id"]), {"exact": 0, "any": 0, "total": 0})
        entry["total"] += 1
        entry["exact"] += int(row["exact"])
        entry["any"] += int(row["triggered"])
    by_variant: dict[str, dict[str, int]] = {}
    for row in positives:
        entry = by_variant.setdefault(str(row["variant"]), {"any": 0, "total": 0})
        entry["total"] += 1
        entry["any"] += int(row["triggered"])

    exact_hits = sum(1 for row in positives if row["exact"])
    any_hits = sum(1 for row in positives if row["triggered"])
    false_alarms = sum(1 for row in controls if row["triggered"])
    return {
        "positive_cases": len(positives),
        "control_cases": len(controls),
        "recall_exact": _rate(exact_hits, len(positives)),
        "recall_any": _rate(any_hits, len(positives)),
        "false_alarm_rate": _rate(false_alarms, len(controls)),
        "exact_hits": exact_hits,
        "any_hits": any_hits,
        "false_alarms": false_alarms,
        "missed": [row for row in positives if not row["triggered"]],
        "wrong_rule": [row for row in positives if row["triggered"] and not row["exact"]],
        "false_alarm_cases": [row for row in controls if row["triggered"]],
        "by_rule": dict(sorted(by_rule.items())),
        "by_variant": dict(sorted(by_variant.items())),
        "rows": rows,
    }


def render_safety_markdown(summary: dict[str, Any]) -> str:
    """Báo cáo Markdown cho bộ kiểm tra an toàn đối kháng."""

    def _pct(value: float | None) -> str:
        return f"{value:.1%}" if value is not None else "N/A"

    lines = [
        "# Kiểm tra đối kháng 10 luật an toàn deterministic",
        "",
        f"- **Recall bắt đúng luật:** {summary['exact_hits']}/{summary['positive_cases']} ({_pct(summary['recall_exact'])})",
        f"- **Recall bắt bất kỳ luật an toàn:** {summary['any_hits']}/{summary['positive_cases']} ({_pct(summary['recall_any'])})",
        f"- **Báo động nhầm trên ca đối chứng:** {summary['false_alarms']}/{summary['control_cases']} ({_pct(summary['false_alarm_rate'])})",
        "",
        "## Theo luật",
        "",
        "| Luật | Bắt đúng luật | Bắt bất kỳ luật | Tổng |",
        "| --- | :---: | :---: | :---: |",
    ]
    for rule_id, entry in summary["by_rule"].items():
        lines.append(f"| `{rule_id}` | {entry['exact']} | {entry['any']} | {entry['total']} |")
    lines += [
        "",
        "## Theo kiểu diễn đạt",
        "",
        "| Kiểu | Bị chặn / Tổng |",
        "| --- | :---: |",
    ]
    for variant, entry in summary["by_variant"].items():
        lines.append(f"| `{variant}` | {entry['any']}/{entry['total']} |")

    def _case_lines(title: str, rows: list[dict[str, Any]]) -> list[str]:
        out = ["", f"## {title}", ""]
        if not rows:
            return [*out, "Không có."]
        out += ["| Case ID | Kỳ vọng | Thực tế | Câu hỏi |", "| --- | --- | --- | --- |"]
        for row in rows:
            out.append(
                f"| `{row['case_id']}` | `{row['expected_rule_id']}` | `{row['actual_rule_id']}` "
                f"| {row['user_input']} |"
            )
        return out

    lines += _case_lines("Ca bỏ sót (nghiêm trọng nhất)", summary["missed"])
    lines += _case_lines("Ca chặn bởi luật khác luật kỳ vọng", summary["wrong_rule"])
    lines += _case_lines("Ca đối chứng bị báo động nhầm", summary["false_alarm_cases"])
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đo recall / báo động nhầm của luật an toàn.")
    parser.add_argument("--dataset", type=Path, default=SAFETY_ADVERSARIAL_DATASET)
    args = parser.parse_args(argv)

    summary = evaluate_safety_cases(load_safety_cases(args.dataset))
    out_dir: Path = PROJECT_ROOT / SAFETY_ADVERSARIAL_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "safety_adversarial_raw.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = out_dir / "safety_adversarial_report.md"
    report.write_text(render_safety_markdown(summary), encoding="utf-8")
    print(
        f"Recall đúng luật {summary['exact_hits']}/{summary['positive_cases']}, "
        f"bất kỳ luật {summary['any_hits']}/{summary['positive_cases']}, "
        f"báo động nhầm {summary['false_alarms']}/{summary['control_cases']}. Báo cáo: {report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
