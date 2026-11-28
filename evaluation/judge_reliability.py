"""Kiểm định độ tin cậy của LLM judge trên chính các câu trả lời đã sinh của hệ thống.

Hai phép kiểm định, đều chỉ chấm lại bản ghi có sẵn (không chạy lại hệ thống, không tốn quota
sinh câu trả lời Gemini; Answer Relevancy/Answer Correctness vẫn gọi embedding):

1. ``rescore``: chấm lại toàn bộ bản ghi bằng **một judge cố định, tắt chuỗi dự phòng**. Chạy
   cùng judge hai lần để đo độ ổn định (nhiễu nền); chạy judge khác để đo mức đồng thuận giữa
   các judge. So sánh kết quả bằng ``python -m evaluation.paired_comparison``.
2. ``planted``: kiểm tra độ nhạy của Faithfulness bằng cách chèn có chủ đích một mệnh đề y khoa
   SAI (trái ngược với kho tri thức) vào các câu trả lời đang đạt Faithfulness = 1.0. Judge
   đáng tin cậy phải hạ Faithfulness ở các ca bị cài lỗi. Đây là dạng kiểm thử độ nhạy bằng
   lỗi cài sẵn (perturbation / fault-injection testing) cho bộ chấm tự động.

Chạy CLI (đọc ``.env`` giống notebook đánh giá):
    python -m evaluation.judge_reliability rescore --input evaluation/results/ragas_evaluation_raw.json --label judge_primary_run2
    python -m evaluation.judge_reliability planted --input evaluation/results/ragas_evaluation_raw.json --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.paired_comparison import load_scored_records  # noqa: E402
from evaluation.ragas_runner import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROVIDER,
    DEFAULT_RESULTS_DIR,
    export_evaluation_artifacts,
    score_records_with_ragas,
    summarize_evaluation,
)

RELIABILITY_RESULTS_DIR = DEFAULT_RESULTS_DIR / "judge_reliability"

# Các trường do bước chấm điểm sinh ra; phải xóa trước khi chấm lại để không lẫn kết quả cũ.
_SCORING_FIELDS = (
    "ragas_scores",
    "metric_errors",
    "metric_judge_models",
    "score_signature",
    "unexpected_abstention",
    "abstention_correct",
)


@dataclass(frozen=True)
class PlantedClaim:
    """Một mệnh đề sai có chủ đích, kèm bằng chứng trong kho cho thấy nó sai."""

    claim_id: str
    text: str
    contradicted_by: str


# Mỗi mệnh đề đều được đối chiếu là TRÁI NGƯỢC với nội dung có trong
# sample_data/Acne_data_source.json (trích dẫn ở ``contradicted_by``). Chúng chỉ dùng để kiểm thử
# judge và tuyệt đối không phải thông tin y khoa đúng.
PLANTED_FALSE_CLAIMS: tuple[PlantedClaim, ...] = (
    PlantedClaim(
        "isotretinoin_pregnancy_safe",
        "Isotretinoin đường uống hoàn toàn an toàn cho phụ nữ đang mang thai nên không cần tránh thai khi dùng thuốc.",
        "Kho ghi isotretinoin gây quái thai và yêu cầu tránh thai (BAD: pregnancy prevention for teratogenic drugs).",
    ),
    PlantedClaim(
        "isotretinoin_doxycycline_combo",
        "Nên dùng phối hợp isotretinoin đường uống với doxycycline để tăng hiệu quả điều trị mụn.",
        "Kho ghi: 'Medications to avoid while taking isotretinoin include: Tetracycline antibiotics "
        "(includes lymecycline and doxycycline)' và 'Không dùng phối hợp với kháng sinh nhóm cyclin đường uống'.",
    ),
    PlantedClaim(
        "benzoyl_peroxide_oral_antibiotic",
        "Benzoyl peroxide là một loại kháng sinh đường uống, được dùng dưới dạng viên nén mỗi ngày.",
        "Kho xếp benzoyl peroxide vào nhóm thuốc bôi tại chỗ (AAD: 'several topical therapies, including: Benzoyl peroxide').",
    ),
    PlantedClaim(
        "picking_no_scar",
        "Tự nặn mụn bằng tay giúp mụn nhanh khỏi và không bao giờ để lại sẹo.",
        "Kho ghi: 'Squeezing or picking blemishes can cause scars or dark blotches to develop'.",
    ),
)


def strip_scoring_fields(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trả về bản sao bản ghi hệ thống đã bỏ toàn bộ kết quả chấm điểm cũ."""

    return [{k: v for k, v in rec.items() if k not in _SCORING_FIELDS} for rec in records]


def _faithfulness(rec: dict[str, Any]) -> float | None:
    value = (rec.get("ragas_scores") or {}).get("faithfulness")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def select_planted_error_cases(
    records: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Chọn tất định các ca trả lời đạt Faithfulness = 1.0, rải đều qua các nhóm chủ đề.

    Chỉ dùng ca đang được judge coi là trung thực tuyệt đối để mọi sụt giảm sau khi cài lỗi
    đều quy được cho mệnh đề sai đã chèn vào.
    """

    eligible = [
        rec
        for rec in records
        if rec.get("expected_behavior") == "answer"
        and not rec.get("unexpected_abstention")
        and _faithfulness(rec) == 1.0
        and str(rec.get("response") or "").strip()
        and rec.get("retrieved_contexts")
    ]
    by_category: dict[str, list[dict[str, Any]]] = {}
    for rec in sorted(eligible, key=lambda r: str(r.get("case_id"))):
        by_category.setdefault(str(rec.get("category") or "-"), []).append(rec)

    # Lấy lần lượt từng nhóm (round-robin) để mẫu không dồn vào một chủ đề.
    selected: list[dict[str, Any]] = []
    queues = [list(items) for _, items in sorted(by_category.items())]
    while len(selected) < limit and any(queues):
        for queue in queues:
            if queue and len(selected) < limit:
                selected.append(queue.pop(0))
    return selected


def build_planted_error_records(
    records: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Tạo bản ghi đã chèn một mệnh đề sai vào cuối câu trả lời (luân phiên 4 mệnh đề)."""

    planted: list[dict[str, Any]] = []
    for idx, rec in enumerate(select_planted_error_cases(records, limit=limit)):
        claim = PLANTED_FALSE_CLAIMS[idx % len(PLANTED_FALSE_CLAIMS)]
        clone = strip_scoring_fields([rec])[0]
        clone["response"] = f"{str(rec.get('response') or '').rstrip()}\n\n{claim.text}"
        clone["planted_claim_id"] = claim.claim_id
        clone["original_faithfulness"] = _faithfulness(rec)
        planted.append(clone)
    return planted


def summarize_planted_errors(planted_scored: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Tổng hợp tỷ lệ judge phát hiện lỗi (Faithfulness giảm so với bản gốc) và mức giảm."""

    rows: list[dict[str, Any]] = []
    for rec in planted_scored:
        original = rec.get("original_faithfulness")
        perturbed = _faithfulness(rec)
        rows.append(
            {
                "case_id": rec.get("case_id"),
                "category": rec.get("category"),
                "planted_claim_id": rec.get("planted_claim_id"),
                "original_faithfulness": original,
                "perturbed_faithfulness": perturbed,
                "detected": (
                    perturbed < original
                    if isinstance(original, (int, float)) and perturbed is not None
                    else None
                ),
                "metric_errors": rec.get("metric_errors") or {},
            }
        )
    scored = [row for row in rows if row["detected"] is not None]
    detected = sum(1 for row in scored if row["detected"])
    drops = [row["original_faithfulness"] - row["perturbed_faithfulness"] for row in scored]
    by_claim: dict[str, dict[str, int]] = {}
    for row in scored:
        entry = by_claim.setdefault(str(row["planted_claim_id"]), {"detected": 0, "total": 0})
        entry["total"] += 1
        entry["detected"] += int(bool(row["detected"]))
    return {
        "planted_cases": len(rows),
        "scored_cases": len(scored),
        "unscored_cases": [row["case_id"] for row in rows if row["detected"] is None],
        "detected": detected,
        "detection_rate": round(detected / len(scored), 4) if scored else None,
        "mean_faithfulness_drop": round(statistics.fmean(drops), 4) if drops else None,
        "by_claim": by_claim,
        "rows": rows,
    }


def render_planted_markdown(summary: dict[str, Any], *, judge_model: str, source: str) -> str:
    """Báo cáo Markdown cho bài kiểm tra cài lỗi có chủ đích."""

    rate = summary["detection_rate"]
    lines = [
        "# Kiểm tra độ nhạy của judge bằng lỗi cài sẵn (Faithfulness)",
        "",
        f"- **Judge:** `{judge_model}` (tắt chuỗi judge dự phòng)",
        f"- **Nguồn bản ghi:** `{source}`",
        f"- **Số ca cài lỗi:** {summary['planted_cases']} (chấm được {summary['scored_cases']})",
        f"- **Tỷ lệ phát hiện (Faithfulness giảm so với bản gốc 1.0):** "
        f"**{summary['detected']}/{summary['scored_cases']}**"
        + (f" (**{rate:.1%}**)" if rate is not None else ""),
        f"- **Mức giảm Faithfulness trung bình:** {summary['mean_faithfulness_drop']}",
        "",
        "## Theo từng mệnh đề sai",
        "",
        "| Mệnh đề | Phát hiện / Tổng | Bằng chứng trong kho cho thấy mệnh đề sai |",
        "| --- | :---: | --- |",
    ]
    claims = {claim.claim_id: claim for claim in PLANTED_FALSE_CLAIMS}
    for claim_id, entry in sorted(summary["by_claim"].items()):
        evidence = claims[claim_id].contradicted_by if claim_id in claims else "-"
        lines.append(f"| `{claim_id}` | {entry['detected']}/{entry['total']} | {evidence} |")
    lines += [
        "",
        "## Chi tiết từng ca",
        "",
        "| Case ID | Nhóm | Mệnh đề cài | Faithfulness gốc | Sau khi cài | Phát hiện |",
        "| --- | --- | --- | :---: | :---: | :---: |",
    ]
    for row in summary["rows"]:
        perturbed = row["perturbed_faithfulness"]
        lines.append(
            f"| `{row['case_id']}` | `{row['category']}` | `{row['planted_claim_id']}` "
            f"| {row['original_faithfulness']} "
            f"| {f'{perturbed:.3f}' if perturbed is not None else 'N/A'} "
            f"| {'Có' if row['detected'] else ('Không' if row['detected'] is False else 'N/A')} |"
        )
    return "\n".join(lines) + "\n"


def _judge_settings(judge_model: str | None, judge_provider: str | None) -> tuple[str, str]:
    provider = (judge_provider or os.getenv("RAGAS_JUDGE_PROVIDER") or DEFAULT_JUDGE_PROVIDER).strip()
    model = (judge_model or os.getenv("RAGAS_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL).strip()
    return model, provider


async def rescore_with_single_judge(
    records: Sequence[dict[str, Any]],
    *,
    judge_model: str,
    judge_provider: str,
    embedding_model: str,
    checkpoint_path: Path | None,
    metric_keys: Sequence[str] | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Chấm lại bằng đúng MỘT judge: tắt chuỗi dự phòng để kết quả không trộn nhiều giám khảo."""

    return await score_records_with_ragas(
        strip_scoring_fields(records),
        judge_model=judge_model,
        judge_provider=judge_provider,
        fallback_judge_models=(),
        embedding_model=embedding_model,
        checkpoint_path=checkpoint_path,
        metric_keys=tuple(metric_keys) if metric_keys is not None else None,
        verbose=verbose,
    )


async def _run_rescore(args: argparse.Namespace) -> Path:
    judge_model, judge_provider = _judge_settings(args.judge, args.judge_provider)
    source = load_scored_records(args.input)
    out_dir: Path = RELIABILITY_RESULTS_DIR / str(args.label)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = await rescore_with_single_judge(
        source,
        judge_model=judge_model,
        judge_provider=judge_provider,
        embedding_model=args.embedding_model,
        checkpoint_path=out_dir / "scored_checkpoint.json",
    )
    source_meta = json.loads(Path(args.input).read_text(encoding="utf-8"))
    summary = summarize_evaluation(
        scored,
        dataset_meta={"dataset_name": source_meta.get("dataset_name")} if isinstance(source_meta, dict) else None,
        judge_model=judge_model,
        embedding_model=args.embedding_model,
        kb_build_id=source_meta.get("kb_build_id") if isinstance(source_meta, dict) else None,
    )
    summary["rescored_from"] = str(args.input)
    _, raw_path = export_evaluation_artifacts(summary, output_dir=out_dir)
    return raw_path


async def _run_planted(args: argparse.Namespace) -> Path:
    judge_model, judge_provider = _judge_settings(args.judge, args.judge_provider)
    planted = build_planted_error_records(load_scored_records(args.input), limit=args.limit)
    if not planted:
        raise RuntimeError("Không có ca nào đạt Faithfulness = 1.0 để cài lỗi.")
    out_dir: Path = RELIABILITY_RESULTS_DIR / str(args.label)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = await rescore_with_single_judge(
        planted,
        judge_model=judge_model,
        judge_provider=judge_provider,
        embedding_model=args.embedding_model,
        checkpoint_path=out_dir / "scored_checkpoint.json",
        metric_keys=("faithfulness",),
    )
    # Runner giữ nguyên các trường đầu vào (planted_claim_id, original_faithfulness) trong bản ghi.
    summary = summarize_planted_errors(scored)
    summary["judge_model"] = judge_model
    (out_dir / "planted_errors.json").write_text(
        json.dumps({**summary, "records": scored}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = out_dir / "planted_errors_report.md"
    report.write_text(
        render_planted_markdown(summary, judge_model=judge_model, source=str(args.input)),
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    # Giống notebook đánh giá: .env của dự án là nguồn cấu hình chuẩn.
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    parser = argparse.ArgumentParser(description="Kiểm định độ tin cậy của RAGAS judge.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("rescore", "planted"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--input", type=Path, required=True, help="File raw JSON đã chấm.")
        cmd.add_argument("--judge", default=None, help="Judge model (mặc định RAGAS_JUDGE_MODEL).")
        cmd.add_argument("--judge-provider", default=None)
        cmd.add_argument(
            "--embedding-model",
            default=os.getenv("RAGAS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        )
        cmd.add_argument("--label", required=name == "rescore", default="planted_errors")
        if name == "planted":
            cmd.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)

    if args.command == "rescore":
        path = asyncio.run(_run_rescore(args))
    else:
        path = asyncio.run(_run_planted(args))
    print(f"Đã ghi kết quả: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
