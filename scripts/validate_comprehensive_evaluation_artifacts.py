"""Validate a completed canonical comprehensive evaluation artifact directory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.comprehensive import FINAL_FILES


REQUIRED_PLOTS = {
    "system_scorecard.png",
    "category_scores.png",
    "retrieval_quality.png",
    "origin_distribution.png",
    "safety_quality.png",
    "judge_score_by_category.png",
    "judge_score_by_origin.png",
    "latency_by_origin.png",
    "failure_reason_distribution.png",
}


def jsonl_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def validate(report_dir: Path, *, expected_cases: int = 300, require_completed: bool = True) -> None:
    missing = [name for name in FINAL_FILES if not (report_dir / name).is_file()]
    missing_plots = sorted(name for name in REQUIRED_PLOTS if not (report_dir / "plots" / name).is_file())
    if missing or missing_plots:
        raise ValueError(f"Missing artifacts: files={missing}, plots={missing_plots}")
    raw_count = jsonl_count(report_dir / "raw_responses.jsonl")
    case_count = csv_count(report_dir / "case_results.csv")
    judge_count = csv_count(report_dir / "judge_results.csv")
    manifest = json.loads((report_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    judge = json.loads((report_dir / "judge_summary.json").read_text(encoding="utf-8"))
    if raw_count != expected_cases or case_count != expected_cases or judge_count != expected_cases:
        raise ValueError(f"Row count mismatch raw={raw_count}, case={case_count}, judge={judge_count}, expected={expected_cases}")
    if require_completed and manifest.get("run_status") != "completed":
        raise ValueError(f"Manifest run_status must be completed, got {manifest.get('run_status')!r}")
    if judge.get("judge_cases") != expected_cases:
        raise ValueError(f"Judge cases must be {expected_cases}, got {judge.get('judge_cases')}")
    if judge.get("final_error_count", 0) != 0:
        raise ValueError(f"Final judge errors must be zero, got {judge.get('final_error_count')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=300)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    validate(args.report_dir.resolve(), expected_cases=args.expected_cases, require_completed=not args.allow_smoke)
    print("COMPREHENSIVE_EVALUATION_ARTIFACTS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
