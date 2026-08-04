"""CLI wrapper for the checkpointed canonical comprehensive evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from src.evaluation.comprehensive import ComprehensiveRunner, EvaluationConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "notebooks" / "eval_data" / "acne_rag_eval_comprehensive_v1.jsonl")
    parser.add_argument("--report-root", type=Path, default=ROOT / "reports" / "evaluation")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--question-limit", type=int, default=300)
    parser.add_argument("--live-provider", default="ollama")
    parser.add_argument("--live-model", default="qwen3:8b")
    parser.add_argument("--judge-provider", default="gemini")
    parser.add_argument("--judge-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--run-judge", action="store_true")
    parser.add_argument("--bypass-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-persistence", action="store_true")
    parser.add_argument("--report-label", default="comprehensive_v1")
    args = parser.parse_args()
    if args.judge_provider != "gemini":
        raise SystemExit("BLOCKED: canonical route-aware judge requires --judge-provider gemini.")
    config = EvaluationConfig(dataset_path=args.dataset.resolve(), report_root=args.report_root.resolve(), api_base_url=args.api_base_url, question_limit=args.question_limit, live_provider=args.live_provider, live_model=args.live_model, judge_provider=args.judge_provider, judge_model=args.judge_model, run_live=args.run_live, run_judge=args.run_judge, bypass_cache=args.bypass_cache, smoke=args.smoke, no_persistence=args.no_persistence, report_label=args.report_label)
    report_dir = ComprehensiveRunner(config, ROOT).run(resume=args.resume)
    print(f"COMPREHENSIVE_EVALUATION_REPORT={report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
