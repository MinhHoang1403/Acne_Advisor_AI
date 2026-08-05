"""Run canonical Evaluation V3 in separate live, judge, and finalize stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from evaluation.models import EvaluationConfig
from evaluation.runner import FinalEvaluationRunner


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "data" / "acne_system_eval_v3.jsonl")
    parser.add_argument("--report-root", type=Path, default=ROOT / "reports" / "evaluation")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--question-limit", type=int, default=300)
    parser.add_argument("--judge-limit", type=int, default=300)
    parser.add_argument("--case-id", action="append", default=[], help="Run only these V3 case IDs; useful for a targeted replay.")
    parser.add_argument("--bypass-cache", action="store_true", default=True)
    parser.add_argument("--no-persistence", action="store_true", default=True)
    parser.add_argument("--checkpoint", action="store_true", default=True)
    parser.add_argument("--run-dir", type=Path, help="Explicit run directory inside --report-root.")


def _config(args: argparse.Namespace, *, judge: bool = False) -> EvaluationConfig:
    return EvaluationConfig(
        dataset_path=args.dataset.resolve(),
        report_root=args.report_root.resolve(),
        live_provider="ollama" if judge else args.provider,
        live_model="qwen3:8b" if judge else args.model,
        judge_provider=args.provider if judge else "gemini",
        judge_model=args.model if judge else "gemini-3.1-flash-lite",
        question_limit=args.question_limit,
        judge_limit=args.judge_limit,
        case_ids=tuple(args.case_id),
        bypass_cache=args.bypass_cache,
        no_persistence=args.no_persistence,
        checkpoint=args.checkpoint,
        run_dir=args.run_dir.resolve() if args.run_dir else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    live = subparsers.add_parser("live", help="Run direct, isolated Ollama live evaluation.")
    _add_common_options(live)
    live.add_argument("--resume", action="store_true", help="Resume the explicit --run-dir.")
    live.add_argument("--resume-latest", action="store_true")
    judge = subparsers.add_parser("judge", help="Judge saved V3 live responses only.")
    _add_common_options(judge)
    judge.set_defaults(provider="gemini", model="gemini-3.1-flash-lite")
    judge.add_argument("--retry-transient", action="store_true", default=True)
    judge.add_argument("--resume-latest", action="store_true", default=True)
    finalize = subparsers.add_parser("finalize", help="Render final V3 report and plots from saved stages.")
    _add_common_options(finalize)
    finalize.add_argument("--resume-latest", action="store_true", default=True)
    args = parser.parse_args()

    if args.stage == "live":
        runner = FinalEvaluationRunner(_config(args), ROOT)
        if args.run_dir and args.resume_latest:
            raise SystemExit("--run-dir and --resume-latest cannot be combined.")
        report_dir = runner.run_live(resume=args.resume or args.resume_latest)
    elif args.stage == "judge":
        if args.provider != "gemini":
            raise SystemExit("BLOCKED: Evaluation V3 judge provider must be gemini.")
        runner = FinalEvaluationRunner(_config(args, judge=True), ROOT)
        report_dir = runner.run_judge(resume=True)
    else:
        runner = FinalEvaluationRunner(_config(args), ROOT)
        report_dir = runner.finalize(resume=True)
    print(f"FINAL_EVALUATION_V3_REPORT={report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
