"""Thin canonical operator CLI for the frozen Phase 1 foundation."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pipeline import build_phase1, phase1_status, validate_phase1  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build, validate or inspect frozen Phase 1.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build immutable Qdrant candidates.")
    build.add_argument("--source", type=Path, default=Path("sample_data"))
    build.add_argument("--replace-candidate", action="store_true")
    validate = subparsers.add_parser("validate", help="Run layered Phase 1 validation.")
    validate.add_argument("--offline", action="store_true")
    subparsers.add_parser("status", help="Show expected and current build identity.")
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        result = await build_phase1(source_dir=args.source, replace_candidate=args.replace_candidate)
    elif args.command == "validate":
        result = await validate_phase1(live=not args.offline)
    else:
        result = await phase1_status()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("passed", result.get("offline_validation", True)) else 1


def cli() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(cli())
