"""Operator CLI cho build, validate, activate, freeze và xem knowledge status.

``status`` và ``validate --offline`` không ghi datastore. ``build`` tạo candidate
collections/cache/manifest; ``--activate`` còn chuyển Qdrant aliases và thay entity
graph nên bắt buộc có rollback artifacts. Chi tiết side effect thuộc
``src.ingestion.pipeline``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pipeline import (  # noqa: E402
    activate_knowledge,
    build_knowledge,
    freeze_knowledge,
    inspect_embedding_cache_reuse,
    knowledge_status,
    validate_knowledge,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build, validate, or inspect the knowledge index.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build immutable Qdrant candidates.")
    build.add_argument("--source", type=Path, default=Path("sample_data"))
    build.add_argument("--replace-candidate", action="store_true")
    build.add_argument(
        "--activate",
        action="store_true",
        help="Activate the validated candidate after building it.",
    )
    build.add_argument(
        "--rollback-root",
        type=Path,
        help="Verified pre-mutation Qdrant/Neo4j backup required by --activate.",
    )
    validate = subparsers.add_parser("validate", help="Run layered knowledge-build validation.")
    validate.add_argument("--offline", action="store_true")
    subparsers.add_parser(
        "inspect-cache",
        help="Verify parser and embedding cache reuse without provider or datastore calls.",
    )
    subparsers.add_parser(
        "freeze",
        help="Verify the activated live build and atomically freeze its manifest.",
    )
    subparsers.add_parser("status", help="Show expected and current build identity.")
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        if args.activate and args.rollback_root is None:
            raise SystemExit("--activate requires --rollback-root")
        result = await build_knowledge(
            source_dir=args.source, replace_candidate=args.replace_candidate
        )
        if args.activate:
            result = await activate_knowledge(rollback_root=args.rollback_root)
    elif args.command == "validate":
        result = await validate_knowledge(live=not args.offline)
    elif args.command == "inspect-cache":
        result = await inspect_embedding_cache_reuse()
    elif args.command == "freeze":
        result = await freeze_knowledge()
    else:
        result = await knowledge_status()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("passed", result.get("offline_validation", True)) else 1


def cli() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(cli())
