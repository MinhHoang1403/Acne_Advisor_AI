#!/usr/bin/env python3
"""Build and optionally upsert deterministic taxonomy entity graph into Neo4j."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from src.knowledge.entity_cards import build_entity_cards_from_taxonomy  # noqa: E402
from src.knowledge.graph_index import (  # noqa: E402
    apply_entity_graph_schema,
    get_neo4j_driver,
    upsert_entity_graph,
    validate_entity_graph,
    validate_entity_graph_records,
)
from src.knowledge.graph_schema import (  # noqa: E402
    build_entity_graph_records,
    summarize_graph_records,
)
from src.knowledge.versioning import get_knowledge_versions  # noqa: E402


PREVIEW_NAMES = {"Dalacin T", "Epiduo", "Differin", "Tazorac", "benzoyl_peroxide", "tazarotene"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic entity/drug graph records from taxonomy.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build and print records only.")
    parser.add_argument("--apply-schema", action="store_true", help="Apply Neo4j constraints/indexes.")
    parser.add_argument("--upsert", action="store_true", help="Upsert deterministic entity graph.")
    parser.add_argument("--validate", action="store_true", help="Validate deterministic entity graph.")
    parser.add_argument(
        "--kb-version",
        default=get_knowledge_versions()["kb_version"],
        help="KB version for graph properties and entity IDs.",
    )
    return parser.parse_args(argv)


def build_dry_run_summary(kb_version: str | None = None) -> dict:
    cards = build_entity_cards_from_taxonomy()
    records = build_entity_graph_records(cards, kb_version=kb_version)
    summary = summarize_graph_records(records)
    return {
        "kb_version": kb_version or get_knowledge_versions()["kb_version"],
        "card_count": len(cards),
        "node_count": len(records["nodes"]),
        "relationship_count": len(records["relationships"]),
        **summary,
        "preview": _preview_records(records),
    }


async def main() -> int:
    args = parse_args()
    should_write_or_validate = args.apply_schema or args.upsert or args.validate

    if args.dry_run or not should_write_or_validate:
        print(json.dumps(build_dry_run_summary(args.kb_version), ensure_ascii=False, indent=2))
        return 0

    cards = build_entity_cards_from_taxonomy()
    records = build_entity_graph_records(cards, kb_version=args.kb_version)
    driver = get_neo4j_driver()
    try:
        if args.apply_schema or args.upsert:
            await apply_entity_graph_schema(driver)
            print("Entity graph schema applied.")
        if args.upsert:
            counts = await upsert_entity_graph(driver, records)
            print(json.dumps({"upserted": counts}, ensure_ascii=False, indent=2))
        if args.validate:
            baseline_validation = await validate_entity_graph(driver)
            records_validation = await validate_entity_graph_records(driver, records)
            validation = {
                "passed": bool(baseline_validation.get("passed"))
                and bool(records_validation.get("passed")),
                "baseline": baseline_validation,
                "records": records_validation,
            }
            print(json.dumps({"validation": validation}, ensure_ascii=False, indent=2, default=str))
            return 0 if validation.get("passed") else 1
    finally:
        await driver.close()

    return 0


def _preview_records(records: dict[str, list[dict]]) -> dict[str, list[dict]]:
    preview_nodes = [
        node
        for node in records["nodes"]
        if node.get("canonical_name") in PREVIEW_NAMES
    ]
    preview_relationships = [
        relationship
        for relationship in records["relationships"]
        if (
            relationship.get("source_name") in PREVIEW_NAMES
            or relationship.get("target_name") in PREVIEW_NAMES
        )
    ]
    return {
        "nodes": preview_nodes,
        "relationships": preview_relationships,
    }


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
