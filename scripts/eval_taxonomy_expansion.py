#!/usr/bin/env python3
"""Evaluate taxonomy expansion hardening offline or against local read-only stores."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from scripts.inspect_taxonomy_candidates import main as inspect_candidates_main  # noqa: E402
from scripts.plan_entity_index_update import (  # noqa: E402
    build_entity_index_update_plan,
    build_entity_index_update_plan_from_qdrant,
)
from scripts.plan_taxonomy_graph_update import build_taxonomy_graph_update_plan  # noqa: E402
from scripts.validate_neo4j_schema import collect_neo4j_snapshot, validate_snapshot  # noqa: E402
from src.database.vector_store import qdrant_client_kwargs  # noqa: E402
from src.knowledge.entity_index import get_entity_collection_name  # noqa: E402
from src.knowledge.normalizer import DEFAULT_TAXONOMY_PATH, DrugEntityNormalizer  # noqa: E402
from src.knowledge.taxonomy_models import (  # noqa: E402
    migrate_v1_taxonomy,
    validate_taxonomy_catalog,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate taxonomy expansion hardening.")
    parser.add_argument("--mode", choices=["offline", "integration-readonly"], default="offline")
    parser.add_argument(
        "--taxonomy-path",
        default=str(DEFAULT_TAXONOMY_PATH),
        help="Runtime taxonomy path. Defaults to the active drug_aliases.yaml catalog.",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    warnings: list[str] = []

    def add(name: str, passed: bool, details: dict[str, Any] | None = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details or {}})
        if not passed:
            failures.append(name)

    runtime_normalizer = DrugEntityNormalizer(args.taxonomy_path)
    catalog = migrate_v1_taxonomy(
        runtime_normalizer.taxonomy,
        source_path=runtime_normalizer.taxonomy_path,
    )
    validation = validate_taxonomy_catalog(catalog)
    add("taxonomy_validation", validation.passed, validation.model_dump(mode="json"))
    warnings.extend(validation.warnings)

    migrated = migrate_v1_taxonomy(
        DrugEntityNormalizer._load_taxonomy(DEFAULT_TAXONOMY_PATH),
        source_path=DEFAULT_TAXONOMY_PATH,
    )
    add(
        "v1_migration_preserves_required_products",
        {"Differin", "Epiduo", "Dalacin T"}.issubset(
            {entity.canonical_name for entity in migrated.entities if entity.entity_type == "drug_product"}
        ),
        migrated.entity_counts(),
    )
    add("v1_migration_validates", validate_taxonomy_catalog(migrated).passed)

    qdrant_plan = build_entity_index_update_plan(taxonomy_path=args.taxonomy_path)
    graph_plan = build_taxonomy_graph_update_plan(taxonomy_path=args.taxonomy_path)
    add("qdrant_plan_dry_run", qdrant_plan.get("mutation_executed") is False, _plan_counts(qdrant_plan))
    add("graph_plan_dry_run", graph_plan.get("mutation_executed") is False, _graph_plan_counts(graph_plan))
    add("draft_entities_skipped", not qdrant_plan.get("draft_skipped"), {"draft_skipped": qdrant_plan.get("draft_skipped")})

    if args.mode == "offline":
        add(
            "candidate_inspector_available",
            callable(inspect_candidates_main),
            {"script": "scripts/inspect_taxonomy_candidates.py"},
        )
    else:
        qdrant_snapshot = await _qdrant_entity_snapshot()
        add("qdrant_entity_snapshot_readonly", qdrant_snapshot.get("reachable") is True, qdrant_snapshot)
        qdrant_plan = await build_entity_index_update_plan_from_qdrant(taxonomy_path=args.taxonomy_path)
        add(
            "qdrant_existing_points_reused",
            qdrant_plan.get("conflicts") == [] and qdrant_plan.get("existing_point_ids_reused") in {20, 22, 32},
            _plan_counts(qdrant_plan),
        )
        neo4j_snapshot = await collect_neo4j_snapshot()
        neo4j_report = validate_snapshot(neo4j_snapshot)
        add("neo4j_schema_snapshot_readonly", neo4j_report.get("passed") is True, {
            "node_count": neo4j_snapshot.get("node_count"),
            "relationship_count": neo4j_snapshot.get("relationship_count"),
            "failures": neo4j_report.get("failures"),
        })
        add(
            "no_delete_execution",
            not qdrant_plan.get("mutation_executed") and not graph_plan.get("mutation_executed"),
            {
                "qdrant_delete_candidates": qdrant_plan.get("delete_candidates"),
                "neo4j_delete_candidates": graph_plan.get("delete_candidates"),
            },
        )

    report = {
        "passed": not failures,
        "mode": args.mode,
        "taxonomy_version": runtime_normalizer.taxonomy_version,
        "entity_counts": catalog.entity_counts(),
        "candidate_counts": {"verified_additions": len(qdrant_plan.get("new", []))},
        "planned_changes": {
            "qdrant": _plan_counts(qdrant_plan),
            "neo4j": _graph_plan_counts(graph_plan),
        },
        "total_checks": len(checks),
        "passed_checks": sum(1 for check in checks if check["passed"]),
        "failed_checks": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
        "warnings": warnings,
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["passed"] else 1


async def _qdrant_entity_snapshot() -> dict[str, Any]:
    try:
        from qdrant_client import AsyncQdrantClient  # type: ignore[import]
    except ImportError as exc:
        return {"reachable": False, "error": str(exc)}

    collection = get_entity_collection_name()
    client = AsyncQdrantClient(**qdrant_client_kwargs())
    try:
        info = await client.get_collection(collection_name=collection)
        return {
            "reachable": True,
            "collection": collection,
            "points_count": int(getattr(info, "points_count", 0) or 0),
        }
    except Exception as exc:
        return {"reachable": False, "collection": collection, "error": str(exc)}
    finally:
        await client.close()


def _plan_counts(plan: dict[str, Any]) -> dict[str, int]:
    return {
        "unchanged": len(plan.get("unchanged", [])),
        "new": len(plan.get("new", [])),
        "updated": len(plan.get("updated", [])),
        "conflicts": len(plan.get("conflicts", [])),
        "draft_skipped": len(plan.get("draft_skipped", [])),
        "delete_candidates": len(plan.get("delete_candidates", [])),
    }


def _graph_plan_counts(plan: dict[str, Any]) -> dict[str, int]:
    deletes = plan.get("delete_candidates", {}) or {}
    return {
        "nodes_to_create": len(plan.get("nodes_to_create", [])),
        "nodes_to_update": len(plan.get("nodes_to_update", [])),
        "relationships_to_create": len(plan.get("relationships_to_create", [])),
        "relationships_to_update": len(plan.get("relationships_to_update", [])),
        "conflicts": len(plan.get("conflicts", [])),
        "delete_candidates": len(deletes.get("nodes", [])) + len(deletes.get("relationships", [])),
    }


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
