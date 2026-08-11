#!/usr/bin/env python3
"""Validate the completed Phase 1 knowledge base without writing data."""

from __future__ import annotations

import asyncio
import json
import logging
import os
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

from scripts.eval_phase1_readiness import run_phase1_readiness_eval  # noqa: E402
from scripts.validate_kb_collections import inspect_qdrant_schema  # noqa: E402
from src.database.vector_store import qdrant_client_kwargs  # noqa: E402
from src.knowledge.entity_index import get_entity_collection_name  # noqa: E402
from src.knowledge.versioning import get_embedding_metadata  # noqa: E402

logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "ingestion_manifest.json"
EXPECTED_CHUNK_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge").strip() or "acne_knowledge"
EXPECTED_ENTITY_COLLECTION = get_entity_collection_name()
ACCEPTABLE_ENTITY_POINTS = {20, 22, 32}
EXPECTED_MANIFEST_RECORDS = 4
EXPECTED_NEO4J_LABELS = {
    "ActiveIngredient": 7,
    "Condition": 1,
    "DrugClass": 6,
    "DrugProduct": 3,
    "SafetyContext": 4,
}
REBUILT_NEO4J_LABELS = {
    "ActiveIngredient": 8,
    "Condition": 1,
    "DrugClass": 6,
    "DrugProduct": 4,
    "SafetyContext": 4,
}
EXPECTED_NEO4J_RELATIONSHIPS = {
    "BELONGS_TO_CLASS": 11,
    "CONTRAINDICATED_IN": 0,
    "HAS_ACTIVE_INGREDIENT": 4,
}
REBUILT_NEO4J_RELATIONSHIPS = {
    "BELONGS_TO_CLASS": 13,
    "CONTRAINDICATED_IN": 0,
    "HAS_ACTIVE_INGREDIENT": 5,
}
EXPANDED_NEO4J_LABELS = {
    "ActiveIngredient": 15,
    "Condition": 1,
    "DrugClass": 8,
    "DrugProduct": 4,
    "SafetyContext": 4,
}
EXPANDED_NEO4J_RELATIONSHIPS = {
    "BELONGS_TO_CLASS": 20,
    "CONTRAINDICATED_IN": 2,
    "HAS_ACTIVE_INGREDIENT": 5,
}


def add_check(
    checks: list[dict[str, Any]],
    errors: list[str],
    name: str,
    passed: bool,
    details: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    check = {"name": name, "passed": bool(passed), "details": details or {}}
    if error:
        check["error"] = error
    checks.append(check)
    if not passed:
        errors.append(error or name)


async def validate_qdrant(checks: list[dict[str, Any]], errors: list[str]) -> None:
    from qdrant_client import AsyncQdrantClient  # type: ignore[import]

    expected_dimensions = int(get_embedding_metadata()["embedding_dimensions"])
    client = AsyncQdrantClient(**qdrant_client_kwargs())
    try:
        collections = await client.get_collections()
        existing = {collection.name for collection in collections.collections}
        add_check(
            checks,
            errors,
            "qdrant_reachable",
            True,
            {"collections": sorted(existing)},
        )

        for role, collection_name, expected_points in (
            ("chunk", EXPECTED_CHUNK_COLLECTION, None),
            ("entity", EXPECTED_ENTITY_COLLECTION, ACCEPTABLE_ENTITY_POINTS),
        ):
            if collection_name not in existing:
                add_check(
                    checks,
                    errors,
                    f"qdrant_{role}_collection_exists",
                    False,
                    {"collection": collection_name},
                    f"{role} collection {collection_name!r} missing",
                )
                continue

            info = await client.get_collection(collection_name=collection_name)
            schema = inspect_qdrant_schema(info.config.params)
            points_count = int(getattr(info, "points_count", 0) or 0)
            details = {
                "collection": collection_name,
                "points_count": points_count,
                **schema,
            }
            schema_passed = (
                schema["has_dense"]
                and schema["dense_vector_name"] == "dense"
                and schema["dense_size"] == expected_dimensions
                and schema["has_bm25"]
                and schema["sparse_vector_name"] == "bm25"
            )
            points_passed = points_count > 0 if expected_points is None else points_count in expected_points
            add_check(
                checks,
                errors,
                f"qdrant_{role}_schema_and_points",
                schema_passed and points_passed,
                details,
                (
                    f"{role} collection schema/points mismatch: {details}"
                    if not (schema_passed and points_passed)
                    else None
                ),
            )
    except Exception as exc:
        add_check(checks, errors, "qdrant_reachable", False, error=str(exc))
    finally:
        await client.close()


def validate_manifest(checks: list[dict[str, Any]], errors: list[str]) -> None:
    path = DEFAULT_MANIFEST_PATH
    if not path.exists():
        add_check(checks, errors, "manifest_exists", False, {"path": str(path)}, "manifest missing")
        return

    manifest = json.loads(path.read_text(encoding="utf-8"))
    documents = manifest.get("documents", {})
    records = [record for record in documents.values() if isinstance(record, dict)]
    missing_point_ids = [
        record.get("source_path")
        for record in records
        if not record.get("qdrant_point_ids")
    ]
    total_point_count = sum(int(record.get("qdrant_point_count", 0) or 0) for record in records)
    web_records = [
        record
        for record in records
        if str(record.get("source_file") or "").lower() == "web_raw_dataset.json"
    ]
    details = {
        "path": str(path),
        "record_count": len(records),
        "missing_qdrant_point_ids": missing_point_ids,
        "total_qdrant_point_count": total_point_count,
        "web_raw_dataset_source_types": [record.get("source_type") for record in web_records],
        "status_counts": _counter(record.get("status") for record in records),
    }
    passed = (
        len(records) == EXPECTED_MANIFEST_RECORDS
        and not missing_point_ids
        and total_point_count > 0
        and any(record.get("source_type") == "web_json" for record in web_records)
    )
    add_check(
        checks,
        errors,
        "manifest_phase1_records",
        passed,
        details,
        f"manifest mismatch: {details}" if not passed else None,
    )


async def validate_neo4j(checks: list[dict[str, Any]], errors: list[str]) -> None:
    try:
        from neo4j import AsyncGraphDatabase  # type: ignore[import]
    except ImportError as exc:
        add_check(checks, errors, "neo4j_reachable", False, error=str(exc))
        return

    uri_candidates = _neo4j_uri_candidates()
    last_error: Exception | None = None
    for uri in uri_candidates:
        driver = AsyncGraphDatabase.driver(
            uri,
            auth=(
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password"),
            ),
        )
        try:
            async with driver.session() as session:
                label_counts: dict[str, int] = {}
                for label in EXPECTED_NEO4J_LABELS:
                    result = await session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
                    record = await result.single()
                    label_counts[label] = int(record["count"]) if record else 0

                rel_counts: dict[str, int] = {}
                for rel_type in EXPECTED_NEO4J_RELATIONSHIPS:
                    result = await session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count")
                    record = await result.single()
                    rel_counts[rel_type] = int(record["count"]) if record else 0

            node_total = sum(label_counts.values())
            rel_total = sum(rel_counts.values())
            details = {
                "uri": uri,
                "nodes": node_total,
                "relationships": rel_total,
                "labels": label_counts,
                "relationship_types": rel_counts,
            }
            baseline_ok = (
                node_total == 21
                and rel_total == 15
                and label_counts == EXPECTED_NEO4J_LABELS
                and rel_counts == EXPECTED_NEO4J_RELATIONSHIPS
            )
            rebuilt_ok = (
                node_total == 23
                and rel_total == 18
                and label_counts == REBUILT_NEO4J_LABELS
                and rel_counts == REBUILT_NEO4J_RELATIONSHIPS
            )
            expanded_ok = (
                node_total == 32
                and rel_total == 27
                and label_counts == EXPANDED_NEO4J_LABELS
                and rel_counts == EXPANDED_NEO4J_RELATIONSHIPS
            )
            passed = baseline_ok or rebuilt_ok or expanded_ok
            add_check(
                checks,
                errors,
                "neo4j_deterministic_graph",
                passed,
                details,
                f"Neo4j graph mismatch: {details}" if not passed else None,
            )
            return
        except Exception as exc:
            last_error = exc
        finally:
            await driver.close()

    add_check(
        checks,
        errors,
        "neo4j_reachable",
        False,
        {"uris": uri_candidates},
        str(last_error) if last_error else "Neo4j not reachable",
    )


def validate_readiness(checks: list[dict[str, Any]], errors: list[str]) -> None:
    try:
        summary = run_phase1_readiness_eval()
        add_check(
            checks,
            errors,
            "phase1_readiness_eval",
            summary.get("passed") is True,
            {
                "readiness": summary.get("readiness"),
                "total_cases": summary.get("total_cases"),
                "failures": summary.get("failures"),
            },
            "Phase 1 readiness eval failed" if summary.get("passed") is not True else None,
        )
    except Exception as exc:
        add_check(checks, errors, "phase1_readiness_eval", False, error=str(exc))


def _counter(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "<missing>")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _neo4j_uri_candidates() -> list[str]:
    env_uri = os.getenv("NEO4J_URI", "").strip()
    candidates = []
    if env_uri:
        candidates.append(env_uri)
    fallback = "bolt://127.0.0.1:7687"
    if fallback not in candidates:
        candidates.append(fallback)
    return candidates


async def main() -> int:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    await validate_qdrant(checks, errors)
    validate_manifest(checks, errors)
    await validate_neo4j(checks, errors)
    validate_readiness(checks, errors)

    report = {
        "passed": not errors,
        "checks": checks,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
