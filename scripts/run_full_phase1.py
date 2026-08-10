#!/usr/bin/env python3
"""Authoritative, strict full Phase 1 build for an empty Acne Advisor AI KB.

This command composes existing lower-level ingestion, entity-index, and graph
helpers. It is the only command that may certify ``full_phase1_validation`` in
the ingestion manifest; individual scripts remain useful for diagnostics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
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

from scripts.build_entity_index import build_and_index_entities  # noqa: E402
from scripts.ingest_knowledge import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    EMBEDDING_DIMENSIONS,
    QDRANT_COLLECTION_NAME,
    SAMPLE_DATA_DIR,
    canonical_source_identity,
    discover_source_documents,
    finalize_full_phase1_manifest,
    ingest_pipeline,
    load_ingestion_manifest,
    qdrant_client_kwargs,
    run_preflight_checks,
)
from src.knowledge.entity_cards import build_entity_cards_from_taxonomy  # noqa: E402
from src.knowledge.entity_index import get_entity_collection_name  # noqa: E402
from src.knowledge.graph_index import (  # noqa: E402
    apply_entity_graph_schema,
    get_neo4j_driver,
    upsert_entity_graph,
    validate_entity_graph,
    validate_entity_graph_records,
)
from src.knowledge.graph_schema import build_entity_graph_records  # noqa: E402
from src.knowledge.phase1_validation import (  # noqa: E402
    Phase1ValidationReport,
    manifest_knowledge_expectations,
    reconcile_qdrant_collection,
)
from src.knowledge.versioning import get_knowledge_versions  # noqa: E402


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FullPhase1Plan:
    source_dir: Path
    source_paths: tuple[Path, ...]
    source_identities: tuple[str, ...]
    entity_count: int
    graph_node_count: int
    graph_relationship_count: int
    knowledge_collection: str
    entity_collection: str
    kb_version: str
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_dir": str(self.source_dir),
            "sources": [str(path) for path in self.source_paths],
            "source_identities": list(self.source_identities),
            "entity_count": self.entity_count,
            "graph_node_count": self.graph_node_count,
            "graph_relationship_count": self.graph_relationship_count,
            "knowledge_collection": self.knowledge_collection,
            "entity_collection": self.entity_collection,
            "kb_version": self.kb_version,
            "stages": [
                "preflight",
                "source_validation",
                "knowledge_ingestion",
                "knowledge_qdrant_validation",
                "entity_index",
                "entity_qdrant_validation",
                "deterministic_neo4j_graph",
                "graph_validation",
                "manifest_finalization",
            ],
            "errors": list(self.errors),
        }


def build_full_phase1_plan(source_dir: Path) -> tuple[FullPhase1Plan, dict[str, list[dict[str, Any]]]]:
    """Discover canonical inputs and deterministic outputs without mutation."""

    source_paths = tuple(discover_source_documents(source_dir))
    source_identities = tuple(
        canonical_source_identity(source_path, source_root=source_dir)
        for source_path in source_paths
    )
    errors: list[str] = []
    if not source_paths:
        errors.append(f"No PDF/DOCX/JSON sources found in {source_dir}")
    duplicate_identities = sorted(
        identity
        for identity in set(source_identities)
        if source_identities.count(identity) > 1
    )
    if duplicate_identities:
        errors.append(
            "Canonical source identities are not unique: "
            + ", ".join(duplicate_identities)
        )

    cards = build_entity_cards_from_taxonomy()
    kb_version = get_knowledge_versions()["kb_version"]
    graph_records = build_entity_graph_records(cards, kb_version=kb_version)
    return (
        FullPhase1Plan(
            source_dir=source_dir,
            source_paths=source_paths,
            source_identities=source_identities,
            entity_count=len(cards),
            graph_node_count=len(graph_records["nodes"]),
            graph_relationship_count=len(graph_records["relationships"]),
            knowledge_collection=QDRANT_COLLECTION_NAME,
            entity_collection=get_entity_collection_name(),
            kb_version=kb_version,
            errors=tuple(errors),
        ),
        graph_records,
    )


def _preflight_args(source_dir: Path, manifest_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        source=source_dir,
        manifest_path=manifest_path,
        dry_run=False,
        skip_graph_extraction=False,
        skip_neo4j=False,
        skip_qdrant=False,
        incremental=False,
        force_reingest=False,
        limit_files=None,
    )


async def run_full_phase1(
    *,
    source_dir: Path = SAMPLE_DATA_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the complete Phase 1 contract and return an auditable report.

    ``dry_run`` performs only deterministic source/entity/graph planning and
    does not contact services, embed text, or write a manifest.
    """

    plan, graph_records = build_full_phase1_plan(source_dir)
    if dry_run:
        return {
            "status": "dry_run",
            "passed": not plan.errors,
            "plan": plan.as_dict(),
            "mutated": False,
        }

    report = Phase1ValidationReport(
        sources_expected=len(plan.source_paths),
        sources_accounted_for=0,
    )
    stage_results: dict[str, Any] = {}
    qdrant_client = None
    graph_driver = None
    ingestion_started = False

    try:
        preflight_passed = await run_preflight_checks(
            _preflight_args(source_dir, manifest_path),
            graph_resume_enabled=True,
        )
        if plan.errors:
            report.errors.extend(plan.errors)
        if not preflight_passed:
            report.errors.append("Full Phase 1 preflight failed")
        if plan.errors or not preflight_passed:
            raise RuntimeError("Preflight failed")
        stage_results["preflight"] = "passed"

        stage_results["source_validation"] = "passed"

        ingestion_started = True
        ingestion_stats = await ingest_pipeline(
            source_dir=source_dir,
            dry_run=False,
            limit_files=None,
            limit_chunks=None,
            refresh_markdown=False,
            use_resume=True,
            refresh_graph_cache=False,
            skip_graph_extraction=False,
            skip_neo4j=False,
            skip_qdrant=False,
            incremental=False,
            force_reingest=False,
            manifest_path=manifest_path,
            defer_manifest_completion=True,
            require_zero_graph_errors=True,
        )
        stage_results["knowledge_ingestion"] = ingestion_stats.__dict__.copy()
        if ingestion_stats.parse_errors or ingestion_stats.pdf_files != len(plan.source_paths):
            report.errors.append(
                "Knowledge ingestion did not account for every required source: "
                f"expected={len(plan.source_paths)}, parsed={ingestion_stats.pdf_files}, "
                f"errors={ingestion_stats.parse_errors}"
            )
            raise RuntimeError("Knowledge ingestion completeness failed")

        manifest = load_ingestion_manifest(manifest_path)
        expected_count, expected_by_source, manifest_errors = manifest_knowledge_expectations(
            manifest,
            required_source_identities=set(plan.source_identities),
        )
        report.sources_accounted_for = len(plan.source_identities) - len(manifest_errors)
        if manifest_errors:
            report.errors.extend(manifest_errors)
            raise RuntimeError("Manifest knowledge expectations failed")

        from qdrant_client import AsyncQdrantClient  # type: ignore[import]

        qdrant_client = AsyncQdrantClient(**qdrant_client_kwargs())
        report.knowledge = await reconcile_qdrant_collection(
            client=qdrant_client,
            collection_name=plan.knowledge_collection,
            role="knowledge",
            expected_count=expected_count,
            expected_dimensions=EMBEDDING_DIMENSIONS,
            expected_by_source=expected_by_source,
        )
        stage_results["knowledge_qdrant_validation"] = report.knowledge.as_dict()
        if not report.knowledge.passed:
            raise RuntimeError("Knowledge Qdrant reconciliation failed")

        entity_result = await build_and_index_entities(
            collection_name=plan.entity_collection,
            kb_version=plan.kb_version,
            recreate=False,
        )
        stage_results["entity_index"] = {
            "upserted": entity_result["upserted"],
            "collection": entity_result["collection"],
        }
        report.entities = await reconcile_qdrant_collection(
            client=qdrant_client,
            collection_name=plan.entity_collection,
            role="entity",
            expected_count=plan.entity_count,
            expected_dimensions=EMBEDDING_DIMENSIONS,
            expected_entity_ids=set(entity_result["expected_entity_ids"]),
        )
        stage_results["entity_qdrant_validation"] = report.entities.as_dict()
        if not report.entities.passed:
            raise RuntimeError("Entity Qdrant reconciliation failed")

        graph_driver = get_neo4j_driver()
        await apply_entity_graph_schema(graph_driver)
        graph_upsert = await upsert_entity_graph(graph_driver, graph_records)
        baseline_graph = await validate_entity_graph(graph_driver)
        record_graph = await validate_entity_graph_records(graph_driver, graph_records)
        report.graph = {
            "passed": bool(baseline_graph.get("passed")) and bool(record_graph.get("passed")),
            "upserted": graph_upsert,
            "baseline": baseline_graph,
            "records": record_graph,
        }
        stage_results["graph_validation"] = report.graph
        if not report.graph["passed"]:
            raise RuntimeError("Deterministic Neo4j graph reconciliation failed")

        final = {
            "passed": report.passed,
            "plan": plan.as_dict(),
            "validation": report.as_dict(),
            "stages": stage_results,
        }
        if not final["passed"]:
            raise RuntimeError("Final Phase 1 validation failed")
        finalize_full_phase1_manifest(manifest_path, validation=final["validation"])
        final["status"] = "completed"
        return final
    except Exception as exc:
        if not report.errors:
            report.errors.append(str(exc))
        final = {
            "status": "failed",
            "passed": False,
            "plan": plan.as_dict(),
            "validation": report.as_dict(),
            "stages": stage_results,
        }
        if ingestion_started and manifest_path.exists():
            finalize_full_phase1_manifest(manifest_path, validation=final["validation"])
        return final
    finally:
        if graph_driver is not None:
            await graph_driver.close()
        if qdrant_client is not None:
            await qdrant_client.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Authoritative Full Phase 1 build: preflight -> knowledge ingestion -> "
            "Qdrant reconciliation -> entity index -> Neo4j graph -> final manifest gate."
        ),
    )
    parser.add_argument("--source", type=Path, default=SAMPLE_DATA_DIR, metavar="DIR")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH, metavar="PATH")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the source/entity/graph plan without contacting services or mutating data.",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = await run_full_phase1(
        source_dir=args.source,
        manifest_path=args.manifest_path,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
