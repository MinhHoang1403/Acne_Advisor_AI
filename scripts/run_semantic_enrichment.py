#!/usr/bin/env python3
"""Optional Ollama document-graph enrichment for a validated core Phase 1 KB.

This job reads existing Qdrant knowledge payloads instead of parsing documents or
rebuilding embeddings. It never changes the validity of the core Phase 1 state.
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

from scripts.ingest_knowledge import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    OLLAMA_MODEL,
    QDRANT_COLLECTION_NAME,
    SAMPLE_DATA_DIR,
    GraphPayload,
    IngestionStats,
    SemanticChunk,
    canonical_source_identity,
    discover_source_documents,
    preflight_neo4j,
    preflight_ollama,
    preflight_qdrant,
    qdrant_client_kwargs,
    save_ingestion_manifest,
    semantic_enrichment_fingerprint,
    stage3_and_optional_neo4j_incremental,
    update_semantic_enrichment_manifest,
    load_ingestion_manifest,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexedChunk:
    point_id: Any
    source_path: str
    chunk: SemanticChunk


def build_semantic_enrichment_plan(source_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Build a non-mutating plan from source identities and current manifest state."""

    source_paths = discover_source_documents(source_dir)
    source_identities = [
        canonical_source_identity(source_path, source_root=source_dir)
        for source_path in source_paths
    ]
    manifest = load_ingestion_manifest(manifest_path)
    core_state = manifest.get("core_phase1") or {}
    if core_state.get("status") == "not_validated":
        core_state = manifest.get("full_phase1_validation") or core_state
    return {
        "source_dir": str(source_dir),
        "sources": [str(path) for path in source_paths],
        "source_identities": source_identities,
        "knowledge_collection": QDRANT_COLLECTION_NAME,
        "semantic_enrichment": {
            "status": "planned",
            "provider": "ollama",
            "model": OLLAMA_MODEL,
            "fingerprint": semantic_enrichment_fingerprint(),
            "requires_ollama": True,
            "core_phase1_independent": True,
        },
        "core_phase1_status": core_state.get("status", "not_validated"),
        "stages": [
            "core_manifest_validation",
            "ollama_preflight",
            "indexed_chunk_load",
            "semantic_graph_extraction",
            "optional_document_neo4j_graph",
            "optional_qdrant_graph_nodes",
            "semantic_enrichment_manifest_report",
        ],
        "errors": [] if source_paths else [f"No PDF/DOCX/JSON sources found in {source_dir}"],
    }


def _assert_core_phase1_ready(manifest: dict[str, Any], source_identities: set[str]) -> None:
    core_statuses = {
        str((manifest.get(key) or {}).get("status") or "")
        for key in ("core_phase1", "full_phase1_validation")
        if isinstance(manifest.get(key), dict)
    }
    if not core_statuses.intersection({"completed_validated", "completed"}):
        raise RuntimeError("Core Phase 1 is not validated; run the canonical core workflow first.")

    documents = manifest.get("documents") or {}
    found = {
        str(entry.get("source_identity") or "")
        for entry in documents.values()
        if isinstance(entry, dict)
        and str(entry.get("status") or "") == "completed"
        and str(entry.get("core_phase1_status") or "completed_validated")
        in {"completed_validated", ""}
    }
    missing = sorted(source_identities - found)
    if missing:
        raise RuntimeError(
            "Core Phase 1 manifest is incomplete for: " + ", ".join(missing)
        )


async def load_indexed_chunks(
    *,
    source_identities: set[str],
) -> list[IndexedChunk]:
    """Read eligible core chunks from Qdrant without parsing or embedding again."""

    from qdrant_client import AsyncQdrantClient  # type: ignore[import]

    client = AsyncQdrantClient(**qdrant_client_kwargs())
    offset: Any = None
    indexed_chunks: list[IndexedChunk] = []
    try:
        while True:
            points, offset = await client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(getattr(point, "payload", {}) or {})
                if str(payload.get("source_identity") or "") not in source_identities:
                    continue
                text = str(payload.get("text") or "")
                chunk_id = str(payload.get("chunk_id") or "")
                if not text or not chunk_id:
                    raise RuntimeError("Indexed knowledge payload is missing text or chunk_id")
                chunk = SemanticChunk(
                    source_file=str(payload.get("source_file") or payload.get("source_path") or ""),
                    chunk_index=int(payload.get("chunk_index") or 0),
                    text=text,
                    header_path=str(payload.get("header") or ""),
                    metadata=payload,
                )
                if chunk.chunk_id != chunk_id:
                    raise RuntimeError(
                        f"Indexed chunk identity mismatch for {payload.get('source_path')!r}"
                    )
                indexed_chunks.append(
                    IndexedChunk(
                        point_id=getattr(point, "id", chunk.qdrant_point_id),
                        source_path=str(payload.get("source_path") or ""),
                        chunk=chunk,
                    )
                )
            if offset is None:
                break
    finally:
        await client.close()

    if not indexed_chunks:
        raise RuntimeError("No indexed core chunks were found for the requested sources")
    return indexed_chunks


async def update_qdrant_graph_nodes(
    indexed_chunks: list[IndexedChunk],
    payloads: list[GraphPayload],
) -> int:
    """Attach optional graph-node hints without changing vectors or core fields."""

    from qdrant_client import AsyncQdrantClient  # type: ignore[import]

    payloads_by_chunk = {payload.chunk_id: payload for payload in payloads}
    client = AsyncQdrantClient(**qdrant_client_kwargs())
    updated = 0
    try:
        for indexed in indexed_chunks:
            payload = payloads_by_chunk.get(indexed.chunk.chunk_id)
            if payload is None or payload.extraction_error:
                continue
            await client.set_payload(
                collection_name=QDRANT_COLLECTION_NAME,
                payload={
                    "graph_nodes": [node.name for node in payload.nodes],
                    "semantic_enrichment_status": "completed",
                    "semantic_enrichment_model": OLLAMA_MODEL,
                    "semantic_enrichment_fingerprint": semantic_enrichment_fingerprint(),
                },
                points=[indexed.point_id],
            )
            updated += 1
    finally:
        await client.close()
    return updated


def _document_results(
    indexed_chunks: list[IndexedChunk],
    payloads: list[GraphPayload],
) -> dict[str, dict[str, Any]]:
    payloads_by_chunk = {payload.chunk_id: payload for payload in payloads}
    summaries: dict[str, dict[str, Any]] = {}
    for indexed in indexed_chunks:
        summary = summaries.setdefault(
            indexed.source_path,
            {"status": "completed", "chunks": 0, "errors": 0, "nodes": 0, "edges": 0},
        )
        summary["chunks"] += 1
        payload = payloads_by_chunk.get(indexed.chunk.chunk_id)
        if payload is None or payload.extraction_error:
            summary["status"] = "failed"
            summary["errors"] += 1
            continue
        summary["nodes"] += len(payload.nodes)
        summary["edges"] += len(payload.edges)
    return summaries


async def run_semantic_enrichment(
    *,
    source_dir: Path = SAMPLE_DATA_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    dry_run: bool = False,
    skip_neo4j: bool = False,
    skip_qdrant_graph_nodes: bool = False,
    refresh_graph_cache: bool = False,
) -> dict[str, Any]:
    """Run optional Ollama enrichment while preserving valid core manifest state."""

    plan = build_semantic_enrichment_plan(source_dir, manifest_path)
    if dry_run:
        return {"status": "dry_run", "passed": not plan["errors"], "plan": plan, "mutated": False}
    if plan["errors"]:
        return {"status": "failed", "passed": False, "plan": plan, "errors": plan["errors"]}

    manifest = load_ingestion_manifest(manifest_path)
    source_identities = set(plan["source_identities"])
    indexed_chunks: list[IndexedChunk] = []
    stats = IngestionStats()
    try:
        _assert_core_phase1_ready(manifest, source_identities)
        if not await preflight_ollama():
            raise RuntimeError("Ollama preflight failed for requested semantic enrichment")
        if not await preflight_qdrant():
            raise RuntimeError("Qdrant preflight failed for requested semantic enrichment")
        if not skip_neo4j and not await preflight_neo4j():
            raise RuntimeError("Neo4j preflight failed for requested semantic enrichment")

        update_semantic_enrichment_manifest(manifest, status="running")
        save_ingestion_manifest(manifest_path, manifest)

        indexed_chunks = await load_indexed_chunks(source_identities=source_identities)
        chunks = [indexed.chunk for indexed in indexed_chunks]
        payloads = await stage3_and_optional_neo4j_incremental(
            chunks=chunks,
            dry_run=False,
            use_resume=True,
            refresh_graph_cache=refresh_graph_cache,
            skip_graph_extraction=False,
            skip_neo4j=skip_neo4j,
            stats=stats,
        )
        document_results = _document_results(indexed_chunks, payloads)
        extraction_errors = sum(1 for payload in payloads if payload.extraction_error)
        graph_nodes_updated = 0
        if not skip_qdrant_graph_nodes:
            graph_nodes_updated = await update_qdrant_graph_nodes(indexed_chunks, payloads)

        report = {
            "chunks_considered": len(indexed_chunks),
            "graph_cache_hits": stats.graph_cache_hits,
            "llm_calls": stats.graph_cache_misses,
            "successful_extractions": len(payloads) - extraction_errors,
            "extraction_errors": extraction_errors,
            "document_nodes": sum(len(payload.nodes) for payload in payloads),
            "document_edges": sum(len(payload.edges) for payload in payloads),
            "neo4j_enabled": not skip_neo4j,
            "qdrant_graph_nodes_updated": graph_nodes_updated,
        }
        successful_extractions = len(payloads) - extraction_errors
        if extraction_errors == 0:
            status = "completed"
        elif successful_extractions > 0:
            status = "completed_with_warnings"
            report["warning"] = (
                "Some chunks could not be semantically extracted and remain retryable."
            )
        else:
            status = "failed"
        passed = status != "failed"
        update_semantic_enrichment_manifest(
            manifest,
            status=status,
            report=report,
            document_results=document_results,
        )
        save_ingestion_manifest(manifest_path, manifest)
        return {
            "status": status,
            "passed": passed,
            "plan": plan,
            "report": report,
        }
    except Exception as exc:
        report = {
            "error": str(exc),
            "chunks_considered": len(indexed_chunks),
            "graph_cache_hits": stats.graph_cache_hits,
            "llm_calls": stats.graph_cache_misses,
        }
        document_results = {
            indexed.source_path: {"status": "failed", "error": str(exc)}
            for indexed in indexed_chunks
        }
        update_semantic_enrichment_manifest(
            manifest,
            status="failed",
            report=report,
            document_results=document_results,
        )
        save_ingestion_manifest(manifest_path, manifest)
        return {"status": "failed", "passed": False, "plan": plan, "report": report}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optional Ollama document semantic enrichment for an already validated core Phase 1 KB."
        )
    )
    parser.add_argument("--source", type=Path, default=SAMPLE_DATA_DIR, metavar="DIR")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH, metavar="PATH")
    parser.add_argument("--dry-run", action="store_true", help="Print the optional enrichment plan only.")
    parser.add_argument("--skip-neo4j", action="store_true", help="Do not persist document graph facts to Neo4j.")
    parser.add_argument(
        "--skip-qdrant-graph-nodes",
        action="store_true",
        help="Do not update optional graph_nodes hints in existing Qdrant payloads.",
    )
    parser.add_argument(
        "--refresh-graph-cache",
        action="store_true",
        help="Ignore valid graph cache entries and call Ollama again.",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = await run_semantic_enrichment(
        source_dir=args.source,
        manifest_path=args.manifest_path,
        dry_run=args.dry_run,
        skip_neo4j=args.skip_neo4j,
        skip_qdrant_graph_nodes=args.skip_qdrant_graph_nodes,
        refresh_graph_cache=args.refresh_graph_cache,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
