"""Canonical Phase 1 orchestration; scripts only delegate to this module."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient

from src.database.vector_store import qdrant_client_kwargs
from src.ingestion.build import compile_knowledge, compute_build_identity
from src.ingestion.embedding import EmbeddingCache
from src.ingestion.index import (
    build_entity_candidate,
    build_knowledge_candidate,
    entity_physical_collection,
    resolve_embeddings,
    seed_embedding_cache_from_collection,
)
from src.ingestion.manifest import build_manifest, load_build_manifest, save_build_manifest
from src.ingestion.parser import load_or_parse_source
from src.ingestion.source_manifest import load_source_manifest, verify_source_files
from src.ingestion.validation import (
    combine_validation_layers,
    validate_compiled_knowledge,
    validate_entity_cards,
    validate_graph_records,
    validate_qdrant_collection,
)
from src.knowledge.entity_cards import build_entity_cards_from_taxonomy, entity_card_to_text
from src.knowledge.graph_schema import build_entity_graph_records


DEFAULT_SOURCE_DIR = Path("sample_data")
DEFAULT_SOURCE_MANIFEST = Path("data/sources/manifest.yaml")
DEFAULT_TAXONOMY = Path("data/taxonomy/drug_aliases.yaml")
DEFAULT_PARSED_CACHE = Path("data/cache/phase1/parsed")
DEFAULT_EMBEDDING_CACHE = Path("data/cache/phase1/embeddings")
DEFAULT_BUILD_MANIFEST = Path("data/phase1_build_manifest.json")
LEGACY_INGESTION_MANIFEST = Path("data/ingestion_manifest.json")


async def prepare_phase1(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
    parsed_cache: Path = DEFAULT_PARSED_CACHE,
) -> dict[str, Any]:
    sources = load_source_manifest(source_manifest_path)
    verify_source_files(sources, source_dir)
    artifacts = {}
    parsed_cache_hits = 0
    for source in sources:
        artifact, cache_hit = await load_or_parse_source(
            source,
            source_dir=source_dir,
            cache_root=parsed_cache,
            llama_cloud_api_key=os.getenv("LLAMA_CLOUD_API_KEY", ""),
        )
        artifacts[source.source_id] = artifact
        parsed_cache_hits += int(cache_hit)
    identity = compute_build_identity(source_manifest_path, taxonomy_path)
    compiled = compile_knowledge(sources, artifacts, identity)
    cards = build_entity_cards_from_taxonomy()
    graph_records = build_entity_graph_records(cards, kb_version=identity.build_id)
    taxonomy_source_ids = {
        str(record.get("source_id"))
        for record in json.loads(Path("data/phase1_method_sources.json").read_text(encoding="utf-8"))["sources"]
    }
    layers = combine_validation_layers(
        validate_compiled_knowledge(compiled, sources),
        validate_entity_cards(cards, taxonomy_source_ids),
        validate_graph_records(graph_records),
    )
    if not layers["passed"]:
        raise RuntimeError(f"Offline Phase 1 validation failed: {layers['errors']}")
    return {
        "sources": sources,
        "artifacts": artifacts,
        "identity": identity,
        "compiled": compiled,
        "cards": cards,
        "graph_records": graph_records,
        "offline_validation": layers,
        "parsed_cache_hits": parsed_cache_hits,
    }


async def build_phase1(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    manifest_path: Path = DEFAULT_BUILD_MANIFEST,
    replace_candidate: bool = False,
) -> dict[str, Any]:
    prepared = await prepare_phase1(source_dir=source_dir)
    compiled = prepared["compiled"]
    cards = prepared["cards"]
    cache = EmbeddingCache(DEFAULT_EMBEDDING_CACHE)
    qdrant = AsyncQdrantClient(**qdrant_client_kwargs())
    legacy_reuse = {"loaded": 0, "unreadable": 0}
    try:
        names = {item.name for item in (await qdrant.get_collections()).collections}
        if "acne_knowledge" in names and LEGACY_INGESTION_MANIFEST.is_file():
            legacy_reuse = await seed_embedding_cache_from_collection(
                qdrant,
                collection_name="acne_knowledge",
                point_ids=_legacy_point_ids(LEGACY_INGESTION_MANIFEST),
                cache=cache,
            )

        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        knowledge_vectors, knowledge_embedding_stats = resolve_embeddings(
            [record["text"] for record in compiled.records],
            cache=cache,
            api_key=api_key,
        )
        entity_vectors, entity_embedding_stats = resolve_embeddings(
            [entity_card_to_text(card) for card in cards],
            cache=cache,
            api_key=api_key,
        )
        knowledge_result = await build_knowledge_candidate(
            compiled,
            knowledge_vectors,
            client=qdrant,
            replace_candidate=replace_candidate,
        )
        entity_result = await build_entity_candidate(
            cards,
            entity_vectors,
            build_id=compiled.identity.build_id,
            taxonomy_hash=compiled.identity.taxonomy_hash,
            client=qdrant,
            replace_candidate=replace_candidate,
        )
        qdrant_knowledge = await validate_qdrant_collection(
            qdrant,
            collection_name=knowledge_result["collection"],
            expected_points=len(compiled.records),
        )
        qdrant_entities = await validate_qdrant_collection(
            qdrant,
            collection_name=entity_result["collection"],
            expected_points=len(cards),
            smoke_query="adapalene",
        )
    finally:
        await qdrant.close()

    validation = combine_validation_layers(
        *prepared["offline_validation"]["layers"], qdrant_knowledge, qdrant_entities
    )
    if not validation["passed"]:
        raise RuntimeError(f"Candidate Phase 1 validation failed: {validation['errors']}")
    graph_records = prepared["graph_records"]
    manifest = build_manifest(
        compiled,
        prepared["sources"],
        prepared["artifacts"],
        knowledge_collection=knowledge_result["collection"],
        entity_collection=entity_result["collection"],
        entity_count=len(cards),
        graph_nodes=len(graph_records["nodes"]),
        graph_relationships=len(graph_records["relationships"]),
        git_commit=_git_commit(),
        cache_stats={
            "parsed_hits": prepared["parsed_cache_hits"],
            "parsed_total": len(prepared["sources"]),
            "legacy_dense_reuse": legacy_reuse,
            "knowledge_embeddings": knowledge_embedding_stats,
            "entity_embeddings": entity_embedding_stats,
        },
        validation=validation,
    )
    save_build_manifest(manifest_path, manifest)
    return manifest


async def validate_phase1(
    *,
    manifest_path: Path = DEFAULT_BUILD_MANIFEST,
    live: bool = True,
) -> dict[str, Any]:
    prepared = await prepare_phase1()
    manifest = load_build_manifest(manifest_path) if manifest_path.is_file() else None
    layers = list(prepared["offline_validation"]["layers"])
    if manifest is not None:
        if manifest.get("build_id") != prepared["identity"].build_id:
            layers.append({"layer": "manifest", "passed": False, "errors": ["build ID mismatch"]})
        else:
            layers.append({"layer": "manifest", "passed": True, "errors": []})
    if live and manifest is not None:
        client = AsyncQdrantClient(**qdrant_client_kwargs())
        try:
            layers.append(
                await validate_qdrant_collection(
                    client,
                    collection_name=manifest["collections"]["knowledge_physical"],
                    expected_points=len(prepared["compiled"].records),
                )
            )
            layers.append(
                await validate_qdrant_collection(
                    client,
                    collection_name=manifest["collections"]["entity_physical"],
                    expected_points=len(prepared["cards"]),
                    smoke_query="adapalene",
                )
            )
        finally:
            await client.close()
    return combine_validation_layers(*layers)


async def phase1_status(manifest_path: Path = DEFAULT_BUILD_MANIFEST) -> dict[str, Any]:
    prepared = await prepare_phase1()
    manifest = load_build_manifest(manifest_path) if manifest_path.is_file() else None
    return {
        "expected_build_id": prepared["identity"].build_id,
        "manifest_present": manifest is not None,
        "manifest_build_id": manifest.get("build_id") if manifest else None,
        "phase1_frozen": bool(manifest and manifest.get("phase1_frozen")),
        "sources": len(prepared["sources"]),
        "knowledge_chunks": len(prepared["compiled"].records),
        "entities": len(prepared["cards"]),
        "graph_nodes": len(prepared["graph_records"]["nodes"]),
        "graph_relationships": len(prepared["graph_records"]["relationships"]),
        "offline_validation": prepared["offline_validation"]["passed"],
    }


def _legacy_point_ids(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    identifiers: list[str] = []
    for entry in (raw.get("documents") or {}).values():
        identifiers.extend(str(value) for value in entry.get("qdrant_point_ids", []))
    return identifiers


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


__all__ = ["build_phase1", "phase1_status", "prepare_phase1", "validate_phase1"]
