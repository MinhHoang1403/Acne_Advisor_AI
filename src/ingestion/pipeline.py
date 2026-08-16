"""Điều phối knowledge build được operator CLI sử dụng.

``prepare_knowledge`` parse/compile/validate offline; ``build_knowledge`` tạo physical
Qdrant candidates và manifest; ``activate_knowledge`` chỉ cut over sau khi có
rollback artifacts và validation live. Neo4j graph/EntityCards ở đây là tài sản
cấu trúc của knowledge build, không phải evidence source trong normal answer
runtime. Muốn đổi thuật toán parse/chunk/embed/index nên sửa owner tương ứng thay
vì thêm logic vào orchestrator này.
"""

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
from src.ingestion.filtering import load_claim_exclusions
from src.ingestion.index import (
    build_entity_candidate,
    build_knowledge_candidate,
    resolve_embeddings,
    seed_embedding_cache_from_collection,
    switch_alias,
)
from src.ingestion.manifest import build_manifest, load_build_manifest, save_build_manifest
from src.ingestion.parser import load_or_parse_source
from src.ingestion.source_manifest import (
    load_source_manifest,
    load_web_record_catalog,
    verify_manifest_support_files,
    verify_source_files,
)
from src.ingestion.validation import (
    combine_validation_layers,
    validate_compiled_knowledge,
    validate_entity_cards,
    validate_graph_records,
    validate_qdrant_collection,
)
from src.knowledge.entity_cards import build_entity_cards_from_taxonomy, entity_card_to_text
from src.knowledge.graph_schema import build_entity_graph_records
from src.knowledge.graph_index import get_neo4j_driver, replace_entity_graph


DEFAULT_SOURCE_DIR = Path("sample_data")
DEFAULT_SOURCE_MANIFEST = Path("data/sources/manifest.yaml")
DEFAULT_TAXONOMY = Path("data/taxonomy/drug_aliases.yaml")
DEFAULT_PARSED_CACHE = Path("data/cache/phase1/parsed")
DEFAULT_EMBEDDING_CACHE = Path("data/cache/phase1/embeddings")
DEFAULT_BUILD_MANIFEST = Path("data/knowledge_build_manifest.json")
LEGACY_INGESTION_MANIFEST = Path("data/ingestion_manifest.json")
KNOWLEDGE_LOGICAL_COLLECTION = "acne_knowledge"
ENTITY_LOGICAL_COLLECTION = "acne_entities"


async def prepare_knowledge(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
    parsed_cache: Path = DEFAULT_PARSED_CACHE,
) -> dict[str, Any]:
    """Tạo artifacts deterministic và chạy validation không ghi datastore live."""
    sources = load_source_manifest(source_manifest_path)
    verify_source_files(sources, source_dir)
    verify_manifest_support_files(sources, source_manifest_path)
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
    web_record_catalogs = {
        source.source_id: load_web_record_catalog(
            source,
            manifest_path=source_manifest_path,
            source_dir=source_dir,
        )
        for source in sources
        if source.record_catalog
    }
    claim_exclusions = tuple(
        action
        for source in sources
        for action in load_claim_exclusions(source, manifest_path=source_manifest_path)
    )
    identity = compute_build_identity(source_manifest_path, taxonomy_path)
    compiled = compile_knowledge(
        sources,
        artifacts,
        identity,
        web_record_catalogs=web_record_catalogs,
        claim_exclusions=claim_exclusions,
    )
    cards = build_entity_cards_from_taxonomy()
    graph_records = build_entity_graph_records(cards, kb_version=identity.build_id)
    taxonomy_source_ids = {
        str(record.get("source_id"))
        for record in json.loads(Path("data/method_sources.json").read_text(encoding="utf-8"))[
            "sources"
        ]
    }
    layers = combine_validation_layers(
        validate_compiled_knowledge(compiled, sources),
        validate_entity_cards(cards, taxonomy_source_ids),
        validate_graph_records(graph_records),
    )
    if not layers["passed"]:
        raise RuntimeError(f"Offline knowledge validation failed: {layers['errors']}")
    return {
        "sources": sources,
        "artifacts": artifacts,
        "identity": identity,
        "compiled": compiled,
        "cards": cards,
        "graph_records": graph_records,
        "offline_validation": layers,
        "parsed_cache_hits": parsed_cache_hits,
        "web_record_catalogs": web_record_catalogs,
        "claim_exclusions": claim_exclusions,
    }


async def inspect_embedding_cache_reuse(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    embedding_cache: Path = DEFAULT_EMBEDDING_CACHE,
) -> dict[str, Any]:
    """Read cache entries only; never invokes an embedding provider or datastore."""

    prepared = await prepare_knowledge(source_dir=source_dir)
    cache = EmbeddingCache(embedding_cache)
    knowledge_hits = sum(
        cache.get(record["text"]) is not None for record in prepared["compiled"].records
    )
    entity_hits = sum(
        cache.get(entity_card_to_text(card)) is not None for card in prepared["cards"]
    )
    result = {
        "passed": False,
        "build_id": prepared["identity"].build_id,
        "parsed": {
            "hits": prepared["parsed_cache_hits"],
            "total": len(prepared["sources"]),
        },
        "knowledge_embeddings": {
            "hits": knowledge_hits,
            "misses": len(prepared["compiled"].records) - knowledge_hits,
        },
        "entity_embeddings": {
            "hits": entity_hits,
            "misses": len(prepared["cards"]) - entity_hits,
        },
        "provider_calls": 0,
    }
    result["passed"] = (
        result["parsed"] == {"hits": 4, "total": 4}
        and result["knowledge_embeddings"] == {"hits": 507, "misses": 5}
        and result["entity_embeddings"] == {"hits": 32, "misses": 0}
    )
    return result


async def build_knowledge(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    manifest_path: Path = DEFAULT_BUILD_MANIFEST,
    replace_candidate: bool = False,
) -> dict[str, Any]:
    """Embed dữ liệu thiếu, tạo candidate collections và lưu build manifest.

    Hàm ghi Qdrant candidates/cache/manifest nhưng chưa đổi logical aliases và
    chưa thay graph live; cutover thuộc ``activate_knowledge``.
    """
    prepared = await prepare_knowledge(source_dir=source_dir)
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
        knowledge_vectors, knowledge_embedding_stats = await resolve_embeddings(
            [record["text"] for record in compiled.records],
            cache=cache,
            api_key=api_key,
        )
        entity_vectors, entity_embedding_stats = await resolve_embeddings(
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
        raise RuntimeError(f"Candidate knowledge validation failed: {validation['errors']}")
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


async def validate_knowledge(
    *,
    manifest_path: Path = DEFAULT_BUILD_MANIFEST,
    live: bool = True,
) -> dict[str, Any]:
    """So build identity offline và tùy chọn kiểm candidate collections live."""
    prepared = await prepare_knowledge()
    manifest = load_build_manifest(manifest_path) if manifest_path.is_file() else None
    layers = list(prepared["offline_validation"]["layers"])
    if manifest is not None:
        if manifest.get("build_id") != prepared["identity"].build_id:
            layers.append(
                {
                    "layer": "manifest",
                    "passed": not live,
                    "errors": ["build ID mismatch"] if live else [],
                    "warnings": ["prepared build is not activated"] if not live else [],
                }
            )
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


async def activate_knowledge(
    *,
    manifest_path: Path = DEFAULT_BUILD_MANIFEST,
    rollback_root: Path,
) -> dict[str, Any]:
    """Activate candidate đã validate sau khi chứng minh rollback artifacts tồn tại.

    Tác động ghi gồm thay Neo4j entity graph theo build ID và chuyển hai Qdrant
    logical aliases sang physical candidates. Đây là operator operation có chủ ý,
    không chạy trong API request path.
    """

    _verify_rollback_artifacts(rollback_root)
    validation = await validate_knowledge(manifest_path=manifest_path, live=True)
    if not validation["passed"]:
        raise RuntimeError(f"Candidate cutover validation failed: {validation['errors']}")

    manifest = load_build_manifest(manifest_path)
    prepared = await prepare_knowledge()
    if manifest["build_id"] != prepared["identity"].build_id:
        raise RuntimeError("Candidate manifest does not match the validated input build identity")

    driver = get_neo4j_driver()
    try:
        graph_result = await replace_entity_graph(
            driver,
            prepared["graph_records"],
            build_id=manifest["build_id"],
        )
    finally:
        await driver.close()

    client = AsyncQdrantClient(**qdrant_client_kwargs())
    try:
        collections = {item.name for item in (await client.get_collections()).collections}
        knowledge_physical = manifest["collections"]["knowledge_physical"]
        entity_physical = manifest["collections"]["entity_physical"]
        missing = sorted({knowledge_physical, entity_physical} - collections)
        if missing:
            raise RuntimeError(f"Candidate collection missing before cutover: {missing}")

        # Logical name có thể đang là physical collection. Chỉ xóa nó tại cutover
        # sau khi rollback snapshot đã được kiểm tra ở đầu hàm.
        if KNOWLEDGE_LOGICAL_COLLECTION in collections:
            await client.delete_collection(KNOWLEDGE_LOGICAL_COLLECTION)
        if ENTITY_LOGICAL_COLLECTION in collections:
            await client.delete_collection(ENTITY_LOGICAL_COLLECTION)

        await switch_alias(
            client,
            alias_name=KNOWLEDGE_LOGICAL_COLLECTION,
            target_collection=knowledge_physical,
        )
        await switch_alias(
            client,
            alias_name=ENTITY_LOGICAL_COLLECTION,
            target_collection=entity_physical,
        )
        knowledge_alias_validation = await validate_qdrant_collection(
            client,
            collection_name=KNOWLEDGE_LOGICAL_COLLECTION,
            expected_points=manifest["counts"]["knowledge_chunks"],
        )
        entity_alias_validation = await validate_qdrant_collection(
            client,
            collection_name=ENTITY_LOGICAL_COLLECTION,
            expected_points=manifest["counts"]["entities"],
            smoke_query="adapalene",
        )
    finally:
        await client.close()

    cutover_validation = combine_validation_layers(
        validation,
        knowledge_alias_validation,
        entity_alias_validation,
        graph_result["validation"],
    )
    if not cutover_validation["passed"]:
        raise RuntimeError(f"Post-cutover validation failed: {cutover_validation['errors']}")
    manifest["status"] = "activated"
    manifest["activation"] = {
        "rollback_root": rollback_root.as_posix(),
        "knowledge_alias": KNOWLEDGE_LOGICAL_COLLECTION,
        "entity_alias": ENTITY_LOGICAL_COLLECTION,
        "graph": graph_result,
        "validation": cutover_validation,
    }
    save_build_manifest(manifest_path, manifest)
    return manifest


async def knowledge_status(manifest_path: Path = DEFAULT_BUILD_MANIFEST) -> dict[str, Any]:
    """Tính expected build từ source hiện tại và đối chiếu manifest, không index."""
    prepared = await prepare_knowledge()
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


def _verify_rollback_artifacts(root: Path) -> None:
    qdrant_snapshots = list((root / "qdrant").glob("*.snapshot"))
    neo4j_store = root / "neo4j" / "data" / "databases"
    if len(qdrant_snapshots) < 2 or any(path.stat().st_size <= 0 for path in qdrant_snapshots):
        raise RuntimeError(f"Rollback requires two readable Qdrant snapshots under {root}")
    if not neo4j_store.is_dir() or not any(path.is_file() for path in neo4j_store.rglob("*")):
        raise RuntimeError(f"Rollback requires a non-empty Neo4j cold backup under {root}")


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


__all__ = [
    "activate_knowledge",
    "build_knowledge",
    "inspect_embedding_cache_reuse",
    "knowledge_status",
    "prepare_knowledge",
    "validate_knowledge",
]
