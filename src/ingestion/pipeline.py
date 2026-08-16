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
    entity_physical_collection,
    knowledge_physical_collection,
    resolve_embeddings,
    seed_embedding_cache_from_collection,
    switch_alias,
)
from src.ingestion.manifest import (
    build_manifest,
    load_build_manifest,
    save_build_manifest,
    validate_build_id,
)
from src.ingestion.parser import (
    artifact_path,
    load_or_parse_source,
    load_parsed_artifact,
)
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
EXPECTED_KNOWLEDGE_POINTS = 512
EXPECTED_ENTITY_POINTS = 32
EXPECTED_GRAPH_NODES = 32
EXPECTED_GRAPH_RELATIONSHIPS = 27


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
    prepared = _compile_prepared_knowledge(
        sources=sources,
        artifacts=artifacts,
        source_dir=source_dir,
        source_manifest_path=source_manifest_path,
        taxonomy_path=taxonomy_path,
    )
    return {**prepared, "parsed_cache_hits": parsed_cache_hits}


def _compile_prepared_knowledge(
    *,
    sources: tuple[Any, ...],
    artifacts: dict[str, Any],
    source_dir: Path,
    source_manifest_path: Path,
    taxonomy_path: Path,
) -> dict[str, Any]:
    """Compile artifacts supplied by either normal preparation or cache inspection."""

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
        "web_record_catalogs": web_record_catalogs,
        "claim_exclusions": claim_exclusions,
    }


async def inspect_embedding_cache_reuse(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
    parsed_cache: Path = DEFAULT_PARSED_CACHE,
    embedding_cache: Path = DEFAULT_EMBEDDING_CACHE,
) -> dict[str, Any]:
    """Inspect verified parser/vector cache entries without repairing any miss."""

    sources = load_source_manifest(source_manifest_path)
    verify_source_files(sources, source_dir)
    verify_manifest_support_files(sources, source_manifest_path)
    artifacts = {}
    missing_source_ids: list[str] = []
    for source in sources:
        try:
            artifact = load_parsed_artifact(artifact_path(parsed_cache, source), source)
        except (UnicodeError, ValueError):
            artifact = None
        if artifact is None:
            missing_source_ids.append(source.source_id)
        else:
            artifacts[source.source_id] = artifact

    parsed_stats = {
        "hits": len(artifacts),
        "misses": len(missing_source_ids),
        "total": len(sources),
        "missing_or_invalid_source_ids": missing_source_ids,
    }
    if missing_source_ids:
        skipped = {
            "inspected": False,
            "hits": 0,
            "misses": 0,
            "total": 0,
            "skipped_reason": "parsed_cache_incomplete",
        }
        return {
            "passed": False,
            "build_id": None,
            "parsed": parsed_stats,
            "knowledge_embeddings": dict(skipped),
            "entity_embeddings": dict(skipped),
            "provider_calls": 0,
        }

    prepared = _compile_prepared_knowledge(
        sources=sources,
        artifacts=artifacts,
        source_dir=source_dir,
        source_manifest_path=source_manifest_path,
        taxonomy_path=taxonomy_path,
    )
    cache = EmbeddingCache(embedding_cache)
    knowledge_hits = sum(
        cache.get(record["text"]) is not None for record in prepared["compiled"].records
    )
    entity_hits = sum(
        cache.get(entity_card_to_text(card)) is not None for card in prepared["cards"]
    )
    result = {
        "passed": prepared["offline_validation"]["passed"],
        "build_id": prepared["identity"].build_id,
        "parsed": parsed_stats,
        "knowledge_embeddings": {
            "inspected": True,
            "hits": knowledge_hits,
            "misses": len(prepared["compiled"].records) - knowledge_hits,
            "total": len(prepared["compiled"].records),
        },
        "entity_embeddings": {
            "inspected": True,
            "hits": entity_hits,
            "misses": len(prepared["cards"]) - entity_hits,
            "total": len(prepared["cards"]),
        },
        "provider_calls": 0,
    }
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


async def finalize_knowledge_activation(
    *,
    manifest_path: Path = DEFAULT_BUILD_MANIFEST,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
    parsed_cache: Path = DEFAULT_PARSED_CACHE,
    embedding_cache: Path = DEFAULT_EMBEDDING_CACHE,
) -> dict[str, Any]:
    """Finalize an activated build after read-only identity and live-state checks."""

    manifest = load_build_manifest(manifest_path)
    if manifest.get("status") != "activated":
        raise RuntimeError("Knowledge manifest must have status activated before finalization")

    try:
        configured_build = validate_build_id(os.getenv("KB_VERSION"))
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    manifest_build = manifest.get("build_id")
    if manifest_build != configured_build:
        raise RuntimeError("Activated manifest build_id does not match configured KB_VERSION")

    identity = compute_build_identity(source_manifest_path, taxonomy_path)
    if manifest_build != identity.build_id:
        raise RuntimeError("Activated manifest does not match current prepared build identity")

    expected_counts = {
        "knowledge_chunks": EXPECTED_KNOWLEDGE_POINTS,
        "entities": EXPECTED_ENTITY_POINTS,
        "graph_nodes": EXPECTED_GRAPH_NODES,
        "graph_relationships": EXPECTED_GRAPH_RELATIONSHIPS,
    }
    counts = manifest.get("counts") or {}
    count_errors = [
        f"{name}: expected {expected}, got {counts.get(name)}"
        for name, expected in expected_counts.items()
        if counts.get(name) != expected
    ]
    if count_errors:
        raise RuntimeError(f"Activation finalization manifest counts failed: {count_errors}")

    cache_validation = await inspect_embedding_cache_reuse(
        source_dir=source_dir,
        source_manifest_path=source_manifest_path,
        taxonomy_path=taxonomy_path,
        parsed_cache=parsed_cache,
        embedding_cache=embedding_cache,
    )
    cache_totals_match = (
        cache_validation.get("passed") is True
        and cache_validation.get("build_id") == manifest_build
        and cache_validation.get("provider_calls") == 0
        and (cache_validation.get("parsed") or {}).get("misses") == 0
        and (cache_validation.get("knowledge_embeddings") or {}).get("total")
        == EXPECTED_KNOWLEDGE_POINTS
        and (cache_validation.get("entity_embeddings") or {}).get("total")
        == EXPECTED_ENTITY_POINTS
    )
    if not cache_totals_match:
        raise RuntimeError(
            "Read-only prepared knowledge validation failed before finalization"
        )

    live_validation = await _validate_activated_live_state(manifest)
    if live_validation.get("passed") is not True:
        raise RuntimeError(
            "Live knowledge validation failed before finalization: "
            f"{live_validation.get('errors', [])}"
        )

    finalized_manifest = dict(manifest)
    finalized_manifest["phase1_frozen"] = True
    save_build_manifest(manifest_path, finalized_manifest)
    return {
        "passed": True,
        "build_id": manifest_build,
        "phase1_frozen": True,
        "validation": {
            "prepared_cache": cache_validation,
            "live": live_validation,
        },
    }


async def _validate_activated_live_state(manifest: dict[str, Any]) -> dict[str, Any]:
    """Read aliases, Qdrant schema/counts and Neo4j identity/counts only."""

    build_id = str(manifest["build_id"])
    collections = manifest.get("collections") or {}
    expected_aliases = {
        KNOWLEDGE_LOGICAL_COLLECTION: knowledge_physical_collection(build_id),
        ENTITY_LOGICAL_COLLECTION: entity_physical_collection(build_id),
    }
    alias_errors: list[str] = []
    manifest_aliases = {
        str(collections.get("knowledge_logical")): str(
            collections.get("knowledge_physical")
        ),
        str(collections.get("entity_logical")): str(collections.get("entity_physical")),
    }
    if manifest_aliases != expected_aliases:
        alias_errors.append(
            "Manifest logical collection targets do not identify the activated build"
        )
    qdrant_layers: list[dict[str, Any]] = []
    client = AsyncQdrantClient(**qdrant_client_kwargs())
    try:
        aliases = {
            str(item.alias_name): str(item.collection_name)
            for item in (await client.get_aliases()).aliases
        }
        for alias_name, expected_target in expected_aliases.items():
            if aliases.get(alias_name) != expected_target:
                alias_errors.append(
                    f"Qdrant alias {alias_name} does not resolve to {expected_target}"
                )
        if not alias_errors:
            qdrant_layers.extend(
                [
                    await validate_qdrant_collection(
                        client,
                        collection_name=KNOWLEDGE_LOGICAL_COLLECTION,
                        expected_points=EXPECTED_KNOWLEDGE_POINTS,
                    ),
                    await validate_qdrant_collection(
                        client,
                        collection_name=ENTITY_LOGICAL_COLLECTION,
                        expected_points=EXPECTED_ENTITY_POINTS,
                        smoke_query="adapalene",
                    ),
                ]
            )
    finally:
        await client.close()

    alias_layer = {
        "layer": "qdrant_aliases",
        "passed": not alias_errors,
        "errors": alias_errors,
        "expected": expected_aliases,
    }
    graph_layer = await _inspect_activated_graph_state(build_id)
    return combine_validation_layers(alias_layer, *qdrant_layers, graph_layer)


async def _inspect_activated_graph_state(build_id: str) -> dict[str, Any]:
    """Read global graph counts and ensure every record belongs to ``build_id``."""

    driver = get_neo4j_driver()
    try:
        async with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            node_result = await session.run(
                "MATCH (n) RETURN count(n) AS total, "
                "sum(CASE WHEN n.kb_version = $build_id THEN 1 ELSE 0 END) AS matching",
                build_id=build_id,
            )
            relationship_result = await session.run(
                "MATCH ()-[r]->() RETURN count(r) AS total, "
                "sum(CASE WHEN r.kb_version = $build_id THEN 1 ELSE 0 END) AS matching",
                build_id=build_id,
            )
            node_record = await node_result.single()
            relationship_record = await relationship_result.single()
    finally:
        await driver.close()

    nodes = int((node_record or {}).get("total", 0))
    matching_nodes = int((node_record or {}).get("matching", 0))
    relationships = int((relationship_record or {}).get("total", 0))
    matching_relationships = int((relationship_record or {}).get("matching", 0))
    errors = []
    if nodes != EXPECTED_GRAPH_NODES:
        errors.append(f"Neo4j node count mismatch: expected {EXPECTED_GRAPH_NODES}, got {nodes}")
    if relationships != EXPECTED_GRAPH_RELATIONSHIPS:
        errors.append(
            "Neo4j relationship count mismatch: "
            f"expected {EXPECTED_GRAPH_RELATIONSHIPS}, got {relationships}"
        )
    if matching_nodes != nodes or matching_relationships != relationships:
        errors.append("Neo4j build identity does not match the activated manifest")
    return {
        "layer": "neo4j_build",
        "passed": not errors,
        "errors": errors,
        "build_id": build_id,
        "nodes": nodes,
        "relationships": relationships,
    }


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
    "finalize_knowledge_activation",
    "inspect_embedding_cache_reuse",
    "knowledge_status",
    "prepare_knowledge",
    "validate_knowledge",
]
