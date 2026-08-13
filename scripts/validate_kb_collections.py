#!/usr/bin/env python3
"""Validate chunk/entity Qdrant collections after a clean KB rebuild."""

from __future__ import annotations

import argparse
import asyncio
import json
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

from src.database.vector_store import qdrant_client_kwargs  # noqa: E402
from src.knowledge.entity_index import (  # noqa: E402
    get_chunk_collection_name,
    get_entity_collection_name,
)
from src.knowledge.versioning import (  # noqa: E402
    get_embedding_metadata,
    validate_embedding_config_compatibility,
)


COMMON_REQUIRED_PAYLOAD_FIELDS = (
    "embedding_provider",
    "embedding_model",
    "embedding_dimensions",
    "kb_version",
)
ENTITY_REQUIRED_PAYLOAD_FIELDS = (
    "entity_type",
    "canonical_name",
    "taxonomy_version",
    "entity_schema_version",
)


def get_default_chunk_collection_name() -> str:
    return get_chunk_collection_name()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Acne Advisor AI Qdrant KB collections without writing data.",
    )
    parser.add_argument(
        "--chunk-collection",
        default=get_default_chunk_collection_name(),
        help=(
            "Chunk collection name. Defaults to CHUNK_QDRANT_COLLECTION_NAME "
            "or the canonical QDRANT_COLLECTION_NAME alias."
        ),
    )
    parser.add_argument(
        "--entity-collection",
        default=get_entity_collection_name(),
        help="Entity collection name. Defaults to ENTITY_QDRANT_COLLECTION_NAME.",
    )
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument(
        "--strict",
        choices=["false", "true"],
        default="false",
        help="When true, missing fields or mismatches return a non-zero exit code.",
    )
    return parser.parse_args(argv)


async def validate_collections(args: argparse.Namespace) -> tuple[dict[str, Any], list[str], list[str]]:
    from qdrant_client import AsyncQdrantClient  # type: ignore[import]

    client = AsyncQdrantClient(**qdrant_client_kwargs())
    try:
        collections = await client.get_collections()
        existing = {collection.name for collection in collections.collections}

        warnings: list[str] = []
        errors: list[str] = []
        report: dict[str, Any] = {
            "qdrant_reachable": True,
            "expected_embedding": get_embedding_metadata(),
            "chunk_collection": args.chunk_collection,
            "entity_collection": args.entity_collection,
            "collections": {},
        }

        for role, collection_name in (
            ("chunk", args.chunk_collection),
            ("entity", args.entity_collection),
        ):
            if collection_name not in existing:
                errors.append(f"{role} collection {collection_name!r} does not exist")
                continue

            collection_report = await _inspect_collection(
                client=client,
                collection_name=collection_name,
                role=role,
                sample_size=max(0, args.sample_size),
            )
            report["collections"][role] = collection_report
            warnings.extend(collection_report["warnings"])
            errors.extend(collection_report["errors"])

        chunk_sample_config = _first_payload_config(report, "chunk")
        entity_sample_config = _first_payload_config(report, "entity")
        if chunk_sample_config and entity_sample_config:
            compatibility_issues = validate_embedding_config_compatibility(
                chunk_sample_config,
                entity_sample_config,
            )
            report["embedding_compatibility"] = {
                "compatible": not compatibility_issues,
                "issues": compatibility_issues,
            }
            warnings.extend(compatibility_issues)
        else:
            report["embedding_compatibility"] = {
                "compatible": False,
                "issues": ["Unable to compare embedding metadata from sample payloads."],
            }
            warnings.append("Unable to compare embedding metadata from sample payloads.")

        return report, warnings, errors
    finally:
        await client.close()


async def _inspect_collection(
    client: Any,
    collection_name: str,
    role: str,
    sample_size: int,
) -> dict[str, Any]:
    info = await client.get_collection(collection_name=collection_name)
    params = info.config.params
    schema_report = inspect_qdrant_schema(params)
    points, _ = await client.scroll(
        collection_name=collection_name,
        limit=sample_size,
        with_payload=True,
        with_vectors=False,
    )
    payloads = [point.payload or {} for point in points]
    warnings, errors = validate_sample_payloads(payloads, role)

    expected_dimensions = int(get_embedding_metadata()["embedding_dimensions"])
    if not schema_report["has_dense"]:
        errors.append(f"{role} collection missing dense vector 'dense'")
    if schema_report["dense_size"] != expected_dimensions:
        errors.append(
            f"{role} collection dense dimension {schema_report['dense_size']} != {expected_dimensions}"
        )
    if not schema_report["has_bm25"]:
        warnings.append(f"{role} collection missing sparse vector 'bm25'")

    return {
        "collection": collection_name,
        "points_count": getattr(info, "points_count", None),
        "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
        **schema_report,
        "sample_size": len(payloads),
        "sample_payload_keys": [
            sorted(payload.keys()) for payload in payloads
        ],
        "sample_payload_configs": [
            _payload_config(payload) for payload in payloads
        ],
        "warnings": warnings,
        "errors": errors,
    }


def inspect_qdrant_schema(params: Any) -> dict[str, Any]:
    if isinstance(params, dict):
        vectors_config = params.get("vectors")
        sparse_vectors_config = params.get("sparse_vectors")
    else:
        vectors_config = getattr(params, "vectors", None)
        sparse_vectors_config = getattr(params, "sparse_vectors", None)

    dense_config = _get_named_config(vectors_config, "dense")
    bm25_config = _get_named_config(sparse_vectors_config, "bm25")
    dense_size = (
        dense_config.get("size")
        if isinstance(dense_config, dict)
        else getattr(dense_config, "size", None)
    ) if dense_config is not None else None

    return {
        "has_dense": dense_config is not None,
        "dense_vector_name": "dense" if dense_config is not None else None,
        "dense_size": dense_size,
        "has_bm25": bm25_config is not None,
        "sparse_vector_name": "bm25" if bm25_config is not None else None,
    }


def validate_sample_payloads(payloads: list[dict[str, Any]], role: str) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    if not payloads:
        warnings.append(f"{role} collection returned no sample payloads")
        return warnings, errors

    required = list(COMMON_REQUIRED_PAYLOAD_FIELDS)
    if role == "entity":
        required.extend(ENTITY_REQUIRED_PAYLOAD_FIELDS)

    for index, payload in enumerate(payloads):
        missing = [field for field in required if field not in payload]
        if missing:
            errors.append(
                f"{role} sample {index} missing payload fields: {', '.join(missing)}"
            )
    return warnings, errors


def _payload_config(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        field: payload.get(field)
        for field in COMMON_REQUIRED_PAYLOAD_FIELDS
    }


def _first_payload_config(report: dict[str, Any], role: str) -> dict[str, Any] | None:
    collection_report = report.get("collections", {}).get(role, {})
    configs = collection_report.get("sample_payload_configs") or []
    if not configs:
        return None
    return configs[0]


def _get_named_config(config: Any, name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    if hasattr(config, "get"):
        return config.get(name)
    return None


async def main() -> int:
    args = parse_args()
    strict = args.strict.lower() == "true"
    report, warnings, errors = await validate_collections(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")

    if errors or (strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
