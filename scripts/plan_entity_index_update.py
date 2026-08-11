#!/usr/bin/env python3
"""Dry-run planner for incremental entity-card Qdrant updates."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import defaultdict
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
from src.knowledge.entity_cards import build_entity_cards_from_taxonomy  # noqa: E402
from src.knowledge.entity_index import (  # noqa: E402
    build_entity_point_payload,
    entity_identity_key,
    entity_point_id,
    get_entity_collection_name,
)
from src.knowledge.normalizer import DEFAULT_TAXONOMY_PATH, DrugEntityNormalizer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan entity-card index changes without Qdrant writes.")
    parser.add_argument("--taxonomy-path", default=str(DEFAULT_TAXONOMY_PATH))
    parser.add_argument("--kb-version", default="acne_kb_v1")
    parser.add_argument(
        "--from-qdrant",
        action="store_true",
        help="Read existing entity points from Qdrant for compatibility planning. Read-only.",
    )
    parser.add_argument(
        "--collection",
        default=get_entity_collection_name(),
        help="Entity Qdrant collection used only when --from-qdrant is set.",
    )
    return parser.parse_args(argv)


def build_entity_index_update_plan(
    *,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    kb_version: str = "acne_kb_v1",
    existing_points: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    proposed_normalizer = DrugEntityNormalizer(taxonomy_path)
    proposed_cards = build_entity_cards_from_taxonomy(proposed_normalizer)
    proposed_by_identity = {entity_identity_key(card): card for card in proposed_cards}

    existing_records = (
        _existing_records_from_points(existing_points)
        if existing_points is not None
        else _existing_records_from_current_cards(kb_version=kb_version)
    )
    existing_by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    point_id_to_identity: dict[str, set[str]] = defaultdict(set)
    for record in existing_records:
        identity = record.get("identity")
        point_id = record.get("point_id")
        if not identity:
            continue
        existing_by_identity[str(identity)].append(record)
        if point_id:
            point_id_to_identity[str(point_id)].add(str(identity))

    conflicts = _detect_existing_conflicts(existing_by_identity, point_id_to_identity)
    conflicted_identities = {str(conflict.get("identity")) for conflict in conflicts if conflict.get("identity")}

    unchanged: list[str] = []
    new: list[str] = []
    updated: list[str] = []
    reused: list[dict[str, str]] = []
    preview_payloads: list[dict[str, Any]] = []

    for identity, proposed in sorted(proposed_by_identity.items()):
        if identity in conflicted_identities:
            continue
        matches = existing_by_identity.get(identity, [])
        if not matches:
            new.append(identity)
            planned_point_id = entity_point_id(proposed, kb_version=kb_version)
        else:
            match = matches[0]
            planned_point_id = str(match["point_id"])
            reused.append({"identity": identity, "point_id": planned_point_id})
            proposed_payload = build_entity_point_payload(
                proposed,
                kb_version=kb_version,
                point_id=planned_point_id,
            )
            if _payload_hash(match.get("payload") or {}) == _payload_hash(proposed_payload):
                unchanged.append(identity)
            else:
                updated.append(identity)

        if identity in set(new[:3] + updated[:2]):
            preview_payloads.append(
                build_entity_point_payload(
                    proposed,
                    kb_version=kb_version,
                    point_id=planned_point_id,
                )
            )

    delete_candidates = sorted(set(existing_by_identity) - set(proposed_by_identity))
    return {
        "mutation_executed": False,
        "apply_blocked": bool(conflicts),
        "taxonomy_version": proposed_normalizer.taxonomy_version,
        "current_count": len(existing_records),
        "proposed_verified_count": len(proposed_cards),
        "existing_points_matched": len(reused),
        "existing_point_ids_reused": len(reused),
        "ambiguous_matches": len(conflicts),
        "unchanged": unchanged,
        "new": new,
        "updated": updated,
        "conflicts": conflicts,
        "draft_skipped": [],
        "delete_candidates": delete_candidates,
        "reused_point_ids": reused,
        "preview_payloads": preview_payloads,
    }


async def build_entity_index_update_plan_from_qdrant(
    *,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    kb_version: str = "acne_kb_v1",
    collection_name: str | None = None,
) -> dict[str, Any]:
    points = await _read_qdrant_entity_points(collection_name or get_entity_collection_name())
    return build_entity_index_update_plan(
        taxonomy_path=taxonomy_path,
        kb_version=kb_version,
        existing_points=points,
    )


def _existing_records_from_current_cards(*, kb_version: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for card in build_entity_cards_from_taxonomy():
        payload = build_entity_point_payload(card, kb_version=kb_version)
        records.append(
            {
                "identity": entity_identity_key(payload),
                "point_id": payload["point_id"],
                "payload": payload,
            }
        )
    return records


def _existing_records_from_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for point in points:
        payload = point.get("payload") or {}
        try:
            identity = entity_identity_key(payload)
        except ValueError:
            identity = ""
        records.append(
            {
                "identity": identity,
                "point_id": str(point.get("point_id") or point.get("id") or ""),
                "payload": payload,
            }
        )
    return records


def _detect_existing_conflicts(
    existing_by_identity: dict[str, list[dict[str, Any]]],
    point_id_to_identity: dict[str, set[str]],
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for identity, records in sorted(existing_by_identity.items()):
        point_ids = sorted({str(record.get("point_id") or "") for record in records if record.get("point_id")})
        entity_types = sorted({str((record.get("payload") or {}).get("entity_type") or "") for record in records})
        if len(records) > 1:
            conflicts.append(
                {
                    "identity": identity,
                    "reason": "multiple existing points match the same canonical identity",
                    "point_ids": point_ids,
                }
            )
        if len([item for item in entity_types if item]) > 1:
            conflicts.append(
                {
                    "identity": identity,
                    "reason": "entity type is inconsistent across matching existing points",
                    "entity_types": entity_types,
                }
            )
    for point_id, identities in sorted(point_id_to_identity.items()):
        if len(identities) > 1:
            conflicts.append(
                {
                    "identity": sorted(identities),
                    "reason": "point ID is occupied by multiple canonical identities",
                    "point_id": point_id,
                }
            )
    return conflicts


def _payload_hash(payload: dict[str, Any]) -> str:
    ignored = {"point_id"}
    comparable = {key: value for key, value in payload.items() if key not in ignored}
    raw = json.dumps(comparable, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _read_qdrant_entity_points(collection_name: str) -> list[dict[str, Any]]:
    from qdrant_client import AsyncQdrantClient  # type: ignore[import]

    client = AsyncQdrantClient(**qdrant_client_kwargs())
    records: list[dict[str, Any]] = []
    offset: Any = None
    try:
        while True:
            points, offset = await client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend({"point_id": str(point.id), "payload": point.payload or {}} for point in points)
            if offset is None:
                break
    finally:
        await client.close()
    return records


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.from_qdrant:
        plan = await build_entity_index_update_plan_from_qdrant(
            taxonomy_path=args.taxonomy_path,
            kb_version=args.kb_version,
            collection_name=args.collection,
        )
    else:
        plan = build_entity_index_update_plan(
            taxonomy_path=args.taxonomy_path,
            kb_version=args.kb_version,
        )
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0 if not plan.get("apply_blocked") else 1


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
