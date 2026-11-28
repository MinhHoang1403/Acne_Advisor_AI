"""Đồng bộ đồ thị hai tầng (Canonical Entity + Lexical & Clinical Graph) lên Neo4j.

Script đọc trực tiếp các điểm tri thức đã kích hoạt từ Qdrant (`acne_knowledge`)
và taxonomy chuẩn (`data/taxonomy/drug_aliases.yaml`), đồng bộ 32 nút thực thể gốc
cùng toàn bộ nút `KnowledgeSource`, `KnowledgeDocument`, `KnowledgeChunk`,
`ClinicalFacet` vào `acne_neo4j` mà không tốn bất kỳ lệnh gọi Embedding API nào.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.ingestion.manifest import load_build_manifest, save_build_manifest  # noqa: E402
from src.knowledge.entity_cards import build_entity_cards_from_taxonomy  # noqa: E402
from src.knowledge.graph_index import (  # noqa: E402
    get_neo4j_driver,
    replace_entity_graph,
    validate_entity_graph_records,
)
from src.knowledge.graph_schema import build_entity_graph_records  # noqa: E402
from src.knowledge.lexical_graph import (  # noqa: E402
    build_lexical_graph_records,
    sync_lexical_graph_to_neo4j,
)
from src.knowledge.normalizer import DrugEntityNormalizer  # noqa: E402


async def _scroll_all_qdrant_points(
    client: AsyncQdrantClient,
    collection_name: str,
) -> list[dict[str, Any]]:
    """Đọc toàn bộ points kèm payload từ Qdrant collection hiện hành."""

    points: list[dict[str, Any]] = []
    offset: Any = None
    while True:
        records, next_offset = await client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            points.append(
                {
                    "id": str(record.id),
                    "payload": dict(record.payload or {}),
                }
            )
        if next_offset is None:
            break
        offset = next_offset
    return points


async def sync_live_knowledge_graph(
    *,
    collection_name: str | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Đồng bộ đầy đủ Canonical Entity Graph và Lexical + Clinical Graph vào Neo4j."""

    target_collection = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    manifest_file = manifest_path or (PROJECT_ROOT / "data" / "knowledge_build_manifest.json")
    if not manifest_file.is_file():
        fallback_manifest = PROJECT_ROOT / "data" / "build" / "manifest.json"
        if fallback_manifest.is_file():
            manifest_file = fallback_manifest
    manifest_build_id = ""
    if manifest_file.is_file():
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        manifest_build_id = str(manifest_data.get("build_id") or "").strip()

    from src.database.vector_store import qdrant_client_kwargs

    client_kwargs = qdrant_client_kwargs()
    if qdrant_url:
        client_kwargs["url"] = qdrant_url
    qdrant = AsyncQdrantClient(**client_kwargs, timeout=30)
    try:
        points = await _scroll_all_qdrant_points(qdrant, target_collection)
    finally:
        await qdrant.close()

    if not points:
        raise RuntimeError(f"Qdrant collection '{target_collection}' returned 0 points")

    detected_build_ids = {
        str(item["payload"].get("build_id") or "").strip()
        for item in points
        if item["payload"].get("build_id")
    }
    # Fail-closed: không dựng đồ thị gắn nhãn build của manifest từ các điểm Qdrant thuộc build
    # khác, nếu không tầng GraphRAG sẽ lệch phiên bản với tầng Dense/BM25 mà không ai phát hiện.
    if manifest_build_id and detected_build_ids and detected_build_ids != {manifest_build_id}:
        raise RuntimeError(
            "Qdrant points do not belong to the manifest build: "
            f"manifest={manifest_build_id}, qdrant={sorted(detected_build_ids)}"
        )
    build_id = manifest_build_id or (sorted(detected_build_ids)[0] if detected_build_ids else "")
    if not build_id:
        raise RuntimeError("Could not determine active kb build_id from manifest or Qdrant points")

    normalizer = DrugEntityNormalizer()
    cards = build_entity_cards_from_taxonomy()
    entity_records = build_entity_graph_records(cards, kb_version=build_id)
    lexical_records = build_lexical_graph_records(
        points,
        build_id=build_id,
        taxonomy=normalizer,
    )

    driver = get_neo4j_driver()
    try:
        await replace_entity_graph(
            driver,
            entity_records,
            build_id=build_id,
        )
        lexical_sync = await sync_lexical_graph_to_neo4j(
            driver,
            lexical_records,
            build_id=build_id,
        )
        entity_post_check = await validate_entity_graph_records(driver, entity_records)
    finally:
        await driver.close()

    if not entity_post_check["passed"]:
        raise RuntimeError(
            f"Canonical entity graph check failed after lexical sync: {entity_post_check['errors']}"
        )

    manifest_recorded = False
    if manifest_file.is_file() and manifest_build_id == build_id:
        # Ghi lại kết quả đồng bộ vào manifest đang kích hoạt để đồ thị từ vựng được quản lý
        # phiên bản giống hệt luồng activate_knowledge (activation.lexical_graph).
        manifest = load_build_manifest(manifest_file)
        activation = dict(manifest.get("activation") or {})
        activation["lexical_graph"] = lexical_sync
        manifest["activation"] = activation
        save_build_manifest(manifest_file, manifest)
        manifest_recorded = True

    return {
        "passed": True,
        "build_id": build_id,
        "qdrant_collection": target_collection,
        "qdrant_points_read": len(points),
        "canonical_entity_graph": {
            "nodes": entity_post_check["expected_nodes"],
            "relationships": entity_post_check["expected_relationships"],
            "passed": entity_post_check["passed"],
        },
        "lexical_clinical_graph": lexical_sync["counts"],
        "lexical_catalog_hash": lexical_sync["catalog_hash"],
        "lexical_validation_passed": lexical_sync["validation"]["passed"],
        "manifest_recorded": manifest_recorded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync live Qdrant knowledge chunks and clinical taxonomy into Neo4j GraphRAG."
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Logical Qdrant collection name (defaults to QDRANT_COLLECTION_NAME / acne_knowledge).",
    )
    args = parser.parse_args()
    summary = asyncio.run(sync_live_knowledge_graph(collection_name=args.collection))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
