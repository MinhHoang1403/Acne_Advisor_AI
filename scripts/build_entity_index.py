#!/usr/bin/env python3
"""Build and optionally upsert acne entity cards into Qdrant."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from src.knowledge.entity_cards import build_entity_cards_from_taxonomy  # noqa: E402
from src.knowledge.entity_index import (  # noqa: E402
    ENTITY_COLLECTION_DEFAULT,
    build_entity_point_payload,
    ensure_entity_collection,
    get_chunk_collection_name,
    upsert_entity_cards,
)
from src.integrations.google_genai import embed_texts_sync  # noqa: E402
from src.knowledge.schemas import EntityCard  # noqa: E402
from src.knowledge.versioning import get_embedding_metadata  # noqa: E402


PREVIEW_NAMES = {"Dalacin T", "Epiduo", "Differin", "Tazorac", "benzoyl_peroxide", "tazarotene"}

__all__ = [
    "build_and_index_entities",
    "build_dry_run_summary",
    "embed_entity_texts_sync",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the acne entity-card Qdrant index.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Load taxonomy and print payload previews without Qdrant writes. Default.",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Actually create/validate collection, embed cards, and upsert points.",
    )
    parser.add_argument(
        "--recreate",
        choices=["true", "false"],
        default="false",
        help="Delete/recreate the target entity collection. Default: false.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("ENTITY_QDRANT_COLLECTION_NAME", ENTITY_COLLECTION_DEFAULT),
        help="Entity Qdrant collection name. Default: acne_entities_v1.",
    )
    parser.add_argument(
        "--kb-version",
        default=os.getenv("KB_VERSION", "acne_kb_v1"),
        help="KB version embedded into entity IDs and payloads.",
    )
    return parser.parse_args()


def build_dry_run_summary(
    cards: list[EntityCard],
    *,
    collection: str = ENTITY_COLLECTION_DEFAULT,
    kb_version: str = "acne_kb_v1",
) -> dict:
    counts = Counter(card.entity_type for card in cards)
    embedding_metadata = get_embedding_metadata()
    previews = []
    for card in cards:
        if card.canonical_name in PREVIEW_NAMES:
            previews.append(build_entity_point_payload(card, kb_version=kb_version))
    return {
        "collection": collection,
        "kb_version": kb_version,
        **embedding_metadata,
        "card_count": len(cards),
        "counts_by_entity_type": dict(sorted(counts.items())),
        "preview_payloads": previews,
    }


def embed_entity_texts_sync(texts: list[str]) -> list[list[float]]:
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to .env before running with --no-dry-run."
        )

    embedding_metadata = get_embedding_metadata()
    return embed_texts_sync(
        texts,
        model_name=embedding_metadata["embedding_model"],
        task_type="retrieval_document",
        expected_dimensions=int(embedding_metadata["embedding_dimensions"]),
        api_key=google_api_key,
    )


async def main() -> int:
    args = parse_args()
    recreate = args.recreate.lower() == "true"
    cards = build_entity_cards_from_taxonomy()

    if args.dry_run:
        summary = build_dry_run_summary(
            cards,
            collection=args.collection,
            kb_version=args.kb_version,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    protected_targets = {
        "acne_knowledge",
        "acne_chunks_v1",
        os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge"),
        get_chunk_collection_name(),
    }
    if args.collection in protected_targets:
        raise RuntimeError(
            "Refusing to write entity cards to a chunk collection target: "
            f"{args.collection!r}."
        )

    print(f"Entity collection target: {args.collection}")
    print(f"Chunk collection remains: {get_chunk_collection_name()}")
    embedding_metadata = get_embedding_metadata()
    print(f"Embedding provider: {embedding_metadata['embedding_provider']}")
    print(f"Embedding model: {embedding_metadata['embedding_model']}")
    print(f"Embedding dimensions: {embedding_metadata['embedding_dimensions']}")

    if recreate:
        print(
            "WARNING: --recreate true only targets the entity collection "
            f"{args.collection!r}. Chunk collection remains {get_chunk_collection_name()!r}."
        )

    result = await build_and_index_entities(
        cards=cards,
        collection_name=args.collection,
        kb_version=args.kb_version,
        recreate=recreate,
    )
    print(f"Upserted {result['upserted']} entity cards into Qdrant collection {args.collection!r}.")
    return 0


async def build_and_index_entities(
    *,
    cards: list[EntityCard] | None = None,
    collection_name: str | None = None,
    kb_version: str | None = None,
    recreate: bool = False,
    embeddings: list[list[float]] | None = None,
) -> dict[str, object]:
    """Build, embed, and upsert entity cards for the authoritative Phase 1 workflow."""

    resolved_cards = cards or build_entity_cards_from_taxonomy()
    resolved_collection = collection_name or os.getenv(
        "ENTITY_QDRANT_COLLECTION_NAME", ENTITY_COLLECTION_DEFAULT
    )
    resolved_kb_version = kb_version or os.getenv("KB_VERSION", "acne_kb_v1")
    protected_targets = {
        "acne_knowledge",
        "acne_chunks_v1",
        os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge"),
        get_chunk_collection_name(),
    }
    if resolved_collection in protected_targets:
        raise RuntimeError(
            "Refusing to write entity cards to a chunk collection target: "
            f"{resolved_collection!r}."
        )

    await ensure_entity_collection(
        collection_name=resolved_collection,
        recreate=recreate,
    )
    resolved_embeddings = embeddings or embed_entity_texts_sync(
        [
            build_entity_point_payload(card, kb_version=resolved_kb_version)["text"]
            for card in resolved_cards
        ]
    )
    count = await upsert_entity_cards(
        resolved_cards,
        embeddings=resolved_embeddings,
        collection_name=resolved_collection,
        kb_version=resolved_kb_version,
    )
    return {
        "cards": resolved_cards,
        "collection": resolved_collection,
        "kb_version": resolved_kb_version,
        "upserted": count,
        "expected_entity_ids": {
            build_entity_point_payload(card, kb_version=resolved_kb_version)["entity_id"]
            for card in resolved_cards
        },
    }


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
