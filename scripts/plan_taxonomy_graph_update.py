#!/usr/bin/env python3
"""Dry-run planner for deterministic taxonomy Neo4j graph updates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.entity_cards import build_entity_cards_from_taxonomy  # noqa: E402
from src.knowledge.graph_schema import build_entity_graph_records  # noqa: E402
from src.knowledge.normalizer import DEFAULT_TAXONOMY_PATH, DrugEntityNormalizer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan taxonomy graph changes without Neo4j writes.")
    parser.add_argument("--taxonomy-path", default=str(DEFAULT_TAXONOMY_PATH))
    parser.add_argument("--kb-version", default="acne_kb_v1")
    return parser.parse_args(argv)


def build_taxonomy_graph_update_plan(
    *,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    kb_version: str = "acne_kb_v1",
) -> dict[str, Any]:
    current_records = build_entity_graph_records(
        build_entity_cards_from_taxonomy(),
        kb_version=kb_version,
    )
    proposed_cards = build_entity_cards_from_taxonomy(DrugEntityNormalizer(taxonomy_path))
    proposed_records = build_entity_graph_records(proposed_cards, kb_version=kb_version)

    current_nodes = {_node_key(node): node for node in current_records["nodes"]}
    proposed_nodes = {_node_key(node): node for node in proposed_records["nodes"]}
    current_rels = {_rel_key(rel): rel for rel in current_records["relationships"]}
    proposed_rels = {_rel_key(rel): rel for rel in proposed_records["relationships"]}

    nodes_to_create = sorted(set(proposed_nodes) - set(current_nodes))
    relationships_to_create = sorted(set(proposed_rels) - set(current_rels))
    nodes_to_update = sorted(
        key
        for key in set(proposed_nodes).intersection(current_nodes)
        if _stable_json(proposed_nodes[key]) != _stable_json(current_nodes[key])
    )
    relationships_to_update = sorted(
        key
        for key in set(proposed_rels).intersection(current_rels)
        if _stable_json(proposed_rels[key]) != _stable_json(current_rels[key])
    )

    return {
        "mutation_executed": False,
        "current": {
            "node_count": len(current_records["nodes"]),
            "relationship_count": len(current_records["relationships"]),
        },
        "proposed": {
            "node_count": len(proposed_records["nodes"]),
            "relationship_count": len(proposed_records["relationships"]),
        },
        "nodes_to_create": nodes_to_create,
        "nodes_to_update": nodes_to_update,
        "relationships_to_create": relationships_to_create,
        "relationships_to_update": relationships_to_update,
        "conflicts": [],
        "orphans": [],
        "delete_candidates": {
            "nodes": sorted(set(current_nodes) - set(proposed_nodes)),
            "relationships": sorted(set(current_rels) - set(proposed_rels)),
        },
    }


def _node_key(node: dict[str, Any]) -> str:
    return f"{node.get('label')}:{node.get('canonical_name')}"


def _rel_key(rel: dict[str, Any]) -> str:
    return "|".join(
        str(rel.get(field) or "")
        for field in ("source_label", "source_name", "relationship", "target_label", "target_name")
    )


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_taxonomy_graph_update_plan(
        taxonomy_path=args.taxonomy_path,
        kb_version=args.kb_version,
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
