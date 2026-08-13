from __future__ import annotations

import asyncio

import src.knowledge.graph_index as graph_index
from src.knowledge.graph_index import replace_entity_graph, upsert_entity_graph


def test_entity_graph_upsert_counts_only_materialized_records() -> None:
    records = {
        "nodes": [
            {"label": "DrugProduct", "canonical_name": "Product"},
            {"label": "ActiveIngredient", "canonical_name": "Ingredient"},
        ],
        "relationships": [{
            "source_label": "DrugProduct",
            "source_name": "Product",
            "relationship": "HAS_ACTIVE_INGREDIENT",
            "target_label": "ActiveIngredient",
            "target_name": "Ingredient",
            "properties": {},
        }],
    }

    class Result:
        def __init__(self, record):
            self.record = record

        async def single(self):
            return self.record

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def run(self, query, **_kwargs):
            if "MERGE (n:" in query:
                return Result({"materialized_count": 1})
            return Result({"source_count": 1, "target_count": 1, "materialized_count": 1})

    class Driver:
        def session(self):
            return Session()

    assert asyncio.run(upsert_entity_graph(Driver(), records)) == {
        "nodes": 2,
        "relationships": 1,
    }


def test_replace_entity_graph_removes_stale_relationships_before_nodes(monkeypatch) -> None:
    queries: list[str] = []

    class Result:
        async def single(self):
            return {"removed": 1}

        async def consume(self):
            return None

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def run(self, query, **_kwargs):
            queries.append(query)
            return Result()

    class Driver:
        def session(self):
            return Session()

    async def no_schema(_driver):
        return None

    async def materialize(_driver, _records):
        return {"nodes": 1, "relationships": 1}

    async def valid(_driver, _records):
        return {"passed": True, "errors": []}

    monkeypatch.setattr(graph_index, "apply_entity_graph_schema", no_schema)
    monkeypatch.setattr(graph_index, "upsert_entity_graph", materialize)
    monkeypatch.setattr(graph_index, "validate_entity_graph_records", valid)
    result = asyncio.run(
        replace_entity_graph(Driver(), {"nodes": [], "relationships": []}, build_id="build-a")
    )

    relationship_delete = next(i for i, query in enumerate(queries) if "DELETE r" in query)
    node_delete = next(i for i, query in enumerate(queries) if "DETACH DELETE n" in query)
    assert relationship_delete < node_delete
    assert result["stale_relationships_removed"] == 1
    assert result["stale_nodes_removed"] == 1
