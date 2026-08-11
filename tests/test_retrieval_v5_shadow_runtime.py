from __future__ import annotations

import pytest

from src.database import retriever as retriever_module
from src.database.retriever import HybridRetriever
from src.retrieval.contracts import RetrievedCandidate


class _VectorStore:
    _collection = "acne_knowledge"

    async def search(self, *, query_vector: list[float], top_k: int) -> list[dict[str, object]]:
        assert query_vector == [0.1, 0.2]
        assert top_k == 15
        return [
            {
                "id": "point-a",
                "chunk_id": "chunk-a",
                "text": "Adapalene is a topical retinoid.",
                "source_file": "source-a.pdf",
                "score": 0.82,
            }
        ]

    async def search_sparse(self, *, text: str, top_k: int) -> list[dict[str, object]]:
        assert text
        assert top_k == 15
        return [
            {
                "id": "point-a",
                "chunk_id": "chunk-a",
                "text": "Adapalene is a topical retinoid.",
                "source_file": "source-a.pdf",
                "score": 1.2,
            }
        ]


class _EntityRetriever:
    async def retrieve(self, **kwargs: object) -> list[object]:
        assert kwargs["limit"] == 8
        return []


class _EntitySignalRetriever:
    async def retrieve(self, **kwargs: object) -> list[RetrievedCandidate]:
        assert kwargs["limit"] == 8
        return [
            RetrievedCandidate(
                candidate_id="active_ingredient:adapalene",
                source="entity",
                collection="acne_entities_v1",
                text="adapalene structural entity",
                score=1.0,
                payload={
                    "entity_id": "active_ingredient:adapalene",
                    "canonical_name": "adapalene",
                    "entity_type": "active_ingredient",
                    "aliases": ["adapalene"],
                    "drug_class": ["topical_retinoid"],
                },
                matched_metadata={"canonical_name": ["adapalene"]},
            )
        ]


class _GraphStore:
    async def get_entity_context(self, entity_names: list[str]) -> list[dict[str, object]]:
        assert entity_names
        return []

    async def search_by_keywords(self, keywords: list[str]) -> list[dict[str, object]]:
        assert keywords
        return []


class _RecordingGraphStore(_GraphStore):
    def __init__(self) -> None:
        self.entity_lookups: list[list[str]] = []

    async def get_entity_context(self, entity_names: list[str]) -> list[dict[str, object]]:
        self.entity_lookups.append(entity_names)
        return []


@pytest.mark.asyncio
async def test_v5_shadow_is_passive_and_reports_v4_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _embed(query: str) -> list[float]:
        assert query == "Adapalene la gi?"
        return [0.1, 0.2]

    retriever = object.__new__(HybridRetriever)
    retriever._vector_store = _VectorStore()
    retriever._entity_retriever = _EntityRetriever()
    retriever._graph_store = _GraphStore()
    monkeypatch.setattr(retriever_module, "embed_query", _embed)
    monkeypatch.setattr(retriever_module, "rerank_enabled_from_env", lambda: False)
    monkeypatch.setenv("RETRIEVAL_PIPELINE_VERSION", "v5_shadow")

    result = await retriever.retrieve("Adapalene la gi?", top_k=5)

    shadow = result.metadata["retrieval_v5"]
    assert result.vector_contexts[0]["chunk_id"] == "chunk-a"
    assert shadow["requested_pipeline"] == "v5_shadow"
    assert shadow["execution_pipeline"] == "v4"
    assert shadow["shadow_equivalent"] is True
    assert all(stage["equivalent"] for stage in shadow["comparison"]["stages"])
    assert shadow["source_evidence_candidate_ids"] == ["point-a"]
    assert shadow["entity_signals"] == []
    assert shadow["graph_seed_names"] == []
    assert shadow["graph_signals"] == []

    monkeypatch.setenv("RETRIEVAL_PIPELINE_VERSION", "v4")
    v4_result = await retriever.retrieve("Adapalene la gi?", top_k=5)

    assert v4_result.vector_contexts == result.vector_contexts
    assert v4_result.graph_facts == result.graph_facts
    assert v4_result.sources == result.sources
    assert "retrieval_v5" not in v4_result.metadata


@pytest.mark.asyncio
async def test_explicit_v5_uses_source_evidence_pool_and_entity_graph_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _embed(query: str) -> list[float]:
        assert query == "Adapalene la gi?"
        return [0.1, 0.2]

    graph_store = _RecordingGraphStore()
    retriever = object.__new__(HybridRetriever)
    retriever._vector_store = _VectorStore()
    retriever._entity_retriever = _EntitySignalRetriever()
    retriever._graph_store = graph_store
    monkeypatch.setattr(retriever_module, "embed_query", _embed)
    monkeypatch.setattr(retriever_module, "rerank_enabled_from_env", lambda: False)
    monkeypatch.setenv("RETRIEVAL_PIPELINE_VERSION", "v5")

    result = await retriever.retrieve("Adapalene la gi?", top_k=5)

    shadow = result.metadata["retrieval_v5"]
    assert all(context["retrieval_source"] == "chunk" for context in result.vector_contexts)
    assert shadow["execution_pipeline"] == "v5"
    assert shadow["shadow_equivalent"] is None
    assert shadow["source_evidence_candidate_ids"] == ["point-a"]
    assert [signal["canonical_name"] for signal in shadow["entity_signals"]] == ["adapalene"]
    assert shadow["graph_seed_names"] == ["adapalene"]
    assert graph_store.entity_lookups == [["adapalene"]]
    trace_events = shadow["trace"]["events"]
    assert all(
        not candidate["legacy_compat_only"]
        for event in trace_events
        if event["stage"] in {"RERANK", "PACKER"}
        for candidate in event["candidates"]
    )
