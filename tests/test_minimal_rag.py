import ast
import inspect
from dataclasses import fields
from pathlib import Path

import pytest

from src.database.retriever import HybridRetriever
from src.evaluation.minimal_rag import (
    MINIMAL_RAG_SYSTEM_ID,
    MinimalRagResult,
    MinimalRagService,
    minimal_normalize_query,
)
from src.retrieval.rrf import reciprocal_rank_fusion


class FakeVectorStore:
    def __init__(self, dense=None, sparse=None, *, fail=False):
        self.dense = dense or []
        self.sparse = sparse or []
        self.fail = fail
        self.dense_calls = 0
        self.sparse_calls = 0

    async def search(self, query_vector, top_k=5):
        self.dense_calls += 1
        if self.fail:
            raise RuntimeError("secret provider detail")
        return self.dense[:top_k]

    async def search_sparse(self, text, top_k=5):
        self.sparse_calls += 1
        return self.sparse[:top_k]


def _candidate(candidate_id: str, text: str, *, score: float, source=True):
    return {
        "id": candidate_id,
        "chunk_id": candidate_id,
        "text": text,
        "score": score,
        "source_path": f"sample_data/{candidate_id}.md" if source else "",
        "document_id": f"doc-{candidate_id}" if source else "",
        "source_identity": f"source-{candidate_id}" if source else "",
    }


@pytest.mark.asyncio
async def test_minimal_rag_calls_dense_sparse_fuses_and_preserves_provenance():
    store = FakeVectorStore(
        dense=[_candidate("a", "Dense evidence", score=0.9), _candidate("b", "Both", score=0.7)],
        sparse=[_candidate("b", "Both", score=0.95), _candidate("c", "Sparse evidence", score=0.6)],
    )
    embed_calls = []

    async def embedder(query):
        embed_calls.append(query)
        return [0.1, 0.2]

    service = MinimalRagService(vector_store=store, embedder=embedder)
    result = await service.run("  Benzoyl   peroxide là gì?  ", generate_answer=False)

    assert result.system_id == MINIMAL_RAG_SYSTEM_ID
    assert result.normalized_query == "Benzoyl peroxide là gì?"
    assert embed_calls == [result.normalized_query]
    assert store.dense_calls == 1
    assert store.sparse_calls == 1
    assert result.diagnostics["retrieval_attempts"] == 1
    assert result.evidence[0].evidence_id == "b"
    assert result.evidence[0].source_path == "sample_data/b.md"
    assert result.citations[0] == "source-b"
    assert "[SOURCE source-b | CHUNK b]" in result.context
    assert result.call_counts["neo4j"] == 0
    assert result.call_counts["reranker"] == 0
    assert result.call_counts["generation"] == 0


@pytest.mark.asyncio
async def test_minimal_rag_excludes_missing_provenance_and_bounds_context():
    store = FakeVectorStore(
        dense=[
            _candidate("invalid", "not source backed", score=1.0, source=False),
            _candidate("valid", "x" * 1000, score=0.9),
        ],
        sparse=[],
    )
    service = MinimalRagService(
        vector_store=store,
        embedder=lambda _query: _async_value([0.1]),
        context_max_chars=256,
    )

    result = await service.run("question", generate_answer=False)

    assert [item.evidence_id for item in result.evidence] == ["valid"]
    assert result.diagnostics["excluded_missing_provenance"] == 1
    assert len(result.context) <= 256


@pytest.mark.asyncio
async def test_minimal_rag_provider_failure_is_safe_and_not_retried():
    store = FakeVectorStore(fail=True)
    service = MinimalRagService(vector_store=store, embedder=lambda _query: _async_value([0.1]))

    result = await service.run("question")

    assert result.status == "provider_or_retrieval_error"
    assert result.error == "RuntimeError"
    assert "secret provider detail" not in result.error
    assert store.dense_calls == 1
    assert result.diagnostics["retrieval_attempts"] == 1


@pytest.mark.asyncio
async def test_minimal_rag_generates_once_without_provider_fallback():
    store = FakeVectorStore(dense=[_candidate("a", "Evidence", score=1.0)])
    generator_calls = []

    async def generator(**kwargs):
        generator_calls.append(kwargs)
        return {"text": "Câu trả lời [source-a]", "provider": "gemini", "model": "test-model"}

    service = MinimalRagService(
        vector_store=store,
        embedder=lambda _query: _async_value([0.1]),
        generator=generator,
    )
    result = await service.run("question")

    assert result.status == "completed"
    assert result.answer == "Câu trả lời [source-a]"
    assert len(generator_calls) == 1
    assert generator_calls[0]["allow_fallback"] is False
    assert result.call_counts["generation"] == 1


@pytest.mark.asyncio
async def test_minimal_rag_global_emergency_guard_skips_external_calls():
    store = FakeVectorStore()
    embed_calls = []

    async def embedder(query):
        embed_calls.append(query)
        return [0.1]

    result = await MinimalRagService(vector_store=store, embedder=embedder).run(
        "Tôi bị khó thở và sưng môi sau khi dùng thuốc"
    )

    assert result.status == "hard_safety_response"
    assert not embed_calls
    assert store.dense_calls == 0
    assert store.sparse_calls == 0
    assert all(count == 0 for count in result.call_counts.values())


def test_minimal_rag_contract_is_small_and_stable():
    assert len(fields(MinimalRagResult)) == 12
    assert [item.name for item in fields(MinimalRagResult)] == [
        "system_id",
        "query",
        "normalized_query",
        "evidence",
        "context",
        "answer",
        "citations",
        "latency_ms",
        "call_counts",
        "status",
        "error",
        "diagnostics",
    ]
    assert minimal_normalize_query("  A\u0300   B  ") == "À B"


def test_minimal_rag_import_boundary_excludes_complex_layers():
    module_path = Path(inspect.getfile(MinimalRagService))
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    forbidden_modules = {
        "src.retrieval.candidate_policy_v5",
        "src.retrieval.entity_signal_v5",
        "src.retrieval.graph_signal_v5",
        "src.agent.nodes.evidence_sufficiency",
        "src.agent.nodes.claim_grounding",
        "langgraph",
    }
    forbidden_names = {
        "CandidatePolicyV5",
        "EntitySignalV5",
        "GraphSignalV5",
        "EvidenceSelector",
        "Neo4jClient",
    }
    assert imports.isdisjoint(forbidden_modules)
    assert imported_names.isdisjoint(forbidden_names)


def test_shared_rrf_preserves_current_system_math():
    dense = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}]
    sparse = [{"id": "b", "score": 0.7}, {"id": "c", "score": 0.6}]

    shared = reciprocal_rank_fusion(dense, sparse, dense_weight=0.7, sparse_weight=1.2, k=60)
    production = HybridRetriever._rrf_fusion(
        dense,
        sparse,
        dense_weight=0.7,
        sparse_weight=1.2,
        k=60,
    )

    assert production == shared
    assert [item["id"] for item in shared] == ["b", "c", "a"]


async def _async_value(value):
    return value
