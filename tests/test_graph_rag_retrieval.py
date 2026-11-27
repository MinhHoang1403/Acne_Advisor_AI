"""Kiểm thử toàn diện cho tầng đồ thị hai lớp (Lexical + Clinical GraphRAG) và Tri-Channel RRF."""

from __future__ import annotations

from typing import Any

import pytest

from src.knowledge.lexical_graph import (
    LEXICAL_GRAPH_LABELS,
    LEXICAL_GRAPH_RELATIONSHIPS,
    build_lexical_graph_records,
    extract_query_graph_anchors,
    sync_lexical_graph_to_neo4j,
)
from src.knowledge.graph_schema import (
    ENTITY_GRAPH_LABELS,
    ENTITY_GRAPH_RELATIONSHIPS,
)
from src.observability.trace_exporter import build_observability_event
from src.retrieval.contracts import RetrievedCandidate
from src.retrieval.graph_retriever import GraphSearchResult, Neo4jGraphStore
from src.retrieval.reranker import CandidateScorer
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.service import (
    EvidenceRetriever,
    _augment_rerank_query_with_brand_ingredients,
    _reorder_for_graph_anchor_coverage,
)


class _FakeVectorStore:
    def __init__(
        self,
        dense: list[dict[str, Any]] | None = None,
        sparse: list[dict[str, Any]] | None = None,
    ) -> None:
        self._dense = list(dense or [])
        self._sparse = list(sparse or [])

    async def search(self, query_vector: list[float], top_k: int = 5, filter: dict | None = None) -> list[dict[str, Any]]:
        return self._dense[:top_k]

    async def search_sparse(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self._sparse[:top_k]

    async def close(self) -> None:
        return None


class _FakeGraphStore(Neo4jGraphStore):
    def __init__(
        self,
        result: GraphSearchResult | BaseException,
        neighbors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(driver=object())
        self._result = result
        self._neighbors = list(neighbors or [])

    async def search_graph(
        self,
        query: str,
        *,
        top_k: int = 16,
        kb_version: str | None = None,
    ) -> GraphSearchResult:
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result

    async def fetch_adjacent_table_chunks(
        self,
        seed_chunk_ids: Any,
        *,
        exclude_ids: set[str] | None = None,
        kb_version: str | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        excluded = set(exclude_ids or set())
        return [item for item in self._neighbors if item["id"] not in excluded][:limit]


class _KeywordScorer(CandidateScorer):
    model_name = "keyword-test-reranker"
    device = "cpu"
    precision = "fp32"
    requested_precision = "fp32"
    device_fallback_reason = None
    precision_fallback_reason = None
    model_load_count = 1

    def __init__(self, scores_by_id: dict[str, float]) -> None:
        self._scores = scores_by_id

    async def score(self, query: str, candidates: Any) -> list[float]:
        out: list[float] = []
        for item in candidates:
            text = item.text if hasattr(item, "text") else str(item)
            matched_score = 0.25
            for marker, val in self._scores.items():
                if marker in text:
                    matched_score = val
                    break
            out.append(matched_score)
        return out


def test_lexical_graph_labels_and_relationships_never_collide_with_canonical_entity_graph() -> None:
    assert set(LEXICAL_GRAPH_LABELS).isdisjoint(set(ENTITY_GRAPH_LABELS))
    assert set(LEXICAL_GRAPH_RELATIONSHIPS).isdisjoint(set(ENTITY_GRAPH_RELATIONSHIPS))


def test_query_anchor_extraction_blocks_generic_acne_noise_and_captures_specific_facets() -> None:
    generic = extract_query_graph_anchors("Mụn trứng cá là gì và có phổ biến không?")
    assert not generic.has_specific_anchors
    assert generic.entity_names == []
    assert generic.facet_keys == []

    specific = extract_query_graph_anchors(
        "Cách chọn kem dưỡng ẩm oil-free cho người đang điều trị mụn bằng Epiduo?"
    )
    assert specific.has_specific_anchors
    assert "drug_product" in specific.entities_by_group
    assert "moisturizing_barrier" in specific.facet_keys
    assert "care_during_acne_therapy" in specific.facet_keys

    steroid = extract_query_graph_anchors(
        "Dấu hiệu nhận biết trứng cá do steroid và hội chứng Cushing là gì?"
    )
    assert "steroid_acne" in steroid.facet_keys

    ascii_query = extract_query_graph_anchors("phu nu mang thai co duoc dung isotretinoin khong")
    assert ascii_query.has_specific_anchors
    assert "pregnancy_teratogenicity" in ascii_query.facet_keys or "safety_context" in ascii_query.entities_by_group


def test_build_lexical_graph_records_creates_chunks_facets_entities_and_next_chunk_edges() -> None:
    sample_points = [
        {
            "id": "chunk-table-1",
            "payload": {
                "chunk_id": "chunk-table-1",
                "document_id": "doc-aad",
                "source_id": "aad_acne_guideline_2024",
                "source_title": "AAD 2024",
                "document_title": "AAD Guideline",
                "section_title": "Topical Therapies",
                "content_kind": "table",
                "chunk_index": 0,
                "table_index": 2,
                "active_ingredient": ["adapalene", "benzoyl_peroxide"],
                "care_topic": ["moisturizing"],
                "text": "Bảng liều dùng Adapalene phối hợp Benzoyl Peroxide và kem dưỡng ẩm non-comedogenic.",
            },
        },
        {
            "id": "chunk-table-2",
            "payload": {
                "chunk_id": "chunk-table-2",
                "document_id": "doc-aad",
                "source_id": "aad_acne_guideline_2024",
                "source_title": "AAD 2024",
                "document_title": "AAD Guideline",
                "section_title": "Topical Therapies",
                "content_kind": "table",
                "chunk_index": 1,
                "table_index": 2,
                "special_group": ["Pregnancy"],
                "text": "Phần tiếp theo của bảng: chống chỉ định Isotretinoin và Doxycycline cho phụ nữ mang thai.",
            },
        },
    ]

    records = build_lexical_graph_records(sample_points, build_id="test_build_001")
    assert len(records["sources"]) == 1
    assert len(records["documents"]) == 1
    assert len(records["chunks"]) == 2
    assert len(records["next_chunk_edges"]) == 1

    next_edge = records["next_chunk_edges"][0]
    assert next_edge["from_point_id"] == "chunk-table-1"
    assert next_edge["to_point_id"] == "chunk-table-2"
    assert next_edge["same_table"] is True
    assert next_edge["same_section"] is True

    entity_targets = {edge["canonical_name"].lower() for edge in records["mentions_entity_edges"]}
    assert {"adapalene", "benzoyl peroxide", "isotretinoin", "doxycycline"}.issubset(
        {name.replace("_", " ") for name in entity_targets}
    )

    facet_targets = {edge["facet_key"] for edge in records["has_facet_edges"]}
    assert "moisturizing_barrier" in facet_targets
    assert "pregnancy_teratogenicity" in facet_targets
    assert len(records["interaction_edges"]) >= 6


def test_tri_channel_rrf_preserves_two_channel_contract_and_boosts_graph_consensus() -> None:
    dense = [{"id": "c1", "score": 0.91, "text": "D1"}, {"id": "c2", "score": 0.88, "text": "D2"}]
    sparse = [{"id": "c3", "score": 11.2, "text": "S1"}, {"id": "c2", "score": 9.4, "text": "D2"}]

    two_way = reciprocal_rank_fusion(dense, sparse)
    assert [item["id"] for item in two_way] == ["c2", "c1", "c3"]

    graph = [
        {
            "id": "c3",
            "score": 3.4,
            "text": "S1",
            "graph_evidence": {"matched_entities": ["ActiveIngredient:Adapalene"]},
        },
        {"id": "c3", "score": 3.4, "text": "duplicate-must-be-ignored"},
        {"id": "c4", "score": 2.1, "text": "G-only"},
    ]
    three_way = reciprocal_rank_fusion(dense, sparse, graph_results=graph, graph_weight=1.0)
    by_id = {item["id"]: item for item in three_way}
    assert by_id["c3"]["graph_rank"] == 1
    assert by_id["c3"]["graph_evidence"] == {"matched_entities": ["ActiveIngredient:Adapalene"]}
    assert "c4" in by_id
    assert by_id["c3"]["rrf_score"] > by_id["c1"]["rrf_score"]


@pytest.mark.asyncio
async def test_evidence_retriever_fuses_graph_channel_and_expands_split_table_neighbor(monkeypatch) -> None:
    async def fake_embed_query(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr("src.retrieval.service.embed_query", fake_embed_query)

    vstore = _FakeVectorStore(
        dense=[
            {
                "id": "table-part-1",
                "chunk_id": "table-part-1",
                "source_id": "aad_acne_guideline_2024",
                "content_kind": "table",
                "score": 0.89,
                "text": "[TABLE_PART_1] Bảng hướng dẫn dưỡng ẩm non-comedogenic khi điều trị mụn.",
            }
        ],
        sparse=[],
    )
    gstore = _FakeGraphStore(
        result=GraphSearchResult(
            candidates=[
                {
                    "id": "graph-hit-1",
                    "chunk_id": "graph-hit-1",
                    "source_id": "nice_ng198_2026_08",
                    "content_kind": "text",
                    "score": 3.2,
                    "text": "[GRAPH_HIT] Phối hợp Epiduo (Adapalene + Benzoyl Peroxide) và kem dưỡng ẩm oil-free.",
                }
            ],
            anchors={"entities": ["Epiduo"], "facets": ["moisturizing_barrier"], "anchor_count": 2},
            paths=["Epiduo-[HAS_ACTIVE_INGREDIENT]->Adapalene"],
            status="ok",
            elapsed_ms=4.2,
        ),
        neighbors=[
            {
                "id": "table-part-2",
                "chunk_id": "table-part-2",
                "source_id": "aad_acne_guideline_2024",
                "content_kind": "table",
                "score": 0.0,
                "text": "[TABLE_PART_2] Phần tiếp nối bảng: thoa kem dưỡng ẩm sau thuốc bôi 15-20 phút.",
            },
            {
                "id": "table-noise",
                "chunk_id": "table-noise",
                "source_id": "aad_acne_guideline_2024",
                "content_kind": "table",
                "score": 0.0,
                "text": "[NOISE_ROW] Nội dung bảng khác không liên quan.",
            },
        ],
    )
    scorer = _KeywordScorer(
        {
            "[GRAPH_HIT]": 0.94,
            "[TABLE_PART_1]": 0.86,
            "[TABLE_PART_2]": 0.79,
            "[NOISE_ROW]": 0.05,
        }
    )

    retriever = EvidenceRetriever(
        vector_store=vstore,  # type: ignore[arg-type]
        graph_store=gstore,
        reranker=scorer,
    )
    result = await retriever.retrieve(
        "Cách dùng kem dưỡng ẩm oil-free khi bôi Epiduo?",
        top_k=5,
    )

    trace = result.metadata["retrieval_trace"]
    assert trace["architecture"] == "tri_channel_hybrid_graphrag_local_reranker"
    assert trace["channels"]["graph"]["count"] == 1
    assert trace["channels"]["graph"]["expanded_neighbors"] == ["table-part-2"]
    assert set(trace["selected_ids"]) == {"graph-hit-1", "table-part-1", "table-part-2"}
    assert "table-noise" not in trace["selected_ids"]

    obs_event = build_observability_event(
        query="Cách dùng kem dưỡng ẩm oil-free khi bôi Epiduo?",
        request_id="12345678-1234-5678-1234-567812345678",
        state={
            "retrieval_trace": trace,
            "packed_context": result.metadata["packed_context"],
            "retrieval_status": result.metadata["retrieval_status"],
            "retrieval_attempt": 1,
        },
    )
    assert obs_event.summary.retrieval_candidates_count == 2
    assert any(stage.stage == "graph" and stage.status == "success" for stage in obs_event.summary.stages)


@pytest.mark.asyncio
async def test_evidence_retriever_fails_open_when_neo4j_channel_errors(monkeypatch) -> None:
    async def fake_embed_query(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr("src.retrieval.service.embed_query", fake_embed_query)

    vstore = _FakeVectorStore(
        dense=[
            {
                "id": "dense-1",
                "chunk_id": "dense-1",
                "source_id": "vn_moh_dermatology_4416_2023_acne",
                "score": 0.92,
                "text": "Hướng dẫn Bộ Y tế về điều trị mụn trứng cá thông thường.",
            }
        ],
        sparse=[],
    )
    broken_graph = _FakeGraphStore(result=ConnectionError("Neo4j bolt socket unavailable"))
    retriever = EvidenceRetriever(
        vector_store=vstore,  # type: ignore[arg-type]
        graph_store=broken_graph,
        reranker_enabled=False,
    )

    result = await retriever.retrieve("Điều trị mụn bằng isotretinoin", top_k=3)
    assert result.metadata["retrieval_status"] == "ok"
    assert len(result.vector_contexts) == 1
    trace = result.metadata["retrieval_trace"]
    assert trace["channels"]["graph"]["status"] == "degraded"
    assert trace["channels"]["graph"]["error"] == "ConnectionError"


def test_offline_unit_test_with_fake_vector_store_disables_live_neo4j_by_default() -> None:
    retriever = EvidenceRetriever(
        vector_store=_FakeVectorStore(),  # type: ignore[arg-type]
        reranker_enabled=False,
    )
    assert retriever._graph_enabled is False


class _MockNeo4jResult:
    def __init__(self, rows: list[dict[str, Any]] | None = None, count_val: int = 0) -> None:
        self._rows = list(rows or [])
        self._count = count_val

    async def data(self) -> list[dict[str, Any]]:
        return self._rows

    async def single(self) -> dict[str, int]:
        return {"count": self._count}

    async def consume(self) -> None:
        return None


class _MockNeo4jSession:
    def __init__(self, query_rows: list[dict[str, Any]] | None = None, counts_by_token: dict[str, int] | None = None) -> None:
        self._query_rows = list(query_rows or [])
        self._counts = dict(counts_by_token or {})
        self.executed: list[str] = []

    async def __aenter__(self) -> "_MockNeo4jSession":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    async def run(self, query: str, **params: Any) -> _MockNeo4jResult:
        self.executed.append(query)
        if "RETURN count(" in query:
            for key, count_val in self._counts.items():
                if key in query:
                    return _MockNeo4jResult(count_val=count_val)
            return _MockNeo4jResult(count_val=0)
        return _MockNeo4jResult(rows=self._query_rows)


class _MockNeo4jDriver:
    def __init__(self, session_obj: _MockNeo4jSession) -> None:
        self._session = session_obj
        self.closed = False

    def session(self) -> _MockNeo4jSession:
        return self._session

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_neo4j_graph_store_executes_cypher_and_skips_generic_queries() -> None:
    session = _MockNeo4jSession(
        query_rows=[
            {
                "point_id": "p-1",
                "text": "Adapalene phối hợp Benzoyl Peroxide giảm kháng thuốc.",
                "payload_json": '{"chunk_id":"p-1","source_id":"aad_acne_guideline_2024"}',
                "score": 2.75,
                "matched_entities": ["ActiveIngredient:Adapalene"],
                "matched_facets": ["ClinicalFacet:antibiotic_stewardship"],
                "expanded_paths": ["Epiduo-[HAS_ACTIVE_INGREDIENT]->Adapalene"],
            }
        ]
    )
    driver = _MockNeo4jDriver(session)
    store = Neo4jGraphStore(driver=driver)

    no_anchor_res = await store.search_graph("Mụn là gì?")
    assert no_anchor_res.status == "no_anchors"
    assert no_anchor_res.candidates == []

    hit_res = await store.search_graph("Dùng Epiduo thế nào để giảm kháng thuốc?", top_k=8)
    assert hit_res.status == "ok"
    assert len(hit_res.candidates) == 1
    assert hit_res.candidates[0]["id"] == "p-1"
    assert hit_res.candidates[0]["source_id"] == "aad_acne_guideline_2024"
    assert "ActiveIngredient:Adapalene" in hit_res.paths

    neighbors = await store.fetch_adjacent_table_chunks(["p-0"], exclude_ids={"skip"}, limit=2)
    assert len(neighbors) == 1
    assert neighbors[0]["id"] == "p-1"

    Neo4jGraphStore._shared_driver = driver
    await Neo4jGraphStore.close_shared_driver()
    assert driver.closed is True


@pytest.mark.asyncio
async def test_sync_lexical_graph_to_neo4j_validates_expected_counts() -> None:
    records = build_lexical_graph_records(
        [
            {
                "id": "c-1",
                "payload": {
                    "chunk_id": "c-1",
                    "document_id": "d-1",
                    "source_id": "s-1",
                    "source_title": "AAD",
                    "document_title": "AAD Doc",
                    "chunk_index": 0,
                    "active_ingredient": ["isotretinoin"],
                    "special_group": ["Pregnancy"],
                    "text": "Isotretinoin chống chỉ định tuyệt đối cho phụ nữ mang thai.",
                },
            }
        ],
        build_id="b_sync_01",
    )
    counts = {
        ":KnowledgeSource": len(records["sources"]),
        ":KnowledgeDocument": len(records["documents"]),
        ":KnowledgeChunk": len(records["chunks"]),
        ":ClinicalFacet": len(records["facets"]),
        ":PUBLISHED_DOCUMENT": len(records["published_edges"]),
        ":HAS_CHUNK": len(records["has_chunk_edges"]),
        ":NEXT_CHUNK": len(records["next_chunk_edges"]),
        ":MENTIONS_ENTITY": len(records["mentions_entity_edges"]),
        ":HAS_CLINICAL_FACET": len(records["has_facet_edges"]),
        ":CLINICAL_INTERACTION": len(records["interaction_edges"]),
    }
    driver = _MockNeo4jDriver(_MockNeo4jSession(counts_by_token=counts))
    summary = await sync_lexical_graph_to_neo4j(driver, records, build_id="b_sync_01")
    assert summary["validation"]["passed"] is True
    assert summary["counts"]["chunks"] == 1


def test_brand_query_augmentation_and_submodular_anchor_coverage_promote_uncovered_facets() -> None:
    augmented = _augment_rerank_query_with_brand_ingredients(
        "Đang bôi Epiduo có nên dưỡng ẩm không?",
        graph_paths=[
            "Epiduo-[HAS_ACTIVE_INGREDIENT]->adapalene",
            "Epiduo-[HAS_ACTIVE_INGREDIENT]->benzoyl_peroxide",
        ],
    )
    assert "adapalene" in augmented
    assert "benzoyl peroxide" in augmented

    c1 = RetrievedCandidate(
        candidate_id="c1",
        collection="acne_knowledge",
        text="Dưỡng ẩm 1",
        rerank_score=0.90,
        rerank_rank=1,
        payload={"graph_evidence": {"matched_facets": ["ClinicalFacet:moisturizing_barrier"]}},
        debug={"graph_score": 1.3},
    )
    c2 = RetrievedCandidate(
        candidate_id="c2",
        collection="acne_knowledge",
        text="Dưỡng ẩm 2 trùng lặp",
        rerank_score=0.84,
        rerank_rank=2,
        payload={"graph_evidence": {"matched_facets": ["ClinicalFacet:moisturizing_barrier"]}},
        debug={"graph_score": 1.3},
    )
    c3 = RetrievedCandidate(
        candidate_id="c3",
        collection="acne_knowledge",
        text="Adapalene phối hợp Benzoyl Peroxide trong Epiduo",
        rerank_score=0.78,
        rerank_rank=3,
        payload={
            "graph_evidence": {
                "expanded_paths": [
                    "Epiduo-[HAS_ACTIVE_INGREDIENT]->adapalene",
                    "Epiduo-[HAS_ACTIVE_INGREDIENT]->benzoyl_peroxide",
                ]
            }
        },
        debug={"graph_score": 2.5},
    )
    reordered = _reorder_for_graph_anchor_coverage([c1, c2, c3], top_k=2)
    assert [item.candidate_id for item in reordered[:2]] == ["c1", "c3"]


def test_extended_vietnamese_brand_synonyms_bridge_to_canonical_inn_entities() -> None:
    klenzit = extract_query_graph_anchors("Dùng Klenzit-C kéo dài có gây kháng kháng sinh không?")
    assert klenzit.has_specific_anchors
    assert "adapalene" in [x.lower() for x in klenzit.entity_names]
    assert "clindamycin" in [x.lower() for x in klenzit.entity_names]
    assert any("Klenzit-C-[HAS_ACTIVE_INGREDIENT]->" in p for p in klenzit.brand_bridge_paths)

    roaccutane = extract_query_graph_anchors("Đang uống Roaccutane có dùng chung Doxycycline được không?")
    assert "isotretinoin" in [x.lower() for x in roaccutane.entity_names]
    assert "doxycycline" in [x.lower() for x in roaccutane.entity_names]

    azaduo = extract_query_graph_anchors("Bôi Azaduo bị khô rát da nên chọn kem dưỡng ẩm nào?")
    assert "adapalene" in [x.lower() for x in azaduo.entity_names]
    assert "benzoyl_peroxide" in [x.lower().replace(" ", "_") for x in azaduo.entity_names]


@pytest.mark.asyncio
async def test_stream_events_scope_emits_stage_and_delta_safely() -> None:
    from src.agent.stream_events import (
        begin_chat_stream_scope,
        emit_stream_delta,
        emit_stream_stage,
        is_chat_stream_active,
        reset_chat_stream_scope,
    )

    assert is_chat_stream_active() is False
    emit_stream_stage("guard", "noop")
    emit_stream_delta("noop")

    queue, token = begin_chat_stream_scope()
    try:
        assert is_chat_stream_active() is True
        emit_stream_stage("retrieve", "Đang tra cứu...")
        emit_stream_delta("Đoạn 1")
        ev1 = queue.get_nowait()
        ev2 = queue.get_nowait()
        assert ev1 == {"event": "stage", "data": {"stage": "retrieve", "label": "Đang tra cứu..."}}
        assert ev2 == {"event": "delta", "data": {"delta": "Đoạn 1"}}
    finally:
        reset_chat_stream_scope(token)
    assert is_chat_stream_active() is False



