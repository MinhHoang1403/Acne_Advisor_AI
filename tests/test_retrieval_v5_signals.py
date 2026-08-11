from __future__ import annotations

from src.knowledge.entity_cards import build_entity_cards_from_taxonomy
from src.knowledge.entity_index import build_entity_point_payload
from src.retrieval.contracts import RetrievedCandidate
from src.retrieval.entity_retriever import retrieve_entity_candidates_from_payloads
from src.retrieval.query_expansion import expand_normalized_query
from src.retrieval.query_normalization import normalize_query
from src.retrieval.v5_compat import RetrievalPipelineSelection, build_v5_shadow_trace
from src.retrieval.v5_contracts import RetrievalPipelineVersion, RetrievalStageV5
from src.retrieval.v5_signals import (
    build_entity_signals,
    build_graph_signals,
    entity_graph_seed_names,
)


def _entity_candidates(query: str) -> list[RetrievedCandidate]:
    normalized = normalize_query(query)
    payloads = [
        build_entity_point_payload(card, kb_version="acne_kb_v1")
        for card in build_entity_cards_from_taxonomy()
    ]
    return retrieve_entity_candidates_from_payloads(
        normalized,
        expand_normalized_query(normalized),
        payloads,
    )


def _chunk(candidate_id: str) -> RetrievedCandidate:
    return RetrievedCandidate(
        candidate_id=candidate_id,
        source="chunk",
        collection="acne_knowledge",
        text="Source-backed evidence.",
        score=0.02,
        fused_score=0.02,
        payload={"chunk_id": candidate_id, "source_file": "source.pdf", "rrf_score": 0.02},
        rank=1,
    )


def test_tazorac_and_pregnancy_entity_signals_preserve_product_chain_and_safety() -> None:
    signals = build_entity_signals(_entity_candidates("Tazorac co dung khi mang thai khong?"))
    by_name = {signal.canonical_name: signal for signal in signals}

    assert {"Tazorac", "tazarotene", "topical_retinoid", "pregnancy"} <= set(by_name)
    assert "pregnancy" in by_name["tazarotene"].safety_annotations
    assert "Tazorac" in entity_graph_seed_names(signals)
    assert all(signal.match_confidence >= 0 for signal in signals)


def test_isotretinoin_and_topical_antibiotic_signal_chains_are_retained() -> None:
    isotretinoin_signals = build_entity_signals(
        _entity_candidates("Isotretinoin co dung khi mang thai khong?")
    )
    isotretinoin_by_name = {
        signal.canonical_name: signal for signal in isotretinoin_signals
    }
    antibiotic_signals = build_entity_signals(
        _entity_candidates("So sanh Dalacin T voi adapalene")
    )
    antibiotic_names = {signal.canonical_name for signal in antibiotic_signals}

    assert {"isotretinoin", "oral_retinoid", "pregnancy"} <= set(isotretinoin_by_name)
    assert "pregnancy" in isotretinoin_by_name["isotretinoin"].safety_annotations
    assert {"Dalacin T", "clindamycin", "topical_antibiotic"} <= antibiotic_names


def test_graph_signals_are_structural_only_without_linked_chunk_provenance() -> None:
    entity_signals = build_entity_signals(_entity_candidates("Tazorac la gi?"))
    signals = build_graph_signals(
        [
            {
                "subject": "Tazorac",
                "subject_graph_node_id": "drug_product:tazorac",
                "predicate": "HAS_ACTIVE_INGREDIENT",
                "object": "tazarotene",
                "object_graph_node_id": "active_ingredient:tazarotene",
                "confidence": 1.0,
                "evidence_source": "taxonomy",
            }
        ],
        entity_signals,
    )

    assert len(signals) == 1
    assert signals[0].relation_path == ("HAS_ACTIVE_INGREDIENT", "tazarotene")
    assert signals[0].source_chunk_ids == ()
    assert signals[0].medical_claim_eligible is False


def test_v5_source_evidence_pool_excludes_entity_cards_without_dropping_chunks() -> None:
    normalized = normalize_query("Tazorac la gi?")
    expansion = expand_normalized_query(normalized)
    entity = _entity_candidates(normalized.original_query)[0]
    chunks = [_chunk("chunk-a"), _chunk("chunk-b"), _chunk("chunk-c")]
    selection = RetrievalPipelineSelection(
        requested=RetrievalPipelineVersion.V5_SHADOW,
        execution=RetrievalPipelineVersion.V4,
        shadow_enabled=True,
    )
    trace = build_v5_shadow_trace(
        original_query=normalized.original_query,
        retrieval_query=normalized.original_query,
        normalized_query=normalized,
        expansion=expansion,
        dense_results=[{"id": "chunk-a", "score": 0.8}],
        sparse_results=[{"id": "chunk-a", "score": 1.0}],
        fused_results=[{"id": "chunk-a", "rrf_score": 0.02, "score": 0.02}],
        entity_candidates=[entity],
        chunk_candidates=chunks,
        merged_candidates=[entity, *chunks],
        reranked_candidates=[entity, *chunks],
        rerank_trace=_rerank_trace(),
        packed_context=_packed_context(chunks[0]),
        graph_facts=[],
        timings_ms={},
        selection=selection,
    )

    assert trace.candidate_ids_for(RetrievalStageV5.SOURCE_EVIDENCE_POOL) == (
        "chunk-a",
        "chunk-b",
        "chunk-c",
    )
    assert trace.candidate_ids_for(RetrievalStageV5.LEGACY_CANDIDATE_MERGE) == (
        entity.candidate_id,
        "chunk-a",
        "chunk-b",
        "chunk-c",
    )
    assert trace.entity_signals
    source_pool = next(
        event
        for event in trace.events
        if event.stage == RetrievalStageV5.SOURCE_EVIDENCE_POOL
    )
    assert all(candidate.source == "chunk" for candidate in source_pool.candidates)


def _rerank_trace():
    from src.retrieval.contracts import RerankTrace

    return RerankTrace(
        provider="fixture",
        enabled=False,
        input_count=2,
        output_count=2,
        top_n=2,
    )


def _packed_context(chunk: RetrievedCandidate):
    from src.retrieval.context_packer import pack_context

    return pack_context(normalize_query("Tazorac la gi?"), [chunk], max_items=1)
