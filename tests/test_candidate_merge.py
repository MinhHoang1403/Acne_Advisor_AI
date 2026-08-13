from src.retrieval.candidate_merge import merge_candidates
from src.retrieval.contracts import RetrievedCandidate
from src.retrieval.query_normalization import normalize_query


def test_merge_prioritizes_entity_for_drug_identity():
    normalized = normalize_query("Dalacin T là gì?")
    entity = RetrievedCandidate(
        candidate_id="entity:dalacin",
        source="entity",
        collection="acne_entities",
        text="Dalacin T entity card",
        score=0.7,
        payload={"entity_id": "drug_product:dalacin", "canonical_name": "Dalacin T"},
    )
    chunk = RetrievedCandidate(
        candidate_id="chunk:dalacin",
        source="chunk",
        collection="acne_knowledge",
        text="Dalacin T chunk",
        score=0.75,
        fused_score=0.75,
        payload={"chunk_id": "chunk:dalacin"},
    )

    merged = merge_candidates([entity], [chunk], normalized)

    assert merged[0].source == "entity"
    assert merged[0].rank == 1


def test_merge_dedupes_by_collection_and_payload_ids():
    normalized = normalize_query("Adapalene có phải kháng sinh không?")
    first = RetrievedCandidate(
        candidate_id="first",
        source="chunk",
        collection="acne_knowledge",
        text="A",
        score=0.1,
        fused_score=0.1,
        payload={"chunk_id": "same"},
    )
    second = RetrievedCandidate(
        candidate_id="second",
        source="chunk",
        collection="acne_knowledge",
        text="B",
        score=0.3,
        fused_score=0.3,
        payload={"chunk_id": "same"},
    )

    merged = merge_candidates([], [first, second], normalized)

    assert len(merged) == 1
    assert merged[0].candidate_id == "second"


def test_merge_preserves_chunk_evidence_for_drug_intent_when_entities_fill_limit():
    normalized = normalize_query("Tazorac, Differin và Epiduo khác nhau về hoạt chất như thế nào?")
    entities = [
        RetrievedCandidate(
            candidate_id=f"entity:{index}",
            source="entity",
            collection="acne_entities",
            text=f"Entity {index}",
            score=1.0 - index * 0.01,
            payload={"entity_id": f"entity:{index}", "canonical_name": f"Entity {index}"},
        )
        for index in range(8)
    ]
    chunk = RetrievedCandidate(
        candidate_id="chunk:tazorac",
        source="chunk",
        collection="acne_knowledge",
        text="Tazorac tazarotene Differin adapalene Epiduo benzoyl peroxide",
        score=0.01,
        payload={"chunk_id": "chunk:tazorac"},
        matched_metadata={"drug_product": ["Tazorac"]},
    )

    merged = merge_candidates(entities, [chunk], normalized, limit=5)

    assert len(merged) == 5
    assert any(candidate.source == "chunk" for candidate in merged)
