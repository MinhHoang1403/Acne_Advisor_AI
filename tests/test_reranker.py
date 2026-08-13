from src.retrieval.context_packer import pack_context
from src.retrieval.contracts import RetrievedCandidate
from src.retrieval.query_expansion import expand_normalized_query
from src.retrieval.query_normalization import normalize_query
from src.retrieval.reranker import rerank_candidates


def _entity(candidate_id: str, canonical_name: str, entity_type: str = "active_ingredient", **payload):
    data = {
        "entity_id": candidate_id,
        "entity_type": entity_type,
        "canonical_name": canonical_name,
        "aliases": payload.pop("aliases", []),
        "active_ingredients": payload.pop("active_ingredients", []),
        "drug_class": payload.pop("drug_class", []),
        "text": payload.pop("text", f"Entity card for {canonical_name}"),
        **payload,
    }
    return RetrievedCandidate(
        candidate_id=candidate_id,
        source="entity",
        collection="acne_entities",
        text=str(data["text"]),
        score=0.8,
        fused_score=0.8,
        payload=data,
        matched_metadata={},
    )


def _chunk(candidate_id: str, text: str, score: float = 0.4, **payload):
    data = {
        "chunk_id": candidate_id,
        "text": text,
        "source_file": "test.md",
        **payload,
    }
    return RetrievedCandidate(
        candidate_id=candidate_id,
        source="chunk",
        collection="acne_knowledge",
        text=text,
        score=score,
        fused_score=score,
        payload=data,
        matched_metadata={
            key: value
            for key, value in payload.items()
            if key in {"active_ingredient", "drug_class", "condition", "concern", "content_type", "safety_context", "query_intent_hint"}
        },
    )


def test_local_rules_reranker_is_deterministic_for_drug_identity():
    normalized = normalize_query("Dalacin T là gì?")
    expansion = expand_normalized_query(normalized)
    candidates = [
        _chunk("oral_abx", "Unrelated oral antibiotic chunk.", 0.9, drug_class=["oral_antibiotic"]),
        _entity(
            "dalacin",
            "Dalacin T",
            "drug_product",
            active_ingredients=["clindamycin"],
            drug_class=["topical_antibiotic"],
            aliases=["dalacin"],
        ),
        _chunk(
            "dalacin_chunk",
            "Dalacin T contains clindamycin topical_antibiotic.",
            0.2,
            active_ingredient=["clindamycin"],
            drug_class=["topical_antibiotic"],
        ),
    ]

    first, trace_a = rerank_candidates(normalized, candidates, expansion, top_n=3)
    second, trace_b = rerank_candidates(normalized, candidates, expansion, top_n=3)

    assert [candidate.candidate_id for candidate in first] == [candidate.candidate_id for candidate in second]
    assert first[0].candidate_id == "dalacin"
    assert first[0].debug["rerank_rank"] == 1
    assert trace_a.provider == "local_rules"
    assert trace_b.ranked_candidates[0].score_breakdown.final_score == trace_a.ranked_candidates[0].score_breakdown.final_score


def test_class_check_does_not_prioritize_negative_class():
    normalized = normalize_query("Clindamycin có phải retinoid không?")
    expansion = expand_normalized_query(normalized)
    candidates = [
        _entity("retinoid", "topical_retinoid", "drug_class", aliases=["retinoid"], text="retinoid"),
        _entity("clinda", "clindamycin", "active_ingredient", drug_class=["topical_antibiotic"]),
    ]

    reranked, _ = rerank_candidates(normalized, candidates, expansion, top_n=2)

    assert reranked[0].candidate_id == "clinda"
    assert reranked[0].payload["canonical_name"] == "clindamycin"


def test_acne_type_prioritizes_chunk_not_drug_entity():
    normalized = normalize_query("Mụn đầu đen là gì?")
    expansion = expand_normalized_query(normalized)
    candidates = [
        _entity("drug", "Dalacin T", "drug_product", active_ingredients=["clindamycin"]),
        _chunk(
            "blackhead",
            "Blackheads are comedonal acne.",
            0.2,
            condition=["acne_vulgaris"],
            concern=["blackheads"],
            content_type=["acne_type"],
        ),
    ]

    reranked, _ = rerank_candidates(normalized, candidates, expansion, top_n=2)

    assert reranked[0].candidate_id == "blackhead"
    assert reranked[0].source == "chunk"


def test_side_effect_and_safety_metadata_are_prioritized():
    side_effect_query = normalize_query("Adapalene có tác dụng phụ gì?")
    side_effect_ranked, _ = rerank_candidates(
        side_effect_query,
        [
            _chunk("general", "General acne care.", 0.7),
            _entity(
                "adapalene",
                "adapalene",
                "active_ingredient",
                side_effects=["dryness", "irritation"],
                drug_class=["topical_retinoid"],
            ),
        ],
        expand_normalized_query(side_effect_query),
    )
    assert side_effect_ranked[0].candidate_id == "adapalene"

    safety_query = normalize_query("Retinoid có dùng khi mang thai không?")
    safety_ranked, _ = rerank_candidates(
        safety_query,
        [
            _chunk("general", "General acne care.", 0.7),
            _chunk(
                "pregnancy",
                "Retinoid pregnancy safety warning.",
                0.2,
                safety_context=["pregnancy"],
                drug_class=["topical_retinoid"],
            ),
        ],
        expand_normalized_query(safety_query),
    )
    assert safety_ranked[0].candidate_id == "pregnancy"


def test_provider_fallbacks_do_not_crash(monkeypatch):
    monkeypatch.setenv("SEMANTIC_RERANK_MODEL_PATH", "")
    normalized = normalize_query("Adapalene có phải kháng sinh không?")
    candidates = [_entity("adapalene", "adapalene", "active_ingredient", drug_class=["topical_retinoid"])]

    unknown_ranked, unknown_trace = rerank_candidates(normalized, candidates, provider="mystery")
    model_ranked, model_trace = rerank_candidates(normalized, candidates, provider="local_model")

    assert unknown_ranked
    assert unknown_trace.provider == "local_rules"
    assert unknown_trace.warnings
    assert model_ranked
    assert model_trace.provider == "local_rules"
    assert model_trace.warnings


def test_empty_candidates_trace_shape():
    normalized = normalize_query("Mụn viêm nên xử lý thế nào?")
    reranked, trace = rerank_candidates(normalized, [], top_n=8)

    assert reranked == []
    assert trace.enabled is False
    assert trace.input_count == 0
    assert trace.output_count == 0


def test_pack_context_uses_reranked_candidate_order_for_acne_type():
    normalized = normalize_query("Mụn đầu đen là gì?")
    expansion = expand_normalized_query(normalized)
    candidates = [
        _entity("drug", "Dalacin T", "drug_product", active_ingredients=["clindamycin"]),
        _chunk("blackhead", "Blackheads are comedonal acne.", 0.1, concern=["blackheads"], content_type=["acne_type"]),
    ]
    reranked, _ = rerank_candidates(normalized, candidates, expansion, top_n=2)
    packed = pack_context(normalized, reranked, max_items=2)

    assert reranked[0].candidate_id == "blackhead"
    assert packed.items[0].item_id == "blackhead"
