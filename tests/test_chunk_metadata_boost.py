import pytest

from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate
from src.retrieval.metadata_boost import (
    annotate_chunk_results,
    boost_chunk_results,
    score_candidate_with_metadata,
)
from src.retrieval.query_normalization import normalize_query


def test_metadata_boost_prioritizes_active_ingredient_and_drug_class():
    normalized = normalize_query("Differin thuộc nhóm gì?")
    matching = RetrievedCandidate(
        candidate_id="chunk:match",
        source="chunk",
        collection="acne_knowledge",
        text="Differin contains adapalene.",
        score=0.1,
        payload={
            "chunk_id": "chunk:match",
            "active_ingredient": ["adapalene"],
            "drug_class": ["topical_retinoid"],
        },
    )
    boosted = score_candidate_with_metadata(matching, normalized)

    assert boosted.fused_score is not None
    assert boosted.fused_score > matching.score
    assert boosted.matched_metadata["active_ingredient"] == ["adapalene"]
    assert boosted.matched_metadata["drug_class"] == ["topical_retinoid"]


def test_acne_type_boost_uses_concern_and_condition_metadata():
    normalized = normalize_query("Mụn đầu đen là gì?")
    chunks = [
        {
            "id": "drug",
            "text": "Drug-only chunk",
            "score": 0.2,
            "active_ingredient": ["clindamycin"],
        },
        {
            "id": "blackhead",
            "text": "Blackheads are comedonal acne.",
            "score": 0.1,
            "condition": ["acne_vulgaris"],
            "concern": ["blackheads"],
            "content_type": ["acne_type"],
        },
    ]
    boosted = boost_chunk_results(chunks, normalized)

    assert boosted[0].candidate_id == "blackhead"
    assert "concern" in boosted[0].matched_metadata
    assert boosted[0].payload["text"] == "Blackheads are comedonal acne."


def test_v5_annotation_keeps_ba_nhon_rrf_order_while_retaining_metadata_feature():
    normalized = NormalizedQuery(
        original_query="Bã nhờn ảnh hưởng đến mụn thế nào?",
        normalized_text="ba nhon anh huong den mun the nao",
        intent="acne_type",
        metadata={"concern": ["oiliness"]},
    )
    chunks = [
        {
            "id": "rrf-first",
            "text": "Higher RRF source evidence.",
            "score": 0.03,
            "rrf_score": 0.03,
            "dense_rank": 1,
            "sparse_rank": 1,
        },
        {
            "id": "ba-nhon-match",
            "text": "Sebum is a metadata-matched source.",
            "score": 0.02,
            "rrf_score": 0.02,
            "dense_rank": 2,
            "sparse_rank": 2,
            "concern": ["oiliness"],
        },
    ]

    legacy = boost_chunk_results(chunks, normalized)
    annotated = annotate_chunk_results(chunks, normalized)

    assert legacy[0].candidate_id == "ba-nhon-match"
    assert [candidate.candidate_id for candidate in annotated] == [
        "rrf-first",
        "ba-nhon-match",
    ]
    assert annotated[1].matched_metadata["concern"] == ["oiliness"]
    assert annotated[1].score == annotated[1].fused_score == 0.02
    assert annotated[1].debug["metadata_annotation_only"] is True
    assert "metadata_boost" not in annotated[1].debug


def test_v5_annotation_breaks_equal_rrf_scores_by_stable_candidate_id():
    normalized = NormalizedQuery(
        original_query="Mụn đầu đen là gì?",
        normalized_text="mun dau den la gi",
        intent="acne_type",
    )
    annotated = annotate_chunk_results(
        [
            {"id": "z-candidate", "text": "Z", "rrf_score": 0.02},
            {"id": "a-candidate", "text": "A", "rrf_score": 0.02},
        ],
        normalized,
    )

    assert [candidate.candidate_id for candidate in annotated] == [
        "a-candidate",
        "z-candidate",
    ]


@pytest.mark.parametrize(
    ("case_id", "query_fields", "payload_fields", "matched_field"),
    [
        (
            "sweat",
            {"metadata": {"concern": ["sweat"]}},
            {"concern": ["sweat"]},
            "concern",
        ),
        (
            "tazorac",
            {"drug_product": ["Tazorac"]},
            {"drug_product": ["Tazorac"]},
            "drug_product",
        ),
        (
            "pregnancy",
            {"safety_context": ["pregnancy"]},
            {"safety_context": ["pregnancy"]},
            "safety_context",
        ),
        (
            "blackheads",
            {"metadata": {"concern": ["blackheads"]}},
            {"concern": ["blackheads"]},
            "concern",
        ),
        (
            "acne-vulgaris",
            {"condition": ["acne_vulgaris"]},
            {"condition": ["acne_vulgaris"]},
            "condition",
        ),
    ],
)
def test_v5_annotation_retains_required_metadata_and_source_candidates(
    case_id: str,
    query_fields: dict,
    payload_fields: dict,
    matched_field: str,
) -> None:
    normalized = NormalizedQuery(
        original_query=case_id,
        normalized_text=case_id,
        intent="acne_type",
        **query_fields,
    )
    chunks = [
        {
            "id": f"{case_id}-rrf-first",
            "text": "Higher RRF source-backed evidence.",
            "rrf_score": 0.03,
            "dense_rank": 1,
            "sparse_rank": 1,
            "source_file": "higher-rrf.pdf",
        },
        {
            "id": f"{case_id}-metadata-match",
            "text": "Metadata-matched source-backed evidence.",
            "rrf_score": 0.02,
            "dense_rank": 2,
            "sparse_rank": 2,
            "source_file": "matched-source.pdf",
            **payload_fields,
        },
    ]

    annotated = annotate_chunk_results(chunks, normalized)

    assert [candidate.candidate_id for candidate in annotated] == [
        f"{case_id}-rrf-first",
        f"{case_id}-metadata-match",
    ]
    assert annotated[1].matched_metadata[matched_field]
    assert annotated[1].payload["source_file"] == "matched-source.pdf"
