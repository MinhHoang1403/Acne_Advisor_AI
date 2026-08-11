from __future__ import annotations

from src.database.retriever import annotate_metadata_matches


def test_query_metadata_annotation_preserves_rrf_order_without_additive_score() -> None:
    results = [
        {
            "id": "rrf-first",
            "rrf_score": 0.03,
            "score": 0.03,
            "dense_rank": 1,
            "sparse_rank": 1,
        },
        {
            "id": "metadata-match",
            "rrf_score": 0.02,
            "score": 0.02,
            "dense_rank": 2,
            "sparse_rank": 2,
            "concern": ["oiliness"],
        },
    ]

    annotated = annotate_metadata_matches(results, {"concern": ["oiliness"]})

    assert [result["id"] for result in annotated] == ["rrf-first", "metadata-match"]
    assert annotated[1]["matched_metadata_fields"] == ["concern"]
    assert annotated[1]["score"] == annotated[1]["rrf_score"]
    assert annotated[1]["metadata_annotation_only"] is True
    assert "metadata_boost" not in annotated[1]


def test_query_metadata_annotation_uses_upstream_ranks_then_candidate_id_for_ties() -> None:
    annotated = annotate_metadata_matches(
        [
            {"id": "z", "rrf_score": 0.02, "dense_rank": 2, "sparse_rank": 1},
            {"id": "a", "rrf_score": 0.02, "dense_rank": 1, "sparse_rank": 2},
            {"id": "b", "rrf_score": 0.02, "dense_rank": 1, "sparse_rank": 2},
        ],
        {},
    )

    assert [result["id"] for result in annotated] == ["a", "b", "z"]
