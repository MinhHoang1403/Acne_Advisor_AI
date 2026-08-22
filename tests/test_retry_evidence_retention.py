from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from src.agent.nodes import workflow
from src.retrieval.contracts import RetrievedCandidate
from src.retrieval.reranker import RerankerOperationalError
from src.retrieval.service import (
    EvidenceRetriever,
    merge_retrieval_candidates,
    retrieve_evidence,
)


def _evidence(candidate_id: str, text: str | None = None) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "chunk_id": candidate_id,
        "score": 1.0,
        "text": text or f"Evidence {candidate_id}",
        "source_id": f"source-{candidate_id}",
    }


def _candidate(candidate_id: str, rank: int, text: str = "Same evidence") -> RetrievedCandidate:
    return RetrievedCandidate(
        candidate_id=candidate_id,
        collection="acne_knowledge",
        text=text,
        fused_score=1.0 / rank,
        rank=rank,
        payload={
            "id": candidate_id,
            "chunk_id": candidate_id,
            "source_id": f"source-{candidate_id}",
            "text": text,
        },
    )


class MappingScorer:
    model_name = "fake-local-reranker"

    def __init__(
        self,
        scores: dict[str, float],
        error: RerankerOperationalError | None = None,
    ) -> None:
        self.scores = scores
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def score(
        self,
        query: str,
        candidates: Sequence[RetrievedCandidate],
    ) -> Sequence[float]:
        self.calls.append(
            {
                "query": query,
                "candidate_ids": [candidate.candidate_id for candidate in candidates],
            }
        )
        if self.error is not None:
            raise self.error
        return [self.scores[candidate.candidate_id] for candidate in candidates]


class StaticStore:
    def __init__(self, bm25: list[dict[str, Any]]) -> None:
        self.bm25 = bm25

    async def search_sparse(self, _query: str, top_k: int) -> list[dict[str, Any]]:
        return self.bm25[:top_k]

    async def close(self) -> None:
        return None


class StaticRetriever(EvidenceRetriever):
    def __init__(
        self,
        dense: list[dict[str, Any]],
        bm25: list[dict[str, Any]],
        scorer: MappingScorer,
    ) -> None:
        self.dense = dense
        super().__init__(StaticStore(bm25), reranker=scorer)

    async def _dense_search(self, _query: str, limit: int) -> list[dict[str, Any]]:
        return self.dense[:limit]


@pytest.mark.asyncio
async def test_first_attempt_keeps_stage1_rerank_and_pack_order() -> None:
    scorer = MappingScorer({"broad": 0.1, "direct": 0.9})
    retriever = StaticRetriever(
        [_evidence("broad"), _evidence("direct")],
        [],
        scorer,
    )

    result = await retriever.retrieve("overall question", top_k=2, retrieval_attempt=1)

    assert scorer.calls == [
        {"query": "overall question", "candidate_ids": ["broad", "direct"]}
    ]
    assert [item["id"] for item in result.vector_contexts] == ["direct", "broad"]
    assert result.metadata["retrieval_trace"]["retry_evidence"] == {
        "retrieval_attempt": 1,
        "retained_candidate_ids": [],
        "acquired_candidate_ids": ["broad", "direct"],
        "duplicate_candidate_ids": [],
        "eligible_candidate_ids": ["broad", "direct"],
        "reranked_candidate_ids": ["direct", "broad"],
        "packed_candidate_ids": ["direct", "broad"],
        "fallback_ordering": "within_attempt_rank_then_first_seen_attempt_then_candidate_id",
    }


@pytest.mark.asyncio
async def test_retry_unions_deduplicates_and_reranks_all_candidates() -> None:
    first_scorer = MappingScorer({"a": 0.9, "b": 0.8})
    first = await StaticRetriever(
        [_evidence("a"), _evidence("b")], [], first_scorer
    ).retrieve("overall question", top_k=2, retrieval_attempt=1)
    second_scorer = MappingScorer({"a": 0.8, "b": 0.7, "c": 0.95})

    second = await StaticRetriever(
        [_evidence("b"), _evidence("c")], [], second_scorer
    ).retrieve(
        "targeted missing evidence",
        top_k=3,
        retained_retrieval_candidates=first.metadata["retained_retrieval_candidates"],
        rerank_query="overall question",
        retrieval_attempt=2,
    )

    assert second_scorer.calls[0]["query"] == "overall question"
    assert set(second_scorer.calls[0]["candidate_ids"]) == {"a", "b", "c"}
    assert len(second_scorer.calls[0]["candidate_ids"]) == 3
    assert [item["id"] for item in second.vector_contexts] == ["c", "a", "b"]
    trace = second.metadata["retrieval_trace"]["retry_evidence"]
    assert trace["duplicate_candidate_ids"] == ["b"]
    duplicate = next(
        candidate
        for candidate in second.metadata["retained_retrieval_candidates"]
        if candidate["candidate_id"] == "b"
    )
    assert duplicate["debug"]["seen_in_attempts"] == [1, 2]
    assert duplicate["payload"]["source_id"] == "source-b"


@pytest.mark.asyncio
async def test_candidate_excluded_from_first_pack_remains_retry_eligible() -> None:
    first = await StaticRetriever(
        [_evidence("first-packed"), _evidence("first-unpacked")],
        [],
        MappingScorer({"first-packed": 0.9, "first-unpacked": 0.8}),
    ).retrieve("overall need", top_k=1)

    assert [item["id"] for item in first.vector_contexts] == ["first-packed"]
    assert {
        item["candidate_id"] for item in first.metadata["retained_retrieval_candidates"]
    } == {"first-packed", "first-unpacked"}

    second = await StaticRetriever(
        [_evidence("retry-new")],
        [],
        MappingScorer(
            {"first-packed": 0.1, "first-unpacked": 0.95, "retry-new": 0.8}
        ),
    ).retrieve(
        "targeted gap",
        top_k=2,
        retained_retrieval_candidates=first.metadata["retained_retrieval_candidates"],
        rerank_query="overall need",
        retrieval_attempt=2,
    )

    assert [item["id"] for item in second.vector_contexts] == [
        "first-unpacked",
        "retry-new",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ({"old-direct": 0.8, "old-noise": 0.1, "new-direct": 0.95}, ["new-direct", "old-direct", "old-noise"]),
        ({"old-direct": 0.95, "old-noise": 0.1, "new-direct": 0.7}, ["old-direct", "new-direct", "old-noise"]),
    ],
)
async def test_combined_reranker_does_not_privilege_an_attempt(
    scores: dict[str, float],
    expected: list[str],
) -> None:
    first = await StaticRetriever(
        [_evidence("old-direct"), _evidence("old-noise")],
        [],
        MappingScorer({"old-direct": 0.9, "old-noise": 0.1}),
    ).retrieve("overall need", top_k=2)

    second = await StaticRetriever(
        [_evidence("new-direct")], [], MappingScorer(scores)
    ).retrieve(
        "targeted gap",
        top_k=3,
        retained_retrieval_candidates=first.metadata["retained_retrieval_candidates"],
        rerank_query="overall need",
        retrieval_attempt=2,
    )

    assert [item["id"] for item in second.vector_contexts] == expected


def test_candidate_dedup_uses_stable_identity_not_text_similarity() -> None:
    retained = [_candidate("a", 1), _candidate("b", 2)]
    acquired = [_candidate("a", 1), _candidate("c", 2)]

    merged, duplicates = merge_retrieval_candidates(
        retained, acquired, retrieval_attempt=2
    )

    assert {candidate.candidate_id for candidate in merged} == {"a", "b", "c"}
    assert duplicates == ["a"]
    assert len(merged) == 3


@pytest.mark.asyncio
async def test_retry_without_retained_candidates_uses_new_candidates_only() -> None:
    scorer = MappingScorer({"new": 0.9})

    result = await StaticRetriever([_evidence("new")], [], scorer).retrieve(
        "targeted gap",
        retained_retrieval_candidates=[],
        rerank_query="overall need",
        retrieval_attempt=2,
    )

    assert scorer.calls[0]["candidate_ids"] == ["new"]
    assert [item["id"] for item in result.vector_contexts] == ["new"]


@pytest.mark.asyncio
async def test_retained_input_is_bounded_by_existing_acquisition_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_LIMIT", "2")
    retained = [
        _candidate(f"retained-{index}", index + 1).model_dump(mode="json")
        for index in range(10)
    ]
    scores = {f"retained-{index}": float(10 - index) for index in range(4)}
    scores["new"] = 20.0

    result = await StaticRetriever(
        [_evidence("new")], [], MappingScorer(scores)
    ).retrieve(
        "targeted gap",
        retained_retrieval_candidates=retained,
        rerank_query="overall need",
        retrieval_attempt=2,
    )

    trace = result.metadata["retrieval_trace"]["retry_evidence"]
    assert trace["retained_candidate_ids"] == [
        "retained-0",
        "retained-1",
        "retained-2",
        "retained-3",
    ]
    assert len(trace["eligible_candidate_ids"]) == 5


@pytest.mark.asyncio
async def test_reranker_failure_preserves_deterministic_combined_candidates() -> None:
    first = await StaticRetriever(
        [_evidence("a"), _evidence("b")],
        [],
        MappingScorer({"a": 0.9, "b": 0.8}),
    ).retrieve("overall need", top_k=2)
    failed_scorer = MappingScorer(
        {}, RerankerOperationalError("inference_failed", "synthetic failure")
    )

    second = await StaticRetriever(
        [_evidence("c"), _evidence("d")], [], failed_scorer
    ).retrieve(
        "targeted gap",
        top_k=4,
        retained_retrieval_candidates=first.metadata["retained_retrieval_candidates"],
        rerank_query="overall need",
        retrieval_attempt=2,
    )

    assert failed_scorer.calls[0]["candidate_ids"] == ["a", "c", "b", "d"]
    assert [item["id"] for item in second.vector_contexts] == ["a", "c", "b", "d"]
    trace = second.metadata["retrieval_trace"]
    assert trace["reranker"]["fallback_used"] is True
    assert trace["retry_evidence"]["eligible_candidate_ids"] == ["a", "c", "b", "d"]


@pytest.mark.asyncio
async def test_retry_union_still_uses_bounded_whole_chunk_packer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRIEVAL_CONTEXT_MAX_ITEMS", "8")
    monkeypatch.setenv("RETRIEVAL_CONTEXT_MAX_CHARS", "6000")
    old_ids = [f"old-{index}" for index in range(6)]
    new_ids = [f"new-{index}" for index in range(6)]
    text_by_id = {
        candidate_id: f"{candidate_id}:" + ("x" * 300)
        for candidate_id in [*old_ids, *new_ids]
    }
    first = await StaticRetriever(
        [_evidence(candidate_id, text_by_id[candidate_id]) for candidate_id in old_ids],
        [],
        MappingScorer({candidate_id: 1.0 for candidate_id in old_ids}),
    ).retrieve("overall need", top_k=8)
    combined_scores = {
        candidate_id: float(len([*old_ids, *new_ids]) - index)
        for index, candidate_id in enumerate([*old_ids, *new_ids])
    }

    second = await StaticRetriever(
        [_evidence(candidate_id, text_by_id[candidate_id]) for candidate_id in new_ids],
        [],
        MappingScorer(combined_scores),
    ).retrieve(
        "targeted gap",
        top_k=20,
        retained_retrieval_candidates=first.metadata["retained_retrieval_candidates"],
        rerank_query="overall need",
        retrieval_attempt=2,
    )

    packed = second.metadata["packed_context"]
    assert len(packed["items"]) == 8
    assert len(packed["context_text"]) <= 6000
    assert all(item["text"] == text_by_id[item["item_id"]] for item in packed["items"])


@pytest.mark.asyncio
async def test_workflow_retains_candidates_and_isolates_independent_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict[str, Any]] = []

    async def fake_ainvoke(payload: dict[str, Any]) -> dict[str, Any]:
        payloads.append(payload)
        candidate_id = f"candidate-{len(payloads)}"
        retained = [
            *payload["retained_retrieval_candidates"],
            {"candidate_id": candidate_id},
        ]
        return {
            "vector_contexts": [
                {"id": candidate_id, "text": "Evidence", "source_id": "source"}
            ],
            "sources": ["source"],
            "metadata": {
                "retrieval_status": "ok",
                "retained_retrieval_candidates": retained,
                "retrieval_trace": {
                    "query": payload["query"],
                    "rerank_query": payload["rerank_query"],
                    "retry_evidence": {"retrieval_attempt": payload["retrieval_attempt"]},
                },
                "packed_context": {"items": []},
            },
        }

    class FakeTool:
        ainvoke = staticmethod(fake_ainvoke)

    monkeypatch.setattr(workflow, "retrieve_evidence", FakeTool())
    first = await workflow.retrieve_node(
        {
            "normalized_question": "Overall question",
            "agent_decision": {
                "action": "retrieve",
                "retrieval_query": "self-contained overall query",
            },
            "retrieval_attempt": 0,
            "retry_history": [],
        }
    )
    second = await workflow.retrieve_node(
        {
            "normalized_question": "Overall question",
            "agent_decision": {
                "action": "retry",
                "retrieval_query": "targeted missing evidence",
                "reason_code": "evidence_gap",
            },
            **first,
        }
    )
    await workflow.retrieve_node(
        {
            "normalized_question": "Independent question",
            "agent_decision": {
                "action": "retrieve",
                "retrieval_query": "independent retrieval query",
            },
            "retrieval_attempt": 0,
            "retry_history": [],
        }
    )

    assert payloads[1]["query"] == "targeted missing evidence"
    assert payloads[1]["rerank_query"] == "self-contained overall query"
    assert payloads[1]["retained_retrieval_candidates"] == [
        {"candidate_id": "candidate-1"}
    ]
    assert second["retrieval_attempt_traces"][-1]["retry_evidence"] == {
        "retrieval_attempt": 2
    }
    assert payloads[2]["retained_retrieval_candidates"] == []


def test_retrieval_tool_schema_accepts_bounded_retry_inputs() -> None:
    parsed = retrieve_evidence.args_schema.model_validate(
        {
            "query": "targeted gap",
            "top_k": 8,
            "retained_retrieval_candidates": [{"candidate_id": "candidate-a"}],
            "rerank_query": "overall need",
            "retrieval_attempt": 2,
        }
    )

    assert parsed.retrieval_attempt == 2
    assert parsed.rerank_query == "overall need"
