from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Sequence

import pytest

from src.retrieval.contracts import RetrievedCandidate
from src.retrieval import service as retrieval_service
from src.retrieval.reranker import (
    CandidateReranker,
    RerankerOperationalError,
    RerankerSettings,
    rerank_candidates,
)
from src.retrieval.service import EvidenceRetriever


def _candidate(candidate_id: str, rank: int, text: str | None = None) -> RetrievedCandidate:
    return RetrievedCandidate(
        candidate_id=candidate_id,
        collection="acne_knowledge",
        text=text or f"Evidence {candidate_id}",
        fused_score=1.0 / rank,
        rank=rank,
        payload={"chunk_id": candidate_id, "source_id": f"source-{candidate_id}"},
    )


class FakeScorer:
    model_name = "fake-multilingual-reranker"

    def __init__(
        self,
        scores: Sequence[float] | None = None,
        error: RerankerOperationalError | None = None,
    ) -> None:
        self.scores = scores
        self.error = error
        self.seen_ids: list[str] = []

    async def score(
        self,
        _query: str,
        candidates: Sequence[RetrievedCandidate],
    ) -> Sequence[float]:
        self.seen_ids = [candidate.candidate_id for candidate in candidates]
        if self.error is not None:
            raise self.error
        return list(self.scores or [])


@pytest.mark.asyncio
async def test_direct_support_candidate_outranks_broad_candidate() -> None:
    candidates = [
        _candidate("broad", 1, "General acne information."),
        _candidate("direct", 2, "Direct evidence for the retrieval query."),
    ]

    outcome = await rerank_candidates(
        "specific question",
        candidates,
        scorer=FakeScorer([0.1, 0.9]),
        enabled=True,
    )

    assert [item.candidate_id for item in outcome.candidates] == ["direct", "broad"]
    assert [item.rerank_rank for item in outcome.candidates] == [1, 2]
    assert outcome.status == "succeeded"


@pytest.mark.asyncio
async def test_operational_exception_preserves_exact_rrf_order() -> None:
    candidates = [_candidate("a", 1), _candidate("b", 2), _candidate("c", 3)]
    scorer = FakeScorer(error=RerankerOperationalError("timeout", "timed out"))

    outcome = await rerank_candidates(
        "query", candidates, scorer=scorer, enabled=True
    )

    assert outcome.candidates == candidates
    assert outcome.status == "fallback"
    assert outcome.fallback_reason == "timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scores", "reason"),
    [
        ([0.5], "score_count_mismatch"),
        ([float("nan"), 0.5], "non_finite_score"),
        ([float("inf"), 0.5], "non_finite_score"),
    ],
)
async def test_invalid_scores_preserve_exact_rrf_order(
    scores: list[float], reason: str
) -> None:
    candidates = [_candidate("a", 1), _candidate("b", 2)]

    outcome = await rerank_candidates(
        "query", candidates, scorer=FakeScorer(scores), enabled=True
    )

    assert outcome.candidates == candidates
    assert outcome.fallback_reason == reason


@pytest.mark.asyncio
async def test_equal_scores_preserve_original_rrf_order() -> None:
    candidates = [_candidate("b", 1), _candidate("a", 2)]

    outcome = await rerank_candidates(
        "query", candidates, scorer=FakeScorer([0.5, 0.5]), enabled=True
    )

    assert [item.candidate_id for item in outcome.candidates] == ["b", "a"]


@pytest.mark.asyncio
async def test_disabled_reranker_does_not_call_scorer() -> None:
    candidates = [_candidate("a", 1)]
    scorer = FakeScorer(error=RerankerOperationalError("unexpected", "must not run"))

    outcome = await rerank_candidates(
        "query", candidates, scorer=scorer, enabled=False
    )

    assert outcome.candidates == candidates
    assert outcome.status == "disabled"
    assert scorer.seen_ids == []


@pytest.mark.asyncio
async def test_programming_error_is_not_silently_downgraded() -> None:
    class BuggyScorer:
        model_name = "buggy"

        async def score(self, _query, _candidates):
            raise TypeError("contract bug")

    with pytest.raises(TypeError, match="contract bug"):
        await rerank_candidates(
            "query", [_candidate("a", 1)], scorer=BuggyScorer(), enabled=True
        )


def test_candidate_reranker_is_lazy() -> None:
    reranker = CandidateReranker(
        RerankerSettings(enabled=True, model_name="local-model", device="cpu")
    )

    assert reranker._model is None


@pytest.mark.asyncio
async def test_candidate_reranker_initializes_model_once_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sentence_transformers

    constructions = 0

    class FakeCrossEncoder:
        def __init__(self, *_args, **kwargs) -> None:
            nonlocal constructions
            constructions += 1
            assert kwargs["local_files_only"] is True
            time.sleep(0.05)

        def predict(self, pairs, **_kwargs):
            return [0.5] * len(pairs)

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    reranker = CandidateReranker(
        RerankerSettings(enabled=True, model_name="fake-local-model", device="cpu")
    )

    await asyncio.gather(
        reranker.score("query one", [_candidate("a", 1)]),
        reranker.score("query two", [_candidate("b", 1)]),
    )

    assert constructions == 1
    assert isinstance(reranker._model_init_lock, type(threading.Lock()))


def _evidence(candidate_id: str, text: str) -> dict[str, object]:
    return {
        "id": candidate_id,
        "score": 1.0,
        "text": text,
        "chunk_id": candidate_id,
        "source_id": f"source-{candidate_id}",
    }


class FakeStore:
    def __init__(self, bm25: list[dict[str, object]]) -> None:
        self.bm25 = bm25

    async def search_sparse(self, _query: str, top_k: int) -> list[dict[str, object]]:
        return self.bm25[:top_k]

    async def close(self) -> None:
        return None


class FakeChannelRetriever(EvidenceRetriever):
    def __init__(
        self,
        dense: list[dict[str, object]],
        bm25: list[dict[str, object]],
        scorer: FakeScorer,
    ) -> None:
        self.dense = dense
        super().__init__(FakeStore(bm25), reranker=scorer)

    async def _dense_search(self, _query: str, limit: int) -> list[dict[str, object]]:
        return self.dense[:limit]


@pytest.mark.asyncio
async def test_process_reranker_is_reused_across_tool_calls_and_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer_constructions: list[object] = []
    stores_created = 0

    class SharedFakeScorer:
        model_name = "fake-process-reranker"

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def score(self, _query, candidates):
            ids = [candidate.candidate_id for candidate in candidates]
            self.calls.append(ids)
            return [float(len(ids) - index) for index in range(len(ids))]

    shared_scorer = SharedFakeScorer()

    def fake_reranker(settings):
        scorer_constructions.append(settings)
        return shared_scorer

    class RuntimeFakeStore:
        def __init__(self) -> None:
            nonlocal stores_created
            stores_created += 1
            self.candidate_id = f"runtime-{stores_created}"

        async def search(self, _vector, top_k):
            return [_evidence(self.candidate_id, "Evidence")][:top_k]

        async def search_sparse(self, _query, top_k):
            return []

        async def close(self) -> None:
            return None

    async def fake_embed(_query: str) -> list[float]:
        return [0.0]

    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_MODEL", "fake-local-model")
    monkeypatch.setattr(retrieval_service, "_process_reranker", None)
    monkeypatch.setattr(retrieval_service, "_process_reranker_settings", None)
    monkeypatch.setattr(retrieval_service, "CandidateReranker", fake_reranker)
    monkeypatch.setattr(retrieval_service, "QdrantVectorStore", RuntimeFakeStore)
    monkeypatch.setattr(retrieval_service, "embed_query", fake_embed)

    first = await retrieval_service.retrieve_evidence.ainvoke(
        {"query": "overall", "retrieval_attempt": 1}
    )
    second = await retrieval_service.retrieve_evidence.ainvoke(
        {
            "query": "targeted",
            "rerank_query": "overall",
            "retrieval_attempt": 2,
            "retained_retrieval_candidates": first["metadata"][
                "retained_retrieval_candidates"
            ],
        }
    )
    independent = await retrieval_service.retrieve_evidence.ainvoke(
        {"query": "independent", "retrieval_attempt": 1}
    )

    assert len(scorer_constructions) == 1
    assert len(shared_scorer.calls) == 3
    assert {item["id"] for item in second["vector_contexts"]} == {"runtime-1", "runtime-2"}
    assert [item["id"] for item in independent["vector_contexts"]] == ["runtime-3"]


@pytest.mark.asyncio
async def test_service_reranks_rrf_union_before_packing() -> None:
    dense = [_evidence("dense-only", "Dense evidence")]
    bm25 = [_evidence("bm25-only", "BM25 direct evidence")]
    scorer = FakeScorer([0.1, 0.9])
    retriever = FakeChannelRetriever(dense, bm25, scorer)

    result = await retriever.retrieve("direct query", top_k=2)
    trace = result.metadata["retrieval_trace"]

    assert set(scorer.seen_ids) == {"dense-only", "bm25-only"}
    assert [item["id"] for item in result.vector_contexts] == ["bm25-only", "dense-only"]
    assert trace["reranker"]["status"] == "succeeded"
    assert trace["packer"]["selected_ids"] == ["bm25-only", "dense-only"]


@pytest.mark.asyncio
async def test_service_falls_back_to_exact_rrf_order_before_packing() -> None:
    dense = [_evidence("dense-only", "Dense evidence")]
    bm25 = [_evidence("bm25-only", "BM25 evidence")]
    scorer = FakeScorer(error=RerankerOperationalError("inference_failed", "failed"))
    retriever = FakeChannelRetriever(dense, bm25, scorer)

    result = await retriever.retrieve("query", top_k=2)
    trace = result.metadata["retrieval_trace"]

    assert [item["id"] for item in result.vector_contexts] == ["dense-only", "bm25-only"]
    assert trace["reranker"]["fallback_used"] is True
    assert trace["reranker"]["fallback_reason"] == "inference_failed"
    assert trace["candidate_trace"]["fused"][0]["rrf_rank"] == 1
