from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Sequence

import pytest

from src.retrieval.contracts import RetrievedCandidate
from src.retrieval import reranker as reranker_module
from src.retrieval import service as retrieval_service
from src.retrieval.reranker import (
    CandidateReranker,
    DEFAULT_RERANKER_DEVICE,
    RerankerOperationalError,
    RerankerSettings,
    rerank_candidates,
    shutdown_reranker_executor,
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
    assert reranker.device == "cpu"
    assert reranker.model_load_count == 0


def test_reranker_defaults_to_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RERANKER_DEVICE", raising=False)

    settings = RerankerSettings.from_env()

    assert DEFAULT_RERANKER_DEVICE == "cuda"
    assert settings.device == "cuda"


def test_explicit_cpu_device_does_not_probe_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    def unexpected_probe() -> bool:
        raise AssertionError("CPU configuration must not probe CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_probe)

    reranker = CandidateReranker(RerankerSettings(device="cpu"))

    assert reranker.requested_device == "cpu"
    assert reranker.device == "cpu"
    assert reranker.device_fallback_reason is None


def test_cuda_configuration_falls_back_truthfully_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    reranker = CandidateReranker(RerankerSettings(device="cuda"))

    assert reranker.requested_device == "cuda"
    assert reranker.device == "cpu"
    assert reranker.device_fallback_reason == "cuda_unavailable"


@pytest.mark.asyncio
async def test_prepare_loads_model_once_and_reuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sentence_transformers

    constructions = 0

    class FakeCrossEncoder:
        def __init__(self, *_args, **_kwargs) -> None:
            nonlocal constructions
            constructions += 1

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    reranker = CandidateReranker(
        RerankerSettings(enabled=True, model_name="fake-local-model", device="cpu")
    )

    await reranker.prepare()
    await reranker.prepare()

    assert constructions == 1
    assert reranker.model_load_count == 1


@pytest.mark.asyncio
async def test_startup_warm_uses_process_reranker_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_calls = 0

    class WarmableScorer:
        async def prepare(self) -> None:
            nonlocal prepare_calls
            prepare_calls += 1

    scorer = WarmableScorer()
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setattr(
        retrieval_service,
        "_get_process_reranker",
        lambda _settings: scorer,
    )

    await retrieval_service.warm_process_reranker()

    assert prepare_calls == 1


def test_api_shutdown_registers_reranker_executor_cleanup() -> None:
    from src.api.app import app

    assert shutdown_reranker_executor in app.router.on_shutdown


@pytest.mark.asyncio
async def test_candidate_reranker_initializes_model_once_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sentence_transformers

    constructions = 0

    state_lock = threading.Lock()
    state = {"active": 0, "max_active": 0}

    class FakeCrossEncoder:
        def __init__(self, *_args, **kwargs) -> None:
            nonlocal constructions
            constructions += 1
            assert kwargs["local_files_only"] is True
            time.sleep(0.05)

        def predict(self, pairs, **_kwargs):
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            try:
                time.sleep(0.05)
                return [0.5] * len(pairs)
            finally:
                with state_lock:
                    state["active"] -= 1

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    reranker = CandidateReranker(
        RerankerSettings(enabled=True, model_name="fake-local-model", device="cpu")
    )

    await asyncio.gather(
        reranker.score("query one", [_candidate("a", 1)]),
        reranker.score("query two", [_candidate("b", 1)]),
    )

    assert constructions == 1
    assert reranker.model_load_count == 1
    assert isinstance(reranker._model_init_lock, type(threading.Lock()))
    assert state["max_active"] == 1


@pytest.mark.asyncio
async def test_timeout_does_not_allow_overlapping_background_inference() -> None:
    reranker = CandidateReranker(
        RerankerSettings(
            enabled=True,
            model_name="fake-local-model",
            device="cpu",
            timeout_seconds=0.05,
        )
    )
    release = threading.Event()
    state_lock = threading.Lock()
    state = {"active": 0, "entered": 0, "max_active": 0}

    def blocking_predict(_pairs):
        with state_lock:
            state["active"] += 1
            state["entered"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        release.wait(1)
        with state_lock:
            state["active"] -= 1
        return [0.5]

    reranker._predict = blocking_predict
    outcomes = await asyncio.gather(
        reranker.score("query one", [_candidate("a", 1)]),
        reranker.score("query two", [_candidate("b", 1)]),
        reranker.score("query three", [_candidate("c", 1)]),
        return_exceptions=True,
    )

    try:
        assert all(isinstance(item, RerankerOperationalError) for item in outcomes)
        assert all(item.reason == "timeout" for item in outcomes)
        assert state["entered"] == 1
        assert state["max_active"] == 1
    finally:
        release.set()
        await asyncio.sleep(0.05)

    assert state["entered"] == 1


@pytest.mark.asyncio
async def test_queued_inference_is_cancelled_when_its_request_times_out() -> None:
    running = CandidateReranker(
        RerankerSettings(device="cpu", timeout_seconds=0.05)
    )
    queued = CandidateReranker(
        RerankerSettings(device="cpu", timeout_seconds=0.05)
    )
    first_entered = threading.Event()
    release = threading.Event()
    queued_entries = 0

    def blocking_predict(_pairs):
        first_entered.set()
        release.wait(1)
        return [0.4]

    def queued_predict(_pairs):
        nonlocal queued_entries
        queued_entries += 1
        return [0.6]

    running._predict = blocking_predict
    queued._predict = queued_predict
    first_request = asyncio.create_task(
        running.score("running", [_candidate("running", 1)])
    )
    while not first_entered.is_set():
        await asyncio.sleep(0.005)

    try:
        with pytest.raises(RerankerOperationalError, match="bounded timeout") as exc_info:
            await queued.score("queued", [_candidate("queued", 1)])
        assert exc_info.value.reason == "timeout"
        assert queued_entries == 0
        with pytest.raises(RerankerOperationalError):
            await first_request
    finally:
        release.set()
        await asyncio.sleep(0.05)

    assert queued_entries == 0


@pytest.mark.asyncio
async def test_native_timeout_is_reported_as_rrf_fallback_not_success() -> None:
    reranker = CandidateReranker(
        RerankerSettings(device="cpu", timeout_seconds=0.05)
    )
    release = threading.Event()
    reranker._predict = lambda _pairs: (release.wait(1), [0.9])[1]
    candidates = [_candidate("a", 1)]

    try:
        outcome = await rerank_candidates(
            "query",
            candidates,
            scorer=reranker,
            enabled=True,
        )
        assert outcome.status == "fallback"
        assert outcome.fallback_used is True
        assert outcome.fallback_reason == "timeout"
        assert outcome.candidates == candidates
    finally:
        release.set()
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_executor_shutdown_is_non_blocking_and_prevents_restart_overlap() -> None:
    reranker = CandidateReranker(
        RerankerSettings(device="cpu", timeout_seconds=0.05)
    )
    release = threading.Event()
    finished = threading.Event()

    def blocking_predict(_pairs):
        try:
            release.wait(1)
            return [0.5]
        finally:
            finished.set()

    reranker._predict = blocking_predict
    try:
        with pytest.raises(RerankerOperationalError):
            await reranker.score("query", [_candidate("a", 1)])

        started = time.perf_counter()
        shutdown_reranker_executor()
        assert time.perf_counter() - started < 0.2

        replacement = CandidateReranker(
            RerankerSettings(device="cpu", timeout_seconds=0.05)
        )
        replacement._predict = lambda _pairs: [0.7]
        with pytest.raises(RerankerOperationalError) as exc_info:
            await replacement.score("query", [_candidate("b", 1)])
        assert exc_info.value.reason == "executor_shutdown"
    finally:
        release.set()
        for _ in range(100):
            if finished.is_set() and reranker_module._running_native_inference == 0:
                break
            await asyncio.sleep(0.01)

    restarted = CandidateReranker(
        RerankerSettings(device="cpu", timeout_seconds=0.2)
    )
    restarted._predict = lambda _pairs: [0.8]
    assert await restarted.score("query", [_candidate("c", 1)]) == [0.8]
    shutdown_reranker_executor()


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
