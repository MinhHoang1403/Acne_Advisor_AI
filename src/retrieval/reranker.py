"""Sắp xếp lại RRF candidate bằng cross-encoder chạy cục bộ.

Reranker chỉ nhận retrieval query và text của candidate. Raw model score là tín
hiệu thứ tự, không phải xác suất hay confidence. Model được tải lazy và chỉ từ
local cache/path; lỗi vận hành có kiểu để caller có thể giữ nguyên thứ tự RRF.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from src.retrieval.contracts import RetrievedCandidate

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_DEVICE = "cuda"
SUPPORTED_RERANKER_DEVICES = frozenset({"cpu", "cuda"})

logger = logging.getLogger(__name__)

_inference_executor: ThreadPoolExecutor | None = None
_inference_executor_lock = threading.Lock()
_running_native_inference = 0


def _get_inference_executor() -> ThreadPoolExecutor:
    """Return the one process executor used by local reranker inference."""

    global _inference_executor
    with _inference_executor_lock:
        if _inference_executor is None:
            if _running_native_inference:
                raise RerankerOperationalError(
                    "executor_shutdown",
                    "Local reranker is finishing inference during shutdown.",
                )
            _inference_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="bge-reranker",
            )
        return _inference_executor


def _run_native_inference(
    predict: Any,
    pairs: list[tuple[str, str]],
) -> Sequence[float]:
    global _running_native_inference
    with _inference_executor_lock:
        _running_native_inference += 1
    try:
        return predict(pairs)
    finally:
        with _inference_executor_lock:
            _running_native_inference -= 1


def shutdown_reranker_executor() -> None:
    """Stop accepting inference without waiting on uninterruptible native code.

    Queued work is cancelled. A native call that already started is allowed to
    finish because Python cannot safely pre-empt it; until then a replacement
    executor cannot be created, preventing overlap across a runtime restart.
    """

    global _inference_executor
    with _inference_executor_lock:
        executor = _inference_executor
        _inference_executor = None
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


class RerankerOperationalError(RuntimeError):
    """Lỗi model/runtime dự kiến có thể fallback an toàn về thứ tự RRF."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class CandidateScorer(Protocol):
    """Giao diện nhỏ cho local scorer và deterministic test doubles."""

    model_name: str

    async def score(
        self,
        query: str,
        candidates: Sequence[RetrievedCandidate],
    ) -> Sequence[float]: ...


@dataclass(frozen=True)
class RerankerSettings:
    enabled: bool = False
    model_name: str = DEFAULT_RERANKER_MODEL
    device: str = DEFAULT_RERANKER_DEVICE
    batch_size: int = 4
    timeout_seconds: float = 20.0

    @classmethod
    def from_env(cls) -> RerankerSettings:
        return cls(
            enabled=_env_bool("RERANKER_ENABLED", False),
            model_name=os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL).strip()
            or DEFAULT_RERANKER_MODEL,
            device=_configured_device(),
            batch_size=_bounded_int_env("RERANKER_BATCH_SIZE", 4, 1, 64),
            timeout_seconds=_bounded_float_env(
                "RERANKER_TIMEOUT_SECONDS", 20.0, 0.1, 120.0
            ),
        )


@dataclass(frozen=True)
class RerankOutcome:
    candidates: list[RetrievedCandidate]
    status: str
    enabled: bool
    model_name: str
    fallback_reason: str | None
    elapsed_ms: float

    @property
    def fallback_used(self) -> bool:
        return self.fallback_reason is not None


class CandidateReranker:
    """Lazy local ``sentence-transformers`` CrossEncoder adapter."""

    def __init__(self, settings: RerankerSettings | None = None) -> None:
        self.settings = settings or RerankerSettings.from_env()
        self.model_name = self.settings.model_name
        self.requested_device = self.settings.device
        self.device, self.device_fallback_reason = _resolve_device(
            self.requested_device
        )
        self.model_load_count = 0
        self._model: Any | None = None
        self._model_init_lock = threading.Lock()
        self._inference_gate = asyncio.Lock()
        if self.device_fallback_reason is not None:
            logger.warning(
                "Local reranker requested %s but will use %s: %s",
                self.requested_device,
                self.device,
                self.device_fallback_reason,
            )

    async def score(
        self,
        query: str,
        candidates: Sequence[RetrievedCandidate],
    ) -> Sequence[float]:
        pairs = [(query, candidate.text) for candidate in candidates]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.timeout_seconds
        try:
            await asyncio.wait_for(
                self._inference_gate.acquire(),
                timeout=self.settings.timeout_seconds,
            )
        except TimeoutError as exc:
            raise RerankerOperationalError(
                "timeout", "Local reranker exceeded its bounded timeout."
            ) from exc

        submitted: Future[Sequence[float]] | None = None
        release_owned_by_future = False
        try:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError
            submitted = _get_inference_executor().submit(
                _run_native_inference,
                self._predict,
                pairs,
            )
            submitted.add_done_callback(
                lambda _future: self._release_inference_gate(loop)
            )
            release_owned_by_future = True
            wrapped = asyncio.wrap_future(submitted, loop=loop)
            return await asyncio.wait_for(
                asyncio.shield(wrapped),
                timeout=remaining,
            )
        except TimeoutError as exc:
            if submitted is not None:
                submitted.cancel()
            raise RerankerOperationalError(
                "timeout", "Local reranker exceeded its bounded timeout."
            ) from exc
        except RerankerOperationalError:
            raise
        except (ImportError, ModuleNotFoundError) as exc:
            raise RerankerOperationalError(
                "model_unavailable", "Local reranker dependency is unavailable."
            ) from exc
        except OSError as exc:
            raise RerankerOperationalError(
                "model_unavailable", "Local reranker model is unavailable."
            ) from exc
        except (RuntimeError, ValueError) as exc:
            raise RerankerOperationalError(
                "inference_failed", "Local reranker inference failed."
            ) from exc
        finally:
            if not release_owned_by_future and self._inference_gate.locked():
                self._inference_gate.release()

    def _release_inference_gate(self, loop: asyncio.AbstractEventLoop) -> None:
        def release() -> None:
            if self._inference_gate.locked():
                self._inference_gate.release()

        try:
            loop.call_soon_threadsafe(release)
        except RuntimeError:
            # The event loop already owns default-executor shutdown. No later
            # request can observe this instance once that loop is closed.
            return

    async def prepare(self) -> None:
        """Load the process model before the API accepts its first request."""

        await asyncio.to_thread(self._get_model)

    def _predict(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        model = self._get_model()
        return model.predict(
            pairs,
            batch_size=self.settings.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def _get_model(self) -> Any:
        if self._model is None:
            with self._model_init_lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    model = CrossEncoder(
                        self.settings.model_name,
                        device=self.device,
                        local_files_only=True,
                    )
                    self._model = model
                    self.model_load_count += 1
        return self._model


async def rerank_candidates(
    query: str,
    candidates: Sequence[RetrievedCandidate],
    *,
    scorer: CandidateScorer | None,
    enabled: bool,
    configured_model: str = DEFAULT_RERANKER_MODEL,
) -> RerankOutcome:
    """Rerank candidate union hoặc giữ chính xác thứ tự RRF khi model lỗi."""

    original = list(candidates)
    model_name = str(getattr(scorer, "model_name", configured_model) or configured_model)
    if not enabled:
        return RerankOutcome(original, "disabled", False, model_name, None, 0.0)
    if not original:
        return RerankOutcome(original, "skipped_no_candidates", True, model_name, None, 0.0)
    if scorer is None:
        raise TypeError("Enabled reranking requires a CandidateScorer.")

    started = time.perf_counter()
    try:
        raw_scores = await scorer.score(query, original)
        scores = _validated_scores(raw_scores, len(original))
    except RerankerOperationalError as exc:
        return RerankOutcome(
            original,
            "fallback",
            True,
            model_name,
            exc.reason,
            round((time.perf_counter() - started) * 1000, 3),
        )

    original_positions = {
        candidate.candidate_id: index for index, candidate in enumerate(original, start=1)
    }
    scored = list(zip(original, scores, strict=True))
    scored.sort(
        key=lambda item: (
            -item[1],
            item[0].rank if item[0].rank is not None else original_positions[item[0].candidate_id],
            item[0].candidate_id,
        )
    )
    reranked = [
        candidate.model_copy(update={"rerank_score": score, "rerank_rank": rank})
        for rank, (candidate, score) in enumerate(scored, start=1)
    ]
    return RerankOutcome(
        reranked,
        "succeeded",
        True,
        model_name,
        None,
        round((time.perf_counter() - started) * 1000, 3),
    )


def _validated_scores(raw_scores: Any, expected_count: int) -> list[float]:
    if isinstance(raw_scores, (str, bytes)):
        raise RerankerOperationalError("malformed_output", "Reranker scores are malformed.")
    try:
        values = list(raw_scores)
    except TypeError as exc:
        raise RerankerOperationalError(
            "malformed_output", "Reranker scores are not iterable."
        ) from exc
    if len(values) != expected_count:
        raise RerankerOperationalError(
            "score_count_mismatch", "Reranker returned an unexpected score count."
        )
    try:
        scores = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise RerankerOperationalError(
            "malformed_output", "Reranker returned a non-numeric score."
        ) from exc
    if not all(math.isfinite(score) for score in scores):
        raise RerankerOperationalError(
            "non_finite_score", "Reranker returned a non-finite score."
        )
    return scores


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _configured_device() -> str:
    value = os.getenv("RERANKER_DEVICE", DEFAULT_RERANKER_DEVICE).strip().casefold()
    if value in SUPPORTED_RERANKER_DEVICES:
        return value
    logger.warning(
        "Unsupported RERANKER_DEVICE=%r; using %s.",
        value,
        DEFAULT_RERANKER_DEVICE,
    )
    return DEFAULT_RERANKER_DEVICE


def _resolve_device(requested_device: str) -> tuple[str, str | None]:
    """Resolve the configured device without ever claiming unavailable CUDA."""

    if requested_device == "cpu":
        return "cpu", None
    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        return "cpu", "cuda_runtime_unavailable"
    if not torch.cuda.is_available():
        return "cpu", "cuda_unavailable"
    try:
        probe = torch.ones(1, device="cuda")
        _ = probe + 1
        torch.cuda.synchronize()
    except (AssertionError, RuntimeError):
        return "cpu", "cuda_probe_failed"
    return "cuda", None


def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(maximum, max(minimum, value))


def _bounded_float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(maximum, max(minimum, value))


__all__ = [
    "CandidateReranker",
    "CandidateScorer",
    "DEFAULT_RERANKER_DEVICE",
    "DEFAULT_RERANKER_MODEL",
    "RerankerOperationalError",
    "RerankerSettings",
    "RerankOutcome",
    "rerank_candidates",
    "shutdown_reranker_executor",
]
