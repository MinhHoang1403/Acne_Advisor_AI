"""Local-only semantic and hybrid reranker providers."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.retrieval.reranking.contracts import RerankCandidate, RerankerUnavailable, RerankScore, sort_scores
from src.retrieval.reranking.normalization import clamp_unit, normalize_score_map, sanitize_score

PROVIDER_LOCAL_RULES = "local_rules"
PROVIDER_LOCAL_SEMANTIC = "local_semantic"
PROVIDER_HYBRID = "hybrid"
PROVIDER_ALIASES = {
    "": PROVIDER_LOCAL_RULES,
    "local": PROVIDER_LOCAL_RULES,
    "local_rules": PROVIDER_LOCAL_RULES,
    "local_model": PROVIDER_LOCAL_SEMANTIC,
    "local_cross_encoder": PROVIDER_LOCAL_SEMANTIC,
    "semantic": PROVIDER_LOCAL_SEMANTIC,
    "local_semantic": PROVIDER_LOCAL_SEMANTIC,
    "hybrid": PROVIDER_HYBRID,
}

_SEMANTIC_RERANKER_CACHE: dict[tuple[str, str, int, int, int, int, bool], "LocalSemanticReranker"] = {}
_SEMANTIC_RERANKER_CACHE_LOCK = threading.Lock()


class SemanticBackend(Protocol):
    """Minimal local semantic backend interface."""

    name: str

    def score_pairs(
        self,
        query: str,
        documents: Sequence[str],
        *,
        batch_size: int,
        timeout_seconds: float | None = None,
    ) -> list[float]:
        """Return one raw semantic score per document."""


@dataclass(frozen=True)
class SemanticRerankerConfig:
    model_path: str = ""
    device: str = "cpu"
    batch_size: int = 8
    max_candidates: int = 32
    max_query_chars: int = 1000
    max_document_chars: int = 4000
    allow_fallback: bool = True

    @property
    def sanitized_model_identifier(self) -> str:
        if not self.model_path:
            return ""
        path = Path(self.model_path)
        return path.name or "local_model"


@dataclass(frozen=True)
class HybridFusionConfig:
    semantic_weight: float = 0.70
    rule_weight: float = 0.20
    retrieval_weight: float = 0.10

    def normalized(self) -> "HybridFusionConfig":
        weights = [
            max(0.0, float(self.semantic_weight)),
            max(0.0, float(self.rule_weight)),
            max(0.0, float(self.retrieval_weight)),
        ]
        total = sum(weights)
        if total <= 0:
            return HybridFusionConfig()
        return HybridFusionConfig(
            semantic_weight=round(weights[0] / total, 6),
            rule_weight=round(weights[1] / total, 6),
            retrieval_weight=round(weights[2] / total, 6),
        )


class CrossEncoderSemanticBackend:
    """Sentence-transformers CrossEncoder backend with local-only loading."""

    name = "sentence_transformers_cross_encoder"

    def __init__(self, config: SemanticRerankerConfig) -> None:
        self._config = config
        self._model = None
        self._lock = threading.Lock()

    def score_pairs(
        self,
        query: str,
        documents: Sequence[str],
        *,
        batch_size: int,
        timeout_seconds: float | None = None,
    ) -> list[float]:
        model = self._load_model()
        pairs = [(query, document) for document in documents]
        started = time.perf_counter()
        raw_scores: list[float] = []
        for offset in range(0, len(pairs), max(1, batch_size)):
            if timeout_seconds is not None and time.perf_counter() - started > timeout_seconds:
                raise TimeoutError("semantic_inference_timeout")
            batch = pairs[offset : offset + max(1, batch_size)]
            predicted = model.predict(batch)
            raw_scores.extend(float(score) for score in predicted)
        return raw_scores

    def warmup(self) -> None:
        """Load the local CrossEncoder before a latency-sensitive request."""

        self._load_model()

    def _load_model(self):
        if self._model is not None:
            return self._model
        model_path = self._config.model_path.strip()
        if not model_path:
            raise RerankerUnavailable("SEMANTIC_RERANK_MODEL_PATH is not configured.")
        if not Path(model_path).exists():
            raise RerankerUnavailable("Configured semantic rerank model path does not exist locally.")
        with self._lock:
            if self._model is not None:
                return self._model
            previous_offline = os.environ.get("TRANSFORMERS_OFFLINE")
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                from sentence_transformers import CrossEncoder  # type: ignore[import]

                try:
                    self._model = CrossEncoder(
                        model_path,
                        device=self._config.device,
                        max_length=self._config.max_document_chars,
                        local_files_only=True,
                    )
                except TypeError:
                    self._model = CrossEncoder(
                        model_path,
                        device=self._config.device,
                        max_length=self._config.max_document_chars,
                    )
            finally:
                if previous_offline is None:
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                else:
                    os.environ["TRANSFORMERS_OFFLINE"] = previous_offline
        return self._model


class LocalSemanticReranker:
    """Local semantic provider. It never downloads a model."""

    name = PROVIDER_LOCAL_SEMANTIC

    def __init__(
        self,
        config: SemanticRerankerConfig,
        backend: SemanticBackend | None = None,
    ) -> None:
        self.config = config
        self.backend = backend or CrossEncoderSemanticBackend(config)

    @property
    def available(self) -> bool:
        return bool(self.config.model_path and Path(self.config.model_path).exists())

    def warmup(self) -> None:
        """Eagerly load the built-in local backend without scoring a query."""

        if not self.available and isinstance(self.backend, CrossEncoderSemanticBackend):
            raise RerankerUnavailable("Local semantic reranker model is not provisioned.")
        if isinstance(self.backend, CrossEncoderSemanticBackend):
            self.backend.warmup()

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_n: int,
        timeout_seconds: float | None = None,
    ) -> list[RerankScore]:
        if not candidates:
            return []
        if not self.available and isinstance(self.backend, CrossEncoderSemanticBackend):
            raise RerankerUnavailable("Local semantic reranker model is not provisioned.")
        capped = _dedupe_candidates(list(candidates))[: max(0, self.config.max_candidates)]
        query_text = query[: self.config.max_query_chars]
        docs = [(candidate.text or "")[: self.config.max_document_chars] for candidate in capped]
        raw_scores = self.backend.score_pairs(
            query_text,
            docs,
            batch_size=self.config.batch_size,
            timeout_seconds=timeout_seconds,
        )
        if len(raw_scores) != len(capped):
            raise ValueError("invalid_semantic_scores: backend returned wrong score count.")
        normalized = normalize_score_map(
            {candidate.candidate_id: sanitize_score(score) for candidate, score in zip(capped, raw_scores, strict=True)}
        )
        output = [
            RerankScore(
                candidate_id=candidate.candidate_id,
                semantic_score=normalized[candidate.candidate_id],
                rule_score=None,
                retrieval_score=sanitize_score(candidate.retrieval_score),
                final_score=normalized[candidate.candidate_id],
                original_rank=candidate.original_rank,
                provider=self.name,
                diagnostics={
                    "backend": self.backend.name,
                    "raw_score": sanitize_score(raw_score),
                    "model_identifier": self.config.sanitized_model_identifier,
                },
            )
            for candidate, raw_score in zip(capped, raw_scores, strict=True)
        ]
        return sort_scores(output)[: max(0, min(top_n, len(output)))]


def hybrid_fuse_scores(
    candidates: Sequence[RerankCandidate],
    semantic_scores: dict[str, float],
    rule_scores: dict[str, float],
    *,
    config: HybridFusionConfig | None = None,
    provider: str = PROVIDER_HYBRID,
) -> list[RerankScore]:
    """Fuse semantic, local-rule and retrieval scores after normalization."""

    cfg = (config or HybridFusionConfig()).normalized()
    normalized_rules = normalize_score_map(rule_scores)
    retrieval_scores = {
        candidate.candidate_id: sanitize_score(candidate.retrieval_score)
        for candidate in candidates
    }
    normalized_retrieval = normalize_score_map(retrieval_scores)
    output: list[RerankScore] = []
    for candidate in candidates:
        semantic = clamp_unit(semantic_scores.get(candidate.candidate_id))
        rule = clamp_unit(normalized_rules.get(candidate.candidate_id))
        retrieval = clamp_unit(normalized_retrieval.get(candidate.candidate_id))
        final_score = round(
            cfg.semantic_weight * semantic
            + cfg.rule_weight * rule
            + cfg.retrieval_weight * retrieval,
            6,
        )
        output.append(
            RerankScore(
                candidate_id=candidate.candidate_id,
                semantic_score=semantic,
                rule_score=rule,
                retrieval_score=retrieval,
                final_score=final_score,
                original_rank=candidate.original_rank,
                provider=provider,
                diagnostics={
                    "weights": {
                        "semantic": cfg.semantic_weight,
                        "rule": cfg.rule_weight,
                        "retrieval": cfg.retrieval_weight,
                    }
                },
            )
        )
    return sort_scores(output)


def canonical_provider_name(provider: str) -> str:
    return PROVIDER_ALIASES.get((provider or "").strip().lower(), "unknown")


def _dedupe_candidates(candidates: list[RerankCandidate]) -> list[RerankCandidate]:
    deduped: dict[str, RerankCandidate] = {}
    for candidate in candidates:
        existing = deduped.get(candidate.candidate_id)
        if existing is None or candidate.original_rank < existing.original_rank:
            deduped[candidate.candidate_id] = candidate
    return sorted(deduped.values(), key=lambda item: (item.original_rank, item.candidate_id))


def reranker_provider_config_from_env(env: dict[str, str] | None = None) -> dict[str, object]:
    source = env or os.environ
    return {
        "semantic": semantic_config_from_env(source),
        "hybrid": hybrid_config_from_env(source),
        "provider": canonical_provider_name(source.get("RERANK_PROVIDER", "local_rules")),
    }


def semantic_config_from_env(env: dict[str, str] | None = None) -> SemanticRerankerConfig:
    source = env or os.environ
    return SemanticRerankerConfig(
        model_path=str(source.get("SEMANTIC_RERANK_MODEL_PATH", "") or "").strip(),
        device=str(source.get("SEMANTIC_RERANK_DEVICE", "cpu") or "cpu"),
        batch_size=_env_int(source, "SEMANTIC_RERANK_BATCH_SIZE", 8, minimum=1),
        max_candidates=_env_int(source, "SEMANTIC_RERANK_MAX_CANDIDATES", 32, minimum=1),
        max_query_chars=_env_int(source, "SEMANTIC_RERANK_MAX_QUERY_CHARS", 1000, minimum=1),
        max_document_chars=_env_int(source, "SEMANTIC_RERANK_MAX_DOCUMENT_CHARS", 4000, minimum=1),
        allow_fallback=_env_bool(source, "SEMANTIC_RERANK_ALLOW_FALLBACK", True),
    )


def hybrid_config_from_env(env: dict[str, str] | None = None) -> HybridFusionConfig:
    source = env or os.environ
    return HybridFusionConfig(
        semantic_weight=_env_float(source, "SEMANTIC_RERANK_WEIGHT", 0.70),
        rule_weight=_env_float(source, "RULE_RERANK_WEIGHT", 0.20),
        retrieval_weight=_env_float(source, "RETRIEVAL_RERANK_WEIGHT", 0.10),
    ).normalized()


def build_semantic_reranker_from_env(env: dict[str, str] | None = None) -> LocalSemanticReranker:
    return LocalSemanticReranker(semantic_config_from_env(env))


def get_cached_semantic_reranker_from_env(env: dict[str, str] | None = None) -> LocalSemanticReranker:
    """Return a process-level semantic reranker for the current local model config."""

    config = semantic_config_from_env(env)
    key = _semantic_reranker_cache_key(config)
    with _SEMANTIC_RERANKER_CACHE_LOCK:
        cached = _SEMANTIC_RERANKER_CACHE.get(key)
        if cached is None:
            cached = LocalSemanticReranker(config)
            _SEMANTIC_RERANKER_CACHE[key] = cached
        return cached


def clear_semantic_reranker_cache_for_tests() -> None:
    """Clear the process-level reranker cache for isolated tests."""

    with _SEMANTIC_RERANKER_CACHE_LOCK:
        _SEMANTIC_RERANKER_CACHE.clear()


def _semantic_reranker_cache_key(config: SemanticRerankerConfig) -> tuple[str, str, int, int, int, int, bool]:
    return (
        str(Path(config.model_path).expanduser()) if config.model_path else "",
        config.device,
        int(config.batch_size),
        int(config.max_candidates),
        int(config.max_query_chars),
        int(config.max_document_chars),
        bool(config.allow_fallback),
    )


def _env_bool(env: dict[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value in {None, ""}:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(env: dict[str, str], name: str, default: int, *, minimum: int) -> int:
    try:
        return max(minimum, int(str(env.get(name, default))))
    except (TypeError, ValueError):
        return default


def _env_float(env: dict[str, str], name: str, default: float) -> float:
    try:
        return float(str(env.get(name, default)))
    except (TypeError, ValueError):
        return default


__all__ = [
    "HybridFusionConfig",
    "LocalSemanticReranker",
    "PROVIDER_HYBRID",
    "PROVIDER_LOCAL_RULES",
    "PROVIDER_LOCAL_SEMANTIC",
    "SemanticBackend",
    "SemanticRerankerConfig",
    "build_semantic_reranker_from_env",
    "canonical_provider_name",
    "clear_semantic_reranker_cache_for_tests",
    "get_cached_semantic_reranker_from_env",
    "hybrid_config_from_env",
    "hybrid_fuse_scores",
    "reranker_provider_config_from_env",
    "semantic_config_from_env",
]
