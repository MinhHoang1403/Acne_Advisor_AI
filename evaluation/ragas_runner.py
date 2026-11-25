"""Bộ công cụ đánh giá hệ thống Acne Advisor AI bằng phương pháp RAGAS.

Tham khảo khoa học:
- Es et al. (2023), "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
  (arXiv:2309.15217).
- Tài liệu chính thức RAGAS 0.4.x (https://docs.ragas.io/).

Module cung cấp:
1. Adapter embeddings ``ProjectGeminiEmbeddings`` dùng chung ``google-genai`` của dự án
   để đảm bảo chiều vector đồng nhất (3072 chiều với ``models/gemini-embedding-2``).
2. Hàm chạy trực tiếp ``run_clinical_agent`` trên bộ câu hỏi chuẩn để thu thập câu trả lời,
   ngữ cảnh truy hồi thực tế, độ trễ và lượng token.
3. Hàm chấm 5 chỉ số RAGAS cốt lõi (Faithfulness, AnswerRelevancy, ContextPrecision,
   ContextRecall, AnswerCorrectness) kết hợp tỷ lệ từ chối an toàn (Safe Abstention Accuracy).
4. Hàm xuất báo cáo chi tiết ra file Markdown (``.md``) và JSON thô.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ragas.embeddings.base import BaseRagasEmbedding

DEFAULT_DATASET_PATH = Path("evaluation/dataset.json")
DEFAULT_RESULTS_DIR = Path("evaluation/results")
DEFAULT_JUDGE_PROVIDER = "nvidia"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_JUDGE_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_FALLBACK_JUDGE_MODEL = "mistralai/mistral-large-4-1t-a52b"
DEFAULT_FALLBACK_JUDGE_MODELS = ("mistralai/mistral-large-4-1t-a52b",)
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSIONS = 3072

RAGAS_METRIC_KEYS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)

METRIC_LABELS_VI: dict[str, str] = {
    "faithfulness": "Độ trung thực với bằng chứng (Faithfulness)",
    "answer_relevancy": "Độ liên quan câu trả lời (Answer Relevancy)",
    "context_precision": "Độ chính xác ngữ cảnh (Context Precision)",
    "context_recall": "Độ bao phủ ngữ cảnh (Context Recall)",
    "answer_correctness": "Độ chính xác câu trả lời (Answer Correctness)",
}


class ProjectGeminiEmbeddings(BaseRagasEmbedding):
    """Adapter embeddings cho RAGAS tái sử dụng ``google-genai`` SDK của dự án.

    Giúp tránh lệch số chiều vector giữa single-text và batch-text embedding khi chấm
    chỉ số ``AnswerRelevancy`` bằng ``models/gemini-embedding-2`` (3072 chiều).
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        api_key: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.dimensions = dimensions
        self._api_key = api_key

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        del kwargs
        vectors = self.embed_texts([text])
        return vectors[0]

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        del kwargs
        return await asyncio.to_thread(self.embed_text, text)

    def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        del kwargs
        if not texts:
            return []
        from src.integrations.google_genai import embed_texts_sync

        return embed_texts_sync(
            list(texts),
            model_name=self.model,
            task_type=None,
            expected_dimensions=self.dimensions,
            output_dimensions=self.dimensions,
            api_key=self._api_key,
        )

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        del kwargs
        return await asyncio.to_thread(self.embed_texts, texts)


@dataclass(frozen=True)
class EvaluationCase:
    """Một mẫu câu hỏi chuẩn trong bộ đánh giá RAGAS."""

    case_id: str
    category: str
    expected_behavior: str
    user_input: str
    reference: str


def load_evaluation_dataset(
    path: Path = DEFAULT_DATASET_PATH,
) -> tuple[dict[str, Any], list[EvaluationCase]]:
    """Đọc và kiểm tra tính hợp lệ của file dataset JSON."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise ValueError(f"Định dạng dataset không hợp lệ tại {path}.")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for item in raw["cases"]:
        case_id = str(item.get("case_id") or "").strip()
        category = str(item.get("category") or "general").strip()
        expected = str(item.get("expected_behavior") or "answer").strip()
        user_input = str(item.get("user_input") or "").strip()
        reference = str(item.get("reference") or "").strip()
        if not case_id or not user_input or not reference:
            raise ValueError(f"Thiếu trường bắt buộc trong mẫu đánh giá: {item}")
        if expected not in {"answer", "abstain"}:
            raise ValueError(f"expected_behavior không hợp lệ ({expected}) ở ca {case_id}")
        if case_id in seen_ids:
            raise ValueError(f"Trùng case_id: {case_id}")
        seen_ids.add(case_id)
        cases.append(
            EvaluationCase(
                case_id=case_id,
                category=category,
                expected_behavior=expected,
                user_input=user_input,
                reference=reference,
            )
        )
    return raw, cases


def _extract_retrieved_contexts(agent_result: dict[str, Any]) -> list[str]:
    """Trích xuất danh sách đoạn chứng cứ thực tế mà hệ thống đã dùng để sinh câu trả lời."""

    contexts: list[str] = []
    packed = agent_result.get("packed_context")
    if isinstance(packed, dict):
        for item in packed.get("items") or []:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
                if text:
                    contexts.append(text)
    if contexts:
        return contexts
    for item in agent_result.get("vector_contexts") or []:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or "").strip()
            if text:
                contexts.append(text)
    return contexts


async def _warmup_reranker() -> None:
    """Tải trước model reranker lên GPU/CPU một lần để tránh cold-start timeout ở câu đầu."""

    from src.retrieval.service import warm_process_reranker

    try:
        await warm_process_reranker()
    except Exception:
        return


async def run_system_on_dataset(
    cases: list[EvaluationCase],
    *,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    verbose: bool = False,
    max_provider_retries: int = 2,
    provider_retry_delay_seconds: float = 12.0,
    case_delay_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    """Chạy trực tiếp hệ thống ``run_clinical_agent`` trên từng ca đánh giá.

    Tắt đọc cache (``bypass_cache=True``) để mỗi lần đánh giá phản ánh đúng chất lượng
    truy hồi và sinh câu trả lời hiện hành của hệ thống. Tự động thử lại có giới hạn nếu
    gặp lỗi hạ tầng tạm thời từ provider (429/503/``provider_unavailable``) khi chạy batch dài.
    """

    from src.agent.graph import run_clinical_agent

    await _warmup_reranker()

    records: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        started = time.perf_counter()
        error_text: str | None = None
        result: dict[str, Any] = {}
        for attempt in range(max_provider_retries + 1):
            error_text = None
            result = {}
            try:
                result = await run_clinical_agent(
                    message=case.user_input,
                    conversation_history=[],
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    bypass_cache=True,
                    include_generation_diagnostics=True,
                    session_id=f"ragas-eval-{case.case_id.lower()}",
                )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
            daily_quota_detail = _extract_daily_quota_detail(result, error_text)
            if daily_quota_detail is not None:
                raise RuntimeError(
                    f"System LLM ({llm_provider or 'default'}/{llm_model or 'default'}) đã cạn hạn mức theo ngày (RPD) "
                    f"tại ca {case.case_id}: {daily_quota_detail}. "
                    "Hãy chờ reset hạn mức ngày hoặc đổi model/API key trước khi chạy lại bộ đánh giá."
                )
            transient_fallback = result.get("fallback_reason_code") in {
                "provider_unavailable",
                "generation_unavailable",
            }
            transient_error = error_text is not None and _is_rate_limit_error(RuntimeError(error_text))
            if (transient_fallback or transient_error) and attempt < max_provider_retries:
                backoff = provider_retry_delay_seconds * (attempt + 1)
                if verbose:
                    print(
                        f"[System {idx}/{len(cases)}] {case.case_id} gặp lỗi tạm thời từ provider "
                        f"(attempt {attempt + 1}/{max_provider_retries + 1}), chờ {backoff:.0f}s rồi thử lại...",
                        flush=True,
                    )
                await asyncio.sleep(backoff)
                started = time.perf_counter()
                continue
            break
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)

        decision = result.get("agent_decision") or {}
        token_usage = result.get("generation_token_usage") or {}
        record = {
            "index": idx,
            "case_id": case.case_id,
            "category": case.category,
            "expected_behavior": case.expected_behavior,
            "user_input": case.user_input,
            "reference": case.reference,
            "response": str(result.get("answer") or result.get("final_answer") or "").strip(),
            "retrieved_contexts": _extract_retrieved_contexts(result),
            "sources": list(result.get("sources") or []),
            "action": str(decision.get("action") or ""),
            "reason_code": str(decision.get("reason_code") or ""),
            "fallback_applied": bool(result.get("fallback_applied")),
            "safety_override": bool(result.get("safety_decision")),
            "actual_provider": result.get("actual_provider"),
            "actual_model": result.get("actual_model"),
            "latency_ms": elapsed_ms,
            "token_usage": dict(token_usage) if isinstance(token_usage, dict) else {},
            "error": error_text,
        }
        records.append(record)
        if verbose:
            print(
                f"[System {idx}/{len(cases)}] {case.case_id} | action={record['action'] or '-'} "
                f"| contexts={len(record['retrieved_contexts'])} | {elapsed_ms:.0f}ms"
                + (f" | ERROR={error_text}" if error_text else ""),
                flush=True,
            )
        if case_delay_seconds > 0 and idx < len(cases):
            await asyncio.sleep(case_delay_seconds)
    return records


def build_ragas_evaluators(
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_provider: str = DEFAULT_JUDGE_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
    judge_api_key: str | None = None,
    judge_base_url: str | None = None,
) -> tuple[Any, ProjectGeminiEmbeddings]:
    """Khởi tạo Judge LLM (NVIDIA NIM, Gemini hoặc OpenAI) và Embeddings cho RAGAS 0.4.x."""

    from ragas.llms import llm_factory

    google_key = (api_key or os.getenv("GOOGLE_API_KEY", "")).strip()
    if not google_key:
        raise RuntimeError("Thiếu GOOGLE_API_KEY trong biến môi trường cho embeddings.")

    provider_norm = judge_provider.strip().lower()
    if provider_norm == "nvidia":
        import instructor
        from openai import AsyncOpenAI

        nvidia_key = (judge_api_key or os.getenv("NVIDIA_API_KEY", "")).strip()
        if not nvidia_key:
            raise RuntimeError("Thiếu NVIDIA_API_KEY để chạy RAGAS judge với provider='nvidia'.")
        base_url = (
            judge_base_url
            or os.getenv("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL)
            or DEFAULT_NVIDIA_BASE_URL
        ).strip()
        nvidia_client = AsyncOpenAI(
            api_key=nvidia_key,
            base_url=base_url,
            timeout=60.0,
            max_retries=1,
        )
        # Dùng MD_JSON và tắt thinking mặc định để model 550B/1T trên NVIDIA NIM
        # xuất JSON chuẩn Pydantic nhanh, ổn định và không bị cắt bởi chuỗi suy nghĩ dài.
        judge_llm = llm_factory(
            judge_model,
            provider="openai",
            client=nvidia_client,
            mode=instructor.Mode.MD_JSON,
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    elif provider_norm == "openai":
        from openai import AsyncOpenAI

        openai_key = (judge_api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        if not openai_key:
            raise RuntimeError("Thiếu OPENAI_API_KEY để chạy RAGAS judge với provider='openai'.")
        openai_kwargs: dict[str, Any] = {"api_key": openai_key, "timeout": 60.0}
        if judge_base_url:
            openai_kwargs["base_url"] = judge_base_url.strip()
        openai_client = AsyncOpenAI(**openai_kwargs)
        judge_llm = llm_factory(judge_model, provider="openai", client=openai_client)
    else:
        from google import genai

        google_client = genai.Client(api_key=google_key)
        judge_llm = llm_factory(judge_model, provider="google", client=google_client)

    embeddings = ProjectGeminiEmbeddings(
        model=embedding_model,
        dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
        api_key=google_key,
    )
    return judge_llm, embeddings


def _clean_score(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return round(numeric, 4)


def _is_daily_quota_error(exc: BaseException | str) -> bool:
    """Phát hiện lỗi cạn hạn mức theo ngày (RPD) để dừng/chuyển judge ngay, tránh chờ retry vô ích."""

    text = exc.lower() if isinstance(exc, str) else f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "requestsperday",
            "per day",
            "per day limit exceeded",
            "daily project quota",
            "daily quota",
            "generate_content_free_tier_requests",
        )
    )


def _extract_daily_quota_detail(result: dict[str, Any], error_text: str | None) -> str | None:
    """Trích thông điệp lỗi RPD từ exception hoặc metadata fallback của ``run_clinical_agent``."""

    candidates: list[str] = []
    if error_text:
        candidates.append(error_text)
    if isinstance(result, dict):
        for key in ("fallback_reason", "retrieval_error"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        decision = result.get("agent_decision")
        if isinstance(decision, dict):
            for key in ("error", "rationale", "reason"):
                value = decision.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())
        for item in result.get("agent_decision_history") or []:
            if isinstance(item, dict):
                for key in ("error", "rationale", "reason"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        candidates.append(value.strip())
    for detail in candidates:
        if _is_daily_quota_error(detail):
            return detail
    return None


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in ("429", "503", "resource_exhausted", "rate limit", "unavailable"))


async def _run_metric_with_retry(
    factory: Any,
    *,
    max_retries: int = 2,
    base_backoff: float = 10.0,
) -> Any:
    """Chạy 1 metric RAGAS kèm tự động lùi thời gian khi gặp 429 RPM/503 tạm thời."""

    for attempt in range(max_retries + 1):
        try:
            return await factory()
        except Exception as exc:
            if _is_daily_quota_error(exc) or attempt >= max_retries or not _is_rate_limit_error(exc):
                raise
            await asyncio.sleep(base_backoff * (attempt + 1))
    raise RuntimeError("Unreachable retry state.")


def _strip_presentation_disclaimer(text: str) -> str:
    """Loại bỏ dòng disclaimer trình bày của hệ thống trước khi tách mệnh đề chấm RAGAS.

    Disclaimer là thông báo pháp lý cố định do presentation layer chèn vào sau khi sinh,
    không phải mệnh đề y khoa của LLM nên không được tính là mệnh đề sai/không có bằng chứng.
    """

    from src.agent.answer_formatting import CANONICAL_DISCLAIMER, LEGACY_DISCLAIMERS

    cleaned = text.strip()
    for disclaimer in (CANONICAL_DISCLAIMER, *LEGACY_DISCLAIMERS):
        cleaned = cleaned.replace(disclaimer, "").strip()
    return cleaned


async def score_records_with_ragas(
    records: list[dict[str, Any]],
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_provider: str = DEFAULT_JUDGE_PROVIDER,
    fallback_judge_model: str | None = DEFAULT_FALLBACK_JUDGE_MODEL,
    fallback_judge_models: tuple[str, ...] | list[str] | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
    judge_api_key: str | None = None,
    judge_base_url: str | None = None,
    request_delay_seconds: float | None = None,
    max_rate_limit_retries: int = 2,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Chấm 5 chỉ số RAGAS trên các ca cần trả lời và kiểm tra tỷ lệ từ chối an toàn."""

    from ragas.metrics.collections import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    provider_norm = judge_provider.strip().lower()
    judge_llm, embeddings = build_ragas_evaluators(
        judge_model=judge_model,
        judge_provider=provider_norm,
        embedding_model=embedding_model,
        api_key=api_key,
        judge_api_key=judge_api_key,
        judge_base_url=judge_base_url,
    )

    def _build_metric_set(llm_instance: Any) -> dict[str, Any]:
        return {
            "faithfulness": Faithfulness(llm=llm_instance),
            "answer_relevancy": AnswerRelevancy(llm=llm_instance, embeddings=embeddings),
            "context_precision": ContextPrecision(llm=llm_instance),
            "context_recall": ContextRecall(llm=llm_instance),
            "answer_correctness": AnswerCorrectness(
                llm=llm_instance,
                embeddings=embeddings,
                weights=[0.75, 0.25],
            ),
        }

    # Xây chuỗi judge ưu tiên: judge chính + các model dự phòng cùng provider (tự động loại trùng).
    judge_chain: list[tuple[str, dict[str, Any]]] = [(judge_model, _build_metric_set(judge_llm))]
    if provider_norm in {"google", "nvidia"}:
        raw_fallbacks: list[str] = []
        if fallback_judge_models is not None:
            raw_fallbacks.extend(str(m).strip() for m in fallback_judge_models if str(m).strip())
        elif fallback_judge_model:
            raw_fallbacks.append(str(fallback_judge_model).strip())
        seen_models = {judge_model}
        for fb_model in raw_fallbacks:
            if fb_model in seen_models:
                continue
            seen_models.add(fb_model)
            fb_llm, _ = build_ragas_evaluators(
                judge_model=fb_model,
                judge_provider=provider_norm,
                embedding_model=embedding_model,
                api_key=api_key,
                judge_api_key=judge_api_key,
                judge_base_url=judge_base_url,
            )
            judge_chain.append((fb_model, _build_metric_set(fb_llm)))

    exhausted_daily_models: set[str] = set()

    use_parallel = provider_norm == "openai"
    pacing_delay = (
        0.0
        if request_delay_seconds is None and use_parallel
        else 1.0
        if request_delay_seconds is None and provider_norm == "nvidia"
        else 2.5
        if request_delay_seconds is None
        else max(0.0, request_delay_seconds)
    )

    short_labels = {
        "faithfulness": "faith",
        "answer_relevancy": "rel",
        "context_precision": "prec",
        "context_recall": "rec",
        "answer_correctness": "corr",
    }

    scored_records: list[dict[str, Any]] = []
    for rec in records:
        enriched = dict(rec)
        scores: dict[str, float | None] = {key: None for key in RAGAS_METRIC_KEYS}
        metric_errors: dict[str, str] = {}

        if rec["expected_behavior"] == "abstain":
            # Với ca ngoài phạm vi hoặc thiếu bằng chứng, chấp nhận cả abstain và safety override.
            safe_handled = (
                bool(rec.get("fallback_applied"))
                or rec.get("action") == "abstain"
                or bool(rec.get("safety_override"))
                or rec.get("actual_provider") == "system"
            )
            enriched["abstention_correct"] = safe_handled
            enriched["ragas_scores"] = scores
            enriched["metric_errors"] = metric_errors
            scored_records.append(enriched)
            continue

        enriched["abstention_correct"] = None
        response_text = _strip_presentation_disclaimer(str(rec.get("response") or ""))
        contexts = list(rec.get("retrieved_contexts") or [])
        user_input = str(rec.get("user_input") or "")
        reference = str(rec.get("reference") or "")

        if response_text and contexts:
            def _make_call(mset: dict[str, Any], key: str) -> Any:
                if key == "faithfulness":
                    return mset[key].ascore(user_input=user_input, response=response_text, retrieved_contexts=contexts)
                if key == "answer_relevancy":
                    return mset[key].ascore(user_input=user_input, response=response_text)
                if key == "context_precision":
                    return mset[key].ascore(user_input=user_input, reference=reference, retrieved_contexts=contexts)
                if key == "context_recall":
                    return mset[key].ascore(user_input=user_input, retrieved_contexts=contexts, reference=reference)
                return mset[key].ascore(user_input=user_input, response=response_text, reference=reference)

            async def _score_one_metric(key: str) -> Any:
                last_exc: Exception | None = None
                for idx_j, (model_name, mset) in enumerate(judge_chain):
                    if model_name in exhausted_daily_models:
                        continue
                    try:
                        return await _run_metric_with_retry(
                            lambda ms=mset, k=key: _make_call(ms, k),
                            max_retries=max_rate_limit_retries if idx_j == 0 else 1,
                        )
                    except Exception as exc:
                        last_exc = exc
                        if _is_daily_quota_error(exc):
                            exhausted_daily_models.add(model_name)
                            if verbose:
                                print(
                                    f"  [Judge] {model_name} hết quota theo ngày -> chuyển sang model dự phòng.",
                                    flush=True,
                                )
                            continue
                        if _is_rate_limit_error(exc):
                            continue
                        raise
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError("Tất cả judge model đều đã cạn quota theo ngày.")

            if use_parallel:
                results = await asyncio.gather(
                    *(_score_one_metric(key) for key in RAGAS_METRIC_KEYS),
                    return_exceptions=True,
                )
                for key, outcome in zip(RAGAS_METRIC_KEYS, results, strict=True):
                    if isinstance(outcome, BaseException):
                        metric_errors[key] = f"{type(outcome).__name__}: {outcome}"
                    else:
                        scores[key] = _clean_score(getattr(outcome, "value", outcome))
            else:
                for key in RAGAS_METRIC_KEYS:
                    try:
                        outcome = await _score_one_metric(key)
                        scores[key] = _clean_score(getattr(outcome, "value", outcome))
                    except Exception as exc:
                        metric_errors[key] = f"{type(exc).__name__}: {exc}"
                    if pacing_delay > 0:
                        await asyncio.sleep(pacing_delay)
        else:
            metric_errors["skipped"] = "Không có câu trả lời hoặc ngữ cảnh truy hồi để chấm RAGAS."

        enriched["ragas_scores"] = scores
        enriched["metric_errors"] = metric_errors
        scored_records.append(enriched)
        if verbose:
            score_str = ", ".join(
                f"{short_labels[k]}={_fmt_score(scores[k])}" for k in RAGAS_METRIC_KEYS
            )
            print(
                f"[RAGAS {len(scored_records)}/{len(records)}] {rec['case_id']} | {score_str}"
                + (f" | errs={list(metric_errors.keys())}" if metric_errors else ""),
                flush=True,
            )

    return scored_records


def summarize_evaluation(
    scored_records: list[dict[str, Any]],
    *,
    dataset_meta: dict[str, Any] | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    kb_build_id: str | None = None,
) -> dict[str, Any]:
    """Tổng hợp điểm số RAGAS toàn cục, theo từng nhóm câu hỏi, độ trễ và token."""

    answer_cases = [r for r in scored_records if r.get("expected_behavior") == "answer"]
    abstain_cases = [r for r in scored_records if r.get("expected_behavior") == "abstain"]

    metric_summary: dict[str, dict[str, Any]] = {}
    for key in RAGAS_METRIC_KEYS:
        vals = [
            float(r["ragas_scores"][key])
            for r in answer_cases
            if isinstance(r.get("ragas_scores", {}).get(key), (int, float))
        ]
        metric_summary[key] = {
            "mean": round(statistics.mean(vals), 4) if vals else None,
            "min": round(min(vals), 4) if vals else None,
            "max": round(max(vals), 4) if vals else None,
            "count": len(vals),
        }

    # Tổng hợp theo từng category câu hỏi.
    categories: dict[str, list[dict[str, Any]]] = {}
    for rec in scored_records:
        categories.setdefault(str(rec.get("category") or "general"), []).append(rec)

    category_summary: dict[str, dict[str, Any]] = {}
    for cat, items in categories.items():
        cat_metrics: dict[str, float | None] = {}
        for key in RAGAS_METRIC_KEYS:
            vals = [
                float(r["ragas_scores"][key])
                for r in items
                if isinstance(r.get("ragas_scores", {}).get(key), (int, float))
            ]
            cat_metrics[key] = round(statistics.mean(vals), 4) if vals else None
        latencies = [
            float(r["latency_ms"]) for r in items if isinstance(r.get("latency_ms"), (int, float))
        ]
        category_summary[cat] = {
            "count": len(items),
            "metrics": cat_metrics,
            "mean_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        }

    abstain_hits = sum(1 for r in abstain_cases if r.get("abstention_correct") is True)
    abstain_accuracy = round(abstain_hits / len(abstain_cases), 4) if abstain_cases else None

    all_latencies = sorted(
        float(r["latency_ms"])
        for r in scored_records
        if isinstance(r.get("latency_ms"), (int, float))
    )
    p50_ms = round(statistics.median(all_latencies), 1) if all_latencies else None
    p95_idx = (
        min(len(all_latencies) - 1, math.ceil(0.95 * len(all_latencies)) - 1)
        if all_latencies
        else 0
    )
    p95_ms = round(all_latencies[p95_idx], 1) if all_latencies else None
    mean_ms = round(statistics.mean(all_latencies), 1) if all_latencies else None

    total_input_tokens = sum(
        int((r.get("token_usage") or {}).get("input") or 0) for r in scored_records
    )
    total_output_tokens = sum(
        int((r.get("token_usage") or {}).get("output") or 0) for r in scored_records
    )
    total_tokens = sum(int((r.get("token_usage") or {}).get("total") or 0) for r in scored_records)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": (dataset_meta or {}).get("dataset_name", "acne_advisor_ragas_benchmark"),
        "judge_model": judge_model,
        "embedding_model": embedding_model,
        "kb_build_id": kb_build_id,
        "counts": {
            "total_cases": len(scored_records),
            "answer_cases": len(answer_cases),
            "abstain_cases": len(abstain_cases),
            "runtime_errors": sum(1 for r in scored_records if r.get("error")),
        },
        "ragas_summary": metric_summary,
        "safe_abstention": {
            "correct": abstain_hits,
            "total": len(abstain_cases),
            "accuracy": abstain_accuracy,
        },
        "latency_ms": {
            "mean": mean_ms,
            "p50": p50_ms,
            "p95": p95_ms,
        },
        "token_usage": {
            "input": total_input_tokens,
            "output": total_output_tokens,
            "total": total_tokens,
        },
        "by_category": category_summary,
        "records": scored_records,
    }


def _fmt_score(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "N/A"


def _status_badge(value: float | None) -> str:
    if not isinstance(value, (int, float)):
        return "Chưa chấm"
    if value >= 0.80:
        return "Tốt (>= 0.80)"
    if value >= 0.65:
        return "Khá (0.65 - 0.79)"
    return "Cần cải thiện (< 0.65)"


def render_markdown_report(summary: dict[str, Any]) -> str:
    """Kết xuất báo cáo đánh giá RAGAS chi tiết sang định dạng Markdown sạch."""

    counts = summary["counts"]
    latency = summary["latency_ms"]
    tokens = summary["token_usage"]
    abstention = summary["safe_abstention"]

    lines: list[str] = [
        "# Báo cáo Đánh giá Hệ thống Acne Advisor AI (RAGAS)",
        "",
        f"- **Thời điểm chạy (UTC):** `{summary['generated_at']}`",
        f"- **Bộ dữ liệu đánh giá:** `{summary['dataset_name']}` ({counts['total_cases']} câu hỏi: {counts['answer_cases']} câu trả lời, {counts['abstain_cases']} câu từ chối an toàn)",
        f"- **Judge LLM (RAGAS):** `{summary['judge_model']}`",
        f"- **Embedding Model:** `{summary['embedding_model']}`",
        f"- **Knowledge Build ID:** `{summary.get('kb_build_id') or 'unknown'}`",
        "",
        "## 1. Tổng quan chỉ số RAGAS",
        "",
        "Phương pháp đánh giá tham chiếu bài báo *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (Es et al., 2023, `arXiv:2309.15217`). Thang điểm từ `0.000` đến `1.000` (càng cao càng tốt).",
        "",
        "| Chỉ số | Trung bình | Thấp nhất | Cao nhất | Số ca chấm | Đánh giá |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for key in RAGAS_METRIC_KEYS:
        m = summary["ragas_summary"][key]
        lines.append(
            f"| {METRIC_LABELS_VI[key]} | {_fmt_score(m['mean'])} | {_fmt_score(m['min'])} | "
            f"{_fmt_score(m['max'])} | {m['count']} | {_status_badge(m['mean'])} |"
        )

    lines.extend(
        [
            "",
            "## 2. Chỉ số Vận hành & An toàn",
            "",
            "| Hạng mục | Kết quả |",
            "| --- | --- |",
            f"| Tỷ lệ từ chối an toàn (Safe Abstention Accuracy) | {abstention['correct']}/{abstention['total']} ({_fmt_score(abstention['accuracy'])}) |",
            f"| Độ trễ trung bình (Mean Latency) | {latency['mean']} ms |",
            f"| Độ trễ trung vị (P50 Latency) | {latency['p50']} ms |",
            f"| Độ trễ P95 (P95 Latency) | {latency['p95']} ms |",
            f"| Tổng token tiêu thụ (Generation) | input={tokens['input']}, output={tokens['output']}, total={tokens['total']} |",
            f"| Số lỗi runtime | {counts['runtime_errors']} |",
            "",
            "## 3. Phân tích theo nhóm câu hỏi",
            "",
            "| Nhóm câu hỏi | Số ca | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Độ trễ TB (ms) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for cat, info in summary["by_category"].items():
        cm = info["metrics"]
        lines.append(
            f"| `{cat}` | {info['count']} | {_fmt_score(cm['faithfulness'])} | "
            f"{_fmt_score(cm['answer_relevancy'])} | {_fmt_score(cm['context_precision'])} | "
            f"{_fmt_score(cm['context_recall'])} | {_fmt_score(cm['answer_correctness'])} | "
            f"{info['mean_latency_ms']} |"
        )

    lines.extend(
        [
            "",
            "## 4. Bảng điểm chi tiết từng câu hỏi",
            "",
            "| Case ID | Nhóm | Hành động | Faithfulness | Relevancy | Precision | Recall | Correctness | Từ chối đúng | Độ trễ (ms) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for rec in summary["records"]:
        sc = rec.get("ragas_scores") or {}
        abstain_col = (
            "Đạt"
            if rec.get("abstention_correct") is True
            else "Chưa đạt"
            if rec.get("abstention_correct") is False
            else "-"
        )
        lines.append(
            f"| `{rec['case_id']}` | `{rec['category']}` | `{rec.get('action') or '-'}` | "
            f"{_fmt_score(sc.get('faithfulness'))} | {_fmt_score(sc.get('answer_relevancy'))} | "
            f"{_fmt_score(sc.get('context_precision'))} | {_fmt_score(sc.get('context_recall'))} | "
            f"{_fmt_score(sc.get('answer_correctness'))} | {abstain_col} | {rec.get('latency_ms')} |"
        )

    lines.extend(["", "## 5. Chi tiết câu trả lời & ngữ cảnh từng ca", ""])
    for rec in summary["records"]:
        sc = rec.get("ragas_scores") or {}
        lines.append(f"### {rec['case_id']} — `{rec['category']}`")
        lines.append("")
        lines.append(f"- **Câu hỏi:** {rec['user_input']}")
        lines.append(f"- **Đáp án chuẩn tham chiếu:** {rec['reference']}")
        lines.append(
            f"- **Hành động hệ thống:** `{rec.get('action')}` (fallback=`{rec.get('fallback_applied')}`, nguồn=`{len(rec.get('sources') or [])}`, độ trễ=`{rec.get('latency_ms')} ms`)"
        )
        if rec["expected_behavior"] == "answer":
            lines.append(
                "- **Điểm RAGAS:** "
                f"Faithfulness=`{_fmt_score(sc.get('faithfulness'))}`, "
                f"Relevancy=`{_fmt_score(sc.get('answer_relevancy'))}`, "
                f"Precision=`{_fmt_score(sc.get('context_precision'))}`, "
                f"Recall=`{_fmt_score(sc.get('context_recall'))}`, "
                f"Correctness=`{_fmt_score(sc.get('answer_correctness'))}`"
            )
        else:
            lines.append(f"- **Kiểm tra từ chối an toàn:** `{rec.get('abstention_correct')}`")
        if rec.get("error"):
            lines.append(f"- **Lỗi runtime:** `{rec['error']}`")
        if rec.get("metric_errors"):
            lines.append(f"- **Lỗi chấm metric:** `{rec['metric_errors']}`")
        lines.append("")
        lines.append("**Câu trả lời thực tế của hệ thống:**")
        lines.append("")
        answer_preview = str(rec.get("response") or "(trống)").replace("\r\n", "\n")
        for paragraph_line in answer_preview.splitlines():
            lines.append(f"> {paragraph_line}" if paragraph_line.strip() else ">")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def export_evaluation_artifacts(
    summary: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_RESULTS_DIR,
    report_filename: str = "ragas_evaluation_report.md",
    raw_filename: str = "ragas_evaluation_raw.json",
) -> tuple[Path, Path]:
    """Lưu báo cáo Markdown chi tiết và file JSON thô vào thư mục kết quả."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / report_filename
    raw_path = target_dir / raw_filename

    report_path.write_text(render_markdown_report(summary), encoding="utf-8")
    raw_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path, raw_path
