"""Bộ công cụ đánh giá hệ thống Acne Advisor AI bằng phương pháp RAGAS.

Tham khảo khoa học:
- Es et al. (2023), "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
  (arXiv:2309.15217).
- Tài liệu chính thức RAGAS 0.4.x (https://docs.ragas.io/).

Module cung cấp:
1. Adapter embeddings ``ProjectGeminiEmbeddings`` dùng chung ``google-genai`` của dự án
   để đảm bảo chiều vector đồng nhất (3072 chiều với ``models/gemini-embedding-2``).
2. Hàm chạy trực tiếp ``run_clinical_agent`` trên bộ câu hỏi chuẩn (hỗ trợ cả câu đơn lượt
   và hội thoại đa lượt có ``conversation_history``) kèm lưu checkpoint từng ca.
3. Hàm chấm 5 chỉ số RAGAS cốt lõi (Faithfulness, AnswerRelevancy, ContextPrecision,
   ContextRecall, AnswerCorrectness) kết hợp tỷ lệ từ chối an toàn (Safe Abstention Accuracy),
   tự động đóng kết nối HTTP bất đồng bộ sau khi chấm xong.
4. Hàm xuất báo cáo chi tiết theo cả nhóm chủ đề y khoa và dạng hội thoại ra Markdown/JSON.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

from ragas.embeddings.base import BaseRagasEmbedding

DEFAULT_DATASET_PATH = Path("evaluation/dataset.json")
DEFAULT_RESULTS_DIR = Path("evaluation/results")
DEFAULT_JUDGE_PROVIDER = "nvidia"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_JUDGE_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
DEFAULT_FALLBACK_JUDGE_MODEL = "mistralai/mistral-large-4-1t-a52b"
DEFAULT_FALLBACK_JUDGE_MODELS = (
    "mistralai/mistral-large-4-1t-a52b",
    "nvidia/nemotron-3-ultra-550b-a55b",
)
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSIONS = 3072
DEFAULT_JUDGE_EMBEDDING_MODEL = "nvidia/nemotron-3.5-embed-8b"
DEFAULT_JUDGE_EMBEDDING_DIMENSIONS = 4096
DEFAULT_JUDGE_MAX_TOKENS = 8192

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

VALID_DIALOGUE_TYPES = {
    "single_turn",
    "colloquial_patient",
    "brand_bridge",
    "multi_turn",
    "safety_abstention",
}

INFRA_FAILURE_REASONS = {
    "provider_unavailable",
    "retrieval_unavailable",
    "generation_unavailable",
}


class ProjectGeminiEmbeddings(BaseRagasEmbedding):
    """Adapter embeddings cho RAGAS tái sử dụng ``google-genai`` SDK của dự án.

    Giúp tránh lệch số chiều vector giữa single-text và batch-text embedding khi chấm
    chỉ số ``AnswerRelevancy`` và ``AnswerCorrectness`` bằng ``models/gemini-embedding-2`` (3072 chiều).
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
        vectors = await self.aembed_texts([text])
        return vectors[0]

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
            request_timeout=30.0,
        )

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        del kwargs
        if not texts:
            return []
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self.embed_texts, texts),
                    timeout=35.0,
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= 2 or _is_daily_quota_error(exc):
                    raise
                await asyncio.sleep(3.0 * (attempt + 1))
        if last_exc is not None:
            raise last_exc
        return []


class ProjectNvidiaEmbeddings(BaseRagasEmbedding):
    """Adapter embeddings đa ngôn ngữ độc lập trên NVIDIA NIM (4096 chiều) cho bộ chấm RAGAS.

    Tách biệt hoàn toàn tầng chấm tương đồng ngữ nghĩa của RAGAS (AnswerRelevancy,
    AnswerCorrectness) khỏi hạn mức Google Embedding của tầng truy xuất hệ thống.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_JUDGE_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_JUDGE_EMBEDDING_DIMENSIONS,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.dimensions = dimensions
        self._api_key = (api_key or os.getenv("NVIDIA_API_KEY", "")).strip()
        self._base_url = (
            base_url or os.getenv("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL) or DEFAULT_NVIDIA_BASE_URL
        ).strip()
        self._async_client: Any = None

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            from openai import AsyncOpenAI

            if not self._api_key:
                raise RuntimeError("Thiếu NVIDIA_API_KEY để chạy ProjectNvidiaEmbeddings.")
            self._async_client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=35.0,
                max_retries=1,
            )
        return self._async_client

    async def aclose(self) -> None:
        if self._async_client is not None:
            try:
                await self._async_client.close()
            except Exception:
                pass
            self._async_client = None

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        del kwargs
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        del kwargs
        if not texts:
            return []
        from openai import OpenAI

        if not self._api_key:
            raise RuntimeError("Thiếu NVIDIA_API_KEY để chạy ProjectNvidiaEmbeddings.")
        with OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=35.0, max_retries=1) as client:
            resp = client.embeddings.create(
                model=self.model,
                input=list(texts),
                extra_body={"input_type": "query", "truncate": "END"},
            )
            return [list(item.embedding) for item in resp.data]

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        del kwargs
        vectors = await self.aembed_texts([text])
        return vectors[0]

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        del kwargs
        if not texts:
            return []
        client = self._get_async_client()
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = await asyncio.wait_for(
                    client.embeddings.create(
                        model=self.model,
                        input=list(texts),
                        extra_body={"input_type": "query", "truncate": "END"},
                    ),
                    timeout=40.0,
                )
                return [list(item.embedding) for item in resp.data]
            except Exception as exc:
                last_exc = exc
                if attempt >= 2:
                    raise
                await asyncio.sleep(2.5 * (attempt + 1))
        if last_exc is not None:
            raise last_exc
        return []


@dataclass(frozen=True)
class EvaluationCase:
    """Một mẫu câu hỏi chuẩn trong bộ đánh giá RAGAS (hỗ trợ cả đơn lượt và đa lượt)."""

    case_id: str
    category: str
    expected_behavior: str
    user_input: str
    reference: str
    dialogue_type: str = "single_turn"
    conversation_history: tuple[dict[str, str], ...] = ()
    standalone_question: str | None = None


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
        dialogue_type = str(item.get("dialogue_type") or "single_turn").strip()
        standalone_raw = str(item.get("standalone_question") or "").strip() or None

        if not case_id or not user_input or not reference:
            raise ValueError(f"Thiếu trường bắt buộc trong mẫu đánh giá: {item}")
        if expected not in {"answer", "abstain"}:
            raise ValueError(f"expected_behavior không hợp lệ ({expected}) ở ca {case_id}")
        if dialogue_type not in VALID_DIALOGUE_TYPES:
            raise ValueError(f"dialogue_type không hợp lệ ({dialogue_type}) ở ca {case_id}")
        if case_id in seen_ids:
            raise ValueError(f"Trùng case_id: {case_id}")

        raw_history = item.get("conversation_history") or []
        if not isinstance(raw_history, list):
            raise ValueError(f"conversation_history phải là danh sách ở ca {case_id}")
        normalized_history: list[dict[str, str]] = []
        for turn_item in raw_history:
            if not isinstance(turn_item, dict):
                raise ValueError(f"Phần tử conversation_history không hợp lệ ở ca {case_id}")
            role = str(turn_item.get("role") or "").strip()
            content = str(turn_item.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                raise ValueError(f"Lượt hội thoại thiếu role/content hợp lệ ở ca {case_id}")
            normalized_history.append({"role": role, "content": content})

        if dialogue_type == "multi_turn" and not normalized_history:
            raise ValueError(f"Ca multi_turn {case_id} bắt buộc phải có conversation_history.")
        if normalized_history and expected == "answer" and not standalone_raw:
            raise ValueError(
                f"Ca hội thoại đa lượt {case_id} cần có standalone_question đã giải chiếu đại từ."
            )

        seen_ids.add(case_id)
        cases.append(
            EvaluationCase(
                case_id=case_id,
                category=category,
                expected_behavior=expected,
                user_input=user_input,
                reference=reference,
                dialogue_type=dialogue_type,
                conversation_history=tuple(normalized_history),
                standalone_question=standalone_raw,
            )
        )
    return raw, cases


def _extract_retrieved_contexts(agent_result: dict[str, Any]) -> list[str]:
    """Trích xuất danh sách đoạn chứng cứ thực tế (kèm header nguồn/mục/thực thể) mà LLM đã nhận."""

    from src.retrieval.context_packer import _render_header

    contexts: list[str] = []
    packed = agent_result.get("packed_context")
    if isinstance(packed, dict):
        for idx, item in enumerate(packed.get("items") or [], 1):
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
                if text:
                    raw_payload = item.get("payload")
                    payload = raw_payload if isinstance(raw_payload, dict) and raw_payload else item
                    chunk_id = str(payload.get("chunk_id") or item.get("item_id") or f"chunk-{idx}")
                    header = _render_header(payload, chunk_id, idx)
                    contexts.append(f"{header}\n{text}")
    if contexts:
        return contexts
    for idx, item in enumerate(agent_result.get("vector_contexts") or [], 1):
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or "").strip()
            if text:
                chunk_id = str(item.get("chunk_id") or item.get("id") or f"chunk-{idx}")
                header = _render_header(item, chunk_id, idx)
                contexts.append(f"{header}\n{text}")
    return contexts


async def _warmup_reranker() -> None:
    """Tải trước model reranker lên GPU/CPU một lần để tránh cold-start timeout ở câu đầu."""

    from src.retrieval.service import warm_process_reranker

    try:
        await warm_process_reranker()
    except Exception:
        return


def _case_system_signature(
    case: EvaluationCase,
    llm_provider: str | None,
    llm_model: str | None,
    allow_model_fallback: bool = False,
    pipeline_fingerprint: str | None = None,
) -> str:
    # Chữ ký checkpoint phải gồm cả pipeline fingerprint (GraphRAG, reranker, context budget, KB
    # build...) để khi đổi cấu hình (vd ablation tắt GraphRAG) runner không lặng lẽ dùng lại
    # câu trả lời đã sinh bởi một cấu hình khác.
    payload = {
        "case_id": case.case_id,
        "pipeline_fingerprint": pipeline_fingerprint or "unknown",
        "user_input": case.user_input,
        "conversation_history": list(case.conversation_history),
        "llm_provider": llm_provider or "default",
        "llm_model": llm_model or "default",
        "allow_model_fallback": bool(allow_model_fallback),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


async def run_system_on_dataset(
    cases: list[EvaluationCase],
    *,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    allow_model_fallback: bool = False,
    verbose: bool = False,
    max_provider_retries: int = 2,
    provider_retry_delay_seconds: float = 12.0,
    case_delay_seconds: float = 0.8,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Chạy trực tiếp hệ thống ``run_clinical_agent`` trên từng ca đánh giá.

    Tắt đọc cache (``bypass_cache=True``) để mỗi lần đánh giá phản ánh đúng chất lượng
    truy hồi và sinh câu trả lời hiện hành của hệ thống. Hỗ trợ lưu checkpoint từng ca.
    """

    from src.agent.graph import run_clinical_agent

    from src.observability.versioning import current_pipeline_fingerprint

    await _warmup_reranker()
    pipeline_fingerprint = current_pipeline_fingerprint()

    cached_by_sig: dict[str, dict[str, Any]] = {}
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        try:
            loaded = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                for item in loaded:
                    sig = str(item.get("system_signature") or "")
                    if (
                        sig
                        and not item.get("error")
                        and item.get("fallback_reason_code") not in INFRA_FAILURE_REASONS
                    ):
                        cached_by_sig[sig] = item
        except Exception:
            cached_by_sig = {}

    records: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        sig = _case_system_signature(
            case,
            llm_provider,
            llm_model,
            allow_model_fallback,
            pipeline_fingerprint=pipeline_fingerprint,
        )
        if sig in cached_by_sig:
            restored = dict(cached_by_sig[sig])
            restored["index"] = idx
            restored["category"] = case.category
            restored["expected_behavior"] = case.expected_behavior
            restored["dialogue_type"] = case.dialogue_type
            restored["reference"] = case.reference
            restored["standalone_question"] = case.standalone_question
            records.append(restored)
            if verbose:
                print(
                    f"[System {idx}/{len(cases)}] {case.case_id} (resumed) "
                    f"| action={restored.get('action') or '-'} "
                    f"| contexts={len(restored.get('retrieved_contexts') or [])} "
                    f"| {float(restored.get('latency_ms') or 0):.0f}ms",
                    flush=True,
                )
            continue

        started = time.perf_counter()
        error_text: str | None = None
        result: dict[str, Any] = {}
        history_payload = [dict(m) for m in case.conversation_history]

        for attempt in range(max_provider_retries + 1):
            error_text = None
            result = {}
            try:
                result = await run_clinical_agent(
                    message=case.user_input,
                    conversation_history=history_payload,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    allow_model_fallback=allow_model_fallback,
                    bypass_cache=True,
                    include_generation_diagnostics=True,
                    session_id=f"ragas-eval-{case.case_id.lower()}",
                )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
            daily_quota_detail = _extract_daily_quota_detail(result, error_text)
            if daily_quota_detail is not None and (not allow_model_fallback or attempt >= max_provider_retries):
                raise RuntimeError(
                    f"System LLM ({llm_provider or 'default'}/{llm_model or 'default'}) đã cạn hạn mức theo ngày (RPD) "
                    f"tại ca {case.case_id}: {daily_quota_detail}. "
                    "Hãy chờ reset hạn mức ngày hoặc đổi model/API key trước khi chạy lại bộ đánh giá."
                )
            transient_fallback = result.get("fallback_reason_code") in {
                "provider_unavailable",
                "generation_unavailable",
                "retrieval_unavailable",
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
        safety_dec = result.get("safety_decision") or {}
        token_usage = result.get("generation_token_usage") or {}
        record = {
            "index": idx,
            "case_id": case.case_id,
            "category": case.category,
            "dialogue_type": case.dialogue_type,
            "expected_behavior": case.expected_behavior,
            "user_input": case.user_input,
            "conversation_history": history_payload,
            "standalone_question": case.standalone_question,
            "retrieval_query": str(result.get("retrieval_query") or "").strip() or None,
            "reference": case.reference,
            "response": str(result.get("answer") or result.get("final_answer") or "").strip(),
            "retrieved_contexts": _extract_retrieved_contexts(result),
            "sources": list(result.get("sources") or []),
            "action": str(decision.get("action") or ""),
            "reason_code": str(decision.get("reason_code") or ""),
            "fallback_applied": bool(result.get("fallback_applied")),
            "fallback_reason_code": str(result.get("fallback_reason_code") or "") or None,
            "safety_override": bool(safety_dec),
            "safety_rule_id": str(safety_dec.get("rule_id") or "") or None,
            "actual_provider": result.get("actual_provider"),
            "actual_model": result.get("actual_model"),
            "requested_model": result.get("requested_model") or llm_model,
            "llm_fallback_used": bool(result.get("llm_fallback_used")),
            "latency_ms": elapsed_ms,
            "token_usage": dict(token_usage) if isinstance(token_usage, dict) else {},
            "error": error_text,
            "pipeline_fingerprint": pipeline_fingerprint,
            "system_signature": sig,
        }
        records.append(record)
        if checkpoint_path is not None:
            try:
                target_cp = Path(checkpoint_path)
                target_cp.parent.mkdir(parents=True, exist_ok=True)
                target_cp.write_text(
                    json.dumps(records, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
        if verbose:
            action_display = (
                f"safety:{record['safety_rule_id']}"
                if record["safety_override"] and record["safety_rule_id"]
                else (record["action"] or "-")
            )
            print(
                f"[System {idx}/{len(cases)}] {case.case_id} ({case.dialogue_type}) "
                f"| action={action_display} "
                f"| contexts={len(record['retrieved_contexts'])} | {elapsed_ms:.0f}ms"
                + (f" | ERROR={error_text}" if error_text else ""),
                flush=True,
            )
        if case_delay_seconds > 0 and idx < len(cases):
            await asyncio.sleep(case_delay_seconds)
    return records


def build_ragas_evaluators(
    *,
    judge_model: str | None = None,
    judge_provider: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
    judge_api_key: str | None = None,
    judge_base_url: str | None = None,
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    shared_embeddings: ProjectGeminiEmbeddings | None = None,
) -> tuple[Any, ProjectGeminiEmbeddings]:
    """Khởi tạo Judge LLM (NVIDIA NIM, OpenAI hoặc Gemini) và Embeddings cho RAGAS 0.4.x."""

    from ragas.llms import llm_factory

    effective_provider = (
        judge_provider
        or os.getenv("RAGAS_JUDGE_PROVIDER", DEFAULT_JUDGE_PROVIDER)
        or DEFAULT_JUDGE_PROVIDER
    ).strip().lower()
    effective_model = (
        judge_model
        or os.getenv("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        or DEFAULT_JUDGE_MODEL
    ).strip()

    use_nvidia_embeddings = str(embedding_model or "").strip().lower().startswith("nvidia/")
    google_key = (api_key or os.getenv("GOOGLE_API_KEY", "")).strip()
    if not google_key and (effective_provider == "google" or (not use_nvidia_embeddings and shared_embeddings is None)):
        raise RuntimeError("Thiếu GOOGLE_API_KEY trong biến môi trường cho embeddings.")

    if effective_provider == "nvidia":
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
        # Dùng MD_JSON và cấp đủ 8192 output tokens để câu trả lời tiếng Việt dài không bị cắt cụt;
        # chỉ truyền chat_template_kwargs.enable_thinking=False cho họ mô hình hỗ trợ vLLM template này.
        factory_kwargs: dict[str, Any] = {
            "provider": "openai",
            "client": nvidia_client,
            "mode": instructor.Mode.MD_JSON,
            "max_tokens": judge_max_tokens,
        }
        if any(family in effective_model.lower() for family in ("nemotron", "qwen", "deepseek")):
            factory_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        judge_llm = llm_factory(effective_model, **factory_kwargs)
    elif effective_provider == "openai":
        from openai import AsyncOpenAI

        openai_key = (judge_api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        if not openai_key:
            raise RuntimeError("Thiếu OPENAI_API_KEY để chạy RAGAS judge với provider='openai'.")
        openai_kwargs: dict[str, Any] = {"api_key": openai_key, "timeout": 60.0, "max_retries": 1}
        if judge_base_url:
            openai_kwargs["base_url"] = judge_base_url.strip()
        openai_client = AsyncOpenAI(**openai_kwargs)
        judge_llm = llm_factory(
            effective_model,
            provider="openai",
            client=openai_client,
            max_tokens=judge_max_tokens,
        )
    else:
        from google import genai

        google_client = genai.Client(api_key=google_key)
        judge_llm = llm_factory(effective_model, provider="google", client=google_client)

    if shared_embeddings is not None:
        embeddings: BaseRagasEmbedding = shared_embeddings
    elif use_nvidia_embeddings:
        embeddings = ProjectNvidiaEmbeddings(
            model=embedding_model,
            dimensions=DEFAULT_JUDGE_EMBEDDING_DIMENSIONS,
            api_key=judge_api_key,
            base_url=judge_base_url,
        )
    else:
        embeddings = ProjectGeminiEmbeddings(
            model=embedding_model,
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            api_key=google_key,
        )
    return judge_llm, embeddings


async def _close_judge_llm_client(judge_llm: Any) -> None:
    """Đóng kết nối HTTP bất đồng bộ bên trong InstructorLLM/AsyncOpenAI để tránh rò rỉ socket."""

    instructor_client = getattr(judge_llm, "client", None)
    for target in (getattr(instructor_client, "client", None), instructor_client):
        close_fn = getattr(target, "close", None)
        if callable(close_fn):
            try:
                maybe_coro = close_fn()
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro
                return
            except Exception:
                pass


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
    if isinstance(exc, TimeoutError):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "resource_exhausted",
            "rate limit",
            "unavailable",
            "connection error",
            "connecterror",
            "timeout",
            "timed out",
            "timeouterror",
            "retryexhaustederror",
            "retry exhausted",
            "instructorretryexception",
            "validationerror",
            "jsondecodeerror",
        )
    )


async def _run_metric_with_retry(
    factory: Any,
    *,
    max_retries: int = 1,
    base_backoff: float = 6.0,
    metric_timeout_seconds: float = 180.0,
) -> Any:
    """Chạy 1 metric RAGAS kèm giới hạn timeout cứng và lùi thời gian khi gặp lỗi tạm thời."""

    for attempt in range(max_retries + 1):
        try:
            return await asyncio.wait_for(factory(), timeout=metric_timeout_seconds)
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


def _score_checkpoint_signature(rec: dict[str, Any], judge_model: str) -> str:
    contexts_list = [str(c) for c in (rec.get("retrieved_contexts") or [])]
    payload = {
        "case_id": rec.get("case_id"),
        "eval_question": rec.get("standalone_question") or rec.get("user_input"),
        "reference": rec.get("reference"),
        "response": rec.get("response"),
        "contexts_sha": hashlib.sha256("\n||\n".join(contexts_list).encode("utf-8")).hexdigest()[:16],
        "judge_model": judge_model,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _evaluate_safe_abstention(rec: dict[str, Any]) -> bool:
    """Kiểm tra ca từ chối an toàn, loại trừ lỗi sập hạ tầng/provider."""

    if rec.get("error"):
        return False
    if rec.get("fallback_reason_code") in INFRA_FAILURE_REASONS:
        return False
    if rec.get("reason_code") == "provider_unavailable":
        return False
    return (
        bool(rec.get("safety_override"))
        or rec.get("action") == "abstain"
        or bool(rec.get("fallback_applied"))
        or rec.get("actual_provider") == "system"
    )


async def score_records_with_ragas(
    records: list[dict[str, Any]],
    *,
    judge_model: str | None = None,
    judge_provider: str | None = None,
    fallback_judge_model: str | None = DEFAULT_FALLBACK_JUDGE_MODEL,
    fallback_judge_models: tuple[str, ...] | list[str] | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
    judge_api_key: str | None = None,
    judge_base_url: str | None = None,
    request_delay_seconds: float | None = None,
    max_rate_limit_retries: int = 1,
    metric_timeout_seconds: float = 180.0,
    metric_concurrency: int | None = None,
    metric_keys: tuple[str, ...] | list[str] | None = None,
    checkpoint_path: Path | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Chấm 5 chỉ số RAGAS trên các ca cần trả lời và kiểm tra tỷ lệ từ chối an toàn.

    ``metric_keys`` cho phép chỉ chấm một tập con chỉ số (vd chỉ ``faithfulness`` cho bài kiểm
    tra cài lỗi có chủ đích) để tiết kiệm quota judge; chỉ số không được chọn giữ giá trị ``None``.
    """

    # Giữ đúng thứ tự chuẩn của RAGAS_METRIC_KEYS; từ chối tên chỉ số không tồn tại để tránh
    # chạy nhầm một lần chấm "rỗng" mà vẫn tưởng là đã đánh giá.
    requested_keys = set(metric_keys) if metric_keys is not None else set(RAGAS_METRIC_KEYS)
    unknown_keys = requested_keys - set(RAGAS_METRIC_KEYS)
    if unknown_keys or not requested_keys:
        raise ValueError(f"metric_keys không hợp lệ: {sorted(unknown_keys) or 'rỗng'}")
    active_keys = tuple(key for key in RAGAS_METRIC_KEYS if key in requested_keys)

    from ragas.metrics.collections import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    provider_norm = (
        judge_provider
        or os.getenv("RAGAS_JUDGE_PROVIDER", DEFAULT_JUDGE_PROVIDER)
        or DEFAULT_JUDGE_PROVIDER
    ).strip().lower()
    effective_judge_model = (
        judge_model
        or os.getenv("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        or DEFAULT_JUDGE_MODEL
    ).strip()

    created_llms: list[Any] = []
    judge_llm, embeddings = build_ragas_evaluators(
        judge_model=effective_judge_model,
        judge_provider=provider_norm,
        embedding_model=embedding_model,
        api_key=api_key,
        judge_api_key=judge_api_key,
        judge_base_url=judge_base_url,
    )
    created_llms.append(judge_llm)

    def _build_metric_set(llm_instance: Any) -> dict[str, Any]:
        rel_metric = AnswerRelevancy(llm=llm_instance, embeddings=embeddings)
        # Bổ sung quy tắc giữ đúng ngôn ngữ gốc của câu trả lời để tránh mất điểm tương đồng chéo Việt-Anh.
        prompt_obj = getattr(rel_metric, "prompt", None)
        instruction = getattr(prompt_obj, "instruction", None)
        lang_rule = "Always generate the question in the exact same language as the given response (for Vietnamese responses, generate a natural Vietnamese question)."
        if isinstance(instruction, str) and lang_rule not in instruction:
            try:
                prompt_obj.instruction = f"{instruction.rstrip()} {lang_rule}"
            except Exception:
                pass
        return {
            "faithfulness": Faithfulness(llm=llm_instance),
            "answer_relevancy": rel_metric,
            "context_precision": ContextPrecision(llm=llm_instance),
            "context_recall": ContextRecall(llm=llm_instance),
            "answer_correctness": AnswerCorrectness(
                llm=llm_instance,
                embeddings=embeddings,
                weights=[0.75, 0.25],
            ),
        }

    try:
        # Xây chuỗi judge ưu tiên: judge chính + các model dự phòng cùng provider (tự động loại trùng).
        judge_chain: list[tuple[str, dict[str, Any]]] = [
            (effective_judge_model, _build_metric_set(judge_llm))
        ]
        if provider_norm in {"google", "nvidia"}:
            raw_fallbacks: list[str] = []
            env_fallbacks = os.getenv("RAGAS_FALLBACK_JUDGE_MODELS", "").strip()
            if fallback_judge_models is not None:
                raw_fallbacks.extend(str(m).strip() for m in fallback_judge_models if str(m).strip())
            elif env_fallbacks:
                raw_fallbacks.extend(m.strip() for m in env_fallbacks.split(",") if m.strip())
            elif fallback_judge_model:
                raw_fallbacks.append(str(fallback_judge_model).strip())
            seen_models = {effective_judge_model}
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
                    shared_embeddings=embeddings,
                )
                created_llms.append(fb_llm)
                judge_chain.append((fb_model, _build_metric_set(fb_llm)))

        exhausted_daily_models: set[str] = set()
        degraded_timeout_models: set[str] = set()
        consecutive_transient_failures: dict[str, int] = {m: 0 for m, _ in judge_chain}

        effective_concurrency = (
            max(1, int(metric_concurrency))
            if metric_concurrency is not None
            else (5 if provider_norm == "openai" else 3 if provider_norm == "nvidia" else 1)
        )
        pacing_delay = (
            0.0
            if request_delay_seconds is None and provider_norm == "openai"
            else 0.25
            if request_delay_seconds is None and provider_norm == "nvidia"
            else 2.5
            if request_delay_seconds is None
            else max(0.0, request_delay_seconds)
        )
        if provider_norm == "nvidia" and effective_concurrency > 1 and pacing_delay > 0.5:
            pacing_delay = 0.25

        env_cp = os.getenv("RAGAS_SCORED_CHECKPOINT_PATH", "").strip()
        resolved_checkpoint = (
            Path(checkpoint_path) if checkpoint_path is not None else (Path(env_cp) if env_cp else None)
        )
        cached_by_sig: dict[str, dict[str, Any]] = {}
        if resolved_checkpoint is not None and resolved_checkpoint.is_file():
            try:
                loaded_cp = json.loads(resolved_checkpoint.read_text(encoding="utf-8"))
                if isinstance(loaded_cp, list):
                    for item in loaded_cp:
                        sig = str(item.get("score_signature") or "")
                        sc_map = item.get("ragas_scores") or {}
                        errs = item.get("metric_errors") or {}
                        if sig and not errs and all(isinstance(sc_map.get(k), (int, float)) for k in active_keys):
                            cached_by_sig[sig] = item
            except Exception:
                cached_by_sig = {}

        short_labels = {
            "faithfulness": "faith",
            "answer_relevancy": "rel",
            "context_precision": "prec",
            "context_recall": "rec",
            "answer_correctness": "corr",
        }

        scored_records: list[dict[str, Any]] = []
        for rec in records:
            cid = str(rec.get("case_id") or "")
            enriched = dict(rec)
            scores: dict[str, float | None] = {key: None for key in RAGAS_METRIC_KEYS}
            metric_errors: dict[str, str] = {}
            # Ghi lại chính xác judge model nào đã trả điểm cho từng metric. Chuỗi dự phòng
            # và circuit-breaker có thể đổi judge giữa chừng; thiếu thông tin này thì không
            # thể biết một lần chạy có bị trộn nhiều "giám khảo" khác nhau hay không.
            metric_judge_models: dict[str, str] = {}

            if rec["expected_behavior"] == "abstain":
                enriched["abstention_correct"] = _evaluate_safe_abstention(rec)
                enriched["unexpected_abstention"] = False
                enriched["ragas_scores"] = scores
                enriched["metric_errors"] = metric_errors
                enriched["metric_judge_models"] = metric_judge_models
                scored_records.append(enriched)
                continue

            score_sig = _score_checkpoint_signature(rec, effective_judge_model)
            cached_item = cached_by_sig.get(score_sig)
            if cached_item is not None:
                enriched["abstention_correct"] = None
                enriched["unexpected_abstention"] = bool(cached_item.get("unexpected_abstention"))
                enriched["ragas_scores"] = dict(cached_item["ragas_scores"])
                enriched["metric_errors"] = {}
                # Checkpoint cũ (trước khi có trường này) không biết judge thực tế -> để rỗng,
                # tuyệt đối không tự gán judge chính để tránh ghi nhận sai nguồn điểm.
                enriched["metric_judge_models"] = dict(cached_item.get("metric_judge_models") or {})
                enriched["score_signature"] = score_sig
                scored_records.append(enriched)
                if verbose:
                    score_str = ", ".join(
                        f"{short_labels[k]}={_fmt_score(enriched['ragas_scores'].get(k))}"
                        for k in RAGAS_METRIC_KEYS
                    )
                    print(
                        f"[RAGAS {len(scored_records)}/{len(records)}] {cid} (resumed) | {score_str}",
                        flush=True,
                    )
                continue

            response_text = _strip_presentation_disclaimer(str(rec.get("response") or ""))
            contexts = list(rec.get("retrieved_contexts") or [])
            # Mọi ca expected="answer" bị từ chối, vướng safety nhầm, rỗng câu trả lời hoặc lỗi runtime
            # đều tính là không hoàn thành trả lời (0.0 cho các metric sinh câu trả lời thay vì bỏ qua mẫu số).
            unexpected_abstain = (
                bool(rec.get("error"))
                or bool(rec.get("safety_override"))
                or rec.get("action") == "abstain"
                or bool(rec.get("fallback_applied"))
                or not response_text
            )
            enriched["abstention_correct"] = None
            enriched["unexpected_abstention"] = unexpected_abstain
            # Với ca hội thoại đa lượt, dùng câu hỏi độc lập đã giải chiếu đại từ để các metric
            # RAGAS (đặc biệt AnswerRelevancy & ContextPrecision) đối chiếu ngữ nghĩa chính xác.
            eval_question = str(
                rec.get("standalone_question")
                or (rec.get("retrieval_query") if rec.get("conversation_history") else None)
                or rec.get("user_input")
                or ""
            ).strip()
            reference = str(rec.get("reference") or "")

            def _make_call(mset: dict[str, Any], key: str) -> Any:
                if key == "faithfulness":
                    return mset[key].ascore(
                        user_input=eval_question,
                        response=response_text,
                        retrieved_contexts=contexts,
                    )
                if key == "answer_relevancy":
                    return mset[key].ascore(user_input=eval_question, response=response_text)
                if key == "context_precision":
                    return mset[key].ascore(
                        user_input=eval_question,
                        reference=reference,
                        retrieved_contexts=contexts,
                    )
                if key == "context_recall":
                    return mset[key].ascore(
                        user_input=eval_question,
                        retrieved_contexts=contexts,
                        reference=reference,
                    )
                return mset[key].ascore(
                    user_input=eval_question,
                    response=response_text,
                    reference=reference,
                )

            sem = asyncio.Semaphore(effective_concurrency)

            async def _score_one_metric(key: str) -> Any:
                async with sem:
                    last_exc: Exception | None = None
                    # Ưu tiên các model đang phản hồi tốt lên trước; đẩy các model bị nghẽn liên tiếp xuống cuối chuỗi dự phòng.
                    ordered_chain = [
                        item for item in judge_chain if item[0] not in degraded_timeout_models
                    ] + [
                        item for item in judge_chain if item[0] in degraded_timeout_models
                    ]
                    for idx_j, (model_name, mset) in enumerate(ordered_chain):
                        if model_name in exhausted_daily_models:
                            continue
                        try:
                            res = await _run_metric_with_retry(
                                lambda ms=mset, k=key: _make_call(ms, k),
                                max_retries=max_rate_limit_retries if idx_j == 0 else 1,
                                metric_timeout_seconds=metric_timeout_seconds,
                            )
                            consecutive_transient_failures[model_name] = 0
                            degraded_timeout_models.discard(model_name)
                            metric_judge_models[key] = model_name
                            if pacing_delay > 0:
                                await asyncio.sleep(pacing_delay)
                            return res
                        except Exception as exc:
                            last_exc = exc
                            exc_lower = f"{type(exc).__name__}: {exc}".lower()
                            is_foreign_embed_quota = (
                                provider_norm != "google" and "google genai" in exc_lower
                            )
                            if _is_daily_quota_error(exc) and not is_foreign_embed_quota:
                                exhausted_daily_models.add(model_name)
                                if verbose:
                                    print(
                                        f"  [Judge] {model_name} hết quota theo ngày -> chuyển sang model dự phòng.",
                                        flush=True,
                                    )
                                continue
                            if _is_rate_limit_error(exc) and not is_foreign_embed_quota:
                                consecutive_transient_failures[model_name] = (
                                    consecutive_transient_failures.get(model_name, 0) + 1
                                )
                                if (
                                    consecutive_transient_failures[model_name] >= 3
                                    and len(judge_chain) > 1
                                    and model_name not in degraded_timeout_models
                                ):
                                    degraded_timeout_models.add(model_name)
                                    if verbose:
                                        print(
                                            f"  [Judge] {model_name} timeout liên tiếp -> tự động ưu tiên model dự phòng phản hồi nhanh.",
                                            flush=True,
                                        )
                                continue
                            raise
                    if last_exc is not None:
                        raise last_exc
                    raise RuntimeError("Tất cả judge model đều đã cạn quota theo ngày.")

            if unexpected_abstain:
                # Không chấm boilerplate từ chối bằng LLM cho các chỉ số sinh câu trả lời (gán 0.0 trung thực);
                # nếu tầng truy hồi vẫn lấy được ngữ cảnh hợp lệ thì vẫn chấm ContextPrecision & ContextRecall.
                for key in active_keys:
                    scores[key] = 0.0
                retrieval_keys = tuple(
                    key for key in ("context_precision", "context_recall") if key in active_keys
                )
                if contexts and not rec.get("error") and retrieval_keys:
                    ret_results = await asyncio.gather(
                        *(_score_one_metric(k) for k in retrieval_keys),
                        return_exceptions=True,
                    )
                    for k, outcome in zip(retrieval_keys, ret_results, strict=True):
                        if isinstance(outcome, BaseException):
                            metric_errors[k] = f"{type(outcome).__name__}: {outcome}"
                        else:
                            scores[k] = _clean_score(getattr(outcome, "value", outcome))
            elif response_text and contexts:
                if effective_concurrency > 1:
                    results = await asyncio.gather(
                        *(_score_one_metric(key) for key in active_keys),
                        return_exceptions=True,
                    )
                    for key, outcome in zip(active_keys, results, strict=True):
                        if isinstance(outcome, BaseException):
                            metric_errors[key] = f"{type(outcome).__name__}: {outcome}"
                        else:
                            scores[key] = _clean_score(getattr(outcome, "value", outcome))
                else:
                    for key in active_keys:
                        try:
                            outcome = await _score_one_metric(key)
                            scores[key] = _clean_score(getattr(outcome, "value", outcome))
                        except Exception as exc:
                            metric_errors[key] = f"{type(exc).__name__}: {exc}"
            else:
                for key in active_keys:
                    scores[key] = 0.0
                metric_errors["skipped"] = "Không có ngữ cảnh truy hồi hợp lệ cho câu trả lời."

            enriched["ragas_scores"] = scores
            enriched["metric_errors"] = metric_errors
            enriched["metric_judge_models"] = metric_judge_models
            enriched["score_signature"] = score_sig
            scored_records.append(enriched)
            if resolved_checkpoint is not None:
                try:
                    resolved_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    resolved_checkpoint.write_text(
                        json.dumps(scored_records, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
            if verbose:
                score_str = ", ".join(
                    f"{short_labels[k]}={_fmt_score(scores[k])}" for k in RAGAS_METRIC_KEYS
                )
                print(
                    f"[RAGAS {len(scored_records)}/{len(records)}] {cid} | {score_str}"
                    + (" | UNEXPECTED_ABSTAIN" if unexpected_abstain else "")
                    + (f" | errs={list(metric_errors.keys())}" if metric_errors else ""),
                    flush=True,
                )

        return scored_records
    finally:
        for llm_inst in created_llms:
            await _close_judge_llm_client(llm_inst)
        aclose_emb = getattr(embeddings, "aclose", None)
        if callable(aclose_emb):
            await aclose_emb()


def _group_metrics_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    group_metrics: dict[str, float | None] = {}
    for key in RAGAS_METRIC_KEYS:
        vals = [
            float(r["ragas_scores"][key])
            for r in items
            if isinstance(r.get("ragas_scores", {}).get(key), (int, float))
        ]
        group_metrics[key] = round(statistics.mean(vals), 4) if vals else None
    latencies = [
        float(r["latency_ms"]) for r in items if isinstance(r.get("latency_ms"), (int, float))
    ]
    return {
        "count": len(items),
        "metrics": group_metrics,
        "mean_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
    }


def summarize_evaluation(
    scored_records: list[dict[str, Any]],
    *,
    dataset_meta: dict[str, Any] | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    kb_build_id: str | None = None,
) -> dict[str, Any]:
    """Tổng hợp điểm số RAGAS toàn cục, theo nhóm chủ đề, theo dạng hội thoại, độ trễ và token."""

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

    categories: dict[str, list[dict[str, Any]]] = {}
    dialogue_types: dict[str, list[dict[str, Any]]] = {}
    for rec in scored_records:
        categories.setdefault(str(rec.get("category") or "general"), []).append(rec)
        dialogue_types.setdefault(str(rec.get("dialogue_type") or "single_turn"), []).append(rec)

    category_summary = {cat: _group_metrics_summary(items) for cat, items in categories.items()}
    dialogue_type_summary = {
        dtype: _group_metrics_summary(items) for dtype, items in dialogue_types.items()
    }

    abstain_hits = sum(1 for r in abstain_cases if r.get("abstention_correct") is True)
    abstain_accuracy = round(abstain_hits / len(abstain_cases), 4) if abstain_cases else None
    unexpected_abstentions = sum(1 for r in answer_cases if r.get("unexpected_abstention") is True)
    answer_completion_rate = (
        round((len(answer_cases) - unexpected_abstentions) / len(answer_cases), 4)
        if answer_cases
        else None
    )

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

    # Đếm số lượt điểm metric do từng judge model trả về; điểm không rõ nguồn (checkpoint cũ)
    # được tách riêng thành "unknown" để báo cáo không ngầm quy hết cho judge chính.
    judge_model_usage: dict[str, int] = {}
    for rec in answer_cases:
        recorded = rec.get("metric_judge_models") or {}
        for key in RAGAS_METRIC_KEYS:
            if not isinstance(rec.get("ragas_scores", {}).get(key), (int, float)):
                continue
            owner = str(recorded.get(key) or "unknown")
            judge_model_usage[owner] = judge_model_usage.get(owner, 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": (dataset_meta or {}).get("dataset_name", "acne_advisor_ragas_benchmark"),
        "judge_model": judge_model,
        "judge_model_usage": dict(sorted(judge_model_usage.items())),
        "embedding_model": embedding_model,
        "kb_build_id": kb_build_id,
        "counts": {
            "total_cases": len(scored_records),
            "answer_cases": len(answer_cases),
            "abstain_cases": len(abstain_cases),
            "multi_turn_cases": sum(1 for r in scored_records if r.get("conversation_history")),
            "unexpected_abstentions": unexpected_abstentions,
            "answer_completion_rate": answer_completion_rate,
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
        "by_dialogue_type": dialogue_type_summary,
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


def _display_action(rec: dict[str, Any]) -> str:
    if rec.get("safety_override"):
        rule_id = rec.get("safety_rule_id")
        return f"safety:{rule_id}" if rule_id else "safety_override"
    return str(rec.get("action") or "-")


def render_markdown_report(summary: dict[str, Any]) -> str:
    """Kết xuất báo cáo đánh giá RAGAS chi tiết sang định dạng Markdown sạch."""

    counts = summary["counts"]
    latency = summary["latency_ms"]
    tokens = summary["token_usage"]
    abstention = summary["safe_abstention"]
    usage = summary.get("judge_model_usage") or {}
    usage_text = (
        ", ".join(f"`{model}`: {count}" for model, count in usage.items())
        if usage
        else "không có dữ liệu"
    )

    lines: list[str] = [
        "# Báo cáo Đánh giá Hệ thống Acne Advisor AI (RAGAS)",
        "",
        f"- **Thời điểm chạy (UTC):** `{summary['generated_at']}`",
        f"- **Bộ dữ liệu đánh giá:** `{summary['dataset_name']}` ({counts['total_cases']} câu hỏi: {counts['answer_cases']} câu trả lời, {counts['abstain_cases']} câu từ chối an toàn, {counts.get('multi_turn_cases', 0)} ca đa lượt)",
        f"- **Judge LLM (RAGAS):** `{summary['judge_model']}`",
        f"- **Judge thực tế đã chấm (số lượt điểm metric):** {usage_text}",
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
            f"| Tỷ lệ hoàn thành câu trả lời có bằng chứng (Answer Completion Rate) | {counts['answer_cases'] - counts.get('unexpected_abstentions', 0)}/{counts['answer_cases']} ({_fmt_score(counts.get('answer_completion_rate'))}) |",
            f"| Tỷ lệ từ chối an toàn (Safe Abstention Accuracy) | {abstention['correct']}/{abstention['total']} ({_fmt_score(abstention['accuracy'])}) |",
            f"| Độ trễ trung bình (Mean Latency) | {latency['mean']} ms |",
            f"| Độ trễ trung vị (P50 Latency) | {latency['p50']} ms |",
            f"| Độ trễ P95 (P95 Latency) | {latency['p95']} ms |",
            f"| Tổng token tiêu thụ (Generation) | input={tokens['input']}, output={tokens['output']}, total={tokens['total']} |",
            f"| Số lỗi runtime | {counts['runtime_errors']} |",
            "",
            "## 3. Phân tích theo nhóm chủ đề y khoa",
            "",
            "| Nhóm chủ đề | Số ca | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Độ trễ TB (ms) |",
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

    by_dtype = summary.get("by_dialogue_type") or {}
    if by_dtype:
        lines.extend(
            [
                "",
                "## 4. Phân tích theo dạng hội thoại",
                "",
                "| Dạng hội thoại | Số ca | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Độ trễ TB (ms) |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for dtype, info in by_dtype.items():
            dm = info["metrics"]
            lines.append(
                f"| `{dtype}` | {info['count']} | {_fmt_score(dm['faithfulness'])} | "
                f"{_fmt_score(dm['answer_relevancy'])} | {_fmt_score(dm['context_precision'])} | "
                f"{_fmt_score(dm['context_recall'])} | {_fmt_score(dm['answer_correctness'])} | "
                f"{info['mean_latency_ms']} |"
            )

    lines.extend(
        [
            "",
            "## 5. Bảng điểm chi tiết từng câu hỏi",
            "",
            "| Case ID | Nhóm | Dạng hội thoại | Hành động | Faithfulness | Relevancy | Precision | Recall | Correctness | Từ chối đúng | Độ trễ (ms) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
            f"| `{rec['case_id']}` | `{rec['category']}` | `{rec.get('dialogue_type', 'single_turn')}` | `{_display_action(rec)}` | "
            f"{_fmt_score(sc.get('faithfulness'))} | {_fmt_score(sc.get('answer_relevancy'))} | "
            f"{_fmt_score(sc.get('context_precision'))} | {_fmt_score(sc.get('context_recall'))} | "
            f"{_fmt_score(sc.get('answer_correctness'))} | {abstain_col} | {rec.get('latency_ms')} |"
        )

    lines.extend(["", "## 6. Chi tiết câu trả lời & ngữ cảnh từng ca", ""])
    for rec in summary["records"]:
        sc = rec.get("ragas_scores") or {}
        lines.append(
            f"### {rec['case_id']} — `{rec['category']}` (`{rec.get('dialogue_type', 'single_turn')}`)"
        )
        lines.append("")
        if rec.get("conversation_history"):
            lines.append("- **Ngữ cảnh hội thoại trước đó:**")
            for h_item in rec["conversation_history"]:
                lines.append(f"  - `{h_item.get('role')}`: {h_item.get('content')}")
        lines.append(f"- **Câu hỏi hiện tại:** {rec['user_input']}")
        if rec.get("standalone_question"):
            lines.append(f"- **Câu hỏi độc lập (giải chiếu đại từ):** {rec['standalone_question']}")
        lines.append(f"- **Đáp án chuẩn tham chiếu:** {rec['reference']}")
        lines.append(
            f"- **Hành động hệ thống:** `{_display_action(rec)}` (fallback=`{rec.get('fallback_applied')}`, nguồn=`{len(rec.get('sources') or [])}`, độ trễ=`{rec.get('latency_ms')} ms`)"
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
