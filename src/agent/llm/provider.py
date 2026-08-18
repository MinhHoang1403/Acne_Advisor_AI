"""Chọn Gemini/Ollama và thực thi provider fallback trong deadline chung.

Module resolve provider/model, gọi adapter tương ứng, áp dụng retry contract và
ghi nhận chuỗi fallback. Gemini/Ollama mới thực thi inference; Python không sinh
nội dung model. Permanent error dừng ngay, còn fallback chỉ chạy khi caller và
runtime settings cho phép, sử dụng phần ``DeadlineBudget`` còn lại.
"""

import asyncio
import os
import logging
from typing import Optional

from src.agent.llm.ollama_client import generate_ollama_response, list_ollama_models
from src.agent.text_encoding import repair_mojibake
from src.integrations.google_genai import generate_text_async, generate_text_sync
from src.quality.safe_fallback import sanitize_fallback_reason
from src.resilience.budget import DeadlineBudget
from src.resilience.contracts import RuntimeResilienceSettings, runtime_resilience_settings_from_env
from src.resilience.exceptions import (
    PermanentProviderError,
    RuntimeResilienceError,
)
from src.resilience.provider import call_provider_with_resilience
from src.resilience.retry import RetryPolicy

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_FALLBACK_MODELS = ("gemini-3.1-flash-lite",)
DEFAULT_OLLAMA_MODEL = "qwen3:8b"

def _retry_policy(settings: RuntimeResilienceSettings) -> RetryPolicy:
    return RetryPolicy(
        max_retries=settings.llm_max_retries,
        base_delay_seconds=settings.llm_retry_base_delay_seconds,
        max_delay_seconds=settings.llm_retry_max_delay_seconds,
    )


def _resolve_model(provider: str, model: Optional[str]) -> tuple[str, str]:
    provider = (provider or "gemini").lower()
    if provider == "local":
        provider = "ollama"
    if provider == "gemini":
        resolved = model or os.getenv("GOOGLE_MODEL", DEFAULT_GEMINI_MODEL)
        if resolved == "gemini-1.5-flash":
            resolved = DEFAULT_GEMINI_MODEL
        return provider, resolved
    if provider == "ollama":
        configured_model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        resolved = model or configured_model
        if ":" not in resolved:
            resolved = f"{resolved}:latest"
        return provider, resolved
    return provider, model or ""


def parse_google_fallback_models(
    value: str | None = None,
    *,
    primary_model: str | None = None,
) -> list[str]:
    """Parse danh sách Gemini fallback, giữ thứ tự và loại primary/trùng lặp."""

    raw = os.getenv("GOOGLE_FALLBACK_MODELS") if value is None else value
    if raw is None:
        candidates = list(DEFAULT_GEMINI_FALLBACK_MODELS)
    else:
        candidates = [item.strip() for item in raw.split(",")]

    primary = (primary_model or os.getenv("GOOGLE_MODEL", DEFAULT_GEMINI_MODEL)).strip()
    if primary == "gemini-1.5-flash":
        primary = DEFAULT_GEMINI_MODEL

    seen: set[str] = set()
    parsed: list[str] = []
    for candidate in candidates:
        if not candidate or candidate == primary or candidate in seen:
            continue
        seen.add(candidate)
        parsed.append(candidate)
    return parsed


def _provider_model_key(provider: str, model: str) -> str:
    return f"{provider}:{model}"


def _fallback_reason_from_error(exc: BaseException) -> str:
    error_code = getattr(exc, "error_code", "")
    cause = getattr(exc, "__cause__", None)
    text = f"{exc} {cause or ''}".lower()
    if "resource_exhausted" in text or "quota" in text:
        return "quota_exhausted"
    if "429" in text or "rate limit" in text or "rate_limited" in text:
        return "rate_limited"
    if error_code == "provider_timeout" or "timed out" in text or "timeout" in text:
        return "provider_timeout"
    if error_code == "retry_exhausted":
        return "retry_exhausted"
    if error_code == "provider_unavailable" or "503" in text or "unavailable" in text:
        return "provider_unavailable"
    return error_code or "provider_unavailable"


def _safe_error_text(exc: BaseException) -> str:
    return sanitize_fallback_reason(exc)


def _attach_fallback_trace(
    exc: BaseException,
    *,
    requested_provider: str,
    requested_model: str,
    fallback_chain: list[dict[str, object]],
) -> BaseException:
    """Giữ trace provider đã sanitize khi caller phải xử lý failure cuối cùng."""

    exc.requested_provider = requested_provider  # type: ignore[attr-defined]
    exc.requested_model = requested_model  # type: ignore[attr-defined]
    exc.fallback_chain = [dict(entry) for entry in fallback_chain]  # type: ignore[attr-defined]
    return exc


async def _call_gemini(
    prompt: str,
    system_prompt: Optional[str],
    model_name: str,
    temperature: float,
    request_timeout: float | None = None,
) -> str:
    """Gọi Gemini qua Google GenAI adapter bất đồng bộ."""
    return await generate_text_async(
        prompt=prompt,
        system_prompt=system_prompt,
        model_name=model_name,
        temperature=temperature,
        request_timeout=request_timeout,
    )

async def _call_gemini_sync(
    prompt: str,
    system_prompt: Optional[str],
    model_name: str,
    temperature: float,
    request_timeout: float | None = None,
) -> str:
    """Đưa lời gọi Google GenAI đồng bộ sang worker thread có timeout."""
    def _generate_sync() -> str:
        return generate_text_sync(
            prompt=prompt,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
        )

    if request_timeout and request_timeout > 0:
        async with asyncio.timeout(request_timeout):
            return await asyncio.to_thread(_generate_sync)
    return await asyncio.to_thread(_generate_sync)


async def _call_provider_once(
    *,
    provider: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    use_sync: bool,
    request_timeout: float,
) -> str:
    if provider == "gemini":
        logger.info("Calling Gemini (%s)...", model)
        if use_sync:
            return repair_mojibake(
                await _call_gemini_sync(prompt, system_prompt, model, temperature, request_timeout)
            )
        return repair_mojibake(
            await _call_gemini(prompt, system_prompt, model, temperature, request_timeout)
        )

    if provider == "ollama":
        logger.info("Calling Ollama (%s)...", model)
        return repair_mojibake(
            await generate_ollama_response(
                model,
                system_prompt,
                prompt,
                temperature,
                request_timeout=request_timeout,
            )
        )

    raise ValueError(f"Unknown provider: {provider}")


async def _call_provider_resilient(
    *,
    provider: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    use_sync: bool,
    budget: DeadlineBudget,
    settings: RuntimeResilienceSettings,
) -> tuple[str, dict]:
    timeout_seconds = (
        settings.gemini_timeout_seconds
        if provider == "gemini"
        else settings.ollama_timeout_seconds
    )

    async def operation(effective_timeout: float) -> str:
        return await _call_provider_once(
            provider=provider,
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            use_sync=use_sync,
            request_timeout=effective_timeout,
        )

    return await call_provider_with_resilience(
        provider_name=_provider_model_key(provider, model),
        operation=operation,
        budget=budget,
        timeout_seconds=timeout_seconds,
        retry_policy=_retry_policy(settings),
    )

async def generate_llm_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    provider: str = "gemini",
    model: Optional[str] = None,
    temperature: float = 0.2,
    allow_fallback: bool = True,
    use_sync: bool = False,
    budget: DeadlineBudget | None = None,
    resilience_settings: RuntimeResilienceSettings | None = None,
) -> dict:
    """Sinh text qua primary provider và chuỗi fallback hữu hạn nếu được phép.

    Kết quả giữ cả requested và actual provider/model để API, cache và telemetry
    không nhầm output fallback với primary. Mỗi target có retry riêng nhưng tất
    cả cùng chia sẻ ``budget`` của request.
    """
    settings = resilience_settings or runtime_resilience_settings_from_env()
    budget = budget or DeadlineBudget.from_timeout(settings.agent_total_timeout_seconds)
    provider, model = _resolve_model(provider or "gemini", model)
    requested_provider = provider
    requested_model = model
    
    result = {
        "text": "",
        "provider": provider,
        "model": model,
        "requested_provider": requested_provider,
        "requested_model": requested_model,
        "fallback_used": False,
        "fallback_provider": None,
        "fallback_model": None,
        "fallback_reason": None,
        "fallback_chain": [
            {
                "provider": provider,
                "model": model,
                "role": "primary",
                "status": "pending",
            }
        ],
        "error": None,
        "resilience": None,
    }
    
    try:
        # Primary target luôn được thử trước và được ghi ở đầu fallback_chain.
        text, resilience_meta = await _call_provider_resilient(
            provider=provider,
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            use_sync=use_sync,
            budget=budget,
            settings=settings,
        )
        result["text"] = text
        result["resilience"] = resilience_meta
        result["fallback_chain"][0]["status"] = "success"
        return result
            
    except asyncio.CancelledError:
        raise
    except PermanentProviderError as e:
        error_text = _safe_error_text(e)
        logger.warning("Primary LLM (%s/%s) failed permanently: %s", provider, model, error_text)
        result["error"] = error_text
        raise
    except RuntimeResilienceError as e:
        error_text = _safe_error_text(e)
        logger.warning("Primary LLM (%s/%s) failed: %s", provider, model, error_text)
        if not allow_fallback:
            result["error"] = error_text
            raise
        primary_error = e
    except Exception as e:
        error_text = _safe_error_text(e)
        logger.warning("Primary LLM (%s/%s) failed: %s", provider, model, error_text)
        
        # Chỉ chuẩn bị fallback khi caller bật; lỗi permanent không đi vào nhánh này.
        if not allow_fallback:
            logger.error("Fallback is disabled. Failing.")
            result["error"] = error_text
            raise e

        primary_error = e

    result["fallback_chain"][0]["status"] = "failed"
    result["fallback_chain"][0]["reason"] = _fallback_reason_from_error(primary_error)

    fallback_targets: list[tuple[str, str]] = []
    if allow_fallback and provider == "gemini":
        fallback_targets.extend(
            ("gemini", fallback_model)
            for fallback_model in parse_google_fallback_models(primary_model=model)
        )
        available_models = await list_ollama_models(timeout_seconds=min(5.0, budget.remaining_seconds()))
        configured_model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        configured_model = configured_model if ":" in configured_model else f"{configured_model}:latest"
        if configured_model in available_models:
            fallback_targets.append(("ollama", configured_model))
        elif DEFAULT_OLLAMA_MODEL in available_models:
            fallback_targets.append(("ollama", DEFAULT_OLLAMA_MODEL))
        elif "qwen3:latest" in available_models:
            fallback_targets.append(("ollama", "qwen3:latest"))
        elif "qwen2.5:latest" in available_models:
            fallback_targets.append(("ollama", "qwen2.5:latest"))
    elif allow_fallback and provider == "ollama":
        fallback_targets = []

    last_error: BaseException = primary_error
    for fallback_provider, fallback_model in fallback_targets:
        chain_entry = {
            "provider": fallback_provider,
            "model": fallback_model,
            "role": "fallback",
            "status": "pending",
        }
        result["fallback_chain"].append(chain_entry)
        try:
            logger.info("Fallback to %s (%s)...", fallback_provider, fallback_model)
            text, resilience_meta = await _call_provider_resilient(
                provider=fallback_provider,
                model=fallback_model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                use_sync=use_sync,
                budget=budget,
                settings=settings,
            )
            chain_entry["status"] = "success"
            result["text"] = text
            result["provider"] = fallback_provider
            result["model"] = fallback_model
            result["resilience"] = resilience_meta
            result["fallback_used"] = True
            result["fallback_provider"] = fallback_provider
            result["fallback_model"] = fallback_model
            result["fallback_reason"] = result["fallback_chain"][0].get("reason")
            return result
        except asyncio.CancelledError:
            raise
        except PermanentProviderError:
            chain_entry["status"] = "failed"
            chain_entry["reason"] = "permanent_provider_error"
            raise
        except RuntimeResilienceError as fb_err:
            chain_entry["status"] = "failed"
            chain_entry["reason"] = _fallback_reason_from_error(fb_err)
            last_error = fb_err
            logger.warning(
                "Fallback %s/%s failed: %s",
                fallback_provider,
                fallback_model,
                _safe_error_text(fb_err),
            )
            continue
        except Exception as fb_err:
            chain_entry["status"] = "failed"
            chain_entry["reason"] = _fallback_reason_from_error(fb_err)
            last_error = fb_err
            logger.warning(
                "Fallback %s/%s failed: %s",
                fallback_provider,
                fallback_model,
                _safe_error_text(fb_err),
            )
            continue

    if fallback_targets:
        logger.error("All LLM fallback targets failed.")
        raise _attach_fallback_trace(
            last_error,
            requested_provider=requested_provider,
            requested_model=requested_model,
            fallback_chain=result["fallback_chain"],
        )
    raise _attach_fallback_trace(
        primary_error,
        requested_provider=requested_provider,
        requested_model=requested_model,
        fallback_chain=result["fallback_chain"],
    )


__all__ = [
    "DEFAULT_GEMINI_FALLBACK_MODELS",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_OLLAMA_MODEL",
    "generate_llm_response",
    "parse_google_fallback_models",
]
