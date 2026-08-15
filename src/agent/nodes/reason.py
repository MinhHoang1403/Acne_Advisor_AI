"""Evidence-grounded answer generation node."""

import logging
import os
import time
from typing import Any

from src.agent.state import ClinicalState
from src.resilience.budget import DeadlineBudget
from src.resilience.contracts import RuntimeResilienceSettings, runtime_resilience_settings_from_env
from src.resilience.exceptions import ProviderUnavailableError, RuntimeResilienceError
from src.quality.safe_fallback import sanitize_fallback_reason

logger = logging.getLogger(__name__)


from src.agent.llm.provider import generate_llm_response


def _runtime_settings(state: ClinicalState) -> RuntimeResilienceSettings:
    configured = state.get("runtime_resilience_settings")
    if isinstance(configured, dict):
        return RuntimeResilienceSettings(**configured)
    return runtime_resilience_settings_from_env()


def _runtime_budget(state: ClinicalState, settings: RuntimeResilienceSettings) -> DeadlineBudget:
    budget = state.get("runtime_budget")
    if isinstance(budget, DeadlineBudget):
        return budget
    return DeadlineBudget.from_timeout(settings.agent_total_timeout_seconds)


def _select_answer_contexts(contexts: list[dict[str, Any]], limit: int = 5, query: str = "") -> list[dict[str, Any]]:
    """Preserve the canonical packer's fused order for answer generation."""

    del query
    return [dict(context) for context in contexts[:limit]]

async def generate_answer_node(state: ClinicalState) -> dict:
    """Generate an answer to the current question from retrieved chunk evidence."""
    question = state.get("normalized_question") or state.get("user_question", "")
    contexts = state.get("vector_contexts", [])
    conversation_context = state.get("conversation_context") or {}
    conversation_history = list(conversation_context.get("messages") or [])
    source_allowlist = state.get("source_allowlist", [])
    prompt_history = conversation_history

    try:
        from src.agent.prompts.medical_answer import (
            build_medical_prompt,
            build_medical_system_instruction,
            observe_medical_prompt_budget,
        )

        answer_contexts = _select_answer_contexts(contexts, limit=5, query=question)
        packed_context = state.get("packed_context") or {}
        packed_context_text = str(packed_context.get("context_text") or "")
        
        prompt_started = time.perf_counter()
        prompt = build_medical_prompt(
            question=question,
            contexts=answer_contexts,
            conversation_history=prompt_history,
            available_sources=source_allowlist,
            packed_context_text=packed_context_text,
        )
        system_prompt = build_medical_system_instruction(question)
        prompt_budget = observe_medical_prompt_budget(prompt)
        prompt_ms = round((time.perf_counter() - prompt_started) * 1000, 3)
        
        llm_provider = state.get("llm_provider") or os.getenv("LLM_PROVIDER", "gemini")
        llm_model = state.get("llm_model")
        allow_model_fallback = state.get("allow_model_fallback", False)
        settings = _runtime_settings(state)
        
        logger.info(f"Generating answer with LLM: provider={llm_provider}, model={llm_model}")
        
        generation_started = time.perf_counter()
        response_data = await generate_llm_response(
            prompt=prompt,
            system_prompt=system_prompt,
            provider=llm_provider,
            model=llm_model,
            temperature=0.2,
            allow_fallback=allow_model_fallback,
            budget=_runtime_budget(state, settings),
            resilience_settings=settings,
        )
        generation_ms = round((time.perf_counter() - generation_started) * 1000, 3)
        
        draft = response_data.get("text")
        logger.info("LLM generation successful.")
        
        return {
            "draft_answer": draft,
            "sources": [
                str(entry.get("source_id"))
                for entry in source_allowlist
                if entry.get("source_id") and not str(entry.get("source_id")).startswith("entity:")
            ]
            or list(dict.fromkeys(
                ctx.get("source_file", "")
                for ctx in answer_contexts
                if ctx.get("source_file") and not str(ctx.get("source_file")).startswith("entity:")
            ))
            or state.get("sources", []),
            "requested_provider": response_data.get("requested_provider"),
            "requested_model": response_data.get("requested_model"),
            "actual_provider": response_data["provider"],
            "actual_model": response_data["model"],
            "llm_fallback_used": response_data["fallback_used"],
            "fallback_provider": response_data["fallback_provider"],
            "fallback_model": response_data["fallback_model"],
            "fallback_reason": response_data.get("fallback_reason"),
            "fallback_chain": response_data.get("fallback_chain"),
            "prompt_budget": prompt_budget.model_dump(mode="json"),
            "runtime_resilience": {
                **(state.get("runtime_resilience") or {}),
                "llm": response_data.get("resilience"),
            },
            "performance_timings": {
                **(state.get("performance_timings") or {}),
                "prompt_construction": prompt_ms,
                "llm_generation": generation_ms,
            },
        }
        
    except RuntimeResilienceError:
        raise
    except Exception as e:
        safe_error = sanitize_fallback_reason(e)
        logger.error("LLM generation provider error: %s", safe_error)
        raise ProviderUnavailableError("LLM provider unavailable or returned an error.") from e
