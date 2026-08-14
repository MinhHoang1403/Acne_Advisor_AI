"""Small request preparation helpers for the final agent."""

from __future__ import annotations

import logging

from src.agent.llm.provider import generate_llm_response
from src.agent.state import ClinicalState
from src.quality.safe_fallback import sanitize_fallback_reason
from src.resilience.budget import DeadlineBudget
from src.resilience.contracts import RuntimeResilienceSettings, runtime_resilience_settings_from_env

logger = logging.getLogger(__name__)

_COREFERENCE_MARKERS = (
    "nó",
    "loại đó",
    "thuốc đó",
    "hoạt chất đó",
    "hoạt chất thứ",
    "cái này",
    "cái đó",
    "vậy còn",
    "như trên",
    "then what about",
    "that medicine",
)


async def normalize_question_node(state: ClinicalState) -> dict[str, str]:
    """Apply only lossless whitespace normalization."""

    question = " ".join(str(state.get("user_question") or "").split())
    return {"normalized_question": question}


async def rewrite_question_node(state: ClinicalState) -> dict[str, object]:
    """Use the configured language model only for genuine multi-turn references."""

    question = state.get("normalized_question") or state.get("user_question", "")
    history = list(state.get("conversation_history") or [])[-6:]
    context = {
        "history_messages": len(history),
        "rewrite_requested": False,
    }
    if not history or not _needs_history(question):
        return {
            "standalone_question": question,
            "use_history_context": False,
            "conversation_context": context,
        }

    context["rewrite_requested"] = True
    history_text = "\n".join(
        f"{str(item.get('role') or 'user')}: {str(item.get('content') or '')}"
        for item in history
    )
    prompt = (
        "Viết lại câu hỏi cuối thành một câu hỏi y khoa độc lập bằng tiếng Việt. "
        "Chỉ dùng thông tin có trong lịch sử, giữ nguyên mọi tên thuốc/hoạt chất, "
        "không trả lời và không thêm dữ kiện.\n\n"
        f"Lịch sử:\n{history_text}\n\nCâu hỏi cuối:\n{question}\n\nCâu hỏi độc lập:"
    )
    try:
        response = await generate_llm_response(
            prompt=prompt,
            provider=state.get("llm_provider") or "gemini",
            model=state.get("llm_model"),
            temperature=0.0,
            allow_fallback=state.get("allow_model_fallback", True),
            use_sync=False,
            budget=_runtime_budget(state),
            resilience_settings=_runtime_settings(state),
        )
        rewritten = " ".join(str(response.get("text") or "").split())
        if rewritten:
            return {
                "standalone_question": rewritten,
                "use_history_context": True,
                "conversation_context": {**context, "rewrite_succeeded": True},
            }
    except Exception as exc:
        logger.warning("Conversation rewrite failed safely: %s", sanitize_fallback_reason(exc))
    return {
        "standalone_question": question,
        "use_history_context": True,
        "conversation_context": {**context, "rewrite_succeeded": False},
    }


async def extract_symptoms_node(state: ClinicalState) -> dict[str, list[str]]:
    """Keep symptom interpretation with the LLM; no keyword diagnosis is performed."""

    return {"symptoms": []}


def _needs_history(question: str) -> bool:
    folded = f" {question.casefold()} "
    return any(marker in folded for marker in _COREFERENCE_MARKERS)


def _runtime_settings(state: ClinicalState) -> RuntimeResilienceSettings:
    configured = state.get("runtime_resilience_settings")
    if isinstance(configured, dict):
        return RuntimeResilienceSettings(**configured)
    return runtime_resilience_settings_from_env()


def _runtime_budget(state: ClinicalState) -> DeadlineBudget:
    budget = state.get("runtime_budget")
    if isinstance(budget, DeadlineBudget):
        return budget
    return DeadlineBudget.from_timeout(_runtime_settings(state).agent_total_timeout_seconds)


__all__ = ["extract_symptoms_node", "normalize_question_node", "rewrite_question_node"]
