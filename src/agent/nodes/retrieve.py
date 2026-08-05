"""
src/agent/nodes/retrieve.py
===========================
LangGraph nodes for processing input and retrieving context.
"""

import logging

import asyncio
import os
import re
from src.agent.llm.provider import generate_llm_response
from src.agent.state import ClinicalState
from src.database.retriever import HybridRetriever
from src.knowledge import DrugEntityNormalizer
from src.quality.vietnamese_text import build_matching_views
from src.resilience.budget import DeadlineBudget
from src.resilience.contracts import RuntimeResilienceSettings, runtime_resilience_settings_from_env
from src.resilience.exceptions import RuntimeResilienceError, StageTimeoutError
from src.quality.safe_fallback import has_usable_evidence, sanitize_fallback_reason

logger = logging.getLogger(__name__)


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


async def normalize_question_node(state: ClinicalState) -> dict:
    """Normalize the user's question (e.g., lowercasing, stripping)."""
    question = state.get("user_question", "").strip()
    logger.debug("Normalizing question: chars=%d", len(question))
    
    # Simple normalization for Phase 2 (can be upgraded to LLM rewriting later)
    normalized = question.lower()
    
    return {"normalized_question": normalized}


async def rewrite_question_node(state: ClinicalState) -> dict:
    """Rewrite question based on conversation history for multi-turn context."""
    normalized = state.get("normalized_question", "")
    history = state.get("conversation_history", [])
    
    if not history:
        return {
            "standalone_question": normalized,
            "use_history_context": False,
        }

    explicit_primary_entities = [
        "benzoyl peroxide",
        "bp",
        "adapalene",
        "adapalen",
        "clindamycin",
        "erythromycin",
        "isotretinoin",
        "retinoid",
        "tretinoin",
        "tazarotene",
        "trifarotene",
        "tazorac",
        "differin",
        "epiduo",
        "dalacin",
    ]
    _, matching_question = build_matching_views(state.get("user_question", "") or normalized)
    has_coreference = _has_coreference_marker(matching_question)
    if any(entity in normalized for entity in explicit_primary_entities) and not has_coreference:
        return {
            "standalone_question": normalized,
            "use_history_context": False,
        }
        
    ambiguous_keywords = [
        "nó", "loại đó", "cái đó", "thuốc đó", "vậy còn", "như trên", "còn cái này", "vậy",
        "nhắc lại", "tình trạng da", "tuổi của tôi", "bắt đầu chăm sóc", "chăm sóc như thế nào",
        "có cần", "kháng sinh không", "uống kháng sinh", "routine",
    ]
    needs_rewrite = has_coreference or any(kw in normalized for kw in ambiguous_keywords)
    
    if not needs_rewrite:
        return {
            "standalone_question": normalized,
            "use_history_context": any(
                kw in normalized for kw in [
                    "vậy", "nhắc lại", "tình trạng da", "tuổi của tôi", "bắt đầu chăm sóc",
                    "chăm sóc như thế nào", "có cần", "kháng sinh không", "uống kháng sinh",
                    "routine",
                ]
            ),
        }

    deterministic_rewrite = _deterministic_followup_rewrite(
        normalized=normalized,
        original_question=state.get("user_question", ""),
        history=history,
    )
    if deterministic_rewrite:
        logger.info("Rewrote follow-up question with deterministic coreference resolver.")
        return {
            "standalone_question": deterministic_rewrite,
            "use_history_context": True,
        }
        
    logger.info("Question contains ambiguous keywords, attempting rewrite based on history.")
    
    # Format history for prompt
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    prompt = f"""
Dựa vào lịch sử hội thoại dưới đây, hãy viết lại câu hỏi cuối cùng của người dùng thành một câu hỏi độc lập (standalone question) đầy đủ ngữ cảnh.
Chỉ trả về câu hỏi đã được viết lại, không thêm bất kỳ thông tin nào khác, không trả lời câu hỏi. Giữ nguyên ngôn ngữ tiếng Việt.

Lịch sử hội thoại:
{history_text}

Câu hỏi hiện tại của người dùng:
User: {state.get('user_question', '')}

Câu hỏi độc lập:
"""
    try:
        llm_provider = state.get("llm_provider", "gemini")
        llm_model = state.get("llm_model")
        allow_model_fallback = state.get("allow_model_fallback", True)
        
        response_data = await generate_llm_response(
            prompt=prompt,
            provider=llm_provider,
            model=llm_model,
            temperature=0.0,
            allow_fallback=allow_model_fallback,
            use_sync=False,
            budget=_runtime_budget(state, _runtime_settings(state)),
            resilience_settings=_runtime_settings(state),
        )
        
        rewritten = response_data["text"].strip()
        logger.info("Rewrote question using conversation history: chars=%d", len(rewritten))
        return {
            "standalone_question": rewritten,
            "use_history_context": True,
        }
    except Exception as e:
        logger.error(
            "Failed to rewrite question, using original. Error: %s",
            sanitize_fallback_reason(e),
        )
        return {
            "standalone_question": normalized,
            "use_history_context": True,
        }


def _deterministic_followup_rewrite(
    *,
    normalized: str,
    original_question: str,
    history: list[dict],
) -> str | None:
    """Resolve common acne-drug coreference before falling back to LLM rewrite."""

    _, question_norm = build_matching_views(original_question or normalized)
    if not _has_coreference_marker(question_norm):
        return None

    product = _last_product_mention(history)
    if _contains_any(question_norm, ["hoat chat thu hai", "thanh phan thu hai"]):
        ingredient = _ingredient_for_product(product, position=2)
        if ingredient:
            return _rewrite_for_intent(question_norm, ingredient, product)

    target = _last_active_ingredient_mention(history)
    if not target and product:
        ingredient = _ingredient_for_product(product, position=1)
        if ingredient and _contains_any(question_norm, ["thuoc nhom", "nhom nao", "nhom gi", "thuoc gi"]):
            target = f"{product}/{ingredient}"
        else:
            target = product

    if not target:
        return None
    return _rewrite_for_intent(question_norm, target, product)


def _has_coreference_marker(text: str) -> bool:
    return _contains_any(
        text,
        [
            " no ",
            " no?",
            "thuoc do",
            "san pham do",
            "loai do",
            "cai do",
            "hoat chat do",
            "hoat chat thu hai",
            "thanh phan thu hai",
            "vay",
            "vay thi",
        ],
    )


def _rewrite_for_intent(question_norm: str, target: str, product: str | None) -> str:
    product_text = f" trong {product}" if product and product not in target else ""
    comparison_target = _comparison_target(question_norm, target)
    if comparison_target:
        return f"{target} khác {comparison_target} ở điểm nào?"
    if target.casefold() in {"clindamycin", "erythromycin"} and _contains_any(
        question_norm,
        ["dung rieng", "dung don doc", "keo dai"],
    ):
        return f"Có nên dùng {target} đơn độc hoặc kéo dài để trị mụn không?"
    if _contains_any(question_norm, ["khang sinh khong", "co phai khang sinh", "antibiotic"]):
        return f"{target}{product_text} có phải kháng sinh không?"
    if _contains_any(question_norm, ["thuoc nhom", "nhom nao", "nhom gi"]):
        return f"{target} thuộc nhóm thuốc nào?"
    if _contains_any(question_norm, ["tai sao", "vi sao", "why", "how"]) and _contains_any(
        question_norm,
        ["khang khuan", "antimicrobial", "vi khuan", "c. acnes"],
    ):
        return f"Vì sao {target}{product_text} có tác dụng kháng khuẩn/antimicrobial với C. acnes?"
    return f"{target}{product_text}: {question_norm}"


def _comparison_target(question_norm: str, target: str) -> str | None:
    """Return an explicit comparison entity while resolving the prior product."""

    if not _contains_any(question_norm, ["khac", "so sanh", "voi"]):
        return None
    target_norm = build_matching_views(target)[1]
    for candidate in ("Epiduo", "Differin", "Tazorac", "Dalacin T"):
        candidate_norm = build_matching_views(candidate)[1]
        if candidate_norm in question_norm and candidate_norm != target_norm:
            return candidate
    return None


def _last_product_mention(history: list[dict]) -> str | None:
    for message in reversed(history[-8:]):
        _, text = build_matching_views(str(message.get("content") or ""))
        for product in ["Epiduo", "Tazorac", "Differin", "Dalacin T"]:
            _, product_norm = build_matching_views(product)
            if product_norm in text:
                return product
    return None


def _last_active_ingredient_mention(history: list[dict]) -> str | None:
    aliases = [
        ("benzoyl peroxide", ["benzoyl peroxide", "bpo", "bp"]),
        ("adapalene", ["adapalene", "adapalen"]),
        ("tazarotene", ["tazarotene", "tazaroten"]),
        ("clindamycin", ["clindamycin"]),
        ("isotretinoin", ["isotretinoin"]),
        ("tretinoin", ["tretinoin"]),
    ]
    for message in reversed(history[-8:]):
        _, text = build_matching_views(str(message.get("content") or ""))
        for label, values in aliases:
            if any(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", f" {text} ") for alias in values):
                return label
    return None


def _ingredient_for_product(product: str | None, *, position: int) -> str | None:
    if not product:
        return None
    try:
        matches = DrugEntityNormalizer().normalize_mention(product)
    except Exception:
        matches = []
    if not matches:
        return None
    ingredients = list(matches[0].active_ingredients or [])
    index = position - 1
    if index < 0 or index >= len(ingredients):
        return None
    return _display_ingredient(ingredients[index])


def _display_ingredient(value: str) -> str:
    return str(value or "").replace("_", " ")


def _contains_any(text: str, needles: list[str]) -> bool:
    padded = f" {text} "
    return any(needle in padded for needle in needles)


async def extract_symptoms_node(state: ClinicalState) -> dict:
    """Extract symptoms and patient profile from the question."""
    question = state.get("standalone_question") or state.get("normalized_question", "")
    question = question.lower()
    
    # Phase 2 basic rule-based extraction (can be upgraded to LLM extraction)
    symptoms = []
    if "mụn viêm" in question or "sẩn viêm" in question:
        symptoms.append("mụn viêm")
    if "đỏ" in question:
        symptoms.append("đỏ")
    if "má" in question:
        symptoms.append("má")
    if "mụn mủ" in question:
        symptoms.append("mụn mủ")
    if "sẹo" in question:
        symptoms.append("sẹo")
        
    logger.debug(f"Extracted symptoms: {symptoms}")
    
    # Empty patient profile for now
    patient_profile = {}
    
    return {
        "symptoms": symptoms,
        "patient_profile": patient_profile
    }


async def retrieve_context_node(state: ClinicalState) -> dict:
    """Retrieve context using the HybridRetriever."""
    query = state.get("standalone_question") or state.get("normalized_question", "")
    
    if not query or not str(query).strip():
        return {
            "vector_contexts": [],
            "graph_facts": [],
            "sources": [],
            "retrieval_status": "empty_query",
            "retrieval_error": None,
        }
        
    logger.info("Retrieving context: query_chars=%d", len(str(query)))
    
    retriever = HybridRetriever()
    try:
        settings = _runtime_settings(state)
        budget = _runtime_budget(state, settings)
        timeout_seconds = budget.cap_timeout(settings.retrieval_timeout_seconds)
        if timeout_seconds <= 0:
            raise StageTimeoutError("No remaining deadline budget for retrieval.")
        async with asyncio.timeout(timeout_seconds):
            result = await retriever.retrieve(query, top_k=5)
        payload = {
            "vector_contexts": result.vector_contexts,
            "graph_facts": result.graph_facts,
            "sources": result.sources,
            "retrieval_trace": result.metadata.get("retrieval_trace"),
            "packed_context": result.metadata.get("packed_context"),
            "retrieval_error": None,
        }
        payload["retrieval_status"] = "success" if has_usable_evidence(payload) else "no_evidence"
        return {
            **payload,
        }
    except asyncio.CancelledError:
        raise
    except TimeoutError as exc:
        raise StageTimeoutError(f"Retrieval exceeded timeout of {timeout_seconds:.1f}s.") from exc
    except StageTimeoutError:
        raise
    except RuntimeResilienceError:
        raise
    except Exception as e:
        safe_error = sanitize_fallback_reason(e)
        logger.error("Recoverable retrieval error: %s", safe_error)
        return {
            "vector_contexts": [],
            "graph_facts": [],
            "sources": [],
            "retrieval_trace": None,
            "packed_context": None,
            "retrieval_status": "recoverable_error",
            "retrieval_error": safe_error,
            "errors": state.get("errors", []) + [f"Retrieval failed: {safe_error}"],
        }
    finally:
        await retriever.close()
