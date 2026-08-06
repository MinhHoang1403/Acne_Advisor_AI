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
from src.agent.source_presentation import build_source_allowlist
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


_CONVERSATION_TOPIC_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mụn đầu đen", ("mụn đầu đen", "mun dau den", "open comedone")),
    ("mụn đầu trắng", ("mụn đầu trắng", "mun dau trang", "closed comedone")),
    ("mụn lưng", ("mụn lưng", "mun lung", "acne on the back", "back acne")),
)


def build_conversation_context(history: list[dict[str, str]] | None) -> dict[str, object]:
    """Extract only user-provided follow-up facts for the current request."""

    user_messages = [
        str(message.get("content") or "")
        for message in (history or [])[-8:]
        if str(message.get("role") or "").casefold() == "user"
    ]
    joined = "\n".join(user_messages)
    _, folded = build_matching_views(joined)
    active_topic = next(
        (
            topic
            for topic, markers in _CONVERSATION_TOPIC_MARKERS
            if any(marker in folded for marker in markers)
        ),
        None,
    )
    tolerance = next(
        (
            label
            for label, markers in (
                ("khô nhẹ", ("khô nhẹ", "kho nhe")),
                ("rát", ("rát", "rat", "châm chích", "cham chich")),
                ("bong tróc", ("bong tróc", "bong troc", "bong da")),
            )
            if any(marker in folded for marker in markers)
        ),
        None,
    )
    candidate_entities = _user_entity_candidates(history or [])
    active_treatment_class = next(
        (
            label
            for label, markers in (
                ("retinoid bôi", ("retinoid bôi", "retinoid boi", "topical retinoid")),
            )
            if any(marker in folded for marker in markers)
        ),
        None,
    )
    return {
        "active_topic": active_topic,
        "active_product": _last_product_mention(history or [], user_only=True),
        "active_ingredient": _last_active_ingredient_mention(history or [], user_only=True),
        "active_treatment_class": active_treatment_class,
        "tolerance_context": tolerance,
        "pregnancy_context": any(marker in folded for marker in ("mang thai", "co thai", "thai ky", "cho con bu")),
        "antibiotic_context": any(marker in folded for marker in ("kháng sinh", "khang sinh", "clindamycin", "erythromycin", "doxycycline")),
        "last_user_message": user_messages[-1] if user_messages else None,
        "candidate_entities": candidate_entities,
        "unresolved_user_reference": False,
    }


async def rewrite_question_node(state: ClinicalState) -> dict:
    """Rewrite question based on conversation history for multi-turn context."""
    normalized = state.get("normalized_question", "")
    history = state.get("conversation_history", [])
    conversation_context = build_conversation_context(history)
    
    if not history:
        return {
            "standalone_question": normalized,
            "use_history_context": False,
            "conversation_context": conversation_context,
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
            "conversation_context": conversation_context,
        }
        
    ambiguous_keywords = [
        "nó", "loại đó", "cái đó", "thuốc đó", "vậy còn", "như trên", "còn cái này", "vậy",
        "nhắc lại", "tình trạng da", "tuổi của tôi", "bắt đầu chăm sóc", "chăm sóc như thế nào",
        "có cần", "kháng sinh không", "uống kháng sinh", "routine",
    ]
    has_implicit_treatment_followup = _has_implicit_treatment_followup(
        matching_question,
        history,
        conversation_context,
    )
    ambiguity_options = _ambiguous_reference_options(
        matching_question,
        conversation_context,
        has_coreference=has_coreference,
    )
    if ambiguity_options:
        return {
            "standalone_question": normalized,
            "use_history_context": True,
            "conversation_context": {
                **conversation_context,
                "unresolved_user_reference": True,
                "clarification_options": ambiguity_options,
            },
        }
    needs_rewrite = has_coreference or has_implicit_treatment_followup or any(kw in normalized for kw in ambiguous_keywords)
    
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
            "conversation_context": conversation_context,
        }

    deterministic_rewrite = _deterministic_followup_rewrite(
        normalized=normalized,
        original_question=state.get("user_question", ""),
        history=history,
        conversation_context=conversation_context,
        allow_implicit_context=has_implicit_treatment_followup,
    )
    if deterministic_rewrite:
        logger.info("Rewrote follow-up question with deterministic coreference resolver.")
        return {
            "standalone_question": deterministic_rewrite,
            "use_history_context": True,
            "conversation_context": conversation_context,
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
            "conversation_context": conversation_context,
        }
    except Exception as e:
        logger.error(
            "Failed to rewrite question, using original. Error: %s",
            sanitize_fallback_reason(e),
        )
        return {
            "standalone_question": normalized,
            "use_history_context": True,
            "conversation_context": conversation_context,
        }


def _deterministic_followup_rewrite(
    *,
    normalized: str,
    original_question: str,
    history: list[dict],
    conversation_context: dict[str, object] | None = None,
    allow_implicit_context: bool = False,
) -> str | None:
    """Resolve common acne-drug coreference before falling back to LLM rewrite."""

    _, question_norm = build_matching_views(original_question or normalized)
    if not _has_coreference_marker(question_norm) and not allow_implicit_context:
        return None

    context = conversation_context or {}
    product = str(context.get("active_product") or "") or _last_product_mention(history)
    active_topic = str(context.get("active_topic") or "")
    active_tolerance = str(context.get("tolerance_context") or "")
    active_treatment_class = str(context.get("active_treatment_class") or "")
    if active_tolerance and _contains_any(question_norm, ["hoat chat", "luc nay"]):
        return (
            f"Da đang {active_tolerance} khi dùng các hoạt chất bôi: nên giảm hoặc tạm ngưng hoạt chất nào "
            "và dưỡng ẩm thế nào để hạn chế kích ứng?"
        )
    if _is_frequency_adjustment_followup(question_norm) and active_tolerance:
        target = str(context.get("active_ingredient") or "") or product
        if target:
            return (
                f"Da {active_tolerance} khi dùng {target}: nên điều chỉnh tần suất thế nào và có cần dưỡng ẩm không?"
            )
    if active_topic == "mụn đầu đen" and _contains_any(question_norm, ["dang do", "mun viem"]):
        return "Mụn đầu đen có phải là mụn viêm không?"
    if active_topic == "mụn lưng" and _contains_any(question_norm, ["thoi quen", "chu y them"]):
        return "Với mụn lưng sau khi tập, thói quen nào liên quan đến mồ hôi và ma sát nên chú ý thêm?"
    if active_treatment_class == "retinoid bôi" and _contains_any(question_norm, ["ban ngay", "chong nang"]):
        return "Khi dùng retinoid bôi buổi tối, ban ngày cần làm gì để bảo vệ da và hạn chế kích ứng?"
    if _contains_any(question_norm, ["hoat chat thu hai", "thanh phan thu hai"]):
        ingredient = _ingredient_for_product(product, position=2)
        if ingredient:
            return _rewrite_for_intent(question_norm, ingredient, product)

    if product and _contains_any(question_norm, ["hoat chat chinh", "thanh phan chinh"]):
        ingredient = _ingredient_for_product(product, position=1)
        if ingredient:
            return f"Hoạt chất chính của {product} là gì? (Theo taxonomy: {ingredient}.)"

    target = str(context.get("active_ingredient") or "") or _last_active_ingredient_mention(history)
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
            "dang do",
            "hoat chat do",
            "hoat chat thu hai",
            "thanh phan thu hai",
            "ten do",
            "day la",
            "day la thuoc",
            "vay",
            "vay thi",
        ],
    )


def _has_implicit_treatment_followup(
    question_norm: str,
    history: list[dict],
    conversation_context: dict[str, object] | None = None,
) -> bool:
    """Carry the active treatment forward for narrow, context-dependent follow-ups."""

    context = conversation_context or {}
    if not (
        context.get("active_product")
        or context.get("active_ingredient")
        or context.get("active_treatment_class")
        or context.get("active_topic")
        or context.get("tolerance_context")
        or _last_product_mention(history)
        or _last_active_ingredient_mention(history)
    ):
        return False
    return _contains_any(
        question_norm,
        [
            "nhieu hoat chat",
            "them hoat chat",
            "dieu chinh tan suat",
            "ban ngay co buoc",
            "thoi quen nao",
            "tan suat the nao",
            "dang do",
            "hoat chat",
            "luc nay",
        ],
    )


def _is_frequency_adjustment_followup(question_norm: str) -> bool:
    return _contains_any(
        question_norm,
        ["dieu chinh tan suat", "tan suat the nao", "dung may lan", "giam tan suat"],
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
        if product and product not in target:
            return f"{product} ({target}) thuộc nhóm thuốc nào?"
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


def _last_product_mention(history: list[dict], *, user_only: bool = False) -> str | None:
    for message in reversed(history[-8:]):
        if user_only and str(message.get("role") or "").casefold() != "user":
            continue
        _, text = build_matching_views(str(message.get("content") or ""))
        for product in ["Epiduo", "Tazorac", "Differin", "Dalacin T"]:
            _, product_norm = build_matching_views(product)
            if product_norm in text:
                return product
    return None


def _last_active_ingredient_mention(history: list[dict], *, user_only: bool = False) -> str | None:
    aliases = [
        ("benzoyl peroxide", ["benzoyl peroxide", "bpo", "bp"]),
        ("adapalene", ["adapalene", "adapalen"]),
        ("tazarotene", ["tazarotene", "tazaroten"]),
        ("clindamycin", ["clindamycin"]),
        ("isotretinoin", ["isotretinoin"]),
        ("tretinoin", ["tretinoin"]),
    ]
    for message in reversed(history[-8:]):
        if user_only and str(message.get("role") or "").casefold() != "user":
            continue
        _, text = build_matching_views(str(message.get("content") or ""))
        for label, values in aliases:
            if any(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", f" {text} ") for alias in values):
                return label
    return None


def _user_entity_candidates(history: list[dict]) -> list[str]:
    """Return distinct explicit entities from user turns, newest turn first."""

    candidates: list[str] = []
    known_entities = [
        "Epiduo",
        "Tazorac",
        "Differin",
        "Dalacin T",
        "benzoyl peroxide",
        "adapalene",
        "tazarotene",
        "clindamycin",
        "isotretinoin",
        "tretinoin",
    ]
    for message in reversed(history[-8:]):
        if str(message.get("role") or "").casefold() != "user":
            continue
        _, text = build_matching_views(str(message.get("content") or ""))
        for entity in known_entities:
            _, entity_norm = build_matching_views(entity)
            if entity_norm in text and entity not in candidates:
                candidates.append(entity)
    return candidates


def _ambiguous_reference_options(
    question_norm: str,
    conversation_context: dict[str, object],
    *,
    has_coreference: bool,
) -> list[str]:
    """Return choices only when an implicit reference has competing user entities."""

    if not has_coreference:
        return []
    explicit_entities = _user_entity_candidates(
        [{"role": "user", "content": question_norm}]
    )
    if explicit_entities:
        return []
    candidates = [str(value) for value in conversation_context.get("candidate_entities", []) if value]
    return candidates[:2] if len(candidates) >= 2 else []


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
            "graph_relation_found": False,
            "sources": [],
            "source_allowlist": [],
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
            "graph_relation_found": any(
                fact.get("predicate") or fact.get("relationship")
                for fact in result.graph_facts
                if isinstance(fact, dict)
            ),
            "sources": result.sources,
            "source_allowlist": build_source_allowlist(result.sources, result.vector_contexts),
            "retrieval_trace": result.metadata.get("retrieval_trace"),
            "packed_context": result.metadata.get("packed_context"),
            "retrieval_error": None,
            "performance_timings": {
                **(state.get("performance_timings") or {}),
                **{
                    f"retrieval_{name}": float(value)
                    for name, value in (result.metadata.get("retrieval_trace", {}).get("timings_ms", {}) or {}).items()
                    if isinstance(value, (int, float))
                },
            },
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
            "graph_relation_found": False,
            "sources": [],
            "source_allowlist": [],
            "retrieval_trace": None,
            "packed_context": None,
            "retrieval_status": "recoverable_error",
            "retrieval_error": safe_error,
            "errors": state.get("errors", []) + [f"Retrieval failed: {safe_error}"],
        }
    finally:
        await retriever.close()
