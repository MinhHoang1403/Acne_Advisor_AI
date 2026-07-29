"""
src/agent/nodes/reason.py
=========================
LangGraph nodes for safety checks and reasoning (generating answers).
"""

import logging
import re
from typing import Any

from src.agent.state import ClinicalState
from src.resilience.budget import DeadlineBudget
from src.resilience.contracts import RuntimeResilienceSettings, runtime_resilience_settings_from_env
from src.resilience.exceptions import ProviderUnavailableError, RuntimeResilienceError
from src.quality.safe_fallback import sanitize_fallback_reason

logger = logging.getLogger(__name__)


async def safety_check_node(state: ClinicalState) -> dict:
    """Check the query and contexts for safety issues (e.g., severe conditions)."""
    question = state.get("normalized_question", "")
    symptoms = state.get("symptoms", [])
    
    flags = []
    
    # Simple rule-based safety check for Phase 2
    severe_keywords = ["chảy máu", "nhiễm trùng", "mủ nhiều", "sốt", "đau nhức dữ dội"]
    for kw in severe_keywords:
        if kw in question:
            flags.append(f"Cảnh báo: Có dấu hiệu nghiêm trọng ({kw}).")

    emergency_keywords = ["đau ngực", "khó thở", "tức ngực", "ngất", "choáng", "sốc phản vệ"]
    for kw in emergency_keywords:
        if kw in question:
            flags.append(
                f"Cảnh báo khẩn cấp: {kw} không phải triệu chứng điển hình của mụn; "
                "nên đi cấp cứu hoặc liên hệ cơ sở y tế ngay nếu triệu chứng đang xảy ra."
            )

    # ── Retinoid + Pregnancy safety flags ─────────────────────────────
    question_lower = question.lower()
    retinoid_keywords = [
        "isotretinoin", "tretinoin", "retinoid",
        "adapalene", "adapalen",
        "tazarotene", "tazaroten",
    ]
    pregnancy_keywords = [
        "mang thai", "có thai", "có bầu",
        "chuẩn bị mang thai", "kế hoạch mang thai",
        "pregnancy", "pregnant",
    ]
    has_retinoid = any(kw in question_lower for kw in retinoid_keywords)
    has_pregnancy = any(kw in question_lower for kw in pregnancy_keywords)

    if has_retinoid and has_pregnancy:
        # Presentation V3 renders the pregnancy warning inside the answer body.
        # Avoid a second UI-level "Cảnh báo nghiêm trọng" banner for the same risk.
        logger.debug("Retinoid pregnancy safety context detected; warning will be integrated in final answer.")

    logger.debug(f"Safety flags: {flags}")
    return {"safety_flags": flags}


import os
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


def _is_reference_context(ctx: dict[str, Any]) -> bool:
    header = str(ctx.get("header") or ctx.get("parent_header_path") or "").lower()
    role = str(ctx.get("context_role") or "").lower()
    content_type = ctx.get("content_type", [])
    if isinstance(content_type, str):
        content_type = [content_type]
    content_type_text = " ".join(str(item).lower() for item in content_type)
    markers = ("references", "reference", "bibliography", "tài liệu tham khảo", "tham khảo")
    return role == "reference" or any(marker in header or marker in content_type_text for marker in markers)


_LOW_VALUE_SECTION_MARKERS = (
    "abbreviations",
    "abbreviation",
    "references",
    "reference",
    "bibliography",
    "funding",
    "acknowledgements",
    "acknowledgments",
    "table of contents",
    "contents",
    "author",
    "correspondence",
    "tài liệu tham khảo",
    "mục lục",
)

_CLINICAL_SECTION_MARKERS = (
    "recommendation",
    "management",
    "treatment",
    "therapy",
    "safety",
    "adverse",
    "side effect",
    "contraindication",
    "pregnancy",
    "maintenance",
    "skin care",
    "referral",
    "cơ chế",
    "điều trị",
    "tác dụng phụ",
    "chống chỉ định",
    "chăm sóc",
    "khuyến cáo",
    "chuyển tuyến",
)

_DOCUMENT_CODE_RE = re.compile(r"^(?:ng\s*198|ng198|nice\s*ng\s*198|aad\s*2024|\d{2,})$", re.IGNORECASE)


def _context_header_text(ctx: dict[str, Any]) -> str:
    return str(
        ctx.get("header")
        or ctx.get("parent_header_path")
        or ctx.get("section")
        or ""
    ).lower()


def _context_text(ctx: dict[str, Any]) -> str:
    return str(ctx.get("text") or ctx.get("content") or ctx.get("page_content") or "")


def _is_low_value_context(ctx: dict[str, Any]) -> bool:
    header = _context_header_text(ctx)
    text = _context_text(ctx).strip()
    if any(marker in header for marker in _LOW_VALUE_SECTION_MARKERS):
        return True
    if len(text) < 80 and re.search(r"\b[A-Z]{2,8}\s*[:=]\s*[A-Za-z][A-Za-z\s-]{2,40}$", text):
        return True
    return False


def _is_bp_antibiotic_identity_query(query: str) -> bool:
    query_lower = query.lower()
    has_bp = bool(re.search(r"\bbenzoyl\s+peroxide\b|\bbp\b", query_lower))
    asks_antibiotic_identity = any(
        marker in query_lower
        for marker in [
            "có phải kháng sinh không",
            "phải kháng sinh không",
            "là kháng sinh không",
            "is benzoyl peroxide an antibiotic",
            "is bp an antibiotic",
        ]
    )
    return has_bp and asks_antibiotic_identity


def _context_quality_score(ctx: dict[str, Any], query: str = "") -> float:
    score = float(ctx.get("score") or ctx.get("boosted_score") or 0.0)
    header = _context_header_text(ctx)
    text = _context_text(ctx)
    text_lower = text.lower()
    query_lower = query.lower()

    if _is_low_value_context(ctx):
        score -= 0.35
    if _is_reference_context(ctx):
        score -= 0.25
    if any(marker in header or marker in text_lower[:500] for marker in _CLINICAL_SECTION_MARKERS):
        score += 0.20
    if len(text.strip()) >= 250:
        score += 0.05
    if _is_bp_antibiotic_identity_query(query_lower):
        has_bp = "benzoyl peroxide" in text_lower or re.search(r"\bbp\b", text_lower)
        has_direct_antibiotic_contrast = any(
            marker in text_lower
            for marker in [
                "does not contain antibiotics",
                "not an antibiotic",
                "không phải kháng sinh",
                "benzoyl peroxide",
                "kháng sinh bôi tại chỗ",
                "dùng dạng phối hợp với bp",
            ]
        )
        mentions_oral_antibiotics = any(
            marker in text_lower
            for marker in ["oral antibiotic", "kháng sinh uống", "kháng sinh đường uống", "doxycycline", "lymecycline", "minocycline"]
        )
        if has_bp and has_direct_antibiotic_contrast:
            score += 0.45
        if mentions_oral_antibiotics and not has_bp:
            score -= 0.45
    return score


def _select_answer_contexts(contexts: list[dict[str, Any]], limit: int = 5, query: str = "") -> list[dict[str, Any]]:
    """Select prompt contexts, preferring clinical chunks over abbreviations/references."""
    if not contexts:
        return []
    ranked = sorted(
        (dict(ctx) for ctx in contexts),
        key=lambda ctx: _context_quality_score(ctx, query=query),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    low_value_fallback: list[dict[str, Any]] = []
    for ctx in ranked:
        if _is_low_value_context(ctx) or _is_reference_context(ctx):
            ctx["context_role"] = "supporting"
            low_value_fallback.append(ctx)
            continue
        ctx["context_role"] = "main"
        selected.append(ctx)
        if len(selected) >= limit:
            return selected
    selected.extend(low_value_fallback[: max(0, limit - len(selected))])
    return selected[:limit]


def _tokenize_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-ZÀ-ỹ0-9][a-zA-ZÀ-ỹ0-9_.-]{2,}", text.lower())
        if token not in {"the", "and", "for", "with", "trong", "của", "với", "này"}
    }


def _is_bad_entity_name(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text or len(text) < 3:
        return True
    if _DOCUMENT_CODE_RE.match(text):
        return True
    if text.isdigit():
        return True
    return False


def _is_mechanism_or_bacteria(value: str) -> bool:
    text = value.lower()
    markers = (
        "c. acnes",
        "cutibacterium",
        "propionibacterium",
        "vi khuẩn",
        "bacteria",
        "sebum",
        "bã nhờn",
        "comedogenesis",
        "keratin",
        "inflammation",
        "viêm",
        "pathogenesis",
    )
    return any(marker in text for marker in markers)


def filter_graph_facts_for_prompt(
    query: str,
    contexts: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Filter noisy Neo4j facts before they are allowed into the LLM prompt."""
    if not graph_facts:
        return []

    context_text = " ".join(_context_text(ctx) for ctx in contexts).lower()
    graph_node_terms: set[str] = set()
    for ctx in contexts:
        nodes = ctx.get("graph_nodes", [])
        if isinstance(nodes, list):
            graph_node_terms.update(str(node).lower() for node in nodes if node)

    query_terms = _tokenize_terms(query)
    ranked: list[tuple[float, dict[str, Any]]] = []
    seen: set[tuple[str, str, str]] = set()

    for fact in graph_facts:
        entity = str(fact.get("entity") or "").strip()
        related = str(fact.get("related_entity") or "").strip()
        rel = str(fact.get("relationship") or "").strip().upper()
        description = str(fact.get("description") or "").strip()
        related_description = str(fact.get("related_description") or "").strip()
        evidence = str(fact.get("evidence") or "").strip()

        if _is_bad_entity_name(entity) or (related and _is_bad_entity_name(related)):
            continue
        if not evidence and not description and not related_description:
            continue
        if rel == "TREATS" and (_is_mechanism_or_bacteria(entity) or _is_mechanism_or_bacteria(related)):
            if entity.lower() not in context_text and related.lower() not in context_text:
                continue

        key = (entity.lower(), rel, related.lower())
        if key in seen:
            continue
        seen.add(key)

        fact_terms = _tokenize_terms(f"{entity} {related} {description} {related_description} {evidence}")
        overlap = len(fact_terms & query_terms)
        node_overlap = int(entity.lower() in graph_node_terms or related.lower() in graph_node_terms)
        evidence_bonus = 0.5 if evidence else 0.0
        ranked.append((overlap + node_overlap + evidence_bonus, fact))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [fact for _, fact in ranked[:limit]]

async def generate_answer_node(state: ClinicalState) -> dict:
    """Generate the answer based on vector contexts and graph facts using LLM."""
    question = state.get("standalone_question") or state.get("user_question", "")
    contexts = state.get("vector_contexts", [])
    graph_facts = state.get("graph_facts", [])
    safety_flags = state.get("safety_flags", [])
    symptoms = state.get("symptoms", [])
    conversation_history = state.get("conversation_history", [])
    prompt_history = conversation_history if state.get("use_history_context") else []
    ignored_out_of_domain_part = state.get("ignored_out_of_domain_part", False)
    
    if state.get("is_in_domain") is False:
        logger.debug("Domain guardrail triggered. Skipping LLM generation.")
        return {}
        
    try:
        from src.agent.prompts.medical_answer import build_medical_prompt

        answer_contexts = _select_answer_contexts(contexts, limit=5, query=question)
        prompt_graph_facts = filter_graph_facts_for_prompt(
            query=question,
            contexts=answer_contexts,
            graph_facts=graph_facts,
            limit=10,
        )
        
        prompt = build_medical_prompt(
            question=question,
            symptoms=symptoms,
            safety_flags=safety_flags,
            contexts=answer_contexts,
            graph_facts=prompt_graph_facts,
            conversation_history=prompt_history,
            ignored_out_of_domain_part=ignored_out_of_domain_part
        )
        
        llm_provider = state.get("llm_provider") or os.getenv("LLM_PROVIDER", "gemini")
        llm_model = state.get("llm_model")
        allow_model_fallback = state.get("allow_model_fallback", True)
        settings = _runtime_settings(state)
        
        logger.info(f"Generating answer with LLM: provider={llm_provider}, model={llm_model}")
        
        response_data = await generate_llm_response(
            prompt=prompt,
            provider=llm_provider,
            model=llm_model,
            temperature=0.2,
            allow_fallback=allow_model_fallback,
            budget=_runtime_budget(state, settings),
            resilience_settings=settings,
        )
        
        draft = response_data.get("text")
        logger.info("LLM generation successful.")
        
        return {
            "draft_answer": draft,
            "sources": list(dict.fromkeys(
                ctx.get("source_file", "")
                for ctx in answer_contexts
                if ctx.get("source_file")
            ))[:2] or state.get("sources", [])[:2],
            "requested_provider": response_data.get("requested_provider"),
            "requested_model": response_data.get("requested_model"),
            "actual_provider": response_data["provider"],
            "actual_model": response_data["model"],
            "llm_fallback_used": response_data["fallback_used"],
            "fallback_provider": response_data["fallback_provider"],
            "fallback_model": response_data["fallback_model"],
            "fallback_reason": response_data.get("fallback_reason"),
            "fallback_chain": response_data.get("fallback_chain"),
            "runtime_resilience": {
                **(state.get("runtime_resilience") or {}),
                "llm": response_data.get("resilience"),
            },
        }
        
    except RuntimeResilienceError:
        raise
    except Exception as e:
        safe_error = sanitize_fallback_reason(e)
        logger.error("LLM generation provider error: %s", safe_error)
        raise ProviderUnavailableError("LLM provider unavailable or returned an error.") from e
