"""Bounded, provenance-preserving packing for fused source evidence."""

from __future__ import annotations

from typing import Any

from src.retrieval.contracts import ContextItem, NormalizedQuery, PackedContext, RetrievedCandidate


def pack_context(
    normalized_query: NormalizedQuery,
    candidates: list[RetrievedCandidate],
    max_items: int = 8,
    max_chars: int = 6000,
) -> PackedContext:
    """Pack candidates in fused-rank order without changing relevance.

    ``max_items`` and ``max_chars`` are resource limits, not relevance
    heuristics. Every selected item retains its point, chunk, document, and
    source identifiers in the payload.
    """

    item_limit = max(1, max_items)
    char_limit = max(256, max_chars)
    selected: list[ContextItem] = []
    warnings: list[str] = []
    dropped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    rendered_chars = 0

    for candidate in candidates:
        candidate_id = candidate.candidate_id.strip()
        text = candidate.text.strip()
        if not candidate_id or not text:
            dropped.append({"candidate_id": candidate_id, "reason": "missing_id_or_text"})
            continue
        if candidate_id in seen_ids:
            dropped.append({"candidate_id": candidate_id, "reason": "duplicate_id"})
            continue
        if len(selected) >= item_limit:
            dropped.append({"candidate_id": candidate_id, "reason": "item_limit"})
            continue

        index = len(selected) + 1
        block = _render_block(candidate, index)
        separator = 2 if selected else 0
        remaining = char_limit - rendered_chars - separator
        if remaining <= 0:
            dropped.append({"candidate_id": candidate_id, "reason": "character_limit"})
            continue
        if len(block) > remaining:
            if selected:
                dropped.append({"candidate_id": candidate_id, "reason": "character_limit"})
                continue
            header = _render_header(candidate.payload, candidate.candidate_id, index)
            text_budget = max(0, remaining - len(header) - 1)
            text = text[:text_budget].rstrip()
            if not text:
                dropped.append({"candidate_id": candidate_id, "reason": "character_limit"})
                continue
            block = f"{header}\n{text}"
            warnings.append("The first evidence item was truncated to the context character limit.")

        selected.append(
            ContextItem(
                item_id=candidate_id,
                role="medical_evidence",
                text=text,
                payload=_provenance_payload(candidate.payload),
                score=candidate.score,
                fused_score=candidate.fused_score,
                rank=candidate.rank,
                matched_metadata={},
                reason="rrf_rank",
            )
        )
        seen_ids.add(candidate_id)
        rendered_chars += len(block) + separator

    if not selected:
        warnings.append("No usable source evidence was available for the prompt.")

    context_text = "\n\n".join(_render_block_from_item(item, index) for index, item in enumerate(selected, 1))
    if len(context_text) > char_limit:
        context_text = context_text[:char_limit].rstrip()

    return PackedContext(
        original_query=normalized_query.original_query,
        items=selected,
        context_text=context_text,
        chunk_items_count=len(selected),
        warnings=warnings,
        debug={
            "limits": {"max_items": item_limit, "max_chars": char_limit},
            "selected_ids": [item.item_id for item in selected],
            "dropped": dropped,
            "ordering": "rrf_rank",
        },
    )


def packed_context_to_response_contexts(packed_context: PackedContext) -> list[dict[str, Any]]:
    """Expose packed evidence in the existing API/prompt context shape."""

    contexts: list[dict[str, Any]] = []
    for item in packed_context.items:
        payload = dict(item.payload)
        payload.update(
            {
                "id": item.item_id,
                "text": item.text,
                "content": item.text,
                "score": item.fused_score if item.fused_score is not None else item.score,
                "rrf_score": item.fused_score,
                "rank": item.rank,
                "retrieval_source": "chunk",
                "context_role": "medical_evidence",
                "context_pack_reason": item.reason,
            }
        )
        contexts.append(payload)
    return contexts


def _render_block(candidate: RetrievedCandidate, index: int) -> str:
    item = ContextItem(
        item_id=candidate.candidate_id,
        role="medical_evidence",
        text=candidate.text.strip(),
        payload=_provenance_payload(candidate.payload),
        score=candidate.score,
        fused_score=candidate.fused_score,
        rank=candidate.rank,
        reason="rrf_rank",
    )
    return _render_block_from_item(item, index)


def _render_block_from_item(item: ContextItem, index: int) -> str:
    return f"{_render_header(item.payload, item.item_id, index)}\n{item.text.strip()}"


def _render_header(payload: dict[str, Any], item_id: str, index: int) -> str:
    source_id = _source_id(payload) or "unknown"
    chunk_id = str(payload.get("chunk_id") or item_id)
    return f"[Evidence {index} | source={source_id} | chunk={chunk_id}]"


def _source_id(payload: dict[str, Any]) -> str:
    return str(
        payload.get("source_id")
        or payload.get("source_path")
        or payload.get("source_file")
        or payload.get("document_id")
        or ""
    ).strip()


def _provenance_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep provenance/metadata without duplicate unbounded document bodies."""

    body_keys = {"text", "content", "page_content"}
    return {key: value for key, value in payload.items() if key not in body_keys}


__all__ = ["pack_context", "packed_context_to_response_contexts"]
