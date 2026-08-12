"""Deterministic R7 serialization for already-selected V5 evidence."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.retrieval.contracts import (
    ContextItem,
    ContextPackTrace,
    NormalizedQuery,
    PackedContext,
    RetrievedCandidate,
)

from src.retrieval.v5_contracts import (
    CandidateDropV5,
    DropReasonV5,
    EvidencePackingStatusV5,
    PackedEvidenceV5,
    SelectedEvidenceV5,
)


_APPROXIMATE_TOKEN_CHARACTERS = 4
_MIN_CLIPPED_BLOCK_CHARACTERS = 96
_CLIP_SUFFIX = "\n...[truncated]"


def pack_selected_evidence_v5(
    *,
    selected_evidence: Iterable[SelectedEvidenceV5],
    max_items: int,
    max_characters: int,
    max_tokens: int | None = None,
) -> PackedEvidenceV5:
    """Serialize selector output within finite budgets without semantic reranking.

    The selector order is retained for emitted blocks. Critical records are
    reserved before non-critical records so a later safety record cannot be
    displaced by redundant earlier evidence. Any omission or clipping is
    returned explicitly; this function never retries retrieval.
    """

    if max_items < 1:
        raise ValueError("max_items must be positive")
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    effective_max_tokens = max_tokens if max_tokens is not None else max(1, max_characters // 4)
    if effective_max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    records = tuple(
        _EvidenceRecord(index=index, selected=item, block=_render_block(index, item))
        for index, item in enumerate(selected_evidence, start=1)
    )
    included: dict[int, str] = {}
    omitted_indexes: list[int] = []
    clipped_indexes: list[int] = []
    drops: list[CandidateDropV5] = []

    critical_records = tuple(record for record in records if record.selected.critical)
    for record in critical_records:
        if not _has_evidence_text(record.selected):
            _omit(record, DropReasonV5.PACKER_EMPTY_TEXT, omitted_indexes, drops)
            continue
        if len(included) >= max_items or not _fits(
            _compose_blocks(included.values(), record.block),
            max_characters=max_characters,
            max_tokens=effective_max_tokens,
        ):
            _omit(record, DropReasonV5.PACKER_BUDGET_REMOVED, omitted_indexes, drops)
            continue
        included[record.index] = record.block

    critical_evidence_preserved = all(record.index in included for record in critical_records)
    if critical_evidence_preserved:
        for record in records:
            if record.selected.critical:
                continue
            if not _has_evidence_text(record.selected):
                _omit(record, DropReasonV5.PACKER_EMPTY_TEXT, omitted_indexes, drops)
                continue
            if len(included) >= max_items:
                _omit(record, DropReasonV5.PACKER_BUDGET_REMOVED, omitted_indexes, drops)
                continue
            full_text = _compose_blocks(included.values(), record.block)
            if _fits(
                full_text,
                max_characters=max_characters,
                max_tokens=effective_max_tokens,
            ):
                included[record.index] = record.block
                continue

            clipped = _clip_to_remaining_budget(
                included_blocks=included.values(),
                block=record.block,
                max_characters=max_characters,
                max_tokens=effective_max_tokens,
            )
            if clipped:
                included[record.index] = clipped
                clipped_indexes.append(record.index)
                drops.append(
                    CandidateDropV5(
                        candidate_id=_evidence_id(record.selected),
                        reason=DropReasonV5.PACKER_CLIPPED,
                    )
                )
            else:
                _omit(record, DropReasonV5.PACKER_BUDGET_REMOVED, omitted_indexes, drops)
    else:
        # A critical record could not be represented in full. Do not fill the
        # remaining budget with non-critical text that could mask this state.
        for record in records:
            if record.selected.critical:
                continue
            _omit(record, DropReasonV5.PACKER_BUDGET_REMOVED, omitted_indexes, drops)

    ordered_indexes = sorted(included)
    rendered_blocks = tuple(included[index] for index in ordered_indexes)
    context_text = _compose_blocks(rendered_blocks)
    assert _fits(
        context_text,
        max_characters=max_characters,
        max_tokens=effective_max_tokens,
    )
    selected_ids = tuple(_evidence_id(records[index - 1].selected) for index in ordered_indexes)
    critical_ids = tuple(
        _evidence_id(records[index - 1].selected)
        for index in ordered_indexes
        if records[index - 1].selected.critical
    )
    omitted_ids = tuple(
        _evidence_id(records[index - 1].selected)
        for index in _dedupe_indexes(omitted_indexes)
    )
    clipped_ids = tuple(
        _evidence_id(records[index - 1].selected)
        for index in _dedupe_indexes(clipped_indexes)
    )
    if not records:
        status = EvidencePackingStatusV5.INSUFFICIENT
    elif not critical_evidence_preserved:
        status = EvidencePackingStatusV5.CRITICAL_EVIDENCE_OVERFLOW
    elif omitted_ids or clipped_ids:
        status = EvidencePackingStatusV5.OVERFLOW
    else:
        status = EvidencePackingStatusV5.SUFFICIENT

    return PackedEvidenceV5(
        selected_evidence_ids=selected_ids,
        rendered_blocks=rendered_blocks,
        context_text=context_text,
        clipped_evidence_ids=clipped_ids,
        omitted_evidence_ids=omitted_ids,
        drops=tuple(drops),
        character_count=len(context_text),
        max_characters=max_characters,
        token_count=_estimate_tokens(context_text),
        max_tokens=effective_max_tokens,
        max_items=max_items,
        source_paths=_source_paths(records, ordered_indexes),
        critical_evidence_ids=critical_ids,
        critical_evidence_preserved=critical_evidence_preserved,
        status=status,
    )


def packed_evidence_to_legacy_context_v5(
    *,
    normalized_query: NormalizedQuery,
    selected_evidence: Iterable[SelectedEvidenceV5],
    packed_evidence: PackedEvidenceV5,
    candidates: Iterable[RetrievedCandidate],
) -> PackedContext:
    """Adapt the V5 serialized output for legacy prompt/result consumers.

    The adapter follows the V5 included-ID order and uses the V5 rendered
    blocks as item text. It intentionally does not call the legacy packer,
    which would apply a second relevance/intent selection pass.
    """

    evidence_by_id = {
        _evidence_id(item): item
        for item in selected_evidence
    }
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    items: list[ContextItem] = []
    for evidence_id, block in zip(
        packed_evidence.selected_evidence_ids,
        packed_evidence.rendered_blocks,
        strict=True,
    ):
        selected = evidence_by_id[evidence_id]
        candidate = candidates_by_id.get(evidence_id)
        source = candidate.source if candidate is not None else "chunk"
        payload: dict[str, Any] = dict(candidate.payload) if candidate is not None else {}
        provenance = selected.evidence.candidate.candidate.provenance
        payload.setdefault("chunk_id", provenance.chunk_id or evidence_id)
        if provenance.document_id:
            payload.setdefault("document_id", provenance.document_id)
        if provenance.source_path:
            payload.setdefault("source_path", provenance.source_path)
            payload.setdefault("source_file", provenance.source_path)
        payload["text"] = block
        items.append(
            ContextItem(
                item_id=evidence_id,
                source=source,
                role=_legacy_role(selected),
                text=block,
                payload=payload,
                score=(candidate.score if candidate is not None else None),
                fused_score=(candidate.fused_score if candidate is not None else None),
                rank=(candidate.rank if candidate is not None else None),
                matched_metadata=(dict(candidate.matched_metadata) if candidate is not None else {}),
                reason=f"v5_packer: {selected.selection_reason}; roles={','.join(selected.roles)}",
            )
        )
    warnings = []
    if packed_evidence.status != EvidencePackingStatusV5.SUFFICIENT:
        warnings.append(f"V5 evidence packing status: {packed_evidence.status.value}.")
    trace = ContextPackTrace(
        intent=normalized_query.intent,
        selected_chunk_ids=list(packed_evidence.selected_evidence_ids),
        selection_reasons=[item.reason for item in items],
        dropped_candidates=[
            {"candidate_id": drop.candidate_id, "reason": drop.reason.value}
            for drop in packed_evidence.drops
        ],
        warnings=warnings,
    )
    return PackedContext(
        original_query=normalized_query.original_query,
        intent=normalized_query.intent,
        items=items,
        context_text=packed_evidence.context_text,
        entity_items_count=sum(1 for item in items if item.source == "entity"),
        chunk_items_count=sum(1 for item in items if item.source == "chunk"),
        warnings=warnings,
        debug={"pack_trace": trace.model_dump(mode="json"), "v5_packed_evidence": packed_evidence.model_dump(mode="json")},
    )


class _EvidenceRecord:
    def __init__(self, *, index: int, selected: SelectedEvidenceV5, block: str) -> None:
        self.index = index
        self.selected = selected
        self.block = block


def _render_block(index: int, selected: SelectedEvidenceV5) -> str:
    candidate = selected.evidence.candidate.candidate
    provenance = candidate.provenance
    source = provenance.source_path or provenance.document_id or "unknown"
    roles = ", ".join(selected.roles) or "supporting"
    return "\n".join(
        (
            f"[EVIDENCE #{index}]",
            f"Source: {source}",
            f"Roles: {roles}",
            f"Text: {candidate.text.strip()}",
        )
    )


def _has_evidence_text(selected: SelectedEvidenceV5) -> bool:
    return bool(selected.evidence.candidate.candidate.text.strip())


def _legacy_role(selected: SelectedEvidenceV5) -> str:
    if "critical" in selected.roles or "safety" in selected.roles:
        return "safety_context"
    if "primary" in selected.roles:
        return "primary_evidence"
    if "treatment" in selected.roles:
        return "treatment_context"
    return "supporting_evidence"


def _evidence_id(selected: SelectedEvidenceV5) -> str:
    return selected.evidence.candidate.candidate.candidate_id


def _omit(
    record: _EvidenceRecord,
    reason: DropReasonV5,
    omitted_indexes: list[int],
    drops: list[CandidateDropV5],
) -> None:
    omitted_indexes.append(record.index)
    drops.append(CandidateDropV5(candidate_id=_evidence_id(record.selected), reason=reason))


def _compose_blocks(existing_blocks: Iterable[str], next_block: str | None = None) -> str:
    blocks = [*existing_blocks]
    if next_block is not None:
        blocks.append(next_block)
    return "\n\n".join(blocks)


def _fits(text: str, *, max_characters: int, max_tokens: int) -> bool:
    return len(text) <= max_characters and _estimate_tokens(text) <= max_tokens


def _estimate_tokens(text: str) -> int:
    return (len(text) + _APPROXIMATE_TOKEN_CHARACTERS - 1) // _APPROXIMATE_TOKEN_CHARACTERS


def _clip_to_remaining_budget(
    *,
    included_blocks: Iterable[str],
    block: str,
    max_characters: int,
    max_tokens: int,
) -> str:
    existing = _compose_blocks(included_blocks)
    separator_size = 2 if existing else 0
    character_room = max_characters - len(existing) - separator_size
    token_room = max_tokens - _estimate_tokens(existing)
    allowed = min(character_room, token_room * _APPROXIMATE_TOKEN_CHARACTERS)
    if allowed < _MIN_CLIPPED_BLOCK_CHARACTERS:
        return ""
    clipped = block[: allowed - len(_CLIP_SUFFIX)].rstrip() + _CLIP_SUFFIX
    candidate = _compose_blocks((existing,) if existing else (), clipped)
    return clipped if _fits(candidate, max_characters=max_characters, max_tokens=max_tokens) else ""


def _source_paths(records: tuple[_EvidenceRecord, ...], included_indexes: list[int]) -> tuple[str, ...]:
    paths: list[str] = []
    for index in included_indexes:
        provenance = records[index - 1].selected.evidence.candidate.candidate.provenance
        source = provenance.source_path or provenance.document_id
        if source and source not in paths:
            paths.append(source)
    return tuple(paths)


def _dedupe_indexes(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(values))


__all__ = ["pack_selected_evidence_v5", "packed_evidence_to_legacy_context_v5"]
