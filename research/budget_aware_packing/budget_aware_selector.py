"""Exact rank-weighted budget-aware whole-chunk selector V1.

This research-only selector consumes an already ordered candidate list. It does
not retrieve, rerank, infer facets, inspect gold labels, or alter chunk text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.retrieval.context_packer import _render_block, pack_context
from src.retrieval.contracts import NormalizedQuery, PackedContext, RetrievedCandidate


SELECTOR_NAME = "RANK_WEIGHTED_BUDGET_AWARE_WHOLE_CHUNK_SELECTION_V1"


class SelectionInputError(ValueError):
    """Raised when a candidate pool violates the fixed selector contract."""


@dataclass(frozen=True)
class _Solution:
    utility: int
    positions: tuple[int, ...]
    candidate_ids: tuple[str, ...]
    rendered_chars: int


@dataclass(frozen=True)
class SelectionResult:
    selector_name: str
    packed_context: PackedContext
    total_utility: int
    selected_original_positions: tuple[int, ...]
    selected_candidate_ids: tuple[str, ...]
    candidate_utilities: tuple[int, ...]
    candidates_considered: int
    transitions_considered: int
    peak_state_count: int


def rank_utility(one_based_position: int) -> int:
    if one_based_position < 1:
        raise SelectionInputError("Candidate positions must be one-based positive integers")
    return int(round(1_000_000 / math.log2(one_based_position + 1)))


def _validate_candidates(candidates: list[RetrievedCandidate]) -> None:
    seen: set[str] = set()
    for position, candidate in enumerate(candidates, 1):
        candidate_id = candidate.candidate_id.strip()
        if not candidate_id or not candidate.text.strip():
            raise SelectionInputError(
                f"Candidate at position {position} is missing a stable ID or text"
            )
        if candidate_id in seen:
            raise SelectionInputError(f"Duplicate candidate ID: {candidate_id}")
        seen.add(candidate_id)


def _is_better(candidate: _Solution, incumbent: _Solution | None) -> bool:
    if incumbent is None:
        return True
    if candidate.utility != incumbent.utility:
        return candidate.utility > incumbent.utility
    if candidate.positions != incumbent.positions:
        return candidate.positions < incumbent.positions
    if candidate.rendered_chars != incumbent.rendered_chars:
        return candidate.rendered_chars < incumbent.rendered_chars
    return candidate.candidate_ids < incumbent.candidate_ids


def select_budget_aware(
    normalized_query: NormalizedQuery,
    candidates: list[RetrievedCandidate],
    *,
    max_items: int = 9,
    max_chars: int = 7000,
) -> SelectionResult:
    """Return the exact optimum for the preregistered V1 objective."""

    if max_items < 1:
        raise SelectionInputError("max_items must be positive")
    if max_chars < 1:
        raise SelectionInputError("max_chars must be positive")
    _validate_candidates(candidates)

    empty = _Solution(utility=0, positions=(), candidate_ids=(), rendered_chars=0)
    states: dict[tuple[int, int], _Solution] = {(0, 0): empty}
    transitions_considered = 0
    peak_state_count = 1

    for original_position, candidate in enumerate(candidates, 1):
        utility = rank_utility(original_position)
        snapshot = list(states.items())
        updates: dict[tuple[int, int], _Solution] = {}
        for (selected_count, used_chars), solution in snapshot:
            new_count = selected_count + 1
            if new_count > max_items:
                continue
            transitions_considered += 1
            block_chars = len(_render_block(candidate, new_count))
            new_chars = used_chars + block_chars + (2 if selected_count else 0)
            if new_chars > max_chars:
                continue
            proposed = _Solution(
                utility=solution.utility + utility,
                positions=(*solution.positions, original_position),
                candidate_ids=(*solution.candidate_ids, candidate.candidate_id.strip()),
                rendered_chars=new_chars,
            )
            key = (new_count, new_chars)
            incumbent = updates.get(key) or states.get(key)
            if _is_better(proposed, incumbent):
                updates[key] = proposed
        for key, proposed in updates.items():
            if _is_better(proposed, states.get(key)):
                states[key] = proposed
        peak_state_count = max(peak_state_count, len(states))

    optimum: _Solution | None = None
    for solution in states.values():
        if _is_better(solution, optimum):
            optimum = solution
    assert optimum is not None

    selected = [candidates[position - 1] for position in optimum.positions]
    packed = pack_context(
        normalized_query,
        selected,
        max_items=max_items,
        max_chars=max_chars,
    )
    packed_ids = tuple(item.item_id for item in packed.items)
    if packed_ids != optimum.candidate_ids:
        raise AssertionError("Exact solver selected a subset that production renderer rejected")
    if len(packed.context_text) != optimum.rendered_chars:
        raise AssertionError("DP character accounting disagrees with production rendering")
    if len(packed.items) > max_items or len(packed.context_text) > max_chars:
        raise AssertionError("Experimental selector exceeded a fixed resource budget")

    return SelectionResult(
        selector_name=SELECTOR_NAME,
        packed_context=packed,
        total_utility=optimum.utility,
        selected_original_positions=optimum.positions,
        selected_candidate_ids=optimum.candidate_ids,
        candidate_utilities=tuple(rank_utility(position) for position in optimum.positions),
        candidates_considered=len(candidates),
        transitions_considered=transitions_considered,
        peak_state_count=peak_state_count,
    )


__all__ = [
    "SELECTOR_NAME",
    "SelectionInputError",
    "SelectionResult",
    "rank_utility",
    "select_budget_aware",
]
