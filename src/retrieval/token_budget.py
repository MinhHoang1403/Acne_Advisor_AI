"""Deterministic token-budget observations without provider dependencies."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict


APPROXIMATE_TOKEN_CHARACTERS = 4
APPROXIMATE_TOKEN_COUNT_MODE = "approximate_chars_div_4"


def estimate_tokens_approximately(text: str) -> int:
    """Return the documented deterministic estimate ``ceil(chars / 4)``."""

    return (len(text) + APPROXIMATE_TOKEN_CHARACTERS - 1) // APPROXIMATE_TOKEN_CHARACTERS


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PromptComponentBudgetObservation(_FrozenModel):
    """Size-only observation for one rendered prompt component."""

    name: str
    characters: int
    estimated_tokens: int


class FullPromptBudgetObservation(_FrozenModel):
    """Complete rendered prompt accounting when no total limit is configured."""

    accounting_mode: str = "observation_only"
    token_count_mode: str = APPROXIMATE_TOKEN_COUNT_MODE
    total_characters: int
    estimated_total_tokens: int
    evidence_characters: int
    evidence_estimated_tokens: int
    graph_characters: int
    graph_estimated_tokens: int
    non_evidence_characters: int
    non_evidence_estimated_tokens: int
    enforced_max_tokens: int | None = None
    overflow_status: str = "NOT_EVALUATED_NO_TOTAL_LIMIT"
    components: tuple[PromptComponentBudgetObservation, ...] = ()


def observe_prompt_components(
    components: Iterable[tuple[str, str]],
) -> FullPromptBudgetObservation:
    """Account for every rendered component without retaining prompt contents."""

    rendered_components = tuple(components)
    observations = tuple(
        PromptComponentBudgetObservation(
            name=name,
            characters=len(text),
            estimated_tokens=estimate_tokens_approximately(text),
        )
        for name, text in rendered_components
    )
    total_characters = sum(item.characters for item in observations)
    evidence_characters = sum(
        item.characters for item in observations if item.name == "evidence"
    )
    graph_characters = sum(item.characters for item in observations if item.name == "graph")
    return FullPromptBudgetObservation(
        total_characters=total_characters,
        estimated_total_tokens=estimate_tokens_approximately(
            "".join(text for _, text in rendered_components)
        ),
        evidence_characters=evidence_characters,
        evidence_estimated_tokens=estimate_tokens_approximately(
            "".join(text for name, text in rendered_components if name == "evidence")
        ),
        graph_characters=graph_characters,
        graph_estimated_tokens=estimate_tokens_approximately(
            "".join(text for name, text in rendered_components if name == "graph")
        ),
        non_evidence_characters=total_characters - evidence_characters,
        non_evidence_estimated_tokens=estimate_tokens_approximately(
            "".join(text for name, text in rendered_components if name != "evidence")
        ),
        components=observations,
    )


__all__ = [
    "APPROXIMATE_TOKEN_CHARACTERS",
    "APPROXIMATE_TOKEN_COUNT_MODE",
    "FullPromptBudgetObservation",
    "PromptComponentBudgetObservation",
    "estimate_tokens_approximately",
    "observe_prompt_components",
]
