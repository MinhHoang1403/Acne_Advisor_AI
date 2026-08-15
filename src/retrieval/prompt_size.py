"""Đếm chính xác số Unicode characters trong prompt đã render.

Observation tách ``evidence`` và ``non_evidence`` bằng phép cộng độ dài component:
``total = sum(len(component))`` và ``non_evidence = total - evidence``. Đây là
resource telemetry, không ước lượng token, cost hay semantic quality.
"""

from __future__ import annotations

from collections.abc import Iterable
from pydantic import BaseModel, ConfigDict


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PromptComponentSize(_FrozenModel):
    name: str
    characters: int


class PromptSizeObservation(_FrozenModel):
    accounting_mode: str = "character_observation_only"
    total_characters: int
    evidence_characters: int
    non_evidence_characters: int
    components: tuple[PromptComponentSize, ...] = ()


def observe_prompt_components(
    components: Iterable[tuple[str, str]],
) -> PromptSizeObservation:
    """Tổng hợp character count mà không cắt hoặc thay đổi prompt."""
    observations = tuple(
        PromptComponentSize(name=name, characters=len(text)) for name, text in components
    )
    total = sum(item.characters for item in observations)
    evidence = sum(item.characters for item in observations if item.name == "evidence")
    return PromptSizeObservation(
        total_characters=total,
        evidence_characters=evidence,
        non_evidence_characters=total - evidence,
        components=observations,
    )


__all__ = ["PromptComponentSize", "PromptSizeObservation", "observe_prompt_components"]
