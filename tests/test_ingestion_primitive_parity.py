from __future__ import annotations

from scripts import ingest_knowledge
from src.ingestion.chunking import naive_split
from src.ingestion.filtering import is_noisy_chunk


def test_ingestion_script_reexports_canonical_primitives() -> None:
    assert ingest_knowledge.naive_split is naive_split
    assert ingest_knowledge.is_noisy_chunk is is_noisy_chunk


def test_fixed_width_split_golden_contract() -> None:
    assert naive_split("abcdefghij", size=4, overlap=1) == ["abcd", "defg", "ghij", "j"]


def test_noisy_chunk_decision_golden_contract() -> None:
    assert is_noisy_chunk("...............", "Contents")[0] is True
    assert is_noisy_chunk("42\n43\nclinical context long enough to inspect")[0] is True
    assert is_noisy_chunk("Không dùng adapalene khi mang thai.")[0] is False
    assert is_noisy_chunk("Benzoyl peroxide")[0] is False
    assert is_noisy_chunk("navigation")[0] is True
