from __future__ import annotations

from src.ingestion.chunking import CHUNK_MAX_CHARS, naive_split, structural_chunks
from src.ingestion.filtering import deduplicate_chunks, is_noisy_chunk
from src.ingestion.normalization import normalize_parsed_text


def test_structure_first_chunking_preserves_heading_and_has_no_overlap() -> None:
    text = "# Treatment\n\n" + ("Sentence about acne. " * 180) + "\n\n## Safety\n\nDo not use in pregnancy."
    chunks = structural_chunks(text)

    assert all(len(chunk.text) <= CHUNK_MAX_CHARS for chunk in chunks)
    assert chunks[0].section_path == ("Treatment",)
    assert chunks[-1].section_path == ("Treatment", "Safety")
    assert chunks[-1].text == "Do not use in pregnancy."
    assert sum(chunk.text.count("Sentence about acne.") for chunk in chunks) == 180


def test_short_medical_content_is_never_generic_noise() -> None:
    for text in (
        "Adults, adolescents and children aged 9 years and over.",
        "Overall Relative Risk",
        "Chống chỉ định trong thai kỳ.",
        "Trứng cá tối cấp là thể nặng.",
    ):
        assert is_noisy_chunk(text) == (False, "")


def test_only_proven_artifacts_are_rejected() -> None:
    assert is_noisy_chunk("   ") == (True, "empty")
    assert is_noisy_chunk("Page 12 of 56") == (True, "page_number_only")
    assert is_noisy_chunk("Treatment .......... 10", "Contents") == (True, "toc_dot_leaders")
    assert is_noisy_chunk("© notice plus clinically meaningful text") == (False, "")


def test_normalization_is_conservative_and_deterministic() -> None:
    raw = "A\u0301p dụng\r\nPage 2 of 56\r\n\r\n\r\nKhông đổi nghĩa.  \r\n"
    assert normalize_parsed_text(raw) == "Áp dụng\n\nKhông đổi nghĩa."
    assert normalize_parsed_text(normalize_parsed_text(raw)) == normalize_parsed_text(raw)


def test_exact_deduplication_keeps_first_occurrence() -> None:
    assert deduplicate_chunks(["a", "b", "a"]) == (["a", "b"], [2])


def test_fixed_width_split_primitive_remains_deterministic() -> None:
    assert naive_split("abcdefghij", size=4, overlap=1) == ["abcd", "defg", "ghij", "j"]
