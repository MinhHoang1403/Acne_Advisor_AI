from __future__ import annotations

from src.ingestion.chunking import CHUNK_MAX_CHARS, structural_chunks
from src.ingestion.filtering import is_noisy_chunk
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


def test_normalization_removes_only_confirmed_quoted_and_unquoted_nice_artifacts() -> None:
    raw = """# Safety

> © NICE 2026. All rights reserved. Subject to Notice of rights (https://example.test).
> Page 8 of 56 1.4.3
Page 9 of 56

> Clinicians should preserve this normal medical blockquote.

- Keep this list item.
- Keep the next item.

The patient-facing page explains when to seek help.
"""

    normalized = normalize_parsed_text(raw)

    assert "© NICE" not in normalized
    assert "Page 8 of 56" not in normalized
    assert "Page 9 of 56" not in normalized
    assert "> 1.4.3" in normalized
    assert "> Clinicians should preserve this normal medical blockquote." in normalized
    assert "# Safety" in normalized
    assert "- Keep this list item." in normalized
    assert "The patient-facing page explains when to seek help." in normalized


def test_structure_preserving_chunking_keeps_claim_qualifier_and_list_intro() -> None:
    text = (
        "# Treatment\n\n"
        + ("Background sentence. " * 7)
        + "\n\nOnly continue antibiotics in exceptional circumstances; review regularly,"
        + "\n\nand stop the antibiotic as soon as possible."
        + "\n\nUse the following precautions:\n\n"
        + "• start gradually.\n\n• stop if a serious reaction occurs."
    )

    chunks = structural_chunks(text, max_chars=190)

    qualifier_chunk = next(chunk.text for chunk in chunks if "Only continue" in chunk.text)
    assert "and stop the antibiotic as soon as possible" in qualifier_chunk
    list_chunk = next(chunk.text for chunk in chunks if "Use the following precautions:" in chunk.text)
    assert "• start gradually." in list_chunk


def test_structure_preserving_chunking_splits_markdown_table_between_rows() -> None:
    text = """# Options

| Treatment | Important qualifier |
| --- | --- |
| Adapalene | Do not use during pregnancy. |
| Benzoyl peroxide | May bleach hair and fabrics. |
"""

    chunks = structural_chunks(text, max_chars=95)

    assert all(len(chunk.text) <= 95 for chunk in chunks)
    assert any(
        "| Adapalene | Do not use during pregnancy. |" in chunk.text
        for chunk in chunks
    )
    assert any(
        "| Benzoyl peroxide | May bleach hair and fabrics. |" in chunk.text
        for chunk in chunks
    )


def test_parent_list_introduction_is_carried_into_each_child_section() -> None:
    text = """# Systemic treatment

## Antibiotics

Choose one of the following options:

### Doxycycline

- Use the source-backed regimen.

### Lymecycline

- Use the alternative source-backed regimen.
"""

    chunks = structural_chunks(text)
    child_chunks = [chunk for chunk in chunks if len(chunk.section_path) == 3]

    assert len(child_chunks) == 2
    assert all("Choose one of the following options:" in chunk.text for chunk in child_chunks)
