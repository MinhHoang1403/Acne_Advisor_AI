from __future__ import annotations

from pathlib import Path

from src.ingestion.normalization import normalize_parsed_text
from src.ingestion.parser import _parse_markdown


def test_markdown_page_markers_become_portable_page_units(tmp_path: Path) -> None:
    path = tmp_path / "source.md"
    path.write_text(
        """Cover without a reliable marker.

> © NICE 2026. All rights reserved. Subject to Notice of rights (https://example.test).
> Page 8 of 56 1.4.3

# Referral

Consider referral when acne is leading to scarring.

> © NICE 2026. All rights reserved. Subject to Notice of rights (https://example.test).
> Page 9 of 56

# Review

Review response and adverse effects.
""",
        encoding="utf-8",
    )

    units = _parse_markdown(path)

    assert [(unit.locator, unit.page_start, unit.page_end) for unit in units] == [
        ("document:prelude", 0, 0),
        ("page:8", 8, 8),
        ("page:9", 9, 9),
    ]
    page_eight = normalize_parsed_text(units[1].text)
    assert page_eight.startswith("> 1.4.3")
    assert "# Referral" in page_eight
    assert "© NICE" not in page_eight


def test_markdown_without_page_markers_reports_page_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "flat.md"
    path.write_text("# Heading\n\nSource text.", encoding="utf-8")

    units = _parse_markdown(path)

    assert len(units) == 1
    assert units[0].locator == "document:prelude"
    assert units[0].page_start == units[0].page_end == 0


def test_claim_qualifier_crossing_explicit_page_boundary_stays_together(
    tmp_path: Path,
) -> None:
    path = tmp_path / "continued.md"
    path.write_text(
        """> Page 16 of 56
Review the treatment regularly,

> Page 17 of 56
and stop it as soon as appropriate.

> 1.5.14

Start the next recommendation here.
""",
        encoding="utf-8",
    )

    units = _parse_markdown(path)

    assert units[0].page_start == 16
    assert units[0].page_end == 17
    assert "and stop it as soon as appropriate." in units[0].text
    assert units[1].locator == "page:17"
    assert units[1].text.startswith("> 1.5.14")
