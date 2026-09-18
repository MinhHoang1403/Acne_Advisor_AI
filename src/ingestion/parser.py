"""Parse source thành intermediate representation có content identity.

Markdown/JSON được đọc trực tiếp; PDF được gửi tới LlamaParse khi verified cache
không còn hợp lệ. Output được normalize rồi lưu theo source hash và parser
contract, nên cache chỉ được reuse khi source và contract cùng khớp.

Module không chunk, embed hay index. Muốn đổi provider/config PDF bắt đầu tại
``PARSER_CONFIGURATION`` và ``_parse_pdf``; muốn đổi normalization đọc
``src/ingestion/normalization.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.ingestion.json_loader import load_web_json_documents
from src.ingestion.normalization import NORMALIZATION_CONTRACT_ID, normalize_parsed_text
from src.ingestion.provenance import sha256_text
from src.ingestion.source_manifest import CanonicalSource


PARSER_CONTRACT_ID = "llamaparse_0_6_94_markdown_page_ranges_or_direct_utf8"
PARSER_CONFIGURATION = {
    "pdf": {
        "implementation": "llama-parse",
        "version": "0.6.94",
        "result_type": "markdown",
        "instruction": "Preserve headings, tables, lists, clinical terminology and page order.",
    },
    "markdown": {"implementation": "python_utf8_direct"},
    "json": {"implementation": "python_json_direct", "record_field": "raw_text"},
}
_MARKDOWN_PAGE_MARKER_RE = re.compile(
    r"^\s*>?\s*Page\s+(?P<page>\d{1,4})\s+of\s+\d{1,4}"
    r"(?P<continuation>\s+.*)?$",
    re.IGNORECASE,
)
_REUSABLE_CACHE_CONTRACTS = {
    (
        "llamaparse_0_6_94_markdown_or_direct_utf8",
        "unicode_nfc_lf_exact_artifacts",
    ),
    (
        "llamaparse_0_6_94_markdown_page_units_or_direct_utf8",
        "unicode_nfc_lf_confirmed_nice_artifacts",
    ),
}


@dataclass(frozen=True)
class ParsedUnit:
    locator: str
    text: str
    page_start: int = 0
    page_end: int = 0
    source_url: str = ""


@dataclass(frozen=True)
class ParsedArtifact:
    source_id: str
    source_hash: str
    parser_contract_id: str
    normalization_contract_id: str
    parsed_output_hash: str
    normalized_output_hash: str
    units: tuple[ParsedUnit, ...]


def artifact_path(cache_root: Path, source: CanonicalSource) -> Path:
    """Tạo cache path từ source hash và parser-configuration hash."""

    contract_hash = sha256_text(
        json.dumps(PARSER_CONFIGURATION, sort_keys=True, separators=(",", ":"))
    )[:16]
    return cache_root / source.source_id / f"{source.sha256}.{contract_hash}.json"


def load_parsed_artifact(path: Path, source: CanonicalSource) -> ParsedArtifact | None:
    """Chỉ trả cache khi identity, contract và normalized-output hash đều hợp lệ."""

    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        units = tuple(ParsedUnit(**unit) for unit in raw["units"])
        artifact = ParsedArtifact(
            source_id=raw["source_id"],
            source_hash=raw["source_hash"],
            parser_contract_id=raw["parser_contract_id"],
            normalization_contract_id=raw["normalization_contract_id"],
            parsed_output_hash=raw["parsed_output_hash"],
            normalized_output_hash=raw["normalized_output_hash"],
            units=units,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if (
        artifact.source_id != source.source_id
        or artifact.source_hash != source.sha256
        or artifact.parser_contract_id != PARSER_CONTRACT_ID
        or artifact.normalization_contract_id != NORMALIZATION_CONTRACT_ID
        or not artifact.units
    ):
        return None
    normalized_joined = _joined(unit.text for unit in artifact.units)
    if sha256_text(normalized_joined) != artifact.normalized_output_hash:
        return None
    return artifact


def save_parsed_artifact(path: Path, artifact: ParsedArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(artifact), "units": [asdict(unit) for unit in artifact.units]}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


async def load_or_parse_source(
    source: CanonicalSource,
    *,
    source_dir: Path,
    cache_root: Path,
    llama_cloud_api_key: str = "",
) -> tuple[ParsedArtifact, bool]:
    """Trả parsed artifact đã kiểm tra cùng cờ cho biết có cache hit hay không."""

    cache_path = artifact_path(cache_root, source)
    cached = load_parsed_artifact(cache_path, source)
    if cached is not None:
        return cached, True

    source_path = source_dir / source.local_filename
    suffix = source_path.suffix.casefold()
    if suffix not in {".md", ".markdown"}:
        migrated = _migrate_compatible_cached_artifact(cache_path, source)
        if migrated is not None:
            save_parsed_artifact(cache_path, migrated)
            return migrated, True
    if suffix == ".json":
        raw_units = _parse_json(source_path)
    elif suffix in {".md", ".markdown"}:
        raw_units = _parse_markdown(source_path)
    elif suffix == ".pdf":
        raw_units = await _parse_pdf(source_path, api_key=llama_cloud_api_key)
    else:
        raise ValueError(f"Unsupported canonical source media: {source.local_filename}")

    normalized_units_list: list[ParsedUnit] = []
    for unit in raw_units:
        normalized_text = normalize_parsed_text(unit.text)
        if not normalized_text:
            continue
        normalized_units_list.append(
            ParsedUnit(
                locator=unit.locator,
                text=normalized_text,
                page_start=unit.page_start,
                page_end=unit.page_end,
                source_url=unit.source_url,
            )
        )
    normalized_units = tuple(normalized_units_list)
    if not normalized_units:
        raise ValueError(f"Parser returned no content for {source.source_id}")
    artifact = ParsedArtifact(
        source_id=source.source_id,
        source_hash=source.sha256,
        parser_contract_id=PARSER_CONTRACT_ID,
        normalization_contract_id=NORMALIZATION_CONTRACT_ID,
        parsed_output_hash=sha256_text(_joined(unit.text for unit in raw_units)),
        normalized_output_hash=sha256_text(_joined(unit.text for unit in normalized_units)),
        units=normalized_units,
    )
    save_parsed_artifact(cache_path, artifact)
    return artifact, False


def _migrate_compatible_cached_artifact(
    path: Path,
    source: CanonicalSource,
) -> ParsedArtifact | None:
    """Reuse parser output when only the Markdown page parser changed.

    PDF and JSON parser behavior is unchanged. Re-normalizing their verified
    cached units avoids unnecessary provider calls while still writing the
    current contract and content hash. Markdown is deliberately excluded
    because its unit/page structure changed.
    """

    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        units = tuple(ParsedUnit(**unit) for unit in raw["units"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if (
        raw.get("source_id") != source.source_id
        or raw.get("source_hash") != source.sha256
        or (raw.get("parser_contract_id"), raw.get("normalization_contract_id"))
        not in _REUSABLE_CACHE_CONTRACTS
        or not units
        or sha256_text(_joined(unit.text for unit in units))
        != raw.get("normalized_output_hash")
    ):
        return None

    migrated_units_list: list[ParsedUnit] = []
    for unit in units:
        normalized_text = normalize_parsed_text(unit.text)
        if normalized_text:
            migrated_units_list.append(
                ParsedUnit(
                    locator=unit.locator,
                    text=normalized_text,
                    page_start=unit.page_start,
                    page_end=unit.page_end,
                    source_url=unit.source_url,
                )
            )
    migrated_units = tuple(migrated_units_list)
    return ParsedArtifact(
        source_id=source.source_id,
        source_hash=source.sha256,
        parser_contract_id=PARSER_CONTRACT_ID,
        normalization_contract_id=NORMALIZATION_CONTRACT_ID,
        parsed_output_hash=str(raw.get("parsed_output_hash") or ""),
        normalized_output_hash=sha256_text(_joined(unit.text for unit in migrated_units)),
        units=migrated_units,
    )


def _parse_markdown(path: Path) -> tuple[ParsedUnit, ...]:
    text = path.read_text(encoding="utf-8")
    marker = "Markdown Content:"
    if marker in text[:500]:
        text = text.split(marker, 1)[1].lstrip()
    return _markdown_page_units(text)


def _markdown_page_units(text: str) -> tuple[ParsedUnit, ...]:
    """Split only on explicit rendered PDF page markers.

    The NICE transport places a page marker before the following page content.
    Content before the first marker keeps page ``0`` because no reliable page
    locator exists for that prelude.
    """

    units: list[ParsedUnit] = []
    current_page = 0
    lines: list[str] = []

    def flush() -> None:
        content = "\n".join(lines).strip()
        if not content:
            lines.clear()
            return
        locator = f"page:{current_page}" if current_page else "document:prelude"
        units.append(
            ParsedUnit(
                locator=locator,
                text=content,
                page_start=current_page,
                page_end=current_page,
            )
        )
        lines.clear()

    for line in text.splitlines():
        marker = _MARKDOWN_PAGE_MARKER_RE.fullmatch(line)
        if marker is None:
            lines.append(line)
            continue
        flush()
        current_page = int(marker.group("page"))
        continuation = str(marker.group("continuation") or "").strip()
        if continuation:
            lines.append(f"> {continuation}")
    flush()
    return _join_cross_page_continuations(tuple(units))


def _join_cross_page_continuations(
    units: tuple[ParsedUnit, ...],
) -> tuple[ParsedUnit, ...]:
    """Attach explicit next-page continuations to the preceding page range."""

    if not units:
        return ()
    joined: list[ParsedUnit] = [units[0]]
    for current in units[1:]:
        previous = joined[-1]
        previous_text = normalize_parsed_text(previous.text).rstrip()
        pages_are_adjacent = (
            previous.page_end > 0
            and current.page_start > 0
            and current.page_start == previous.page_end + 1
        )
        if not pages_are_adjacent or not _needs_next_page_continuation(previous_text):
            joined.append(current)
            continue

        blocks = [
            block.strip()
            for block in re.split(r"\n\s*\n", current.text)
            if block.strip()
        ]
        if not blocks or _is_structural_start(blocks[0]):
            joined.append(current)
            continue

        moved = [blocks.pop(0)]
        while blocks and _is_list_start(moved[-1]) and _is_list_start(blocks[0]):
            moved.append(blocks.pop(0))
        moved_text = "\n\n".join(moved)
        joined[-1] = ParsedUnit(
            locator=previous.locator,
            text=f"{previous.text.rstrip()}\n\n{moved_text}",
            page_start=previous.page_start,
            page_end=current.page_end,
            source_url=previous.source_url,
        )
        if blocks:
            joined.append(
                ParsedUnit(
                    locator=current.locator,
                    text="\n\n".join(blocks),
                    page_start=current.page_start,
                    page_end=current.page_end,
                    source_url=current.source_url,
                )
            )
    return tuple(joined)


def _needs_next_page_continuation(text: str) -> bool:
    return bool(text) and text[-1] not in ".!?"


def _is_structural_start(block: str) -> bool:
    first = block.splitlines()[0].strip()
    return bool(
        first.startswith("#")
        or re.fullmatch(r">\s*\d+(?:\.\d+)+", first)
    )


def _is_list_start(block: str) -> bool:
    first = block.splitlines()[0].lstrip("> ")
    return bool(re.match(r"^(?:[-*•－]|\d+[.)])\s+", first))


def _parse_json(path: Path) -> tuple[ParsedUnit, ...]:
    documents = load_web_json_documents(path)
    return tuple(
        ParsedUnit(
            locator=(str(item["metadata"].get("source_url") or f"record:{index}")),
            text=item["text"],
            source_url=str(item["metadata"].get("source_url") or ""),
        )
        for index, item in enumerate(documents)
    )


async def _parse_pdf(path: Path, *, api_key: str) -> tuple[ParsedUnit, ...]:
    if not api_key.strip():
        raise RuntimeError(
            f"Parsed cache missing for {path.name}; LLAMA_CLOUD_API_KEY is required."
        )
    from llama_parse import LlamaParse  # type: ignore[import-not-found]

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        parsing_instruction=PARSER_CONFIGURATION["pdf"]["instruction"],
        verbose=False,
    )
    documents = await parser.aload_data(str(path))
    units: list[ParsedUnit] = []
    for index, document in enumerate(documents or []):
        text = str(getattr(document, "text", "") or "").strip()
        metadata: dict[str, Any] = getattr(document, "metadata", {}) or {}
        if not text:
            continue
        page = int(metadata.get("page_label") or metadata.get("page") or 0)
        units.append(ParsedUnit(locator=f"parser_document:{index}", text=text, page_start=page, page_end=page))
    return tuple(units)


def _joined(values: Any) -> str:
    return "\n\n".join(str(value) for value in values)


__all__ = [
    "PARSER_CONFIGURATION",
    "PARSER_CONTRACT_ID",
    "ParsedArtifact",
    "ParsedUnit",
    "artifact_path",
    "load_or_parse_source",
    "load_parsed_artifact",
    "save_parsed_artifact",
]
