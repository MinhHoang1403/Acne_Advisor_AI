"""Content-addressed parsed intermediate representation for Phase 1."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.ingestion.json_loader import load_web_json_documents
from src.ingestion.normalization import NORMALIZATION_CONTRACT_ID, normalize_parsed_text
from src.ingestion.provenance import sha256_text
from src.ingestion.source_manifest import CanonicalSource


PARSER_CONTRACT_ID = "llamaparse_0_6_94_markdown_or_direct_utf8"
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
    contract_hash = sha256_text(
        json.dumps(PARSER_CONFIGURATION, sort_keys=True, separators=(",", ":"))
    )[:16]
    return cache_root / source.source_id / f"{source.sha256}.{contract_hash}.json"


def load_parsed_artifact(path: Path, source: CanonicalSource) -> ParsedArtifact | None:
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
    """Return a verified parsed artifact and whether it was a cache hit."""

    cache_path = artifact_path(cache_root, source)
    cached = load_parsed_artifact(cache_path, source)
    if cached is not None:
        return cached, True

    source_path = source_dir / source.local_filename
    suffix = source_path.suffix.casefold()
    if suffix == ".json":
        raw_units = _parse_json(source_path)
    elif suffix in {".md", ".markdown"}:
        raw_units = _parse_markdown(source_path)
    elif suffix == ".pdf":
        raw_units = await _parse_pdf(source_path, api_key=llama_cloud_api_key)
    else:
        raise ValueError(f"Unsupported canonical source media: {source.local_filename}")

    normalized_units = tuple(
        ParsedUnit(
            locator=unit.locator,
            text=normalize_parsed_text(unit.text),
            page_start=unit.page_start,
            page_end=unit.page_end,
            source_url=unit.source_url,
        )
        for unit in raw_units
        if normalize_parsed_text(unit.text)
    )
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


def _parse_markdown(path: Path) -> tuple[ParsedUnit, ...]:
    text = path.read_text(encoding="utf-8")
    marker = "Markdown Content:"
    if marker in text[:500]:
        text = text.split(marker, 1)[1].lstrip()
    return (ParsedUnit(locator="document", text=text),)


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
