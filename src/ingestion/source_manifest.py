"""Portable canonical source-manifest loading and verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SOURCE_MANIFEST_SCHEMA = "phase1_source_manifest"


@dataclass(frozen=True)
class CanonicalSource:
    source_id: str
    title: str
    authority: str
    source_type: str
    version_date: str
    original_url: str
    local_filename: str
    media_type: str
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_manifest(path: Path) -> tuple[CanonicalSource, ...]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported source manifest schema at {path}")
    records = raw.get("sources")
    if not isinstance(records, list) or not records:
        raise ValueError("Source manifest must contain at least one source")
    sources = tuple(_source_from_record(record) for record in records)
    ids = [source.source_id for source in sources]
    filenames = [source.local_filename for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("Source manifest contains duplicate source_id")
    if len(filenames) != len(set(filenames)):
        raise ValueError("Source manifest contains duplicate local_filename")
    return sources


def verify_source_files(sources: tuple[CanonicalSource, ...], source_dir: Path) -> None:
    for source in sources:
        path = source_dir / source.local_filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical source: {source.local_filename}")
        actual = sha256_file(path)
        if actual != source.sha256:
            raise ValueError(
                f"Canonical source hash mismatch for {source.source_id}: "
                f"expected {source.sha256}, got {actual}"
            )


def source_manifest_hash(path: Path) -> str:
    sources = load_source_manifest(path)
    canonical = "\n".join(
        f"{item.source_id}\0{item.sha256}\0{item.version_date}" for item in sources
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_from_record(record: Any) -> CanonicalSource:
    if not isinstance(record, dict):
        raise ValueError("Every source manifest record must be an object")
    required = (
        "source_id", "title", "authority", "source_type", "version_date",
        "original_url", "local_filename", "media_type", "sha256",
    )
    values = {key: str(record.get(key) or "").strip() for key in required}
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ValueError(f"Source manifest record missing: {', '.join(missing)}")
    if len(values["sha256"]) != 64:
        raise ValueError(f"Invalid SHA-256 for source {values['source_id']}")
    if Path(values["local_filename"]).name != values["local_filename"]:
        raise ValueError("local_filename must be portable and contain no directory")
    return CanonicalSource(**values)


__all__ = [
    "CanonicalSource",
    "SOURCE_MANIFEST_SCHEMA",
    "load_source_manifest",
    "sha256_file",
    "source_manifest_hash",
    "verify_source_files",
]
