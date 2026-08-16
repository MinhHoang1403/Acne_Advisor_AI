"""Portable canonical source and child-record provenance verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


SOURCE_MANIFEST_SCHEMA = "phase1_source_manifest"
SOURCE_MANIFEST_SCHEMA_VERSION = 2
WEB_RECORD_CATALOG_SCHEMA = "web_record_provenance"
WEB_RECORD_CATALOG_SCHEMA_VERSION = 1


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
    record_catalog: str = ""
    record_catalog_sha256: str = ""
    claim_exclusions: str = ""
    claim_exclusions_sha256: str = ""


@dataclass(frozen=True)
class WebRecordSource:
    parent_source_id: str
    source_id: str
    title: str
    authority: str
    source_type: str
    source_url: str
    raw_text_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_portable_text_file(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def load_source_manifest(path: Path) -> tuple[CanonicalSource, ...]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported source manifest schema at {path}")
    if raw.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported source manifest version at {path}")
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


def verify_manifest_support_files(
    sources: tuple[CanonicalSource, ...],
    manifest_path: Path,
) -> None:
    """Verify immutable sidecar catalogs before they can affect compilation."""

    for source in sources:
        for filename, expected_hash in (
            (source.record_catalog, source.record_catalog_sha256),
            (source.claim_exclusions, source.claim_exclusions_sha256),
        ):
            if not filename:
                continue
            path = manifest_path.parent / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing source-manifest support file: {filename}")
            actual_hash = sha256_portable_text_file(path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Source-manifest support hash mismatch for {filename}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )


def load_web_record_catalog(
    source: CanonicalSource,
    *,
    manifest_path: Path,
    source_dir: Path,
) -> dict[str, WebRecordSource]:
    """Load and cross-check one child source for every raw web JSON record."""

    if not source.record_catalog:
        return {}
    catalog_path = manifest_path.parent / source.record_catalog
    raw_catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw_catalog, dict)
        or raw_catalog.get("schema") != WEB_RECORD_CATALOG_SCHEMA
        or raw_catalog.get("schema_version") != WEB_RECORD_CATALOG_SCHEMA_VERSION
        or raw_catalog.get("parent_source_id") != source.source_id
    ):
        raise ValueError(f"Unsupported web record catalog at {catalog_path}")

    publishers = raw_catalog.get("publishers")
    catalog_records = raw_catalog.get("records")
    if not isinstance(publishers, dict) or not isinstance(catalog_records, list):
        raise ValueError("Web record catalog requires publishers and records")

    raw_records = _json_records(source_dir / source.local_filename)
    raw_by_url: dict[str, dict[str, Any]] = {}
    for record in raw_records:
        url = str(record.get("source_url") or "").strip()
        if not url or url in raw_by_url:
            raise ValueError("Raw web dataset contains a missing or duplicate source_url")
        raw_by_url[url] = record

    records_by_url: dict[str, WebRecordSource] = {}
    source_ids: set[str] = set()
    for record in catalog_records:
        if not isinstance(record, dict):
            raise ValueError("Every web record catalog entry must be an object")
        url = str(record.get("source_url") or "").strip()
        publisher_key = str(record.get("publisher") or "").strip()
        publisher = publishers.get(publisher_key)
        if not isinstance(publisher, dict):
            raise ValueError(f"Unknown publisher {publisher_key!r} for {url}")
        if not url or url in records_by_url:
            raise ValueError("Web record catalog contains a missing or duplicate source_url")
        if url not in raw_by_url:
            raise ValueError(f"Web record catalog contains an extra source_url: {url}")

        hostname = (urlparse(url).hostname or "").casefold().removeprefix("www.")
        domains = {
            str(value).casefold().removeprefix("www.") for value in publisher.get("domains", [])
        }
        if hostname not in domains:
            raise ValueError(f"Publisher/domain mismatch for {url}")
        expected_id = web_record_source_id(publisher_key, url)
        if record.get("source_id") != expected_id:
            raise ValueError(f"Non-deterministic child source_id for {url}")
        raw_hash = hashlib.sha256(
            str(raw_by_url[url].get("raw_text") or "").encode("utf-8")
        ).hexdigest()
        if record.get("raw_text_sha256") != raw_hash:
            raise ValueError(f"Raw text hash mismatch for {url}")

        child = WebRecordSource(
            parent_source_id=source.source_id,
            source_id=expected_id,
            title=_required_text(record, "title"),
            authority=_required_text(publisher, "authority"),
            source_type=_required_text(publisher, "source_type"),
            source_url=url,
            raw_text_sha256=raw_hash,
        )
        if child.source_id in source_ids:
            raise ValueError(f"Duplicate child source_id: {child.source_id}")
        source_ids.add(child.source_id)
        records_by_url[url] = child

    missing = sorted(set(raw_by_url) - set(records_by_url))
    if missing or len(records_by_url) != len(raw_records):
        raise ValueError(f"Web record catalog does not cover raw dataset: missing={missing}")
    return records_by_url


def source_manifest_hash(path: Path) -> str:
    load_source_manifest(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def web_record_source_id(publisher_key: str, source_url: str) -> str:
    url_hash = hashlib.sha256(source_url.strip().encode("utf-8")).hexdigest()[:16]
    return f"web_{publisher_key.strip().casefold()}_{url_hash}"


def _source_from_record(record: Any) -> CanonicalSource:
    if not isinstance(record, dict):
        raise ValueError("Every source manifest record must be an object")
    required = (
        "source_id",
        "title",
        "authority",
        "source_type",
        "version_date",
        "original_url",
        "local_filename",
        "media_type",
        "sha256",
    )
    values = {key: str(record.get(key) or "").strip() for key in required}
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ValueError(f"Source manifest record missing: {', '.join(missing)}")
    if len(values["sha256"]) != 64:
        raise ValueError(f"Invalid SHA-256 for source {values['source_id']}")
    if Path(values["local_filename"]).name != values["local_filename"]:
        raise ValueError("local_filename must be portable and contain no directory")
    optional = {
        key: str(record.get(key) or "").strip()
        for key in (
            "record_catalog",
            "record_catalog_sha256",
            "claim_exclusions",
            "claim_exclusions_sha256",
        )
    }
    for filename_key, hash_key in (
        ("record_catalog", "record_catalog_sha256"),
        ("claim_exclusions", "claim_exclusions_sha256"),
    ):
        filename = optional[filename_key]
        expected_hash = optional[hash_key]
        if bool(filename) != bool(expected_hash):
            raise ValueError(f"{filename_key} and {hash_key} must be configured together")
        if filename and Path(filename).name != filename:
            raise ValueError(f"{filename_key} must contain no directory")
        if expected_hash and len(expected_hash) != 64:
            raise ValueError(f"Invalid SHA-256 in {hash_key}")
    return CanonicalSource(**values, **optional)


def _json_records(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        records = next(
            (
                raw[key]
                for key in ("records", "data", "items", "documents", "pages")
                if isinstance(raw.get(key), list)
            ),
            None,
        )
    else:
        records = None
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Unsupported web JSON structure at {path}")
    return records


def _required_text(record: dict[str, Any], key: str) -> str:
    value = str(record.get(key) or "").strip()
    if not value:
        raise ValueError(f"Web provenance record missing {key}")
    return value


__all__ = [
    "CanonicalSource",
    "SOURCE_MANIFEST_SCHEMA",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "WEB_RECORD_CATALOG_SCHEMA",
    "WebRecordSource",
    "load_web_record_catalog",
    "load_source_manifest",
    "sha256_file",
    "sha256_portable_text_file",
    "source_manifest_hash",
    "verify_manifest_support_files",
    "verify_source_files",
    "web_record_source_id",
]
