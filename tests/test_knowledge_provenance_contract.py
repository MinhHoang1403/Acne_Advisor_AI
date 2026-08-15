from __future__ import annotations

from src.ingestion.provenance import (
    base_source_provenance,
    chunk_id,
    document_id,
    record_id,
    validate_provenance,
)
from src.ingestion.source_manifest import CanonicalSource


def _source() -> CanonicalSource:
    return CanonicalSource(
        source_id="canonical_source",
        title="Title",
        authority="Authority",
        source_type="clinical_guideline",
        version_date="2026-01-01",
        original_url="https://example.test/source",
        local_filename="source.pdf",
        media_type="application/pdf",
        sha256="a" * 64,
    )


def test_document_and_chunk_identity_are_path_independent_and_content_bound() -> None:
    source = _source()
    doc = document_id(source.source_id, source.sha256)
    rec = record_id(doc, "page:1")
    first = chunk_id(doc, rec, ("Treatment",), 0, "b" * 64)
    second = chunk_id(doc, rec, ("Treatment",), 0, "b" * 64)

    assert first == second
    assert first != chunk_id(doc, rec, ("Treatment",), 1, "b" * 64)
    assert doc != document_id(source.source_id, "c" * 64)
    assert "C:\\" not in doc


def test_base_source_provenance_uses_canonical_manifest_fields() -> None:
    payload = base_source_provenance(_source())
    assert payload["source_id"] == "canonical_source"
    assert payload["source_content_hash"] == "a" * 64
    assert payload["document_id"].startswith("doc_")


def test_provenance_validation_requires_every_contract_field() -> None:
    errors = validate_provenance({"source_id": "x"})
    assert "missing:chunk_id" in errors
    assert "missing:build_id" in errors
