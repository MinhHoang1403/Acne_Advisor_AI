from __future__ import annotations

import json

import pytest

from src.ingestion.domain_metadata import enrich_domain_metadata
from src.ingestion.json_loader import load_web_json_documents


LONG_TEXT = (
    "Benzoyl peroxide can help treat mild acne. "
    "Adapalene is a retinoid used for comedonal acne and inflammation."
)


def _write_json(path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_load_web_json_documents_array(tmp_path) -> None:
    path = tmp_path / "web_raw_dataset.json"
    _write_json(
        path,
        [{
            "seed_url": "https://www.aad.org/public/diseases/acne",
            "source_url": "https://www.aad.org/public/diseases/acne/diy/adult-acne-treatment",
            "raw_text": LONG_TEXT,
        }],
    )

    documents = load_web_json_documents(path)

    assert len(documents) == 1
    assert "Benzoyl peroxide" in documents[0]["text"]
    assert documents[0]["metadata"]["source_type"] == "web_json"
    assert documents[0]["metadata"]["record_index"] == 0


def test_load_web_json_documents_retains_short_nonempty_records(tmp_path) -> None:
    path = tmp_path / "web_raw_dataset.json"
    _write_json(
        path,
        [
            {"source_url": "https://example.test/empty", "raw_text": ""},
            {"source_url": "https://example.test/short", "raw_text": "short"},
            {"source_url": "https://example.test/ok", "raw_text": LONG_TEXT},
        ],
    )

    documents = load_web_json_documents(path)

    assert [item["text"] for item in documents] == ["short", LONG_TEXT]
    assert documents[0]["metadata"]["record_index"] == 1


def test_json_metadata_enrichment_maps_entities(tmp_path) -> None:
    path = tmp_path / "web_raw_dataset.json"
    _write_json(path, [{"source_url": "https://example.test/acne", "raw_text": LONG_TEXT}])
    document = load_web_json_documents(path)[0]

    metadata = enrich_domain_metadata(document["text"], existing_metadata=document["metadata"])

    assert "benzoyl_peroxide" in metadata["active_ingredient"]
    assert "adapalene" in metadata["active_ingredient"]
    assert "topical_retinoid" in metadata["drug_class"]


def test_json_loader_supports_records_mapping(tmp_path) -> None:
    path = tmp_path / "web_raw_dataset.json"
    _write_json(path, {"records": [{"source_url": "https://example.test/acne", "raw_text": LONG_TEXT}]})

    documents = load_web_json_documents(path)

    assert len(documents) == 1
    assert documents[0]["metadata"]["record_index"] == 0


def test_json_loader_malformed_file_raises_clear_error(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON file"):
        load_web_json_documents(path)
