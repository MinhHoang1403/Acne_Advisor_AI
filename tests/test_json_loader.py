import json

import pytest

from scripts.ingest_knowledge import (
    _file_manifest_info,
    stage2_chunk_web_json_file,
)
from src.ingestion.domain_metadata import enrich_domain_metadata
from src.ingestion.json_loader import load_web_json_documents


LONG_TEXT = (
    "Benzoyl peroxide can help treat mild acne. "
    "Adapalene is a retinoid used for comedonal acne and inflammation."
)


def _write_json(path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_load_web_json_documents_array(tmp_path) -> None:
    json_path = tmp_path / "web_raw_dataset.json"
    _write_json(
        json_path,
        [
            {
                "seed_url": "https://www.aad.org/public/diseases/acne",
                "source_url": "https://www.aad.org/public/diseases/acne/diy/adult-acne-treatment",
                "raw_text": LONG_TEXT,
            }
        ],
    )

    documents = load_web_json_documents(json_path)

    assert len(documents) == 1
    assert "Benzoyl peroxide" in documents[0]["text"]
    metadata = documents[0]["metadata"]
    assert metadata["source_type"] == "web_json"
    assert metadata["source_url"] == "https://www.aad.org/public/diseases/acne/diy/adult-acne-treatment"
    assert metadata["seed_url"] == "https://www.aad.org/public/diseases/acne"
    assert metadata["record_index"] == 0


def test_load_web_json_documents_skips_empty_raw_text(tmp_path) -> None:
    json_path = tmp_path / "web_raw_dataset.json"
    _write_json(
        json_path,
        [
            {"seed_url": "", "source_url": "https://example.test/empty", "raw_text": ""},
            {"seed_url": "", "source_url": "https://example.test/short", "raw_text": "short"},
            {"seed_url": "", "source_url": "https://example.test/ok", "raw_text": LONG_TEXT},
        ],
    )

    documents = load_web_json_documents(json_path)

    assert len(documents) == 2
    assert documents[0]["text"] == "short"
    assert documents[0]["metadata"]["record_index"] == 1
    assert documents[1]["metadata"]["source_url"] == "https://example.test/ok"
    assert documents[1]["metadata"]["record_index"] == 2


def test_json_metadata_enrichment_maps_entities(tmp_path) -> None:
    json_path = tmp_path / "web_raw_dataset.json"
    _write_json(
        json_path,
        [{"seed_url": "", "source_url": "https://example.test/acne", "raw_text": LONG_TEXT}],
    )
    document = load_web_json_documents(json_path)[0]

    metadata = enrich_domain_metadata(
        document["text"],
        existing_metadata=document["metadata"],
    )

    assert "benzoyl_peroxide" in metadata["active_ingredient"]
    assert "adapalene" in metadata["active_ingredient"]
    assert "benzoyl_peroxide" not in metadata["drug_class"]
    assert "topical_retinoid" in metadata["drug_class"]


def test_json_loader_supports_dict_records_key(tmp_path) -> None:
    json_path = tmp_path / "web_raw_dataset.json"
    _write_json(
        json_path,
        {"records": [{"seed_url": "", "source_url": "https://example.test/acne", "raw_text": LONG_TEXT}]},
    )

    documents = load_web_json_documents(json_path)

    assert len(documents) == 1
    assert documents[0]["metadata"]["record_index"] == 0


def test_json_loader_malformed_file_raises_clear_error(tmp_path) -> None:
    json_path = tmp_path / "bad.json"
    json_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON file"):
        load_web_json_documents(json_path)


def test_stage2_json_chunks_preserve_source_metadata(tmp_path) -> None:
    json_path = tmp_path / "web_raw_dataset.json"
    _write_json(
        json_path,
        [
            {
                "seed_url": "https://www.aad.org/public/diseases/acne",
                "source_url": "https://www.aad.org/public/diseases/acne/diy/adult-acne-treatment",
                "raw_text": LONG_TEXT,
            }
        ],
    )
    file_info = _file_manifest_info(json_path)

    chunks = stage2_chunk_web_json_file(json_path, file_info)

    assert len(chunks) == 1
    metadata = chunks[0].metadata
    assert metadata["source_type"] == "web_json"
    assert metadata["source_file"] == "web_raw_dataset.json"
    assert metadata["source_path"] == file_info["source_path"]
    assert metadata["source_url"] == "https://www.aad.org/public/diseases/acne/diy/adult-acne-treatment"
    assert metadata["seed_url"] == "https://www.aad.org/public/diseases/acne"
    assert metadata["record_index"] == 0
    assert metadata["document_id"]
    assert metadata["chunk_id"] == chunks[0].chunk_id
    assert file_info["json_record_count"] == 1
    assert file_info["document_count"] == 1
    assert file_info["skipped_record_count"] == 0
