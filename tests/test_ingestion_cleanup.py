import asyncio

from src.ingestion.cleanup import (
    build_qdrant_cleanup_plan,
    cleanup_previous_qdrant_points,
    is_safe_chunk_collection_for_cleanup,
)
from scripts.ingest_knowledge import (
    GraphPayload,
    SemanticChunk,
    _file_manifest_info,
    finalize_manifest_for_documents,
    get_incremental_file_plan,
    ingestion_config_fingerprint,
    update_manifest_after_success,
)


def test_manifest_backward_compatible_without_point_ids(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "acne_knowledge")

    plan = build_qdrant_cleanup_plan(
        collection_name="acne_knowledge",
        manifest_record={
            "document_id": "doc-1",
            "source_path": "sample/a.pdf",
            "content_hash": "old",
        },
        expected_document_id="doc-1",
        expected_source_path="sample/a.pdf",
    )

    assert plan["safe"] is True
    assert plan["cleanup_required"] is True
    assert plan["mode"] == "filter"
    assert {"key": "document_id", "match": {"value": "doc-1"}} in plan["filter"]["must"]


def test_cleanup_prefers_point_ids(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "acne_knowledge")

    plan = build_qdrant_cleanup_plan(
        collection_name="acne_knowledge",
        manifest_record={
            "document_id": "doc-1",
            "qdrant_point_ids": ["id1", "id2", "id1"],
        },
        expected_document_id="doc-1",
    )

    assert plan["safe"] is True
    assert plan["mode"] == "ids"
    assert plan["point_ids"] == ["id1", "id2"]
    assert plan["point_count"] == 2


def test_cleanup_blocks_entity_collection(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "acne_knowledge")
    monkeypatch.setenv("ENTITY_QDRANT_COLLECTION_NAME", "acne_entities_v1")

    class FakeQdrantClient:
        def __init__(self) -> None:
            self.delete_calls = []

        async def delete(self, **kwargs) -> None:
            self.delete_calls.append(kwargs)

    client = FakeQdrantClient()
    result = asyncio.run(
        cleanup_previous_qdrant_points(
            qdrant_client=client,
            collection_name="acne_entities_v1",
            manifest_record={"document_id": "doc-1", "qdrant_point_ids": ["id1"]},
            expected_document_id="doc-1",
        )
    )

    assert result["safe"] is False
    assert client.delete_calls == []


def test_cleanup_blocks_empty_collection(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "acne_knowledge")

    assert is_safe_chunk_collection_for_cleanup("") is False
    plan = build_qdrant_cleanup_plan(
        collection_name="",
        manifest_record={"document_id": "doc-1", "qdrant_point_ids": ["id1"]},
        expected_document_id="doc-1",
    )

    assert plan["safe"] is False
    assert plan["cleanup_required"] is False


def test_cleanup_filter_requires_document_id(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "acne_knowledge")

    plan = build_qdrant_cleanup_plan(
        collection_name="acne_knowledge",
        manifest_record={},
        expected_document_id="",
    )

    assert plan["safe"] is False
    assert plan["cleanup_required"] is False
    assert "expected_document_id" in plan["reason"]


def test_manifest_record_updated_with_point_ids(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "acne_knowledge")
    monkeypatch.setenv("KB_VERSION", "acne_kb_v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")

    manifest = {"documents": {}}
    update_manifest_after_success(
        manifest=manifest,
        source_path="sample/a.pdf",
        document_id="doc-1",
        content_hash="hash-1",
        file_size=123,
        modified_time="2026-01-01T00:00:00+00:00",
        chunk_count=2,
        qdrant_point_ids=["point-1", "point-2", "point-1"],
        ingestion_run_id="run-1",
        last_ingested_at="2026-01-01T00:00:01+00:00",
    )

    record = manifest["documents"]["sample/a.pdf"]
    assert record["qdrant_point_ids"] == ["point-1", "point-2"]
    assert record["qdrant_point_count"] == 2
    assert record["qdrant_collection"] == "acne_knowledge"
    assert record["kb_version"] == "acne_kb_v1"
    assert record["embedding_model"] == "models/gemini-embedding-2"
    assert record["embedding_dimensions"] == 3072
    assert record["cleanup_version"] == "qdrant_cleanup_v1"


def test_changed_file_requires_cleanup_before_upsert(tmp_path) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"new")
    file_info = _file_manifest_info(source)
    manifest = {
        "documents": {
            file_info["source_path"]: {
                **file_info,
                "path": None,
                "content_hash": "old-hash",
                "status": "completed",
            }
        }
    }

    plan = get_incremental_file_plan([source], manifest)

    item = plan["to_ingest"][0]
    assert item["reason"] == "changed"
    assert item["cleanup_required"] is True


def test_unchanged_completed_skips_cleanup(tmp_path) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"same")
    file_info = _file_manifest_info(source)
    manifest = {
        "documents": {
            file_info["source_path"]: {
                **file_info,
                "path": None,
                "status": "completed",
                "graph_validated": True,
                "ingestion_config_fingerprint": ingestion_config_fingerprint(),
            }
        }
    }

    plan = get_incremental_file_plan([source], manifest)

    item = plan["skipped"][0]
    assert item["reason"] == "skip"
    assert item["cleanup_required"] is False


def test_partial_retry_requires_cleanup(tmp_path) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"same")
    file_info = _file_manifest_info(source)
    manifest = {
        "documents": {
            file_info["source_path"]: {
                **file_info,
                "path": None,
                "status": "partial",
            }
        }
    }

    plan = get_incremental_file_plan([source], manifest)

    item = plan["to_ingest"][0]
    assert item["reason"] == "retry"
    assert item["cleanup_required"] is True


def test_manifest_completed_with_graph_skipped_when_qdrant_indexed(tmp_path) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"same")
    file_info = _file_manifest_info(source)
    chunk = SemanticChunk(
        source_file=source.name,
        chunk_index=0,
        text="Benzoyl peroxide can help treat mild acne.",
        metadata={
            "document_id": file_info["document_id"],
            "source_path": file_info["source_path"],
        },
    )
    payload = GraphPayload(chunk_id=chunk.chunk_id, extraction_error=False)
    manifest = {"documents": {}}

    finalize_manifest_for_documents(
        manifest=manifest,
        doc_chunks=[(file_info, [chunk])],
        payloads=[payload],
        ingestion_run_id="run-1",
        ingested_at="2026-07-05T00:00:00+00:00",
        skip_neo4j=True,
        skip_qdrant=False,
        skip_graph_extraction=True,
        qdrant_ids_available=True,
        limit_chunks_truncated=False,
    )

    record = manifest["documents"][file_info["source_path"]]
    assert record["status"] == "completed_with_graph_skipped"
    assert record["graph_extraction_skipped"] is True
    assert record["neo4j_skipped"] is True
    assert record["qdrant_indexed"] is True
    assert record["qdrant_point_count"] == 1
    assert "error_message" not in record


def test_manifest_partial_when_qdrant_upsert_has_no_point_ids(tmp_path) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"same")
    file_info = _file_manifest_info(source)
    chunk = SemanticChunk(
        source_file=source.name,
        chunk_index=0,
        text="References only.",
        metadata={
            "document_id": file_info["document_id"],
            "source_path": file_info["source_path"],
            "is_noisy": True,
        },
    )
    payload = GraphPayload(chunk_id=chunk.chunk_id, extraction_error=False)
    manifest = {"documents": {}}

    finalize_manifest_for_documents(
        manifest=manifest,
        doc_chunks=[(file_info, [chunk])],
        payloads=[payload],
        ingestion_run_id="run-1",
        ingested_at="2026-07-05T00:00:00+00:00",
        skip_neo4j=False,
        skip_qdrant=False,
        skip_graph_extraction=False,
        qdrant_ids_available=True,
        limit_chunks_truncated=False,
    )

    record = manifest["documents"][file_info["source_path"]]
    assert record["status"] == "partial"
    assert record["qdrant_point_count"] == 0
    assert "Qdrant upsert produced no point IDs" in record["error_message"]
