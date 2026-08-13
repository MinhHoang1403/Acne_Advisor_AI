from __future__ import annotations

import asyncio
from types import SimpleNamespace

import scripts.ingest_knowledge as ingestion
from scripts.run_full_phase1 import build_full_phase1_plan, run_full_phase1
import src.knowledge.graph_index as graph_index
from src.knowledge.graph_index import (
    replace_entity_graph,
    upsert_entity_graph,
    validate_entity_graph_records,
)
from src.knowledge.phase1_validation import (
    CollectionReconciliation,
    Phase1ValidationReport,
    reconcile_qdrant_collection,
)


def test_short_safety_evidence_survives_without_whitelisting_exact_sentences() -> None:
    statements = (
        "Avoid use during pregnancy.",
        "Do not use during pregnancy.",
        "Avoid while breastfeeding.",
        "Stop use and seek medical care if severe swelling occurs.",
    )

    for statement in statements:
        noisy, _ = ingestion.is_noisy_chunk(statement)
        assert noisy is False

    chunks = ingestion.chunk_markdown_text(
        "# Safety\nAvoid use during pregnancy.",
        "safety.md",
    )
    stats = ingestion.IngestionStats()
    ingestion.account_chunk_quality(chunks, stats)
    assert stats.chunks_rescued_short_safety == 1
    assert stats.chunks_rejected_noise == 0


def test_nonmedical_short_noise_remains_rejectable() -> None:
    for fragment in ("Page 12", "References", "Back to top", "© All rights reserved"):
        noisy, _ = ingestion.is_noisy_chunk(fragment)
        assert noisy is True


def test_document_identity_is_stable_across_absolute_checkout_roots() -> None:
    left = r"C:\\machine-a\\acne-agent-system\\sample_data\\nested\\guide.pdf"
    right = r"D:\\machine-b\\acne-agent-system\\sample_data\\nested\\guide.pdf"

    assert ingestion.canonical_source_identity(left) == "sample_data/nested/guide.pdf"
    assert ingestion.canonical_source_identity(left) == ingestion.canonical_source_identity(right)
    assert ingestion.document_id_from_source_path(left) == ingestion.document_id_from_source_path(right)


def test_incremental_requires_graph_validated_and_matching_config(tmp_path, monkeypatch) -> None:
    source = tmp_path / "guide.pdf"
    source.write_bytes(b"same content")
    file_info = ingestion._file_manifest_info(source)
    completed = {
        **file_info,
        "path": None,
        "status": "completed",
        "graph_validated": True,
        "ingestion_config_fingerprint": ingestion.ingestion_config_fingerprint(),
    }

    plan = ingestion.get_incremental_file_plan(
        [source], {"documents": {file_info["source_path"]: completed}}
    )
    assert plan["summary"]["skipped"] == 1

    completed["graph_validated"] = False
    graph_skipped = ingestion.get_incremental_file_plan(
        [source], {"documents": {file_info["source_path"]: completed}}
    )
    assert graph_skipped["to_ingest"][0]["reason"] == "retry"

    completed["graph_validated"] = True
    monkeypatch.setattr(ingestion, "CHUNK_SIZE", ingestion.CHUNK_SIZE + 1)
    changed_config = ingestion.get_incremental_file_plan(
        [source], {"documents": {file_info["source_path"]: completed}}
    )
    assert changed_config["to_ingest"][0]["reason"] == "config_changed"


def test_incremental_retries_graph_skipped_or_warning_completion(tmp_path) -> None:
    source = tmp_path / "guide.pdf"
    source.write_bytes(b"same content")
    file_info = ingestion._file_manifest_info(source)
    for status in ("completed_with_graph_skipped", "completed_with_warnings"):
        plan = ingestion.get_incremental_file_plan(
            [source],
            {
                "documents": {
                    file_info["source_path"]: {
                        **file_info,
                        "path": None,
                        "status": status,
                        "graph_validated": False,
                        "ingestion_config_fingerprint": ingestion.ingestion_config_fingerprint(),
                    }
                }
            },
        )
        assert plan["to_ingest"][0]["reason"] == "retry"


def test_parser_completeness_rejects_observable_partial_output() -> None:
    valid = [SimpleNamespace(text="A short valid document.", metadata={})]
    explicit_failure = [SimpleNamespace(text="Some text", metadata={"page_error": "timeout"})]

    assert ingestion.validate_parser_documents(valid).passed is True
    failed = ingestion.validate_parser_documents(explicit_failure)
    assert failed.passed is False
    assert failed.status == "failed"
    assert ingestion.validate_parser_documents([]).passed is False


def test_neo4j_edge_missing_endpoint_is_not_counted_as_success() -> None:
    class Result:
        async def single(self):
            return {"source_count": 0, "target_count": 1, "relationship_count": 0}

    class Session:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def run(self, _query: str, **kwargs):
            self.calls.append(kwargs)
            return Result()

    payload = ingestion.GraphPayload(
        chunk_id="chunk-1",
        edges=[
            ingestion.GraphEdge(
                source_name="missing",
                source_type="DRUG",
                target_name="target",
                target_type="DISEASE",
                relation="TREATS",
            )
        ],
    )
    session = Session()
    result = asyncio.run(
        ingestion.upsert_neo4j_payloads(
            session,
            [payload],
            payload_document_ids={"chunk-1": "document-1"},
        )
    )

    assert result.relationships_upserted == 0
    assert result.missing_source_endpoint == 1
    assert result.has_failures is True


def test_neo4j_edge_provenance_is_written_only_after_endpoints_validate() -> None:
    class Result:
        def __init__(self, record: dict) -> None:
            self.record = record

        async def single(self):
            return self.record

    class Session:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def run(self, query: str, **kwargs):
            self.calls.append((query, kwargs))
            if "source_count" in query:
                return Result({"source_count": 1})
            if "target_count" in query:
                return Result({"target_count": 1})
            return Result({"relationship_count": 1})

    payload = ingestion.GraphPayload(
        chunk_id="chunk-1",
        edges=[
            ingestion.GraphEdge(
                source_name="source",
                source_type="DRUG",
                target_name="target",
                target_type="DISEASE",
                relation="TREATS",
            )
        ],
    )
    session = Session()
    result = asyncio.run(
        ingestion.upsert_neo4j_payloads(
            session,
            [payload],
            payload_document_ids={"chunk-1": "document-1"},
        )
    )

    assert result.relationships_upserted == 1
    assert result.has_failures is False
    assert session.calls[-1][1]["source_document_ids"] == ["document-1"]


def test_source_managed_graph_reconciliation_is_document_scoped() -> None:
    class Result:
        def __init__(self, record):
            self.record = record

        async def single(self):
            return self.record

    class Session:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def run(self, query: str, **kwargs):
            self.calls.append((query, kwargs))
            return Result({"deleted": 2} if "deleted" in query else {"detached": 1})

    session = Session()
    result = asyncio.run(ingestion.reconcile_neo4j_document_graph(session, "document-1"))

    assert result == {"deleted": 2, "detached": 1}
    assert all(call[1]["document_id"] == "document-1" for call in session.calls)
    assert "size(r.source_document_ids) = 1" in session.calls[0][0]


def _knowledge_payload(source_path: str, *, chunk_id: str = "chunk-1") -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "document-1",
        "source_path": source_path,
        "content_hash": "hash",
        "chunk_index": 0,
        "ingestion_run_id": "run-1",
        "ingested_at": "2026-08-11T00:00:00+00:00",
        "text": "Evidence",
        "embedding_provider": "google",
        "embedding_model": "models/gemini-embedding-2",
        "embedding_dimensions": 3072,
        "kb_version": "acne_kb_v1",
    }


def test_qdrant_reconciliation_rejects_count_and_per_source_mismatch() -> None:
    class Client:
        async def get_collection(self, **_kwargs):
            return SimpleNamespace(
                points_count=1,
                config=SimpleNamespace(
                    params={
                        "vectors": {"dense": {"size": 3072}},
                        "sparse_vectors": {"bm25": {}},
                    }
                ),
            )

        async def scroll(self, **_kwargs):
            return ([SimpleNamespace(payload=_knowledge_payload("sample_data/a.pdf"))], None)

    result = asyncio.run(
        reconcile_qdrant_collection(
            client=Client(),
            collection_name="acne_knowledge",
            role="knowledge",
            expected_count=2,
            expected_dimensions=3072,
            expected_by_source={"sample_data/b.pdf": 2},
        )
    )

    assert result.passed is False
    assert any("count mismatch" in error for error in result.errors)
    assert any("per-source" in error for error in result.errors)


def test_entity_qdrant_reconciliation_rejects_identity_mismatch() -> None:
    entity_payload = {
        "entity_id": "active_ingredient:adapalene",
        "canonical_name": "adapalene",
        "entity_type": "active_ingredient",
        "taxonomy_version": "drug_taxonomy_v1",
        "entity_schema_version": "entity_schema_v1",
        "point_id": "point-1",
        "text": "adapalene",
        "embedding_provider": "google",
        "embedding_model": "models/gemini-embedding-2",
        "embedding_dimensions": 3072,
        "kb_version": "acne_kb_v1",
    }

    class Client:
        async def get_collection(self, **_kwargs):
            return SimpleNamespace(
                points_count=1,
                config=SimpleNamespace(
                    params={
                        "vectors": {"dense": {"size": 3072}},
                        "sparse_vectors": {"bm25": {}},
                    }
                ),
            )

        async def scroll(self, **_kwargs):
            return ([SimpleNamespace(payload=entity_payload)], None)

    result = asyncio.run(
        reconcile_qdrant_collection(
            client=Client(),
            collection_name="acne_entities_v1",
            role="entity",
            expected_count=1,
            expected_dimensions=3072,
            expected_entity_ids={"active_ingredient:benzoyl_peroxide"},
        )
    )

    assert result.passed is False
    assert any("identity mismatch" in error for error in result.errors)


def test_strict_graph_record_validation_detects_missing_relationship() -> None:
    records = {
        "nodes": [{"label": "DrugProduct", "canonical_name": "Product"}],
        "relationships": [
            {
                "source_label": "DrugProduct",
                "source_name": "Product",
                "relationship": "HAS_ACTIVE_INGREDIENT",
                "target_label": "ActiveIngredient",
                "target_name": "Ingredient",
            }
        ],
    }

    class Result:
        def __init__(self, count: int) -> None:
            self.count = count

        async def single(self):
            return {"count": self.count}

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def run(self, query: str, **_kwargs):
            if "MATCH (src)-[r:" in query:
                return Result(0)
            if "canonical_name" in query or "MATCH (n:DrugProduct)" in query:
                return Result(1)
            return Result(0)

    class Driver:
        def session(self):
            return Session()

    validation = asyncio.run(validate_entity_graph_records(Driver(), records))
    assert validation["passed"] is False
    assert len(validation["errors"]) >= 1
    assert "relationship" in validation["errors"][0]


def test_entity_graph_upsert_counts_only_materialized_records() -> None:
    records = {
        "nodes": [
            {"label": "DrugProduct", "canonical_name": "Product"},
            {"label": "ActiveIngredient", "canonical_name": "Ingredient"},
        ],
        "relationships": [
            {
                "source_label": "DrugProduct",
                "source_name": "Product",
                "relationship": "HAS_ACTIVE_INGREDIENT",
                "target_label": "ActiveIngredient",
                "target_name": "Ingredient",
                "properties": {},
            }
        ],
    }

    class Result:
        def __init__(self, record: dict) -> None:
            self.record = record

        async def single(self):
            return self.record

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def run(self, query: str, **_kwargs):
            if "MERGE (n:" in query:
                return Result({"materialized_count": 1})
            return Result(
                {
                    "source_count": 1,
                    "target_count": 1,
                    "materialized_count": 1,
                }
            )

    class Driver:
        def session(self):
            return Session()

    assert asyncio.run(upsert_entity_graph(Driver(), records)) == {
        "nodes": 2,
        "relationships": 1,
    }


def test_replace_entity_graph_removes_stale_relationships_before_nodes(monkeypatch) -> None:
    queries: list[str] = []

    class Result:
        async def single(self):
            return {"removed": 1}

        async def consume(self):
            return None

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def run(self, query: str, **_kwargs):
            queries.append(query)
            return Result()

    class Driver:
        def session(self):
            return Session()

    async def no_schema(_driver):
        return None

    async def materialize(_driver, _records):
        return {"nodes": 1, "relationships": 1}

    async def valid(_driver, _records):
        return {"passed": True, "errors": []}

    monkeypatch.setattr(graph_index, "apply_entity_graph_schema", no_schema)
    monkeypatch.setattr(graph_index, "upsert_entity_graph", materialize)
    monkeypatch.setattr(graph_index, "validate_entity_graph_records", valid)

    result = asyncio.run(
        replace_entity_graph(
            Driver(),
            {"nodes": [], "relationships": []},
            build_id="build-a",
        )
    )

    relationship_delete = next(i for i, query in enumerate(queries) if "DELETE r" in query)
    node_delete = next(i for i, query in enumerate(queries) if "DETACH DELETE n" in query)
    assert relationship_delete < node_delete
    assert result["stale_relationships_removed"] == 1
    assert result["stale_nodes_removed"] == 1


def test_strict_graph_record_validation_detects_stale_aggregate_count() -> None:
    records = {"nodes": [{"label": "DrugProduct", "canonical_name": "Product"}], "relationships": []}

    class Result:
        def __init__(self, count: int) -> None:
            self.count = count

        async def single(self):
            return {"count": self.count}

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def run(self, query: str, **_kwargs):
            if "canonical_name" in query:
                return Result(1)
            if "MATCH (n:DrugProduct)" in query:
                return Result(2)
            return Result(0)

    class Driver:
        def session(self):
            return Session()

    validation = asyncio.run(validate_entity_graph_records(Driver(), records))
    assert validation["passed"] is False
    assert validation["actual_nodes_by_label"]["DrugProduct"] == 2
    assert "node count mismatch" in validation["errors"][0]


def test_full_phase1_plan_and_dry_run_are_non_mutating(tmp_path) -> None:
    source_dir = tmp_path / "sample_data"
    source_dir.mkdir()
    (source_dir / "knowledge.json").write_text("[]", encoding="utf-8")
    plan, records = build_full_phase1_plan(source_dir)
    result = asyncio.run(run_full_phase1(source_dir=source_dir, dry_run=True))

    assert not plan.errors
    assert len(plan.source_paths) == 1
    assert plan.entity_count > 0
    assert len(records["relationships"]) == plan.graph_relationship_count
    assert result["passed"] is True
    assert result["status"] == "dry_run"
    assert result["mutated"] is False


def test_final_phase1_validation_requires_every_child_invariant() -> None:
    passing_collection = CollectionReconciliation(
        role="knowledge", collection="acne_knowledge", expected_count=1, actual_count=1
    )
    report = Phase1ValidationReport(
        sources_expected=1,
        sources_accounted_for=1,
        knowledge=passing_collection,
        entities=CollectionReconciliation(
            role="entity", collection="acne_entities_v1", expected_count=1, actual_count=1
        ),
        graph={"passed": True},
    )

    assert report.passed is True
    report.graph = {"passed": False}
    assert report.passed is False


def test_manifest_is_completed_only_after_full_phase1_validation(tmp_path) -> None:
    manifest_path = tmp_path / "ingestion_manifest.json"
    manifest = {
        "documents": {
            "sample_data/a.pdf": {
                "source_path": "sample_data/a.pdf",
                "status": "knowledge_indexed_pending_phase1_validation",
                "graph_validated": True,
            }
        }
    }
    ingestion.save_ingestion_manifest(manifest_path, manifest)

    ingestion.finalize_full_phase1_manifest(
        manifest_path,
        validation={"passed": False, "errors": ["entity mismatch"]},
    )
    failed = ingestion.load_ingestion_manifest(manifest_path)
    assert failed["full_phase1_validation"]["status"] == "failed"
    assert failed["documents"]["sample_data/a.pdf"]["status"] == (
        "knowledge_indexed_pending_phase1_validation"
    )

    ingestion.finalize_full_phase1_manifest(
        manifest_path,
        validation={"passed": True, "errors": []},
    )
    completed = ingestion.load_ingestion_manifest(manifest_path)
    assert completed["full_phase1_validation"]["status"] == "completed"
    assert completed["core_phase1"]["status"] == "completed_validated"
    assert completed["semantic_enrichment"]["status"] == "not_run"
    assert completed["documents"]["sample_data/a.pdf"]["status"] == "completed"
    assert completed["documents"]["sample_data/a.pdf"]["graph_validated"] is False
