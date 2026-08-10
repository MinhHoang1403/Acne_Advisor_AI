from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import scripts.ingest_knowledge as ingestion
import scripts.run_full_phase1 as full_phase1
import scripts.run_semantic_enrichment as enrichment
from src.database.retriever import _get_graph_facts_with_fallback
from src.knowledge.phase1_validation import reconcile_qdrant_collection


def _source_file(tmp_path: Path) -> tuple[Path, str]:
    source_dir = tmp_path / "sample_data"
    source_dir.mkdir()
    source = source_dir / "knowledge.json"
    source.write_text("[]", encoding="utf-8")
    return source_dir, "sample_data/knowledge.json"


def test_core_preflight_never_calls_ollama(monkeypatch, tmp_path: Path) -> None:
    source_dir, _ = _source_file(tmp_path)
    calls = {"ollama": 0, "neo4j": 0, "qdrant": 0}

    async def unexpected_ollama() -> bool:
        calls["ollama"] += 1
        raise AssertionError("core preflight must not call Ollama")

    async def available_neo4j() -> bool:
        calls["neo4j"] += 1
        return True

    async def available_qdrant() -> bool:
        calls["qdrant"] += 1
        return True

    monkeypatch.setattr(ingestion, "LLAMA_CLOUD_API_KEY", "configured")
    monkeypatch.setattr(ingestion, "GOOGLE_API_KEY", "configured")
    monkeypatch.setattr(ingestion, "preflight_ollama", unexpected_ollama)
    monkeypatch.setattr(ingestion, "preflight_neo4j", available_neo4j)
    monkeypatch.setattr(ingestion, "preflight_qdrant", available_qdrant)

    args = full_phase1._preflight_args(source_dir, tmp_path / "manifest.json")
    assert args.skip_graph_extraction is True
    assert args.skip_neo4j is True
    assert asyncio.run(
        ingestion.run_preflight_checks(args, graph_resume_enabled=False, require_neo4j=True)
    )
    assert calls == {"ollama": 0, "neo4j": 1, "qdrant": 1}


def test_core_ingestion_bypasses_stage3_entirely(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "sample_data"
    source_dir.mkdir()
    source = source_dir / "knowledge.pdf"
    source.write_bytes(b"placeholder")
    source_path = "sample_data/knowledge.pdf"
    file_info = {
        **ingestion._file_manifest_info(source, source_root=source_dir),
        "source_path": source_path,
    }

    async def fake_sources(**_kwargs):
        return [(file_info, source.name, "Adapalene is a topical retinoid for acne treatment.")]

    async def unexpected_stage3(**_kwargs):
        raise AssertionError("core ingestion must not call Stage 3")

    monkeypatch.setattr(ingestion, "ensure_cache_dirs", lambda: None)
    monkeypatch.setattr(ingestion, "stage1_extract_sources", fake_sources)
    monkeypatch.setattr(ingestion, "stage3_and_optional_neo4j_incremental", unexpected_stage3)

    stats = asyncio.run(
        ingestion.ingest_pipeline(
            source_dir=source_dir,
            dry_run=True,
            skip_graph_extraction=True,
            skip_neo4j=True,
            skip_semantic_enrichment=True,
        )
    )

    assert stats.chunks_created > 0
    assert stats.llm_errors == 0


def test_empty_graph_nodes_is_valid_core_qdrant_payload() -> None:
    payload = {
        "chunk_id": "chunk-1",
        "document_id": "document-1",
        "source_path": "sample_data/knowledge.json",
        "content_hash": "hash",
        "chunk_index": 0,
        "ingestion_run_id": "run-1",
        "ingested_at": "2026-08-11T00:00:00+00:00",
        "text": "Evidence",
        "graph_nodes": [],
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
            return ([SimpleNamespace(payload=payload)], None)

    result = asyncio.run(
        reconcile_qdrant_collection(
            client=Client(),
            collection_name="acne_knowledge",
            role="knowledge",
            expected_count=1,
            expected_dimensions=3072,
            expected_by_source={"sample_data/knowledge.json": 1},
        )
    )
    assert result.passed is True


def test_retriever_keyword_fallback_works_without_graph_nodes() -> None:
    class GraphStore:
        def __init__(self) -> None:
            self.entity_calls = 0
            self.keyword_calls: list[list[str]] = []

        async def get_entity_context(self, _names):
            self.entity_calls += 1
            return []

        async def search_by_keywords(self, keywords):
            self.keyword_calls.append(keywords)
            return [{"entity": "adapalene"}]

    store = GraphStore()
    facts = asyncio.run(_get_graph_facts_with_fallback(store, set(), "Adapalene for acne"))

    assert facts == [{"entity": "adapalene"}]
    assert store.entity_calls == 0
    assert store.keyword_calls == [["adapalene", "for", "acne"]]


def test_core_manifest_is_valid_when_semantic_enrichment_is_not_run(tmp_path: Path) -> None:
    source_dir, _ = _source_file(tmp_path)
    source = source_dir / "knowledge.json"
    info = ingestion._file_manifest_info(source, source_root=source_dir)
    manifest_path = tmp_path / "manifest.json"
    ingestion.save_ingestion_manifest(
        manifest_path,
        {
            "documents": {
                info["source_path"]: {
                    **info,
                    "path": None,
                    "status": "knowledge_indexed_pending_phase1_validation",
                    "ingestion_config_fingerprint": ingestion.ingestion_config_fingerprint(),
                }
            }
        },
    )

    ingestion.finalize_full_phase1_manifest(manifest_path, validation={"passed": True})
    manifest = ingestion.load_ingestion_manifest(manifest_path)
    entry = manifest["documents"][info["source_path"]]

    assert manifest["core_phase1"]["status"] == "completed_validated"
    assert manifest["semantic_enrichment"]["status"] == "not_run"
    assert entry["core_phase1_status"] == "completed_validated"
    assert entry["semantic_enrichment"]["status"] == "not_run"
    assert entry["graph_validated"] is False
    assert ingestion.get_incremental_file_plan([source], manifest, source_root=source_dir)["summary"]["skipped"] == 1


def test_semantic_failure_does_not_invalidate_core_manifest() -> None:
    manifest = {
        "core_phase1": {"status": "completed_validated"},
        "full_phase1_validation": {"status": "completed"},
        "documents": {
            "sample_data/knowledge.json": {
                "status": "completed",
                "core_phase1_status": "completed_validated",
            }
        },
    }

    ingestion.update_semantic_enrichment_manifest(
        manifest,
        status="failed",
        report={"error": "Ollama unavailable"},
        document_results={"sample_data/knowledge.json": {"status": "failed", "errors": 1}},
    )

    assert manifest["core_phase1"]["status"] == "completed_validated"
    assert manifest["documents"]["sample_data/knowledge.json"]["status"] == "completed"
    assert manifest["semantic_enrichment"]["status"] == "failed"


def test_semantic_fingerprint_does_not_change_core_fingerprint(monkeypatch) -> None:
    core_before = ingestion.ingestion_config_fingerprint()
    semantic_before = ingestion.semantic_enrichment_fingerprint()

    monkeypatch.setattr(ingestion, "OLLAMA_MODEL", "qwen3:14b")

    assert ingestion.ingestion_config_fingerprint() == core_before
    assert ingestion.semantic_enrichment_fingerprint() != semantic_before


def test_optional_semantic_enrichment_reuses_valid_graph_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingestion, "GRAPH_CACHE_DIR", tmp_path / "graph")
    chunk = ingestion.SemanticChunk(
        source_file="knowledge.json",
        chunk_index=0,
        text="Adapalene is a topical retinoid.",
        metadata={"document_id": "document-1", "source_path": "sample_data/knowledge.json"},
    )
    payload = ingestion.GraphPayload(
        chunk_id=chunk.chunk_id,
        nodes=[ingestion.GraphNode(name="adapalene", entity_type="DRUG")],
    )
    assert ingestion.save_graph_payload(payload, chunk)

    class UnexpectedLLM:
        async def ainvoke(self, _messages):
            raise AssertionError("valid optional graph cache should avoid an Ollama call")

    stats = ingestion.IngestionStats()
    restored = asyncio.run(
        ingestion.extract_graph_one_chunk(
            chunk=chunk,
            semaphore=asyncio.Semaphore(1),
            llm=UnexpectedLLM(),
            use_resume=True,
            stats=stats,
        )
    )

    assert restored.from_cache is True
    assert stats.graph_cache_hits == 1
    assert stats.graph_cache_misses == 0


def test_optional_enrichment_calls_ollama_and_keeps_core_valid(monkeypatch, tmp_path: Path) -> None:
    source_dir, source_path = _source_file(tmp_path)
    source = source_dir / "knowledge.json"
    source_identity = ingestion.canonical_source_identity(source, source_root=source_dir)
    chunk = ingestion.SemanticChunk(
        source_file=source.name,
        chunk_index=0,
        text="Adapalene is a topical retinoid.",
        metadata={"document_id": "document-1", "source_path": source_path, "source_identity": source_identity},
    )
    manifest = {
        "core_phase1": {"status": "completed_validated"},
        "full_phase1_validation": {"status": "completed"},
        "documents": {
            source_path: {
                "source_identity": source_identity,
                "status": "completed",
                "core_phase1_status": "completed_validated",
            }
        },
    }
    calls = {"ollama": 0, "stage3": 0, "qdrant": 0, "neo4j": 0, "update": 0}

    async def available(name: str) -> bool:
        calls[name] += 1
        return True

    async def fake_load(**_kwargs):
        return [enrichment.IndexedChunk(point_id="point-1", source_path=source_path, chunk=chunk)]

    async def fake_stage3(**_kwargs):
        calls["stage3"] += 1
        return [
            ingestion.GraphPayload(
                chunk_id=chunk.chunk_id,
                nodes=[ingestion.GraphNode(name="adapalene", entity_type="DRUG")],
            )
        ]

    async def fake_update(*_args, **_kwargs):
        calls["update"] += 1
        return 1

    monkeypatch.setattr(enrichment, "load_ingestion_manifest", lambda _path: manifest)
    monkeypatch.setattr(enrichment, "save_ingestion_manifest", lambda *_args: None)
    monkeypatch.setattr(enrichment, "preflight_ollama", lambda: available("ollama"))
    monkeypatch.setattr(enrichment, "preflight_qdrant", lambda: available("qdrant"))
    monkeypatch.setattr(enrichment, "preflight_neo4j", lambda: available("neo4j"))
    monkeypatch.setattr(enrichment, "load_indexed_chunks", fake_load)
    monkeypatch.setattr(enrichment, "stage3_and_optional_neo4j_incremental", fake_stage3)
    monkeypatch.setattr(enrichment, "update_qdrant_graph_nodes", fake_update)

    result = asyncio.run(enrichment.run_semantic_enrichment(source_dir=source_dir))

    assert result["passed"] is True
    assert calls == {"ollama": 1, "stage3": 1, "qdrant": 1, "neo4j": 1, "update": 1}
    assert manifest["core_phase1"]["status"] == "completed_validated"
    assert manifest["semantic_enrichment"]["status"] == "completed"


def test_partial_semantic_errors_become_warnings_without_invalidating_core(
    monkeypatch, tmp_path: Path
) -> None:
    source_dir, source_path = _source_file(tmp_path)
    source = source_dir / "knowledge.json"
    source_identity = ingestion.canonical_source_identity(source, source_root=source_dir)
    first_chunk = ingestion.SemanticChunk(
        source_file=source.name,
        chunk_index=0,
        text="Adapalene is a topical retinoid.",
        metadata={
            "document_id": "document-1",
            "source_path": source_path,
            "source_identity": source_identity,
        },
    )
    second_chunk = ingestion.SemanticChunk(
        source_file=source.name,
        chunk_index=1,
        text="Benzoyl peroxide is antimicrobial.",
        metadata={
            "document_id": "document-1",
            "source_path": source_path,
            "source_identity": source_identity,
        },
    )
    manifest = {
        "core_phase1": {"status": "completed_validated"},
        "documents": {
            source_path: {
                "source_identity": source_identity,
                "status": "completed",
                "core_phase1_status": "completed_validated",
            }
        },
    }

    async def available() -> bool:
        return True

    async def fake_load(**_kwargs):
        return [
            enrichment.IndexedChunk("point-1", source_path, first_chunk),
            enrichment.IndexedChunk("point-2", source_path, second_chunk),
        ]

    async def fake_stage3(**_kwargs):
        return [
            ingestion.GraphPayload(
                chunk_id=first_chunk.chunk_id,
                nodes=[ingestion.GraphNode(name="adapalene", entity_type="DRUG")],
            ),
            ingestion.GraphPayload(chunk_id=second_chunk.chunk_id, extraction_error=True),
        ]

    async def no_qdrant_update(*_args, **_kwargs):
        return 1

    monkeypatch.setattr(enrichment, "load_ingestion_manifest", lambda _path: manifest)
    monkeypatch.setattr(enrichment, "save_ingestion_manifest", lambda *_args: None)
    monkeypatch.setattr(enrichment, "preflight_ollama", available)
    monkeypatch.setattr(enrichment, "preflight_qdrant", available)
    monkeypatch.setattr(enrichment, "preflight_neo4j", available)
    monkeypatch.setattr(enrichment, "load_indexed_chunks", fake_load)
    monkeypatch.setattr(enrichment, "stage3_and_optional_neo4j_incremental", fake_stage3)
    monkeypatch.setattr(enrichment, "update_qdrant_graph_nodes", no_qdrant_update)

    result = asyncio.run(enrichment.run_semantic_enrichment(source_dir=source_dir))

    assert result["passed"] is True
    assert result["status"] == "completed_with_warnings"
    assert manifest["core_phase1"]["status"] == "completed_validated"
    assert manifest["semantic_enrichment"]["status"] == "completed_with_warnings"
    assert "retryable" in manifest["semantic_enrichment"]["report"]["warning"]


def test_optional_ollama_failure_isolated_from_completed_core(monkeypatch, tmp_path: Path) -> None:
    source_dir, source_path = _source_file(tmp_path)
    source = source_dir / "knowledge.json"
    source_identity = ingestion.canonical_source_identity(source, source_root=source_dir)
    manifest = {
        "core_phase1": {"status": "completed_validated"},
        "full_phase1_validation": {"status": "completed"},
        "documents": {
            source_path: {
                "source_identity": source_identity,
                "status": "completed",
                "core_phase1_status": "completed_validated",
            }
        },
    }

    async def unavailable_ollama() -> bool:
        return False

    monkeypatch.setattr(enrichment, "load_ingestion_manifest", lambda _path: manifest)
    monkeypatch.setattr(enrichment, "save_ingestion_manifest", lambda *_args: None)
    monkeypatch.setattr(enrichment, "preflight_ollama", unavailable_ollama)

    result = asyncio.run(enrichment.run_semantic_enrichment(source_dir=source_dir))

    assert result["passed"] is False
    assert manifest["core_phase1"]["status"] == "completed_validated"
    assert manifest["documents"][source_path]["status"] == "completed"
    assert manifest["semantic_enrichment"]["status"] == "failed"


def test_optional_enrichment_dry_run_is_non_mutating(tmp_path: Path) -> None:
    source_dir, _ = _source_file(tmp_path)
    result = asyncio.run(
        enrichment.run_semantic_enrichment(
            source_dir=source_dir,
            manifest_path=tmp_path / "missing-manifest.json",
            dry_run=True,
        )
    )

    assert result["passed"] is True
    assert result["mutated"] is False
    assert result["plan"]["semantic_enrichment"]["requires_ollama"] is True


def test_core_plan_keeps_deterministic_graph_and_marks_semantic_not_run(tmp_path: Path) -> None:
    source_dir, _ = _source_file(tmp_path)
    plan, _records = full_phase1.build_full_phase1_plan(source_dir)

    plan_data = plan.as_dict()
    assert "deterministic_neo4j_graph" in plan_data["stages"]
    assert plan_data["semantic_enrichment"] == {"status": "not_run", "required": False}
    assert plan.graph_node_count > 0
    assert plan.graph_relationship_count > 0
