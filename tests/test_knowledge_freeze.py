from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import inspect_runtime_readiness
from scripts.knowledge_build import parse_args
from src.ingestion import pipeline
from src.ingestion.build import BUILD_MANIFEST_SCHEMA


BUILD_ID = "a" * 20
OTHER_BUILD_ID = "b" * 20


def _manifest(*, build_id: str = BUILD_ID, status: str = "activated") -> dict[str, object]:
    return {
        "schema": BUILD_MANIFEST_SCHEMA,
        "status": status,
        "phase1_frozen": False,
        "build_id": build_id,
        "collections": {
            "knowledge_logical": "acne_knowledge",
            "knowledge_physical": f"acne_knowledge__{build_id}",
            "entity_logical": "acne_entities",
            "entity_physical": f"acne_entities__{build_id}",
        },
        "counts": {
            "sources": 4,
            "knowledge_chunks": 512,
            "entities": 32,
            "graph_nodes": 32,
            "graph_relationships": 27,
        },
    }


def _write_manifest(path: Path, **overrides: object) -> bytes:
    manifest = _manifest()
    manifest.update(overrides)
    raw = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _patch_successful_freeze_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    prepared_build_id: str = BUILD_ID,
) -> None:
    monkeypatch.setenv("KB_VERSION", BUILD_ID)
    monkeypatch.setattr(
        pipeline,
        "compute_build_identity",
        lambda *_args, **_kwargs: SimpleNamespace(build_id=prepared_build_id),
    )

    async def inspect_cache(**_kwargs):
        return {
            "passed": True,
            "build_id": BUILD_ID,
            "parsed": {"hits": 4, "misses": 0, "total": 4},
            "knowledge_embeddings": {"hits": 507, "misses": 5, "total": 512},
            "entity_embeddings": {"hits": 32, "misses": 0, "total": 32},
            "provider_calls": 0,
        }

    async def validate_live(_manifest):
        return {"passed": True, "errors": [], "layers": []}

    monkeypatch.setattr(pipeline, "inspect_embedding_cache_reuse", inspect_cache)
    monkeypatch.setattr(pipeline, "_validate_freeze_live_state", validate_live)


def test_runtime_readiness_has_no_hardcoded_expected_build() -> None:
    source = Path(inspect_runtime_readiness.__file__).read_text(encoding="utf-8")

    assert "EXPECTED_BUILD" not in source
    assert "ec0a6de32d58ac181af6" not in source


def test_runtime_readiness_follows_configured_kb_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path / "data" / "knowledge_build_manifest.json", phase1_frozen=True)
    monkeypatch.setattr(inspect_runtime_readiness, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("KB_VERSION", BUILD_ID)

    check = inspect_runtime_readiness._knowledge_manifest_check()

    assert check["passed"] is True
    assert check["details"]["configured_kb_version"] == BUILD_ID


@pytest.mark.parametrize("configured", [OTHER_BUILD_ID, "", "not-a-build-id"])
def test_runtime_readiness_fails_closed_for_mismatched_or_invalid_kb_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configured: str,
) -> None:
    _write_manifest(tmp_path / "data" / "knowledge_build_manifest.json", phase1_frozen=True)
    monkeypatch.setattr(inspect_runtime_readiness, "PROJECT_ROOT", tmp_path)
    if configured:
        monkeypatch.setenv("KB_VERSION", configured)
    else:
        monkeypatch.delenv("KB_VERSION", raising=False)

    check = inspect_runtime_readiness._knowledge_manifest_check()

    assert check["passed"] is False


def test_freeze_cli_is_explicit_operator_command() -> None:
    assert parse_args(["freeze"]).command == "freeze"


def test_freeze_refuses_non_activated_manifest_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    before = _write_manifest(manifest_path, status="completed")
    monkeypatch.setenv("KB_VERSION", BUILD_ID)

    with pytest.raises(RuntimeError, match="activated"):
        asyncio.run(pipeline.freeze_knowledge(manifest_path=manifest_path))

    assert manifest_path.read_bytes() == before


def test_freeze_refuses_configured_build_mismatch_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    before = _write_manifest(manifest_path)
    monkeypatch.setenv("KB_VERSION", OTHER_BUILD_ID)

    with pytest.raises(RuntimeError, match="KB_VERSION"):
        asyncio.run(pipeline.freeze_knowledge(manifest_path=manifest_path))

    assert manifest_path.read_bytes() == before


def test_freeze_fails_closed_when_kb_version_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    before = _write_manifest(manifest_path)
    monkeypatch.delenv("KB_VERSION", raising=False)

    with pytest.raises(RuntimeError, match="KB_VERSION"):
        asyncio.run(pipeline.freeze_knowledge(manifest_path=manifest_path))

    assert manifest_path.read_bytes() == before


def test_freeze_refuses_prepared_identity_mismatch_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    before = _write_manifest(manifest_path)
    _patch_successful_freeze_dependencies(
        monkeypatch,
        prepared_build_id=OTHER_BUILD_ID,
    )

    with pytest.raises(RuntimeError, match="prepared build identity"):
        asyncio.run(pipeline.freeze_knowledge(manifest_path=manifest_path))

    assert manifest_path.read_bytes() == before


def test_freeze_sets_true_only_after_all_validation_passes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    _patch_successful_freeze_dependencies(monkeypatch)

    result = asyncio.run(pipeline.freeze_knowledge(manifest_path=manifest_path))
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result["passed"] is True
    assert result["phase1_frozen"] is True
    assert persisted["phase1_frozen"] is True
    assert persisted["status"] == "activated"
    assert persisted["build_id"] == BUILD_ID


def test_freeze_keeps_manifest_unchanged_when_live_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    before = _write_manifest(manifest_path)
    _patch_successful_freeze_dependencies(monkeypatch)

    async def failed_live_validation(_manifest):
        return {"passed": False, "errors": ["alias mismatch"], "layers": []}

    monkeypatch.setattr(
        pipeline,
        "_validate_freeze_live_state",
        failed_live_validation,
    )

    with pytest.raises(RuntimeError, match="Live knowledge validation failed"):
        asyncio.run(pipeline.freeze_knowledge(manifest_path=manifest_path))

    assert manifest_path.read_bytes() == before


def test_freeze_does_not_use_indexing_or_provider_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    _patch_successful_freeze_dependencies(monkeypatch)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("freeze must not call a write/provider path")

    async def forbidden_async(*_args, **_kwargs):
        raise AssertionError("freeze must not call a write/provider path")

    monkeypatch.setattr(pipeline, "prepare_knowledge", forbidden_async)
    monkeypatch.setattr(pipeline, "load_or_parse_source", forbidden_async)
    monkeypatch.setattr(pipeline, "resolve_embeddings", forbidden_async)
    monkeypatch.setattr(pipeline, "build_knowledge_candidate", forbidden_async)
    monkeypatch.setattr(pipeline, "build_entity_candidate", forbidden_async)
    monkeypatch.setattr(pipeline, "replace_entity_graph", forbidden_async)
    monkeypatch.setattr(pipeline, "switch_alias", forbidden_async)
    monkeypatch.setattr(pipeline.EmbeddingCache, "put", forbidden)

    result = asyncio.run(pipeline.freeze_knowledge(manifest_path=manifest_path))

    assert result["passed"] is True
    freeze_source = inspect.getsource(pipeline.freeze_knowledge)
    assert "prepare_knowledge(" not in freeze_source
    assert "load_or_parse_source(" not in freeze_source
    assert "resolve_embeddings(" not in freeze_source


def test_live_freeze_validation_rejects_physical_collection_from_another_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    manifest["collections"] = {
        "knowledge_logical": "acne_knowledge",
        "knowledge_physical": f"acne_knowledge__{OTHER_BUILD_ID}",
        "entity_logical": "acne_entities",
        "entity_physical": f"acne_entities__{OTHER_BUILD_ID}",
    }

    class FakeClient:
        async def get_aliases(self):
            return SimpleNamespace(
                aliases=[
                    SimpleNamespace(
                        alias_name="acne_knowledge",
                        collection_name=f"acne_knowledge__{OTHER_BUILD_ID}",
                    ),
                    SimpleNamespace(
                        alias_name="acne_entities",
                        collection_name=f"acne_entities__{OTHER_BUILD_ID}",
                    ),
                ]
            )

        async def close(self) -> None:
            return None

    async def passing_collection_validation(*_args, **_kwargs):
        return {"layer": "qdrant", "passed": True, "errors": []}

    async def passing_graph_validation(_build_id):
        return {"layer": "neo4j_build", "passed": True, "errors": []}

    monkeypatch.setattr(pipeline, "AsyncQdrantClient", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(
        pipeline,
        "validate_qdrant_collection",
        passing_collection_validation,
    )
    monkeypatch.setattr(
        pipeline,
        "_inspect_freeze_graph_state",
        passing_graph_validation,
    )

    result = asyncio.run(pipeline._validate_freeze_live_state(manifest))

    assert result["passed"] is False
    assert any("activated build" in error for error in result["errors"])


def test_live_freeze_validation_requires_canonical_logical_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    manifest["collections"] = {
        "knowledge_logical": "other_knowledge",
        "knowledge_physical": f"other_knowledge__{BUILD_ID}",
        "entity_logical": "other_entities",
        "entity_physical": f"other_entities__{BUILD_ID}",
    }

    class FakeClient:
        async def get_aliases(self):
            return SimpleNamespace(
                aliases=[
                    SimpleNamespace(
                        alias_name="other_knowledge",
                        collection_name=f"other_knowledge__{BUILD_ID}",
                    ),
                    SimpleNamespace(
                        alias_name="other_entities",
                        collection_name=f"other_entities__{BUILD_ID}",
                    ),
                ]
            )

        async def close(self) -> None:
            return None

    async def passing_collection_validation(*_args, **_kwargs):
        return {"layer": "qdrant", "passed": True, "errors": []}

    async def passing_graph_validation(_build_id):
        return {"layer": "neo4j_build", "passed": True, "errors": []}

    monkeypatch.setattr(pipeline, "AsyncQdrantClient", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(
        pipeline,
        "validate_qdrant_collection",
        passing_collection_validation,
    )
    monkeypatch.setattr(
        pipeline,
        "_inspect_freeze_graph_state",
        passing_graph_validation,
    )

    result = asyncio.run(pipeline._validate_freeze_live_state(manifest))

    assert result["passed"] is False
    assert any("logical collection" in error for error in result["errors"])


def test_graph_freeze_validation_rejects_mixed_build_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def __init__(self, record: dict[str, int]) -> None:
            self.record = record

        async def single(self):
            return self.record

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def run(self, query: str, **_kwargs):
            if "MATCH (n)" in query:
                return FakeResult({"total": 32, "matching": 31})
            return FakeResult({"total": 27, "matching": 27})

    class FakeDriver:
        def session(self, **_kwargs):
            return FakeSession()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(pipeline, "get_neo4j_driver", lambda: FakeDriver())

    result = asyncio.run(pipeline._inspect_freeze_graph_state(BUILD_ID))

    assert result["passed"] is False
    assert result["nodes"] == 32
    assert result["relationships"] == 27
    assert "build identity" in result["errors"][0]
