from __future__ import annotations

from scripts import inspect_phase2_readiness
from scripts.validate_kb_collections import qdrant_addressable_names


class _Item:
    def __init__(self, **values):
        self.__dict__.update(values)


def test_qdrant_addressable_names_includes_logical_aliases():
    collections = _Item(collections=[_Item(name="acne_knowledge__build")])
    aliases = _Item(aliases=[_Item(alias_name="acne_knowledge", collection_name="acne_knowledge__build")])

    assert qdrant_addressable_names(collections, aliases) == {
        "acne_knowledge__build",
        "acne_knowledge",
    }


def test_display_rerank_provider_recognizes_hybrid_with_model(monkeypatch, tmp_path):
    monkeypatch.setenv("RERANK_PROVIDER", "hybrid")
    monkeypatch.setenv("SEMANTIC_RERANK_MODEL_PATH", str(tmp_path))

    display = inspect_phase2_readiness._display_rerank_provider()

    assert display == "hybrid (semantic model available)"


def test_display_rerank_provider_recognizes_hybrid_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("RERANK_PROVIDER", "hybrid")
    monkeypatch.setenv("SEMANTIC_RERANK_MODEL_PATH", str(tmp_path / "missing"))

    display = inspect_phase2_readiness._display_rerank_provider()

    assert display == "hybrid (semantic model missing; falls back to local_rules)"
