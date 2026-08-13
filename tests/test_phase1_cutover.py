from __future__ import annotations

from pathlib import Path

import pytest

from scripts.phase1 import parse_args
from src.ingestion.pipeline import _verify_rollback_artifacts


def test_cutover_requires_qdrant_and_neo4j_rollback_artifacts(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Qdrant snapshots"):
        _verify_rollback_artifacts(tmp_path)

    snapshots = tmp_path / "qdrant"
    snapshots.mkdir()
    (snapshots / "knowledge.snapshot").write_bytes(b"knowledge")
    (snapshots / "entities.snapshot").write_bytes(b"entities")
    with pytest.raises(RuntimeError, match="Neo4j cold backup"):
        _verify_rollback_artifacts(tmp_path)

    database = tmp_path / "neo4j" / "data" / "databases" / "neo4j"
    database.mkdir(parents=True)
    (database / "store").write_bytes(b"neo4j")
    _verify_rollback_artifacts(tmp_path)


def test_canonical_build_cli_exposes_guarded_activation() -> None:
    root = Path("data/backups/pre-s4a")
    args = parse_args(["build", "--activate", "--rollback-root", str(root)])

    assert args.command == "build"
    assert args.activate is True
    assert args.rollback_root == root
