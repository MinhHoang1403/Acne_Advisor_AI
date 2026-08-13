#!/usr/bin/env python3
"""
ingest_knowledge.py
===================
Acne Advisor AI – Knowledge Base Ingestion Pipeline
===================================================

Pipeline
--------
[STAGE 1] PDF EXTRACTION
  - Discover PDF files from SAMPLE_DATA_DIR or --source.
  - Convert PDFs to Markdown via LlamaParse.
  - Cache Markdown output to data/cache/markdown so the same PDF is not parsed again.

[STAGE 2] MARKDOWN CHUNKING
  - Split Markdown by headers (#, ##, ###).
  - Long sections are split with RecursiveCharacterTextSplitter.
  - Supports --limit-chunks for test runs.

[STAGE 3] GRAPH EXTRACTION
  - Use Ollama/Qwen3 to extract clinical nodes/edges from each chunk.
  - Cache graph extraction per chunk to data/cache/graph.
  - Supports resume: already cached chunks are skipped.
  - Supports --skip-graph-extraction to load cached graph only.
  - Writes Neo4j incrementally by graph batch instead of waiting for all chunks.

[STAGE 4A] NEO4J GRAPH INDEXING
  - MERGE nodes and edges.
  - Idempotent: safe to rerun.

[STAGE 4B] QDRANT VECTOR INDEXING
  - Dense vector: Google Gemini embedding.
  - Sparse vector: deterministic hashed sparse vector for Qdrant.
  - Fixes previous rank_bm25 IndexError.
  - Idempotent upsert.

Examples
--------
python scripts/ingest_knowledge.py --dry-run
python scripts/ingest_knowledge.py --limit-files 1 --limit-chunks 50 --dry-run
python scripts/ingest_knowledge.py
python scripts/ingest_knowledge.py --skip-graph-extraction --skip-neo4j
python scripts/ingest_knowledge.py --refresh-markdown --no-resume

Environment Variables
---------------------
LLAMA_CLOUD_API_KEY
OLLAMA_BASE_URL
OLLAMA_MODEL
GOOGLE_API_KEY
EMBEDDING_MODEL
EMBEDDING_DIMENSIONS
QDRANT_URL
QDRANT_COLLECTION_NAME
NEO4J_URI
NEO4J_USERNAME
NEO4J_PASSWORD
SAMPLE_DATA_DIR
CHUNK_SIZE
LLM_CONCURRENCY
INGEST_BATCH_SIZE
GRAPH_BATCH_SIZE
LOG_LEVEL
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import uuid
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any


# =============================================================================
# Bootstrap
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

# Phase 1.5 — Dermatology-Aware Chunking
from src.ingestion.cleanup import (
    CLEANUP_VERSION,
    build_qdrant_cleanup_plan,
    cleanup_previous_qdrant_points,
)
from src.ingestion.chunking import naive_split
from src.ingestion.domain_metadata import enrich_domain_metadata, extract_dermatology_metadata
from src.ingestion.filtering import is_noisy_chunk, is_short_medical_safety_statement
from src.ingestion.json_loader import load_web_json_documents_with_stats
from src.ingestion.sparse_legacy import (
    SPARSE_VECTOR_SCHEMA_VERSION,
    compute_sparse_vectors,
    token_to_sparse_index,
    tokenize_for_sparse,
)
from src.integrations.google_genai import embed_texts_sync
from src.knowledge.versioning import expected_kb_payload_metadata


# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("ingest_knowledge")


# =============================================================================
# Configuration
# =============================================================================

LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY", "")

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))

QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge")

NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

SAMPLE_DATA_DIR: Path = Path(
    os.getenv("SAMPLE_DATA_DIR", str(PROJECT_ROOT / "sample_data"))
)

DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
CACHE_DIR: Path = DATA_DIR / "cache"
MARKDOWN_CACHE_DIR: Path = CACHE_DIR / "markdown"
GRAPH_CACHE_DIR: Path = CACHE_DIR / "graph"
DEFAULT_MANIFEST_PATH: Path = DATA_DIR / "ingestion_manifest.json"
INGESTION_MANIFEST_VERSION: int = 1
INGESTION_CONFIG_SCHEMA_VERSION = "phase1_ingestion_config_v1"
SEMANTIC_ENRICHMENT_VERSION = "semantic_graph_enrichment_v1"

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "2000"))
LLM_CONCURRENCY: int = int(os.getenv("LLM_CONCURRENCY", "2"))
INGEST_BATCH_SIZE: int = int(os.getenv("INGEST_BATCH_SIZE", "16"))
GRAPH_BATCH_SIZE: int = int(os.getenv("GRAPH_BATCH_SIZE", "50"))

GRAPH_CACHE_VERSION: str = os.getenv("GRAPH_CACHE_VERSION", "v2")
GRAPH_PROMPT_SCHEMA_VERSION: str = os.getenv(
    "GRAPH_PROMPT_SCHEMA_VERSION",
    "clinical_graph_prompt_v2",
)
GRAPH_ERROR_TOLERANCE_RATIO: float = float(os.getenv("GRAPH_ERROR_TOLERANCE_RATIO", "0.02"))
GRAPH_ERROR_TOLERANCE_MAX: int = int(os.getenv("GRAPH_ERROR_TOLERANCE_MAX", "3"))

# Gemini embedding free-tier rate limit protection.
# Gemini embedding quota can be counted per text inside a batch. With 16 texts/batch,
# a 12s delay keeps the pipeline below roughly 100 embed contents/minute.
EMBEDDING_BATCH_DELAY: float = float(os.getenv("EMBEDDING_BATCH_DELAY", "12"))
EMBEDDING_MAX_RETRIES: int = int(os.getenv("EMBEDDING_MAX_RETRIES", "10"))
EMBEDDING_RETRY_BASE_DELAY: float = float(os.getenv("EMBEDDING_RETRY_BASE_DELAY", "8"))

ENTITY_TYPES = ["DISEASE", "DRUG", "SYMPTOM", "TREATMENT", "MECHANISM", "BODY_PART"]
RELATION_TYPES = ["CAUSES", "TREATS", "CONTRAINDICATES", "PART_OF"]

LLAMAPARSE_INSTRUCTION = (
    "This document is a medical/dermatology text about acne (mụn trứng cá) and "
    "related skin conditions. Preserve all tables, bullet lists, and section headers "
    "exactly. Output well-structured Markdown."
)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class SemanticChunk:
    source_file: str
    chunk_index: int
    text: str
    header_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    @property
    def document_identity(self) -> str:
        document_id = self.metadata.get("document_id")
        if document_id:
            return str(document_id)

        source_path = self.metadata.get("source_path")
        if source_path:
            return document_id_from_source_path(str(source_path))

        return self.source_file

    @property
    def chunk_id(self) -> str:
        key = f"{self.document_identity}::{self.chunk_index}::{self.content_hash}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]

    @property
    def qdrant_point_id(self) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, self.chunk_id))


@dataclass
class GraphNode:
    name: str
    entity_type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source_name: str
    source_type: str
    target_name: str
    target_type: str
    relation: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphPayload:
    chunk_id: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    from_cache: bool = False
    extraction_error: bool = False


@dataclass
class Neo4jUpsertResult:
    nodes_upserted: int = 0
    relationships_upserted: int = 0
    missing_source_endpoint: int = 0
    missing_target_endpoint: int = 0
    database_errors: int = 0

    @property
    def has_failures(self) -> bool:
        return any(
            (
                self.missing_source_endpoint,
                self.missing_target_endpoint,
                self.database_errors,
            )
        )


@dataclass(frozen=True)
class ParserCompleteness:
    """Structural parser validation without assuming provider-specific page metadata."""

    passed: bool
    status: str
    error: str | None = None


@dataclass
class IngestionStats:
    pdf_files: int = 0
    markdown_cache_hits: int = 0
    markdown_cache_misses: int = 0
    chunks_created: int = 0
    graph_cache_hits: int = 0
    graph_cache_misses: int = 0
    graph_cache_invalid: int = 0
    graph_cache_invalid_json: int = 0
    graph_cache_invalid_schema: int = 0
    graph_cache_invalid_mismatch: int = 0
    nodes_extracted: int = 0
    edges_extracted: int = 0
    nodes_upserted_neo4j: int = 0
    edges_upserted_neo4j: int = 0
    vectors_upserted_qdrant: int = 0
    llm_errors: int = 0
    parse_errors: int = 0
    chunks_rejected_noise: int = 0
    chunks_rescued_short_safety: int = 0
    chunks_accepted_normal: int = 0
    neo4j_missing_endpoints: int = 0

    def report(self) -> None:
        logger.info("─" * 56)
        logger.info("  Ingestion Statistics")
        logger.info("─" * 56)
        logger.info("  PDF files parsed          : %d", self.pdf_files)
        logger.info("  Markdown cache hits       : %d", self.markdown_cache_hits)
        logger.info("  Markdown cache misses     : %d", self.markdown_cache_misses)
        logger.info("  Semantic chunks           : %d", self.chunks_created)
        logger.info("  Graph cache hits          : %d", self.graph_cache_hits)
        logger.info("  Graph cache misses        : %d", self.graph_cache_misses)
        logger.info("  Graph cache invalid       : %d", self.graph_cache_invalid)
        logger.info("    - JSON errors           : %d", self.graph_cache_invalid_json)
        logger.info("    - Schema errors         : %d", self.graph_cache_invalid_schema)
        logger.info("    - Version/model/hash    : %d", self.graph_cache_invalid_mismatch)
        logger.info("  Graph nodes extracted     : %d", self.nodes_extracted)
        logger.info("  Graph edges extracted     : %d", self.edges_extracted)
        logger.info("  Neo4j nodes upserted      : %d", self.nodes_upserted_neo4j)
        logger.info("  Neo4j edges upserted      : %d", self.edges_upserted_neo4j)
        logger.info("  Qdrant vectors upserted   : %d", self.vectors_upserted_qdrant)
        logger.info("  LLM extraction errors     : %d", self.llm_errors)
        logger.info("  Parse errors              : %d", self.parse_errors)
        logger.info("  Chunks rejected as noise  : %d", self.chunks_rejected_noise)
        logger.info("  Short safety chunks kept  : %d", self.chunks_rescued_short_safety)
        logger.info("  Normal chunks accepted    : %d", self.chunks_accepted_normal)
        logger.info("  Neo4j missing endpoints  : %d", self.neo4j_missing_endpoints)
        logger.info("─" * 56)


# =============================================================================
# Cache utilities
# =============================================================================

def ensure_cache_dirs() -> None:
    MARKDOWN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def discover_source_documents(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []

    files: list[Path] = []
    for pattern in ("*.pdf", "*.docx", "*.json"):
        files.extend(source_dir.rglob(pattern))

    return sorted(files)


def discover_knowledge_files(source_dir: Path) -> list[Path]:
    return discover_source_documents(source_dir)


def is_web_json_source(path: Path) -> bool:
    return path.suffix.lower() == ".json"


# =============================================================================
# Incremental ingestion manifest
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_file_hash(path: Path) -> str:
    """Return a SHA256 hash of the real file content."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _source_path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def canonical_source_identity(
    source_path: str | Path,
    *,
    source_root: Path | None = None,
) -> str:
    """Return a portable source identity instead of an absolute machine path.

    The normal Phase 1 source directory is ``sample_data``. Retaining that
    directory marker makes identities stable across checkouts on different
    drives while preserving nested paths. Callers with an arbitrary source
    directory can pass ``source_root`` to retain the same property.
    """

    raw = str(source_path).replace("\\", "/")
    path = Path(source_path)
    if source_root is not None:
        try:
            relative = path.resolve().relative_to(source_root.resolve())
            return (PurePosixPath(source_root.name) / PurePosixPath(relative)).as_posix()
        except (OSError, ValueError):
            pass

    try:
        relative_to_project = path.resolve().relative_to(PROJECT_ROOT.resolve())
        return PurePosixPath(relative_to_project).as_posix()
    except (OSError, ValueError):
        pass

    windows_parts = PureWindowsPath(raw).parts
    posix_parts = PurePosixPath(raw).parts
    parts = windows_parts if len(windows_parts) > len(posix_parts) else posix_parts
    marker_index = next(
        (index for index, part in enumerate(parts) if part.lower() == "sample_data"),
        None,
    )
    if marker_index is not None:
        return PurePosixPath(*parts[marker_index:]).as_posix()

    # External sources do not have a repository anchor. Preserve their final
    # relative shape rather than a drive-specific path; source roots should be
    # provided by production callers for full uniqueness.
    return PurePath(raw).name


def document_id_from_source_path(source_path: str, *, source_root: Path | None = None) -> str:
    identity = canonical_source_identity(source_path, source_root=source_root)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"acne-advisor-ai:{identity}"))


def web_json_record_document_id(
    source_path: str,
    record_index: int,
    source_url: str | None = None,
) -> str:
    identity = (source_url or "").strip() or f"record:{record_index}"
    source_identity = canonical_source_identity(source_path)
    key = f"acne-advisor-ai:web_json:{source_identity}:{identity}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _file_modified_time_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _manifest_base() -> dict[str, Any]:
    return {
        "manifest_version": INGESTION_MANIFEST_VERSION,
        "updated_at": None,
        "documents": {},
        "core_phase1": {
            "status": "not_validated",
            "updated_at": None,
        },
        "full_phase1_validation": {
            "status": "not_validated",
            "updated_at": None,
        },
        "semantic_enrichment": {
            "status": "not_run",
            "updated_at": None,
            "fingerprint": None,
        },
    }


def ingestion_config_fingerprint() -> str:
    """Fingerprint only material Phase 1 derivation settings."""

    config = {
        "config_schema_version": INGESTION_CONFIG_SCHEMA_VERSION,
        "ingestion_pipeline_version": expected_kb_payload_metadata().get(
            "ingestion_pipeline_version"
        ),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": max(100, CHUNK_SIZE // 5),
        "parser_result_type": "markdown",
        "parser_instruction": LLAMAPARSE_INSTRUCTION,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "sparse_vector_schema_version": SPARSE_VECTOR_SCHEMA_VERSION,
    }
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def semantic_enrichment_fingerprint() -> str:
    """Fingerprint only stochastic document semantic-enrichment settings."""

    config = {
        "semantic_enrichment_version": SEMANTIC_ENRICHMENT_VERSION,
        "graph_cache_version": GRAPH_CACHE_VERSION,
        "graph_prompt_schema_version": GRAPH_PROMPT_SCHEMA_VERSION,
        "ollama_model": OLLAMA_MODEL,
        "entity_types": ENTITY_TYPES,
        "relation_types": RELATION_TYPES,
    }
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def load_ingestion_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _manifest_base()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid ingestion manifest JSON at {path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read ingestion manifest at {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Ingestion manifest must be a JSON object: {path}")

    data.setdefault("manifest_version", INGESTION_MANIFEST_VERSION)
    data.setdefault("updated_at", None)
    data.setdefault(
        "core_phase1",
        {"status": "not_validated", "updated_at": None},
    )
    data.setdefault(
        "full_phase1_validation",
        {"status": "not_validated", "updated_at": None},
    )
    data.setdefault(
        "semantic_enrichment",
        {"status": "not_run", "updated_at": None, "fingerprint": None},
    )
    documents = data.setdefault("documents", {})
    if not isinstance(documents, dict):
        raise ValueError(f"Ingestion manifest 'documents' must be an object: {path}")

    return data


def save_ingestion_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["manifest_version"] = INGESTION_MANIFEST_VERSION
    data["updated_at"] = utc_now_iso()

    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                logger.warning("[MANIFEST] Failed to remove temp manifest %s", tmp_path.name)


def finalize_full_phase1_manifest(
    manifest_path: Path,
    *,
    validation: dict[str, Any],
) -> None:
    """Record the full-build gate and complete only validated pending sources."""

    manifest = load_ingestion_manifest(manifest_path)
    passed = bool(validation.get("passed"))
    now = utc_now_iso()
    manifest["core_phase1"] = {
        "status": "completed_validated" if passed else "failed",
        "updated_at": now,
        "report": validation,
    }
    manifest["full_phase1_validation"] = {
        "status": "completed" if passed else "failed",
        "updated_at": now,
        "report": validation,
    }
    semantic_state = manifest.setdefault(
        "semantic_enrichment",
        {"status": "not_run", "updated_at": None, "fingerprint": None},
    )
    if not isinstance(semantic_state, dict):
        manifest["semantic_enrichment"] = {
            "status": "not_run",
            "updated_at": None,
            "fingerprint": None,
        }

    for entry in _manifest_documents(manifest).values():
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "knowledge_indexed_pending_phase1_validation":
            continue
        if passed:
            entry["status"] = "completed"
            entry["core_phase1_status"] = "completed_validated"
            entry["graph_validated"] = False
            entry["phase1_validation_status"] = "completed_full_phase1"
            entry.setdefault(
                "semantic_enrichment",
                {"status": "not_run", "updated_at": now, "fingerprint": None},
            )
            entry.pop("error_message", None)
        else:
            entry["core_phase1_status"] = "failed"
            entry["phase1_validation_status"] = "full_phase1_validation_failed"
            entry["error_message"] = _short_error_message(
                "Full Phase 1 validation failed; source remains retryable."
            )
        entry["last_full_phase1_validation_at"] = now

    save_ingestion_manifest(manifest_path, manifest)


def update_semantic_enrichment_manifest(
    manifest: dict[str, Any],
    *,
    status: str,
    report: dict[str, Any] | None = None,
    document_results: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Record optional semantic-enrichment state without changing core validity."""

    now = utc_now_iso()
    root_state: dict[str, Any] = {
        "status": status,
        "updated_at": now,
        "fingerprint": semantic_enrichment_fingerprint(),
    }
    if report is not None:
        root_state["report"] = report
    manifest["semantic_enrichment"] = root_state

    for source_path, result in (document_results or {}).items():
        entry = _manifest_documents(manifest).get(source_path)
        if not isinstance(entry, dict):
            continue
        entry["semantic_enrichment"] = {
            "status": str(result.get("status") or status),
            "updated_at": now,
            "fingerprint": semantic_enrichment_fingerprint(),
            **{
                key: value
                for key, value in result.items()
                if key != "status" and value is not None
            },
        }


def _manifest_documents(manifest: dict[str, Any]) -> dict[str, Any]:
    documents = manifest.setdefault("documents", {})
    if not isinstance(documents, dict):
        raise ValueError("Ingestion manifest 'documents' must be an object")
    return documents


def _find_manifest_entry(
    manifest: dict[str, Any],
    source_path: str,
) -> tuple[str | None, dict[str, Any] | None]:
    documents = _manifest_documents(manifest)
    direct = documents.get(source_path)
    if isinstance(direct, dict):
        return source_path, direct

    for key, entry in documents.items():
        if isinstance(entry, dict) and entry.get("source_path") == source_path:
            return str(key), entry

    return None, None


def _file_manifest_info(path: Path, *, source_root: Path | None = None) -> dict[str, Any]:
    source_path = _source_path_key(path)
    stat = path.stat()
    source_identity = canonical_source_identity(path, source_root=source_root)
    return {
        "path": path,
        "source_path": source_path,
        "source_identity": source_identity,
        "source_file": path.name,
        "source_type": "web_json" if is_web_json_source(path) else "source_document",
        "document_id": document_id_from_source_path(source_identity),
        "content_hash": compute_file_hash(path),
        "file_size": stat.st_size,
        "modified_time": _file_modified_time_iso(path),
    }


def _dedupe_manifest_point_ids(qdrant_point_ids: list[str] | None) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for raw_id in qdrant_point_ids or []:
        point_id = str(raw_id).strip()
        if not point_id or point_id in seen:
            continue
        seen.add(point_id)
        deduped.append(point_id)
    return deduped


def _manifest_qdrant_metadata(qdrant_point_ids: list[str] | None) -> dict[str, Any]:
    point_ids = _dedupe_manifest_point_ids(qdrant_point_ids)
    return {
        "qdrant_collection": QDRANT_COLLECTION_NAME,
        "qdrant_point_ids": point_ids,
        "qdrant_point_count": len(point_ids),
        "cleanup_version": CLEANUP_VERSION,
        **expected_kb_payload_metadata(),
    }


def _manifest_source_metadata(source_metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not source_metadata:
        return {}

    fields = (
        "source_identity",
        "source_type",
        "document_count",
        "json_record_count",
        "skipped_record_count",
        "parser_completeness",
        "parser_validation_error",
        "graph_extraction_skipped",
        "neo4j_skipped",
        "qdrant_indexed",
        "graph_validated",
        "phase1_validation_status",
    )
    output: dict[str, Any] = {}
    for field_name in fields:
        if field_name in source_metadata:
            output[field_name] = source_metadata[field_name]
    return output


def get_incremental_file_plan(
    files: list[Path],
    manifest: dict[str, Any],
    force_reingest: bool = False,
    source_root: Path | None = None,
) -> dict[str, Any]:
    scanned: list[dict[str, Any]] = []
    to_ingest: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    summary = {
        "scanned": 0,
        "new": 0,
        "changed": 0,
        "retry": 0,
        "force": 0,
        "config_changed": 0,
        "skipped": 0,
        "to_ingest": 0,
    }

    for path in files:
        info = _file_manifest_info(path, source_root=source_root)
        manifest_key, previous = _find_manifest_entry(manifest, info["source_path"])
        previous_status = str(previous.get("status", "")) if previous else None
        previous_hash = str(previous.get("content_hash", "")) if previous else None

        info["manifest_key"] = manifest_key
        info["previous_status"] = previous_status
        info["previous_content_hash"] = previous_hash
        info["previous_manifest_record"] = previous

        if force_reingest:
            reason = "force"
        elif previous is None:
            reason = "new"
        elif previous_status in {"failed", "partial", "cleanup_failed"}:
            if previous_hash != info["content_hash"]:
                reason = "changed"
            else:
                reason = "retry"
        elif previous_hash != info["content_hash"]:
            reason = "changed"
        elif previous.get("ingestion_config_fingerprint") != ingestion_config_fingerprint():
            reason = "config_changed"
        elif previous.get("core_phase1_status") == "completed_validated":
            reason = "skip"
        elif previous_status == "completed" and previous.get("graph_validated") is True:
            reason = "skip"
        else:
            reason = "retry"

        info["reason"] = reason
        info["cleanup_required"] = reason in {"changed", "config_changed", "retry", "force"} and previous is not None
        scanned.append(info)

        if reason == "skip":
            skipped.append(info)
            summary["skipped"] += 1
        else:
            to_ingest.append(info)
            summary["to_ingest"] += 1
            summary[reason] += 1

    summary["scanned"] = len(scanned)
    return {
        "scanned": scanned,
        "to_ingest": to_ingest,
        "skipped": skipped,
        "summary": summary,
    }


def log_incremental_file_plan(plan: dict[str, Any]) -> None:
    summary = plan["summary"]
    logger.info("[INCREMENTAL] Files scanned          : %d", summary["scanned"])
    logger.info("[INCREMENTAL] New files              : %d", summary["new"])
    logger.info("[INCREMENTAL] Changed files          : %d", summary["changed"])
    logger.info("[INCREMENTAL] Failed/partial retries : %d", summary["retry"])
    logger.info("[INCREMENTAL] Force reingest files   : %d", summary["force"])
    logger.info("[INCREMENTAL] Config changed files   : %d", summary["config_changed"])
    logger.info("[INCREMENTAL] Skipped unchanged      : %d", summary["skipped"])
    logger.info("[INCREMENTAL] Files to ingest        : %d", summary["to_ingest"])

    for item in plan["skipped"]:
        logger.info("[INCREMENTAL] Skip unchanged: %s", item["source_path"])

    for item in plan["to_ingest"]:
        logger.info(
            "[INCREMENTAL] Queue %s (%s)",
            item["source_path"],
            item["reason"],
        )


def _short_error_message(error: Exception | str, max_length: int = 500) -> str:
    message = str(error).replace("\r", " ").replace("\n", " ").strip()
    if len(message) <= max_length:
        return message
    return message[: max_length - 3] + "..."


def update_manifest_before_ingest(
    manifest: dict[str, Any],
    file_info: dict[str, Any],
    ingestion_run_id: str,
) -> None:
    documents = _manifest_documents(manifest)
    source_path = str(file_info["source_path"])
    previous = documents.get(source_path)
    previous_chunk_count = 0
    previous_qdrant_ids: list[str] = []
    if isinstance(previous, dict):
        previous_chunk_count = int(previous.get("chunk_count", 0) or 0)
        raw_ids = previous.get("qdrant_point_ids", [])
        if isinstance(raw_ids, list):
            previous_qdrant_ids = [str(point_id) for point_id in raw_ids]

    documents[source_path] = {
        **{
            key: file_info[key]
            for key in (
                "source_path",
                "source_identity",
                "document_id",
                "content_hash",
                "file_size",
                "modified_time",
            )
        },
        "source_file": str(file_info.get("source_file") or Path(source_path).name),
        **_manifest_source_metadata(file_info),
        "status": "partial",
        "chunk_count": previous_chunk_count,
        **_manifest_qdrant_metadata(previous_qdrant_ids),
        "last_ingested_at": None,
        "ingestion_run_id": ingestion_run_id,
        "error_message": "ingestion started but has not completed",
        "ingestion_config_fingerprint": ingestion_config_fingerprint(),
        "core_phase1_status": "pending",
        "semantic_enrichment": (
            dict(previous.get("semantic_enrichment", {}))
            if isinstance(previous, dict) and isinstance(previous.get("semantic_enrichment"), dict)
            else {"status": "not_run", "updated_at": None, "fingerprint": None}
        ),
        "graph_validated": False,
        "phase1_validation_status": "pending",
    }


def update_manifest_after_success(
    manifest: dict[str, Any],
    source_path: str,
    document_id: str,
    content_hash: str,
    file_size: int,
    modified_time: str,
    chunk_count: int,
    qdrant_point_ids: list[str] | None,
    ingestion_run_id: str,
    last_ingested_at: str | None = None,
    status: str = "completed",
    error_message: str | None = None,
    graph_error_count: int | None = None,
    graph_error_tolerance: float | None = None,
    warning: str | None = None,
    source_metadata: dict[str, Any] | None = None,
    graph_validated: bool = False,
    phase1_validation_status: str = "knowledge_only",
    core_phase1_status: str = "knowledge_indexed",
) -> None:
    documents = _manifest_documents(manifest)
    previous = documents.get(source_path)
    qdrant_metadata = _manifest_qdrant_metadata(qdrant_point_ids)
    entry: dict[str, Any] = {
        "source_path": source_path,
        "source_file": Path(source_path).name,
        **_manifest_source_metadata(source_metadata),
        "document_id": document_id,
        "content_hash": content_hash,
        "file_size": file_size,
        "modified_time": modified_time,
        "status": status,
        "chunk_count": chunk_count,
        **qdrant_metadata,
        "last_ingested_at": last_ingested_at or utc_now_iso(),
        "ingestion_run_id": ingestion_run_id,
        "ingestion_config_fingerprint": ingestion_config_fingerprint(),
        "core_phase1_status": core_phase1_status,
        "semantic_enrichment": (
            dict(previous.get("semantic_enrichment", {}))
            if isinstance(previous, dict) and isinstance(previous.get("semantic_enrichment"), dict)
            else {"status": "not_run", "updated_at": None, "fingerprint": None}
        ),
        "graph_validated": graph_validated,
        "phase1_validation_status": phase1_validation_status,
    }
    if error_message:
        entry["error_message"] = _short_error_message(error_message)
    if graph_error_count is not None:
        entry["graph_error_count"] = graph_error_count
    if graph_error_tolerance is not None:
        entry["graph_error_tolerance"] = graph_error_tolerance
    if warning:
        entry["warning"] = warning
    documents[source_path] = entry


def update_manifest_after_failure(
    manifest: dict[str, Any],
    file_info: dict[str, Any],
    ingestion_run_id: str,
    error: Exception | str,
    status: str = "failed",
    chunk_count: int | None = None,
    qdrant_point_ids: list[str] | None = None,
    last_ingested_at: str | None = None,
    graph_error_count: int | None = None,
    graph_error_tolerance: float | None = None,
    warning: str | None = None,
) -> None:
    documents = _manifest_documents(manifest)
    source_path = str(file_info["source_path"])
    previous = documents.get(source_path)
    previous_qdrant_ids = []
    if isinstance(previous, dict):
        raw_ids = previous.get("qdrant_point_ids", [])
        if isinstance(raw_ids, list):
            previous_qdrant_ids = [str(point_id) for point_id in raw_ids]

    manifest_chunk_count = (
        chunk_count
        if chunk_count is not None
        else int(previous.get("chunk_count", 0) or 0) if isinstance(previous, dict) else 0
    )
    manifest_qdrant_point_ids = (
        qdrant_point_ids
        if qdrant_point_ids is not None
        else previous_qdrant_ids
    )
    qdrant_metadata = _manifest_qdrant_metadata(manifest_qdrant_point_ids)

    entry: dict[str, Any] = {
        "source_path": source_path,
        "source_file": str(file_info.get("source_file") or Path(source_path).name),
        **_manifest_source_metadata(file_info),
        "document_id": str(file_info["document_id"]),
        "content_hash": str(file_info["content_hash"]),
        "file_size": int(file_info["file_size"]),
        "modified_time": str(file_info["modified_time"]),
        "status": status,
        "chunk_count": manifest_chunk_count,
        **qdrant_metadata,
        "last_ingested_at": last_ingested_at or utc_now_iso(),
        "ingestion_run_id": ingestion_run_id,
        "error_message": _short_error_message(error),
        "ingestion_config_fingerprint": ingestion_config_fingerprint(),
        "core_phase1_status": "failed",
        "semantic_enrichment": (
            dict(previous.get("semantic_enrichment", {}))
            if isinstance(previous, dict) and isinstance(previous.get("semantic_enrichment"), dict)
            else {"status": "not_run", "updated_at": None, "fingerprint": None}
        ),
        "graph_validated": False,
        "phase1_validation_status": "failed",
    }
    if graph_error_count is not None:
        entry["graph_error_count"] = graph_error_count
    if graph_error_tolerance is not None:
        entry["graph_error_tolerance"] = graph_error_tolerance
    if warning:
        entry["warning"] = warning
    documents[source_path] = entry


def enrich_chunks_with_ingestion_metadata(
    chunks: list[SemanticChunk],
    file_info: dict[str, Any],
    ingestion_run_id: str,
    ingested_at: str,
) -> None:
    for chunk in chunks:
        document_id = str(chunk.metadata.get("document_id") or file_info["document_id"])
        source_type = str(chunk.metadata.get("source_type") or file_info.get("source_type") or "")
        chunk.metadata.update(
            {
                "document_id": document_id,
                "file_document_id": str(file_info["document_id"]),
                "source_type": source_type,
                "source_file": str(file_info.get("source_file") or chunk.source_file),
                "source_path": str(file_info["source_path"]),
                "source_identity": str(file_info.get("source_identity") or ""),
                "content_hash": str(file_info["content_hash"]),
                "file_size": int(file_info["file_size"]),
                "modified_time": str(file_info["modified_time"]),
                "ingestion_run_id": ingestion_run_id,
                "ingested_at": ingested_at,
                **expected_kb_payload_metadata(),
            }
        )
        chunk.metadata.update(
            {
                "chunk_index": chunk.chunk_index,
                "chunk_id": chunk.chunk_id,
                "chunk_hash": chunk.content_hash,
            }
        )


def qdrant_point_ids_for_chunks(chunks: list[SemanticChunk]) -> list[str]:
    return [
        chunk.qdrant_point_id
        for chunk in chunks
        if not chunk.metadata.get("is_noisy", False)
    ]


def account_chunk_quality(chunks: list[SemanticChunk], stats: IngestionStats) -> None:
    """Record mutually exclusive eligibility outcomes for rebuild reconciliation."""

    for chunk in chunks:
        if chunk.metadata.get("is_noisy", False):
            stats.chunks_rejected_noise += 1
        elif chunk.metadata.get("is_short_medical_safety_rescue", False):
            stats.chunks_rescued_short_safety += 1
        else:
            stats.chunks_accepted_normal += 1


def graph_error_tolerance_for_chunks(total_chunks: int) -> float:
    return max(
        float(GRAPH_ERROR_TOLERANCE_MAX),
        float(total_chunks) * GRAPH_ERROR_TOLERANCE_RATIO,
    )


def graph_warning_message(graph_errors: int, tolerance: float) -> str:
    return (
        f"{graph_errors} chunk(s) had graph extraction errors; "
        f"tolerance={tolerance:g}. Some chunks do not have graph extraction."
    )


def finalize_manifest_for_documents(
    manifest: dict[str, Any],
    doc_chunks: list[tuple[dict[str, Any], list[SemanticChunk]]],
    payloads: list[GraphPayload],
    ingestion_run_id: str,
    ingested_at: str,
    skip_neo4j: bool,
    skip_qdrant: bool,
    skip_graph_extraction: bool,
    qdrant_ids_available: bool,
    limit_chunks_truncated: bool,
    global_error: str | None = None,
    defer_completion: bool = False,
    require_zero_graph_errors: bool = False,
) -> None:
    payloads_by_chunk_id = {payload.chunk_id: payload for payload in payloads}

    for file_info, chunks in doc_chunks:
        qdrant_point_ids = (
            qdrant_point_ids_for_chunks(chunks)
            if qdrant_ids_available
            else []
        )
        intentional_graph_skip = (
            skip_graph_extraction
            and skip_neo4j
            and not skip_qdrant
            and qdrant_ids_available
            and bool(qdrant_point_ids)
        )
        file_info["graph_extraction_skipped"] = bool(skip_graph_extraction)
        file_info["neo4j_skipped"] = bool(skip_neo4j)
        file_info["qdrant_indexed"] = bool(qdrant_point_ids)

        reasons: list[str] = []
        if not chunks:
            reasons.append("no chunks processed")
        if limit_chunks_truncated:
            reasons.append("--limit-chunks used")
        if skip_neo4j and not intentional_graph_skip:
            reasons.append("skipped Neo4j")
        if skip_qdrant:
            reasons.append("skipped Qdrant")
        if not skip_qdrant and not qdrant_point_ids:
            reasons.append("Qdrant upsert produced no point IDs")
        if global_error:
            reasons.append(global_error)

        graph_errors = sum(
            1
            for chunk in chunks
            if payloads_by_chunk_id.get(chunk.chunk_id, GraphPayload(chunk_id=chunk.chunk_id)).extraction_error
        )
        graph_tolerance = graph_error_tolerance_for_chunks(len(chunks))
        graph_warning = None
        graph_errors_within_tolerance = (
            graph_errors > 0
            and graph_errors <= graph_tolerance
            and not limit_chunks_truncated
            and not skip_neo4j
            and not skip_qdrant
            and qdrant_ids_available
            and not global_error
            and bool(chunks)
        )
        if graph_errors > 0 and (require_zero_graph_errors or not graph_errors_within_tolerance):
            reasons.append(
                f"graph extraction failed for {graph_errors} chunk(s); "
                f"tolerance={graph_tolerance:g}"
            )
        elif graph_errors_within_tolerance:
            graph_warning = graph_warning_message(graph_errors, graph_tolerance)

        if reasons:
            update_manifest_after_failure(
                manifest=manifest,
                file_info=file_info,
                ingestion_run_id=ingestion_run_id,
                error="; ".join(reasons),
                status="partial",
                chunk_count=len(chunks),
                qdrant_point_ids=qdrant_point_ids,
                last_ingested_at=ingested_at,
                graph_error_count=graph_errors,
                graph_error_tolerance=graph_tolerance,
            )
            continue

        graph_validated = (
            graph_errors == 0
            and not skip_graph_extraction
            and not skip_neo4j
            and not skip_qdrant
            and bool(qdrant_point_ids)
        )
        file_info["graph_validated"] = graph_validated
        file_info["phase1_validation_status"] = (
            "pending_full_validation" if defer_completion else "knowledge_only"
        )
        update_manifest_after_success(
            manifest=manifest,
            source_path=str(file_info["source_path"]),
            document_id=str(file_info["document_id"]),
            content_hash=str(file_info["content_hash"]),
            file_size=int(file_info["file_size"]),
            modified_time=str(file_info["modified_time"]),
            chunk_count=len(chunks),
            qdrant_point_ids=qdrant_point_ids,
            ingestion_run_id=ingestion_run_id,
            last_ingested_at=ingested_at,
            status=(
                "knowledge_indexed_pending_phase1_validation"
                if defer_completion
                else "completed_with_graph_skipped"
                if intentional_graph_skip
                else "completed_with_warnings" if graph_warning else "completed"
            ),
            error_message=graph_warning,
            graph_error_count=graph_errors,
            graph_error_tolerance=graph_tolerance,
            warning=graph_warning,
            source_metadata=file_info,
            graph_validated=graph_validated,
            phase1_validation_status=str(file_info["phase1_validation_status"]),
        )


def warn_changed_document_cleanup(
    file_info: dict[str, Any],
    skip_neo4j: bool,
    skip_qdrant: bool,
    dry_run: bool,
) -> None:
    if file_info.get("reason") != "changed" or dry_run:
        return

    if not skip_neo4j:
        logger.info(
            "[INCREMENTAL] Changed file detected: %s. Source-managed Neo4j "
            "relationship provenance will be reconciled by document_id before upsert. "
            "Legacy relationships without provenance are left untouched.",
            file_info["source_path"],
        )


async def cleanup_qdrant_before_reingest(
    *,
    manifest_record: dict[str, Any] | None,
    file_info: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    if not file_info.get("cleanup_required"):
        return {
            "cleanup_required": False,
            "safe": True,
            "mode": "none",
            "reason": "no previous Qdrant points expected for this file action",
        }

    if dry_run:
        return build_qdrant_cleanup_plan(
            collection_name=QDRANT_COLLECTION_NAME,
            manifest_record=manifest_record,
            expected_document_id=str(file_info["document_id"]),
            expected_source_path=str(file_info["source_path"]),
        )

    try:
        from qdrant_client import AsyncQdrantClient  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install qdrant-client") from exc

    client = AsyncQdrantClient(**qdrant_client_kwargs())
    try:
        result = await cleanup_previous_qdrant_points(
            qdrant_client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            manifest_record=manifest_record,
            expected_document_id=str(file_info["document_id"]),
            expected_source_path=str(file_info["source_path"]),
            dry_run=False,
        )
    finally:
        await client.close()

    if not result.get("safe", False):
        raise RuntimeError(result.get("reason") or "Qdrant cleanup was blocked")

    return result


def clear_graph_cache_dir() -> int:
    ensure_cache_dirs()
    deleted = 0
    for path in GRAPH_CACHE_DIR.glob("*.json"):
        if path.is_file():
            path.unlink()
            deleted += 1
    for path in GRAPH_CACHE_DIR.glob("*.tmp"):
        if path.is_file():
            path.unlink()
            deleted += 1
    return deleted


def file_fingerprint(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.name.encode("utf-8"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()[:24]


def legacy_file_fingerprint(path: Path) -> str:
    stat = path.stat()
    key = f"{path.name}::{stat.st_size}::{int(stat.st_mtime)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def markdown_cache_path(pdf_path: Path) -> Path:
    return MARKDOWN_CACHE_DIR / f"{pdf_path.stem}.{file_fingerprint(pdf_path)}.md"


def legacy_markdown_cache_path(source_path: Path) -> Path:
    return MARKDOWN_CACHE_DIR / f"{source_path.stem}.{legacy_file_fingerprint(source_path)}.md"


def graph_cache_key(chunk: SemanticChunk) -> str:
    key = "::".join(
        [
            GRAPH_CACHE_VERSION,
            GRAPH_PROMPT_SCHEMA_VERSION,
            OLLAMA_MODEL,
            chunk.document_identity,
            str(chunk.chunk_index),
            chunk.content_hash,
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def graph_cache_path(chunk: SemanticChunk) -> Path:
    safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(chunk.source_file).stem)
    filename = (
        f"{safe_source}.{chunk.chunk_index:05d}."
        f"{chunk.content_hash}.{graph_cache_key(chunk)}.json"
    )
    return GRAPH_CACHE_DIR / filename


def save_text(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
        return True
    except OSError as exc:
        logger.warning("[CACHE] Failed to write text cache %s: %s", path.name, exc)
        return False
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                logger.warning("[CACHE] Failed to remove temp text cache %s", tmp_path.name)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_markdown_cache(
    cache_path: Path,
    source_name: str,
    stats: IngestionStats | None = None,
) -> str | None:
    if not cache_path.exists():
        return None

    try:
        markdown = read_text(cache_path)
    except OSError as exc:
        logger.warning(
            "[STAGE 1] Markdown cache unreadable for %s (%s). Re-parsing.",
            source_name,
            exc,
        )
        return None

    if not markdown.strip():
        logger.warning(
            "[STAGE 1] Markdown cache is empty for %s. Re-parsing.",
            source_name,
        )
        return None

    logger.info("[STAGE 1] Markdown cache hit: %s", source_name)
    if stats:
        stats.markdown_cache_hits += 1
    return markdown


def graph_payload_to_dict(payload: GraphPayload) -> dict[str, Any]:
    return {
        "chunk_id": payload.chunk_id,
        "nodes": [
            {
                "name": node.name,
                "type": node.entity_type,
                "properties": node.properties,
            }
            for node in payload.nodes
        ],
        "edges": [
            {
                "source": edge.source_name,
                "source_type": edge.source_type,
                "target": edge.target_name,
                "target_type": edge.target_type,
                "relation": edge.relation,
                "properties": edge.properties,
            }
            for edge in payload.edges
        ],
    }


def graph_payload_from_dict(data: dict[str, Any], from_cache: bool = True) -> GraphPayload:
    nodes = [
        GraphNode(
            name=str(n.get("name", "")).strip().lower(),
            entity_type=str(n.get("entity_type", n.get("type", n.get("label", "")))).upper(),
            properties=dict(n.get("properties", {})),
        )
        for n in data.get("nodes", [])
    ]

    edges = [
        GraphEdge(
            source_name=str(e.get("source_name", e.get("source", ""))).strip().lower(),
            source_type=str(e.get("source_type", "")).upper(),
            target_name=str(e.get("target_name", e.get("target", ""))).strip().lower(),
            target_type=str(e.get("target_type", "")).upper(),
            relation=str(e.get("relation", "")).upper(),
            properties=dict(e.get("properties", {})),
        )
        for e in data.get("edges", [])
    ]

    nodes = [
        n for n in nodes
        if n.name and n.entity_type in ENTITY_TYPES
    ]

    edges = [
        e for e in edges
        if (
            e.source_name
            and e.target_name
            and e.source_type in ENTITY_TYPES
            and e.target_type in ENTITY_TYPES
            and e.relation in RELATION_TYPES
        )
    ]

    return GraphPayload(
        chunk_id=str(data.get("chunk_id", "")),
        nodes=nodes,
        edges=edges,
        from_cache=from_cache,
    )

def _record_graph_cache_invalid(
    stats: IngestionStats | None,
    reason: str,
) -> None:
    if stats is None:
        return
    stats.graph_cache_invalid += 1
    if reason == "json":
        stats.graph_cache_invalid_json += 1
    elif reason == "mismatch":
        stats.graph_cache_invalid_mismatch += 1
    else:
        stats.graph_cache_invalid_schema += 1


def _validate_graph_payload_data(data: Any, chunk_id: str, source: str) -> bool:
    if not isinstance(data, dict):
        logger.warning("[CACHE] %s graph payload is not a JSON object for chunk %s", source, chunk_id)
        return False
    if data.get("extraction_error") is True:
        logger.warning("[CACHE] %s graph payload has extraction_error=True for chunk %s", source, chunk_id)
        return False

    nodes = data.get("nodes")
    edges = data.get("edges")

    if not isinstance(nodes, list):
        logger.warning("[CACHE] %s graph payload nodes is not a list for chunk %s", source, chunk_id)
        return False
    if not isinstance(edges, list):
        logger.warning("[CACHE] %s graph payload edges is not a list for chunk %s", source, chunk_id)
        return False

    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            logger.warning("[CACHE] %s graph node %d is not an object for chunk %s", source, idx, chunk_id)
            return False
        name = str(node.get("name", "")).strip()
        entity_type = str(
            node.get("type", node.get("label", node.get("entity_type", "")))
        ).strip()
        if not name or not entity_type:
            logger.warning(
                "[CACHE] %s graph node %d missing name/type for chunk %s",
                source,
                idx,
                chunk_id,
            )
            return False

    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            logger.warning("[CACHE] %s graph edge %d is not an object for chunk %s", source, idx, chunk_id)
            return False
        source_name = str(edge.get("source", edge.get("source_name", ""))).strip()
        target_name = str(edge.get("target", edge.get("target_name", ""))).strip()
        relation = str(edge.get("relation", edge.get("type", ""))).strip()
        if not source_name or not target_name or not relation:
            logger.warning(
                "[CACHE] %s graph edge %d missing source/target/relation for chunk %s",
                source,
                idx,
                chunk_id,
            )
            return False

    return True


def save_graph_payload(payload: GraphPayload, chunk: SemanticChunk) -> bool:
    if payload.extraction_error:
        logger.warning("[CACHE] Not caching failed graph extraction for chunk %s", chunk.chunk_id)
        return False
    if not _validate_graph_payload_data(graph_payload_to_dict(payload), chunk.chunk_id, "new"):
        logger.warning("[CACHE] Not caching invalid graph payload for chunk %s", chunk.chunk_id)
        return False

    path = graph_cache_path(chunk)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    cache_data = {
        "cache_version": GRAPH_CACHE_VERSION,
        "model": OLLAMA_MODEL,
        "prompt_schema_version": GRAPH_PROMPT_SCHEMA_VERSION,
        "source_file": chunk.source_file,
        "chunk_index": chunk.chunk_index,
        "chunk_id": chunk.chunk_id,
        "chunk_hash": chunk.content_hash,
        "extraction_error": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **graph_payload_to_dict(payload),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp_path.write_text(
            json.dumps(cache_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
        return True
    except OSError as exc:
        logger.warning(
            "[CACHE] Failed to write graph cache %s atomically: %s",
            path.name,
            exc,
        )
        return False
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                logger.warning("[CACHE] Failed to remove temp graph cache %s", tmp_path.name)


def load_graph_payload(
    chunk: SemanticChunk,
    stats: IngestionStats | None = None,
) -> GraphPayload | None:
    path = graph_cache_path(chunk)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _record_graph_cache_invalid(stats, "json")
        logger.warning("[CACHE] Invalid graph cache JSON %s: %s", path.name, exc)
        return None
    except OSError as exc:
        _record_graph_cache_invalid(stats, "schema")
        logger.warning("[CACHE] Failed to read graph cache %s: %s", path.name, exc)
        return None

    required_fields = {
        "cache_version",
        "model",
        "prompt_schema_version",
        "chunk_hash",
        "nodes",
        "edges",
        "created_at",
    }
    missing_fields = sorted(required_fields - set(data))
    if missing_fields:
        _record_graph_cache_invalid(stats, "schema")
        logger.warning(
            "[CACHE] Invalid graph cache %s: missing fields %s",
            path.name,
            missing_fields,
        )
        return None

    if data.get("extraction_error") is True:
        _record_graph_cache_invalid(stats, "schema")
        logger.warning(
            "[CACHE] Graph cache %s has extraction_error=True for chunk %s",
            path.name,
            chunk.chunk_id,
        )
        return None

    mismatches = []
    if data.get("cache_version") != GRAPH_CACHE_VERSION:
        mismatches.append(
            f"cache_version={data.get('cache_version')!r} expected {GRAPH_CACHE_VERSION!r}"
        )
    if data.get("model") != OLLAMA_MODEL:
        mismatches.append(f"model={data.get('model')!r} expected {OLLAMA_MODEL!r}")
    if data.get("prompt_schema_version") != GRAPH_PROMPT_SCHEMA_VERSION:
        mismatches.append(
            "prompt_schema_version="
            f"{data.get('prompt_schema_version')!r} expected {GRAPH_PROMPT_SCHEMA_VERSION!r}"
        )
    if data.get("chunk_hash") != chunk.content_hash:
        mismatches.append(f"chunk_hash={data.get('chunk_hash')!r} expected {chunk.content_hash!r}")

    if mismatches:
        _record_graph_cache_invalid(stats, "mismatch")
        logger.warning("[CACHE] Graph cache mismatch %s: %s", path.name, "; ".join(mismatches))
        return None

    if not _validate_graph_payload_data(data, chunk.chunk_id, "cached"):
        _record_graph_cache_invalid(stats, "schema")
        logger.warning("[CACHE] Invalid graph cache schema %s", path.name)
        return None

    payload = graph_payload_from_dict(data, from_cache=True)
    if payload.extraction_error:
        _record_graph_cache_invalid(stats, "schema")
        logger.warning("[CACHE] Graph cache normalized to failed payload %s", path.name)
        return None

    return payload


# =============================================================================
# STAGE 1 – PDF Extraction with Markdown cache
# =============================================================================

def build_llamaparse_parser() -> Any:
    try:
        from llama_parse import LlamaParse  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install llama-parse") from exc

    return LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        parsing_instruction=LLAMAPARSE_INSTRUCTION,
        verbose=False,
    )


def validate_parser_documents(documents: Any) -> ParserCompleteness:
    """Reject explicit parser failures while retaining valid small documents.

    LlamaParse does not guarantee a portable page-completion field on every
    returned document. This validator therefore rejects only observable
    structural failures and records ``partial_observability`` otherwise.
    """

    if not isinstance(documents, (list, tuple)) or not documents:
        return ParserCompleteness(False, "failed", "parser returned no document objects")

    text_count = 0
    for index, document in enumerate(documents):
        metadata = getattr(document, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        status = str(metadata.get("status") or metadata.get("parse_status") or "").lower()
        explicit_error = metadata.get("error") or metadata.get("page_error") or metadata.get("failed")
        if explicit_error or status in {"failed", "error", "incomplete", "partial_failure"}:
            return ParserCompleteness(
                False,
                "failed",
                f"parser document {index} reported failure: {explicit_error or status}",
            )
        if str(getattr(document, "text", "") or "").strip():
            text_count += 1

    if text_count == 0:
        return ParserCompleteness(False, "failed", "parser returned no meaningful text")
    return ParserCompleteness(True, "partial_observability")


async def stage1_extract_one_source(
    source_path: Path,
    parser: Any,
    refresh_markdown: bool = False,
    stats: IngestionStats | None = None,
) -> tuple[str, str, ParserCompleteness] | None:
    cache_path = markdown_cache_path(source_path)

    if not refresh_markdown:
        cached_markdown = load_markdown_cache(cache_path, source_path.name, stats=stats)
        if cached_markdown is not None:
            return source_path.name, cached_markdown, ParserCompleteness(
                True,
                "cached_partial_observability",
            )

        legacy_cache_path = legacy_markdown_cache_path(source_path)
        if legacy_cache_path != cache_path:
            legacy_markdown = load_markdown_cache(
                legacy_cache_path,
                source_path.name,
                stats=None,
            )
            if legacy_markdown is not None:
                logger.info(
                    "[STAGE 1] Migrating legacy Markdown cache for %s to content-hash key.",
                    source_path.name,
                )
                save_text(cache_path, legacy_markdown)
                if stats:
                    stats.markdown_cache_hits += 1
                return source_path.name, legacy_markdown, ParserCompleteness(
                    True,
                    "cached_partial_observability",
                )

    if stats:
        stats.markdown_cache_misses += 1

    try:
        logger.info("[STAGE 1] Parsing with LlamaParse: %s", source_path.name)
        documents = await parser.aload_data(str(source_path))
        parser_completeness = validate_parser_documents(documents)
        if not parser_completeness.passed:
            raise ValueError(parser_completeness.error or "parser completeness validation failed")
        markdown = "\n\n".join(doc.text for doc in documents if getattr(doc, "text", None))
        if not markdown.strip():
            raise ValueError("LlamaParse returned empty Markdown")
        save_text(cache_path, markdown)
        logger.info(
            "[STAGE 1] ✓ %s — %d chars extracted (parser=%s)",
            source_path.name,
            len(markdown),
            parser_completeness.status,
        )
        return source_path.name, markdown, parser_completeness
    except Exception as exc:
        logger.error("[STAGE 1] ✗ Failed to parse %s: %s", source_path.name, exc)
        if stats:
            stats.parse_errors += 1
        return None


async def stage1_extract_pdfs(
    source_dir: Path,
    limit_files: int | None = None,
    refresh_markdown: bool = False,
    stats: IngestionStats | None = None,
) -> list[tuple[dict[str, Any], str, str]]:
    source_files = discover_source_documents(source_dir)

    if limit_files is not None and limit_files > 0:
        source_files = source_files[:limit_files]

    if not source_files:
        logger.warning("[STAGE 1] No PDF/DOCX files found in %s", source_dir)
        return []

    logger.info("[STAGE 1] Found %d PDF/DOCX file(s) in %s", len(source_files), source_dir)

    parser = build_llamaparse_parser()

    raw_results = await asyncio.gather(
        *(
            stage1_extract_one_source(
                source_path=p,
                parser=parser,
                refresh_markdown=refresh_markdown,
                stats=stats,
            )
            for p in source_files
        )
    )
    results = [
        (_file_manifest_info(source_path, source_root=source_dir), item[0], item[1])
        for source_path, item in zip(source_files, raw_results)
        if item is not None
    ]

    logger.info(
        "[STAGE 1] Completed: %d/%d documents available",
        len(results),
        len(source_files),
    )

    return results


async def stage1_extract_sources(
    source_dir: Path,
    limit_files: int | None = None,
    refresh_markdown: bool = False,
    stats: IngestionStats | None = None,
) -> list[tuple[dict[str, Any], str, str | None]]:
    source_files = discover_source_documents(source_dir)

    if limit_files is not None and limit_files > 0:
        source_files = source_files[:limit_files]

    if not source_files:
        logger.warning("[STAGE 1] No PDF/DOCX/JSON files found in %s", source_dir)
        return []

    json_count = sum(1 for path in source_files if is_web_json_source(path))
    parsed_count = len(source_files) - json_count
    logger.info(
        "[STAGE 1] Found %d knowledge file(s) in %s: parseable=%d json=%d",
        len(source_files),
        source_dir,
        parsed_count,
        json_count,
    )

    parser = None
    results: list[tuple[dict[str, Any], str, str | None]] = []
    for source_path in source_files:
        file_info = _file_manifest_info(source_path, source_root=source_dir)
        if is_web_json_source(source_path):
            logger.info("[STAGE 1] Found JSON file: %s", source_path.name)
            file_info["parser_completeness"] = "not_applicable_json"
            results.append((file_info, source_path.name, None))
            continue

        if parser is None:
            parser = build_llamaparse_parser()
        parsed = await stage1_extract_one_source(
            source_path=source_path,
            parser=parser,
            refresh_markdown=refresh_markdown,
            stats=stats,
        )
        if parsed is not None:
            file_info["parser_completeness"] = parsed[2].status
            results.append((file_info, parsed[0], parsed[1]))

    logger.info(
        "[STAGE 1] Completed: %d/%d source file(s) available",
        len(results),
        len(source_files),
    )
    return results


# =============================================================================
# STAGE 2 – Markdown chunking
# =============================================================================

def _enrich_chunk_metadata(
    base_metadata: dict[str, Any],
    text: str,
    header_path: str,
) -> dict[str, Any]:
    """Merge dermatology metadata into existing chunk metadata.

    Renames ``confidence`` → ``metadata_confidence`` and
    ``extraction_method`` → ``metadata_extraction_method`` to
    avoid name clashes with other pipeline fields.
    """
    derm_meta = extract_dermatology_metadata(text=text, header_path=header_path)

    # Rename fields to avoid collision with top-level pipeline metadata
    enriched = dict(base_metadata)
    enriched["domain_topic"] = derm_meta["domain_topic"]
    enriched["content_type"] = derm_meta["content_type"]
    enriched["concern"] = derm_meta["concern"]
    enriched["ingredient"] = derm_meta["ingredient"]
    enriched["skin_type"] = derm_meta["skin_type"]
    enriched["body_area"] = derm_meta["body_area"]
    enriched["safety_context"] = derm_meta["safety_context"]
    enriched["evidence_type"] = derm_meta["evidence_type"]
    enriched["metadata_confidence"] = derm_meta["confidence"]
    enriched["metadata_extraction_method"] = derm_meta["extraction_method"]

    return enrich_domain_metadata(
        text=f"{header_path}\n{text}" if header_path else text,
        existing_metadata=enriched,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.5 Step 6.5 – Noisy chunk detection
# ─────────────────────────────────────────────────────────────────────────────

def chunk_markdown_text(
    markdown_text: str,
    source_file: str,
    max_section_chars: int = 2000,
) -> list[SemanticChunk]:
    """Parse Markdown into semantically-chunked pieces with dermatology metadata.

    This is a standalone helper that does NOT depend on any external service
    (no LlamaParse, no Qdrant, no Neo4j).  It is safe to import and call from
    test scripts.

    Parameters
    ----------
    markdown_text : str
        Raw Markdown content.
    source_file : str
        Filename label stored in each chunk.
    max_section_chars : int
        Maximum characters per section before further splitting.
    """
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    headers = list(header_pattern.finditer(markdown_text))

    sections: list[tuple[str, str]] = []

    if not headers:
        sections.append(("", markdown_text))
    else:
        pre = markdown_text[:headers[0].start()].strip()
        if pre:
            sections.append(("", pre))

        for i, match in enumerate(headers):
            title = match.group(2).strip()
            start = match.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(markdown_text)
            content = markdown_text[start:end].strip()
            sections.append((title, content))

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_section_chars,
            chunk_overlap=max(100, max_section_chars // 5),
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        use_splitter = True
    except ImportError:
        logger.warning("langchain-text-splitters not installed. Using naive splitter.")
        splitter = None
        use_splitter = False

    chunks: list[SemanticChunk] = []
    idx = 0

    for header_path, content in sections:
        if not content.strip():
            continue

        # ── Hierarchical parent metadata (Phase 1.5 Step 3) ──────────
        section_text_stripped = content.strip()
        parent_text_hash = hashlib.sha1(
            section_text_stripped.encode("utf-8")
        ).hexdigest()[:16]
        parent_id = hashlib.sha1(
            f"{source_file}|{header_path}|{parent_text_hash}".encode("utf-8")
        ).hexdigest()[:16]
        section_char_length = len(section_text_stripped)
        # ─────────────────────────────────────────────────────────────

        if len(content) <= max_section_chars:
            sub_texts = [content]
        elif use_splitter and splitter is not None:
            sub_texts = splitter.split_text(content)
        else:
            sub_texts = naive_split(content, max_section_chars, max(100, max_section_chars // 5))

        child_index = 0
        for sub in sub_texts:
            text = sub.strip()
            if not text:
                continue

            base_metadata = {
                "source_file": source_file,
                "header": header_path,
                "chunk_index": idx,
                # Hierarchical parent-child fields
                "parent_id": parent_id,
                "chunk_level": "child",
                "parent_header_path": header_path,
                "child_index_in_parent": child_index,
                "parent_text_hash": parent_text_hash,
                "section_char_length": section_char_length,
            }

            enriched = _enrich_chunk_metadata(base_metadata, text, header_path)

            # Phase 1.5 Step 6.5 – Noisy chunk tagging
            noisy, noise_reason = is_noisy_chunk(text, header_path)
            enriched["is_noisy"] = noisy
            enriched["noise_reason"] = noise_reason if noisy else None
            enriched["is_short_medical_safety_rescue"] = (
                len(text) < 80
                and not noisy
                and is_short_medical_safety_statement(text, header_path)
            )

            chunks.append(
                SemanticChunk(
                    source_file=source_file,
                    chunk_index=idx,
                    text=text,
                    header_path=header_path,
                    metadata=enriched,
                )
            )
            idx += 1
            child_index += 1

    return chunks


def stage2_chunk_web_json_file(
    source_path: Path,
    file_info: dict[str, Any],
) -> list[SemanticChunk]:
    documents, summary = load_web_json_documents_with_stats(source_path)
    file_info["json_record_count"] = summary["total_records"]
    file_info["document_count"] = len(documents)
    file_info["skipped_record_count"] = summary["skipped_records"]

    logger.info(
        "[STAGE 1] Loaded %d JSON documents from %s; skipped=%d",
        len(documents),
        source_path.name,
        summary["skipped_records"],
    )

    chunks: list[SemanticChunk] = []
    for document in documents:
        text = str(document["text"])
        metadata = dict(document["metadata"])
        record_index = int(metadata.get("record_index", 0) or 0)
        record_document_id = web_json_record_document_id(
            source_path=str(file_info["source_path"]),
            record_index=record_index,
            source_url=str(metadata.get("source_url") or ""),
        )

        record_chunks = chunk_markdown_text(
            markdown_text=text,
            source_file=str(file_info.get("source_file") or source_path.name),
            max_section_chars=CHUNK_SIZE,
        )

        for chunk in record_chunks:
            chunk.metadata.update(
                {
                    **metadata,
                    "source_type": "web_json",
                    "source_file": str(file_info.get("source_file") or source_path.name),
                    "source_path": str(file_info["source_path"]),
                    "record_index": record_index,
                    "document_id": record_document_id,
                    "file_document_id": str(file_info["document_id"]),
                }
            )
            chunk.metadata = enrich_domain_metadata(
                text=chunk.text,
                existing_metadata=chunk.metadata,
            )
            chunk.metadata["chunk_index"] = chunk.chunk_index
            chunk.metadata["chunk_id"] = chunk.chunk_id
            chunks.append(chunk)

    logger.info("[STAGE 2] %s -> %d JSON semantic chunks", source_path.name, len(chunks))
    return chunks


def stage2_chunk_markdown(filename: str, markdown: str) -> list[SemanticChunk]:
    """Stage 2 entry point — delegates to chunk_markdown_text()."""
    chunks = chunk_markdown_text(
        markdown_text=markdown,
        source_file=filename,
        max_section_chars=CHUNK_SIZE,
    )
    logger.info("[STAGE 2] %s → %d semantic chunks", filename, len(chunks))
    return chunks


# =============================================================================
# STAGE 3 – Graph extraction
# =============================================================================

_EXTRACTION_PROMPT_TEMPLATE = """\
You are a clinical knowledge extraction expert specialising in dermatology and acne treatment.

Given the following text chunk, extract all relevant medical entities and their relationships.

## Entity Types
{entity_types}

## Relationship Types
{relation_types}

## Rules
- Only extract entities explicitly mentioned in the text.
- Entity names must be concise, normalised to lowercase.
- Prefer medical concepts related to acne, dermatology, symptoms, drugs, mechanisms, body parts, and treatments.
- Do not invent entities.
- Return ONLY valid JSON — no markdown fences, no explanation.

## Required Output Format
{{
  "nodes": [
    {{"name": "<entity_name>", "type": "<ENTITY_TYPE>", "description": "<short description>"}}
  ],
  "edges": [
    {{"source": "<entity_name>", "source_type": "<ENTITY_TYPE>",
      "target": "<entity_name>", "target_type": "<ENTITY_TYPE>",
      "relation": "<RELATION_TYPE>", "evidence": "<quote from text>"}}
  ]
}}

## Text Chunk
---
{chunk_text}
---

JSON:"""


def build_extraction_prompt(chunk_text: str) -> str:
    return _EXTRACTION_PROMPT_TEMPLATE.format(
        entity_types="\n".join(f"- {t}" for t in ENTITY_TYPES),
        relation_types="\n".join(f"- {t}" for t in RELATION_TYPES),
        chunk_text=chunk_text[:4000],
    )


def parse_llm_json(raw: str, chunk_id: str) -> GraphPayload:
    text = raw.strip()

    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            logger.warning("[STAGE 3] LLM JSON root is not an object for chunk %s", chunk_id)
            return GraphPayload(chunk_id=chunk_id, extraction_error=True)
        return build_graph_payload(data, chunk_id)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if not isinstance(data, dict):
                logger.warning("[STAGE 3] LLM JSON root is not an object for chunk %s", chunk_id)
                return GraphPayload(chunk_id=chunk_id, extraction_error=True)
            return build_graph_payload(data, chunk_id)
        except json.JSONDecodeError:
            pass

    logger.warning("[STAGE 3] Could not parse JSON for chunk %s", chunk_id)
    return GraphPayload(chunk_id=chunk_id, extraction_error=True)


def build_graph_payload(data: dict[str, Any], chunk_id: str) -> GraphPayload:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])

    if not isinstance(raw_nodes, list):
        logger.warning("[STAGE 3] LLM nodes field is not a list for chunk %s", chunk_id)
        raw_nodes = []
    if not isinstance(raw_edges, list):
        logger.warning("[STAGE 3] LLM edges field is not a list for chunk %s", chunk_id)
        raw_edges = []

    for n in raw_nodes:
        if not isinstance(n, dict):
            logger.warning("[STAGE 3] Skipping non-object node for chunk %s", chunk_id)
            continue
        name = str(n.get("name", "")).strip().lower()
        etype = str(n.get("type", n.get("label", n.get("entity_type", "")))).strip().upper()

        if name and etype in ENTITY_TYPES:
            nodes.append(
                GraphNode(
                    name=name,
                    entity_type=etype,
                    properties={
                        "description": str(n.get("description", "")).strip()
                    },
                )
            )

    for e in raw_edges:
        if not isinstance(e, dict):
            logger.warning("[STAGE 3] Skipping non-object edge for chunk %s", chunk_id)
            continue
        src = str(e.get("source", e.get("source_name", ""))).strip().lower()
        src_type = str(e.get("source_type", "")).strip().upper()
        tgt = str(e.get("target", e.get("target_name", ""))).strip().lower()
        tgt_type = str(e.get("target_type", "")).strip().upper()
        rel = str(e.get("relation", e.get("type", ""))).strip().upper()

        if (
            src
            and tgt
            and src_type in ENTITY_TYPES
            and tgt_type in ENTITY_TYPES
            and rel in RELATION_TYPES
        ):
            edges.append(
                GraphEdge(
                    source_name=src,
                    source_type=src_type,
                    target_name=tgt,
                    target_type=tgt_type,
                    relation=rel,
                    properties={
                        "evidence": str(e.get("evidence", "")).strip()
                    },
                )
            )

    return GraphPayload(chunk_id=chunk_id, nodes=nodes, edges=edges)


async def extract_graph_one_chunk(
    chunk: SemanticChunk,
    semaphore: asyncio.Semaphore,
    llm: Any,
    use_resume: bool = True,
    refresh_graph_cache: bool = False,
    skip_graph_extraction: bool = False,
    stats: IngestionStats | None = None,
) -> GraphPayload:
    if use_resume and not refresh_graph_cache:
        cached = load_graph_payload(chunk, stats=stats)
        if cached is not None:
            if stats:
                stats.graph_cache_hits += 1
            logger.debug("[CACHE] Graph cache hit: %s", graph_cache_path(chunk).name)
            return cached

    if stats:
        stats.graph_cache_misses += 1

    if skip_graph_extraction:
        return GraphPayload(chunk_id=chunk.chunk_id, extraction_error=False)

    async with semaphore:
        try:
            from langchain_core.messages import HumanMessage  # type: ignore

            prompt = build_extraction_prompt(chunk.text)
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = response.content if hasattr(response, "content") else str(response)
            payload = parse_llm_json(str(raw), chunk.chunk_id)

            save_graph_payload(payload, chunk)

            logger.debug(
                "[STAGE 3] Chunk %s → %d nodes, %d edges",
                chunk.chunk_id,
                len(payload.nodes),
                len(payload.edges),
            )

            return payload
        except Exception as exc:
            logger.error("[STAGE 3] LLM error on chunk %s: %s", chunk.chunk_id, exc)
            return GraphPayload(chunk_id=chunk.chunk_id, extraction_error=True)


async def build_ollama_llm() -> Any:
    try:
        from langchain_ollama import ChatOllama  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run: pip install langchain-ollama"
        ) from exc

    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.0,
        format="json",
    )


def _http_get_json(url: str, timeout: float = 5.0) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def qdrant_client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    return kwargs


def preflight_source_documents(source_dir: Path) -> bool:
    docs = discover_source_documents(source_dir)
    if not source_dir.exists():
        logger.error("[PREFLIGHT] Source directory does not exist: %s", source_dir)
        return False
    if not docs:
        logger.error("[PREFLIGHT] No PDF/DOCX files found in %s", source_dir)
        return False
    logger.info("[PREFLIGHT] Source documents found: %d", len(docs))
    return True


def incremental_plan_has_no_work(args: argparse.Namespace) -> bool:
    if not getattr(args, "incremental", False):
        return False
    if getattr(args, "force_reingest", False):
        return False

    source_files = discover_source_documents(args.source)
    if args.limit_files is not None and args.limit_files > 0:
        source_files = source_files[:args.limit_files]
    if not source_files:
        return False

    manifest = load_ingestion_manifest(args.manifest_path)
    plan = get_incremental_file_plan(
        source_files,
        manifest,
        force_reingest=False,
        source_root=args.source,
    )
    return not plan["to_ingest"]


async def preflight_ollama() -> bool:
    try:
        data = await asyncio.to_thread(_http_get_json, f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags")
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.error("[PREFLIGHT] Ollama is not reachable at %s: %s", OLLAMA_BASE_URL, exc)
        return False

    models = data.get("models", []) if isinstance(data, dict) else []
    model_names = {
        str(model.get("name", "")).split(":")[0]
        for model in models
        if isinstance(model, dict)
    }
    full_model_names = {
        str(model.get("name", ""))
        for model in models
        if isinstance(model, dict)
    }
    if OLLAMA_MODEL not in model_names and OLLAMA_MODEL not in full_model_names:
        logger.error(
            "[PREFLIGHT] Ollama model '%s' is not installed. Available models: %s",
            OLLAMA_MODEL,
            ", ".join(sorted(full_model_names)) or "(none)",
        )
        return False

    logger.info("[PREFLIGHT] Ollama reachable and model '%s' is available.", OLLAMA_MODEL)
    return True


async def preflight_qdrant() -> bool:
    try:
        from qdrant_client import AsyncQdrantClient  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install qdrant-client") from exc

    client = AsyncQdrantClient(**qdrant_client_kwargs())
    try:
        await client.get_collections()
        logger.info("[PREFLIGHT] Qdrant reachable at %s", QDRANT_URL)
        return True
    except Exception as exc:
        logger.error("[PREFLIGHT] Qdrant is not reachable at %s: %s", QDRANT_URL, exc)
        return False
    finally:
        await client.close()


async def preflight_neo4j() -> bool:
    driver = None
    try:
        driver = await create_neo4j_driver()
        await driver.verify_connectivity()
        logger.info("[PREFLIGHT] Neo4j reachable at %s", NEO4J_URI)
        return True
    except Exception as exc:
        logger.error("[PREFLIGHT] Neo4j is not reachable at %s: %s", NEO4J_URI, exc)
        return False
    finally:
        if driver is not None:
            await driver.close()


async def run_preflight_checks(
    args: argparse.Namespace,
    graph_resume_enabled: bool,
    require_neo4j: bool | None = None,
) -> bool:
    checks_ok = True

    checks_ok = preflight_source_documents(args.source) and checks_ok

    if checks_ok:
        try:
            if incremental_plan_has_no_work(args):
                logger.info(
                    "[PREFLIGHT] Incremental manifest has no files to ingest; "
                    "external service checks skipped."
                )
                return True
        except Exception as exc:
            logger.error("[PREFLIGHT] Failed to evaluate incremental manifest: %s", exc)
            return False

    if not LLAMA_CLOUD_API_KEY:
        logger.error("[PREFLIGHT] LLAMA_CLOUD_API_KEY is not set.")
        checks_ok = False
    else:
        logger.info("[PREFLIGHT] LLAMA_CLOUD_API_KEY is set.")

    if not GOOGLE_API_KEY and not args.dry_run and not args.skip_qdrant:
        logger.error("[PREFLIGHT] GOOGLE_API_KEY is not set; Qdrant embedding cannot run.")
        checks_ok = False
    elif args.dry_run or args.skip_qdrant:
        logger.info("[PREFLIGHT] GOOGLE_API_KEY check skipped because Qdrant is disabled.")
    else:
        logger.info("[PREFLIGHT] GOOGLE_API_KEY is set.")

    if not args.skip_graph_extraction:
        checks_ok = await preflight_ollama() and checks_ok
    elif graph_resume_enabled:
        logger.info("[PREFLIGHT] Ollama check skipped because graph extraction is disabled.")

    should_check_neo4j = (
        not args.dry_run
        and (not args.skip_neo4j if require_neo4j is None else require_neo4j)
    )
    if should_check_neo4j:
        checks_ok = await preflight_neo4j() and checks_ok
    else:
        logger.info("[PREFLIGHT] Neo4j check skipped.")

    if not args.dry_run and not args.skip_qdrant:
        checks_ok = await preflight_qdrant() and checks_ok
    else:
        logger.info("[PREFLIGHT] Qdrant check skipped.")

    return checks_ok


# =============================================================================
# STAGE 4A – Neo4j indexing
# =============================================================================

async def create_neo4j_driver() -> Any:
    try:
        from neo4j import AsyncGraphDatabase  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install neo4j") from exc

    return AsyncGraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )


async def ensure_neo4j_constraints(session: Any) -> None:
    for etype in ENTITY_TYPES:
        await session.run(
            f"CREATE CONSTRAINT IF NOT EXISTS "
            f"FOR (n:{etype}) REQUIRE n.name IS UNIQUE"
        )


async def upsert_neo4j_payloads(
    session: Any,
    payloads: list[GraphPayload],
    *,
    payload_document_ids: dict[str, str] | None = None,
) -> Neo4jUpsertResult:
    """Upsert extracted graph facts and report only materialized relationships."""

    result_summary = Neo4jUpsertResult()
    payload_document_ids = payload_document_ids or {}

    all_nodes: list[GraphNode] = []
    all_edges: list[GraphEdge] = []

    for payload in payloads:
        all_nodes.extend(payload.nodes)
        all_edges.extend(payload.edges)

    unique_nodes: dict[tuple[str, str], GraphNode] = {}
    for node in all_nodes:
        unique_nodes.setdefault((node.entity_type, node.name), node)

    unique_edges: dict[
        tuple[str, str, str, str, str], tuple[GraphEdge, set[str], set[str]]
    ] = {}
    for payload in payloads:
        document_id = str(payload_document_ids.get(payload.chunk_id) or "").strip()
        for edge in payload.edges:
            key = (
                edge.source_type,
                edge.source_name,
                edge.relation,
                edge.target_type,
                edge.target_name,
            )
            existing = unique_edges.get(key)
            if existing is None:
                existing = (edge, set(), set())
                unique_edges[key] = existing
            if document_id:
                existing[1].add(document_id)
            existing[2].add(payload.chunk_id)

    for node in unique_nodes.values():
        cypher = (
            f"MERGE (n:{node.entity_type} {{name: $name}}) "
            f"ON CREATE SET "
            f"n.description = $description, "
            f"n.extraction_source = $extraction_source, "
            f"n.extraction_model = $extraction_model, "
            f"n.created_at = timestamp() "
            f"ON MATCH SET "
            f"n.description = CASE "
            f"WHEN n.description IS NULL OR n.description = '' "
            f"THEN $description ELSE n.description END, "
            f"n.extraction_source = coalesce(n.extraction_source, $extraction_source), "
            f"n.extraction_model = coalesce(n.extraction_model, $extraction_model) "
            f"RETURN count(n) AS node_count"
        )

        node_result = await session.run(
            cypher,
            name=node.name,
            description=node.properties.get("description", ""),
            extraction_source="ollama_semantic_enrichment",
            extraction_model=OLLAMA_MODEL,
        )
        node_record = await _neo4j_single(node_result)
        result_summary.nodes_upserted += _neo4j_record_int(node_record, "node_count", default=1)

    for edge, source_document_ids, source_chunk_ids in unique_edges.values():
        try:
            source_result = await session.run(
                f"MATCH (src:{edge.source_type} {{name: $src_name}}) "
                "RETURN count(src) AS source_count",
                src_name=edge.source_name,
            )
            source_count = _neo4j_record_int(
                await _neo4j_single(source_result), "source_count"
            )
            if source_count == 0:
                result_summary.missing_source_endpoint += 1
                logger.error(
                    "[STAGE 4A] Missing source endpoint for edge (%s)-[%s]->(%s)",
                    edge.source_name,
                    edge.relation,
                    edge.target_name,
                )
                continue
            if source_count != 1:
                result_summary.database_errors += 1
                logger.error(
                    "[STAGE 4A] Source endpoint is not unique for edge (%s)-[%s]->(%s): %d",
                    edge.source_name,
                    edge.relation,
                    edge.target_name,
                    source_count,
                )
                continue

            target_result = await session.run(
                f"MATCH (tgt:{edge.target_type} {{name: $tgt_name}}) "
                "RETURN count(tgt) AS target_count",
                tgt_name=edge.target_name,
            )
            target_count = _neo4j_record_int(
                await _neo4j_single(target_result), "target_count"
            )
            if target_count == 0:
                result_summary.missing_target_endpoint += 1
                logger.error(
                    "[STAGE 4A] Missing target endpoint for edge (%s)-[%s]->(%s)",
                    edge.source_name,
                    edge.relation,
                    edge.target_name,
                )
                continue
            if target_count != 1:
                result_summary.database_errors += 1
                logger.error(
                    "[STAGE 4A] Target endpoint is not unique for edge (%s)-[%s]->(%s): %d",
                    edge.source_name,
                    edge.relation,
                    edge.target_name,
                    target_count,
                )
                continue

            edge_result = await session.run(
                (
                    f"MATCH (src:{edge.source_type} {{name: $src_name}}) "
                    f"MATCH (tgt:{edge.target_type} {{name: $tgt_name}}) "
                    f"MERGE (src)-[r:{edge.relation}]->(tgt) "
                    f"ON CREATE SET "
                    f"r.evidence = $evidence, "
                    f"r.extraction_source = $extraction_source, "
                    f"r.extraction_model = $extraction_model, "
                    f"r.created_at = timestamp(), "
                    f"r.source_document_ids = $source_document_ids, "
                    f"r.source_chunk_ids = $source_chunk_ids "
                    f"ON MATCH SET "
                    f"r.evidence = CASE "
                    f"WHEN r.evidence IS NULL OR r.evidence = '' "
                    f"THEN $evidence ELSE r.evidence END, "
                    f"r.extraction_source = coalesce(r.extraction_source, $extraction_source), "
                    f"r.extraction_model = coalesce(r.extraction_model, $extraction_model), "
                    f"r.source_document_ids = reduce(ids = coalesce(r.source_document_ids, []), "
                    f"document_id IN $source_document_ids | "
                    f"CASE WHEN document_id IN ids THEN ids ELSE ids + document_id END), "
                    f"r.source_chunk_ids = reduce(ids = coalesce(r.source_chunk_ids, []), "
                    f"chunk_id IN $source_chunk_ids | "
                    f"CASE WHEN chunk_id IN ids THEN ids ELSE ids + chunk_id END) "
                    f"RETURN count(r) AS relationship_count"
                ),
                src_name=edge.source_name,
                tgt_name=edge.target_name,
                evidence=edge.properties.get("evidence", ""),
                source_document_ids=sorted(source_document_ids),
                source_chunk_ids=sorted(source_chunk_ids),
                extraction_source="ollama_semantic_enrichment",
                extraction_model=OLLAMA_MODEL,
            )
            record = await _neo4j_single(edge_result)
            relationship_count = _neo4j_record_int(record, "relationship_count")
            if relationship_count == 1:
                result_summary.relationships_upserted += 1
            else:
                result_summary.database_errors += 1
                logger.error(
                    "[STAGE 4A] Edge did not materialize (%s)-[%s]->(%s)",
                    edge.source_name,
                    edge.relation,
                    edge.target_name,
                )
        except Exception as exc:
            result_summary.database_errors += 1
            logger.error(
                "[STAGE 4A] Edge database error (%s)-[%s]->(%s): %s",
                edge.source_name,
                edge.relation,
                edge.target_name,
                exc,
            )

    return result_summary


async def _neo4j_single(result: Any) -> Any:
    single = getattr(result, "single", None)
    if single is None:
        return None
    record = single()
    if hasattr(record, "__await__"):
        record = await record
    return record


def _neo4j_record_int(record: Any, key: str, default: int = 0) -> int:
    if record is None:
        return default
    try:
        value = record.get(key, default) if hasattr(record, "get") else record[key]
        return int(value or 0)
    except (KeyError, TypeError, ValueError):
        return default


async def reconcile_neo4j_document_graph(session: Any, document_id: str) -> dict[str, int]:
    """Remove only source-managed graph facts before a changed document is rebuilt."""

    source_document_id = str(document_id).strip()
    if not source_document_id:
        return {"deleted": 0, "detached": 0}

    delete_result = await session.run(
        "MATCH ()-[r]->() "
        "WHERE $document_id IN coalesce(r.source_document_ids, []) "
        "AND size(r.source_document_ids) = 1 "
        "DELETE r RETURN count(r) AS deleted",
        document_id=source_document_id,
    )
    detach_result = await session.run(
        "MATCH ()-[r]->() "
        "WHERE $document_id IN coalesce(r.source_document_ids, []) "
        "AND size(r.source_document_ids) > 1 "
        "SET r.source_document_ids = [item IN r.source_document_ids WHERE item <> $document_id] "
        "RETURN count(r) AS detached",
        document_id=source_document_id,
    )
    return {
        "deleted": _neo4j_record_int(await _neo4j_single(delete_result), "deleted"),
        "detached": _neo4j_record_int(await _neo4j_single(detach_result), "detached"),
    }


# =============================================================================
# STAGE 4B – Qdrant vector indexing
# =============================================================================

async def ensure_qdrant_collection(client: Any) -> None:
    try:
        from qdrant_client.models import Distance, SparseVectorParams, VectorParams  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install qdrant-client") from exc

    collections = await client.get_collections()
    existing_names = {c.name for c in collections.collections}

    if QDRANT_COLLECTION_NAME not in existing_names:
        logger.info(
            "[STAGE 4B] Creating Qdrant collection '%s' (dense=%d, sparse=bm25)",
            QDRANT_COLLECTION_NAME,
            EMBEDDING_DIMENSIONS,
        )

        await client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=EMBEDDING_DIMENSIONS,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams()
            },
        )

        logger.info("[STAGE 4B] ✓ Qdrant collection created.")
    else:
        info = await client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        params = info.config.params

        if isinstance(params, dict):
            vectors_config = params.get("vectors")
            sparse_vectors_config = params.get("sparse_vectors")
        else:
            vectors_config = getattr(params, "vectors", None)
            sparse_vectors_config = getattr(params, "sparse_vectors", None)

        def get_named_config(config: Any, name: str) -> Any | None:
            if config is None:
                return None
            if isinstance(config, dict):
                return config.get(name)
            if hasattr(config, "get"):
                return config.get(name)
            return None

        dense_config = get_named_config(vectors_config, "dense")
        bm25_config = get_named_config(sparse_vectors_config, "bm25")

        schema_errors: list[str] = []

        if dense_config is None:
            schema_errors.append("missing named dense vector 'dense'")
        else:
            if isinstance(dense_config, dict):
                dense_size = dense_config.get("size")
            else:
                dense_size = getattr(dense_config, "size", None)

            if dense_size != EMBEDDING_DIMENSIONS:
                schema_errors.append(
                    f"named vector 'dense' has size {dense_size}, "
                    f"expected {EMBEDDING_DIMENSIONS}"
                )

        if bm25_config is None:
            schema_errors.append("missing sparse vector 'bm25'")

        if schema_errors:
            raise ValueError(
                "Qdrant collection schema mismatch for "
                f"'{QDRANT_COLLECTION_NAME}': "
                + "; ".join(schema_errors)
                + ". Aborting upsert. Recreate or migrate the collection manually "
                "with named dense vector 'dense' and sparse vector 'bm25'."
            )

        logger.info(
            "[STAGE 4B] Collection '%s' already exists — schema validated.",
            QDRANT_COLLECTION_NAME,
        )


def embed_dense_sync(texts: list[str]) -> list[list[float]]:
    return embed_texts_sync(
        texts,
        model_name=EMBEDDING_MODEL,
        task_type="retrieval_document",
        expected_dimensions=EMBEDDING_DIMENSIONS,
        api_key=GOOGLE_API_KEY,
    )



def _extract_retry_delay_seconds(error: Exception) -> float | None:
    """Try to read retry delay from Google API error text.

    Google provider quota errors sometimes include text like:
        Please retry in 3.654895925s
    We parse that hint when available, then apply a safer minimum delay in the
    async retry wrapper.
    """
    message = str(error)
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _is_retryable_embedding_error(error: Exception) -> bool:
    message = str(error).lower()
    retryable_markers = [
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "deadline",
        "timeout",
        "temporarily unavailable",
        "503",
        "500",
    ]
    return any(marker in message for marker in retryable_markers)


async def embed_dense_batch_with_retry(
    texts: list[str],
    batch_number: int,
    total_batches: int,
) -> list[list[float]]:
    """Call Gemini embedding with retry/backoff for quota and transient errors."""
    loop = asyncio.get_event_loop()
    last_error: Exception | None = None

    for attempt in range(1, EMBEDDING_MAX_RETRIES + 1):
        try:
            return await loop.run_in_executor(None, embed_dense_sync, texts)
        except Exception as exc:
            last_error = exc

            if not _is_retryable_embedding_error(exc):
                raise

            retry_hint = _extract_retry_delay_seconds(exc)
            exponential_delay = EMBEDDING_RETRY_BASE_DELAY * min(2 ** (attempt - 1), 8)
            sleep_seconds = max(
                EMBEDDING_BATCH_DELAY,
                retry_hint or 0.0,
                exponential_delay,
            )

            logger.warning(
                "[STAGE 4B] Embedding batch %d/%d hit a retryable error "
                "(attempt %d/%d): %s",
                batch_number,
                total_batches,
                attempt,
                EMBEDDING_MAX_RETRIES,
                exc,
            )
            logger.warning(
                "[STAGE 4B] Sleeping %.1fs before retrying embedding batch %d/%d",
                sleep_seconds,
                batch_number,
                total_batches,
            )
            await asyncio.sleep(sleep_seconds)

    raise RuntimeError(
        f"Gemini embedding failed after {EMBEDDING_MAX_RETRIES} retries "
        f"for batch {batch_number}/{total_batches}: {last_error}"
    ) from last_error


def compute_hashed_sparse_vectors(texts: list[str]) -> list[dict[str, list]]:
    """Compatibility wrapper for the current custom sparse representation."""

    return compute_sparse_vectors(texts)


async def stage4b_upsert_qdrant(
    chunks: list[SemanticChunk],
    payloads: list[GraphPayload],
    ingestion_run_id: str | None = None,
    ingested_at: str | None = None,
) -> int:
    logger.info(
        "[STAGE 4B] Upserting %d chunks into Qdrant '%s'",
        len(chunks),
        QDRANT_COLLECTION_NAME,
    )

    try:
        from qdrant_client import AsyncQdrantClient  # type: ignore
        from qdrant_client.models import PointStruct, SparseVector  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install qdrant-client") from exc

    client = AsyncQdrantClient(**qdrant_client_kwargs())
    payload_ingested_at = ingested_at or utc_now_iso()

    try:
        await ensure_qdrant_collection(client)

        chunk_nodes: dict[str, list[str]] = {
            payload.chunk_id: [node.name for node in payload.nodes]
            for payload in payloads
        }

        # Phase 1.5 Step 6.5 – Filter out noisy chunks before embedding
        clean_chunks = [
            c for c in chunks if not c.metadata.get("is_noisy", False)
        ]
        noisy_count = len(chunks) - len(clean_chunks)
        if noisy_count > 0:
            logger.info(
                "[STAGE 4B] Skipping %d noisy chunks (of %d total). "
                "Upserting %d clean chunks.",
                noisy_count,
                len(chunks),
                len(clean_chunks),
            )

        if not clean_chunks:
            logger.warning("[STAGE 4B] No clean chunks available after noisy filtering. Skipping Qdrant upsert.")
            return 0

        total_upserted = 0
        total_batches = max(1, (len(clean_chunks) - 1) // INGEST_BATCH_SIZE + 1)

        for batch_start in range(0, len(clean_chunks), INGEST_BATCH_SIZE):
            batch = clean_chunks[batch_start:batch_start + INGEST_BATCH_SIZE]
            texts = [chunk.text for chunk in batch]

            logger.info(
                "[STAGE 4B] Batch %d/%d — embedding %d texts",
                batch_start // INGEST_BATCH_SIZE + 1,
                total_batches,
                len(texts),
            )

            loop = asyncio.get_event_loop()
            batch_number = batch_start // INGEST_BATCH_SIZE + 1

            dense_vecs = await embed_dense_batch_with_retry(
                texts=texts,
                batch_number=batch_number,
                total_batches=total_batches,
            )

            sparse_dicts = await loop.run_in_executor(
                None,
                compute_hashed_sparse_vectors,
                texts,
            )

            if len(dense_vecs) != len(batch):
                raise ValueError(
                    f"Dense embedding count mismatch: got {len(dense_vecs)}, "
                    f"expected {len(batch)}"
                )
            if len(sparse_dicts) != len(batch):
                raise ValueError(
                    f"Sparse vector count mismatch: got {len(sparse_dicts)}, "
                    f"expected {len(batch)}"
                )

            points: list[PointStruct] = []

            for chunk, dense, sparse_d in zip(batch, dense_vecs, sparse_dicts):
                if len(dense) != EMBEDDING_DIMENSIONS:
                    raise ValueError(
                        f"Embedding dimension mismatch for chunk {chunk.chunk_id}: "
                        f"got {len(dense)}, expected {EMBEDDING_DIMENSIONS}. "
                        f"Check EMBEDDING_MODEL and EMBEDDING_DIMENSIONS in .env, "
                        f"then recreate Qdrant collection."
                    )

                source_path = str(chunk.metadata.get("source_path") or chunk.source_file)
                document_id = str(
                    chunk.metadata.get("document_id")
                    or document_id_from_source_path(source_path)
                )
                document_content_hash = str(
                    chunk.metadata.get("content_hash")
                    or chunk.content_hash
                )
                payload_run_id = str(
                    chunk.metadata.get("ingestion_run_id")
                    or ingestion_run_id
                    or ""
                )
                payload_time = str(
                    chunk.metadata.get("ingested_at")
                    or payload_ingested_at
                )

                payload = {
                    **chunk.metadata,
                    **expected_kb_payload_metadata(),
                    "document_id": document_id,
                    "source_path": source_path,
                    "content_hash": document_content_hash,
                    "chunk_index": chunk.chunk_index,
                    "chunk_id": chunk.chunk_id,
                    "chunk_hash": chunk.content_hash,
                    "ingestion_run_id": payload_run_id,
                    "ingested_at": payload_time,
                    "text": chunk.text,
                    "header": chunk.header_path,
                    "graph_nodes": chunk_nodes.get(chunk.chunk_id, []),
                }

                points.append(
                    PointStruct(
                        id=chunk.qdrant_point_id,
                        vector={
                            "dense": dense,
                            "bm25": SparseVector(
                                indices=sparse_d["indices"],
                                values=sparse_d["values"],
                            ),
                        },
                        payload=payload,
                    )
                )

            await client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=points,
            )

            total_upserted += len(points)
            logger.info(
                "[STAGE 4B] ✓ Batch upserted %d points, total=%d",
                len(points),
                total_upserted,
            )

            if EMBEDDING_BATCH_DELAY > 0 and batch_start + INGEST_BATCH_SIZE < len(clean_chunks):
                logger.info(
                    "[STAGE 4B] Sleeping %.1fs to respect Gemini embedding quota",
                    EMBEDDING_BATCH_DELAY,
                )
                await asyncio.sleep(EMBEDDING_BATCH_DELAY)

        logger.info("[STAGE 4B] ✓ Qdrant upsert complete — %d vectors", total_upserted)
        return total_upserted

    finally:
        await client.close()


# =============================================================================
# Pipeline orchestration
# =============================================================================

def chunk_list(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]


async def stage3_and_optional_neo4j_incremental(
    chunks: list[SemanticChunk],
    dry_run: bool,
    use_resume: bool,
    refresh_graph_cache: bool,
    skip_graph_extraction: bool,
    skip_neo4j: bool,
    stats: IngestionStats,
    replace_document_ids: set[str] | None = None,
) -> list[GraphPayload]:
    logger.info("=" * 60)
    logger.info("[STAGE 3] Knowledge Graph Extraction (Ollama/%s)", OLLAMA_MODEL)
    logger.info("=" * 60)
    logger.info(
        "[STAGE 3] Processing %d chunks | concurrency=%d | graph_batch_size=%d | resume=%s | refresh_cache=%s | skip_graph=%s",
        len(chunks),
        LLM_CONCURRENCY,
        GRAPH_BATCH_SIZE,
        use_resume,
        refresh_graph_cache,
        skip_graph_extraction,
    )

    llm = None
    if not skip_graph_extraction:
        llm = await build_ollama_llm()

    semaphore = asyncio.Semaphore(LLM_CONCURRENCY)
    all_payloads: list[GraphPayload] = []

    neo4j_driver = None
    neo4j_session = None

    if not dry_run and not skip_neo4j:
        logger.info("=" * 60)
        logger.info("[STAGE 4A] Neo4j Graph Indexing will run incrementally")
        logger.info("=" * 60)
        logger.info("[STAGE 4A] Connecting to Neo4j at %s", NEO4J_URI)
        neo4j_driver = await create_neo4j_driver()
        neo4j_session = neo4j_driver.session()
        await ensure_neo4j_constraints(neo4j_session)
        for document_id in sorted(replace_document_ids or set()):
            reconciliation = await reconcile_neo4j_document_graph(neo4j_session, document_id)
            logger.info(
                "[STAGE 4A] Reconciled managed graph facts for document %s: deleted=%d detached=%d",
                document_id,
                reconciliation["deleted"],
                reconciliation["detached"],
            )

    try:
        batches = chunk_list(chunks, GRAPH_BATCH_SIZE)
        skip_graph_cache_misses_before = stats.graph_cache_misses

        for batch_idx, batch in enumerate(batches, start=1):
            logger.info(
                "[STAGE 3] Graph batch %d/%d — %d chunks",
                batch_idx,
                len(batches),
                len(batch),
            )

            payloads = await asyncio.gather(
                *[
                    extract_graph_one_chunk(
                        chunk=chunk,
                        semaphore=semaphore,
                        llm=llm,
                        use_resume=use_resume,
                        refresh_graph_cache=refresh_graph_cache,
                        skip_graph_extraction=skip_graph_extraction,
                        stats=stats,
                    )
                    for chunk in batch
                ]
            )

            all_payloads.extend(payloads)

            batch_nodes = sum(len(p.nodes) for p in payloads)
            batch_edges = sum(len(p.edges) for p in payloads)
            batch_errors = sum(1 for p in payloads if p.extraction_error)

            stats.nodes_extracted += batch_nodes
            stats.edges_extracted += batch_edges
            stats.llm_errors += batch_errors

            logger.info(
                "[STAGE 3] ✓ Batch %d/%d extracted — nodes=%d, edges=%d, errors=%d",
                batch_idx,
                len(batches),
                batch_nodes,
                batch_edges,
                batch_errors,
            )

            if not dry_run and not skip_neo4j and neo4j_session is not None:
                payload_document_ids = {
                    chunk.chunk_id: str(chunk.metadata.get("file_document_id") or chunk.metadata.get("document_id") or "")
                    for chunk in batch
                }
                upsert_result = await upsert_neo4j_payloads(
                    neo4j_session,
                    payloads,
                    payload_document_ids=payload_document_ids,
                )
                stats.nodes_upserted_neo4j += upsert_result.nodes_upserted
                stats.edges_upserted_neo4j += upsert_result.relationships_upserted
                stats.neo4j_missing_endpoints += (
                    upsert_result.missing_source_endpoint + upsert_result.missing_target_endpoint
                )
                if upsert_result.has_failures:
                    raise RuntimeError(
                        "Neo4j graph upsert was incomplete: "
                        f"missing_source={upsert_result.missing_source_endpoint}, "
                        f"missing_target={upsert_result.missing_target_endpoint}, "
                        f"database_errors={upsert_result.database_errors}"
                    )

                logger.info(
                    "[STAGE 4A] ✓ Batch %d/%d Neo4j upsert — nodes=%d, edges=%d",
                    batch_idx,
                    len(batches),
                    upsert_result.nodes_upserted,
                    upsert_result.relationships_upserted,
                )

        if skip_graph_extraction:
            cache_misses_due_to_skip = stats.graph_cache_misses - skip_graph_cache_misses_before
            logger.info(
                "[STAGE 3] Graph extraction skipped by flag — chunks=%d, cache_misses=%d",
                len(chunks),
                cache_misses_due_to_skip,
            )

        logger.info(
            "[STAGE 3] ✓ Extraction complete — %d nodes, %d edges across %d chunks",
            stats.nodes_extracted,
            stats.edges_extracted,
            len(all_payloads),
        )

        if not dry_run and not skip_neo4j:
            logger.info(
                "[STAGE 4A] ✓ Neo4j incremental upsert complete — nodes=%d, edges=%d",
                stats.nodes_upserted_neo4j,
                stats.edges_upserted_neo4j,
            )

        return all_payloads

    finally:
        if neo4j_session is not None:
            await neo4j_session.close()

        if neo4j_driver is not None:
            await neo4j_driver.close()


async def ingest_pipeline_incremental(
    source_dir: Path,
    dry_run: bool = False,
    limit_files: int | None = None,
    limit_chunks: int | None = None,
    refresh_markdown: bool = False,
    use_resume: bool = True,
    refresh_graph_cache: bool = False,
    skip_graph_extraction: bool = False,
    skip_neo4j: bool = False,
    skip_qdrant: bool = False,
    force_reingest: bool = False,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> IngestionStats:
    ensure_cache_dirs()
    stats = IngestionStats()

    logger.info("=" * 60)
    logger.info("[INCREMENTAL] Planning source document ingestion")
    logger.info("=" * 60)

    source_files = discover_source_documents(source_dir)
    if limit_files is not None and limit_files > 0:
        logger.warning("[INCREMENTAL] Applying --limit-files=%d", limit_files)
        source_files = source_files[:limit_files]

    if not source_files:
        logger.warning("[INCREMENTAL] No PDF/DOCX/JSON files found in %s", source_dir)
        return stats

    manifest = load_ingestion_manifest(manifest_path)
    plan = get_incremental_file_plan(
        files=source_files,
        manifest=manifest,
        force_reingest=force_reingest,
        source_root=source_dir,
    )
    log_incremental_file_plan(plan)

    if not plan["to_ingest"]:
        logger.info("[INCREMENTAL] No new, changed, failed, or partial files to ingest.")
        return stats

    ingestion_run_id = uuid.uuid4().hex
    logger.info("[INCREMENTAL] Ingestion run id: %s", ingestion_run_id)
    logger.info("[INCREMENTAL] Manifest path: %s", manifest_path)

    parser = None
    remaining_chunks = limit_chunks if limit_chunks is not None and limit_chunks > 0 else None

    for doc_number, file_info in enumerate(plan["to_ingest"], start=1):
        if remaining_chunks is not None and remaining_chunks <= 0:
            logger.warning(
                "[INCREMENTAL] --limit-chunks reached. Leaving %s unmodified in manifest.",
                file_info["source_path"],
            )
            continue

        logger.info("=" * 60)
        logger.info(
            "[INCREMENTAL] Document %d/%d (%s): %s",
            doc_number,
            len(plan["to_ingest"]),
            file_info["reason"],
            file_info["source_path"],
        )
        logger.info("=" * 60)

        if not dry_run:
            update_manifest_before_ingest(manifest, file_info, ingestion_run_id)
            save_ingestion_manifest(manifest_path, manifest)

        warn_changed_document_cleanup(
            file_info=file_info,
            skip_neo4j=skip_neo4j,
            skip_qdrant=skip_qdrant,
            dry_run=dry_run,
        )

        stage1_already_recorded_error = False

        try:
            logger.info("=" * 60)
            logger.info("[STAGE 2] Markdown-Aware Chunking (chunk_size=%d)", CHUNK_SIZE)
            logger.info("=" * 60)

            if is_web_json_source(file_info["path"]):
                logger.info("[STAGE 1] Found JSON file: %s", file_info["path"].name)
                chunks = stage2_chunk_web_json_file(file_info["path"], file_info)
                display_name = str(file_info.get("source_file") or file_info["path"].name)
            else:
                if parser is None:
                    parser = build_llamaparse_parser()
                parsed = await stage1_extract_one_source(
                    source_path=file_info["path"],
                    parser=parser,
                    refresh_markdown=refresh_markdown,
                    stats=stats,
                )
                if parsed is None:
                    stage1_already_recorded_error = True
                    raise RuntimeError("Stage 1 extraction failed")
                filename, markdown, parser_completeness = parsed
                file_info["parser_completeness"] = parser_completeness.status
                chunks = stage2_chunk_markdown(filename, markdown)
                display_name = filename

            stats.pdf_files += 1
            original_chunk_count = len(chunks)
            limit_chunks_used_for_doc = remaining_chunks is not None
            truncated_by_limit = False

            if remaining_chunks is not None:
                if len(chunks) > remaining_chunks:
                    logger.warning(
                        "[STAGE 2] Applying --limit-chunks. %s produced %d chunks; "
                        "processing only %d in this run.",
                        display_name,
                        len(chunks),
                        remaining_chunks,
                    )
                    chunks = chunks[:remaining_chunks]
                    truncated_by_limit = True
                remaining_chunks -= len(chunks)

            if not chunks:
                raise RuntimeError("No chunks produced")

            ingested_at = utc_now_iso()
            enrich_chunks_with_ingestion_metadata(
                chunks=chunks,
                file_info=file_info,
                ingestion_run_id=ingestion_run_id,
                ingested_at=ingested_at,
            )

            stats.chunks_created += len(chunks)
            account_chunk_quality(chunks, stats)
            logger.info("[STAGE 2] Total semantic chunks for document: %d", len(chunks))

            payloads = await stage3_and_optional_neo4j_incremental(
                chunks=chunks,
                dry_run=dry_run,
                use_resume=use_resume,
                refresh_graph_cache=refresh_graph_cache,
                skip_graph_extraction=skip_graph_extraction,
                skip_neo4j=skip_neo4j,
                stats=stats,
                replace_document_ids=(
                    {str(file_info["document_id"])}
                    if file_info.get("cleanup_required") and not skip_neo4j
                    else None
                ),
            )

            graph_errors = sum(1 for payload in payloads if payload.extraction_error)
            graph_tolerance = graph_error_tolerance_for_chunks(len(chunks))

            qdrant_point_ids: list[str] = []

            if dry_run:
                if file_info.get("cleanup_required") and not skip_qdrant:
                    cleanup_plan = await cleanup_qdrant_before_reingest(
                        manifest_record=file_info.get("previous_manifest_record"),
                        file_info=file_info,
                        dry_run=True,
                    )
                    logger.info(
                        "[DRY RUN] Qdrant cleanup plan for %s: mode=%s safe=%s reason=%s",
                        file_info["source_path"],
                        cleanup_plan.get("mode"),
                        cleanup_plan.get("safe"),
                        cleanup_plan.get("reason"),
                    )
                logger.info("[DRY RUN] Stage 4A/4B writes and manifest update skipped.")
                continue

            if not skip_qdrant:
                logger.info("=" * 60)
                logger.info("[STAGE 4B] Qdrant Vector Indexing (dense + hashed sparse)")
                logger.info("=" * 60)

                try:
                    cleanup_result = await cleanup_qdrant_before_reingest(
                        manifest_record=file_info.get("previous_manifest_record"),
                        file_info=file_info,
                        dry_run=False,
                    )
                    if cleanup_result.get("cleanup_required"):
                        logger.info(
                            "[QDRANT CLEANUP] Completed before re-ingest for %s: "
                            "mode=%s deleted=%s",
                            file_info["source_path"],
                            cleanup_result.get("mode"),
                            cleanup_result.get("deleted"),
                        )
                except Exception as cleanup_exc:
                    update_manifest_after_failure(
                        manifest=manifest,
                        file_info=file_info,
                        ingestion_run_id=ingestion_run_id,
                        error=f"Qdrant cleanup failed: {cleanup_exc}",
                        status="cleanup_failed",
                        chunk_count=len(chunks),
                        last_ingested_at=ingested_at,
                    )
                    save_ingestion_manifest(manifest_path, manifest)
                    logger.error(
                        "[QDRANT CLEANUP] Failed for %s. Upsert skipped to avoid duplicates: %s",
                        file_info["source_path"],
                        cleanup_exc,
                    )
                    continue

                vectors = await stage4b_upsert_qdrant(
                    chunks,
                    payloads,
                    ingestion_run_id=ingestion_run_id,
                    ingested_at=ingested_at,
                )
                stats.vectors_upserted_qdrant += vectors
                qdrant_point_ids = qdrant_point_ids_for_chunks(chunks)
            else:
                logger.warning("[STAGE 4B] Skipped by --skip-qdrant")

            skipped_components = []
            intentional_graph_skip = (
                skip_graph_extraction
                and skip_neo4j
                and not skip_qdrant
                and bool(qdrant_point_ids)
            )
            file_info["graph_extraction_skipped"] = bool(skip_graph_extraction)
            file_info["neo4j_skipped"] = bool(skip_neo4j)
            file_info["qdrant_indexed"] = bool(qdrant_point_ids)
            file_info["graph_validated"] = (
                graph_errors == 0
                and not skip_graph_extraction
                and not skip_neo4j
                and not skip_qdrant
                and bool(qdrant_point_ids)
            )
            file_info["phase1_validation_status"] = "knowledge_only"

            if skip_neo4j and not intentional_graph_skip:
                skipped_components.append("Neo4j")
            if skip_qdrant:
                skipped_components.append("Qdrant")
            qdrant_missing_reason = None
            if not skip_qdrant and not qdrant_point_ids:
                qdrant_missing_reason = "Qdrant upsert produced no point IDs"

            graph_warning = None
            graph_errors_within_tolerance = (
                graph_errors > 0
                and graph_errors <= graph_tolerance
                and not limit_chunks_used_for_doc
                and not skipped_components
            )

            if graph_errors_within_tolerance:
                graph_warning = graph_warning_message(graph_errors, graph_tolerance)

            if (
                limit_chunks_used_for_doc
                or skipped_components
                or qdrant_missing_reason
                or (graph_errors > 0 and not graph_errors_within_tolerance)
            ):
                reasons = []
                if limit_chunks_used_for_doc:
                    reasons.append(
                        f"--limit-chunks processed {len(chunks)}/{original_chunk_count} chunks"
                        if truncated_by_limit
                        else "--limit-chunks used"
                    )
                if skipped_components:
                    reasons.append("skipped " + ", ".join(skipped_components))
                if qdrant_missing_reason:
                    reasons.append(qdrant_missing_reason)
                if graph_errors > 0 and not graph_errors_within_tolerance:
                    reasons.append(
                        f"graph extraction failed for {graph_errors} chunk(s); "
                        f"tolerance={graph_tolerance:g}"
                    )
                partial_reason = "; ".join(reasons)
                update_manifest_after_failure(
                    manifest=manifest,
                    file_info=file_info,
                    ingestion_run_id=ingestion_run_id,
                    error=partial_reason,
                    status="partial",
                    chunk_count=len(chunks),
                    qdrant_point_ids=qdrant_point_ids,
                    last_ingested_at=ingested_at,
                    graph_error_count=graph_errors,
                    graph_error_tolerance=graph_tolerance,
                )
                save_ingestion_manifest(manifest_path, manifest)
                logger.warning(
                    "[MANIFEST] Marked partial for %s: %s",
                    file_info["source_path"],
                    partial_reason,
                )
                continue

            update_manifest_after_success(
                manifest=manifest,
                source_path=str(file_info["source_path"]),
                document_id=str(file_info["document_id"]),
                content_hash=str(file_info["content_hash"]),
                file_size=int(file_info["file_size"]),
                modified_time=str(file_info["modified_time"]),
                chunk_count=len(chunks),
                qdrant_point_ids=qdrant_point_ids,
                ingestion_run_id=ingestion_run_id,
                last_ingested_at=ingested_at,
                status=(
                    "completed_with_graph_skipped"
                    if intentional_graph_skip
                    else "completed_with_warnings" if graph_warning else "completed"
                ),
                error_message=graph_warning,
                graph_error_count=graph_errors,
                graph_error_tolerance=graph_tolerance,
                warning=graph_warning,
                source_metadata=file_info,
                graph_validated=bool(file_info["graph_validated"]),
                phase1_validation_status="knowledge_only",
            )
            save_ingestion_manifest(manifest_path, manifest)
            logger.info(
                "[MANIFEST] Marked %s: %s",
                (
                    "completed_with_graph_skipped"
                    if intentional_graph_skip
                    else "completed_with_warnings" if graph_warning else "completed"
                ),
                file_info["source_path"],
            )

        except Exception as exc:
            logger.error(
                "[INCREMENTAL] Failed to ingest %s: %s",
                file_info["source_path"],
                exc,
                exc_info=True,
            )
            if not stage1_already_recorded_error:
                stats.parse_errors += 1
            if not dry_run:
                update_manifest_after_failure(
                    manifest=manifest,
                    file_info=file_info,
                    ingestion_run_id=ingestion_run_id,
                    error=exc,
                    status="failed",
                )
                save_ingestion_manifest(manifest_path, manifest)

    return stats


async def ingest_pipeline(
    source_dir: Path,
    dry_run: bool = False,
    limit_files: int | None = None,
    limit_chunks: int | None = None,
    refresh_markdown: bool = False,
    use_resume: bool = True,
    refresh_graph_cache: bool = False,
    skip_graph_extraction: bool = False,
    skip_neo4j: bool = False,
    skip_qdrant: bool = False,
    incremental: bool = False,
    force_reingest: bool = False,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    defer_manifest_completion: bool = False,
    require_zero_graph_errors: bool = False,
    skip_semantic_enrichment: bool = False,
) -> IngestionStats:
    if incremental or force_reingest:
        return await ingest_pipeline_incremental(
            source_dir=source_dir,
            dry_run=dry_run,
            limit_files=limit_files,
            limit_chunks=limit_chunks,
            refresh_markdown=refresh_markdown,
            use_resume=use_resume,
            refresh_graph_cache=refresh_graph_cache,
            skip_graph_extraction=skip_graph_extraction,
            skip_neo4j=skip_neo4j,
            skip_qdrant=skip_qdrant,
            force_reingest=force_reingest,
            manifest_path=manifest_path,
        )

    ensure_cache_dirs()
    stats = IngestionStats()
    ingestion_run_id = uuid.uuid4().hex
    ingested_at = utc_now_iso()

    logger.info("=" * 60)
    logger.info("[STAGE 1] Source Extraction via LlamaParse + local JSON loader")
    logger.info("=" * 60)

    parsed_docs = await stage1_extract_sources(
        source_dir=source_dir,
        limit_files=limit_files,
        refresh_markdown=refresh_markdown,
        stats=stats,
    )

    stats.pdf_files = len(parsed_docs)

    if not parsed_docs:
        logger.warning("No documents extracted. Aborting pipeline.")
        return stats

    logger.info("=" * 60)
    logger.info("[STAGE 2] Markdown-Aware Chunking (chunk_size=%d)", CHUNK_SIZE)
    logger.info("=" * 60)

    doc_chunks: list[tuple[dict[str, Any], list[SemanticChunk]]] = []

    for file_info, filename, markdown in parsed_docs:
        if is_web_json_source(file_info["path"]):
            chunks = stage2_chunk_web_json_file(file_info["path"], file_info)
        else:
            chunks = stage2_chunk_markdown(filename, markdown or "")
        enrich_chunks_with_ingestion_metadata(
            chunks=chunks,
            file_info=file_info,
            ingestion_run_id=ingestion_run_id,
            ingested_at=ingested_at,
        )
        doc_chunks.append((file_info, chunks))

    limit_chunks_truncated = limit_chunks is not None and limit_chunks > 0

    if limit_chunks is not None and limit_chunks > 0:
        logger.warning("[STAGE 2] Applying --limit-chunks=%d", limit_chunks)
        remaining_chunks = limit_chunks
        limited_doc_chunks: list[tuple[dict[str, Any], list[SemanticChunk]]] = []
        for file_info, chunks in doc_chunks:
            if remaining_chunks <= 0:
                limited_doc_chunks.append((file_info, []))
                if chunks:
                    limit_chunks_truncated = True
                continue

            if len(chunks) > remaining_chunks:
                limited_doc_chunks.append((file_info, chunks[:remaining_chunks]))
                remaining_chunks = 0
                limit_chunks_truncated = True
            else:
                limited_doc_chunks.append((file_info, chunks))
                remaining_chunks -= len(chunks)

        doc_chunks = limited_doc_chunks

    all_chunks: list[SemanticChunk] = [
        chunk
        for _, chunks in doc_chunks
        for chunk in chunks
    ]

    stats.chunks_created = len(all_chunks)
    account_chunk_quality(all_chunks, stats)

    logger.info("[STAGE 2] Total semantic chunks: %d", stats.chunks_created)

    if not all_chunks:
        logger.warning("No chunks produced. Aborting pipeline.")
        return stats

    manifest: dict[str, Any] | None = None
    if not dry_run:
        manifest = load_ingestion_manifest(manifest_path)
        for file_info, _ in doc_chunks:
            update_manifest_before_ingest(manifest, file_info, ingestion_run_id)
        save_ingestion_manifest(manifest_path, manifest)

    if skip_semantic_enrichment:
        if not skip_graph_extraction or not skip_neo4j:
            raise ValueError(
                "skip_semantic_enrichment requires skip_graph_extraction and skip_neo4j"
            )
        logger.info(
            "[STAGE 3/4A] Optional semantic enrichment is not part of this core ingestion run."
        )
        payloads = [GraphPayload(chunk_id=chunk.chunk_id) for chunk in all_chunks]
    else:
        payloads = await stage3_and_optional_neo4j_incremental(
            chunks=all_chunks,
            dry_run=dry_run,
            use_resume=use_resume,
            refresh_graph_cache=refresh_graph_cache,
            skip_graph_extraction=skip_graph_extraction,
            skip_neo4j=skip_neo4j,
            stats=stats,
        )

    if dry_run:
        logger.info("─" * 60)
        logger.info("[DRY RUN] Stage 4A/4B and manifest writes skipped.")
        logger.info("─" * 60)
        stats.report()
        return stats

    qdrant_upsert_succeeded = False

    if not skip_qdrant:
        logger.info("=" * 60)
        logger.info("[STAGE 4B] Qdrant Vector Indexing (dense + hashed sparse)")
        logger.info("=" * 60)

        try:
            vectors = await stage4b_upsert_qdrant(all_chunks, payloads)
            stats.vectors_upserted_qdrant = vectors
            qdrant_upsert_succeeded = True
        except Exception as exc:
            logger.error("[STAGE 4B] Failed: %s", exc, exc_info=True)
            stats.parse_errors += 1
    else:
        logger.warning("[STAGE 4B] Skipped by --skip-qdrant")

    if manifest is not None:
        global_error = None
        if stats.parse_errors > 0:
            global_error = f"pipeline completed with {stats.parse_errors} error(s)"
        finalize_manifest_for_documents(
            manifest=manifest,
            doc_chunks=doc_chunks,
            payloads=payloads,
            ingestion_run_id=ingestion_run_id,
            ingested_at=ingested_at,
            skip_neo4j=skip_neo4j,
            skip_qdrant=skip_qdrant,
            skip_graph_extraction=skip_graph_extraction,
            qdrant_ids_available=qdrant_upsert_succeeded,
            limit_chunks_truncated=limit_chunks_truncated,
            global_error=global_error,
            defer_completion=defer_manifest_completion,
            require_zero_graph_errors=require_zero_graph_errors,
        )
        save_ingestion_manifest(manifest_path, manifest)
        logger.info("[MANIFEST] Full ingestion manifest updated: %s", manifest_path)

    return stats


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Acne Advisor AI – Knowledge Base Ingestion Pipeline\n"
            "Stages: LlamaParse → Markdown Cache → Chunking → "
            "Graph Cache → Neo4j Batch MERGE → Qdrant Upsert"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=SAMPLE_DATA_DIR,
        metavar="DIR",
        help=f"Directory containing PDF files. Default: {SAMPLE_DATA_DIR}",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run parse/chunk/graph extraction only. Skip Neo4j and Qdrant.",
    )

    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Limit number of PDF files processed. Useful for testing.",
    )

    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Limit number of chunks processed. Useful for testing.",
    )

    parser.add_argument(
        "--refresh-markdown",
        action="store_true",
        help="Ignore Markdown cache and call LlamaParse again.",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Ingest only new, changed, failed, or partial source files using the manifest.",
    )

    parser.add_argument(
        "--force-reingest",
        action="store_true",
        help="Ignore manifest skip decisions and reingest scanned files.",
    )

    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        metavar="PATH",
        help=f"Incremental ingestion manifest path. Default: {DEFAULT_MANIFEST_PATH}",
    )

    parser.add_argument(
        "--no-resume",
        "--no-resume-graph-cache",
        dest="no_resume_graph_cache",
        action="store_true",
        help="Do not read graph cache; re-extract chunks with Ollama.",
    )

    parser.add_argument(
        "--refresh-graph-cache",
        action="store_true",
        help="Do not read existing graph cache, but write fresh valid cache after extraction.",
    )

    parser.add_argument(
        "--clear-graph-cache",
        action="store_true",
        help="Delete graph cache files before running. If used alone, delete and exit.",
    )

    parser.add_argument(
        "--skip-graph-extraction",
        action="store_true",
        help="Do not call Ollama. Load graph payloads from cache only.",
    )

    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Skip Neo4j graph upsert.",
    )

    parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Skip Qdrant vector upsert.",
    )

    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    graph_resume_enabled = not (args.no_resume_graph_cache or args.refresh_graph_cache)

    logger.info("=" * 60)
    logger.info("  Acne Advisor AI – Knowledge Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info("  Source dir              : %s", args.source)
    logger.info("  LlamaParse key          : %s", "✓ set" if LLAMA_CLOUD_API_KEY else "✗ MISSING")
    logger.info("  Ollama model            : %s @ %s", OLLAMA_MODEL, OLLAMA_BASE_URL)
    logger.info("  Embedding model         : %s (dim=%d)", EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)
    logger.info("  Qdrant                  : %s / %s", QDRANT_URL, QDRANT_COLLECTION_NAME)
    logger.info("  Neo4j                   : %s", NEO4J_URI)
    logger.info("  Chunk size              : %d", CHUNK_SIZE)
    logger.info("  LLM concurrency         : %d", LLM_CONCURRENCY)
    logger.info("  Graph batch size        : %d", GRAPH_BATCH_SIZE)
    logger.info("  Graph cache path        : %s", GRAPH_CACHE_DIR)
    logger.info("  Graph cache version     : %s", GRAPH_CACHE_VERSION)
    logger.info("  Graph prompt schema     : %s", GRAPH_PROMPT_SCHEMA_VERSION)
    logger.info("  Qdrant batch size       : %d", INGEST_BATCH_SIZE)
    logger.info("  Embedding batch delay   : %.1fs", EMBEDDING_BATCH_DELAY)
    logger.info("  Embedding max retries   : %d", EMBEDDING_MAX_RETRIES)
    logger.info("  Dry run                 : %s", args.dry_run)
    logger.info("  Limit files             : %s", args.limit_files)
    logger.info("  Limit chunks            : %s", args.limit_chunks)
    logger.info("  Refresh markdown        : %s", args.refresh_markdown)
    logger.info("  Incremental             : %s", args.incremental)
    logger.info("  Force reingest          : %s", args.force_reingest)
    logger.info("  Manifest path           : %s", args.manifest_path)
    logger.info("  Resume graph cache      : %s", graph_resume_enabled)
    logger.info("  Refresh graph cache     : %s", args.refresh_graph_cache)
    logger.info("  Clear graph cache       : %s", args.clear_graph_cache)
    logger.info("  Skip graph extraction   : %s", args.skip_graph_extraction)
    logger.info("  Skip Neo4j              : %s", args.skip_neo4j)
    logger.info("  Skip Qdrant             : %s", args.skip_qdrant)
    logger.info("=" * 60)

    if args.clear_graph_cache:
        deleted = clear_graph_cache_dir()
        logger.warning(
            "[CACHE] Cleared %d graph cache file(s) from %s",
            deleted,
            GRAPH_CACHE_DIR,
        )
        run_after_clear = any(
            [
                args.dry_run,
                args.limit_files is not None,
                args.limit_chunks is not None,
                args.refresh_markdown,
                args.incremental,
                args.force_reingest,
                args.manifest_path != DEFAULT_MANIFEST_PATH,
                args.no_resume_graph_cache,
                args.refresh_graph_cache,
                args.skip_graph_extraction,
                args.skip_neo4j,
                args.skip_qdrant,
                args.source != SAMPLE_DATA_DIR,
            ]
        )
        if not run_after_clear:
            return 0

    if not await run_preflight_checks(args, graph_resume_enabled):
        logger.error("[PREFLIGHT] One or more checks failed. Aborting before ingestion.")
        return 1

    try:
        stats = await ingest_pipeline(
            source_dir=args.source,
            dry_run=args.dry_run,
            limit_files=args.limit_files,
            limit_chunks=args.limit_chunks,
            refresh_markdown=args.refresh_markdown,
            use_resume=graph_resume_enabled,
            refresh_graph_cache=args.refresh_graph_cache,
            skip_graph_extraction=args.skip_graph_extraction,
            skip_neo4j=args.skip_neo4j,
            skip_qdrant=args.skip_qdrant,
            incremental=args.incremental,
            force_reingest=args.force_reingest,
            manifest_path=args.manifest_path,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. You can rerun; cached chunks will resume.")
        return 130
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        return 1

    logger.info("=" * 60)
    stats.report()

    if stats.parse_errors > 0:
        logger.warning("Pipeline completed with %d error(s).", stats.parse_errors)
        return 1

    logger.info("✅ Ingestion pipeline completed successfully.")
    logger.info("=" * 60)
    return 0


def cli() -> int:
    """Synchronous console-script wrapper."""

    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(cli())
