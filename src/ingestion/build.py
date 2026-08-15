"""Pure deterministic compilation of source snapshots into knowledge records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.ingestion.bm25 import BM25_CONTRACT_ID, bm25_config
from src.ingestion.chunking import CHUNK_CONTRACT_ID, structural_chunks
from src.ingestion.domain_metadata import enrich_domain_metadata
from src.ingestion.embedding import EMBEDDING_CONTRACT_ID
from src.ingestion.filtering import FILTER_CONTRACT_ID, is_noisy_chunk
from src.ingestion.normalization import NORMALIZATION_CONTRACT_ID
from src.ingestion.parser import PARSER_CONTRACT_ID, ParsedArtifact
from src.ingestion.provenance import (
    PROVENANCE_CONTRACT_ID,
    base_source_provenance,
    chunk_id,
    record_id,
    sha256_text,
)
from src.ingestion.source_manifest import CanonicalSource, source_manifest_hash


BUILD_MANIFEST_SCHEMA = "phase1_build_manifest"
BUILD_CONTRACT_ID = "frozen_phase1_build"
GRAPH_CONTRACT_ID = "deterministic_source_backed_taxonomy_graph"
ENTITY_CARD_CONTRACT_ID = "narrow_source_backed_entity_cards"


@dataclass(frozen=True)
class BuildIdentity:
    build_id: str
    source_manifest_hash: str
    taxonomy_hash: str
    contract_hash: str


@dataclass(frozen=True)
class CompiledKnowledge:
    identity: BuildIdentity
    records: tuple[dict[str, Any], ...]
    source_counts: dict[str, int]
    filtered_counts: dict[str, int]
    structural_hash: str


def compute_build_identity(source_manifest_path: Path, taxonomy_path: Path) -> BuildIdentity:
    source_hash = source_manifest_hash(source_manifest_path)
    taxonomy_bytes = (
        taxonomy_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    )
    taxonomy_hash = hashlib.sha256(taxonomy_bytes).hexdigest()
    contract = {
        "build": BUILD_CONTRACT_ID,
        "parser": PARSER_CONTRACT_ID,
        "normalization": NORMALIZATION_CONTRACT_ID,
        "chunk": CHUNK_CONTRACT_ID,
        "filter": FILTER_CONTRACT_ID,
        "provenance": PROVENANCE_CONTRACT_ID,
        "embedding": EMBEDDING_CONTRACT_ID,
        "bm25": BM25_CONTRACT_ID,
        "entity_card": ENTITY_CARD_CONTRACT_ID,
        "graph": GRAPH_CONTRACT_ID,
    }
    raw_contract = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract_hash = sha256_text(raw_contract)
    build_id = sha256_text("\0".join((source_hash, taxonomy_hash, contract_hash)))[:20]
    return BuildIdentity(build_id, source_hash, taxonomy_hash, contract_hash)


def compile_knowledge(
    sources: tuple[CanonicalSource, ...],
    artifacts: dict[str, ParsedArtifact],
    identity: BuildIdentity,
    *,
    ingested_at: str | None = None,
) -> CompiledKnowledge:
    """Compile parser artifacts into deterministic Qdrant payload records."""

    timestamp = ingested_at or datetime.now(UTC).isoformat()
    records: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    filtered_counts: dict[str, int] = {}

    for source in sources:
        artifact = artifacts.get(source.source_id)
        if artifact is None:
            raise ValueError(f"Missing parsed artifact for {source.source_id}")
        source_base = base_source_provenance(source)
        seen_content: set[str] = set()
        chunk_index = 0
        filtered = 0
        for unit in artifact.units:
            unit_record_id = record_id(source_base["document_id"], unit.locator)
            for chunk in structural_chunks(unit.text):
                noisy, _reason = is_noisy_chunk(
                    chunk.text,
                    chunk.section_path[-1] if chunk.section_path else None,
                )
                content_hash = sha256_text(chunk.text)
                if noisy or content_hash in seen_content:
                    filtered += 1
                    continue
                seen_content.add(content_hash)
                identifier = chunk_id(
                    source_base["document_id"],
                    unit_record_id,
                    chunk.section_path,
                    chunk_index,
                    content_hash,
                )
                source_url = unit.source_url or source.original_url
                payload: dict[str, Any] = {
                    **source_base,
                    "source_url": source_url,
                    "source_file": source.local_filename,
                    "source_path": source.local_filename,
                    "record_id": unit_record_id,
                    "page_start": unit.page_start,
                    "page_end": unit.page_end,
                    "section_path": list(chunk.section_path),
                    "header": chunk.section_path[-1] if chunk.section_path else "",
                    "chunk_index": chunk_index,
                    "chunk_id": identifier,
                    "chunk_content_hash": content_hash,
                    "content_hash": content_hash,
                    "text": chunk.text,
                    "content": chunk.text,
                    "parser_contract_id": PARSER_CONTRACT_ID,
                    "parsed_output_hash": artifact.parsed_output_hash,
                    "normalized_output_hash": artifact.normalized_output_hash,
                    "normalization_contract_id": NORMALIZATION_CONTRACT_ID,
                    "chunk_contract_id": CHUNK_CONTRACT_ID,
                    "filter_contract_id": FILTER_CONTRACT_ID,
                    "provenance_contract_id": PROVENANCE_CONTRACT_ID,
                    "embedding_contract_id": EMBEDDING_CONTRACT_ID,
                    "bm25_contract_id": BM25_CONTRACT_ID,
                    "bm25_config": bm25_config().model_dump(mode="json"),
                    "build_id": identity.build_id,
                    "ingestion_run_id": identity.build_id,
                    "ingested_at": timestamp,
                    "semantic_enrichment": "removed",
                }
                payload.update(enrich_domain_metadata(chunk.text, existing_metadata=payload))
                records.append(payload)
                chunk_index += 1
        source_counts[source.source_id] = chunk_index
        filtered_counts[source.source_id] = filtered

    structural_hash = structural_records_hash(records)
    return CompiledKnowledge(
        identity=identity,
        records=tuple(records),
        source_counts=source_counts,
        filtered_counts=filtered_counts,
        structural_hash=structural_hash,
    )


def structural_records_hash(records: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    stable_records = []
    for record in records:
        stable = {key: value for key, value in record.items() if key != "ingested_at"}
        stable_records.append(stable)
    raw = json.dumps(stable_records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(raw)


__all__ = [
    "BUILD_CONTRACT_ID",
    "BUILD_MANIFEST_SCHEMA",
    "BuildIdentity",
    "CompiledKnowledge",
    "ENTITY_CARD_CONTRACT_ID",
    "GRAPH_CONTRACT_ID",
    "compile_knowledge",
    "compute_build_identity",
    "structural_records_hash",
]
