"""Biên dịch parsed source snapshots thành knowledge records deterministic.

Module nối parser output với chunking, filtering, provenance và domain metadata.
Nó chưa gọi embedding provider hay Qdrant; hai side effect đó thuộc
``src/ingestion/index.py``. Build identity là SHA-256 rút gọn của source-manifest
hash, taxonomy hash và serialized contract hash.

Muốn đổi record payload hoặc thứ tự compilation bắt đầu tại ``compile_knowledge``;
muốn đổi identity đọc ``compute_build_identity``. Mọi thay đổi contract hợp lệ
sẽ tạo build ID khác thay vì ghi đè identity cũ.
"""

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
from src.ingestion.filtering import (
    CLAIM_CURATION_CONTRACT_ID,
    FILTER_CONTRACT_ID,
    ClaimExclusion,
    apply_claim_exclusion,
    is_noisy_chunk,
)
from src.ingestion.normalization import NORMALIZATION_CONTRACT_ID
from src.ingestion.parser import PARSER_CONTRACT_ID, ParsedArtifact
from src.ingestion.provenance import (
    PROVENANCE_CONTRACT_ID,
    base_source_provenance,
    base_web_record_provenance,
    chunk_id,
    document_id,
    record_id,
    sha256_text,
)
from src.ingestion.source_manifest import CanonicalSource, WebRecordSource, source_manifest_hash


BUILD_MANIFEST_SCHEMA = "phase1_build_manifest"
BUILD_CONTRACT_ID = "record_provenance_curated_phase1_build_v3"
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
    """Tính build ID từ source, taxonomy và toàn bộ compilation contracts."""

    source_hash = source_manifest_hash(source_manifest_path)
    taxonomy_bytes = taxonomy_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    taxonomy_hash = hashlib.sha256(taxonomy_bytes).hexdigest()
    contract = {
        "build": BUILD_CONTRACT_ID,
        "parser": PARSER_CONTRACT_ID,
        "normalization": NORMALIZATION_CONTRACT_ID,
        "chunk": CHUNK_CONTRACT_ID,
        "filter": FILTER_CONTRACT_ID,
        "claim_curation": CLAIM_CURATION_CONTRACT_ID,
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


_STRUCTURED_CLINICAL_METADATA_FIELDS = (
    "language",
    "scope",
    "content_kind",
    "evidence_tier",
    "acne_type",
    "differential",
    "category",
    "level",
    "special_group",
    "skin_type",
    "care_topic",
    "doi",
    "published_year",
    "last_updated_year",
    "accessed_at",
    "table_index",
    "row_index",
    "table_title",
    "section_title",
    "table_representation",
    "license",
    "license_status",
    "license_url",
    "redistribution_status",
    "provenance",
)


def _structured_record_provenance(
    parent: CanonicalSource,
    unit_meta: dict[str, Any],
    unit_source_url: str,
) -> dict[str, Any]:
    """Dựng provenance chuẩn hóa cho từng bản ghi đã gắn nhãn nguồn sẵn trong JSON có cấu trúc."""

    child_source_id = str(unit_meta.get("source_id") or parent.source_id).strip()
    child_title = str(unit_meta.get("title") or parent.title).strip()
    child_authority = str(unit_meta.get("publisher") or parent.authority).strip()
    child_source_type = str(unit_meta.get("source_type") or parent.source_type).strip()
    child_url = str(
        unit_source_url
        or unit_meta.get("source_url")
        or unit_meta.get("document_url")
        or unit_meta.get("final_url")
        or unit_meta.get("seed_url")
        or parent.original_url
    ).strip()
    child_version = str(
        unit_meta.get("last_updated_year")
        or unit_meta.get("published_year")
        or unit_meta.get("accessed_at")
        or parent.version_date
    ).strip()
    child_hash = str(
        unit_meta.get("document_hash")
        or unit_meta.get("content_hash")
        or parent.sha256
    ).strip()
    child_doc_id = str(
        unit_meta.get("document_id") or document_id(child_source_id, child_hash)
    ).strip()
    return {
        "source_id": child_source_id,
        "source_title": child_title,
        "document_title": child_title,
        "source_authority": child_authority,
        "source_type": child_source_type,
        "source_url": child_url,
        "source_version_date": child_version,
        "source_content_hash": child_hash,
        "document_id": child_doc_id,
        "parent_source_id": parent.source_id,
        "parent_source_content_hash": parent.sha256,
    }


def compile_knowledge(
    sources: tuple[CanonicalSource, ...],
    artifacts: dict[str, ParsedArtifact],
    identity: BuildIdentity,
    *,
    web_record_catalogs: dict[str, dict[str, WebRecordSource]] | None = None,
    claim_exclusions: tuple[ClaimExclusion, ...] = (),
    ingested_at: str | None = None,
) -> CompiledKnowledge:
    """Chuyển parsed units thành ordered payload records cho bước indexing."""

    timestamp = ingested_at or datetime.now(UTC).isoformat()
    records: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    filtered_counts: dict[str, int] = {}
    web_record_catalogs = web_record_catalogs or {}
    actions_by_source: dict[str, dict[str, ClaimExclusion]] = {}
    for action in claim_exclusions:
        actions_by_source.setdefault(action.source_id, {})[action.pre_curation_content_hash] = (
            action
        )
    applied_actions: set[str] = set()

    for source in sources:
        artifact = artifacts.get(source.source_id)
        if artifact is None:
            raise ValueError(f"Missing parsed artifact for {source.source_id}")
        if source.record_catalog and source.source_id not in web_record_catalogs:
            raise ValueError(f"Missing child provenance catalog for {source.source_id}")
        if source.claim_exclusions and source.source_id not in actions_by_source:
            raise ValueError(f"Missing claim exclusions for {source.source_id}")
        record_catalog = web_record_catalogs.get(source.source_id, {})
        seen_content: set[str] = set()
        source_chunk_count = 0
        filtered = 0
        for unit in artifact.units:
            unit_meta = unit.metadata if isinstance(unit.metadata, dict) else {}
            is_structured_unit = bool(
                unit_meta.get("structured_record")
                and unit_meta.get("source_id")
                and unit_meta.get("title")
            )
            if record_catalog:
                child_source = record_catalog.get(unit.source_url)
                if child_source is None:
                    raise ValueError(f"Parsed web unit has no child provenance: {unit.source_url}")
                source_base = base_web_record_provenance(source, child_source)
            elif is_structured_unit:
                source_base = _structured_record_provenance(source, unit_meta, unit.source_url)
            else:
                source_base = base_source_provenance(source)
            unit_record_id = record_id(source_base["document_id"], unit.locator)
            unit_chunk_index = 0
            curated_section_path = tuple(
                part
                for part in (
                    str(unit_meta.get("section_title") or "").strip(),
                    str(unit_meta.get("table_title") or "").strip(),
                )
                if part
            )
            for chunk in structural_chunks(unit.text):
                effective_section_path = chunk.section_path or curated_section_path
                curated_text, pre_curation_hash, action_ids = apply_claim_exclusion(
                    chunk.text,
                    source=source,
                    actions_by_hash=actions_by_source.get(source.source_id, {}),
                )
                applied_actions.update(action_ids)
                noisy, _reason = is_noisy_chunk(
                    curated_text,
                    effective_section_path[-1] if effective_section_path else None,
                )
                content_hash = sha256_text(curated_text)
                if noisy or content_hash in seen_content:
                    filtered += 1
                    continue
                seen_content.add(content_hash)
                scoped_chunk_index = (
                    unit_chunk_index
                    if (record_catalog or is_structured_unit)
                    else source_chunk_count
                )
                identifier = chunk_id(
                    source_base["document_id"],
                    unit_record_id,
                    effective_section_path,
                    scoped_chunk_index,
                    content_hash,
                )
                source_url = unit.source_url or source_base["source_url"] or source.original_url
                header_text = effective_section_path[-1] if effective_section_path else ""
                payload: dict[str, Any] = {
                    **source_base,
                    "source_url": source_url,
                    "source_file": source.local_filename,
                    "source_path": source.local_filename,
                    "record_id": unit_record_id,
                    "page_start": unit.page_start,
                    "page_end": unit.page_end,
                    "section_path": list(effective_section_path),
                    "header": header_text,
                    "chunk_index": scoped_chunk_index,
                    "chunk_id": identifier,
                    "chunk_content_hash": content_hash,
                    "content_hash": content_hash,
                    "text": curated_text,
                    "content": curated_text,
                    "parser_contract_id": PARSER_CONTRACT_ID,
                    "parsed_output_hash": artifact.parsed_output_hash,
                    "normalized_output_hash": artifact.normalized_output_hash,
                    "normalization_contract_id": NORMALIZATION_CONTRACT_ID,
                    "chunk_contract_id": CHUNK_CONTRACT_ID,
                    "filter_contract_id": FILTER_CONTRACT_ID,
                    "claim_curation_contract_id": CLAIM_CURATION_CONTRACT_ID,
                    "provenance_contract_id": PROVENANCE_CONTRACT_ID,
                    "embedding_contract_id": EMBEDDING_CONTRACT_ID,
                    "bm25_contract_id": BM25_CONTRACT_ID,
                    "bm25_config": bm25_config().model_dump(mode="json"),
                    "build_id": identity.build_id,
                    "ingestion_run_id": identity.build_id,
                    "ingested_at": timestamp,
                    "semantic_enrichment": "removed",
                }
                if is_structured_unit:
                    if unit.page_start > 0:
                        payload["page"] = unit.page_start
                    if unit_meta.get("chunk_id"):
                        payload["original_chunk_id"] = str(unit_meta["chunk_id"])
                    if header_text and header_text.casefold() not in curated_text[:200].casefold():
                        payload["embedding_title"] = f"{source_base['source_title']} — {header_text}"
                    for extra_key in _STRUCTURED_CLINICAL_METADATA_FIELDS:
                        if extra_key in unit_meta and unit_meta[extra_key] is not None:
                            payload[extra_key] = unit_meta[extra_key]
                if action_ids:
                    payload["pre_curation_content_hash"] = pre_curation_hash
                    payload["curation_action_ids"] = list(action_ids)
                payload.update(enrich_domain_metadata(curated_text, existing_metadata=payload))
                records.append(payload)
                unit_chunk_index += 1
                source_chunk_count += 1
        source_counts[source.source_id] = source_chunk_count
        filtered_counts[source.source_id] = filtered

    expected_actions = {action.action_id for action in claim_exclusions}
    if applied_actions != expected_actions:
        missing = sorted(expected_actions - applied_actions)
        unexpected = sorted(applied_actions - expected_actions)
        raise ValueError(
            f"Claim exclusion application mismatch: missing={missing}, unexpected={unexpected}"
        )

    structural_hash = structural_records_hash(records)
    return CompiledKnowledge(
        identity=identity,
        records=tuple(records),
        source_counts=source_counts,
        filtered_counts=filtered_counts,
        structural_hash=structural_hash,
    )


def structural_records_hash(records: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    """Hash canonical records, bỏ timestamp để cùng content cho cùng kết quả."""

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
