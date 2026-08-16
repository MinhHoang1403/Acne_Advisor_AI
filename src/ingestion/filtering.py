"""Loại artifact chỉ khi pattern deterministic chứng minh được loại nội dung.

Filter bỏ text rỗng, page-number-only, một số standalone artifact chính xác và
dot leaders trong mục lục. Text ngắn không bị loại chỉ vì ngắn; module không dùng
LLM và không đánh giá medical quality. Exact deduplication giữ lần xuất hiện đầu.
Muốn đổi quy tắc lọc bắt đầu tại ``is_noisy_chunk()`` và các pattern trong file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml

from src.ingestion.provenance import sha256_text
from src.ingestion.source_manifest import CanonicalSource, sha256_portable_text_file


FILTER_CONTRACT_ID = "proof_only_artifact_filter"
CLAIM_CURATION_CONTRACT_ID = "source_bound_exact_claim_exclusion_v1"
CLAIM_EXCLUSION_SCHEMA = "source_bound_claim_exclusions"
CLAIM_EXCLUSION_SCHEMA_VERSION = 1

_PAGE_NUMBER_ONLY_RE = re.compile(r"^(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?$", re.IGNORECASE)
_DOT_LEADER_RE = re.compile(r"^.{1,180}?\.{3,}\s*\d{1,4}\s*$")
_DOTS_ONLY_RE = re.compile(r"^\.{3,}$")
_TOC_HEADERS = frozenset({"contents", "table of contents", "mục lục"})
_EXACT_STANDALONE_ARTIFACTS = frozenset({"references", "back to top", "© all rights reserved"})


@dataclass(frozen=True)
class ClaimExclusion:
    action_id: str
    source_id: str
    source_sha256: str
    pre_curation_content_hash: str
    exact_texts: tuple[str, ...]


def load_claim_exclusions(
    source: CanonicalSource,
    *,
    manifest_path: Path,
) -> tuple[ClaimExclusion, ...]:
    """Load exact source-bound removals and reject any sidecar mismatch."""

    if not source.claim_exclusions:
        return ()
    path = manifest_path.parent / source.claim_exclusions
    actual_hash = sha256_portable_text_file(path)
    if actual_hash != source.claim_exclusions_sha256:
        raise ValueError(f"Claim exclusion catalog hash mismatch for {source.source_id}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != CLAIM_EXCLUSION_SCHEMA
        or raw.get("schema_version") != CLAIM_EXCLUSION_SCHEMA_VERSION
        or raw.get("source_id") != source.source_id
        or raw.get("source_sha256") != source.sha256
    ):
        raise ValueError(f"Unsupported claim exclusion catalog at {path}")
    records = raw.get("actions")
    if not isinstance(records, list) or not records:
        raise ValueError("Claim exclusion catalog must contain actions")

    actions = tuple(_claim_exclusion(record, source) for record in records)
    action_ids = [action.action_id for action in actions]
    content_hashes = [action.pre_curation_content_hash for action in actions]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("Claim exclusion catalog contains duplicate action_id")
    if len(content_hashes) != len(set(content_hashes)):
        raise ValueError("Claim exclusion catalog contains duplicate pre-curation hash")
    return actions


def apply_claim_exclusion(
    text: str,
    *,
    source: CanonicalSource,
    actions_by_hash: dict[str, ClaimExclusion],
) -> tuple[str, str, tuple[str, ...]]:
    """Remove only exact approved text from the exact approved source chunk."""

    pre_hash = sha256_text(text)
    action = actions_by_hash.get(pre_hash)
    if action is None:
        return text, pre_hash, ()
    if action.source_id != source.source_id or action.source_sha256 != source.sha256:
        raise ValueError(f"Claim exclusion source mismatch for {action.action_id}")

    curated = text
    for exact_text in action.exact_texts:
        if curated.count(exact_text) != 1:
            raise ValueError(f"Exact stale claim mismatch for {action.action_id}")
        with_newline = exact_text + "\n"
        curated = (
            curated.replace(with_newline, "", 1)
            if with_newline in curated
            else curated.replace(exact_text, "", 1)
        )
    curated = curated.strip()
    if not curated or curated == text:
        raise ValueError(f"Claim exclusion made no valid change for {action.action_id}")
    return curated, pre_hash, (action.action_id,)


def is_noisy_chunk(text: str, header: str | None = None) -> tuple[bool, str]:
    """Chỉ reject khi text khớp chính xác một artifact rule đã khai báo."""

    stripped = text.strip()
    if not stripped:
        return True, "empty"
    if _PAGE_NUMBER_ONLY_RE.fullmatch(stripped):
        return True, "page_number_only"
    if stripped.casefold() in _EXACT_STANDALONE_ARTIFACTS:
        return True, "exact_standalone_artifact"
    if (header or "").strip().casefold() in _TOC_HEADERS and all(
        _DOT_LEADER_RE.fullmatch(line.strip()) or _DOTS_ONLY_RE.fullmatch(line.strip())
        for line in stripped.splitlines()
        if line.strip()
    ):
        return True, "toc_dot_leaders"
    return False, ""


def is_short_medical_safety_statement(text: str, header: str | None = None) -> bool:
    """Helper tương thích: mọi text ngắn không rỗng đều được giữ lại."""

    del header
    return bool(text.strip())


def deduplicate_chunks(texts: list[str]) -> tuple[list[str], list[int]]:
    """Giữ exact normalized chunk đầu tiên và trả index của bản trùng bị bỏ."""

    seen: set[str] = set()
    kept: list[str] = []
    removed: list[int] = []
    for index, text in enumerate(texts):
        key = text.strip()
        if key in seen:
            removed.append(index)
            continue
        seen.add(key)
        kept.append(text)
    return kept, removed


def _claim_exclusion(record: Any, source: CanonicalSource) -> ClaimExclusion:
    if not isinstance(record, dict):
        raise ValueError("Every claim exclusion action must be an object")
    action_id = str(record.get("action_id") or "").strip()
    pre_hash = str(record.get("pre_curation_content_hash") or "").strip()
    exact_texts = record.get("exact_texts")
    if not action_id or len(pre_hash) != 64:
        raise ValueError("Claim exclusion action has invalid identity")
    if (
        not isinstance(exact_texts, list)
        or not exact_texts
        or not all(isinstance(value, str) and value for value in exact_texts)
    ):
        raise ValueError(f"Claim exclusion {action_id} requires exact_texts")
    return ClaimExclusion(
        action_id=action_id,
        source_id=source.source_id,
        source_sha256=source.sha256,
        pre_curation_content_hash=pre_hash,
        exact_texts=tuple(exact_texts),
    )


__all__ = [
    "CLAIM_CURATION_CONTRACT_ID",
    "ClaimExclusion",
    "FILTER_CONTRACT_ID",
    "apply_claim_exclusion",
    "deduplicate_chunks",
    "is_noisy_chunk",
    "is_short_medical_safety_statement",
    "load_claim_exclusions",
]
