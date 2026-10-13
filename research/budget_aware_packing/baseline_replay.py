"""Exact offline replay of the frozen production context-packing baseline.

The reconstruction phase may perform read-only Qdrant exact-ID retrieval. The
replay phase is deliberately separate and operates only on the resulting local,
ignored cache. It never performs vector search, embedding, reranking, generation,
or database writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.retrieval.context_packer import _render_block, _render_block_from_item, pack_context
from src.retrieval.contracts import NormalizedQuery, PackedContext, RetrievedCandidate


JSONDict = dict[str, Any]
FORENSIC_SHA = "2d32317d6abd335aedd69b7b066fca53c13b0bb8"
FROZEN_PRODUCTION_SHA = "366f3ad836c36be36565caad4a94ad17a7a6278f"
FROZEN_RUN_ID = "thesis_evaluation_366f3ad8"
FROZEN_RAW_CANONICAL_SHA256 = "1f525efb44091c2444d298d45de4f31637b9be4d1d512f88db38ae9738125feb"
FROZEN_RAW_BYTE_SHA256 = "b078858d61ced4ed072339fd9cb576517a2ce585724e425c6fc0a2f772f57f16"
BENCHMARK_CANONICAL_SHA256 = "f61d6807c0ce39f902936844d810562c486f1bcadaa57a8a2da0e460ad7e534b"
KB_BUILD_ID = "d4a1819fe7fb77fe1f40"
PIPELINE_FINGERPRINT = "e4c420923b79b37c992cf36d"
ARTIFACT_REF = "evaluation/thesis-final-evaluation"
ARTIFACT_COMMIT = "5a19e3c079ef82006fbb42cba44e92560544b7da"

SOURCE_ARTIFACTS: dict[str, dict[str, str]] = {
    "benchmark": {
        "path": "evaluation/benchmark_100.json",
        "owner_commit": "bd3db1a87fc1e6491c11cab983dfa0b62112db00",
        "blob_oid": "d2aabf26e32fa16d031bfe69ff36555dec9cf36d",
    },
    "benchmark_manifest": {
        "path": "evaluation/benchmark_manifest.json",
        "owner_commit": "88b28c7e74ce1f110339ccd75e526bf96046ffcc",
        "blob_oid": "181f3edcb3778ffead0638be2c8ea217945b2c0d",
    },
    "gold": {
        "path": "evaluation/locked_gold/current_kb_adjudication.json",
        "owner_commit": "afc60883e9060c1dd041b2105c3e9a69a8313bf1",
        "blob_oid": "e91dbe5ccfcf455c8e7f6c61436afd39b58194c9",
    },
    "gold_manifest": {
        "path": "evaluation/locked_gold/current_kb_adjudication_manifest.json",
        "owner_commit": "afc60883e9060c1dd041b2105c3e9a69a8313bf1",
        "blob_oid": "08f2ac84773b097c8d516acc8ea9934ee5271436",
    },
    "raw_results": {
        "path": "evaluation/results/thesis_evaluation_366f3ad8/raw_results.json",
        "owner_commit": "c679994efc4463e96f7154dfb80296378c32ff02",
        "blob_oid": "04f75c350b5c2c8ba308f309b00dcc06c1905897",
    },
    "reference_metrics": {
        "path": "evaluation/results/thesis_evaluation_366f3ad8/deterministic_metrics.json",
        "owner_commit": ARTIFACT_COMMIT,
        "blob_oid": "c25df3cb89cf569f09d2bf470691e9ec1a7b021a",
    },
    "run_manifest": {
        "path": "evaluation/results/thesis_evaluation_366f3ad8/run_manifest.json",
        "owner_commit": ARTIFACT_COMMIT,
        "blob_oid": "c2c92d419bf6dda0b3f29626043468bbdccadcd5",
    },
    "system_output_manifest": {
        "path": "evaluation/results/thesis_evaluation_366f3ad8/system_output_manifest.json",
        "owner_commit": "c679994efc4463e96f7154dfb80296378c32ff02",
        "blob_oid": "f8998c91d3cb35988e14ca16aeceb8e4cd62a3e8",
    },
}

PRODUCTION_FILES = (
    "src/retrieval/context_packer.py",
    "src/retrieval/contracts.py",
    "src/retrieval/service.py",
    "src/retrieval/reranker.py",
)


class ReplayInputError(ValueError):
    """Raised when frozen identity or reconstructed inputs fail closed."""


@dataclass(frozen=True)
class FrozenInputs:
    raw_bytes: bytes
    raw: JSONDict
    benchmark: JSONDict
    benchmark_manifest: JSONDict
    gold: JSONDict
    gold_manifest: JSONDict
    reference_metrics: JSONDict
    run_manifest: JSONDict
    system_output_manifest: JSONDict
    blob_checks: dict[str, bool]


def canonical_json_bytes(value: Any) -> bytes:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (serialized + "\n").encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _git(repository: Path, *args: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8").strip()


def _load_json_bytes(value: bytes, label: str) -> JSONDict:
    try:
        loaded = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayInputError(f"Invalid UTF-8 JSON for {label}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ReplayInputError(f"{label} must be a JSON object")
    return loaded


def load_frozen_inputs(repository: Path) -> FrozenInputs:
    values: dict[str, JSONDict] = {}
    raw_bytes = b""
    blob_checks: dict[str, bool] = {}
    for label, identity in SOURCE_ARTIFACTS.items():
        actual_oid = str(
            _git(
                repository,
                "rev-parse",
                f"{identity['owner_commit']}:{identity['path']}",
            )
        )
        blob_checks[label] = actual_oid == identity["blob_oid"]
        content = _git(repository, "cat-file", "blob", identity["blob_oid"], binary=True)
        assert isinstance(content, bytes)
        values[label] = _load_json_bytes(content, label)
        if label == "raw_results":
            raw_bytes = content
    if not all(blob_checks.values()):
        failed = sorted(name for name, passed in blob_checks.items() if not passed)
        raise ReplayInputError(f"DATA_IDENTITY_BLOCKED: Git blob mismatch: {failed}")
    return FrozenInputs(
        raw_bytes=raw_bytes,
        raw=values["raw_results"],
        benchmark=values["benchmark"],
        benchmark_manifest=values["benchmark_manifest"],
        gold=values["gold"],
        gold_manifest=values["gold_manifest"],
        reference_metrics=values["reference_metrics"],
        run_manifest=values["run_manifest"],
        system_output_manifest=values["system_output_manifest"],
        blob_checks=blob_checks,
    )


def validate_frozen_identity(inputs: FrozenInputs) -> dict[str, bool]:
    identity = inputs.raw.get("identity") or {}
    raw_canonical = canonical_json_sha256(inputs.raw)
    raw_bytes = sha256_bytes(inputs.raw_bytes)
    benchmark_canonical = canonical_json_sha256(inputs.benchmark)
    checks = {
        "git_blobs_match": all(inputs.blob_checks.values()),
        "raw_canonical_matches": raw_canonical == FROZEN_RAW_CANONICAL_SHA256,
        "raw_bytes_match": raw_bytes == FROZEN_RAW_BYTE_SHA256,
        "benchmark_canonical_matches": benchmark_canonical == BENCHMARK_CANONICAL_SHA256,
        "production_sha_matches": identity.get("system_under_test_sha") == FROZEN_PRODUCTION_SHA,
        "run_id_matches": identity.get("run_id") == FROZEN_RUN_ID,
        "kb_build_matches": identity.get("kb_build_id") == KB_BUILD_ID,
        "pipeline_fingerprint_matches": identity.get("pipeline_fingerprint")
        == PIPELINE_FINGERPRINT,
        "record_count_matches": len(inputs.raw.get("records") or []) == 100,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ReplayInputError(f"DATA_IDENTITY_BLOCKED: failed checks: {failed}")
    return checks


def production_implementation_identity(repository: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in PRODUCTION_FILES:
        frozen_blob = str(_git(repository, "rev-parse", f"{FROZEN_PRODUCTION_SHA}:{path}"))
        current_blob = str(_git(repository, "rev-parse", f"HEAD:{path}"))
        files[path] = {
            "frozen_blob_oid": frozen_blob,
            "current_blob_oid": current_blob,
            "match": frozen_blob == current_blob,
        }
    if not all(item["match"] for item in files.values()):
        raise ReplayInputError("Baseline implementation differs from frozen production SHA")
    return {
        "evaluated_production_sha": FROZEN_PRODUCTION_SHA,
        "current_head": str(_git(repository, "rev-parse", "HEAD")),
        "files": files,
        "all_match": True,
    }


def frozen_candidate_ids(raw: JSONDict) -> list[str]:
    values = {
        str(candidate["candidate_id"])
        for record in raw.get("records") or []
        for attempt in record.get("retrieval_attempt_traces") or []
        for candidate in (attempt.get("candidate_trace") or {}).get("fused") or []
    }
    return sorted(values)


def _http_json(url: str, *, body: JSONDict | None = None) -> JSONDict:
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"} if encoded is not None else {},
        method="POST" if encoded is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ReplayInputError(f"Unexpected response shape from {url}")
    return value


def fetch_exact_candidate_cache(
    raw: JSONDict,
    *,
    qdrant_url: str,
    collection_alias: str = "acne_knowledge",
) -> JSONDict:
    """Fetch only known frozen IDs; never execute a similarity/search query."""

    base = qdrant_url.rstrip("/")
    aliases = (_http_json(f"{base}/aliases").get("result") or {}).get("aliases") or []
    target = next(
        (
            item.get("collection_name")
            for item in aliases
            if item.get("alias_name") == collection_alias
        ),
        None,
    )
    expected_target = f"acne_knowledge__{KB_BUILD_ID}"
    if target != expected_target:
        raise ReplayInputError(
            f"Qdrant alias mismatch: expected {expected_target}, received {target}"
        )
    collection = _http_json(f"{base}/collections/{collection_alias}").get("result") or {}
    if collection.get("status") != "green" or collection.get("points_count") != 536:
        raise ReplayInputError("Active Qdrant collection identity/health mismatch")

    candidate_ids = frozen_candidate_ids(raw)
    points: dict[str, JSONDict] = {}
    for start in range(0, len(candidate_ids), 128):
        response = _http_json(
            f"{base}/collections/{collection_alias}/points",
            body={
                "ids": candidate_ids[start : start + 128],
                "with_payload": True,
                "with_vector": False,
            },
        )
        for point in response.get("result") or []:
            point_id = str(point.get("id") or "")
            payload = point.get("payload")
            if point_id and isinstance(payload, dict):
                points[point_id] = payload

    missing = sorted(set(candidate_ids) - set(points))
    if missing:
        raise ReplayInputError(f"Missing exact candidate IDs in active build: {missing}")
    invalid = sorted(
        point_id
        for point_id, payload in points.items()
        if payload.get("build_id") != KB_BUILD_ID
        or str(payload.get("chunk_id") or "") != point_id
        or not str(payload.get("text") or "").strip()
    )
    if invalid:
        raise ReplayInputError(f"Candidate payload identity mismatch: {invalid}")
    return {
        "schema": "budget_aware_candidate_cache_v1",
        "source": {
            "kind": "qdrant_exact_id_lookup",
            "collection_alias": collection_alias,
            "collection_target": target,
            "kb_build_id": KB_BUILD_ID,
            "read_only": True,
            "vector_search": False,
            "embedding_calls": 0,
            "reranker_calls": 0,
            "database_writes": 0,
        },
        "candidate_count": len(candidate_ids),
        "candidate_ids_sha256": sha256_text("\n".join(candidate_ids) + "\n"),
        "points": {key: points[key] for key in sorted(points)},
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_candidate_cache(path: Path, cache: JSONDict) -> None:
    _write_json(path, cache)


def load_candidate_cache(path: Path) -> JSONDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayInputError(f"Cannot read candidate cache {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != "budget_aware_candidate_cache_v1":
        raise ReplayInputError("Candidate cache schema mismatch")
    points = value.get("points")
    if not isinstance(points, dict) or len(points) != value.get("candidate_count"):
        raise ReplayInputError("Candidate cache point count mismatch")
    ids = sorted(points)
    if sha256_text("\n".join(ids) + "\n") != value.get("candidate_ids_sha256"):
        raise ReplayInputError("Candidate cache identity checksum mismatch")
    return value


def reconstruct_candidate(trace: JSONDict, payload: JSONDict) -> RetrievedCandidate:
    candidate_id = str(trace.get("candidate_id") or "")
    if not candidate_id or str(payload.get("chunk_id") or "") != candidate_id:
        raise ReplayInputError(f"Candidate identity mismatch for {candidate_id or '<missing>'}")
    if payload.get("build_id") != KB_BUILD_ID:
        raise ReplayInputError(f"Candidate {candidate_id} belongs to a different KB build")
    text = str(payload.get("text") or "")
    if not text.strip():
        raise ReplayInputError(f"Candidate {candidate_id} is missing text")
    source_id = _source_id(payload)
    if not source_id:
        raise ReplayInputError(f"Candidate {candidate_id} is missing provenance")
    if trace.get("source_id") is not None and str(trace["source_id"]) != source_id:
        raise ReplayInputError(f"Candidate {candidate_id} source provenance mismatch")
    if trace.get("section") is not None and trace.get("section") != _section(payload):
        raise ReplayInputError(f"Candidate {candidate_id} section provenance mismatch")
    if trace.get("text_sha256") is not None and trace["text_sha256"] != sha256_text(text):
        raise ReplayInputError(f"Candidate {candidate_id} text checksum mismatch")
    candidate_payload = dict(payload)
    candidate_payload["id"] = candidate_id
    return RetrievedCandidate(
        candidate_id=candidate_id,
        collection="acne_knowledge",
        text=text,
        score=float(trace.get("fused_score") or 0.0),
        fused_score=float(trace.get("fused_score") or 0.0),
        rerank_score=(
            float(trace["rerank_score"]) if trace.get("rerank_score") is not None else None
        ),
        rerank_rank=(int(trace["rerank_rank"]) if trace.get("rerank_rank") is not None else None),
        payload=candidate_payload,
        rank=int(trace["rank"]) if trace.get("rank") is not None else None,
        debug={
            "dense_rank": trace.get("dense_rank"),
            "bm25_rank": trace.get("bm25_rank"),
            "seen_in_attempts": list(trace.get("seen_in_attempts") or []),
            "rank_by_attempt": dict(trace.get("rank_by_attempt") or {}),
        },
    )


def _validate_attempt_traces(traces: list[JSONDict]) -> None:
    seen: set[str] = set()
    for trace in traces:
        candidate_id = str(trace.get("candidate_id") or "")
        if not candidate_id:
            raise ReplayInputError("Frozen attempt contains a candidate without an ID")
        if candidate_id in seen:
            raise ReplayInputError(
                f"Frozen attempt contains duplicate candidate ID: {candidate_id}"
            )
        seen.add(candidate_id)


def candidate_from_cache(trace: JSONDict, cache: JSONDict) -> RetrievedCandidate:
    candidate_id = str(trace.get("candidate_id") or "")
    payload = (cache.get("points") or {}).get(candidate_id)
    if not isinstance(payload, dict):
        raise ReplayInputError(f"Missing cached payload for candidate {candidate_id}")
    return reconstruct_candidate(trace, payload)


def _source_id(payload: JSONDict) -> str:
    return str(
        payload.get("source_id")
        or payload.get("source_path")
        or payload.get("source_file")
        or payload.get("document_id")
        or ""
    ).strip()


def _section(payload: JSONDict) -> Any:
    section_path = payload.get("section_path")
    return payload.get("header") or (
        section_path[-1] if isinstance(section_path, list) and section_path else None
    )


def packed_evidence_summary(packed: PackedContext) -> list[JSONDict]:
    return [
        {
            "item_id": item.item_id,
            "source_id": _source_id(item.payload),
            "section": _section(item.payload),
        }
        for item in packed.items
    ]


def _safe_value(value: Any) -> Any:
    """Keep mismatch artifacts compact and free of medical corpus text."""

    if isinstance(value, str) and len(value) > 300:
        return {"sha256": sha256_text(value), "length": len(value)}
    if isinstance(value, list) and len(value) > 50:
        return {"count": len(value), "sha256": canonical_json_sha256(value)}
    return value


def _compare(
    mismatches: list[JSONDict],
    *,
    case_id: str,
    attempt_index: int,
    field: str,
    expected: Any,
    actual: Any,
) -> None:
    if expected != actual:
        mismatches.append(
            {
                "case_id": case_id,
                "attempt_index": attempt_index,
                "field": field,
                "expected": _safe_value(expected),
                "actual": _safe_value(actual),
            }
        )


def _header_metadata(payload: JSONDict, candidate_id: str) -> JSONDict:
    return {
        "source_id": _source_id(payload),
        "chunk_id": str(payload.get("chunk_id") or candidate_id),
        "section_path": payload.get("section_path"),
        "header": payload.get("header"),
        "drug_product": payload.get("drug_product"),
        "active_ingredient": payload.get("active_ingredient"),
        "drug_class": payload.get("drug_class"),
    }


def replay_baseline(
    inputs: FrozenInputs,
    cache: JSONDict,
    *,
    implementation_identity: JSONDict,
) -> tuple[JSONDict, JSONDict, JSONDict]:
    """Replay all attempts with no network or provider access."""

    records = inputs.raw.get("records") or []
    mismatches: list[JSONDict] = []
    attempts_total = 0
    attempts_with_complete_inputs = 0
    selected_matches = 0
    order_matches = 0
    context_length_matches = 0
    context_hash_verifiable = 0
    context_hash_matches = 0
    dropped_matches = 0
    not_examined_matches = 0
    provenance_matches = 0
    final_selected_by_case: dict[str, list[str]] = {}
    candidate_audit: list[JSONDict] = []

    for record in records:
        case_id = str(record["case_id"])
        record_attempts = record.get("retrieval_attempt_traces") or []
        for attempt_offset, attempt in enumerate(record_attempts):
            attempts_total += 1
            attempt_index = int(attempt.get("attempt_index") or attempt_offset + 1)
            traces = (attempt.get("candidate_trace") or {}).get("fused") or []
            _validate_attempt_traces(traces)
            candidates = [candidate_from_cache(trace, cache) for trace in traces]
            attempts_with_complete_inputs += 1
            normalized = NormalizedQuery(
                original_query=str(
                    attempt.get("overall_rerank_query") or record.get("query") or ""
                ),
                normalized_text=str(
                    attempt.get("overall_rerank_query") or attempt.get("rerank_query") or ""
                ),
            )
            expected_packer = attempt.get("packer") or {}
            limits = expected_packer.get("limits") or {}
            packed = pack_context(
                normalized,
                candidates,
                max_items=int(limits.get("max_items") or 9),
                max_chars=int(limits.get("max_chars") or 7000),
            )
            actual_ids = [item.item_id for item in packed.items]
            expected_ids = list(expected_packer.get("selected_ids") or [])
            before = len(mismatches)
            _compare(
                mismatches,
                case_id=case_id,
                attempt_index=attempt_index,
                field="selected_ids",
                expected=expected_ids,
                actual=actual_ids,
            )
            if len(mismatches) == before:
                selected_matches += 1
                order_matches += 1

            before = len(mismatches)
            _compare(
                mismatches,
                case_id=case_id,
                attempt_index=attempt_index,
                field="context_chars",
                expected=expected_packer.get("context_chars"),
                actual=len(packed.context_text),
            )
            if len(mismatches) == before:
                context_length_matches += 1

            before = len(mismatches)
            _compare(
                mismatches,
                case_id=case_id,
                attempt_index=attempt_index,
                field="dropped",
                expected=list(expected_packer.get("dropped") or []),
                actual=list(packed.debug.get("dropped") or []),
            )
            if len(mismatches) == before:
                dropped_matches += 1

            before = len(mismatches)
            _compare(
                mismatches,
                case_id=case_id,
                attempt_index=attempt_index,
                field="not_examined",
                expected=list(expected_packer.get("not_examined") or []),
                actual=list(packed.debug.get("not_examined") or []),
            )
            if len(mismatches) == before:
                not_examined_matches += 1

            before = len(mismatches)
            _compare(
                mismatches,
                case_id=case_id,
                attempt_index=attempt_index,
                field="packed_provenance",
                expected=list(attempt.get("packed_evidence") or []),
                actual=packed_evidence_summary(packed),
            )
            if len(mismatches) == before:
                provenance_matches += 1

            is_final = attempt_offset == len(record_attempts) - 1
            if is_final:
                final_selected_by_case[case_id] = actual_ids
                context_hash_verifiable += 1
                before = len(mismatches)
                _compare(
                    mismatches,
                    case_id=case_id,
                    attempt_index=attempt_index,
                    field="final_context_text_sha256",
                    expected=record.get("generation_context_text_sha256"),
                    actual=sha256_text(packed.context_text),
                )
                _compare(
                    mismatches,
                    case_id=case_id,
                    attempt_index=attempt_index,
                    field="final_context_text",
                    expected=record.get("generation_context_text"),
                    actual=packed.context_text,
                )
                frozen_contexts = list(record.get("generation_contexts") or [])
                rendered = [
                    {
                        "position": index,
                        "chunk_id": item.item_id,
                        "source_id": _source_id(item.payload),
                        "source_title": item.payload.get("source_title"),
                        "source_url": item.payload.get("source_url"),
                        "section_path": item.payload.get("section_path"),
                        "text_sha256": sha256_text(item.text),
                        "rendered_text_sha256": sha256_text(_render_block_from_item(item, index)),
                    }
                    for index, item in enumerate(packed.items, 1)
                ]
                expected_rendered = [
                    {
                        key: item.get(key)
                        for key in (
                            "position",
                            "chunk_id",
                            "source_id",
                            "source_title",
                            "source_url",
                            "section_path",
                            "text_sha256",
                            "rendered_text_sha256",
                        )
                    }
                    for item in frozen_contexts
                ]
                _compare(
                    mismatches,
                    case_id=case_id,
                    attempt_index=attempt_index,
                    field="final_context_items",
                    expected=expected_rendered,
                    actual=rendered,
                )
                if len(mismatches) == before:
                    context_hash_matches += 1

            reranker_events = record.get("reranker_events") or []
            reranker_status = (
                reranker_events[attempt_offset].get("status")
                if attempt_offset < len(reranker_events)
                else None
            )
            for final_order, (trace, candidate) in enumerate(
                zip(traces, candidates, strict=True), 1
            ):
                metadata = _header_metadata(candidate.payload, candidate.candidate_id)
                trace_source_matches = str(trace.get("source_id") or "") == _source_id(
                    candidate.payload
                )
                trace_section_matches = trace.get("section") == _section(candidate.payload)
                candidate_audit.append(
                    {
                        "case_id": case_id,
                        "attempt_index": attempt_index,
                        "candidate_id": candidate.candidate_id,
                        "frozen_final_order": final_order,
                        "rrf_rank": trace.get("rrf_rank"),
                        "rerank_rank": trace.get("rerank_rank"),
                        "reranker_status": reranker_status,
                        "text_source": "qdrant_exact_id_lookup",
                        "text_sha256": sha256_text(candidate.text),
                        "text_length": len(candidate.text),
                        "source_id": _source_id(candidate.payload),
                        "chunk_id": str(
                            candidate.payload.get("chunk_id") or candidate.candidate_id
                        ),
                        "header_relevant_metadata_sha256": canonical_json_sha256(metadata),
                        "reconstructed_rendered_cost": len(_render_block(candidate, 1)),
                        "identity_status": (
                            "MATCH"
                            if trace_source_matches and trace_section_matches
                            else "MISMATCH"
                        ),
                        "reconstruction_confidence": "EXACT_BUILD_ID_LOOKUP_REPLAY_VERIFIED",
                    }
                )

    requirements: list[tuple[str, str]] = []
    claims = inputs.gold.get("gold_claim_adjudications") or []
    by_case: dict[str, list[str]] = {}
    for claim in claims:
        case_id = str(claim.get("case_id") or "")
        for evidence_id in claim.get("current_evidence_ids") or []:
            key = (case_id, str(evidence_id))
            if key not in requirements:
                requirements.append(key)
                by_case.setdefault(case_id, []).append(str(evidence_id))
    records_by_case = {str(record["case_id"]): record for record in records}
    acquired = sum(
        evidence_id in set(records_by_case[case_id].get("acquisition_candidate_ids") or [])
        for case_id, evidence_id in requirements
    )
    packed_hits = sum(
        evidence_id in set(final_selected_by_case.get(case_id, []))
        for case_id, evidence_id in requirements
    )
    complete_cases = sum(
        all(evidence_id in set(final_selected_by_case.get(case_id, [])) for evidence_id in ids)
        for case_id, ids in by_case.items()
    )
    stage_counts = {"character_limit": 0, "item_limit_reached": 0}
    for case_id, evidence_id in requirements:
        if evidence_id in set(final_selected_by_case.get(case_id, [])):
            continue
        record = records_by_case[case_id]
        if evidence_id not in set(record.get("acquisition_candidate_ids") or []):
            continue
        final_fused = {
            str(item.get("candidate_id")): item
            for item in ((record.get("retrieval_trace") or {}).get("candidate_trace") or {}).get(
                "fused"
            )
            or []
        }
        item = final_fused.get(evidence_id) or {}
        if item.get("drop_reason") == "character_limit":
            stage_counts["character_limit"] += 1
        if item.get("not_examined_reason") == "item_limit_reached":
            stage_counts["item_limit_reached"] += 1

    metric_checks = {
        "cases_100": len(records) == 100,
        "attempts_104": attempts_total == 104,
        "requirements_88": len(requirements) == 88,
        "acquired_85": acquired == 85,
        "packed_58": packed_hits == 58,
        "character_losses_20": stage_counts["character_limit"] == 20,
        "item_losses_7": stage_counts["item_limit_reached"] == 7,
        "complete_cases_46": complete_cases == 46,
    }
    gate_passed = (
        not mismatches
        and all(metric_checks.values())
        and attempts_with_complete_inputs == attempts_total
        and selected_matches == attempts_total
        and context_length_matches == attempts_total
        and dropped_matches == attempts_total
        and not_examined_matches == attempts_total
        and provenance_matches == attempts_total
        and context_hash_matches == context_hash_verifiable
    )
    results = {
        "schema": "budget_aware_baseline_replay_results_v1",
        "status": "EXACT_BASELINE_REPLAY_CONFIRMED" if gate_passed else "BASELINE_REPLAY_BLOCKED",
        "gate_passed": gate_passed,
        "execution_type": "deterministic_offline_packer_replay",
        "network_during_replay": False,
        "provider_calls": 0,
        "retrieval_search_calls": 0,
        "database_writes": 0,
        "cases": len(records),
        "retrieval_attempts": attempts_total,
        "attempts_with_complete_inputs": attempts_with_complete_inputs,
        "selected_id_matches": selected_matches,
        "selected_order_matches": order_matches,
        "context_length_matches": context_length_matches,
        "context_hash_verifiable": context_hash_verifiable,
        "context_hash_matches": context_hash_matches,
        "dropped_decision_matches": dropped_matches,
        "not_examined_decision_matches": not_examined_matches,
        "provenance_matches": provenance_matches,
        "mismatch_count": len(mismatches),
        "metrics": {
            "required_evidence": len(requirements),
            "acquired": acquired,
            "packed": packed_hits,
            "character_losses": stage_counts["character_limit"],
            "item_losses": stage_counts["item_limit_reached"],
            "complete_answerable_cases": complete_cases,
        },
        "metric_checks": metric_checks,
        "baseline_implementation_identity": implementation_identity,
    }
    cache_bytes = canonical_json_bytes(cache)
    manifest = {
        "schema": "budget_aware_reconstruction_manifest_v1",
        "status": "VERIFIED" if gate_passed else "RECONSTRUCTION_MISMATCH",
        "source_priority_a": {
            "status": "INSUFFICIENT",
            "reason": "Frozen trace omits nonpacked candidate text and full payload.",
        },
        "source_priority_b": {
            "status": "NOT_USED",
            "reason": (
                "Build metadata and parsed sources exist, but no exact persisted candidate "
                "object array is bound to every frozen attempt."
            ),
        },
        "source_priority_c": {
            **dict(cache.get("source") or {}),
            "status": "VERIFIED_BY_EXACT_BASELINE_REPLAY" if gate_passed else "MISMATCH",
        },
        "candidate_cache": {
            "tracked": False,
            "candidate_count": cache.get("candidate_count"),
            "canonical_sha256": sha256_bytes(cache_bytes),
            "candidate_ids_sha256": cache.get("candidate_ids_sha256"),
        },
        "candidate_occurrences": len(candidate_audit),
        "candidate_audit": candidate_audit,
    }
    mismatch_output = {
        "schema": "budget_aware_baseline_replay_mismatches_v1",
        "status": "NO_MISMATCHES" if not mismatches else "MISMATCHES_FOUND",
        "count": len(mismatches),
        "mismatches": mismatches,
    }
    return results, mismatch_output, manifest


def run(
    *,
    repository: Path,
    cache_path: Path,
    output_dir: Path,
    qdrant_url: str,
    refresh_cache: bool,
) -> JSONDict:
    inputs = load_frozen_inputs(repository)
    validate_frozen_identity(inputs)
    implementation_identity = production_implementation_identity(repository)
    if refresh_cache:
        cache = fetch_exact_candidate_cache(inputs.raw, qdrant_url=qdrant_url)
        write_candidate_cache(cache_path, cache)
    else:
        cache = load_candidate_cache(cache_path)
    results, mismatches, manifest = replay_baseline(
        inputs,
        cache,
        implementation_identity=implementation_identity,
    )
    _write_json(output_dir / "reconstruction_manifest.json", manifest)
    _write_json(output_dir / "baseline_replay_results.json", results)
    _write_json(output_dir / "baseline_replay_mismatches.json", mismatches)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/cache/research_budget_aware_packing/candidates.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/results/budget_aware_packing"),
    )
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(
            repository=args.repository.resolve(),
            cache_path=(args.repository / args.cache_path).resolve(),
            output_dir=(args.repository / args.output_dir).resolve(),
            qdrant_url=args.qdrant_url,
            refresh_cache=args.refresh_cache,
        )
    except (ReplayInputError, OSError, subprocess.CalledProcessError) as exc:
        print(f"BASELINE_REPLAY_BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
