"""
src/database/retriever.py – Hybrid Retriever
=============================================
Combines Qdrant semantic search (dense + sparse BM25) with Neo4j
knowledge graph context for comprehensive retrieval.

Pipeline
--------
  User Query
    ├─→ [1] Qdrant Dense Search   (semantic similarity)
    ├─→ [2] Qdrant Sparse BM25    (lexical keyword match)
    ├─→ [3] RRF Fusion            (merge & re-rank)
    ├─→ [3.5] Metadata Boost      (Phase 1.5 query-adaptive boost)
    └─→ [4] Neo4j KG Enrichment   (graph_nodes → relationships)

  Output: RetrievalResult(vector_contexts, graph_facts, sources)
"""

from __future__ import annotations

import logging
import asyncio
import contextlib
import os
import time
from dataclasses import dataclass, field
from typing import Any

from src.database.graph_store import Neo4jGraphStore
from src.database.vector_store import QdrantVectorStore, embed_query
from src.retrieval.candidate_merge import candidate_debug_summary, merge_candidates
from src.retrieval.candidate_policy import (
    CandidatePolicyResult,
    apply_candidate_policy,
    candidate_policy_budget,
)
from src.retrieval.contracts import RerankTrace, RetrievalTrace
from src.retrieval.context_packer import pack_context, packed_context_to_legacy_contexts
from src.retrieval.diagnostics import ConceptMatcher, build_retrieval_diagnostic_trace
from src.retrieval.entity_retriever import EntityRetriever
from src.retrieval.metadata_boost import annotate_chunk_results, boost_chunk_results
from src.retrieval.query_expansion import expand_normalized_query
from src.retrieval.query_normalization import normalize_query
from src.retrieval.reranker import (
    rerank_candidates,
    rerank_enabled_from_env,
    rerank_provider_from_env,
    rerank_top_n_from_env,
)
from src.retrieval.v5_compat import (
    build_v5_shadow_trace,
    compare_v4_to_v5_shadow,
    retrieval_pipeline_selection_from_env,
)
from src.retrieval.v5_contracts import RetrievalPipelineVersion, RetrievalStageV5
from src.retrieval.v5_signals import build_entity_signals, entity_graph_seed_names
from src.quality.safe_fallback import sanitize_fallback_reason
from src.resilience.contracts import runtime_resilience_settings_from_env

# Phase 1.5 — Dermatology-aware query boost
from src.ingestion.domain_metadata import extract_dermatology_metadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1.5 — Query-Adaptive Metadata Boost
# ---------------------------------------------------------------------------

# Per-field boost weights
_FIELD_BOOST_WEIGHTS: dict[str, float] = {
    "ingredient": 0.15,
    "concern": 0.10,
    "content_type": 0.10,
    "domain_topic": 0.08,
    "skin_type": 0.08,
    "safety_context": 0.10,
    "body_area": 0.05,
}

# Maximum total boost to prevent over-ranking
_MAX_METADATA_BOOST: float = 0.30

# Boosted metadata field names expected in Qdrant payload
_BOOST_FIELDS: list[str] = list(_FIELD_BOOST_WEIGHTS.keys())

_REFERENCE_SECTION_MARKERS = (
    "references",
    "reference",
    "bibliography",
    "tài liệu tham khảo",
    "tham khảo",
)


def _is_reference_context(result: dict[str, Any]) -> bool:
    header = str(
        result.get("header")
        or result.get("parent_header_path")
        or result.get("section")
        or ""
    ).lower()
    content_type = result.get("content_type", [])
    if isinstance(content_type, str):
        content_type = [content_type]
    content_type_text = " ".join(str(item).lower() for item in content_type)
    return any(marker in header or marker in content_type_text for marker in _REFERENCE_SECTION_MARKERS)


def prioritize_main_contexts(results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Prefer guideline/body contexts over references for answer generation."""
    main_contexts: list[dict[str, Any]] = []
    reference_contexts: list[dict[str, Any]] = []

    for result in results:
        entry = dict(result)
        if _is_reference_context(entry):
            entry["context_role"] = "reference"
            entry["score"] = round(float(entry.get("score", 0.0)) - 0.05, 6)
            reference_contexts.append(entry)
        else:
            entry["context_role"] = "main"
            main_contexts.append(entry)

    selected = main_contexts[:top_k]
    if len(selected) < top_k:
        selected.extend(reference_contexts[: top_k - len(selected)])
    return selected


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _rerank_timeout_within_retrieval_budget() -> float:
    settings = runtime_resilience_settings_from_env()
    rerank_timeout = float(settings.rerank_timeout_seconds)
    retrieval_timeout = float(settings.retrieval_timeout_seconds)
    if retrieval_timeout <= 1:
        return max(0.1, min(rerank_timeout, retrieval_timeout))
    # Reranker must time out before the enclosing retrieval stage so fallback
    # can preserve a 200 response instead of leaking a retrieval-level 504.
    budgeted_timeout = max(1.0, retrieval_timeout * 0.5)
    return max(0.1, min(rerank_timeout, budgeted_timeout))


def extract_query_dermatology_metadata(query: str) -> dict[str, Any]:
    """Extract dermatology metadata from a user query string.

    Reuses the same rule-based extractor used during ingestion so
    that query-side and document-side taxonomies are consistent.
    """
    return extract_dermatology_metadata(text=query, header_path="")


def apply_metadata_boost(
    results: list[dict[str, Any]],
    query_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply soft metadata boost to retrieval results.

    For each result, check if its payload fields overlap with the
    query metadata.  Each matching field adds a weight-based bonus
    to the score, capped at ``_MAX_METADATA_BOOST``.

    The function is **backward-compatible**:
    - If a result has no payload or no dermatology fields, it passes
      through with ``metadata_boost_applied = False``.
    - If ``score`` is None, it defaults to 0.0.
    - Original score is always preserved as ``original_score``.

    Parameters
    ----------
    results : list[dict]
        Search results (each dict has at minimum ``"score"`` and
        optional dermatology payload fields).
    query_metadata : dict
        Output of :func:`extract_query_dermatology_metadata`.

    Returns
    -------
    list[dict]
        Same results list, sorted descending by ``boosted_score``.
    """
    boosted: list[dict[str, Any]] = []

    for result in results:
        original_score = result.get("score")
        if original_score is None:
            original_score = 0.0
        original_score = float(original_score)

        total_boost = 0.0
        matched_fields: list[str] = []

        for field_name in _BOOST_FIELDS:
            query_values = query_metadata.get(field_name, [])
            if not query_values:
                continue

            doc_values = result.get(field_name, [])
            if not isinstance(doc_values, list):
                continue
            if not doc_values:
                continue

            # Check for any overlap between query and document values
            if set(query_values) & set(doc_values):
                total_boost += _FIELD_BOOST_WEIGHTS[field_name]
                matched_fields.append(field_name)

        # Cap total boost
        total_boost = min(total_boost, _MAX_METADATA_BOOST)

        boosted_score = original_score + total_boost
        boost_applied = total_boost > 0

        # Attach boost debug info
        entry = dict(result)
        entry["original_score"] = round(original_score, 6)
        entry["metadata_boost"] = round(total_boost, 6)
        entry["boosted_score"] = round(boosted_score, 6)
        entry["score"] = round(boosted_score, 6)
        entry["metadata_boost_applied"] = boost_applied
        entry["matched_metadata_fields"] = matched_fields

        boosted.append(entry)

    # Re-sort by boosted score descending
    boosted.sort(key=lambda x: x["boosted_score"], reverse=True)

    return boosted


def annotate_metadata_matches(
    results: list[dict[str, Any]],
    query_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach query metadata matches while preserving knowledge RRF ranking.

    This is the V5 R3 path. It deliberately does not calculate or apply an
    additive metadata score; metadata remains a feature for later stages.
    """

    annotated: list[dict[str, Any]] = []
    for result in results:
        entry = dict(result)
        rrf_score = float(entry.get("rrf_score", entry.get("score", 0.0)) or 0.0)
        matched_fields: list[str] = []
        for field_name in _BOOST_FIELDS:
            query_values = query_metadata.get(field_name, [])
            doc_values = entry.get(field_name, [])
            if not query_values or not isinstance(doc_values, list):
                continue
            if set(query_values) & set(doc_values):
                matched_fields.append(field_name)

        entry["score"] = round(rrf_score, 6)
        entry["original_score"] = round(rrf_score, 6)
        entry["matched_metadata_fields"] = matched_fields
        entry["metadata_annotation_only"] = True
        annotated.append(entry)

    annotated.sort(
        key=lambda item: (
            -float(item.get("rrf_score", item.get("score", 0.0)) or 0.0),
            _rank_or_max(item.get("dense_rank")),
            _rank_or_max(item.get("sparse_rank")),
            str(item.get("id") or item.get("chunk_id") or ""),
        )
    )
    return annotated


def _rank_or_max(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 10**9
    return parsed if parsed > 0 else 10**9


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Structured output from the hybrid retrieval pipeline."""

    vector_contexts: list[dict[str, Any]]
    """Ranked chunks from Qdrant (text, score, metadata)."""

    graph_facts: list[dict[str, Any]]
    """Related entities and relationships from Neo4j."""

    sources: list[str]
    """Unique source file names referenced by retrieved chunks."""

    query: str
    """Original user query."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Timing and scoring breakdown for observability."""


def _sources_for_contexts(contexts: list[dict[str, Any]]) -> list[str]:
    """Return every distinct document source used in the packed prompt context."""

    return list(
        dict.fromkeys(
            str(context.get("source_file"))
            for context in contexts
            if context.get("source_file") and context.get("retrieval_source") != "entity"
        )
    )


async def _get_graph_facts_with_fallback(
    graph_store: Neo4jGraphStore,
    entity_names: set[str],
    query: str,
) -> list[dict[str, Any]]:
    """Resolve graph facts, using keywords when optional semantic hints are empty."""

    if entity_names:
        return await graph_store.get_entity_context(list(entity_names))
    logger.info("No graph_nodes in payloads, falling back to keyword search.")
    return await graph_store.search_by_keywords(query.lower().split())


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Hybrid retrieval combining dense semantic search, sparse BM25 lexical
    search, RRF fusion, and Neo4j knowledge graph enrichment.

    Usage
    -----
    retriever = HybridRetriever()
    result = await retriever.retrieve("isotretinoin side effects", top_k=5)
    print(result.vector_contexts)
    print(result.graph_facts)
    await retriever.close()
    """

    def __init__(self) -> None:
        self._vector_store = QdrantVectorStore()
        self._graph_store = Neo4jGraphStore()
        self._entity_retriever = EntityRetriever()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        rrf_k: int = 60,
        diagnostic_expected_concepts: list[str] | None = None,
        diagnostic_original_query: str | None = None,
        diagnostic_concept_matcher: ConceptMatcher | None = None,
        diagnostic_critical: bool = False,
    ) -> RetrievalResult:
        """Run the full hybrid retrieval pipeline.

        Parameters
        ----------
        query : str
            User question or search query.
        top_k : int
            Number of final chunks to return after fusion.
        dense_weight : float
            Weight for dense search in RRF fusion.
        sparse_weight : float
            Weight for sparse BM25 search in RRF fusion.
        rrf_k : int
            RRF constant (default 60, higher = more conservative fusion).
        """
        t_start = time.time()
        warnings: list[str] = []
        pipeline_selection = retrieval_pipeline_selection_from_env()

        # ── Phase 2A: normalize + taxonomy expansion ────────────────
        t_norm_start = time.time()
        normalized_query = normalize_query(query)
        expansion = expand_normalized_query(normalized_query)
        t_norm = time.time() - t_norm_start

        # ── Step 1: Embed query + resolve entity cards ───────────────
        # Entity-card lookup only needs normalized query metadata, so it can
        # overlap the remote embedding request without changing retrieval
        # semantics.
        t_embed_start = time.time()
        t_entity_start = time.time()
        embedding_task = asyncio.create_task(embed_query(query))
        entity_task = asyncio.create_task(
            self._entity_retriever.retrieve(
                normalized_query=normalized_query,
                expansion=expansion,
                limit=8,
            )
        )
        try:
            query_vector = await embedding_task
        except BaseException:
            # The entity lookup overlaps the embedding call.  Do not leave it
            # running after an embedding failure because its result can no
            # longer contribute to this request.
            entity_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await entity_task
            raise
        t_embed = time.time() - t_embed_start
        logger.info("Embedded query in %.2fs (dim=%d)", t_embed, len(query_vector))

        entity_candidates = []
        try:
            entity_candidates = await entity_task
        except Exception as exc:
            warning = f"Entity retrieval skipped: {sanitize_fallback_reason(exc)}"
            warnings.append(warning)
            logger.warning(warning)
        t_entity = time.time() - t_entity_start

        # ── Step 2: Dense search ─────────────────────────────────────
        fetch_k = max(top_k * 3, 15)  # Fetch more candidates for fusion

        # ── Step 2 + 3: Dense and sparse search ─────────────────────
        # Both queries are independent once the dense embedding is available.
        t_dense_start = time.time()
        t_sparse_start = t_dense_start
        sparse_query = " ".join(expansion.expanded_terms) or query
        dense_results, sparse_results = await asyncio.gather(
            self._vector_store.search(query_vector=query_vector, top_k=fetch_k),
            self._vector_store.search_sparse(text=sparse_query, top_k=fetch_k),
        )
        t_dense = time.time() - t_dense_start
        t_sparse = time.time() - t_sparse_start
        logger.info("Dense search: %d results in %.2fs", len(dense_results), t_dense)
        logger.info("Sparse search: %d results in %.2fs", len(sparse_results), t_sparse)

        # ── Step 4: RRF Fusion ───────────────────────────────────────
        fused = self._rrf_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            k=rrf_k,
        )

        logger.info(
            "RRF fusion: %d dense + %d sparse → %d fused",
            len(dense_results),
            len(sparse_results),
            len(fused),
        )

        # ── Step 4.5: Legacy boost or V5 metadata annotation ────────
        t_boost_start = time.time()
        query_metadata = extract_query_dermatology_metadata(query)
        metadata_mode = "legacy_additive"
        if pipeline_selection.execution == RetrievalPipelineVersion.V5:
            metadata_mode = "annotation_first"
            annotated = annotate_metadata_matches(fused, query_metadata)
            chunk_candidates = annotate_chunk_results(
                annotated,
                normalized_query=normalized_query,
                collection_name=self._vector_store._collection,
            )
        else:
            boosted = apply_metadata_boost(fused, query_metadata)
            chunk_candidates = boost_chunk_results(
                boosted,
                normalized_query=normalized_query,
                collection_name=self._vector_store._collection,
            )
        t_boost = time.time() - t_boost_start

        metadata_match_count = sum(
            1
            for candidate in chunk_candidates
            if candidate.matched_metadata
        )
        logger.info(
            "Metadata %s: %d/%d candidates annotated in %.4fs",
            metadata_mode,
            metadata_match_count,
            len(chunk_candidates),
            t_boost,
        )

        # ── Step 4.6: Candidate merge ────────────────────────────────
        merged_candidates = merge_candidates(
            entity_candidates=entity_candidates,
            chunk_candidates=chunk_candidates,
            normalized_query=normalized_query,
            limit=max(top_k * 2, 8),
        )
        source_evidence_candidates = [
            candidate for candidate in chunk_candidates if candidate.source == "chunk"
        ]
        reranker_input_candidates = merged_candidates
        candidate_policy_result: CandidatePolicyResult | None = None
        t_candidate_policy = 0.0
        if pipeline_selection.execution == RetrievalPipelineVersion.V5:
            t_policy_start = time.time()
            candidate_policy_result = apply_candidate_policy(
                source_evidence_candidates,
                normalized_query,
                budget=candidate_policy_budget(top_k),
            )
            reranker_input_candidates = list(candidate_policy_result.candidates)
            t_candidate_policy = time.time() - t_policy_start
        t_rerank_start = time.time()
        rerank_top_n = rerank_top_n_from_env(default=max(top_k * 2, 8))
        rerank_timeout_seconds = _rerank_timeout_within_retrieval_budget()
        if rerank_enabled_from_env():
            try:
                reranked_candidates, rerank_trace = await asyncio.wait_for(
                    asyncio.to_thread(
                        rerank_candidates,
                        normalized_query=normalized_query,
                        candidates=reranker_input_candidates,
                        expansion=expansion,
                        top_n=rerank_top_n,
                        provider=rerank_provider_from_env(),
                        timeout_seconds=rerank_timeout_seconds,
                    ),
                    timeout=rerank_timeout_seconds,
                )
            except asyncio.TimeoutError:
                warning = "Reranker timed out; using merged candidate order."
                warnings.append(warning)
                logger.warning(warning)
                reranked_candidates = reranker_input_candidates
                rerank_trace = RerankTrace(
                    provider=rerank_provider_from_env(),
                    enabled=False,
                    input_count=len(reranker_input_candidates),
                    output_count=len(reranker_input_candidates),
                    top_n=rerank_top_n,
                    ranked_candidates=[],
                    warnings=[warning],
                    timings_ms={},
                )
        else:
            reranked_candidates = reranker_input_candidates
            rerank_trace = RerankTrace(
                provider=rerank_provider_from_env(),
                enabled=False,
                input_count=len(reranker_input_candidates),
                output_count=len(reranker_input_candidates),
                top_n=rerank_top_n,
                ranked_candidates=[],
                warnings=["Reranker disabled by RERANK_ENABLED=false."],
                timings_ms={},
            )
        t_rerank = time.time() - t_rerank_start
        warnings.extend(rerank_trace.warnings)

        t_pack_start = time.time()
        context_max_items = _env_int("RETRIEVAL_CONTEXT_MAX_ITEMS", 5, minimum=3, maximum=10)
        context_max_chars = _env_int("RETRIEVAL_CONTEXT_MAX_CHARS", 4200, minimum=1200, maximum=12000)
        packed_context = pack_context(
            normalized_query=normalized_query,
            merged_candidates=reranked_candidates,
            max_items=min(max(top_k, 5), context_max_items),
            max_chars=context_max_chars,
        )
        t_pack = time.time() - t_pack_start
        selected_candidates = _candidates_for_packed_items(reranked_candidates, packed_context)
        top_chunks = prioritize_main_contexts(packed_context_to_legacy_contexts(packed_context), top_k)

        # ── Step 5: Extract graph_nodes from Qdrant payloads ────────
        entity_names: set[str] = set()
        for chunk in top_chunks:
            if chunk.get("retrieval_source") == "entity" and chunk.get("canonical_name"):
                entity_names.add(str(chunk["canonical_name"]))
            graph_nodes = chunk.get("graph_nodes", [])
            if isinstance(graph_nodes, list):
                entity_names.update(
                    n for n in graph_nodes if isinstance(n, str) and n.strip()
                )
        if pipeline_selection.execution == RetrievalPipelineVersion.V5:
            entity_names.update(entity_graph_seed_names(build_entity_signals(entity_candidates)))

        # ── Step 6: Neo4j knowledge graph context ────────────────────
        t_neo4j_start = time.time()

        neo4j_timeout_seconds = float(os.getenv("NEO4J_TIMEOUT_SECONDS", "10"))
        try:
            async with asyncio.timeout(neo4j_timeout_seconds):
                graph_facts = await _get_graph_facts_with_fallback(
                    self._graph_store,
                    entity_names,
                    query,
                )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            warning = f"Neo4j context skipped after {neo4j_timeout_seconds:.1f}s timeout."
            warnings.append(warning)
            logger.warning(warning)
            graph_facts = []

        t_neo4j = time.time() - t_neo4j_start
        logger.info(
            "Neo4j context: %d entity names → %d facts in %.2fs",
            len(entity_names),
            len(graph_facts),
            t_neo4j,
        )

        # ── Step 7: Build sources list ───────────────────────────────
        sources = _sources_for_contexts(top_chunks)

        t_total = time.time() - t_start
        trace = RetrievalTrace(
            original_query=query,
            normalized_query=normalized_query,
            expansion=expansion,
            entity_candidates=entity_candidates[:8],
            chunk_candidates=chunk_candidates[:8],
            merged_candidates=reranker_input_candidates[:8],
            selected_context=selected_candidates,
            packed_context=packed_context,
            rerank_trace=rerank_trace,
            warnings=warnings,
            timings_ms={
                "normalize_expand": round(t_norm * 1000, 3),
                "embed": round(t_embed * 1000, 3),
                "dense": round(t_dense * 1000, 3),
                "sparse": round(t_sparse * 1000, 3),
                "boost": round(t_boost * 1000, 3),
                "entity": round(t_entity * 1000, 3),
                "candidate_policy": round(t_candidate_policy * 1000, 3),
                "rerank": round(t_rerank * 1000, 3),
                "pack": round(t_pack * 1000, 3),
                "neo4j": round(t_neo4j * 1000, 3),
                "total": round(t_total * 1000, 3),
            },
        )
        retrieval_diagnostics = None
        if diagnostic_expected_concepts and diagnostic_concept_matcher is not None:
            try:
                retrieval_diagnostics = build_retrieval_diagnostic_trace(
                    expected_concepts=diagnostic_expected_concepts,
                    critical=diagnostic_critical,
                    original_query=diagnostic_original_query or query,
                    retrieval_query=query,
                    normalized_query=normalized_query,
                    expansion=expansion,
                    dense_results=dense_results,
                    sparse_results=sparse_results,
                    entity_candidates=entity_candidates,
                    fused_results=fused,
                    chunk_candidates=chunk_candidates,
                    merged_candidates=reranker_input_candidates,
                    reranked_candidates=reranked_candidates,
                    packed_context=packed_context,
                    rerank_trace=rerank_trace,
                    context_max_items=context_max_items,
                    context_max_chars=context_max_chars,
                    warnings=warnings,
                    concept_matcher=diagnostic_concept_matcher,
                ).model_dump(mode="json")
            except Exception as exc:
                warning = f"Evaluation retrieval diagnostics unavailable: {sanitize_fallback_reason(exc)}"
                warnings.append(warning)
                logger.warning(warning)

        # V5 remains opt-in. Shadow mode preserves V4 output for equivalence
        # checks, while explicit V5 uses source-backed chunks as evidence and
        # retains entity/graph data as dedicated structural side signals.
        v5_shadow_payload = None
        if pipeline_selection.shadow_enabled:
            try:
                v5_trace = build_v5_shadow_trace(
                    original_query=query,
                    retrieval_query=query,
                    normalized_query=normalized_query,
                    expansion=expansion,
                    dense_results=dense_results,
                    sparse_results=sparse_results,
                    fused_results=fused,
                    entity_candidates=entity_candidates,
                    chunk_candidates=chunk_candidates,
                    merged_candidates=merged_candidates,
                    reranked_candidates=reranked_candidates,
                    rerank_trace=rerank_trace,
                    packed_context=packed_context,
                    graph_facts=graph_facts,
                    timings_ms=trace.timings_ms,
                    selection=pipeline_selection,
                    candidate_policy_candidates=(
                        list(candidate_policy_result.candidates)
                        if candidate_policy_result is not None
                        else None
                    ),
                    candidate_policy_drops=(
                        candidate_policy_result.drops
                        if candidate_policy_result is not None
                        else ()
                    ),
                )
                comparison = None
                if pipeline_selection.execution == RetrievalPipelineVersion.V4:
                    comparison = compare_v4_to_v5_shadow(
                        trace=v5_trace,
                        dense_results=dense_results,
                        sparse_results=sparse_results,
                        fused_results=fused,
                        merged_candidates=merged_candidates,
                        reranked_candidates=reranked_candidates,
                        packed_context=packed_context,
                    )
                    v5_trace = v5_trace.model_copy(update={"shadow_comparison": comparison})
                v5_shadow_payload = {
                    "requested_pipeline": pipeline_selection.requested.value,
                    "execution_pipeline": pipeline_selection.execution.value,
                    "shadow_equivalent": comparison.equivalent if comparison else None,
                    "comparison": comparison.model_dump(mode="json") if comparison else None,
                    "entity_signals": [
                        signal.model_dump(mode="json")
                        for signal in v5_trace.entity_signals
                    ],
                    "graph_seed_names": list(v5_trace.graph_seed_names),
                    "graph_signals": [
                        signal.model_dump(mode="json")
                        for signal in v5_trace.graph_signals
                    ],
                    "source_evidence_candidate_ids": list(
                        v5_trace.candidate_ids_for(RetrievalStageV5.SOURCE_EVIDENCE_POOL)
                    ),
                    "trace": v5_trace.model_dump(mode="json"),
                }
            except Exception as exc:
                logger.warning("V5 shadow trace unavailable: %s", sanitize_fallback_reason(exc))

        result_metadata = {
                "total_time_s": round(t_total, 3),
                "normalize_expand_time_s": round(t_norm, 3),
                "embed_time_s": round(t_embed, 3),
                "dense_time_s": round(t_dense, 3),
                "sparse_time_s": round(t_sparse, 3),
                "boost_time_s": round(t_boost, 3),
                "entity_time_s": round(t_entity, 3),
                "candidate_policy_time_s": round(t_candidate_policy, 3),
                "rerank_time_s": round(t_rerank, 3),
                "context_pack_time_s": round(t_pack, 3),
                "neo4j_time_s": round(t_neo4j, 3),
                "dense_count": len(dense_results),
                "sparse_count": len(sparse_results),
                "fused_count": len(fused),
                "boosted_count": metadata_match_count if metadata_mode == "legacy_additive" else 0,
                "metadata_annotation_count": metadata_match_count,
                "metadata_mode": metadata_mode,
                "candidate_policy": (
                    candidate_policy_result.debug_summary()
                    if candidate_policy_result is not None
                    else {"mode": "legacy_candidate_merge"}
                ),
                "entity_count": len(entity_candidates),
                    "merged_count": len(reranker_input_candidates),
                "reranked_count": len(reranked_candidates),
                "top_k": top_k,
                "entity_names_count": len(entity_names),
                "graph_facts_count": len(graph_facts),
                "query_metadata": query_metadata,
                "phase2a": {
                    "intent": normalized_query.intent,
                    "normalized_entities": {
                        "drug_product": normalized_query.drug_product,
                        "active_ingredient": normalized_query.active_ingredient,
                        "drug_class": normalized_query.drug_class,
                        "condition": normalized_query.condition,
                        "safety_context": normalized_query.safety_context,
                    },
                    "expanded_terms": expansion.expanded_terms,
                    "top_entity_candidates": [
                        candidate_debug_summary(candidate)
                        for candidate in entity_candidates[:5]
                    ],
                    "top_chunk_candidates": [
                        candidate_debug_summary(candidate)
                        for candidate in chunk_candidates[:5]
                    ],
                    "merged_candidates": [
                        candidate_debug_summary(candidate)
                        for candidate in reranker_input_candidates[:5]
                    ],
                    "rerank": {
                        "enabled": rerank_trace.enabled,
                        "provider": rerank_trace.provider,
                        "input_count": rerank_trace.input_count,
                        "output_count": rerank_trace.output_count,
                        "top_n": rerank_trace.top_n,
                        "ranked_candidates": [
                            {
                                "rank": item.rerank_rank,
                                "candidate_id": item.candidate.candidate_id,
                                "source": item.candidate.source,
                                "canonical_name": item.candidate.payload.get("canonical_name"),
                                "chunk_id": item.candidate.payload.get("chunk_id"),
                                "rerank_score": item.rerank_score,
                                "reasons": item.score_breakdown.reasons,
                            }
                            for item in rerank_trace.ranked_candidates[:5]
                        ],
                        "warnings": rerank_trace.warnings,
                    },
                    "packed_context": {
                        "entity_items_count": packed_context.entity_items_count,
                        "chunk_items_count": packed_context.chunk_items_count,
                        "selected_entity_ids": packed_context.debug.get("pack_trace", {}).get("selected_entity_ids", []),
                        "selected_chunk_ids": packed_context.debug.get("pack_trace", {}).get("selected_chunk_ids", []),
                        "selection_reasons": packed_context.debug.get("pack_trace", {}).get("selection_reasons", []),
                        "warnings": packed_context.warnings,
                    },
                    "warnings": warnings,
                },
                "retrieval_trace": trace.model_dump(mode="json"),
                "rerank_trace": rerank_trace.model_dump(mode="json"),
                "packed_context": packed_context.model_dump(mode="json"),
            "retrieval_diagnostics": retrieval_diagnostics,
        }
        if v5_shadow_payload is not None:
            result_metadata["retrieval_v5"] = v5_shadow_payload

        return RetrievalResult(
            vector_contexts=top_chunks,
            graph_facts=graph_facts,
            sources=sources,
            query=query,
            metadata=result_metadata,
        )
    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fusion(
        dense_results: list[dict],
        sparse_results: list[dict],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion to merge dense and sparse search results.

        Formula: ``score(doc) = Σ weight / (k + rank)``
        where rank is 1-indexed.

        Uses Qdrant point ID as the document key for deduplication.
        """
        doc_scores: dict[str, float] = {}
        doc_data: dict[str, dict] = {}

        # Score dense results
        for rank, doc in enumerate(dense_results, start=1):
            doc_id = str(doc.get("id", ""))
            if not doc_id:
                continue
            rrf_score = dense_weight / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = doc.copy()
            doc_data[doc_id]["dense_rank"] = rank
            doc_data[doc_id]["dense_score"] = doc.get("score", 0.0)

        # Score sparse results
        for rank, doc in enumerate(sparse_results, start=1):
            doc_id = str(doc.get("id", ""))
            if not doc_id:
                continue
            rrf_score = sparse_weight / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = doc.copy()
            doc_data[doc_id]["sparse_rank"] = rank
            doc_data[doc_id]["sparse_score"] = doc.get("score", 0.0)

        # Sort by fused RRF score descending
        ranked_ids = sorted(
            doc_scores,
            key=lambda did: doc_scores[did],
            reverse=True,
        )

        result: list[dict] = []
        for doc_id in ranked_ids:
            doc = doc_data[doc_id]
            doc["rrf_score"] = round(doc_scores[doc_id], 6)
            doc["score"] = doc["rrf_score"]  # Override score with RRF
            result.append(doc)

        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close all backend connections."""
        try:
            await self._vector_store.close()
        except Exception as exc:
            logger.warning("Error closing vector store: %s", sanitize_fallback_reason(exc))
        try:
            await self._graph_store.close()
        except Exception as exc:
            logger.warning("Error closing graph store: %s", sanitize_fallback_reason(exc))
        try:
            await self._entity_retriever.close()
        except Exception as exc:
            logger.warning("Error closing entity retriever: %s", sanitize_fallback_reason(exc))


def _candidates_for_packed_items(merged_candidates: list[Any], packed_context: Any) -> list[Any]:
    by_id = {candidate.candidate_id: candidate for candidate in merged_candidates}
    selected = []
    for item in packed_context.items:
        candidate = by_id.get(item.item_id)
        if candidate is not None:
            selected.append(candidate)
    return selected
