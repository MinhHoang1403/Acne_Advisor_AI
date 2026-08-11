"""Entity and graph side-channel adapters for the staged Retrieval V5 path."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from src.retrieval.contracts import RetrievedCandidate
from src.retrieval.v5_contracts import EntitySignalV5, GraphSignalV5


def build_entity_signals(
    candidates: Iterable[RetrievedCandidate],
) -> tuple[EntitySignalV5, ...]:
    """Convert legacy entity candidates into structural side-channel signals.

    Entity lexical confidence remains inside `EntitySignalV5`; it is never used
    as a source-evidence score or compared to a chunk RRF score.
    """

    signals: list[EntitySignalV5] = []
    for candidate in candidates:
        if candidate.source != "entity":
            continue
        payload = candidate.payload
        entity_id = _text(payload.get("entity_id")) or candidate.candidate_id
        canonical_name = _text(payload.get("canonical_name")) or candidate.candidate_id
        entity_type = _text(payload.get("entity_type")) or "unknown"
        matched_terms = tuple(
            _stable_strings(
                value
                for values in candidate.matched_metadata.values()
                for value in _as_list(values)
            )
        )
        safety_annotations = tuple(
            _stable_strings(
                [
                    *_as_list(payload.get("safety_contexts")),
                    *_as_list(payload.get("contraindications")),
                ]
            )
        )
        confidence = _finite_score(candidate.score)
        signals.append(
            EntitySignalV5(
                entity_id=entity_id,
                canonical_name=canonical_name,
                entity_type=entity_type,
                matched_terms=matched_terms,
                match_confidence=confidence,
                graph_seed_ids=tuple(_stable_strings([entity_id, canonical_name])),
                safety_annotations=safety_annotations,
            )
        )
    return tuple(_dedupe_entity_signals(signals))


def build_graph_signals(
    graph_facts: Iterable[Mapping[str, Any]],
    entity_signals: Iterable[EntitySignalV5],
) -> tuple[GraphSignalV5, ...]:
    """Convert read-only Neo4j facts into non-grounding relational signals.

    Current graph facts have graph/taxonomy provenance but do not expose linked
    knowledge chunk IDs. They are therefore explicitly not medical-claim
    eligible; R6 can only promote a graph assertion with validated chunk
    provenance.
    """

    fallback_entity_ids = {
        signal.canonical_name.casefold(): signal.entity_id
        for signal in entity_signals
    }
    signals: list[GraphSignalV5] = []
    for index, fact in enumerate(graph_facts):
        subject = _text(fact.get("subject") or fact.get("entity"))
        relation = _text(fact.get("predicate") or fact.get("relationship"))
        target = _text(fact.get("object") or fact.get("related_entity"))
        if not subject:
            continue
        source_entity_id = (
            _text(fact.get("subject_graph_node_id"))
            or fallback_entity_ids.get(subject.casefold())
            or subject
        )
        target_entity_id = _text(fact.get("object_graph_node_id")) or target or None
        path = tuple(value for value in (relation, target) if value)
        confidence = _finite_optional_score(fact.get("confidence"))
        signals.append(
            GraphSignalV5(
                signal_id=f"graph:{source_entity_id}:{relation or 'isolated'}:{target or index}",
                source_entity_id=source_entity_id,
                relation_path=path,
                target_entity_id=target_entity_id,
                path_confidence=confidence,
                source_chunk_ids=(),
                medical_claim_eligible=False,
            )
        )
    return tuple(_dedupe_graph_signals(signals))


def entity_graph_seed_names(entity_signals: Iterable[EntitySignalV5]) -> tuple[str, ...]:
    """Return deterministic canonical names to seed graph lookup in later V5 stages."""

    return tuple(_stable_strings(signal.canonical_name for signal in entity_signals))


def _dedupe_entity_signals(signals: Iterable[EntitySignalV5]) -> list[EntitySignalV5]:
    deduped: dict[str, EntitySignalV5] = {}
    for signal in signals:
        existing = deduped.get(signal.entity_id)
        if existing is None or signal.match_confidence > existing.match_confidence:
            deduped[signal.entity_id] = signal
    return sorted(
        deduped.values(),
        key=lambda signal: (signal.canonical_name.casefold(), signal.entity_id),
    )


def _dedupe_graph_signals(signals: Iterable[GraphSignalV5]) -> list[GraphSignalV5]:
    deduped = {signal.signal_id: signal for signal in signals}
    return [deduped[signal_id] for signal_id in sorted(deduped)]


def _finite_score(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return value


def _finite_optional_score(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)] if value else []


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _stable_strings(values: Iterable[Any]) -> list[str]:
    return sorted({_text(value) for value in values if _text(value)}, key=str.casefold)


__all__ = ["build_entity_signals", "build_graph_signals", "entity_graph_seed_names"]
