"""Coverage-oriented V5 evidence selection, deliberately separate from packing."""

from __future__ import annotations

from collections.abc import Iterable

from src.retrieval.v5_contracts import (
    EntitySignalV5,
    EvidenceSelectionRequirementsV5,
    EvidenceSelectionResultV5,
    EvidenceSufficiencyV5,
    GraphSignalV5,
    QueryContextV5,
    RankedEvidenceV5,
    SelectedEvidenceV5,
)


_SAFETY_FEATURES = {"safety_context", "safety_contexts", "contraindications"}
_TREATMENT_INTENTS = {"treatment", "treatment_recommendation", "general_acne_question"}
_CONTRADICTION_FEATURES = {"contraindication", "contraindications", "not_antibiotic"}
_GRAPH_RELATION_ROLES = {
    "BELONGS_TO_CLASS": "drug_class",
    "HAS_ACTIVE_INGREDIENT": "ingredient",
    "CONTRAINDICATED_IN": "safety",
}


def select_evidence_v5(
    *,
    query_context: QueryContextV5,
    ranked_evidence: Iterable[RankedEvidenceV5],
    entity_signals: Iterable[EntitySignalV5] = (),
    graph_signals: Iterable[GraphSignalV5] = (),
    requirements: EvidenceSelectionRequirementsV5 | None = None,
) -> EvidenceSelectionResultV5:
    """Classify source-backed evidence without reranking or applying a budget.

    R6 deliberately preserves the full R5 order. Entity and graph signals can
    inform required coverage, but neither can satisfy a medical-evidence role
    because only `RankedEvidenceV5` carries chunk provenance.
    """

    ranked = tuple(ranked_evidence)
    entity_signals = tuple(entity_signals)
    graph_signals = tuple(graph_signals)
    requirements = requirements or _requirements_from_query(query_context, graph_signals)
    selected = tuple(
        _selected_evidence(
            evidence=evidence,
            query_context=query_context,
            requirements=requirements,
            is_primary=index == 1,
        )
        for index, evidence in enumerate(ranked, start=1)
    )
    covered_roles = {
        role
        for selected_item in selected
        for role in selected_item.roles
    }
    missing_roles = tuple(
        role for role in requirements.required_roles if role not in covered_roles
    )
    critical_required = bool(requirements.critical_safety_flags)
    critical_covered = any(item.critical for item in selected)
    if critical_required and not critical_covered:
        status = EvidenceSufficiencyV5.CRITICAL_EVIDENCE_MISSING
    elif missing_roles:
        status = EvidenceSufficiencyV5.INSUFFICIENT
    else:
        status = EvidenceSufficiencyV5.SUFFICIENT
    return EvidenceSelectionResultV5(
        selected_evidence=selected,
        status=status,
        missing_roles=missing_roles,
        satisfied_roles=tuple(sorted(covered_roles)),
        requirements=requirements,
        entity_signal_count=len(entity_signals),
        graph_signal_count=len(graph_signals),
    )


def _requirements_from_query(
    query_context: QueryContextV5,
    graph_signals: tuple[GraphSignalV5, ...],
) -> EvidenceSelectionRequirementsV5:
    required_roles = ["primary", "source_traceability"]
    if query_context.safety_flags:
        required_roles.append("safety")
    graph_required_roles = tuple(
        dict.fromkeys(
            role
            for signal in graph_signals
            for role in (_graph_requirement_role(signal),)
            if role
        )
    )
    required_roles.extend(role for role in graph_required_roles if role not in required_roles)
    return EvidenceSelectionRequirementsV5(
        required_roles=tuple(required_roles),
        critical_safety_flags=query_context.safety_flags,
        graph_required_roles=graph_required_roles,
    )


def _selected_evidence(
    *,
    evidence: RankedEvidenceV5,
    query_context: QueryContextV5,
    requirements: EvidenceSelectionRequirementsV5,
    is_primary: bool,
) -> SelectedEvidenceV5:
    metadata_features = set(evidence.candidate.candidate.metadata_features)
    roles: list[str] = []
    if is_primary:
        roles.append("primary")
    if _has_source_provenance(evidence):
        roles.append("source_traceability")
    safety_evidence = bool(metadata_features & _SAFETY_FEATURES)
    if safety_evidence:
        roles.append("safety")
    if query_context.intent in _TREATMENT_INTENTS:
        roles.append("treatment")
    if metadata_features & _CONTRADICTION_FEATURES:
        roles.append("contradiction")
    if "drug_class" in metadata_features:
        roles.append("drug_class")
    if metadata_features & {"active_ingredient", "active_ingredients", "ingredient"}:
        roles.append("ingredient")
    critical = bool(requirements.critical_safety_flags) and safety_evidence
    if critical:
        roles.append("critical")
    return SelectedEvidenceV5(
        evidence=evidence,
        roles=tuple(roles),
        selection_reason="r6_pass_through_coverage_classification",
        critical=critical,
    )


def _has_source_provenance(evidence: RankedEvidenceV5) -> bool:
    provenance = evidence.candidate.candidate.provenance
    return bool(provenance.chunk_id and (provenance.source_path or provenance.document_id))


def _graph_requirement_role(signal: GraphSignalV5) -> str | None:
    relation = signal.relation_path[0].upper() if signal.relation_path else ""
    return _GRAPH_RELATION_ROLES.get(relation)


__all__ = ["select_evidence_v5"]
