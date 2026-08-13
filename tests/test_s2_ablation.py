from scripts.evaluate_s2_controlled_ablation import (
    _complexity_inventory,
    _write_machine_outputs,
    _write_markdown_outputs,
)
from src.evaluation.s2_ablation import (
    COMPONENT_STATUSES,
    run_candidate_policy_ablation,
    run_entity_ablation,
    run_locked_component_diagnostics,
    run_reranker_ablation,
    run_s2_ablation,
    run_sufficiency_retry_ablation,
)


def test_s2_reranker_is_diagnostic_and_exposes_exact_metric_denominators():
    result = run_reranker_ablation(provider="local_rules")

    assert result["case_count"] == 12
    assert result["gold_grade"] == "NOT_DECISION_GRADE_GOLD"
    assert result["baseline"]["recall@1"]["denominator"] == 12
    assert result["variant"]["mrr"]["denominator"] == 12
    assert result["requested_provider"] == "local_rules"


def test_current_candidate_policy_fixture_does_not_exercise_budget():
    result = run_candidate_policy_ablation()

    assert result["policy_mode"] == "budget_only"
    assert result["cases_exercising_budget"] == 0
    assert all(row["retention"]["value"] == 1.0 for row in result["cases"])


def test_sufficiency_and_retry_are_reported_separately():
    result = run_sufficiency_retry_ablation()

    assert result["case_count"] == 17
    assert result["gold_grade"] == "NOT_DECISION_GRADE_GOLD"
    assert result["retry"] == {
        "triggered": 3,
        "recovered": 1,
        "still_insufficient": 2,
        "unnecessary": 0,
        "external_calls": 0,
    }


def test_entity_fixture_is_not_promoted_to_source_grounded_gold():
    result = run_entity_ablation()

    assert result["case_count"] == 8
    assert result["gold_grade"] == "NOT_DECISION_GRADE_GOLD"
    assert result["normalized_hit_count"] >= result["literal_hit_count"]


def test_locked_graph_selector_packer_scope_is_explicit():
    result = run_locked_component_diagnostics()

    assert result["graph"]["isolated_quality_delta"] == "N/A"
    assert result["graph"]["medical_claim_eligible_count"] == 0
    assert result["packer"]["all_within_item_budget"] is True
    assert result["packer"]["all_source_backed"] is True
    assert result["packer"]["primary_retained"] == result["packer"]["primary_denominator"]


def test_s2_decisions_use_only_allowed_statuses_and_keep_production_frozen():
    result = run_s2_ablation(reranker_provider="local_rules")

    assert result["overall_status"] == "S2_PARTIAL_EVIDENCE_READY_FOR_S3"
    assert result["a0"] == {
        "dense_depth": 15,
        "sparse_depth": 15,
        "rrf_k": 60,
        "rrf_weights": "equal",
        "top_k": 5,
        "frozen_from_s1": True,
    }
    statuses = {decision["status"] for decision in result["component_decisions"].values()}
    assert statuses <= COMPONENT_STATUSES
    assert result["component_decisions"]["packer"]["status"] == "KEEP_EVIDENCE_SUPPORTED"
    assert result["component_decisions"]["graph"]["status"] == "NOT_CLEANLY_ISOLATABLE"
    assert result["call_counts"]["qdrant_mutations"] == 0
    assert result["call_counts"]["neo4j_mutations"] == 0


def test_s2_writer_emits_every_required_artifact(tmp_path):
    result = run_s2_ablation(reranker_provider="local_rules")
    runtime = {
        "knowledge": 639,
        "entities": 32,
        "neo4j_nodes": 32,
        "neo4j_relationships": 27,
        "semantic_enrichment": "not_run",
        "ingestion": "not_run",
        "reindex": "not_run",
        "reembedding": "not_run",
    }
    complexity = _complexity_inventory()
    git = {
        "branch": "feat/s2-controlled-ablation",
        "head": "test-head",
        "main": "test-main",
        "origin_main": "test-origin-main",
        "status": "",
    }

    _write_machine_outputs(tmp_path, result, runtime, complexity, git)
    _write_markdown_outputs(tmp_path, result, runtime, complexity, git)

    required_markdown = {
        "S2_CONTROLLED_ABLATION_REPORT.md",
        "S2_METHOD_SOURCE_REGISTRY.md",
        "S2_METRIC_DEFINITIONS.md",
        "S2_REFERENCES.md",
        "S2_GROUND_TRUTH_AUDIT.md",
        "S2_DATA_LEAKAGE_AUDIT.md",
        "S2_HEURISTIC_AUDIT.md",
        "S2_RERANKER_ABLATION.md",
        "S2_CANDIDATE_POLICY_ABLATION.md",
        "S2_SUFFICIENCY_RETRY_ABLATION.md",
        "S2_ENTITY_ABLATION.md",
        "S2_GRAPH_ABLATION.md",
        "S2_SELECTOR_PACKER_ABLATION.md",
        "S2_COMPLEXITY_COST.md",
        "S2_COMPONENT_DECISIONS.md",
        "S2_S3_CLEANUP_HANDOFF.md",
        "S2_TEST_RESULTS.md",
        "S2_RUNTIME_INTEGRITY.md",
        "S2_GIT_INTEGRITY.md",
    }
    required_json = {
        "s2_cases.json",
        "s2_metric_results.json",
        "s2_component_deltas.json",
        "s2_component_decisions.json",
        "s2_latency.json",
        "s2_call_counts.json",
        "s2_gold_audit.json",
    }
    names = {path.name for path in tmp_path.iterdir()}
    assert required_markdown <= names
    assert required_json <= names
    assert "S2_PARTIAL_EVIDENCE_READY_FOR_S3" in (
        tmp_path / "S2_CONTROLLED_ABLATION_REPORT.md"
    ).read_text(encoding="utf-8")
