from __future__ import annotations

from scripts.plan_taxonomy_graph_update import build_taxonomy_graph_update_plan


def test_taxonomy_graph_planner_is_dry_run() -> None:
    plan = build_taxonomy_graph_update_plan()

    assert plan["mutation_executed"] is False
    assert plan["current"]["node_count"] == 32
    assert plan["proposed"]["node_count"] == 32


def test_taxonomy_graph_planner_reports_no_delete_execution() -> None:
    plan = build_taxonomy_graph_update_plan()

    assert plan["delete_candidates"]["nodes"] == []
    assert plan["delete_candidates"]["relationships"] == []
    assert plan["conflicts"] == []
