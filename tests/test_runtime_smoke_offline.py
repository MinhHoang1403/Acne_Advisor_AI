from __future__ import annotations

from scripts.smoke_runtime import run_offline_smoke


def test_runtime_smoke_offline_passes_without_live_chat():
    report = run_offline_smoke()

    assert report["mode"] == "offline"
    assert report["passed"] is True
    assert report["errors"] == []
    assert len(report["cases"]) >= 8
    assert all(case["passed"] for case in report["cases"])
