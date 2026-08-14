from pathlib import Path

from scripts import pre_ui_runtime_check as pre_ui


def test_environment_summary_redacts_url_credentials(monkeypatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://user:secret@localhost:6379/0")
    summary = pre_ui.environment_summary()

    assert "secret" not in summary["redis_url"]
    assert "[REDACTED]" in summary["redis_url"]


def test_frontend_config_has_api_contract() -> None:
    status = pre_ui.frontend_config_status(Path(__file__).resolve().parents[1])

    assert status["frontend_exists"] is True
    assert status["package_json"] is True
    assert status["api_contract_source"] is True


def test_pre_ui_requires_s4b_readiness_and_cache_v7(monkeypatch) -> None:
    monkeypatch.setattr(pre_ui, "inspect_readiness", lambda: {"passed": True, "checks": []})
    monkeypatch.setattr(pre_ui, "get_answer_cache_version", lambda: "v7")
    monkeypatch.setattr(
        pre_ui,
        "frontend_config_status",
        lambda: {"frontend_exists": True, "package_json": True, "api_contract_source": True},
    )

    report = pre_ui.run_pre_ui_check()

    assert report["passed"] is True
    assert {item["name"] for item in report["checks"]} == {
        "backend_readiness",
        "cache_version",
        "frontend_config",
    }
