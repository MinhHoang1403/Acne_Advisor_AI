from __future__ import annotations

import subprocess

from scripts import check_runtime_contracts


def test_runtime_contract_commands_never_use_live_chat_or_ingestion() -> None:
    flattened = " ".join(" ".join(command) for command in check_runtime_contracts.CHECKS)
    assert "--live-chat" not in flattened
    assert "--mode offline" in flattened
    assert "ingest_knowledge.py" not in flattened
    assert "eval_" not in flattened


def test_runtime_contract_summary_with_mocked_subprocess(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout='{"passed": true}', stderr="")

    monkeypatch.setattr(check_runtime_contracts.subprocess, "run", fake_run)
    summary = check_runtime_contracts.run_runtime_contracts(timeout_seconds=1)
    assert summary["passed"] is True
    assert summary["total_checks"] == len(check_runtime_contracts.CHECKS)
