"""Attempt-history regression coverage for report audits."""

from __future__ import annotations

import sys
from pathlib import Path

DEPLOYMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOYMENT / "badgers-foundation"))

from foundation import job_state  # noqa: E402


def test_mark_running_appends_attempt_and_clears_stale_completion(monkeypatch) -> None:
    calls: list[dict] = []

    class Table:
        def update_item(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(job_state, "enabled", lambda: True)
    monkeypatch.setattr(job_state, "_table", lambda: Table())
    job_state.mark_running("job", "specialist#page", specialist="specialist")

    expression = calls[0]["UpdateExpression"]
    assert "list_append" in expression
    assert "attempt_count" in expression
    assert "REMOVE #e, completed_at" in expression


def test_report_attempt_helpers_preserve_legacy_rows() -> None:
    import importlib.util

    report_dir = DEPLOYMENT / "lambdas" / "code" / "html_report_specialist"
    sys.path.insert(0, str(report_dir))
    spec = importlib.util.spec_from_file_location(
        "report_attempt_handler", report_dir / "lambda_handler.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    records = [
        {
            "subtask_id": "keyword#page_1",
            "status": "COMPLETE",
            "attempts": [
                {"status": "FAILED", "error": "first"},
                {"status": "COMPLETE"},
            ],
        },
        {"subtask_id": "legacy#page_1", "status": "FAILED"},
    ]
    counts = module._execution_counts(records)

    assert counts == {
        "invocation_count": 3,
        "complete_count": 1,
        "failed_count": 2,
        "retried_count": 1,
    }
    assert module._attempts(records[1])[0]["status"] == "FAILED"


def test_report_ui_lists_attempts_and_retry_count() -> None:
    report_ui = (DEPLOYMENT.parent / "ui" / "src" / "components" / "Reports.jsx").read_text()
    renderer = (
        DEPLOYMENT / "lambdas" / "code" / "html_report_specialist" / "renderer.py"
    ).read_text()

    assert "Attempt {attemptIndex + 1}" in report_ui
    assert 'Metric label="Retried"' in report_ui
    assert "Attempt ${n+1}:" in renderer
