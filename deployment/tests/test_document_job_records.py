"""Document-level report record merging."""

from __future__ import annotations

import sys
from pathlib import Path

DEPLOYMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOYMENT / "badgers-foundation"))

from foundation import job_state  # noqa: E402


def test_merge_document_records_retains_attempts_from_retry_jobs() -> None:
    records = job_state.merge_job_records(
        [
            [
                {
                    "subtask_id": "keyword#page_1",
                    "status": "FAILED",
                    "started_at": "2026-11-18T10:00:00+00:00",
                    "attempts": [{"status": "FAILED", "error": "first run"}],
                }
            ],
            [
                {
                    "subtask_id": "keyword#page_1",
                    "status": "COMPLETE",
                    "completed_at": "2026-11-18T11:00:00+00:00",
                    "result_s3_key": "s3://output/keyword.json",
                    "attempts": [{"status": "COMPLETE"}],
                }
            ],
        ]
    )

    if not (len(records) == 1):
        raise AssertionError
    if not (records[0]["status"] == "COMPLETE"):
        raise AssertionError
    if not ([item["status"] for item in records[0]["attempts"]] == [
        "FAILED",
        "COMPLETE",
    ]):
        raise AssertionError


def test_report_prompt_and_iam_allow_document_history_reads() -> None:
    prompt = (
        DEPLOYMENT / "s3_files" / "agent_system_prompt" / "agent_system_prompt.xml"
    ).read_text()
    iam = (DEPLOYMENT / "stacks" / "iam_stack.py").read_text()

    if not ("After a successful re-run of a document" in prompt):
        raise AssertionError
    if not ('f"{jobs_table.table_arn}/index/doc-index"' in iam):
        raise AssertionError
