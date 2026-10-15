"""Tests for extension-keyed document intake routing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROUTER_DIRECTORY = Path(__file__).resolve().parents[2] / "lambdas" / "code" / "document_intake_router"
sys.path.insert(0, str(ROUTER_DIRECTORY))

from lambda_handler import lambda_handler  # noqa: E402


def _body(response: dict[str, object]) -> dict[str, object]:
    return json.loads(str(response["body"]))


def test_docx_selects_the_native_page_plan_processor() -> None:
    response = lambda_handler(
        {
            "document_path": "s3://input-bucket/documents/Research.DOCX",
            "session_id": "session-1",
            "job_id": "job-1",
            "doc_id": "doc-1",
        },
        None,
    )
    body = _body(response)
    result = json.loads(str(body["result"]))

    assert response["statusCode"] == 200
    assert result["input_type"] == "docx"
    assert result["processor_tool"] == "process_docx_tool"
    assert result["result_contract"] == "page_plan"
    assert result["session_id"] == "session-1"


def test_legacy_and_future_office_formats_return_actionable_errors() -> None:
    legacy = _body(
        lambda_handler({"document_path": "s3://input-bucket/legacy.doc"}, None)
    )
    future = _body(
        lambda_handler({"document_path": "s3://input-bucket/deck.pptx"}, None)
    )

    assert "Save the document as .docx or PDF" in str(legacy["error"])
    assert "future Office intake feature" in str(future["error"])
