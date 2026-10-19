"""Focused tests for private workflow-runner stage boundaries."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
FOUNDATION_DIR = DEPLOYMENT_DIR / "badgers-foundation"
RUNNER_PATH = (
    DEPLOYMENT_DIR
    / "lambdas"
    / "code"
    / "workflow_runner"
    / "lambda_handler.py"
)
sys.path.insert(0, str(FOUNDATION_DIR))

SPEC = importlib.util.spec_from_file_location("badgers_workflow_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeS3:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.requests: list[dict[str, str]] = []

    def head_object(self, **kwargs: str) -> dict[str, Any]:
        self.requests.append(kwargs)
        return self.response


def _record(status: str = "running") -> dict[str, Any]:
    return {
        "operation_id": "op_" + ("a" * 32),
        "owner_sub": "tenant-a",
        "status": status,
        "source_document_s3_uri": (
            "s3://badgers-source/tenants/tenant-a/contracts/msa.pdf"
        ),
    }


def _event() -> dict[str, Any]:
    return {
        "action": "validate",
        "state": {
            "operation_id": "op_" + ("a" * 32),
            "owner_sub": "tenant-a",
            "audit_mode": False,
        },
    }


def test_validate_stage_requires_tenant_scoped_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages: list[dict[str, str]] = []
    fake_s3 = FakeS3({"ContentLength": 1024, "ContentType": "application/pdf"})
    monkeypatch.setenv("SOURCE_BUCKET_NAME", "badgers-source")
    monkeypatch.setattr(
        runner.operation_state, "get_operation", lambda *_a, **_k: _record()
    )
    monkeypatch.setattr(
        runner.operation_state,
        "update_operation_stage",
        lambda *_a, **kwargs: stages.append(kwargs) or {},
    )
    monkeypatch.setattr(runner.boto3, "client", lambda service: fake_s3)

    result = runner.lambda_handler(_event(), SimpleNamespace())

    assert result["operation_id"] == "op_" + ("a" * 32)
    assert result["source_content_type"] == "application/pdf"
    assert result["source_size_bytes"] == 1024
    assert fake_s3.requests == [
        {
            "Bucket": "badgers-source",
            "Key": "tenants/tenant-a/contracts/msa.pdf",
        }
    ]
    assert stages[0]["status"] == "validating"


def test_validate_stage_rejects_cross_tenant_source_before_s3_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record()
    record["source_document_s3_uri"] = (
        "s3://badgers-source/tenants/tenant-b/contracts/msa.pdf"
    )
    monkeypatch.setenv("SOURCE_BUCKET_NAME", "badgers-source")
    monkeypatch.setattr(
        runner.operation_state, "get_operation", lambda *_a, **_k: record
    )
    monkeypatch.setattr(
        runner.operation_state, "update_operation_stage", lambda *_a, **_k: {}
    )

    with pytest.raises(
        runner.WorkflowRunnerError, match="outside the caller tenant prefix"
    ):
        runner.lambda_handler(_event(), SimpleNamespace())


def test_cancelled_operation_short_circuits_before_any_stage_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.operation_state,
        "get_operation",
        lambda *_a, **_k: _record(status="cancelled"),
    )

    result = runner.lambda_handler(_event(), SimpleNamespace())

    assert result["cancelled"] is True


def test_runner_requires_readable_artifact_in_private_operation_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_s3 = FakeS3({"ContentLength": 1})
    operation_id = "op_" + ("b" * 32)
    monkeypatch.setenv("OUTPUT_BUCKET_NAME", "badgers-output")
    monkeypatch.setattr(runner.boto3, "client", lambda service: fake_s3)

    runner._require_operation_artifact(
        (
            "s3://badgers-output/operations/tenant-a/"
            f"{operation_id}/specialists/full_text_specialist/page_001.xml"
        ),
        operation_id=operation_id,
        owner_sub="tenant-a",
    )

    assert fake_s3.requests == [
        {
            "Bucket": "badgers-output",
            "Key": (
                f"operations/tenant-a/{operation_id}/specialists/"
                "full_text_specialist/page_001.xml"
            ),
        }
    ]

    with pytest.raises(
        runner.WorkflowRunnerError, match="private operation prefix"
    ):
        runner._require_operation_artifact(
            "s3://badgers-output/operations/tenant-b/op_elsewhere/page.xml",
            operation_id=operation_id,
            owner_sub="tenant-a",
        )
