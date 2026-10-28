"""Focused tests for private workflow-runner stage boundaries."""

from __future__ import annotations

import importlib.util
import json
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


class FakeArtifactS3:
    def __init__(self) -> None:
        self.objects: list[dict[str, Any]] = []

    def put_object(self, **kwargs: Any) -> None:
        self.objects.append(kwargs)


def _record(status: str = "running") -> dict[str, Any]:
    return {
        "operation_id": "op_" + ("a" * 32),
        "owner_sub": "tenant-a",
        "status": status,
        "source_document_s3_uri": (
            "s3://badgers-source/tenants/tenant-a/contracts/msa.pdf"
        ),
        "workflow_id": "standard-document-analysis",
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
    monkeypatch.setenv("SPECIALIST_FUNCTIONS_JSON", "{}")
    monkeypatch.setattr(
        runner.operation_state, "get_operation", lambda *_a, **_k: _record()
    )
    monkeypatch.setattr(
        runner.operation_state,
        "update_operation_stage",
        lambda *_a, **kwargs: stages.append(kwargs) or {},
    )
    monkeypatch.setattr(
        runner.workflow_planner,
        "load_release_workflow_catalog",
        lambda *_a, **_k: (
            {
                "input_contract": {
                    "accepted_source_types": ["application/pdf"],
                }
            },
            {},
            {},
        ),
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


def test_native_processor_result_is_retained_in_the_private_operation_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation_id = "op_" + ("n" * 32)
    record = _record()
    record["operation_id"] = operation_id
    record["workflow_id"] = "native-docx-analysis"
    artifact_store = FakeArtifactS3()
    completed: list[tuple[str, str, str]] = []
    stages: list[dict[str, str]] = []
    monkeypatch.setenv("OUTPUT_BUCKET_NAME", "badgers-output")
    monkeypatch.setattr(
        runner.operation_state, "get_operation", lambda *_a, **_k: record
    )
    monkeypatch.setattr(
        runner.operation_state,
        "update_operation_stage",
        lambda *_a, **kwargs: stages.append(kwargs) or {},
    )
    monkeypatch.setattr(
        runner,
        "_invoke_specialist",
        lambda *_a, **_k: {"body": '{"success": true, "result": "{}"}'},
    )
    monkeypatch.setattr(runner.boto3, "client", lambda _service: artifact_store)
    monkeypatch.setattr(
        runner.job_state,
        "mark_complete",
        lambda job_id, subtask, uri: completed.append((job_id, subtask, uri)),
    )

    result = runner._native_process(
        {
            "operation_id": operation_id,
            "owner_sub": "tenant-a",
            "plan": {"resolved_specialists": ["docx_processor"]},
        }
    )

    expected_uri = (
        f"s3://badgers-output/operations/tenant-a/{operation_id}/"
        "specialists/docx_processor/native-result.json"
    )
    assert result["native_result_uri"] == expected_uri
    assert completed == [
        (
            operation_id,
            runner.job_state.subtask_id(
                "docx_processor", _record()["source_document_s3_uri"]
            ),
            expected_uri,
        )
    ]
    assert artifact_store.objects[0]["Key"].endswith(
        "specialists/docx_processor/native-result.json"
    )
    assert stages[0]["stage"] == "native_document"


def test_xlsx_native_workflow_returns_the_single_processor_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation_id = "op_" + ("x" * 32)
    record = _record()
    record["operation_id"] = operation_id
    record["workflow_id"] = "native-xlsx-analysis"
    artifact_store = FakeArtifactS3()
    calls: list[tuple[str, dict[str, object]]] = []
    verified: list[str] = []
    prefix = f"s3://badgers-output/operations/tenant-a/{operation_id}/xlsx"
    native_analysis = f"{prefix}/digest/workbook-analysis.json"
    content_tree = f"{prefix}/digest/content-tree.xml"

    monkeypatch.setenv("OUTPUT_BUCKET_NAME", "badgers-output")
    monkeypatch.setattr(
        runner.operation_state, "get_operation", lambda *_a, **_k: record
    )
    monkeypatch.setattr(
        runner.operation_state, "update_operation_stage", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        runner,
        "_invoke_specialist",
        lambda name, payload: calls.append((name, dict(payload)))
        or {
            "body": json.dumps(
                {
                    "success": True,
                    "result": json.dumps(
                        {
                            "workbook_analysis_uri": native_analysis,
                            "content_tree_uri": content_tree,
                        }
                    ),
                }
            )
        },
    )
    monkeypatch.setattr(
        runner,
        "_require_operation_artifact",
        lambda uri, **_kwargs: verified.append(uri),
    )
    monkeypatch.setattr(runner.boto3, "client", lambda _service: artifact_store)
    monkeypatch.setattr(runner.job_state, "mark_complete", lambda *_a, **_k: None)

    result = runner._native_process(
        {
            "operation_id": operation_id,
            "owner_sub": "tenant-a",
            "plan": {
                "resolved_specialists": ["xlsx_processor"]
            },
        }
    )

    assert [name for name, _ in calls] == ["xlsx_processor"]
    assert verified == [content_tree]
    assert result["content_tree_uri"] == content_tree


def test_native_workflow_finalizes_without_a_correlation_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation_id = "op_" + ("q" * 32)
    record = _record()
    record["operation_id"] = operation_id
    record["workflow_id"] = "native-docx-analysis"
    verified: list[str] = []
    monkeypatch.setenv("DEPLOYMENT_ID", "beta")
    monkeypatch.setattr(
        runner.operation_state, "get_operation", lambda *_a, **_k: record
    )
    monkeypatch.setattr(
        runner.operation_state, "update_operation_stage", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        runner,
        "_require_operation_artifact",
        lambda uri, **_kwargs: verified.append(uri),
    )
    monkeypatch.setattr(
        runner.operation_state,
        "complete_operation",
        lambda **kwargs: {"status": "completed", **kwargs},
    )

    uri = (
        f"s3://badgers-output/operations/tenant-a/{operation_id}/"
        "specialists/docx_processor/native-result.json"
    )
    result = runner._finalize(
        {
            "operation_id": operation_id,
            "owner_sub": "tenant-a",
            "native_result_uri": uri,
        }
    )

    assert verified == [uri]
    assert result["result"]["status"] == "completed"
    assert result["result"]["summary"].startswith("Native document processing")
