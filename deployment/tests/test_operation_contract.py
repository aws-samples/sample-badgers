"""Tests for versioned Headless operation contracts and pointer persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator


DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
FOUNDATION_DIR = DEPLOYMENT_DIR / "badgers-foundation"
sys.path.insert(0, str(FOUNDATION_DIR))

from foundation import operation_contract, operation_state  # noqa: E402
from foundation import s3_result_saver  # noqa: E402


CONTRACTS_DIR = DEPLOYMENT_DIR / "s3_files" / "contracts"


class FakeTable:
    """Small DynamoDB table double for operation-state behavior."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict[str, Any]] = {}
        self.update_calls: list[dict[str, Any]] = []

    def put_item(self, *, Item: dict[str, Any], **_kwargs: Any) -> None:
        key = (Item["job_id"], Item["subtask_id"])
        if key in self.items:
            raise RuntimeError("ConditionalCheckFailedException")
        self.items[key] = dict(Item)

    def update_item(
        self,
        *,
        Key: dict[str, str],
        ExpressionAttributeValues: dict[str, Any],
        **_kwargs: Any,
    ) -> None:
        self.update_calls.append(
            {
                "Key": dict(Key),
                "ExpressionAttributeValues": dict(ExpressionAttributeValues),
                **_kwargs,
            }
        )
        item = self.items[(Key["job_id"], Key["subtask_id"])]
        values = ExpressionAttributeValues
        if "ConditionExpression" in _kwargs and ":owner" in values:
            allowed_statuses = {
                value
                for key, value in values.items()
                if key.startswith(":active_")
            }
            if (
                item.get("owner_sub") != values[":owner"]
                or item.get("status") not in allowed_statuses
            ):
                raise RuntimeError("ConditionalCheckFailedException")
        if ":uri" in values:
            if ":status" in values:
                item["completion_manifest_uri"] = values[":uri"]
            else:
                item["status_manifest_uri"] = values[":uri"]
        if ":plan_uri" in values:
            item["plan_uri"] = values[":plan_uri"]
        if ":status" in values:
            item["status"] = values[":status"]
        if ":stage" in values:
            item["current_stage"] = values[":stage"]
        if ":execution_arn" in values:
            item["workflow_execution_arn"] = values[":execution_arn"]
        if ":updated" in values:
            item["updated_at"] = values[":updated"]
        if ":ttl" in values:
            item["ttl"] = values[":ttl"]
        if ":error_code" in values:
            item["error_code"] = values[":error_code"]

    def get_item(self, *, Key: dict[str, str], **_kwargs: Any) -> dict[str, Any]:
        item = self.items.get((Key["job_id"], Key["subtask_id"]))
        return {"Item": dict(item)} if item else {}


class FakeS3:
    """Captures private JSON objects without making any network calls."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}

    def put_object(self, **kwargs: Any) -> None:
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = kwargs


def _load_schema(name: str) -> dict[str, Any]:
    return json.loads((CONTRACTS_DIR / name).read_text(encoding="utf-8"))


def test_contract_schemas_are_valid_draft_2020_12_schemas() -> None:
    for path in sorted(CONTRACTS_DIR.glob("*.json")):
        Draft202012Validator.check_schema(
            json.loads(path.read_text(encoding="utf-8"))
        )


def test_versioned_agent_request_rejects_control_plane_inputs() -> None:
    request = operation_contract.normalize_agent_request(
        {
            "schema_version": "badgers-agent-request/v1",
            "session_id": "review-104",
            "document_uri": "s3://customer-source/tenant-a/msa.pdf",
            "instruction": "Identify termination clauses.",
            "routing_mode": "manual",
            "workflow_id": "legal-contract-review",
            "caller_context": {"correlation_id": "case-7281"},
        }
    )

    assert request.versioned is True
    assert request.routing_mode == "manual"
    assert request.document_uri.endswith("msa.pdf")

    for forbidden_field in ("output_prefix", "role_arn", "config_uri"):
        with pytest.raises(
            operation_contract.OperationContractError, match=forbidden_field
        ):
            operation_contract.normalize_agent_request(
                {
                    "schema_version": "badgers-agent-request/v1",
                    "session_id": "review-104",
                    "document_uri": "s3://customer-source/tenant-a/msa.pdf",
                    "instruction": "Identify termination clauses.",
                    forbidden_field: "s3://another-tenant/results/",
                }
            )


def test_completion_manifest_is_bounded_and_schema_valid() -> None:
    summary = "x" * 4_000
    result = operation_contract.build_operation_result(
        status="completed",
        operation_id="op_" + ("a" * 32),
        session_id="review-104",
        workflow_id="standard-document-analysis",
        routing_mode="auto",
        source_document_s3_uri="s3://customer-source/tenant-a/msa.pdf",
        s3_classification_uri="s3://badgers-output/tenant-a/classification.xml",
        s3_correlated_uri="s3://badgers-output/tenant-a/correlated.xml",
        completion_manifest_uri="s3://badgers-output/tenant-a/completion.json",
        individual_specialists=[
            {
                "specialist": "table_specialist",
                "s3_uri": "s3://badgers-output/tenant-a/tables.xml",
                "content_type": "application/xml",
            }
        ],
        summary=summary,
    )

    assert len(result["summary"]) == 2_000
    assert result["original_pdf_s3_uri"].endswith("msa.pdf")
    assert "artifact_body" not in result
    Draft202012Validator(_load_schema("badgers-operation-result-v1.json")).validate(
        result
    )


@pytest.mark.parametrize(
    ("status", "source_uri", "trace_id"),
    [
        ("completed", "s3://customer-source/tenant-a/msa.pdf", "1-abcdef01-traced"),
        ("failed", "s3://customer-source/tenant-a/photo.png", ""),
        ("cancelled", "s3://customer-source/tenant-a/photo.png", ""),
    ],
)
def test_result_schema_supports_terminal_pdf_image_and_trace_variants(
    status: str,
    source_uri: str,
    trace_id: str,
) -> None:
    result = operation_contract.build_operation_result(
        status=status,
        operation_id="op_" + ("d" * 32),
        session_id="review-104",
        workflow_id="standard-document-analysis",
        routing_mode="auto",
        source_document_s3_uri=source_uri,
        trace_id=trace_id,
        error=(
            {"code": "OPERATION_STOPPED", "message": "Stopped safely."}
            if status != "completed"
            else None
        ),
    )

    if source_uri.endswith(".pdf"):
        assert result["original_pdf_s3_uri"] == source_uri
    else:
        assert "original_pdf_s3_uri" not in result
    if trace_id:
        assert result["trace_id"] == trace_id
    else:
        assert "trace_id" not in result
    Draft202012Validator(_load_schema("badgers-operation-result-v1.json")).validate(
        result
    )


def test_operation_state_persists_private_status_and_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = FakeTable()
    s3 = FakeS3()
    operation_id = "op_" + ("b" * 32)

    monkeypatch.setenv("JOBS_TABLE_NAME", "badgers-jobs-test")
    monkeypatch.setenv("OUTPUT_BUCKET_NAME", "badgers-output-test")
    monkeypatch.setenv("CONFIG_PREFIX", "releases/" + ("c" * 64))
    monkeypatch.setattr(operation_state, "_table", lambda: table)
    monkeypatch.setattr(operation_state, "_s3_client", lambda: s3)
    monkeypatch.setattr(operation_state.job_state, "create_job", lambda *_a, **_k: None)
    monkeypatch.setattr(
        operation_state.job_state,
        "get_job_records",
        lambda _operation_id: [
            {
                "status": "COMPLETE",
                "specialist": "classify_pdf_content",
                "result_s3_key": (
                    f"s3://badgers-output-test/operations/tenant-a-workload/"
                    f"{operation_id}/classification/classification.xml"
                ),
            },
            {
                "status": "COMPLETE",
                "specialist": "full_text_specialist",
                "result_s3_key": (
                    f"s3://badgers-output-test/operations/tenant-a-workload/"
                    f"{operation_id}/specialists/full-text.xml"
                ),
            },
            {
                "status": "COMPLETE",
                "specialist": "correlation_specialist",
                "result_s3_key": (
                    f"s3://badgers-output-test/operations/tenant-a-workload/"
                    f"{operation_id}/correlation/correlated.xml"
                ),
            },
        ],
    )

    created = operation_state.create_operation(
        operation_id=operation_id,
        session_id="review-104",
        owner_sub="tenant-a-workload",
        workflow_id="standard-document-analysis",
        routing_mode="auto",
        source_document_s3_uri="s3://customer-source/tenant-a/msa.pdf",
        deployment_id="test",
    )

    assert created["config_release_id"] == "c" * 64
    assert created["status_manifest_uri"].startswith("s3://badgers-output-test/")
    assert created["plan_uri"].endswith("plan/plan.json")
    assert len(s3.objects) == 2
    assert operation_state.get_operation(operation_id, owner_sub="tenant-a-workload")

    with pytest.raises(operation_state.OperationOwnershipError):
        operation_state.get_operation(operation_id, owner_sub="tenant-b-workload")

    completed = operation_state.complete_operation(
        operation_id=operation_id,
        owner_sub="tenant-a-workload",
        deployment_id="test",
        summary="Completed document analysis.",
    )

    assert completed["status"] == "completed"
    assert completed["s3_classification_uri"].endswith("classification.xml")
    assert completed["s3_correlated_uri"].endswith("correlated.xml")
    assert completed["individual_specialists"] == [
        {
            "specialist": "full_text_specialist",
            "s3_uri": (
                f"s3://badgers-output-test/operations/tenant-a-workload/"
                f"{operation_id}/specialists/full-text.xml"
            ),
            "content_type": "application/xml",
        }
    ]
    assert completed["completion_manifest_uri"].endswith("completion.json")
    assert completed["plan_uri"].endswith("plan/plan.json")
    assert len(s3.objects) == 3
    status_body = s3.objects[
        (
            "badgers-output-test",
            f"operations/tenant-a-workload/{operation_id}/status.json",
        )
    ]["Body"]
    assert json.loads(status_body)["status"] == "completed"
    Draft202012Validator(_load_schema("badgers-operation-result-v1.json")).validate(
        completed
    )


def test_operation_state_persists_cancelled_pointer_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = FakeTable()
    s3 = FakeS3()
    operation_id = "op_" + ("e" * 32)
    monkeypatch.setenv("JOBS_TABLE_NAME", "badgers-jobs-test")
    monkeypatch.setenv("OUTPUT_BUCKET_NAME", "badgers-output-test")
    monkeypatch.setenv("CONFIG_PREFIX", "releases/" + ("f" * 64))
    monkeypatch.setattr(operation_state, "_table", lambda: table)
    monkeypatch.setattr(operation_state, "_s3_client", lambda: s3)
    monkeypatch.setattr(operation_state.job_state, "create_job", lambda *_a, **_k: None)
    monkeypatch.setattr(operation_state.job_state, "get_job_records", lambda _id: [])

    operation_state.create_operation(
        operation_id=operation_id,
        session_id="review-104",
        owner_sub="tenant-a-workload",
        workflow_id="standard-document-analysis",
        routing_mode="manual",
        source_document_s3_uri="s3://customer-source/tenant-a/msa.pdf",
    )
    cancelled = operation_state.cancel_operation(
        operation_id,
        owner_sub="tenant-a-workload",
        reason="User cancelled before specialist execution.",
    )

    assert cancelled["status"] == "cancelled"
    assert cancelled["completion_manifest_uri"].endswith("cancelled.json")
    assert cancelled["error"]["code"] == "CANCELLED"
    record = operation_state.get_operation(operation_id, owner_sub="tenant-a-workload")
    assert record["status"] == "cancelled"


def test_operation_stage_persists_lifecycle_and_execution_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = FakeTable()
    s3 = FakeS3()
    operation_id = "op_" + ("f" * 32)
    monkeypatch.setenv("JOBS_TABLE_NAME", "badgers-jobs-test")
    monkeypatch.setenv("OUTPUT_BUCKET_NAME", "badgers-output-test")
    monkeypatch.setenv("CONFIG_PREFIX", "releases/" + ("a" * 64))
    monkeypatch.setattr(operation_state, "_table", lambda: table)
    monkeypatch.setattr(operation_state, "_s3_client", lambda: s3)
    monkeypatch.setattr(operation_state.job_state, "create_job", lambda *_a, **_k: None)

    operation_state.create_operation(
        operation_id=operation_id,
        session_id="review-104",
        owner_sub="tenant-a-workload",
        workflow_id="standard-document-analysis",
        routing_mode="auto",
        source_document_s3_uri="s3://customer-source/tenant-a/msa.pdf",
    )

    execution_arn = (
        "arn:aws:states:us-west-2:123456789012:execution:"
        "badgers-workflow-test:run-001"
    )
    updated = operation_state.update_operation_stage(
        operation_id,
        owner_sub="tenant-a-workload",
        status="validating",
        stage="validate_source",
        execution_arn=execution_arn,
    )

    assert updated["status"] == "validating"
    assert updated["current_stage"] == "validate_source"
    assert updated["workflow_execution_arn"] == execution_arn
    status_body = s3.objects[
        (
            "badgers-output-test",
            f"operations/tenant-a-workload/{operation_id}/status.json",
        )
    ]["Body"]
    status = json.loads(status_body)
    assert status["current_stage"] == "validate_source"
    assert "ConditionExpression" in table.update_calls[-1]
    assert "owner_sub = :owner" in table.update_calls[-1]["ConditionExpression"]
    assert "#status IN" in table.update_calls[-1]["ConditionExpression"]


def test_operation_manifest_ignores_out_of_prefix_artifact_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = FakeTable()
    s3 = FakeS3()
    operation_id = "op_" + ("9" * 32)
    monkeypatch.setenv("JOBS_TABLE_NAME", "badgers-jobs-test")
    monkeypatch.setenv("OUTPUT_BUCKET_NAME", "badgers-output-test")
    monkeypatch.setenv("CONFIG_PREFIX", "releases/" + ("d" * 64))
    monkeypatch.setattr(operation_state, "_table", lambda: table)
    monkeypatch.setattr(operation_state, "_s3_client", lambda: s3)
    monkeypatch.setattr(operation_state.job_state, "create_job", lambda *_a, **_k: None)
    monkeypatch.setattr(
        operation_state.job_state,
        "get_job_records",
        lambda _operation_id: [
            {
                "status": "COMPLETE",
                "specialist": "full_text_specialist",
                "result_s3_key": (
                    "s3://badgers-output-test/operations/another-owner/"
                    "op_elsewhere/private.xml"
                ),
            }
        ],
    )
    operation_state.create_operation(
        operation_id=operation_id,
        session_id="review-104",
        owner_sub="tenant-a-workload",
        workflow_id="standard-document-analysis",
        routing_mode="auto",
    )

    result = operation_state.complete_operation(
        operation_id=operation_id,
        owner_sub="tenant-a-workload",
    )

    assert result["individual_specialists"] == []


def test_terminal_operation_state_cannot_be_overwritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = FakeTable()
    s3 = FakeS3()
    operation_id = "op_" + ("8" * 32)
    monkeypatch.setenv("JOBS_TABLE_NAME", "badgers-jobs-test")
    monkeypatch.setenv("OUTPUT_BUCKET_NAME", "badgers-output-test")
    monkeypatch.setenv("CONFIG_PREFIX", "releases/" + ("e" * 64))
    monkeypatch.setattr(operation_state, "_table", lambda: table)
    monkeypatch.setattr(operation_state, "_s3_client", lambda: s3)
    monkeypatch.setattr(operation_state.job_state, "create_job", lambda *_a, **_k: None)
    monkeypatch.setattr(operation_state.job_state, "get_job_records", lambda _id: [])
    operation_state.create_operation(
        operation_id=operation_id,
        session_id="review-104",
        owner_sub="tenant-a-workload",
        workflow_id="standard-document-analysis",
        routing_mode="auto",
    )
    operation_state.complete_operation(
        operation_id=operation_id,
        owner_sub="tenant-a-workload",
    )

    with pytest.raises(operation_state.OperationStateError, match="terminal"):
        operation_state.fail_operation(
            operation_id=operation_id,
            owner_sub="tenant-a-workload",
            error_code="WORKFLOW_STAGE_FAILED",
            error_message="A late workflow error must not replace completion.",
        )


def test_runner_managed_specialist_result_uses_server_derived_operation_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ResultS3:
        def __init__(self) -> None:
            self.puts: list[dict[str, Any]] = []

        def put_object(self, **kwargs: Any) -> None:
            self.puts.append(kwargs)

    result_s3 = ResultS3()
    monkeypatch.setattr(s3_result_saver.boto3, "client", lambda _service: result_s3)
    monkeypatch.setattr(
        operation_state,
        "get_operation",
        lambda _id: {"owner_sub": "tenant-a"},
    )
    operation_id = "op_" + ("a" * 32)
    s3_uri = s3_result_saver.save_result_to_s3(
        result="<result/>",
        specialist_name="full_text_specialist",
        output_bucket="badgers-output-test",
        session_id="untrusted-session",
        image_path="s3://badgers-output-test/temp/op/page_001.b64",
        job_id=operation_id,
    )

    assert s3_uri.startswith(
        "s3://badgers-output-test/operations/tenant-a/" + operation_id + "/specialists/"
    )
    assert result_s3.puts[0]["Key"].startswith(
        "operations/tenant-a/" + operation_id + "/specialists/full_text_specialist/"
    )


def test_runner_managed_specialist_rejects_unknown_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operation_state, "get_operation", lambda _id: {})
    monkeypatch.setattr(s3_result_saver.boto3, "client", lambda _service: object())

    with pytest.raises(
        s3_result_saver.S3ResultSaverError, match="no durable operation owner"
    ):
        s3_result_saver.save_result_to_s3(
            result="<result/>",
            specialist_name="full_text_specialist",
            output_bucket="badgers-output-test",
            session_id="untrusted-session",
            job_id="op_" + ("b" * 32),
        )


def test_runner_managed_converter_pages_use_server_derived_operation_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    converter_path = (
        DEPLOYMENT_DIR
        / "lambdas"
        / "code"
        / "pdf_to_images_converter"
        / "lambda_handler.py"
    )
    spec = spec_from_file_location("badgers_pdf_converter", converter_path)
    assert spec and spec.loader
    converter = module_from_spec(spec)
    spec.loader.exec_module(converter)

    class PageS3:
        def __init__(self) -> None:
            self.puts: list[dict[str, Any]] = []

        def put_object(self, **kwargs: Any) -> None:
            self.puts.append(kwargs)

    page_s3 = PageS3()
    operation_id = "op_" + ("c" * 32)
    monkeypatch.setattr(converter, "OUTPUT_BUCKET", "badgers-output-test")
    import boto3

    monkeypatch.setattr(boto3, "client", lambda _service: page_s3)
    monkeypatch.setattr(
        converter.operation_state,
        "get_operation",
        lambda _id: {"owner_sub": "tenant-a"},
    )

    paths = converter._store_images_to_s3(
        ["cGFnZS0x"],
        "untrusted-session",
        job_id=operation_id,
    )

    expected_key = f"operations/tenant-a/{operation_id}/pages/page_001.b64"
    assert paths == [f"s3://badgers-output-test/{expected_key}"]
    assert page_s3.puts[0]["Key"] == expected_key
    assert page_s3.puts[0]["IfNoneMatch"] == "*"


def test_runner_managed_converter_reuses_existing_page_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    converter_path = (
        DEPLOYMENT_DIR
        / "lambdas"
        / "code"
        / "pdf_to_images_converter"
        / "lambda_handler.py"
    )
    spec = spec_from_file_location("badgers_pdf_converter_retry", converter_path)
    assert spec and spec.loader
    converter = module_from_spec(spec)
    spec.loader.exec_module(converter)

    class AlreadyWrittenError(Exception):
        response = {"Error": {"Code": "PreconditionFailed"}}

    class ExistingPageS3:
        def put_object(self, **_kwargs: Any) -> None:
            raise AlreadyWrittenError()

    operation_id = "op_" + ("d" * 32)
    monkeypatch.setattr(converter, "OUTPUT_BUCKET", "badgers-output-test")
    import boto3

    monkeypatch.setattr(boto3, "client", lambda _service: ExistingPageS3())
    monkeypatch.setattr(
        converter.operation_state,
        "get_operation",
        lambda _id: {"owner_sub": "tenant-a"},
    )

    assert converter._store_images_to_s3(
        ["cGFnZS0x"], "untrusted-session", job_id=operation_id
    ) == [
        f"s3://badgers-output-test/operations/tenant-a/{operation_id}/pages/page_001.b64"
    ]
