"""Durable operation state and pointer-manifest persistence for BADGERS.

This module reuses the deployment's DynamoDB jobs table for an operation-level
row and writes small JSON status/completion manifests into the private output
bucket.  Specialist artifacts are never copied into these manifests; only
their S3 URIs and minimal metadata are recorded.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Mapping

from . import job_state
from .operation_contract import (
    OPERATION_RECORD_SCHEMA_VERSION,
    artifact_content_type,
    build_operation_result,
    operation_record,
    utc_now,
)
from .workflow_planner import WorkflowPlanningError, validate_resolved_plan

logger = logging.getLogger(__name__)

OPERATION_SUBTASK = "operation"
_TTL_DAYS = 30
_dynamodb = None
_s3 = None
_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]")


class OperationStateError(RuntimeError):
    """Raised when a versioned operation cannot be safely persisted."""


class OperationOwnershipError(PermissionError):
    """Raised when an operation is not owned by the requested caller."""


_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_ACTIVE_STATUSES = frozenset(
    {"accepted", "validating", "converting", "classifying", "planning", "running", "correlating", "finalizing"}
)


def _table_name() -> str:
    return os.environ.get("JOBS_TABLE_NAME", "")


def _output_bucket() -> str:
    return os.environ.get("OUTPUT_BUCKET_NAME", "")


def enabled() -> bool:
    """True when the runtime has the resources needed for durable operations."""

    return bool(_table_name() and _output_bucket())


def _table():
    global _dynamodb
    if _dynamodb is None:
        import boto3

        _dynamodb = boto3.resource(
            "dynamodb",
            region_name=(
                os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or "us-west-2"
            ),
        )
    return _dynamodb.Table(_table_name())


def _s3_client():
    global _s3
    if _s3 is None:
        import boto3

        _s3 = boto3.client(
            "s3",
            region_name=(
                os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or "us-west-2"
            ),
        )
    return _s3


def _ttl() -> int:
    return int(time.time()) + (_TTL_DAYS * 86400)


def _safe_segment(value: str, *, fallback: str) -> str:
    cleaned = _SAFE_SEGMENT_RE.sub("_", value.strip())
    return cleaned[:128] or fallback


def operation_prefix(owner_sub: str, operation_id: str) -> str:
    """Return the server-derived private output prefix for an operation."""

    return (
        "operations/"
        f"{_safe_segment(owner_sub, fallback='unattributed')}/"
        f"{_safe_segment(operation_id, fallback='invalid-operation')}"
    )


def tenant_segment(owner_sub: str) -> str:
    """Return the server-safe tenant segment used in source/output prefixes."""

    return _safe_segment(owner_sub, fallback="unattributed")


def _s3_uri(key: str) -> str:
    return f"s3://{_output_bucket()}/{key}"


def _is_operation_artifact_uri(
    uri: str, *, operation_id: str, owner_sub: str
) -> bool:
    """Return whether a persisted artifact pointer belongs to this operation.

    DynamoDB job records are written by specialists and are therefore treated
    as untrusted metadata at the operation-result boundary. A record may point
    only into the server-derived prefix for this owned operation.
    """

    if not isinstance(uri, str) or not uri.startswith("s3://"):
        return False
    bucket_and_key = uri[5:].split("/", 1)
    if len(bucket_and_key) != 2:
        return False
    bucket, key = bucket_and_key
    return (
        bucket == _output_bucket()
        and key.startswith(f"{operation_prefix(owner_sub, operation_id)}/")
    )


def _require_enabled() -> None:
    if not enabled():
        raise OperationStateError(
            "JOBS_TABLE_NAME and OUTPUT_BUCKET_NAME are required for "
            "versioned BADGERS operations"
        )


def _is_conditional_conflict(error: Exception) -> bool:
    """Identify a DynamoDB compare-and-set failure without SDK coupling."""

    return (
        type(error).__name__ == "ConditionalCheckFailedException"
        or "ConditionalCheckFailed" in str(error)
    )


def _put_json(key: str, value: Mapping[str, Any]) -> str:
    try:
        _s3_client().put_object(
            Bucket=_output_bucket(),
            Key=key,
            Body=(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
            ContentType="application/json",
        )
    except Exception as error:
        raise OperationStateError(f"Could not persist operation manifest: {error}") from error
    return _s3_uri(key)


def _status_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a small status document without internal DynamoDB key names."""

    return {
        key: value
        for key, value in record.items()
        if key
        in {
            "schema_version",
            "operation_id",
            "session_id",
            "status",
            "current_stage",
            "workflow_execution_arn",
            "workflow_id",
            "routing_mode",
            "workflow_version",
            "routing_policy_id",
            "routing_policy_version",
            "source_document_s3_uri",
            "config_release_id",
            "correlation_id",
            "principal_id",
            "identity_provider",
            "plan_uri",
            "started_at",
            "updated_at",
            "completion_manifest_uri",
        }
        and value not in ("", None)
    }


def configured_release_id() -> str:
    """Derive the pinned release ID from the deploy-injected configuration prefix."""

    prefix = os.environ.get("CONFIG_PREFIX", "").strip("/")
    if prefix.startswith("releases/"):
        return prefix.split("/", 1)[1]
    return ""


def current_trace_id() -> str:
    """Extract the root X-Ray trace identifier only when the runtime knows it."""

    trace_header = os.environ.get("_X_AMZN_TRACE_ID", "")
    for segment in trace_header.split(";"):
        key, separator, value = segment.partition("=")
        if separator and key == "Root" and value:
            return value
    return ""


def create_operation(
    *,
    operation_id: str,
    session_id: str,
    owner_sub: str,
    workflow_id: str,
    routing_mode: str,
    source_document_s3_uri: str = "",
    correlation_id: str = "",
    principal_id: str = "",
    identity_provider: str = "",
    executor: str = "agent_runtime",
    deployment_id: str = "",
) -> dict[str, Any]:
    """Create an operation row and initial private status artifact.

    ``owner_sub`` is the authenticated/injected workload identity at this
    boundary.  Gateway will later derive it from its verified caller identity;
    this function never accepts an output prefix from a caller.
    """

    _require_enabled()
    release_id = configured_release_id()
    if not re.fullmatch(r"[a-f0-9]{64}", release_id):
        raise OperationStateError(
            "Versioned BADGERS operations require a SHA-pinned CONFIG_PREFIX "
            "of the form releases/<64 lowercase hex characters>"
        )
    record = operation_record(
        operation_id=operation_id,
        session_id=session_id,
        owner_sub=owner_sub,
        status="accepted",
        workflow_id=workflow_id,
        routing_mode=routing_mode,
        source_document_s3_uri=source_document_s3_uri,
        config_release_id=release_id,
        correlation_id=correlation_id,
        principal_id=principal_id,
        identity_provider=identity_provider,
    )
    item = {
        "job_id": operation_id,
        "subtask_id": OPERATION_SUBTASK,
        "entity_type": "operation",
        "ttl": _ttl(),
        **record,
    }
    try:
        _table().put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(job_id)",
        )
    except Exception as error:
        raise OperationStateError(f"Could not create operation record: {error}") from error

    try:
        plan_uri = _put_json(
            f"{operation_prefix(owner_sub, operation_id)}/plan/plan.json",
            {
                "schema_version": "badgers-operation-plan/v1",
                "operation_id": operation_id,
                "session_id": session_id,
                "workflow_id": workflow_id,
                "routing_mode": routing_mode,
                "config_release_id": release_id,
                "source_document_s3_uri": source_document_s3_uri,
                "executor": executor,
                "created_at": utc_now(),
            },
        )
        status_uri = _put_json(
            f"{operation_prefix(owner_sub, operation_id)}/status.json",
            _status_payload(record),
        )
        _table().update_item(
            Key={"job_id": operation_id, "subtask_id": OPERATION_SUBTASK},
            UpdateExpression=(
                "SET status_manifest_uri = :uri, plan_uri = :plan_uri, "
                "updated_at = :updated, #ttl = :ttl"
            ),
            ExpressionAttributeNames={"#ttl": "ttl"},
            ExpressionAttributeValues={
                ":uri": status_uri,
                ":plan_uri": plan_uri,
                ":updated": utc_now(),
                ":ttl": _ttl(),
            },
        )
    except Exception as error:
        raise OperationStateError(
            f"Could not persist initial operation status: {error}"
        ) from error

    # Keep legacy job-oriented readers working. The future operation API uses
    # the dedicated operation row above, while specialists continue to write
    # their deterministic subtask rows under the same partition key.
    job_state.create_job(
        operation_id,
        session_id=session_id,
        reason=f"versioned {routing_mode} workflow: {workflow_id}",
        owner_sub=owner_sub,
    )

    return {
        **record,
        "status_manifest_uri": status_uri,
        "plan_uri": plan_uri,
        "deployment_id": deployment_id,
    }


def get_operation(operation_id: str, *, owner_sub: str = "") -> dict[str, Any]:
    """Read an operation and enforce ownership when an owner is supplied."""

    _require_enabled()
    try:
        response = _table().get_item(
            Key={"job_id": operation_id, "subtask_id": OPERATION_SUBTASK},
            ConsistentRead=True,
        )
    except Exception as error:
        raise OperationStateError(f"Could not read operation record: {error}") from error
    item = response.get("Item", {}) or {}
    if not item:
        return {}
    if owner_sub and item.get("owner_sub") != owner_sub:
        raise OperationOwnershipError("Operation is not owned by this caller")
    return item


def persist_resolved_plan(
    operation_id: str, *, owner_sub: str, plan: Mapping[str, Any]
) -> str:
    """Replace the initial plan stub with a validated deterministic plan.

    The runner calls this after classification and before invoking any
    optional specialist.  The same server-derived key is retained, so the
    operation manifest always points to one stable private plan URI.
    """

    record = get_operation(operation_id, owner_sub=owner_sub)
    if not record:
        raise OperationStateError(f"Operation does not exist: {operation_id}")
    if record.get("status") in {"completed", "failed", "cancelled"}:
        raise OperationStateError("Cannot change a terminal operation plan")
    if plan.get("schema_version") != "badgers-operation-plan/v1":
        raise OperationStateError("Plan must use badgers-operation-plan/v1")
    try:
        validate_resolved_plan(plan)
    except WorkflowPlanningError as error:
        raise OperationStateError(f"Resolved plan is invalid: {error}") from error
    if plan.get("operation_id") != operation_id:
        raise OperationStateError("Plan operation_id does not match operation")
    if plan.get("session_id") != record.get("session_id"):
        raise OperationStateError("Plan session_id does not match operation")
    if plan.get("workflow_id") != record.get("workflow_id"):
        raise OperationStateError("Plan workflow_id does not match operation")
    if plan.get("routing_mode") != record.get("routing_mode"):
        raise OperationStateError("Plan routing_mode does not match operation")
    if plan.get("config_release_id") != record.get("config_release_id"):
        raise OperationStateError("Plan config_release_id does not match operation")

    key = f"{operation_prefix(owner_sub, operation_id)}/plan/plan.json"
    plan_uri = _put_json(key, plan)
    try:
        _table().update_item(
            Key={"job_id": operation_id, "subtask_id": OPERATION_SUBTASK},
            UpdateExpression=(
                "SET plan_uri = :plan_uri, workflow_version = :workflow_version, "
                "routing_policy_id = :routing_policy_id, "
                "routing_policy_version = :routing_policy_version, "
                "updated_at = :updated, #ttl = :ttl"
            ),
            ExpressionAttributeNames={"#ttl": "ttl"},
            ExpressionAttributeValues={
                ":plan_uri": plan_uri,
                ":workflow_version": plan["workflow_version"],
                ":routing_policy_id": plan["routing_policy_id"],
                ":routing_policy_version": plan["routing_policy_version"],
                ":updated": utc_now(),
                ":ttl": _ttl(),
            },
        )
    except Exception as error:
        raise OperationStateError(f"Could not index resolved operation plan: {error}") from error
    return plan_uri


def update_operation_stage(
    operation_id: str,
    *,
    owner_sub: str,
    status: str,
    stage: str,
    execution_arn: str = "",
) -> dict[str, Any]:
    """Persist a non-terminal runner stage and refresh the private status object."""

    record = get_operation(operation_id, owner_sub=owner_sub)
    if not record:
        raise OperationStateError(f"Operation does not exist: {operation_id}")
    if record.get("status") in _TERMINAL_STATUSES:
        raise OperationStateError("Cannot transition a terminal operation")
    if status in _TERMINAL_STATUSES:
        raise OperationStateError("Use a terminal operation method for terminal states")
    if not status or not stage:
        raise OperationStateError("Operation status and stage are required")

    values: dict[str, Any] = {
        ":status": status,
        ":stage": stage,
        ":updated": utc_now(),
        ":ttl": _ttl(),
        ":owner": owner_sub,
    }
    expression = (
        "SET #status = :status, current_stage = :stage, updated_at = :updated, "
        "#ttl = :ttl"
    )
    names = {"#status": "status", "#ttl": "ttl"}
    active_status_values = {
        f":active_{index}": value
        for index, value in enumerate(sorted(_ACTIVE_STATUSES))
    }
    values.update(active_status_values)
    condition = (
        "attribute_exists(job_id) AND owner_sub = :owner AND "
        "#status IN (" + ", ".join(active_status_values) + ")"
    )
    if execution_arn:
        expression += ", workflow_execution_arn = :execution_arn"
        values[":execution_arn"] = execution_arn
    try:
        _table().update_item(
            Key={"job_id": operation_id, "subtask_id": OPERATION_SUBTASK},
            UpdateExpression=expression,
            ConditionExpression=condition,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )
    except Exception as error:
        if _is_conditional_conflict(error):
            raise OperationStateError(
                "Operation transition was rejected because its state changed or it is not owned"
            ) from error
        raise OperationStateError(f"Could not update operation stage: {error}") from error

    updated = dict(record)
    updated.update(
        {
            "status": status,
            "current_stage": stage,
            "updated_at": values[":updated"],
        }
    )
    if execution_arn:
        updated["workflow_execution_arn"] = execution_arn
    _put_json(
        f"{operation_prefix(owner_sub, operation_id)}/status.json",
        _status_payload(updated),
    )
    return updated


def cancel_operation(
    operation_id: str,
    *,
    owner_sub: str,
    reason: str = "Operation cancelled by authorized caller.",
    runtime_session_id: str = "",
    deployment_id: str = "",
) -> dict[str, Any]:
    """Persist a terminal cancelled manifest while retaining safe artifacts."""

    record = get_operation(operation_id, owner_sub=owner_sub)
    if not record:
        raise OperationStateError(f"Operation does not exist: {operation_id}")
    if record.get("status") == "cancelled":
        completion_uri = str(record.get("completion_manifest_uri") or "")
        if completion_uri:
            return build_operation_result(
                status="cancelled",
                operation_id=operation_id,
                session_id=str(record["session_id"]),
                workflow_id=str(record["workflow_id"]),
                routing_mode=str(record["routing_mode"]),
                config_release_id=str(record.get("config_release_id") or ""),
                source_document_s3_uri=str(record.get("source_document_s3_uri") or ""),
                plan_uri=str(record.get("plan_uri") or ""),
                completion_manifest_uri=completion_uri,
                error={"code": "CANCELLED", "message": reason},
            )
    if record.get("status") in _TERMINAL_STATUSES:
        raise OperationStateError("Cannot cancel a terminal operation")
    classification_uri, correlated_uri, specialists, reports = _artifact_records(
        operation_id, owner_sub=owner_sub
    )
    key = f"{operation_prefix(owner_sub, operation_id)}/manifest/cancelled.json"
    manifest = build_operation_result(
        status="cancelled",
        operation_id=operation_id,
        session_id=str(record["session_id"]),
        runtime_session_id=runtime_session_id,
        trace_id=current_trace_id(),
        deployment_id=deployment_id,
        workflow_id=str(record["workflow_id"]),
        routing_mode=str(record["routing_mode"]),
        config_release_id=str(record.get("config_release_id") or ""),
        source_document_s3_uri=str(record.get("source_document_s3_uri") or ""),
        correlation_id=str(record.get("correlation_id") or ""),
        s3_classification_uri=classification_uri,
        s3_correlated_uri=correlated_uri,
        plan_uri=str(record.get("plan_uri") or ""),
        individual_specialists=specialists,
        report_artifacts=reports,
        completion_manifest_uri=_s3_uri(key),
        error={"code": "CANCELLED", "message": reason},
    )
    completion_uri = _put_json(key, manifest)
    _update_terminal_record(
        operation_id=operation_id,
        owner_sub=owner_sub,
        status="cancelled",
        completion_manifest_uri=completion_uri,
        error_code="CANCELLED",
    )
    _write_terminal_status(
        record=record,
        status="cancelled",
        completion_manifest_uri=completion_uri,
        error_code="CANCELLED",
    )
    return manifest


def _artifact_records(
    operation_id: str, *, owner_sub: str
) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Read specialist result pointers from existing job-state records."""

    classification_uri = ""
    correlated_uri = ""
    specialists: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for record in job_state.get_job_records(operation_id):
        if record.get("status") != "COMPLETE":
            continue
        specialist = str(record.get("specialist") or "")
        result_uri = str(record.get("result_s3_key") or "")
        if not specialist or not _is_operation_artifact_uri(
            result_uri, operation_id=operation_id, owner_sub=owner_sub
        ):
            if specialist and result_uri:
                logger.warning(
                    "Ignoring artifact pointer outside operation prefix "
                    "(operation=%s, specialist=%s)",
                    operation_id,
                    specialist,
                )
            continue
        artifact = {
            "specialist": specialist,
            "s3_uri": result_uri,
            "content_type": artifact_content_type(result_uri),
        }
        if specialist == "classify_pdf_content":
            classification_uri = result_uri
        elif specialist == "correlation_specialist":
            correlated_uri = result_uri
        elif specialist == "html_report_specialist":
            reports.append({"kind": "html_report", **artifact})
        else:
            specialists.append(artifact)
    return classification_uri, correlated_uri, specialists, reports


def _update_terminal_record(
    *,
    operation_id: str,
    owner_sub: str,
    status: str,
    completion_manifest_uri: str,
    error_code: str = "",
) -> None:
    try:
        values: dict[str, Any] = {
            ":status": status,
            ":uri": completion_manifest_uri,
            ":updated": utc_now(),
            ":ttl": _ttl(),
            ":owner": owner_sub,
        }
        expression = (
            "SET #status = :status, completion_manifest_uri = :uri, "
            "updated_at = :updated, #ttl = :ttl"
        )
        names = {"#status": "status", "#ttl": "ttl"}
        active_status_values = {
            f":active_{index}": value
            for index, value in enumerate(sorted(_ACTIVE_STATUSES))
        }
        values.update(active_status_values)
        condition = (
            "attribute_exists(job_id) AND owner_sub = :owner AND "
            "#status IN (" + ", ".join(active_status_values) + ")"
        )
        if error_code:
            expression += ", error_code = :error_code"
            values[":error_code"] = error_code
        _table().update_item(
            Key={"job_id": operation_id, "subtask_id": OPERATION_SUBTASK},
            UpdateExpression=expression,
            ConditionExpression=condition,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )
    except Exception as error:
        if _is_conditional_conflict(error):
            raise OperationStateError(
                "Operation terminal transition was rejected because it is already terminal or not owned"
            ) from error
        raise OperationStateError(f"Could not update terminal operation: {error}") from error


def _write_terminal_status(
    *,
    record: Mapping[str, Any],
    status: str,
    completion_manifest_uri: str,
    error_code: str = "",
) -> None:
    """Replace the small status document after a terminal state transition."""

    payload = dict(record)
    payload["status"] = status
    payload["completion_manifest_uri"] = completion_manifest_uri
    payload["updated_at"] = utc_now()
    if error_code:
        payload["error_code"] = error_code
    _put_json(
        f"{operation_prefix(str(record['owner_sub']), str(record['operation_id']))}/status.json",
        _status_payload(payload),
    )


def complete_operation(
    *,
    operation_id: str,
    owner_sub: str,
    runtime_session_id: str = "",
    deployment_id: str = "",
    summary: str = "",
) -> dict[str, Any]:
    """Persist and return the bounded completion manifest for an owned operation."""

    record = get_operation(operation_id, owner_sub=owner_sub)
    if not record:
        raise OperationStateError(f"Operation does not exist: {operation_id}")
    classification_uri, correlated_uri, specialists, reports = _artifact_records(
        operation_id, owner_sub=owner_sub
    )
    manifest = build_operation_result(
        status="completed",
        operation_id=operation_id,
        session_id=str(record["session_id"]),
        runtime_session_id=runtime_session_id,
        trace_id=current_trace_id(),
        deployment_id=deployment_id,
        workflow_id=str(record["workflow_id"]),
        routing_mode=str(record["routing_mode"]),
        config_release_id=str(record.get("config_release_id") or ""),
        source_document_s3_uri=str(record.get("source_document_s3_uri") or ""),
        correlation_id=str(record.get("correlation_id") or ""),
        s3_classification_uri=classification_uri,
        s3_correlated_uri=correlated_uri,
        plan_uri=str(record.get("plan_uri") or ""),
        individual_specialists=specialists,
        report_artifacts=reports,
        completion_manifest_uri=_s3_uri(
            f"{operation_prefix(owner_sub, operation_id)}/manifest/completion.json"
        ),
        summary=summary,
    )
    key = f"{operation_prefix(owner_sub, operation_id)}/manifest/completion.json"
    completion_uri = _put_json(key, manifest)
    _update_terminal_record(
        operation_id=operation_id,
        owner_sub=owner_sub,
        status="completed",
        completion_manifest_uri=completion_uri,
    )
    _write_terminal_status(
        record=record,
        status="completed",
        completion_manifest_uri=completion_uri,
    )
    return manifest


def fail_operation(
    *,
    operation_id: str,
    owner_sub: str,
    error_code: str,
    error_message: str,
    runtime_session_id: str = "",
    deployment_id: str = "",
) -> dict[str, Any]:
    """Persist a terminal failure pointer manifest without exposing a traceback."""

    record = get_operation(operation_id, owner_sub=owner_sub)
    if not record:
        raise OperationStateError(f"Operation does not exist: {operation_id}")
    classification_uri, correlated_uri, specialists, reports = _artifact_records(
        operation_id, owner_sub=owner_sub
    )
    manifest = build_operation_result(
        status="failed",
        operation_id=operation_id,
        session_id=str(record["session_id"]),
        runtime_session_id=runtime_session_id,
        trace_id=current_trace_id(),
        deployment_id=deployment_id,
        workflow_id=str(record["workflow_id"]),
        routing_mode=str(record["routing_mode"]),
        config_release_id=str(record.get("config_release_id") or ""),
        source_document_s3_uri=str(record.get("source_document_s3_uri") or ""),
        correlation_id=str(record.get("correlation_id") or ""),
        s3_classification_uri=classification_uri,
        s3_correlated_uri=correlated_uri,
        plan_uri=str(record.get("plan_uri") or ""),
        individual_specialists=specialists,
        report_artifacts=reports,
        completion_manifest_uri=_s3_uri(
            f"{operation_prefix(owner_sub, operation_id)}/manifest/failure.json"
        ),
        error={"code": error_code, "message": error_message},
    )
    key = f"{operation_prefix(owner_sub, operation_id)}/manifest/failure.json"
    completion_uri = _put_json(key, manifest)
    _update_terminal_record(
        operation_id=operation_id,
        owner_sub=owner_sub,
        status="failed",
        completion_manifest_uri=completion_uri,
        error_code=error_code,
    )
    _write_terminal_status(
        record=record,
        status="failed",
        completion_manifest_uri=completion_uri,
        error_code=error_code,
    )
    return manifest


def reset_clients_for_tests() -> None:
    """Clear boto client caches for isolated unit tests."""

    global _dynamodb, _s3
    _dynamodb = None
    _s3 = None
