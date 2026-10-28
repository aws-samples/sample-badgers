"""Private Step Functions stage handler for the BADGERS workflow runner.

The handler is deliberately not an API.  Its caller is the deployment-scoped
State Machine role; every action reloads the durable operation record and
uses only the workflow, policy, and specialist functions pinned by the
configuration release.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Mapping

import boto3

from foundation import job_state, operation_state, workflow_planner

logger = logging.getLogger()
logger.setLevel(getattr(logging, os.environ.get("LOGGING_LEVEL", "INFO").upper()))

_MAX_SOURCE_BYTES = int(os.environ.get("MAX_SOURCE_BYTES", str(200 * 1024 * 1024)))
_DEFAULT_SOURCE_TYPES = frozenset({"application/pdf", "image/png", "image/jpeg"})


class WorkflowRunnerError(RuntimeError):
    """Failure that must transition the durable operation to failed."""


class WorkflowCancelled(RuntimeError):
    """Raised internally when a prior cancellation has reached this stage."""


def _specialist_functions() -> dict[str, str]:
    try:
        value = json.loads(os.environ["SPECIALIST_FUNCTIONS_JSON"])
    except (KeyError, json.JSONDecodeError) as error:
        raise WorkflowRunnerError("Runner specialist function map is unavailable") from error
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(function, str)
        for name, function in value.items()
    ):
        raise WorkflowRunnerError("Runner specialist function map is invalid")
    return value


def _operation(event: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    operation_id = event.get("operation_id")
    owner_sub = event.get("owner_sub")
    if not isinstance(operation_id, str) or not operation_id.startswith("op_"):
        raise WorkflowRunnerError("operation_id is required")
    if not isinstance(owner_sub, str) or not owner_sub:
        raise WorkflowRunnerError("owner_sub is required")
    record = operation_state.get_operation(operation_id, owner_sub=owner_sub)
    if not record:
        raise WorkflowRunnerError("Operation does not exist")
    if record.get("status") == "cancelled":
        raise WorkflowCancelled("Operation was cancelled")
    if record.get("status") in {"completed", "failed"}:
        raise WorkflowRunnerError("Operation is already terminal")
    return operation_id, owner_sub, record


def _next_state(event: Mapping[str, Any], **updates: Any) -> dict[str, Any]:
    """Return compact Step Functions state, retaining no native artifact bodies."""

    state = {
        key: value
        for key, value in event.items()
        if key
        in {
            "operation_id",
            "owner_sub",
            "requested_capabilities",
            "audit_mode",
            "execution_arn",
            "images",
            "classification_facts",
            "classification_artifacts",
            "plan",
        }
    }
    state.update(updates)
    return state


def _split_s3_uri(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise WorkflowRunnerError("Source document must be an S3 object URI")
    bucket_and_key = uri[5:].split("/", 1)
    if len(bucket_and_key) != 2 or not bucket_and_key[0] or not bucket_and_key[1]:
        raise WorkflowRunnerError("Source document must identify an S3 object")
    return bucket_and_key[0], bucket_and_key[1]


def _require_operation_artifact(
    uri: str, *, operation_id: str, owner_sub: str
) -> None:
    """Require an existing artifact inside this operation's private prefix."""

    bucket, key = _split_s3_uri(uri)
    if bucket != os.environ.get("OUTPUT_BUCKET_NAME", ""):
        raise WorkflowRunnerError("Artifact is outside the configured output bucket")
    expected_prefix = f"{operation_state.operation_prefix(owner_sub, operation_id)}/"
    if not key.startswith(expected_prefix):
        raise WorkflowRunnerError("Artifact is outside the private operation prefix")
    try:
        boto3.client("s3").head_object(Bucket=bucket, Key=key)
    except Exception as error:
        raise WorkflowRunnerError("Recorded artifact is not readable") from error


def _expected_source_content_type(
    key: str, head: Mapping[str, Any], accepted_types: set[str]
) -> str:
    content_type = str(head.get("ContentType") or "").lower().split(";", 1)[0]
    if content_type in accepted_types:
        return content_type
    extension_types = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    for suffix, inferred in extension_types.items():
        if key.lower().endswith(suffix) and inferred in accepted_types:
            return inferred
    raise WorkflowRunnerError("Source document type is not accepted by this workflow")


def _validate_source(event: Mapping[str, Any]) -> dict[str, Any]:
    operation_id, owner_sub, record = _operation(event)
    operation_state.update_operation_stage(
        operation_id,
        owner_sub=owner_sub,
        status="validating",
        stage="validate_source",
        execution_arn=str(event.get("execution_arn") or ""),
    )
    source_uri = str(record.get("source_document_s3_uri") or "")
    bucket, key = _split_s3_uri(source_uri)
    expected_bucket = os.environ.get("SOURCE_BUCKET_NAME", "")
    if bucket != expected_bucket:
        raise WorkflowRunnerError("Source bucket is not approved for this deployment")
    expected_prefix = f"tenants/{operation_state.tenant_segment(owner_sub)}/"
    if not key.startswith(expected_prefix):
        raise WorkflowRunnerError("Source object is outside the caller tenant prefix")
    try:
        head = boto3.client("s3").head_object(Bucket=bucket, Key=key)
    except Exception as error:
        raise WorkflowRunnerError("Source document could not be read") from error
    content_length = int(head.get("ContentLength") or 0)
    if content_length <= 0 or content_length > _MAX_SOURCE_BYTES:
        raise WorkflowRunnerError("Source document size is outside workflow limits")
    try:
        workflow, _, _ = workflow_planner.load_release_workflow_catalog(
            os.environ.get("CONFIG_BUCKET", ""),
            str(record["workflow_id"]),
            tuple(sorted(_specialist_functions())),
        )
        accepted_types = set(
            workflow.get("input_contract", {}).get(
                "accepted_source_types", _DEFAULT_SOURCE_TYPES
            )
        )
    except (ValueError, KeyError, TypeError) as error:
        raise WorkflowRunnerError(f"Could not load the pinned workflow: {error}") from error
    return _next_state(
        event,
        source_content_type=_expected_source_content_type(key, head, accepted_types),
        source_size_bytes=content_length,
        audit_mode=bool(event.get("audit_mode", False)),
        execution_mode=workflow.get("execution_mode", "page_images"),
    )


def _invoke_specialist(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    function_name = _specialist_functions().get(name)
    if not function_name:
        raise WorkflowRunnerError(f"Specialist is not available in the pinned pack: {name}")
    try:
        response = boto3.client("lambda").invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )
        if response.get("FunctionError"):
            raise WorkflowRunnerError(f"Specialist '{name}' invocation failed")
        decoded = json.loads(response["Payload"].read().decode("utf-8"))
    except WorkflowRunnerError:
        raise
    except Exception as error:
        raise WorkflowRunnerError(f"Specialist '{name}' invocation failed") from error
    if not isinstance(decoded, dict) or int(decoded.get("statusCode", 500)) >= 300:
        raise WorkflowRunnerError(f"Specialist '{name}' returned an error")
    return decoded


def _response_body(response: Mapping[str, Any], specialist: str) -> dict[str, Any]:
    body = response.get("body")
    try:
        decoded = json.loads(body) if isinstance(body, str) else body
    except json.JSONDecodeError as error:
        raise WorkflowRunnerError(f"Specialist '{specialist}' returned invalid JSON") from error
    if not isinstance(decoded, dict) or decoded.get("success") is False:
        raise WorkflowRunnerError(f"Specialist '{specialist}' did not complete")
    return decoded


def _convert(event: Mapping[str, Any]) -> dict[str, Any]:
    operation_id, owner_sub, record = _operation(event)
    operation_state.update_operation_stage(
        operation_id, owner_sub=owner_sub, status="converting", stage="convert_source"
    )
    response = _invoke_specialist(
        "pdf_to_images_converter",
        {
            "document_path": record["source_document_s3_uri"],
            "session_id": operation_id,
            "job_id": operation_id,
            "doc_id": operation_id,
        },
    )
    body = _response_body(response, "pdf_to_images_converter")
    try:
        result = json.loads(body["result"])
        images = result["images"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise WorkflowRunnerError("Converter did not return page images") from error
    if not isinstance(images, list) or not images or not all(
        isinstance(image, str) and image.startswith("s3://") for image in images
    ):
        raise WorkflowRunnerError("Converter returned invalid page images")
    if len(images) > int(os.environ.get("MAX_WORKFLOW_PAGES", "500")):
        raise WorkflowRunnerError("Source document exceeds workflow page limit")
    for image in images:
        _require_operation_artifact(
            image, operation_id=operation_id, owner_sub=owner_sub
        )
    return _next_state(event, images=images)


def _classification_artifacts(operation_id: str, owner_sub: str) -> list[str]:
    artifacts = sorted(
        {
            str(record.get("result_s3_key"))
            for record in job_state.get_job_records(operation_id)
            if record.get("status") == "COMPLETE"
            and record.get("specialist") == "classify_pdf_content"
            and str(record.get("result_s3_key") or "").startswith("s3://")
        }
    )
    for artifact in artifacts:
        _require_operation_artifact(
            artifact, operation_id=operation_id, owner_sub=owner_sub
        )
    return artifacts


def _classification_facts(result: str) -> list[dict[str, Any]]:
    """Normalize the existing classifier's XML/text result deterministically."""

    native = result.split("<!-- S3_RESULT_URI:", 1)[0].strip()
    try:
        pages = json.loads(native)
    except json.JSONDecodeError as error:
        raise WorkflowRunnerError("Classifier returned an unreadable result") from error
    if not isinstance(pages, list) or not pages:
        raise WorkflowRunnerError("Classifier returned no page classifications")
    facts: list[dict[str, Any]] = []
    for page in pages:
        text = str(page.get("classification") or "").lower()
        match = re.search(r"<confidence>\s*(high|medium|low)\s*</confidence>", text)
        confidence = match.group(1) if match else "medium"
        content_type = "table_dominant" if "table" in text else "text"
        facts.append({"content_type": content_type, "confidence": confidence})
    return facts


def _classify(event: Mapping[str, Any]) -> dict[str, Any]:
    operation_id, owner_sub, _record = _operation(event)
    images = event.get("images")
    if not isinstance(images, list) or not images:
        raise WorkflowRunnerError("No converted page images are available")
    operation_state.update_operation_stage(
        operation_id, owner_sub=owner_sub, status="classifying", stage="classify_pages"
    )
    response = _invoke_specialist(
        "classify_pdf_content",
        {
            "image_paths": json.dumps(images),
            "session_id": operation_id,
            "job_id": operation_id,
            "doc_id": operation_id,
            "audit_mode": bool(event.get("audit_mode", False)),
        },
    )
    body = _response_body(response, "classify_pdf_content")
    facts = _classification_facts(str(body.get("result") or ""))
    artifacts = _classification_artifacts(operation_id, owner_sub)
    if not artifacts:
        raise WorkflowRunnerError("Classifier completed without a persisted artifact")
    return _next_state(event, classification_facts=facts, classification_artifacts=artifacts)


def _plan(event: Mapping[str, Any]) -> dict[str, Any]:
    operation_id, owner_sub, record = _operation(event)
    operation_state.update_operation_stage(
        operation_id, owner_sub=owner_sub, status="planning", stage="resolve_plan"
    )
    functions = _specialist_functions()
    try:
        workflow, policy, capability_index = workflow_planner.load_release_workflow_catalog(
            os.environ.get("CONFIG_BUCKET", ""),
            str(record["workflow_id"]),
            tuple(sorted(functions)),
        )
        plan = workflow_planner.resolve_plan(
            operation_id=operation_id,
            session_id=str(record["session_id"]),
            config_release_id=str(record["config_release_id"]),
            workflow=workflow,
            policy=policy,
            capability_index=capability_index,
            routing_mode=str(record["routing_mode"]),
            classifier_facts=event.get("classification_facts", []),
            classification_artifacts=event.get("classification_artifacts", []),
            requested_capabilities=event.get("requested_capabilities", []),
        )
    except (ValueError, KeyError, TypeError) as error:
        raise WorkflowRunnerError(f"Could not resolve approved workflow plan: {error}") from error
    operation_state.persist_resolved_plan(operation_id, owner_sub=owner_sub, plan=plan)
    return _next_state(event, plan=plan)


def _native_process(event: Mapping[str, Any]) -> dict[str, Any]:
    """Run one declared native processor and retain its result privately."""

    operation_id, owner_sub, record = _operation(event)
    plan = event.get("plan")
    if not isinstance(plan, dict):
        raise WorkflowRunnerError("Native workflow has no resolved plan")
    specialists = plan.get("resolved_specialists")
    if not isinstance(specialists, list) or len(specialists) != 1:
        raise WorkflowRunnerError("Native workflow must resolve exactly one processor")
    specialist = specialists[0]
    if specialist not in {"docx_processor", "pptx_processor", "native_pdf_processor"}:
        raise WorkflowRunnerError("Native workflow resolved an unsupported processor")
    operation_state.update_operation_stage(
        operation_id, owner_sub=owner_sub, status="running", stage="native_document"
    )
    response = _response_body(
        _invoke_specialist(
            specialist,
            {
                "document_path": record["source_document_s3_uri"],
                "session_id": operation_id,
                "job_id": operation_id,
                "doc_id": operation_id,
                # PPTX uses explicit plan/render modes. The other native
                # processors ignore this field, so every native processor gets
                # the same bounded invocation contract.
                "operation": "plan",
            },
        ),
        specialist,
    )
    key = (
        f"{operation_state.operation_prefix(owner_sub, operation_id)}/"
        f"specialists/{specialist}/native-result.json"
    )
    try:
        boto3.client("s3").put_object(
            Bucket=os.environ.get("OUTPUT_BUCKET_NAME", ""),
            Key=key,
            Body=json.dumps(response, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as error:
        raise WorkflowRunnerError("Could not retain native processor result") from error
    result_uri = f"s3://{os.environ.get('OUTPUT_BUCKET_NAME', '')}/{key}"
    job_state.mark_complete(
        operation_id,
        job_state.subtask_id(specialist, str(record["source_document_s3_uri"])),
        result_uri,
    )
    return _next_state(event, native_result_uri=result_uri)


def _run_page_specialists(event: Mapping[str, Any]) -> dict[str, Any]:
    operation_id, owner_sub, _record = _operation(event)
    plan = event.get("plan")
    image = event.get("image")
    if not isinstance(plan, dict) or not isinstance(image, str) or not image.startswith("s3://"):
        raise WorkflowRunnerError("Resolved plan and page image are required")
    excluded = {
        "pdf_to_images_converter",
        "classify_pdf_content",
        "correlation_specialist",
    }
    specialists = [
        name for name in plan.get("resolved_specialists", []) if name not in excluded
    ]
    if not specialists:
        raise WorkflowRunnerError("Resolved plan has no page specialists")
    for specialist in specialists:
        _response_body(
            _invoke_specialist(
                specialist,
                {
                    "image_path": image,
                    "session_id": operation_id,
                    "job_id": operation_id,
                    "doc_id": operation_id,
                    "audit_mode": bool(event.get("audit_mode", False)),
                },
            ),
            specialist,
        )
    return _next_state(event)


def _begin_specialist_work(event: Mapping[str, Any]) -> dict[str, Any]:
    operation_id, owner_sub, _record = _operation(event)
    operation_state.update_operation_stage(
        operation_id, owner_sub=owner_sub, status="running", stage="run_specialists"
    )
    return _next_state(event)


def _completed_artifacts(
    operation_id: str, owner_sub: str, image_path: str
) -> list[dict[str, str]]:
    image_id = job_state.image_identifier(image_path)
    artifacts = []
    for record in job_state.get_job_records(operation_id):
        if (
            record.get("status") == "COMPLETE"
            and record.get("image_identifier") == image_id
            and str(record.get("result_s3_key") or "").startswith("s3://")
        ):
            uri = str(record["result_s3_key"])
            _require_operation_artifact(
                uri, operation_id=operation_id, owner_sub=owner_sub
            )
            artifacts.append(
                {"specialist": str(record.get("specialist") or ""), "s3_uri": uri}
            )
    return artifacts


def _correlate_page(event: Mapping[str, Any]) -> dict[str, Any]:
    operation_id, owner_sub, _record = _operation(event)
    image = event.get("image")
    if not isinstance(image, str) or not image.startswith("s3://"):
        raise WorkflowRunnerError("Page image is required for correlation")
    artifacts = _completed_artifacts(operation_id, owner_sub, image)
    full_text = next(
        (artifact for artifact in artifacts if artifact["specialist"] == "full_text_specialist"),
        None,
    )
    enrichments = [
        {
            "specialist_name": artifact["specialist"],
            "s3_uri": artifact["s3_uri"],
        }
        for artifact in artifacts
        if artifact["specialist"]
        not in {"full_text_specialist", "classify_pdf_content", "pdf_to_images_converter"}
    ]
    if not full_text or not enrichments:
        raise WorkflowRunnerError(
            "Correlation requires completed full-text and enrichment artifacts"
        )
    page_index = event.get("page_index", 0)
    try:
        page_number = str(int(page_index) + 1)
    except (TypeError, ValueError) as error:
        raise WorkflowRunnerError("Page index is invalid") from error
    _response_body(
        _invoke_specialist(
            "correlation_specialist",
            {
                "full_text_uri": full_text["s3_uri"],
                "specialist_uris": enrichments,
                "page_number": page_number,
                "source_image_path": image,
                "session_id": operation_id,
                "job_id": operation_id,
                "doc_id": operation_id,
            },
        ),
        "correlation_specialist",
    )
    return _next_state(event)


def _begin_correlation(event: Mapping[str, Any]) -> dict[str, Any]:
    operation_id, owner_sub, _record = _operation(event)
    operation_state.update_operation_stage(
        operation_id, owner_sub=owner_sub, status="correlating", stage="correlate"
    )
    return _next_state(event)


def _finalize(event: Mapping[str, Any]) -> dict[str, Any]:
    operation_id, owner_sub, _record = _operation(event)
    operation_state.update_operation_stage(
        operation_id, owner_sub=owner_sub, status="finalizing", stage="write_completion_manifest"
    )
    native_result_uri = str(event.get("native_result_uri") or "")
    if native_result_uri:
        _require_operation_artifact(
            native_result_uri, operation_id=operation_id, owner_sub=owner_sub
        )
        result = operation_state.complete_operation(
            operation_id=operation_id,
            owner_sub=owner_sub,
            deployment_id=os.environ.get("DEPLOYMENT_ID", ""),
            summary="Native document processing completed; see the processor artifact.",
        )
        return _next_state(event, result=result)

    correlated_uris = [
        str(record["result_s3_key"])
        for record in job_state.get_job_records(operation_id)
        if record.get("status") == "COMPLETE"
        and record.get("specialist") == "correlation_specialist"
        and str(record.get("result_s3_key") or "").startswith("s3://")
    ]
    if not correlated_uris:
        raise WorkflowRunnerError("Correlation did not produce a persisted artifact")
    for artifact in correlated_uris:
        _require_operation_artifact(
            artifact, operation_id=operation_id, owner_sub=owner_sub
        )
    result = operation_state.complete_operation(
        operation_id=operation_id,
        owner_sub=owner_sub,
        deployment_id=os.environ.get("DEPLOYMENT_ID", ""),
    )
    return _next_state(event, result=result)


def _fail(event: Mapping[str, Any]) -> dict[str, Any]:
    operation_id, owner_sub, _record = _operation(event)
    failure = event.get("failure", {})
    if not isinstance(failure, Mapping):
        failure = {}
    result = operation_state.fail_operation(
        operation_id=operation_id,
        owner_sub=owner_sub,
        error_code=str(failure.get("Error") or "WORKFLOW_STAGE_FAILED")[:128],
        error_message=str(failure.get("Cause") or "Workflow stage failed.")[:1024],
        deployment_id=os.environ.get("DEPLOYMENT_ID", ""),
    )
    return _next_state(event, result=result)


def _cancel(event: Mapping[str, Any]) -> dict[str, Any]:
    operation_id, owner_sub, _record = _operation(event)
    result = operation_state.cancel_operation(
        operation_id=operation_id,
        owner_sub=owner_sub,
        reason=str(event.get("reason") or "Operation cancelled by authorized caller.")[:1024],
        deployment_id=os.environ.get("DEPLOYMENT_ID", ""),
    )
    return _next_state(event, result=result, cancelled=True)


_ACTIONS = {
    "validate": _validate_source,
    "convert": _convert,
    "classify": _classify,
    "plan": _plan,
    "native_process": _native_process,
    "begin_specialist_work": _begin_specialist_work,
    "run_page_specialists": _run_page_specialists,
    "begin_correlation": _begin_correlation,
    "correlate_page": _correlate_page,
    "finalize": _finalize,
    "fail": _fail,
    "cancel": _cancel,
}


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """Execute one authenticated, durable workflow stage."""

    if not isinstance(event, dict):
        raise WorkflowRunnerError("Workflow state must be an object")
    state = event.get("state")
    if isinstance(state, dict):
        event = {
            **state,
            "action": event.get("action"),
            "execution_arn": event.get("execution_arn"),
        }
    action = event.get("action")
    if not isinstance(action, str) or action not in _ACTIONS:
        raise WorkflowRunnerError("Workflow action is not recognized")
    try:
        return _ACTIONS[action](event)
    except WorkflowCancelled:
        return _next_state(event, cancelled=True)
