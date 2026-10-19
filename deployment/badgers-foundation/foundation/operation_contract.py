"""Versioned contracts for BADGERS asynchronous document-analysis operations.

The runtime and a future Gateway workflow adapter use this small stdlib-only
module to agree on request validation and completion-manifest shape.  Native
specialist artifacts remain in S3; this contract deliberately contains pointers
and bounded summaries rather than artifact bodies.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

AGENT_REQUEST_SCHEMA_VERSION = "badgers-agent-request/v1"
OPERATION_RESULT_SCHEMA_VERSION = "badgers-operation-result/v1"
OPERATION_RECORD_SCHEMA_VERSION = "badgers-operation-record/v1"

ROUTING_MODES = frozenset({"auto", "manual"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_WORKFLOW_ID_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_S3_URI_RE = re.compile(r"^s3://[^/\s]+/.+$")
_MAX_SUMMARY_CHARS = 2_000


class OperationContractError(ValueError):
    """Raised when a versioned BADGERS operation contract is invalid."""


@dataclass(frozen=True)
class AgentRequest:
    """Normalized request accepted by the Headless Agent API."""

    instruction: str
    session_id: str
    document_uri: str
    routing_mode: str
    workflow_id: str
    requested_capabilities: tuple[str, ...]
    audit_mode: bool
    correlation_id: str
    versioned: bool


def new_operation_id() -> str:
    """Mint an opaque, server-generated operation identifier."""

    return f"op_{uuid.uuid4().hex}"


def utc_now() -> str:
    """Return a stable UTC timestamp representation for persisted contracts."""

    return datetime.now(timezone.utc).isoformat()


def bounded_summary(value: str, *, limit: int = _MAX_SUMMARY_CHARS) -> str:
    """Keep conversational completion text bounded without changing artifacts."""

    normalized = " ".join(str(value or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _require_string(
    payload: Mapping[str, Any],
    name: str,
    *,
    minimum: int = 1,
    maximum: int = 8_192,
) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise OperationContractError(f"'{name}' must be a string")
    value = value.strip()
    if not (minimum <= len(value) <= maximum):
        raise OperationContractError(
            f"'{name}' must contain between {minimum} and {maximum} characters"
        )
    return value


def _validate_s3_uri(value: str, name: str) -> str:
    if not _S3_URI_RE.fullmatch(value):
        raise OperationContractError(f"'{name}' must be an S3 object URI")
    return value


def _validate_workflow_id(value: str) -> str:
    if not _WORKFLOW_ID_RE.fullmatch(value):
        raise OperationContractError(
            "'workflow_id' must use lowercase letters, digits, and hyphens"
        )
    return value


def _validate_capabilities(value: Any) -> tuple[str, ...]:
    """Validate caller-selected capability tags, never specialist identifiers."""

    if not isinstance(value, list):
        raise OperationContractError("'requested_capabilities' must be an array")
    if len(value) > 32:
        raise OperationContractError("'requested_capabilities' may contain at most 32 entries")
    capabilities: list[str] = []
    capability_re = re.compile(r"^[a-z][a-z0-9_-]{0,127}$")
    for capability in value:
        if not isinstance(capability, str) or not capability_re.fullmatch(capability):
            raise OperationContractError(
                "requested capabilities must use lowercase letters, digits, underscores, and hyphens"
            )
        capabilities.append(capability)
    if len(capabilities) != len(set(capabilities)):
        raise OperationContractError("'requested_capabilities' must not contain duplicates")
    return tuple(capabilities)


def _validate_v1_agent_request(payload: Mapping[str, Any]) -> AgentRequest:
    allowed = {
        "schema_version",
        "session_id",
        "document_uri",
        "instruction",
        "workflow_id",
        "routing_mode",
        "requested_capabilities",
        "audit_mode",
        "caller_context",
    }
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise OperationContractError(
            "Unsupported fields in versioned request: " + ", ".join(unexpected)
        )

    if payload.get("schema_version") != AGENT_REQUEST_SCHEMA_VERSION:
        raise OperationContractError(
            f"'schema_version' must be '{AGENT_REQUEST_SCHEMA_VERSION}'"
        )

    session_id = _require_string(payload, "session_id", maximum=256)
    document_uri = _validate_s3_uri(
        _require_string(payload, "document_uri", maximum=2_048), "document_uri"
    )
    instruction = _require_string(payload, "instruction", maximum=8_192)
    workflow_id = _validate_workflow_id(
        str(payload.get("workflow_id") or "standard-document-analysis")
    )
    routing_mode = str(payload.get("routing_mode") or "auto")
    if routing_mode not in ROUTING_MODES:
        raise OperationContractError(
            "'routing_mode' must be one of: " + ", ".join(sorted(ROUTING_MODES))
        )
    requested_capabilities = _validate_capabilities(
        payload.get("requested_capabilities", [])
    )
    if requested_capabilities and routing_mode != "manual":
        raise OperationContractError(
            "'requested_capabilities' is allowed only when routing_mode is 'manual'"
        )

    audit_mode = payload.get("audit_mode", False)
    if not isinstance(audit_mode, bool):
        raise OperationContractError("'audit_mode' must be a boolean")

    caller_context = payload.get("caller_context", {})
    if not isinstance(caller_context, Mapping):
        raise OperationContractError("'caller_context' must be an object")
    unexpected_context = sorted(set(caller_context) - {"correlation_id"})
    if unexpected_context:
        raise OperationContractError(
            "Unsupported caller_context fields: " + ", ".join(unexpected_context)
        )
    correlation_id = str(caller_context.get("correlation_id") or "").strip()
    if len(correlation_id) > 256:
        raise OperationContractError("'caller_context.correlation_id' is too long")

    return AgentRequest(
        instruction=instruction,
        session_id=session_id,
        document_uri=document_uri,
        routing_mode=routing_mode,
        workflow_id=workflow_id,
        requested_capabilities=requested_capabilities,
        audit_mode=audit_mode,
        correlation_id=correlation_id,
        versioned=True,
    )


def normalize_agent_request(payload: Mapping[str, Any]) -> AgentRequest:
    """Validate v1 requests and retain compatibility with existing callers.

    Requests without ``schema_version`` use the existing ``prompt``/``doc_id``
    shape.  They are intentionally not promoted to v1 because they may lack a
    source URI, which the durable Gateway-compatible contract requires.
    """

    if payload.get("schema_version") is not None:
        return _validate_v1_agent_request(payload)

    prompt = payload.get("prompt", "Hello!")
    if not isinstance(prompt, str):
        raise OperationContractError("'prompt' must be a string")
    session_id = str(payload.get("session_id") or f"session_{uuid.uuid4().hex}")
    document_uri = str(payload.get("document_uri") or "").strip()
    if document_uri:
        _validate_s3_uri(document_uri, "document_uri")
    return AgentRequest(
        instruction=prompt,
        session_id=session_id,
        document_uri=document_uri,
        routing_mode="auto",
        workflow_id="legacy-agent-workflow",
        requested_capabilities=(),
        audit_mode=bool(payload.get("audit_mode", False)),
        correlation_id="",
        versioned=False,
    )


def artifact_content_type(s3_uri: str) -> str:
    """Infer a conservative content type for a known result artifact URI."""

    path = s3_uri.lower().split("?", 1)[0]
    if path.endswith(".xml"):
        return "application/xml"
    if path.endswith(".json"):
        return "application/json"
    if path.endswith(".html") or path.endswith(".htm"):
        return "text/html"
    if path.endswith(".pdf"):
        return "application/pdf"
    if path.endswith(".png"):
        return "image/png"
    if path.endswith(".jpg") or path.endswith(".jpeg"):
        return "image/jpeg"
    if path.endswith(".tif") or path.endswith(".tiff"):
        return "image/tiff"
    if path.endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"


def build_operation_result(
    *,
    status: str,
    operation_id: str,
    session_id: str,
    deployment_id: str = "",
    workflow_id: str = "legacy-agent-workflow",
    routing_mode: str = "auto",
    config_release_id: str = "",
    source_document_s3_uri: str = "",
    runtime_session_id: str = "",
    trace_id: str = "",
    correlation_id: str = "",
    s3_classification_uri: str = "",
    s3_correlated_uri: str = "",
    plan_uri: str = "",
    individual_specialists: list[dict[str, Any]] | None = None,
    report_artifacts: list[dict[str, Any]] | None = None,
    completion_manifest_uri: str = "",
    summary: str = "",
    error: Mapping[str, str] | None = None,
    completed_at: str = "",
) -> dict[str, Any]:
    """Build a bounded, JSON-serializable completion pointer manifest."""

    if status not in TERMINAL_STATUSES:
        raise OperationContractError(
            "'status' must be one of: " + ", ".join(sorted(TERMINAL_STATUSES))
        )
    if not operation_id.startswith("op_"):
        raise OperationContractError("'operation_id' must be server-generated")
    if not session_id:
        raise OperationContractError("'session_id' is required")
    if routing_mode not in ROUTING_MODES:
        raise OperationContractError("Invalid routing mode for result manifest")
    _validate_workflow_id(workflow_id)

    result: dict[str, Any] = {
        "schema_version": OPERATION_RESULT_SCHEMA_VERSION,
        "status": status,
        "operation_id": operation_id,
        "session_id": session_id,
        "workflow_id": workflow_id,
        "routing_mode": routing_mode,
        "individual_specialists": individual_specialists or [],
        "report_artifacts": report_artifacts or [],
        "completed_at": completed_at or utc_now(),
    }
    for key, value in (
        ("runtime_session_id", runtime_session_id),
        ("trace_id", trace_id),
        ("deployment_id", deployment_id),
        ("config_release_id", config_release_id),
        ("correlation_id", correlation_id),
        ("source_document_s3_uri", source_document_s3_uri),
        ("s3_classification_uri", s3_classification_uri),
        ("s3_correlated_uri", s3_correlated_uri),
        ("plan_uri", plan_uri),
        ("completion_manifest_uri", completion_manifest_uri),
    ):
        if value:
            result[key] = value
    if source_document_s3_uri.lower().endswith(".pdf"):
        result["original_pdf_s3_uri"] = source_document_s3_uri
    if summary:
        result["summary"] = bounded_summary(summary)
    if error:
        result["error"] = {
            "code": bounded_summary(str(error.get("code", "OPERATION_FAILED")), limit=128),
            "message": bounded_summary(str(error.get("message", "")), limit=1_024),
        }
    return result


def operation_record(
    *,
    operation_id: str,
    session_id: str,
    owner_sub: str,
    status: str,
    workflow_id: str,
    routing_mode: str,
    source_document_s3_uri: str = "",
    config_release_id: str = "",
    correlation_id: str = "",
    principal_id: str = "",
    identity_provider: str = "",
) -> dict[str, Any]:
    """Build the durable operation-index row without provider-specific keys."""

    if not owner_sub:
        raise OperationContractError("'owner_sub' is required for operation ownership")
    record = {
        "schema_version": OPERATION_RECORD_SCHEMA_VERSION,
        "operation_id": operation_id,
        "session_id": session_id,
        "owner_sub": owner_sub,
        "status": status,
        "workflow_id": workflow_id,
        "routing_mode": routing_mode,
        "started_at": utc_now(),
    }
    for key, value in (
        ("source_document_s3_uri", source_document_s3_uri),
        ("config_release_id", config_release_id),
        ("correlation_id", correlation_id),
        ("principal_id", principal_id),
        ("identity_provider", identity_provider),
    ):
        if value:
            record[key] = value
    return record


def as_request_dict(request: AgentRequest) -> dict[str, Any]:
    """Expose a normalized request for logging/tests without implementation state."""

    return asdict(request)
