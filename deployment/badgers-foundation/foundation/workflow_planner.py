"""Validate immutable workflow configuration and resolve deterministic plans.

Workflow and routing-policy files are declarative release artifacts.  This
module intentionally has no model invocation, Lambda ARN, or prompt selection:
it maps trusted classifier facts and approved capability tags to the selected
specialists in a pinned profile.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9_-]{0,127}$")
_WORKFLOW_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_S3_URI_RE = re.compile(r"^s3://[^/\s]+/.+$")
_CONFIDENCE_NAMES = {"high": 0.9, "medium": 0.6, "low": 0.3}


class WorkflowConfigurationError(ValueError):
    """Raised when a release workflow catalog is invalid."""


class WorkflowPlanningError(ValueError):
    """Raised when a caller's approved workflow cannot produce a safe plan."""


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise WorkflowConfigurationError(f"{description} is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise WorkflowConfigurationError(f"{description} is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise WorkflowConfigurationError(f"{description} must be a JSON object: {path}")
    return value


def _require_string(
    value: Mapping[str, Any], name: str, *, pattern: re.Pattern[str] | None = None
) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result:
        raise WorkflowConfigurationError(f"'{name}' must be a non-empty string")
    if pattern and not pattern.fullmatch(result):
        raise WorkflowConfigurationError(f"'{name}' has an invalid value: {result}")
    return result


def _capability_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise WorkflowConfigurationError(f"'{field}' must be an array")
    if not all(isinstance(item, str) and _CAPABILITY_RE.fullmatch(item) for item in value):
        raise WorkflowConfigurationError(
            f"'{field}' entries must be lowercase capability tags"
        )
    if len(value) != len(set(value)):
        raise WorkflowConfigurationError(f"'{field}' must not contain duplicates")
    return tuple(value)


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise WorkflowConfigurationError(f"'{field}' must be a non-empty string array")
    if len(value) != len(set(value)):
        raise WorkflowConfigurationError(f"'{field}' must not contain duplicates")
    return tuple(value)


def validate_workflow_definition(workflow: Mapping[str, Any]) -> None:
    """Validate the executable-policy boundary of one workflow definition."""

    allowed = {
        "schema_version",
        "workflow_id",
        "version",
        "display_name",
        "execution_mode",
        "supported_routing_modes",
        "input_contract",
        "invariants",
        "invariant_capabilities",
        "routing_policy_id",
        "allowed_capabilities",
        "manual_capability_selection_allowed",
        "result_contract",
    }
    unexpected = sorted(set(workflow) - allowed)
    if unexpected:
        raise WorkflowConfigurationError(
            "Workflow contains unsupported fields: " + ", ".join(unexpected)
        )
    if workflow.get("schema_version") != 1:
        raise WorkflowConfigurationError("Workflow 'schema_version' must equal 1")
    _require_string(workflow, "workflow_id", pattern=_WORKFLOW_RE)
    _require_string(workflow, "version")
    if workflow.get("execution_mode", "page_images") not in {
        "page_images",
        "native_document",
    }:
        raise WorkflowConfigurationError(
            "Workflow execution_mode must be 'page_images' or 'native_document'"
        )
    routing_modes = _string_list(workflow.get("supported_routing_modes"), "supported_routing_modes")
    if set(routing_modes) - {"auto", "manual"}:
        raise WorkflowConfigurationError("Workflow has an unsupported routing mode")
    invariants = _string_list(workflow.get("invariants"), "invariants")
    invariant_capabilities = workflow.get("invariant_capabilities")
    if not isinstance(invariant_capabilities, dict):
        raise WorkflowConfigurationError("'invariant_capabilities' must be an object")
    unknown_steps = sorted(set(invariant_capabilities) - set(invariants))
    if unknown_steps:
        raise WorkflowConfigurationError(
            "invariant_capabilities references unknown invariants: "
            + ", ".join(unknown_steps)
        )
    if not all(
        isinstance(step, str)
        and isinstance(capability, str)
        and _CAPABILITY_RE.fullmatch(capability)
        for step, capability in invariant_capabilities.items()
    ):
        raise WorkflowConfigurationError(
            "'invariant_capabilities' values must be lowercase capability tags"
        )
    _require_string(workflow, "routing_policy_id", pattern=_WORKFLOW_RE)
    allowed_capabilities = _capability_list(
        workflow.get("allowed_capabilities"), "allowed_capabilities"
    )
    if not set(invariant_capabilities.values()).issubset(allowed_capabilities):
        raise WorkflowConfigurationError(
            "Every invariant capability must appear in 'allowed_capabilities'"
        )
    if workflow.get("result_contract") != "badgers-operation-result/v1":
        raise WorkflowConfigurationError(
            "Workflow 'result_contract' must be 'badgers-operation-result/v1'"
        )
    manual_selection = workflow.get("manual_capability_selection_allowed", False)
    if not isinstance(manual_selection, bool):
        raise WorkflowConfigurationError(
            "'manual_capability_selection_allowed' must be a boolean"
        )
    input_contract = workflow.get("input_contract")
    if input_contract is not None:
        if not isinstance(input_contract, dict):
            raise WorkflowConfigurationError("'input_contract' must be an object")
        if set(input_contract) - {"accepted_source_types", "requires_document_uri"}:
            raise WorkflowConfigurationError("'input_contract' contains unsupported fields")
        _string_list(input_contract.get("accepted_source_types"), "accepted_source_types")
        if input_contract.get("requires_document_uri") is not True:
            raise WorkflowConfigurationError(
                "'input_contract.requires_document_uri' must be true"
            )


def validate_routing_policy(policy: Mapping[str, Any]) -> None:
    """Validate a routing policy without evaluating classifier output."""

    allowed = {
        "schema_version",
        "policy_id",
        "policy_version",
        "default_workflow",
        "limits",
        "required_capabilities",
        "routes",
        "low_confidence",
    }
    unexpected = sorted(set(policy) - allowed)
    if unexpected:
        raise WorkflowConfigurationError(
            "Routing policy contains unsupported fields: " + ", ".join(unexpected)
        )
    if policy.get("schema_version") != 1:
        raise WorkflowConfigurationError("Routing policy 'schema_version' must equal 1")
    _require_string(policy, "policy_id", pattern=_WORKFLOW_RE)
    _require_string(policy, "policy_version")
    _require_string(policy, "default_workflow", pattern=_WORKFLOW_RE)
    limits = policy.get("limits")
    if not isinstance(limits, dict) or set(limits) != {
        "max_specialized_tools_per_page",
        "max_pages",
    }:
        raise WorkflowConfigurationError("Routing policy 'limits' is incomplete")
    if not all(isinstance(limits[key], int) and limits[key] >= 0 for key in limits):
        raise WorkflowConfigurationError("Routing policy limits must be non-negative integers")
    if limits["max_pages"] < 1:
        raise WorkflowConfigurationError("Routing policy 'max_pages' must be at least one")
    _capability_list(policy.get("required_capabilities"), "required_capabilities")
    routes = policy.get("routes")
    if not isinstance(routes, list):
        raise WorkflowConfigurationError("'routes' must be an array")
    for route in routes:
        if not isinstance(route, dict) or set(route) != {"when", "add_capabilities"}:
            raise WorkflowConfigurationError("Each route must contain only 'when' and 'add_capabilities'")
        when = route["when"]
        if not isinstance(when, dict) or set(when) != {
            "content_type",
            "minimum_confidence",
        }:
            raise WorkflowConfigurationError("Each route 'when' condition is incomplete")
        if not isinstance(when["content_type"], str) or not when["content_type"]:
            raise WorkflowConfigurationError("Route content_type must be a non-empty string")
        confidence = when["minimum_confidence"]
        if not isinstance(confidence, (float, int)) or not 0 <= confidence <= 1:
            raise WorkflowConfigurationError(
                "Route minimum_confidence must be a number between zero and one"
            )
        _capability_list(route["add_capabilities"], "route.add_capabilities")
    low_confidence = policy.get("low_confidence")
    if not isinstance(low_confidence, dict) or set(low_confidence) != {
        "below",
        "action",
        "capabilities",
    }:
        raise WorkflowConfigurationError("'low_confidence' is incomplete")
    threshold = low_confidence["below"]
    if not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
        raise WorkflowConfigurationError(
            "low_confidence.below must be a number between zero and one"
        )
    if low_confidence["action"] not in {"add_capabilities", "fail"}:
        raise WorkflowConfigurationError(
            "low_confidence.action must be 'add_capabilities' or 'fail'"
        )
    _capability_list(low_confidence["capabilities"], "low_confidence.capabilities")
    if low_confidence["action"] == "fail" and low_confidence["capabilities"]:
        raise WorkflowConfigurationError(
            "low_confidence.capabilities must be empty when action is 'fail'"
        )


def _validate_specialist_metadata(
    metadata: Mapping[str, Any], selected_specialists: Sequence[str]
) -> dict[str, str]:
    if metadata.get("schema_version") != 1 or not isinstance(
        metadata.get("specialists"), dict
    ):
        raise WorkflowConfigurationError("Specialist capability metadata is invalid")
    declared = metadata["specialists"]
    missing = sorted(set(selected_specialists) - set(declared))
    if missing:
        raise WorkflowConfigurationError(
            "Selected specialists have no capability metadata: " + ", ".join(missing)
        )
    capability_index: dict[str, str] = {}
    for specialist in selected_specialists:
        value = declared[specialist]
        if not isinstance(value, dict) or set(value) != {"capabilities"}:
            raise WorkflowConfigurationError(
                f"Specialist metadata for '{specialist}' must contain only capabilities"
            )
        for capability in _capability_list(
            value["capabilities"], f"specialist '{specialist}' capabilities"
        ):
            prior = capability_index.get(capability)
            if prior and prior != specialist:
                raise WorkflowConfigurationError(
                    f"Capability '{capability}' maps to both '{prior}' and '{specialist}'"
                )
            capability_index[capability] = specialist
    return capability_index


def load_workflow_catalog(
    source_root: Path, selected_specialists: Sequence[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    """Load and validate a filesystem catalog before it becomes a release."""

    workflow_directory = source_root / "workflows"
    policy_directory = source_root / "routing_policies"
    workflow_paths = sorted(workflow_directory.glob("*.json"))
    policy_paths = sorted(policy_directory.glob("*.json"))
    if not workflow_paths or not policy_paths:
        raise WorkflowConfigurationError(
            "At least one workflow and one routing policy must be present"
        )

    workflows: dict[str, dict[str, Any]] = {}
    for path in workflow_paths:
        workflow = _load_object(path, "Workflow definition")
        validate_workflow_definition(workflow)
        workflow_id = str(workflow["workflow_id"])
        if path.stem != workflow_id:
            raise WorkflowConfigurationError(
                f"Workflow file name must match workflow_id: {path.name}"
            )
        if workflow_id in workflows:
            raise WorkflowConfigurationError(f"Duplicate workflow_id: {workflow_id}")
        workflows[workflow_id] = workflow

    policies: dict[str, dict[str, Any]] = {}
    for path in policy_paths:
        policy = _load_object(path, "Routing policy")
        validate_routing_policy(policy)
        policy_id = str(policy["policy_id"])
        if path.stem != policy_id:
            raise WorkflowConfigurationError(
                f"Routing policy file name must match policy_id: {path.name}"
            )
        if policy_id in policies:
            raise WorkflowConfigurationError(f"Duplicate routing policy_id: {policy_id}")
        policies[policy_id] = policy

    metadata = _load_object(
        source_root / "workflow_metadata" / "specialist_capabilities.json",
        "Specialist capability metadata",
    )
    capability_index = _validate_specialist_metadata(metadata, selected_specialists)
    for workflow in workflows.values():
        policy_id = str(workflow["routing_policy_id"])
        if policy_id not in policies:
            raise WorkflowConfigurationError(
                f"Workflow '{workflow['workflow_id']}' references unknown policy '{policy_id}'"
            )
        policy = policies[policy_id]
        if policy["default_workflow"] != workflow["workflow_id"]:
            raise WorkflowConfigurationError(
                f"Policy '{policy_id}' does not default to workflow '{workflow['workflow_id']}'"
            )
        allowed_capabilities = set(workflow["allowed_capabilities"])
        required_capabilities = set(policy["required_capabilities"])
        invariant_capabilities = set(workflow["invariant_capabilities"].values())
        if required_capabilities != invariant_capabilities:
            raise WorkflowConfigurationError(
                f"Policy '{policy_id}' required capabilities must equal workflow invariants"
            )
        if not allowed_capabilities.issuperset(required_capabilities):
            raise WorkflowConfigurationError(
                f"Workflow '{workflow['workflow_id']}' does not allow every required capability"
            )
        routed_capabilities = {
            capability
            for route in policy["routes"]
            for capability in route["add_capabilities"]
        }
        routed_capabilities.update(policy["low_confidence"]["capabilities"])
        unavailable = sorted(
            (allowed_capabilities | required_capabilities | routed_capabilities)
            - set(capability_index)
        )
        if unavailable:
            raise WorkflowConfigurationError(
                f"Workflow '{workflow['workflow_id']}' requires unavailable capabilities: "
                + ", ".join(unavailable)
            )
    return workflows, policies, capability_index


def load_release_workflow_catalog(
    config_bucket: str, workflow_id: str, selected_specialists: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Load one workflow and its policy from the immutable S3 release prefix."""

    if not config_bucket:
        raise WorkflowConfigurationError("CONFIG_BUCKET is required for workflow loading")
    if not _WORKFLOW_RE.fullmatch(workflow_id):
        raise WorkflowConfigurationError(f"Invalid workflow ID: {workflow_id}")

    import boto3

    prefix = os.environ.get("CONFIG_PREFIX", "").strip("/")

    def load_key(relative_key: str, description: str) -> dict[str, Any]:
        key = f"{prefix}/{relative_key}" if prefix else relative_key
        try:
            response = boto3.client("s3").get_object(Bucket=config_bucket, Key=key)
            value = json.loads(response["Body"].read().decode("utf-8"))
        except Exception as error:
            raise WorkflowConfigurationError(
                f"Could not load {description} from release: {key}"
            ) from error
        if not isinstance(value, dict):
            raise WorkflowConfigurationError(f"{description} must be a JSON object")
        return value

    workflow = load_key(f"workflows/{workflow_id}.json", "workflow definition")
    validate_workflow_definition(workflow)
    policy = load_key(
        f"routing_policies/{workflow['routing_policy_id']}.json", "routing policy"
    )
    validate_routing_policy(policy)
    metadata = load_key(
        "workflow_metadata/specialist_capabilities.json",
        "specialist capability metadata",
    )
    capability_index = _validate_specialist_metadata(metadata, selected_specialists)

    # Reuse catalog validation semantics without requiring every workflow file
    # to be listed from S3 at runtime.
    workflow_capabilities = set(workflow["allowed_capabilities"])
    invariant_capabilities = set(workflow["invariant_capabilities"].values())
    required_capabilities = set(policy["required_capabilities"])
    if policy["default_workflow"] != workflow_id:
        raise WorkflowConfigurationError(
            f"Policy '{policy['policy_id']}' does not default to workflow '{workflow_id}'"
        )
    if required_capabilities != invariant_capabilities:
        raise WorkflowConfigurationError(
            "Routing policy required capabilities must equal workflow invariants"
        )
    routed_capabilities = {
        capability
        for route in policy["routes"]
        for capability in route["add_capabilities"]
    }
    routed_capabilities.update(policy["low_confidence"]["capabilities"])
    unavailable = sorted(
        (workflow_capabilities | required_capabilities | routed_capabilities)
        - set(capability_index)
    )
    if unavailable:
        raise WorkflowConfigurationError(
            "Pinned specialist pack lacks required capabilities: " + ", ".join(unavailable)
        )
    return workflow, policy, capability_index


def validate_release_configuration(
    source_root: Path, selected_specialists: Sequence[str]
) -> None:
    """Validate the complete workflow catalog against one resolved profile."""

    load_workflow_catalog(source_root, selected_specialists)


def _normalized_classifier_facts(
    facts: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, float], ...]:
    normalized: list[tuple[str, float]] = []
    for fact in facts:
        content_type = fact.get("content_type")
        confidence = fact.get("confidence")
        if not isinstance(content_type, str) or not content_type:
            raise WorkflowPlanningError("Each classifier fact requires content_type")
        if isinstance(confidence, str):
            confidence = _CONFIDENCE_NAMES.get(confidence.lower())
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise WorkflowPlanningError(
                f"Classifier confidence for '{content_type}' must be between zero and one"
            )
        normalized.append((content_type, float(confidence)))
    return tuple(normalized)


def _ordered_capabilities(
    declared_order: Sequence[str], selected: set[str]
) -> list[str]:
    return [capability for capability in declared_order if capability in selected]


def validate_resolved_plan(plan: Mapping[str, Any]) -> None:
    """Validate the persisted plan contract without invoking a workflow."""

    allowed = {
        "schema_version",
        "operation_id",
        "session_id",
        "workflow_id",
        "workflow_version",
        "routing_mode",
        "routing_policy_id",
        "routing_policy_version",
        "config_release_id",
        "invariants",
        "recommended_capabilities",
        "resolved_specialists",
        "classification_artifacts",
        "requested_capabilities",
    }
    unexpected = sorted(set(plan) - allowed)
    if unexpected:
        raise WorkflowPlanningError(
            "Resolved plan contains unsupported fields: " + ", ".join(unexpected)
        )
    required = allowed - {"requested_capabilities"}
    missing = sorted(required - set(plan))
    if missing:
        raise WorkflowPlanningError(
            "Resolved plan is missing required fields: " + ", ".join(missing)
        )
    if plan["schema_version"] != "badgers-operation-plan/v1":
        raise WorkflowPlanningError("Resolved plan has an invalid schema_version")
    if (
        not isinstance(plan["operation_id"], str)
        or not plan["operation_id"].startswith("op_")
        or not isinstance(plan["session_id"], str)
        or not plan["session_id"]
        or not isinstance(plan["config_release_id"], str)
        or not re.fullmatch(r"[a-f0-9]{64}", plan["config_release_id"])
    ):
        raise WorkflowPlanningError("Resolved plan identity or release ID is invalid")
    _require_string(plan, "workflow_id", pattern=_WORKFLOW_RE)
    _require_string(plan, "workflow_version")
    _require_string(plan, "routing_policy_id", pattern=_WORKFLOW_RE)
    _require_string(plan, "routing_policy_version")
    if plan["routing_mode"] not in {"auto", "manual"}:
        raise WorkflowPlanningError("Resolved plan has an invalid routing mode")
    _string_list(plan["invariants"], "plan.invariants")
    _capability_list(plan["recommended_capabilities"], "plan.recommended_capabilities")
    _capability_list(plan["resolved_specialists"], "plan.resolved_specialists")
    artifacts = plan["classification_artifacts"]
    if not isinstance(artifacts, list) or not all(
        isinstance(uri, str) and _S3_URI_RE.fullmatch(uri) for uri in artifacts
    ):
        raise WorkflowPlanningError("Plan classification artifacts must be S3 object URIs")
    if "requested_capabilities" in plan:
        _capability_list(plan["requested_capabilities"], "plan.requested_capabilities")


def resolve_plan(
    *,
    operation_id: str,
    session_id: str,
    config_release_id: str,
    workflow: Mapping[str, Any],
    policy: Mapping[str, Any],
    capability_index: Mapping[str, str],
    routing_mode: str,
    classifier_facts: Sequence[Mapping[str, Any]] = (),
    classification_artifacts: Sequence[str] = (),
    requested_capabilities: Sequence[str] = (),
) -> dict[str, Any]:
    """Resolve one deterministic, capability-safe workflow plan.

    The caller supplies already-normalized classifier facts.  Reading a native
    classifier artifact is a runner concern, so this function remains fully
    deterministic and side-effect free.
    """

    validate_workflow_definition(workflow)
    validate_routing_policy(policy)
    if not operation_id.startswith("op_") or not session_id or not config_release_id:
        raise WorkflowPlanningError(
            "operation_id, session_id, and config_release_id are required"
        )
    if routing_mode not in workflow["supported_routing_modes"]:
        raise WorkflowPlanningError(
            f"Workflow '{workflow['workflow_id']}' does not support '{routing_mode}' routing"
        )
    if policy["policy_id"] != workflow["routing_policy_id"]:
        raise WorkflowPlanningError("Workflow and routing policy do not match")

    allowed = tuple(workflow["allowed_capabilities"])
    invariant_capabilities = tuple(workflow["invariant_capabilities"].values())
    required = set(policy["required_capabilities"])
    if required != set(invariant_capabilities):
        raise WorkflowPlanningError("Routing policy invariants do not match workflow")
    missing = sorted(set(allowed) - set(capability_index))
    if missing:
        raise WorkflowPlanningError(
            "Pinned specialist pack lacks required capabilities: " + ", ".join(missing)
        )

    artifact_uris = tuple(sorted(set(classification_artifacts)))
    if not all(isinstance(uri, str) and _S3_URI_RE.fullmatch(uri) for uri in artifact_uris):
        raise WorkflowPlanningError("Classification artifacts must be S3 object URIs")
    selected: set[str] = set(invariant_capabilities)
    requested = tuple(requested_capabilities)

    execution_mode = workflow.get("execution_mode", "page_images")
    if routing_mode == "manual":
        if requested and not workflow.get("manual_capability_selection_allowed", False):
            raise WorkflowPlanningError(
                f"Workflow '{workflow['workflow_id']}' does not permit manual capability selection"
            )
        if requested:
            if len(requested) != len(set(requested)) or not all(
                isinstance(capability, str) and _CAPABILITY_RE.fullmatch(capability)
                for capability in requested
            ):
                raise WorkflowPlanningError("Manual capabilities must be unique capability tags")
            unsupported = sorted(set(requested) - set(allowed))
            if unsupported:
                raise WorkflowPlanningError(
                    "Manual capabilities are not approved by this workflow: "
                    + ", ".join(unsupported)
                )
            omitted = sorted(required - set(requested))
            if omitted:
                raise WorkflowPlanningError(
                    "Manual selection cannot omit required invariants: "
                    + ", ".join(omitted)
                )
            selected = set(requested)
    elif routing_mode == "auto" and execution_mode == "page_images":
        normalized_facts = _normalized_classifier_facts(classifier_facts)
        if not normalized_facts or not artifact_uris:
            raise WorkflowPlanningError(
                "Automatic routing requires classifier facts and a classifier artifact"
            )
        low_confidence = policy["low_confidence"]
        if any(confidence < low_confidence["below"] for _, confidence in normalized_facts):
            if low_confidence["action"] == "fail":
                raise WorkflowPlanningError("Classifier confidence is below the routing policy threshold")
            selected.update(low_confidence["capabilities"])
        for route in policy["routes"]:
            minimum = float(route["when"]["minimum_confidence"])
            content_type = route["when"]["content_type"]
            if any(
                fact_type == content_type and confidence >= minimum
                for fact_type, confidence in normalized_facts
            ):
                selected.update(route["add_capabilities"])
    elif routing_mode != "auto":
        raise WorkflowPlanningError("Routing mode must be 'auto' or 'manual'")

    ordered_capabilities = _ordered_capabilities(allowed, selected)
    plan = {
        "schema_version": "badgers-operation-plan/v1",
        "operation_id": operation_id,
        "session_id": session_id,
        "workflow_id": workflow["workflow_id"],
        "workflow_version": workflow["version"],
        "routing_mode": routing_mode,
        "routing_policy_id": policy["policy_id"],
        "routing_policy_version": policy["policy_version"],
        "config_release_id": config_release_id,
        "invariants": list(workflow["invariants"]),
        "recommended_capabilities": ordered_capabilities,
        "resolved_specialists": [
            capability_index[capability] for capability in ordered_capabilities
        ],
        "classification_artifacts": list(artifact_uris),
        **(
            {"requested_capabilities": list(_ordered_capabilities(allowed, set(requested)))}
            if routing_mode == "manual" and requested
            else {}
        ),
    }
    validate_resolved_plan(plan)
    return plan
