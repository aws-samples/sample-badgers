"""Persisted, validated workflow-routing configuration for one deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import termios
import tty
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEPLOYMENT_DIR = Path(__file__).parent
STANDARD_WORKFLOW = DEPLOYMENT_DIR / "s3_files/workflows/standard-document-analysis.json"
STANDARD_POLICY = DEPLOYMENT_DIR / "s3_files/routing_policies/standard-document-routing.json"

_TEMPLATES: dict[str, dict[str, Any]] = {
    "standard-document-analysis": {
        "label": "PDF / image analysis",
        "required_specialists": {
            "pdf_to_images_converter",
            "classify_pdf_content",
            "full_text_specialist",
            "elements_specialist",
            "correlation_specialist",
        },
    },
    "native-docx-analysis": {
        "label": "Native DOCX analysis",
        "required_specialists": {"docx_processor"},
        "workflow": {
            "schema_version": 1,
            "workflow_id": "native-docx-analysis",
            "version": "1.0.0",
            "display_name": "Native DOCX Analysis",
            "execution_mode": "native_document",
            "supported_routing_modes": ["auto"],
            "input_contract": {
                "accepted_source_types": [
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ],
                "requires_document_uri": True,
            },
            "invariants": ["validate_source", "native_document", "write_completion_manifest"],
            "invariant_capabilities": {"native_document": "docx_conversion"},
            "routing_policy_id": "native-docx-routing",
            "allowed_capabilities": ["docx_conversion"],
            "manual_capability_selection_allowed": False,
            "result_contract": "badgers-operation-result/v1",
        },
        "policy": {
            "schema_version": 1,
            "policy_id": "native-docx-routing",
            "policy_version": "1.0.0",
            "default_workflow": "native-docx-analysis",
            "limits": {"max_specialized_tools_per_page": 1, "max_pages": 500},
            "required_capabilities": ["docx_conversion"],
            "routes": [],
            "low_confidence": {"below": 0.0, "action": "fail", "capabilities": []},
        },
    },
    "native-pptx-analysis": {
        "label": "Native PPTX analysis",
        "required_specialists": {"pptx_processor"},
        "workflow": {
            "schema_version": 1,
            "workflow_id": "native-pptx-analysis",
            "version": "1.0.0",
            "display_name": "Native PPTX Analysis",
            "execution_mode": "native_document",
            "supported_routing_modes": ["auto"],
            "input_contract": {
                "accepted_source_types": [
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                ],
                "requires_document_uri": True,
            },
            "invariants": ["validate_source", "native_document", "write_completion_manifest"],
            "invariant_capabilities": {"native_document": "pptx_conversion"},
            "routing_policy_id": "native-pptx-routing",
            "allowed_capabilities": ["pptx_conversion"],
            "manual_capability_selection_allowed": False,
            "result_contract": "badgers-operation-result/v1",
        },
        "policy": {
            "schema_version": 1,
            "policy_id": "native-pptx-routing",
            "policy_version": "1.0.0",
            "default_workflow": "native-pptx-analysis",
            "limits": {"max_specialized_tools_per_page": 1, "max_pages": 500},
            "required_capabilities": ["pptx_conversion"],
            "routes": [],
            "low_confidence": {"below": 0.0, "action": "fail", "capabilities": []},
        },
    },
}


class RoutingConfigurationError(ValueError):
    """Raised when a requested routing configuration is not deployable."""


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RoutingConfigurationError(f"Could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise RoutingConfigurationError(f"{path} must contain a JSON object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _template_artifacts(template_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if template_id == "standard-document-analysis":
        return _read(STANDARD_WORKFLOW), _read(STANDARD_POLICY)
    template = _TEMPLATES[template_id]
    return dict(template["workflow"]), dict(template["policy"])


def _selection(state: dict[str, Any]) -> set[str]:
    desired = state.get("desired_specialist_selection")
    if not isinstance(desired, dict):
        raise RoutingConfigurationError(
            "Configure specialist selection first; no desired specialist selection is saved."
        )
    selected = desired.get("resolved_specialists")
    if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
        raise RoutingConfigurationError("Saved specialist selection is invalid.")
    return set(selected)


def _routing_payload(profile: str, template_ids: list[str], selected: set[str]) -> dict[str, Any]:
    if profile == "specialists" and template_ids:
        raise RoutingConfigurationError(
            "The Specialists product has no workflow runner. Use direct Lambda invocation."
        )
    if profile != "specialists" and not template_ids:
        raise RoutingConfigurationError("Choose at least one workflow for an agent product.")
    workflows: dict[str, Any] = {}
    policies: dict[str, Any] = {}
    for template_id in template_ids:
        if template_id not in _TEMPLATES:
            raise RoutingConfigurationError(f"Unknown routing template: {template_id}")
        missing = sorted(_TEMPLATES[template_id]["required_specialists"] - selected)
        if missing:
            raise RoutingConfigurationError(
                f"{_TEMPLATES[template_id]['label']} requires selected specialists: "
                + ", ".join(missing)
            )
        workflow, policy = _template_artifacts(template_id)
        workflows[workflow["workflow_id"]] = workflow
        policies[policy["policy_id"]] = policy
    value = {
        "schema_version": 1,
        "profile": profile,
        "templates": template_ids,
        "workflows": workflows,
        "policies": policies,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return value


def ensure(state_path: Path, profile: str) -> tuple[dict[str, Any], bool]:
    state = _read(state_path)
    selected = _selection(state)
    saved = state.get("desired_routing_configuration")
    if isinstance(saved, dict) and saved.get("profile") == profile:
        templates = list(saved.get("templates", []))
    else:
        templates = [] if profile == "specialists" else ["standard-document-analysis"]
    desired = _routing_payload(profile, templates, selected)
    changed = not isinstance(saved, dict) or saved.get("fingerprint") != desired["fingerprint"]
    if changed:
        state["desired_routing_configuration"] = desired
        state["routing_configuration_revision"] = int(state.get("routing_configuration_revision", 0)) + 1
        state["routing_configuration_updated_at"] = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        )
        _write(state_path, state)
    return desired, changed


def save_templates(
    state_path: Path, profile: str, templates: list[str]
) -> tuple[dict[str, Any], bool]:
    """Validate and persist an explicit set of reviewed workflow templates."""

    state = _read(state_path)
    desired = _routing_payload(profile, templates, _selection(state))
    prior = state.get("desired_routing_configuration")
    changed = not isinstance(prior, dict) or prior.get("fingerprint") != desired["fingerprint"]
    if changed:
        state["desired_routing_configuration"] = desired
        state["routing_configuration_revision"] = int(
            state.get("routing_configuration_revision", 0)
        ) + 1
        state["routing_configuration_updated_at"] = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        )
        _write(state_path, state)
    return desired, changed


def record_applied(state_path: Path, config_release_id: str) -> None:
    state = _read(state_path)
    desired = state.get("desired_routing_configuration")
    if not isinstance(desired, dict):
        raise RoutingConfigurationError("No routing configuration has been saved.")
    applied = dict(desired)
    applied["config_release_id"] = config_release_id
    applied["applied_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    state["applied_routing_configuration"] = applied
    _write(state_path, state)


def _read_key() -> str:
    first = sys.stdin.read(1)
    if first != "\x1b":
        return first
    second, third = sys.stdin.read(1), sys.stdin.read(1)
    return {"\x1b[A": "UP", "\x1b[B": "DOWN"}.get(first + second + third, "ESC")


def configure(state_path: Path, profile: str) -> tuple[dict[str, Any], bool]:
    state = _read(state_path)
    saved = state.get("desired_routing_configuration")
    templates = (
        list(saved.get("templates", []))
        if isinstance(saved, dict) and saved.get("profile") == profile
        else ([] if profile == "specialists" else ["standard-document-analysis"])
    )
    if profile == "specialists":
        return _routing_payload(profile, [], _selection(state)), False
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RoutingConfigurationError("Interactive routing configuration requires a terminal.")
    selected = _selection(state)
    selected_templates = set(templates)
    options = list(_TEMPLATES)
    cursor = 0
    old = termios.tcgetattr(sys.stdin.fileno())
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            os.system("clear")
            print("BADGERS WORKFLOW & ROUTING")
            print("Space toggle  a select available  d deselect all  ↑↓ move  Enter save  q cancel")
            print("")
            for index, template_id in enumerate(options):
                template = _TEMPLATES[template_id]
                required = template["required_specialists"]
                available = required.issubset(selected)
                marker = "[x]" if template_id in selected_templates else "[ ]"
                pointer = "›" if index == cursor else " "
                suffix = "" if available else " (select required specialists first)"
                print(f"{pointer} {marker} {template['label']}{suffix}")
            print("")
            print("  Template routing is validated and release-pinned; custom execution graphs are not accepted.")
            key = _read_key()
            if key == "UP":
                cursor = (cursor - 1) % len(options)
            elif key == "DOWN":
                cursor = (cursor + 1) % len(options)
            elif key == " ":
                template_id = options[cursor]
                if _TEMPLATES[template_id]["required_specialists"].issubset(selected):
                    selected_templates.symmetric_difference_update({template_id})
            elif key in {"a", "A"}:
                selected_templates = {
                    template_id
                    for template_id in options
                    if _TEMPLATES[template_id]["required_specialists"].issubset(selected)
                }
            elif key in {"d", "D"}:
                selected_templates.clear()
            elif key in {"\r", "\n"}:
                return save_templates(state_path, profile, sorted(selected_templates))
            elif key in {"q", "Q", "ESC"}:
                raise RoutingConfigurationError("Routing configuration cancelled.")
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
        print("")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ensure", "configure", "summary", "record-applied"):
        command = sub.add_parser(name)
        command.add_argument("--state", type=Path, required=True)
        command.add_argument("--profile", required=True)
        if name == "record-applied":
            command.add_argument("--config-release-id", required=True)
        else:
            command.add_argument("--lines", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "record-applied":
            record_applied(args.state, args.config_release_id)
            return 0
        desired, changed = (
            configure(args.state, args.profile)
            if args.command == "configure"
            else ensure(args.state, args.profile)
        )
        if args.lines:
            for template in desired["templates"]:
                print(f"template\t{template}")
            print(f"fingerprint\t{desired['fingerprint']}")
            print(f"changed\t{'true' if changed else 'false'}")
        else:
            print(json.dumps(desired, indent=2, sort_keys=True))
        return 0
    except RoutingConfigurationError as error:
        parser.error(str(error))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
