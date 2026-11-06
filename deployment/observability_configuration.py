"""Persisted observability choices for one BADGERS deployment."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_PROFILES = frozenset({"full", "headless", "specialists"})
_MODES = frozenset({"links", "baseline", "enhanced"})
_NOTIFICATION_KINDS = frozenset({"none", "sns", "eventbridge"})
_MAX_TARGET_LENGTH = 2_048


class ObservabilityConfigurationError(ValueError):
    """Raised when a local observability plan is invalid."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ObservabilityConfigurationError(f"Could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ObservabilityConfigurationError(f"{path} must contain a JSON object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _validate_target(notification_kind: str, target: str) -> str:
    normalized = target.strip()
    if len(normalized) > _MAX_TARGET_LENGTH:
        raise ObservabilityConfigurationError("Notification target is too long.")
    if notification_kind == "none":
        if normalized:
            raise ObservabilityConfigurationError(
                "Clear the notification target when notification handoff is disabled."
            )
        return ""
    if not normalized:
        raise ObservabilityConfigurationError(
            "A notification target ARN is required for the selected handoff."
        )
    service = "sns" if notification_kind == "sns" else "events"
    if not normalized.startswith("arn:") or f":{service}:" not in normalized:
        raise ObservabilityConfigurationError(
            "Use the ARN of an SNS topic or EventBridge event bus for this handoff."
        )
    return normalized


def _payload(
    profile: str,
    mode: str,
    notification_kind: str,
    notification_target: str,
) -> dict[str, Any]:
    if profile not in _PROFILES:
        raise ObservabilityConfigurationError("Choose Full, Headless, or Specialists.")
    if mode not in _MODES:
        raise ObservabilityConfigurationError("Choose links, baseline, or enhanced monitoring.")
    if notification_kind not in _NOTIFICATION_KINDS:
        raise ObservabilityConfigurationError(
            "Choose no notification handoff, SNS, or EventBridge."
        )
    if mode == "links" and notification_kind != "none":
        raise ObservabilityConfigurationError(
            "Notification handoff requires baseline or enhanced monitoring."
        )
    if mode == "enhanced" and profile == "specialists":
        raise ObservabilityConfigurationError(
            "Enhanced monitoring needs the operation ledger in Full or Headless."
        )
    target = _validate_target(notification_kind, notification_target)
    value: dict[str, Any] = {
        "schema_version": 1,
        "profile": profile,
        "mode": mode,
        "notification": {"kind": notification_kind, "target": target},
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return value


def load_or_create_desired(state_path: Path, profile: str) -> tuple[dict[str, Any], bool]:
    """Load desired observability state, creating the no-resource links default."""

    state = _read(state_path)
    saved = state.get("desired_observability_configuration")
    if isinstance(saved, dict) and saved.get("profile") == profile:
        notification = saved.get("notification")
        if isinstance(notification, dict):
            desired = _payload(
                profile,
                str(saved.get("mode", "")),
                str(notification.get("kind", "")),
                str(notification.get("target", "")),
            )
        else:
            desired = _payload(profile, "links", "none", "")
    else:
        desired = _payload(profile, "links", "none", "")
    changed = not isinstance(saved, dict) or saved.get("fingerprint") != desired["fingerprint"]
    if changed:
        state["desired_observability_configuration"] = desired
        state["observability_configuration_revision"] = int(
            state.get("observability_configuration_revision", 0)
        ) + 1
        state["observability_configuration_updated_at"] = _timestamp()
        _write(state_path, state)
    return desired, changed


def save(
    state_path: Path,
    profile: str,
    *,
    mode: str,
    notification_kind: str,
    notification_target: str,
) -> tuple[dict[str, Any], bool]:
    """Validate and persist a reviewed observability choice."""

    state = _read(state_path)
    desired = _payload(profile, mode, notification_kind, notification_target)
    prior = state.get("desired_observability_configuration")
    changed = not isinstance(prior, dict) or prior.get("fingerprint") != desired["fingerprint"]
    if changed:
        state["desired_observability_configuration"] = desired
        state["observability_configuration_revision"] = int(
            state.get("observability_configuration_revision", 0)
        ) + 1
        state["observability_configuration_updated_at"] = _timestamp()
        _write(state_path, state)
    return desired, changed


def load_from_state(state_path: Path, profile: str) -> dict[str, Any]:
    """Read a validated existing choice without changing local deployment state."""

    state = _read(state_path)
    saved = state.get("desired_observability_configuration")
    if not isinstance(saved, dict):
        return _payload(profile, "links", "none", "")
    notification = saved.get("notification")
    if not isinstance(notification, dict):
        return _payload(profile, "links", "none", "")
    return _payload(
        profile,
        str(saved.get("mode", "")),
        str(notification.get("kind", "")),
        str(notification.get("target", "")),
    )


def record_applied(state_path: Path, profile: str) -> None:
    """Record that the saved observability selection has been deployed."""

    state = _read(state_path)
    state["applied_observability_configuration"] = load_from_state(state_path, profile)
    state["observability_configuration_applied_at"] = _timestamp()
    _write(state_path, state)


def candidate_plan(
    desired: dict[str, Any], selection: dict[str, Any]
) -> dict[str, Any]:
    """Return a local estimate, not AWS inventory or an enforceable quota result."""

    mode = desired["mode"]
    capabilities = selection.get("capabilities", {})
    specialists = selection.get("resolved_specialists", [])
    components: list[str] = []
    if specialists:
        components.append(
            f"{len(specialists)} selected specialist(s): aggregated Lambda errors and throttles"
        )
    if capabilities.get("workflow_runner"):
        components.append("Workflow failures and timeouts")
        if mode == "enhanced":
            components.append("Workflow execution latency")
    if capabilities.get("agent_runtime") or capabilities.get("gateway") or capabilities.get("ui"):
        components.append("Existing CloudWatch logs and traces for diagnosis")
    if not components:
        components.append("Selected deployment components")

    resources = {
        "cloudwatch_dashboards": 0,
        "standard_alarms": 0,
        "composite_alarms": 0,
        "notification_routes": 0,
    }
    if mode != "links":
        resources["cloudwatch_dashboards"] = 1
        # The deployed baseline deliberately aggregates specialist Lambda signals:
        # POC operators need actionable alarms, not one pager per function.
        resources["standard_alarms"] = 2
        resources["standard_alarms"] += int(bool(capabilities.get("workflow_runner"))) * 2
        if desired["notification"]["kind"] != "none":
            resources["notification_routes"] = 1
        if mode == "enhanced":
            resources["standard_alarms"] += 1

    return {
        "mode": mode,
        "coverage": components,
        "resources": resources,
        "quota_checks": [
            "CloudWatch dashboard count",
            "standard and composite CloudWatch alarm counts",
            "notification destination permissions and service quotas",
        ]
        if mode != "links"
        else [],
        "note": (
            "No added CloudWatch resources: this mode creates no monitoring resources."
            if mode == "links"
            else "The CLI deploys this dashboard and these alarms after validating the "
            "selected configuration. Notification handoff remains optional."
        ),
    }


def summary(state_path: Path, profile: str, selection: dict[str, Any]) -> dict[str, Any]:
    """Return the selected posture, deployment status, and expected footprint."""

    desired, _ = load_or_create_desired(state_path, profile)
    state = _read(state_path)
    applied = state.get("applied_observability_configuration")
    return {
        **desired,
        "revision": state.get("observability_configuration_revision", 0),
        "in_sync": isinstance(applied, dict)
        and applied.get("fingerprint") == desired.get("fingerprint"),
        "candidate_plan": candidate_plan(desired, selection),
    }


def _command(args: Any) -> int:
    path = Path(args.state).resolve()
    if args.command == "ensure":
        _, changed = load_or_create_desired(path, args.profile)
        if args.lines:
            print(f"changed\t{str(changed).lower()}")
        return 0
    if args.command == "record-applied":
        record_applied(path, args.profile)
        return 0
    raise AssertionError(f"Unsupported command: {args.command}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("ensure", "record-applied"))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(_PROFILES), required=True)
    parser.add_argument("--lines", action="store_true")
    return _command(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
