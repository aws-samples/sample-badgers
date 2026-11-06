"""Per-deployment resource tag configuration for BADGERS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_PROFILES = frozenset({"full", "headless", "specialists"})
_KEY = re.compile(r"^[\w\s.:/=+\-@]{1,128}$")
_MAX_VALUE_LENGTH = 256
_MAX_TAGS = 40
_RESERVED_KEYS = frozenset(
    {
        "deployment_id",
        "product_profile",
        "stack_suffix",
        "configuration_fingerprint",
        "managed_by",
        "cdk_stack_prefix",
    }
)
# Customer tags are intentionally opt-in. Placeholder ownership and cost tags
# create misleading inventory in a customer's AWS account when left unchanged.
DEFAULT_TAGS: dict[str, str] = {}


class TagConfigurationError(ValueError):
    """Raised when a deployment tag configuration is invalid."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TagConfigurationError(f"Could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise TagConfigurationError(f"{path} must contain a JSON object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _validate_tags(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise TagConfigurationError("Tags must be a key/value object.")
    if len(value) > _MAX_TAGS:
        raise TagConfigurationError(f"Use at most {_MAX_TAGS} customer-configured tags.")

    tags: dict[str, str] = {}
    for key, tag_value in value.items():
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise TagConfigurationError(
                "Tag keys must be 1-128 characters using letters, numbers, spaces, or .:/=+-@."
            )
        if key.lower().startswith("aws:"):
            raise TagConfigurationError("Tag keys starting with 'aws:' are reserved by AWS.")
        if key in _RESERVED_KEYS:
            raise TagConfigurationError(f"'{key}' is generated for each BADGERS deployment.")
        if not isinstance(tag_value, str) or not tag_value.strip():
            raise TagConfigurationError(f"Tag '{key}' needs a non-empty value.")
        normalized = tag_value.strip()
        if len(normalized) > _MAX_VALUE_LENGTH:
            raise TagConfigurationError(
                f"Tag '{key}' exceeds the {_MAX_VALUE_LENGTH}-character value limit."
            )
        tags[key] = normalized
    return dict(sorted(tags.items()))


def _payload(profile: str, tags: object) -> dict[str, Any]:
    if profile not in _PROFILES:
        raise TagConfigurationError("Choose Full, Headless, or Specialists.")
    value: dict[str, Any] = {
        "schema_version": 1,
        "profile": profile,
        "tags": _validate_tags(tags),
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return value


def load_or_create_desired(state_path: Path, profile: str) -> tuple[dict[str, Any], bool]:
    """Load desired tags, starting with no customer tags when absent."""

    state = _read(state_path)
    saved = state.get("desired_tag_configuration")
    if isinstance(saved, dict) and saved.get("profile") == profile:
        desired = _payload(profile, saved.get("tags", DEFAULT_TAGS))
    else:
        desired = _payload(profile, DEFAULT_TAGS)
    changed = not isinstance(saved, dict) or saved.get("fingerprint") != desired["fingerprint"]
    if changed:
        state["desired_tag_configuration"] = desired
        state["tag_configuration_revision"] = int(state.get("tag_configuration_revision", 0)) + 1
        state["tag_configuration_updated_at"] = _timestamp()
        _write(state_path, state)
    return desired, changed


def save(state_path: Path, profile: str, *, tags: object) -> tuple[dict[str, Any], bool]:
    """Validate and persist tags selected for one deployment."""

    state = _read(state_path)
    desired = _payload(profile, tags)
    prior = state.get("desired_tag_configuration")
    changed = not isinstance(prior, dict) or prior.get("fingerprint") != desired["fingerprint"]
    if changed:
        state["desired_tag_configuration"] = desired
        state["tag_configuration_revision"] = int(state.get("tag_configuration_revision", 0)) + 1
        state["tag_configuration_updated_at"] = _timestamp()
        _write(state_path, state)
    return desired, changed


def load_from_state(state_path: Path, profile: str) -> dict[str, Any]:
    """Read a validated existing tag configuration without changing local state."""

    state = _read(state_path)
    saved = state.get("desired_tag_configuration")
    if not isinstance(saved, dict):
        return _payload(profile, DEFAULT_TAGS)
    return _payload(profile, saved.get("tags", DEFAULT_TAGS))


def effective_tags(
    desired: dict[str, Any],
    *,
    deployment_id: str,
    profile: str,
    stack_suffix: str,
) -> dict[str, str]:
    """Combine customer values with the stable deployment identity tags."""

    tags = dict(desired["tags"])
    tags.update(
        {
            "deployment_id": deployment_id,
            "product_profile": profile,
            "stack_suffix": stack_suffix,
            "configuration_fingerprint": str(desired["fingerprint"])[:16],
            "managed_by": "cdk",
        }
    )
    return tags


def record_applied(state_path: Path, profile: str) -> None:
    """Record that the current desired tag set reached the final deployment step."""

    state = _read(state_path)
    desired = load_from_state(state_path, profile)
    state["applied_tag_configuration"] = desired
    state["tag_configuration_applied_at"] = _timestamp()
    _write(state_path, state)


def _command(args: argparse.Namespace) -> int:
    path = args.state.resolve()
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("ensure", "record-applied"))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(_PROFILES), required=True)
    parser.add_argument("--lines", action="store_true")
    return _command(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
