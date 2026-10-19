"""Resolve deployable BADGERS product profiles into explicit specialist sets.

This module is intentionally independent of CDK.  It is the first step in
splitting the current all-in-one deployment into Full, Headless, and
specialist-pack products: callers can inspect and validate a profile without
creating any AWS resources.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ProfileResolutionError(ValueError):
    """Raised when a profile or specialist pack cannot be resolved safely."""


@dataclass(frozen=True)
class ResolvedProfile:
    """The explicit deployment input produced from a product profile."""

    name: str
    display_name: str
    profile_version: int
    specialist_packs: tuple[str, ...]
    specialists: tuple[str, ...]
    capabilities: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


_DEFAULT_PROFILES_DIR = Path(__file__).parent / "profiles"
_DEFAULT_DEPLOYMENT_CONFIG = Path(__file__).parent / "deployment_config.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            content = json.load(file)
    except FileNotFoundError as error:
        raise ProfileResolutionError(f"Configuration file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ProfileResolutionError(f"Invalid JSON in {path}: {error}") from error

    if not isinstance(content, dict):
        raise ProfileResolutionError(f"Configuration file must contain an object: {path}")
    return content


def _require_string(profile: dict[str, Any], key: str, source: Path) -> str:
    value = profile.get(key)
    if not isinstance(value, str) or not value:
        raise ProfileResolutionError(f"{source}: '{key}' must be a non-empty string")
    return value


def _capabilities(profile: dict[str, Any], source: Path) -> dict[str, bool]:
    capabilities = profile.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ProfileResolutionError(f"{source}: 'capabilities' must be an object")

    non_boolean = [name for name, value in capabilities.items() if not isinstance(value, bool)]
    if non_boolean:
        names = ", ".join(sorted(non_boolean))
        raise ProfileResolutionError(f"{source}: capabilities must be booleans: {names}")

    resolved = dict(capabilities)
    if resolved.get("ui", False) and not resolved.get("ui_identity", False):
        raise ProfileResolutionError(
            f"{source}: UI profiles must enable the 'ui_identity' capability"
        )
    if resolved.get("ui", False) and not resolved.get("agent_runtime", False):
        raise ProfileResolutionError(
            f"{source}: UI profiles must enable the 'agent_runtime' capability"
        )
    if resolved.get("ui", False) and not resolved.get("gateway", False):
        raise ProfileResolutionError(
            f"{source}: UI profiles must enable the 'gateway' capability"
        )
    if resolved.get("websocket_transport", False) and not resolved.get(
        "agent_runtime", False
    ):
        raise ProfileResolutionError(
            f"{source}: websocket_transport requires the 'agent_runtime' capability"
        )
    if resolved.get("session_memory", False) and not resolved.get(
        "agent_runtime", False
    ):
        raise ProfileResolutionError(
            f"{source}: session_memory requires the 'agent_runtime' capability"
        )
    return resolved


def _enabled_specialists(deployment_config: dict[str, Any]) -> set[str]:
    specialists = deployment_config.get("specialists")
    if not isinstance(specialists, dict):
        raise ProfileResolutionError(
            "deployment_config.json: 'specialists' must be an object"
        )
    return {
        name
        for name, specialist in specialists.items()
        if isinstance(specialist, dict) and specialist.get("enabled", True)
    }


def _load_pack(pack_name: str, profiles_dir: Path) -> list[str]:
    pack_path = profiles_dir / "specialist_packs" / f"{pack_name}.json"
    pack = _load_json(pack_path)
    declared_name = _require_string(pack, "name", pack_path)
    if declared_name != pack_name:
        raise ProfileResolutionError(
            f"{pack_path}: pack name '{declared_name}' does not match file name"
        )

    specialists = pack.get("specialists")
    if not isinstance(specialists, list) or not specialists:
        raise ProfileResolutionError(
            f"{pack_path}: 'specialists' must be a non-empty array"
        )
    if not all(isinstance(name, str) and name for name in specialists):
        raise ProfileResolutionError(
            f"{pack_path}: every specialist name must be a non-empty string"
        )
    return specialists


def resolve_profile(
    profile_name: str,
    *,
    profiles_dir: Path = _DEFAULT_PROFILES_DIR,
    deployment_config_path: Path = _DEFAULT_DEPLOYMENT_CONFIG,
    specialist_packs: tuple[str, ...] | None = None,
) -> ResolvedProfile:
    """Resolve a named profile without performing AWS or filesystem writes."""

    profile_path = profiles_dir / f"{profile_name}.json"
    profile = _load_json(profile_path)
    declared_name = _require_string(profile, "name", profile_path)
    if declared_name != profile_name:
        raise ProfileResolutionError(
            f"{profile_path}: profile name '{declared_name}' does not match file name"
        )

    display_name = _require_string(profile, "display_name", profile_path)
    version = profile.get("profile_version")
    if not isinstance(version, int) or version < 1:
        raise ProfileResolutionError(
            f"{profile_path}: 'profile_version' must be a positive integer"
        )

    capabilities = _capabilities(profile, profile_path)
    deployment_config = _load_json(deployment_config_path)
    enabled = _enabled_specialists(deployment_config)

    specialist_selection = profile.get("specialist_selection")
    packs = profile.get("specialist_packs", [])
    if specialist_selection == "enabled":
        if specialist_packs:
            raise ProfileResolutionError(
                f"{profile_path}: this profile does not allow specialist-pack overrides"
            )
        if packs:
            raise ProfileResolutionError(
                f"{profile_path}: 'enabled' selection cannot also declare packs"
            )
        selected = sorted(enabled)
        pack_names: tuple[str, ...] = ()
    elif specialist_selection == "packs":
        if not isinstance(packs, list) or not packs:
            raise ProfileResolutionError(
                f"{profile_path}: pack selection requires a non-empty specialist_packs array"
            )
        if not all(isinstance(pack, str) and pack for pack in packs):
            raise ProfileResolutionError(
                f"{profile_path}: every specialist pack name must be a non-empty string"
            )
        selection_mode = profile.get("pack_selection", "fixed")
        if selection_mode not in ("fixed", "user_selectable"):
            raise ProfileResolutionError(
                f"{profile_path}: 'pack_selection' must be 'fixed' or 'user_selectable'"
            )
        if specialist_packs:
            if selection_mode != "user_selectable":
                raise ProfileResolutionError(
                    f"{profile_path}: this profile does not allow specialist-pack overrides"
                )
            if not all(isinstance(pack, str) and pack for pack in specialist_packs):
                raise ProfileResolutionError(
                    "specialist pack overrides must be non-empty strings"
                )
            pack_names = tuple(dict.fromkeys(specialist_packs))
        else:
            pack_names = tuple(packs)
        selected = []
        for pack_name in pack_names:
            selected.extend(_load_pack(pack_name, profiles_dir))
        selected = list(dict.fromkeys(selected))
    else:
        raise ProfileResolutionError(
            f"{profile_path}: 'specialist_selection' must be 'enabled' or 'packs'"
        )

    unknown = sorted(set(selected) - set(deployment_config["specialists"]))
    if unknown:
        raise ProfileResolutionError(
            f"{profile_path}: unknown specialists: {', '.join(unknown)}"
        )
    disabled = sorted(set(selected) - enabled)
    if disabled:
        raise ProfileResolutionError(
            f"{profile_path}: selected specialists are disabled in deployment_config.json: "
            f"{', '.join(disabled)}"
        )

    return ResolvedProfile(
        name=declared_name,
        display_name=display_name,
        profile_version=version,
        specialist_packs=pack_names,
        specialists=tuple(selected),
        capabilities=capabilities,
    )


def main() -> int:
    """Print a resolved profile for use in planning and CI."""

    parser = argparse.ArgumentParser(
        description="Resolve a BADGERS deployment profile without AWS writes."
    )
    parser.add_argument("profile", help="Profile file name without the .json suffix")
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=_DEFAULT_PROFILES_DIR,
        help="Directory containing profile JSON files",
    )
    parser.add_argument(
        "--deployment-config",
        type=Path,
        default=_DEFAULT_DEPLOYMENT_CONFIG,
        help="Path to deployment_config.json",
    )
    parser.add_argument(
        "--pack",
        action="append",
        default=[],
        help="Override a user-selectable profile's packs; repeat for multiple packs",
    )
    args = parser.parse_args()

    try:
        resolved = resolve_profile(
            args.profile,
            profiles_dir=args.profiles_dir,
            deployment_config_path=args.deployment_config,
            specialist_packs=tuple(args.pack) or None,
        )
    except ProfileResolutionError as error:
        parser.error(str(error))

    print(json.dumps(resolved.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
