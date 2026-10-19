"""Stage the exact configuration needed by a resolved BADGERS profile.

The deploy path uploads this staging directory instead of treating the source
tree as a runtime configuration bucket. That keeps Headless installations
small and gives a release process one concrete directory to hash and publish.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    from .profile_resolver import ResolvedProfile, resolve_profile
except ImportError:  # Script execution: python deployment/config_stager.py ...
    from profile_resolver import ResolvedProfile, resolve_profile

_DEPLOYMENT_DIR = Path(__file__).parent
_FOUNDATION_DIR = _DEPLOYMENT_DIR / "badgers-foundation"
if str(_FOUNDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_FOUNDATION_DIR))

from foundation.workflow_planner import validate_release_configuration


class ConfigurationStagingError(ValueError):
    """Raised when a profile cannot be staged into a complete configuration set."""


_DEFAULT_SOURCE = _DEPLOYMENT_DIR / "s3_files"


def _copy_file(source_root: Path, relative_path: str, destination: Path) -> None:
    source = source_root / relative_path
    if not source.is_file():
        raise ConfigurationStagingError(f"Required configuration file is missing: {source}")
    target = destination / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_tree_if_present(
    source_root: Path, relative_path: str, destination: Path
) -> None:
    source = source_root / relative_path
    if not source.exists():
        return
    if not source.is_dir():
        raise ConfigurationStagingError(
            f"Expected a configuration directory but found a file: {source}"
        )
    shutil.copytree(source, destination / relative_path, dirs_exist_ok=True)


def _stage_workflow_configuration(
    source_root: Path, destination: Path, profile: ResolvedProfile
) -> None:
    """Validate and stage only the workflow metadata relevant to a runtime profile."""

    try:
        validate_release_configuration(source_root, profile.specialists)
    except ValueError as error:
        raise ConfigurationStagingError(
            f"Workflow configuration is invalid for profile '{profile.name}': {error}"
        ) from error

    _copy_tree_if_present(source_root, "workflows", destination)
    _copy_tree_if_present(source_root, "routing_policies", destination)

    metadata_path = source_root / "workflow_metadata" / "specialist_capabilities.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        selected = {
            specialist: metadata["specialists"][specialist]
            for specialist in profile.specialists
        }
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ConfigurationStagingError(
            f"Could not stage specialist capability metadata: {metadata_path}"
        ) from error
    target = destination / "workflow_metadata" / "specialist_capabilities.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {"schema_version": 1, "specialists": selected},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def stage_profile_config(
    profile: ResolvedProfile,
    destination: Path,
    *,
    source_root: Path = _DEFAULT_SOURCE,
) -> tuple[str, ...]:
    """Create a minimal, complete runtime configuration directory.

    The caller owns ``destination``. Refusing a non-empty directory prevents a
    stale file from a prior profile becoming part of deployed configuration.
    """

    if destination.exists() and any(destination.iterdir()):
        raise ConfigurationStagingError(
            f"Destination must be empty before staging profile configuration: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    # Agent and prompt-core files are shared runtime dependencies. They are
    # included as whole directories because their internal references are owned by
    # the prompt-core package rather than duplicated in every profile.
    for directory in (
        "agent_config",
        "agent_system_prompt",
        "config",
        "core_system_prompts",
        "contracts",
        "wrappers",
        "prompts/shared",
    ):
        _copy_tree_if_present(source_root, directory, destination)

    staged: list[str] = []
    for specialist in profile.specialists:
        _copy_file(source_root, f"schemas/{specialist}.json", destination)

        # Utility tools such as conversion may not have a prompt or manifest.
        # Where they exist, both are kept strictly profile-scoped.
        manifest = source_root / f"manifests/{specialist}.json"
        if manifest.is_file():
            _copy_file(source_root, f"manifests/{specialist}.json", destination)
        _copy_tree_if_present(source_root, f"prompts/{specialist}", destination)
        staged.append(specialist)

    # Direct specialist packs need no workflow control plane. Any profile that
    # owns agent orchestration or a deterministic workflow runner stages these
    # artifacts under the same immutable release prefix as prompts/manifests.
    # This lets the future gateway-only profile consume the same release
    # without duplicating workflow configuration.
    if profile.capabilities.get("agent_runtime", False) or profile.capabilities.get(
        "workflow_runner", False
    ):
        _stage_workflow_configuration(source_root, destination, profile)

    return tuple(staged)


def main() -> int:
    """Stage one profile's runtime configuration for deploy or release tooling."""

    parser = argparse.ArgumentParser(
        description="Stage the minimal BADGERS configuration for one product profile."
    )
    parser.add_argument("profile", help="Profile file name without the .json suffix")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=_DEFAULT_SOURCE)
    parser.add_argument(
        "--pack",
        action="append",
        default=[],
        help="Override a user-selectable profile's packs; repeat for multiple packs",
    )
    args = parser.parse_args()

    try:
        profile = resolve_profile(args.profile, specialist_packs=tuple(args.pack) or None)
        staged = stage_profile_config(profile, args.destination, source_root=args.source)
    except (ConfigurationStagingError, ValueError) as error:
        parser.error(str(error))

    print(
        f"Staged {len(staged)} specialists for {profile.display_name}: "
        f"{', '.join(staged)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
