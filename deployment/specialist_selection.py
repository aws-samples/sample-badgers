"""Desired-state selection, pack management, and reconciliation for BADGERS.

The saved selection belongs to one deployment identity and is stored alongside
its local deployment state. Pack definitions remain reusable source-controlled
JSON files. Every read re-resolves packs, so manual JSON edits are visible on
the next CLI run instead of becoming a hidden competing source of truth.
"""

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

try:
    from .profile_resolver import (
        CONTAINER_SPECIALISTS,
        ProfileResolutionError,
        ResolvedProfile,
        resolve_profile,
    )
except ImportError:
    from profile_resolver import (
        CONTAINER_SPECIALISTS,
        ProfileResolutionError,
        ResolvedProfile,
        resolve_profile,
    )


DEPLOYMENT_DIR = Path(__file__).parent
PROFILES_DIR = DEPLOYMENT_DIR / "profiles"
PACKS_DIR = PROFILES_DIR / "specialist_packs"
CATALOG_PATH = DEPLOYMENT_DIR / "deployment_config.json"


class SelectionError(ValueError):
    """Raised when a saved desired selection cannot be resolved safely."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SelectionError(f"Configuration file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise SelectionError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(parsed, dict):
        raise SelectionError(f"Configuration file must contain an object: {path}")
    return parsed


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _catalog() -> dict[str, dict[str, Any]]:
    specialists = _load_json(CATALOG_PATH).get("specialists")
    if not isinstance(specialists, dict):
        raise SelectionError(f"{CATALOG_PATH}: 'specialists' must be an object")
    return {
        name: value
        for name, value in specialists.items()
        if isinstance(name, str) and isinstance(value, dict)
    }


def _profile_defaults(profile_name: str) -> tuple[list[str], list[str]]:
    """Return the default pack and direct selections for a product profile."""

    profile = _load_json(PROFILES_DIR / f"{profile_name}.json")
    selection_kind = profile.get("specialist_selection")
    if selection_kind == "enabled":
        return [], list(resolve_profile(profile_name).specialists)
    if selection_kind == "packs":
        packs = profile.get("specialist_packs")
        if not isinstance(packs, list):
            raise SelectionError(
                f"{PROFILES_DIR / f'{profile_name}.json'}: specialist_packs must be a list"
            )
        return list(packs), []
    raise SelectionError(
        f"{PROFILES_DIR / f'{profile_name}.json'}: unknown specialist_selection"
    )


def _fingerprint(profile_name: str, packs: list[str], direct: list[str], resolved: ResolvedProfile) -> str:
    payload = {
        "profile": profile_name,
        "packs": packs,
        "direct_specialists": direct,
        "resolved_specialists": list(resolved.specialists),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def resolve_selection(
    profile_name: str,
    packs: list[str],
    direct_specialists: list[str],
) -> dict[str, Any]:
    """Validate and return one canonical desired selection."""

    try:
        resolved = resolve_profile(
            profile_name,
            specialist_packs=tuple(packs),
            direct_specialists=tuple(direct_specialists),
            selection_override=True,
        )
    except ProfileResolutionError as error:
        raise SelectionError(str(error)) from error

    selected_containers = sorted(set(resolved.specialists) & CONTAINER_SPECIALISTS)
    return {
        "schema_version": 1,
        "profile": profile_name,
        "packs": list(resolved.specialist_packs),
        "direct_specialists": list(resolved.direct_specialists),
        "resolved_specialists": list(resolved.specialists),
        "container_specialists": selected_containers,
        "capabilities": resolved.capabilities,
        "fingerprint": _fingerprint(
            profile_name,
            list(resolved.specialist_packs),
            list(resolved.direct_specialists),
            resolved,
        ),
    }


def load_or_create_desired(
    state_path: Path,
    profile_name: str,
    *,
    packs: list[str] | None = None,
    direct_specialists: list[str] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Load desired state, applying explicit CLI input or profile defaults."""

    state = _load_json(state_path) if state_path.exists() else {}
    saved = state.get("desired_specialist_selection")
    explicit = packs is not None or direct_specialists is not None

    if explicit:
        desired = resolve_selection(profile_name, packs or [], direct_specialists or [])
    elif isinstance(saved, dict):
        if saved.get("profile") != profile_name:
            raise SelectionError(
                f"Saved specialist selection belongs to profile '{saved.get('profile')}', "
                f"not '{profile_name}'. Use a separate deployment identity."
            )
        desired = resolve_selection(
            profile_name,
            list(saved.get("packs", [])),
            list(saved.get("direct_specialists", [])),
        )
    else:
        default_packs, default_direct = _profile_defaults(profile_name)
        desired = resolve_selection(profile_name, default_packs, default_direct)

    changed = not isinstance(saved, dict) or saved.get("fingerprint") != desired["fingerprint"]
    if changed:
        state["desired_specialist_selection"] = desired
        state["specialist_selection_revision"] = int(
            state.get("specialist_selection_revision", 0)
        ) + 1
        state["specialist_selection_updated_at"] = _timestamp()
        _write_json(state_path, state)
    return desired, changed


def reconciliation(state_path: Path, desired: dict[str, Any]) -> dict[str, Any]:
    """Compare the current desired selection with the last applied selection."""

    state = _load_json(state_path) if state_path.exists() else {}
    applied = state.get("applied_specialist_selection")
    applied_members = set(applied.get("resolved_specialists", [])) if isinstance(applied, dict) else set()
    desired_members = set(desired["resolved_specialists"])
    return {
        "deployment_id": state.get("deployment_id", state_path.stem),
        "profile": desired["profile"],
        "selection_revision": state.get("specialist_selection_revision", 1),
        "desired_fingerprint": desired["fingerprint"],
        "applied_fingerprint": applied.get("fingerprint", "") if isinstance(applied, dict) else "",
        "in_sync": isinstance(applied, dict) and applied.get("fingerprint") == desired["fingerprint"],
        "add": sorted(desired_members - applied_members),
        "remove": sorted(applied_members - desired_members),
        "container_specialists": desired["container_specialists"],
        "required_stages": required_stages(desired, bool(applied)),
    }


def required_stages(desired: dict[str, Any], has_applied_selection: bool) -> list[str]:
    """Return the minimal stage sequence required to reconcile a selection."""

    stages = ["publish configuration", "update specialist Lambdas"]
    if desired["container_specialists"]:
        stages.insert(1, "build and push selected container images")
    if desired["capabilities"].get("gateway", False):
        stages.append("reconcile Gateway targets")
    if desired["capabilities"].get("agent_runtime", False):
        stages.append("redeploy Runtime with the new configuration release")
    if not has_applied_selection:
        return stages
    return stages


def record_applied_selection(state_path: Path, *, config_release_id: str = "") -> None:
    """Record the desired set only after its consumer deployment succeeds."""

    state = _load_json(state_path)
    desired = state.get("desired_specialist_selection")
    if not isinstance(desired, dict):
        raise SelectionError("No desired specialist selection has been saved.")
    applied = dict(desired)
    applied["applied_at"] = _timestamp()
    if config_release_id:
        applied["config_release_id"] = config_release_id
    state["applied_specialist_selection"] = applied
    state["specialist_selection_applied_at"] = applied["applied_at"]
    _write_json(state_path, state)


def pack_definitions(profile_name: str | None = None) -> list[dict[str, Any]]:
    """Return validated pack definitions in a CLI-friendly shape."""

    catalog = _catalog()
    definitions: list[dict[str, Any]] = []
    for path in sorted(PACKS_DIR.glob("*.json")):
        pack = _load_json(path)
        name = pack.get("name")
        members = pack.get("specialists")
        if not isinstance(name, str) or not isinstance(members, list):
            raise SelectionError(f"Invalid pack definition: {path}")
        unknown = sorted(set(members) - set(catalog))
        if unknown:
            raise SelectionError(f"{path}: unknown specialists: {', '.join(unknown)}")
        supported_profiles = pack.get("supported_profiles")
        if supported_profiles is not None:
            if (
                not isinstance(supported_profiles, list)
                or not supported_profiles
                or not all(isinstance(item, str) and item for item in supported_profiles)
            ):
                raise SelectionError(
                    f"{path}: 'supported_profiles' must be a non-empty string array"
                )
            if profile_name is not None and profile_name not in supported_profiles:
                continue
        definitions.append(
            {
                "name": name,
                "display_name": pack.get("display_name", name),
                "description": pack.get("description", ""),
                "version": pack.get("version", 1),
                "specialists": members,
                "supported_profiles": supported_profiles or [],
            }
        )
    return definitions


def save_pack(name: str, description: str, specialists: list[str], *, version: int = 1) -> Path:
    """Create or replace a named pack after validating its members."""

    if not name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in name):
        raise SelectionError("Pack names use lowercase letters, digits, and hyphens only.")
    if not specialists:
        raise SelectionError("A pack must contain at least one specialist.")
    catalog = _catalog()
    unknown = sorted(set(specialists) - set(catalog))
    if unknown:
        raise SelectionError(f"Unknown specialists: {', '.join(unknown)}")
    disabled = sorted(name for name in specialists if not catalog[name].get("enabled", True))
    if disabled:
        raise SelectionError(f"Disabled specialists cannot be packed: {', '.join(disabled)}")
    path = PACKS_DIR / f"{name}.json"
    _write_json(
        path,
        {
            "name": name,
            "display_name": name.replace("-", " ").title(),
            "version": version,
            "description": description,
            "specialists": list(dict.fromkeys(specialists)),
        },
    )
    return path


def pack_references(pack_name: str, state_dir: Path) -> list[str]:
    """Return saved deployment IDs whose desired configuration selects a pack."""

    if not state_dir.exists():
        return []
    referenced: list[str] = []
    for path in state_dir.glob("*.json"):
        try:
            desired = _load_json(path).get("desired_specialist_selection", {})
        except SelectionError:
            continue
        if isinstance(desired, dict) and pack_name in desired.get("packs", []):
            referenced.append(path.stem)
    return sorted(referenced)


def delete_pack(name: str, state_dir: Path, *, confirmed: bool) -> list[str]:
    path = PACKS_DIR / f"{name}.json"
    if not path.is_file():
        raise SelectionError(f"Pack does not exist: {name}")
    references = pack_references(name, state_dir)
    if references:
        raise SelectionError(
            f"Pack '{name}' is selected by: {', '.join(references)}. "
            "Migrate those deployments to another pack or direct selections before deleting it."
        )
    if not confirmed:
        raise SelectionError("Deletion requires --confirm.")
    path.unlink()
    return references


def _read_key() -> str:
    first = sys.stdin.read(1)
    if first != "\x1b":
        return first
    second = sys.stdin.read(1)
    third = sys.stdin.read(1)
    if second == "[":
        return {"A": "UP", "B": "DOWN"}.get(third, "")
    return "ESC"


def _interactive_picker(
    profile_name: str, desired: dict[str, Any], *, include_packs: bool = True
) -> tuple[list[str], list[str]]:
    """Render a compact keyboard picker for packs and direct specialists."""

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise SelectionError(
            "Interactive specialist selection requires a terminal. "
            "Use --pack and --specialist for scripted selection."
        )
    catalog = _catalog()
    packs = pack_definitions(profile_name)
    items = (
        [("pack", pack["name"], pack["display_name"], pack["description"]) for pack in packs]
        if include_packs
        else []
    )
    items.extend(
        (
            "specialist",
            name,
            name.replace("_", " ").title(),
            str(value.get("category", "Specialist")),
        )
        for name, value in sorted(catalog.items())
        if value.get("enabled", True)
    )
    selected_packs = set(desired["packs"])
    selected_direct = set(desired["direct_specialists"])
    cursor = 0
    old = termios.tcgetattr(sys.stdin.fileno())
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            os.system("clear")
            print("BADGERS SPECIALIST SELECTION")
            print(
                f"Deployment product: {profile_name}  |  Space toggle  "
                "a select all  d deselect all  ↑↓ move  Enter save  q cancel"
            )
            print("")
            for index, (kind, identifier, label, detail) in enumerate(items):
                selected = identifier in (selected_packs if kind == "pack" else selected_direct)
                marker = "[x]" if selected else "[ ]"
                pointer = "›" if index == cursor else " "
                kind_label = "PACK" if kind == "pack" else "SPECIALIST"
                print(f"{pointer} {marker} {label:<30} {kind_label:<11} {detail}")
            print("")
            preview = resolve_selection(
                profile_name, sorted(selected_packs), sorted(selected_direct)
            ) if selected_packs or selected_direct else None
            if preview:
                suffix = (
                    f" · containers: {', '.join(preview['container_specialists'])}"
                    if preview["container_specialists"]
                    else ""
                )
                print(f"  {len(preview['resolved_specialists'])} resolved specialists{suffix}")
            else:
                print("  Select at least one pack or specialist.")
            key = _read_key()
            if key == "UP":
                cursor = (cursor - 1) % len(items)
            elif key == "DOWN":
                cursor = (cursor + 1) % len(items)
            elif key == " ":
                kind, identifier, _, _ = items[cursor]
                target = selected_packs if kind == "pack" else selected_direct
                if identifier in target:
                    target.remove(identifier)
                else:
                    target.add(identifier)
            elif key in {"a", "A"}:
                selected_packs = {
                    identifier for kind, identifier, _, _ in items if kind == "pack"
                }
                selected_direct = {
                    identifier for kind, identifier, _, _ in items if kind == "specialist"
                }
            elif key in {"d", "D"}:
                selected_packs.clear()
                selected_direct.clear()
            elif key in ("\r", "\n"):
                if preview:
                    return sorted(selected_packs), sorted(selected_direct)
            elif key in ("q", "Q", "ESC"):
                raise SelectionError("Specialist selection cancelled.")
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
        print("")


def _select_pack_name() -> str:
    packs = pack_definitions()
    if not packs:
        raise SelectionError("There are no specialist packs to edit or delete.")
    print("")
    for index, pack in enumerate(packs, start=1):
        print(f"  {index}) {pack['name']:<28} {len(pack['specialists'])} specialists")
    raw = input("Pack number (blank cancels): ").strip()
    if not raw:
        raise SelectionError("Pack management cancelled.")
    try:
        return packs[int(raw) - 1]["name"]
    except (ValueError, IndexError) as error:
        raise SelectionError("Choose a listed pack number.") from error


def _interactive_pack_manager(state_dir: Path) -> None:
    """Manage reusable packs without modifying any deployment selection implicitly."""

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise SelectionError("Interactive pack management requires a terminal.")
    while True:
        print("")
        print("BADGERS PACK LIBRARY")
        for pack in pack_definitions():
            print(
                f"  {pack['name']:<28} v{pack['version']}  "
                f"{len(pack['specialists'])} specialists  {pack['description']}"
            )
        print("")
        action = input("[c]reate  [e]dit  [d]elete  [q]uit: ").strip().lower()
        if action in {"q", "quit", ""}:
            return
        if action not in {"c", "e", "d"}:
            print("Choose c, e, d, or q.")
            continue
        try:
            if action == "c":
                name = input("New pack name (lowercase letters, digits, hyphens): ").strip()
                description = input("Description: ").strip()
                # The direct-specialist portion of the familiar selector makes creation
                # discoverable without making the new pack active for a deployment.
                _, members = _interactive_picker(
                    "specialists",
                    {"packs": [], "direct_specialists": []},
                    include_packs=False,
                )
                saved = save_pack(name, description, members)
                print(f"Saved {saved.name}.")
            elif action == "e":
                name = _select_pack_name()
                existing = _load_json(PACKS_DIR / f"{name}.json")
                description = input(
                    f"Description [{existing.get('description', '')}]: "
                ).strip() or str(existing.get("description", ""))
                desired = resolve_selection(
                    "specialists", [], list(existing.get("specialists", []))
                )
                _, members = _interactive_picker(
                    "specialists",
                    {
                        **desired,
                        "packs": [],
                        "direct_specialists": list(existing.get("specialists", [])),
                    },
                    include_packs=False,
                )
                saved = save_pack(
                    name,
                    description,
                    members,
                    version=int(existing.get("version", 0)) + 1,
                )
                print(f"Updated {saved.name}.")
            else:
                name = _select_pack_name()
                references = pack_references(name, state_dir)
                if references:
                    print(
                        f"Cannot delete {name}; selected by: {', '.join(references)}. "
                        "Migrate those deployments first."
                    )
                    continue
                confirmation = input(f"Type {name} to delete this pack: ").strip()
                if confirmation != name:
                    print("Delete cancelled.")
                    continue
                delete_pack(name, state_dir, confirmed=True)
                print(f"Deleted {name}.")
        except SelectionError as error:
            print(f"Pack change not applied: {error}")


def _parse_state_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _emit(
    desired: dict[str, Any], state_path: Path, *, lines: bool, changed: bool = False
) -> None:
    if lines:
        for pack in desired["packs"]:
            print(f"pack\t{pack}")
        for specialist in desired["direct_specialists"]:
            print(f"specialist\t{specialist}")
        for specialist in desired["container_specialists"]:
            print(f"container\t{specialist}")
        print(f"fingerprint\t{desired['fingerprint']}")
        print(f"changed\t{'true' if changed else 'false'}")
        print(
            f"in_sync\t{'true' if reconciliation(state_path, desired)['in_sync'] else 'false'}"
        )
        return
    report = dict(desired)
    report["state_path"] = str(state_path)
    report["reconciliation"] = reconciliation(state_path, desired)
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state", type=_parse_state_path, required=True)
    common.add_argument("--profile", required=True)

    ensure = subparsers.add_parser("ensure", parents=[common])
    ensure.add_argument("--pack", action="append")
    ensure.add_argument("--specialist", action="append")
    ensure.add_argument("--lines", action="store_true")

    configure = subparsers.add_parser("configure", parents=[common])
    configure.add_argument("--lines", action="store_true")

    summary = subparsers.add_parser("summary", parents=[common])
    summary.add_argument("--lines", action="store_true")

    applied = subparsers.add_parser("record-applied", parents=[common])
    applied.add_argument("--config-release-id", default="")

    subparsers.add_parser("list-packs")
    create = subparsers.add_parser("create-pack")
    create.add_argument("name")
    create.add_argument("--description", required=True)
    create.add_argument("--specialist", action="append", required=True)
    update = subparsers.add_parser("update-pack")
    update.add_argument("name")
    update.add_argument("--description", required=True)
    update.add_argument("--specialist", action="append", required=True)
    delete = subparsers.add_parser("delete-pack")
    delete.add_argument("name")
    delete.add_argument("--state-dir", type=_parse_state_path, required=True)
    delete.add_argument("--confirm", action="store_true")
    manager = subparsers.add_parser("manage-packs")
    manager.add_argument("--state-dir", type=_parse_state_path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "list-packs":
            print(json.dumps(pack_definitions(), indent=2, sort_keys=True))
            return 0
        if args.command in {"create-pack", "update-pack"}:
            path = PACKS_DIR / f"{args.name}.json"
            version = 1
            if args.command == "update-pack" and path.is_file():
                version = int(_load_json(path).get("version", 0)) + 1
            saved = save_pack(args.name, args.description, args.specialist, version=version)
            print(f"Saved pack: {saved}")
            return 0
        if args.command == "delete-pack":
            references = delete_pack(args.name, args.state_dir, confirmed=args.confirm)
            print(f"Deleted pack '{args.name}'. Referenced deployments: {', '.join(references) or 'none'}")
            return 0
        if args.command == "manage-packs":
            _interactive_pack_manager(args.state_dir)
            return 0

        if args.command == "ensure":
            desired, changed = load_or_create_desired(
                args.state,
                args.profile,
                packs=args.pack,
                direct_specialists=args.specialist,
            )
            _emit(desired, args.state, lines=args.lines, changed=changed)
            return 0
        if args.command == "configure":
            desired, _ = load_or_create_desired(args.state, args.profile)
            packs, direct = _interactive_picker(args.profile, desired)
            desired, changed = load_or_create_desired(
                args.state, args.profile, packs=packs, direct_specialists=direct
            )
            _emit(desired, args.state, lines=args.lines, changed=changed)
            return 0
        if args.command == "summary":
            desired, changed = load_or_create_desired(args.state, args.profile)
            _emit(desired, args.state, lines=args.lines)
            return 0
        if args.command == "record-applied":
            record_applied_selection(args.state, config_release_id=args.config_release_id)
            print("Recorded applied specialist selection.")
            return 0
    except SelectionError as error:
        parser.error(str(error))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
