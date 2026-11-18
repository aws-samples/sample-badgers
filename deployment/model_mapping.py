#!/usr/bin/env python3
"""Resolve specialist model choices against the account availability record.

The mapping is deliberately local desired state.  It is only copied into the
immutable staged release, never written back into source configuration.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import model_discovery
except ImportError:
    import model_discovery

DEPLOYMENT_DIR = Path(__file__).resolve().parent
MANIFESTS = DEPLOYMENT_DIR / "s3_files" / "manifests"


def _load_selection(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection = manifest.get("specialist", manifest).get("model_selections")
    if not isinstance(selection, dict):
        return {}, []
    def entry(value: Any) -> dict[str, Any]:
        return {"model_id": value} if isinstance(value, str) else dict(value)
    primary = entry(selection.get("primary")) if selection.get("primary") else {}
    fallbacks = [entry(item) for item in selection.get("fallback_list", [])]
    return primary, fallbacks


def manifest_models(specialists: list[str] | tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Return primary/fallback choices declared by each selected manifest."""
    result: dict[str, dict[str, Any]] = {}
    for specialist in specialists:
        path = MANIFESTS / f"{specialist}.json"
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        primary, fallbacks = _load_selection(manifest)
        if primary:
            result[specialist] = {"primary": primary, "fallbacks": fallbacks}
    return result


def _availability(state: dict[str, Any]) -> dict[str, Any]:
    mapping = state.get("model_availability")
    if not isinstance(mapping, dict):
        return {}
    models = mapping.get("models")
    return models if isinstance(models, dict) else {}


def resolve(state: dict[str, Any]) -> dict[str, Any]:
    """Choose the first verifiably available configured candidate per specialist."""
    selection = state.get("desired_specialist_selection", {})
    specialists = selection.get("resolved_specialists", []) if isinstance(selection, dict) else []
    available = _availability(state)
    resolved: dict[str, Any] = {}
    unresolved: list[str] = []
    for name, configured in manifest_models(tuple(specialists)).items():
        candidates = [configured["primary"], *configured["fallbacks"]]
        chosen = next(
            (candidate for candidate in candidates if available.get(candidate["model_id"], {}).get("state") == "available"),
            None,
        )
        if chosen is None:
            unresolved.append(name)
            continue
        resolved[name] = {
            "model_id": chosen["model_id"],
            "fallbacks": [candidate["model_id"] for candidate in candidates if candidate != chosen],
            "promoted": chosen != configured["primary"],
        }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "availability_generated_at": state.get("model_availability", {}).get("generated_at", ""),
        "specialists": resolved,
        "unresolved": unresolved,
    }


def set_override(state: dict[str, Any], specialist: str, model_id: str | None) -> dict[str, Any]:
    """Set or clear one explicit choice while retaining the rest of the mapping."""
    mapping = dict(state.get("desired_model_mapping", {}))
    specialists = dict(mapping.get("specialists", {}))
    if model_id is None:
        specialists.pop(specialist, None)
    else:
        current = dict(specialists.get(specialist, {}))
        current["model_id"] = model_id
        current["promoted"] = False
        specialists[specialist] = current
    mapping["specialists"] = specialists
    mapping["generated_at"] = datetime.now(timezone.utc).isoformat()
    state["desired_model_mapping"] = mapping
    return mapping


def apply_mapping(staging_root: Path, mapping: dict[str, Any]) -> None:
    """Overlay a resolved mapping onto staged manifest copies."""
    entries = mapping.get("specialists", {})
    if not isinstance(entries, dict):
        return
    for specialist, record in entries.items():
        if not isinstance(record, dict) or not isinstance(record.get("model_id"), str):
            continue
        path = staging_root / "manifests" / f"{specialist}.json"
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        selection = manifest.get("specialist", {}).get("model_selections")
        if not isinstance(selection, dict):
            continue
        primary = selection.get("primary")
        if isinstance(primary, str):
            selection["primary"] = record["model_id"]
        elif isinstance(primary, dict):
            primary["model_id"] = record["model_id"]
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("show", "resolve", "set"))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--specialist")
    parser.add_argument("--model")
    parser.add_argument("--discover", action="store_true")
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2",
    )
    args = parser.parse_args()
    state = _state(args.state)
    if args.discover:
        registry = model_discovery._registry_module().load_registry(model_discovery.REGISTRY_FILE)
        state["model_availability"] = model_discovery.discover(registry, args.region)
    if args.command == "resolve":
        state["desired_model_mapping"] = resolve(state)
        args.state.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "set":
        if not args.specialist:
            parser.error("--specialist is required for set")
        set_override(state, args.specialist, args.model)
        args.state.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(state.get("desired_model_mapping", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
