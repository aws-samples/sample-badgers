#!/usr/bin/env python3
"""Discover the models an AWS account can use without treating the registry as fact."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEPLOYMENT_DIR = Path(__file__).resolve().parent
REGISTRY_FILE = DEPLOYMENT_DIR / "s3_files" / "config" / "model_registry.json"


def _registry_module() -> Any:
    spec = importlib.util.spec_from_file_location("badgers_model_registry", DEPLOYMENT_DIR / "stacks" / "model_registry.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the model registry module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_error(error: Exception) -> dict[str, str]:
    """Convert provider errors to conservative, actionable availability states."""
    text = str(error)
    response = getattr(error, "response", {})
    code = str(response.get("Error", {}).get("Code", "")) if isinstance(response, dict) else ""
    lowered = f"{code} {text}".lower()
    detail = text[:300]
    if "explicit deny" in lowered or "scp" in lowered:
        return {"state": "denied", "reason": "Blocked by an explicit policy or SCP.", "hint": "Ask the account administrator to allow this model.", "detail": detail}
    if "accessdenied" in lowered or "access denied" in lowered:
        return {"state": "denied", "reason": "The current identity cannot invoke this model.", "hint": "Grant Bedrock model invocation access.", "detail": detail}
    if "subscription" in lowered or "model access" in lowered or "not enabled" in lowered:
        return {"state": "not_enabled", "reason": "Model access is not enabled for this account.", "hint": "Enable model access or complete the required subscription.", "detail": detail}
    if "notfound" in lowered or "not available" in lowered or "validation" in lowered:
        return {"state": "not_in_region", "reason": "The model is not available in this Region.", "hint": "Choose another Region or model.", "detail": detail}
    if "throttl" in lowered:
        return {"state": "throttled", "reason": "The availability probe was throttled.", "hint": "Retry later.", "detail": detail}
    return {"state": "error", "reason": "The account check did not complete.", "hint": "Review the provider error and retry.", "detail": detail}


def _make_clients(region: str) -> dict[str, Any]:
    import boto3

    return {
        "sts": boto3.client("sts", region_name=region),
        "bedrock": boto3.client("bedrock", region_name=region),
        "runtime": boto3.client("bedrock-runtime", region_name=region),
    }


def discover(registry: dict[str, dict[str, Any]], region: str, *, clients: dict[str, Any] | None = None, probe: bool = True) -> dict[str, Any]:
    """Inspect catalog modalities and optionally make a one-token Converse probe."""
    clients = clients or _make_clients(region)
    account = ""
    try:
        account = str(clients["sts"].get_caller_identity().get("Account", ""))
    except Exception as error:  # identity helps persistence but is not required for a report
        identity_error = classify_error(error)
    else:
        identity_error = None
    catalog: dict[str, dict[str, Any]] = {}
    try:
        summaries = clients["bedrock"].list_foundation_models().get("modelSummaries", [])
        catalog = {str(item.get("modelId")): item for item in summaries}
    except Exception as error:
        catalog_error = classify_error(error)
    else:
        catalog_error = None
    models: dict[str, dict[str, Any]] = {}
    for model_id, spec in registry.items():
        item = catalog.get(model_id.removeprefix("us.")) or catalog.get(model_id)
        vision = None if item is None else "IMAGE" in item.get("inputModalities", [])
        record: dict[str, Any] = {"state": "unverified", "reason": "Not probed.", "hint": "Run discovery with credentials.", "detail": "", "vision": vision}
        if catalog_error:
            record.update(catalog_error)
        elif spec.get("transport") == "mantle":
            record["reason"] = "Mantle models are not probed by the Converse checker."
        elif probe:
            try:
                clients["runtime"].converse(
                    modelId=model_id,
                    messages=[{"role": "user", "content": [{"text": "hi"}]}],
                    inferenceConfig={"maxTokens": 1},
                )
                record.update({"state": "available", "reason": "Converse probe succeeded.", "hint": "", "detail": ""})
            except Exception as error:
                record.update(classify_error(error))
        models[model_id] = record
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "account_id": account,
        "region": region,
        "identity_error": identity_error,
        "models": models,
    }


def _availability_path(repo: Path, account: str, region: str) -> Path:
    return repo / ".deploy-state" / "model-availability" / f"{account or 'unknown'}-{region}.json"


def save_availability(repo: Path, availability: dict[str, Any]) -> Path:
    path = _availability_path(repo, str(availability.get("account_id", "")), str(availability["region"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(availability, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def latest_availability(repo: Path, region: str | None = None) -> dict[str, Any] | None:
    """Return the newest saved local account check, optionally for one Region."""
    directory = repo / ".deploy-state" / "model-availability"
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if region is None or record.get("region") == region:
            return record
    return None


def pricing_lines(registry: dict[str, dict[str, Any]]) -> list[str]:
    return [f"{spec['display_name']}: ${spec['price_in']}/M input, ${spec['price_out']}/M output" for spec in registry.values() if spec["status"] == "active"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Report or test BADGERS Bedrock model availability.")
    parser.add_argument("command", choices=("report", "discover"), nargs="?", default="report")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2")
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--repo", type=Path, default=DEPLOYMENT_DIR.parent)
    parser.add_argument(
        "--state",
        type=Path,
        help="Also record the account check in this saved deployment state.",
    )
    args = parser.parse_args()
    registry = _registry_module().load_registry(REGISTRY_FILE)
    if args.command == "report":
        payload: dict[str, Any] = {"models": pricing_lines(registry), "pricing": _registry_module().pricing_notice(REGISTRY_FILE)}
    else:
        payload = discover(registry, args.region, probe=not args.no_probe)
        payload["saved_to"] = str(save_availability(args.repo, payload))
        if args.state:
            try:
                state = json.loads(args.state.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                parser.error(f"Could not read --state: {error}")
            state["model_availability"] = payload
            args.state.write_text(
                json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "report":
        print("\n".join(payload["models"]))
        print(f"Pricing as of {payload['pricing']['as_of']}: {payload['pricing']['url']}")
    else:
        for model_id, status in payload["models"].items():
            print(f"{model_id}\t{status['state']}\t{status['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
