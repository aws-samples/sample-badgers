from __future__ import annotations

import json
from pathlib import Path

from deployment.model_mapping import apply_mapping, resolve


def test_resolve_promotes_first_available_fallback(tmp_path: Path, monkeypatch) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "sample.json").write_text(
        json.dumps(
            {"specialist": {"model_selections": {"primary": "blocked", "fallback_list": ["available"]}}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("deployment.model_mapping.MANIFESTS", manifests)
    result = resolve(
        {
            "desired_specialist_selection": {"resolved_specialists": ["sample"]},
            "model_availability": {"models": {"blocked": {"state": "denied"}, "available": {"state": "available"}}},
        }
    )
    if not (result["unresolved"] == []):
        raise AssertionError
    if not (result["specialists"]["sample"]["model_id"] == "available"):
        raise AssertionError
    if not (result["specialists"]["sample"]["promoted"] is True):
        raise AssertionError


def test_apply_mapping_changes_only_staged_manifest(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    target = manifests / "sample.json"
    target.write_text(json.dumps({"specialist": {"model_selections": {"primary": "old"}}}), encoding="utf-8")
    apply_mapping(tmp_path, {"specialists": {"sample": {"model_id": "new"}}})
    if not (json.loads(target.read_text(encoding="utf-8"))["specialist"]["model_selections"]["primary"] == "new"):
        raise AssertionError
