"""Tests for declarative BADGERS deployment profile resolution."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from deployment.config_stager import stage_profile_config
from deployment.profile_resolver import ProfileResolutionError, resolve_profile
from deployment.release_artifact import build_release


DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
PROFILES_DIR = DEPLOYMENT_DIR / "profiles"
DEPLOYMENT_CONFIG = DEPLOYMENT_DIR / "deployment_config.json"


def test_full_profile_resolves_every_enabled_specialist() -> None:
    resolved = resolve_profile(
        "full",
        profiles_dir=PROFILES_DIR,
        deployment_config_path=DEPLOYMENT_CONFIG,
    )

    config = json.loads(DEPLOYMENT_CONFIG.read_text(encoding="utf-8"))
    enabled = {
        name
        for name, specialist in config["specialists"].items()
        if specialist.get("enabled", True)
    }

    assert resolved.display_name == "BADGERS Full"
    assert set(resolved.specialists) == enabled
    assert resolved.capabilities["ui"] is True
    assert resolved.capabilities["agent_runtime"] is True


def test_headless_profile_resolves_curated_packs_without_ui() -> None:
    resolved = resolve_profile(
        "headless",
        profiles_dir=PROFILES_DIR,
        deployment_config_path=DEPLOYMENT_CONFIG,
    )

    assert resolved.specialist_packs == ("document-essentials", "tables-and-forms")
    assert resolved.specialists == (
        "pdf_to_images_converter",
        "classify_pdf_content",
        "full_text_specialist",
        "layout_specialist",
        "elements_specialist",
        "correlation_specialist",
        "table_specialist",
    )
    assert resolved.capabilities["ui"] is False
    assert resolved.capabilities["websocket_transport"] is False
    assert resolved.capabilities["minimal_operation_ledger"] is True
    assert resolved.capabilities["workflow_runner"] is True


def test_profile_rejects_unknown_specialist(tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles"
    packs_dir = profiles_dir / "specialist_packs"
    packs_dir.mkdir(parents=True)
    (profiles_dir / "headless.json").write_text(
        json.dumps(
            {
                "name": "headless",
                "display_name": "Headless",
                "profile_version": 1,
                "specialist_selection": "packs",
                "specialist_packs": ["unknown-pack"],
                "capabilities": {"agent_runtime": True},
            }
        ),
        encoding="utf-8",
    )
    (packs_dir / "unknown-pack.json").write_text(
        json.dumps({"name": "unknown-pack", "specialists": ["not-a-specialist"]}),
        encoding="utf-8",
    )

    with pytest.raises(ProfileResolutionError, match="unknown specialists"):
        resolve_profile(
            "headless",
            profiles_dir=profiles_dir,
            deployment_config_path=DEPLOYMENT_CONFIG,
        )


def test_profile_rejects_ui_without_ui_identity(tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "invalid.json").write_text(
        json.dumps(
            {
                "name": "invalid",
                "display_name": "Invalid",
                "profile_version": 1,
                "specialist_selection": "enabled",
                "capabilities": {"agent_runtime": True, "ui": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProfileResolutionError, match="ui_identity"):
        resolve_profile(
            "invalid",
            profiles_dir=profiles_dir,
            deployment_config_path=DEPLOYMENT_CONFIG,
        )


def test_headless_staging_excludes_unselected_specialists() -> None:
    profile = resolve_profile(
        "headless",
        profiles_dir=PROFILES_DIR,
        deployment_config_path=DEPLOYMENT_CONFIG,
    )
    source = DEPLOYMENT_DIR / "s3_files"

    with tempfile.TemporaryDirectory() as temporary_directory:
        destination = Path(temporary_directory) / "staged"
        staged = stage_profile_config(profile, destination, source_root=source)

        assert staged == profile.specialists
        assert (destination / "schemas/table_specialist.json").is_file()
        assert (destination / "prompts/table_specialist").is_dir()
        assert (destination / "core_system_prompts").is_dir()
        assert (destination / "contracts/badgers-operation-result-v1.json").is_file()
        assert (destination / "workflows/standard-document-analysis.json").is_file()
        assert (
            destination / "routing_policies/standard-document-routing.json"
        ).is_file()
        staged_metadata = json.loads(
            (destination / "workflow_metadata/specialist_capabilities.json").read_text(
                encoding="utf-8"
            )
        )
        assert set(staged_metadata["specialists"]) == set(profile.specialists)
        assert not (destination / "schemas/charts_specialist.json").exists()
        assert not (destination / "prompts/charts_specialist").exists()


def test_specialists_profile_allows_an_explicit_pack_selection() -> None:
    resolved = resolve_profile(
        "specialists",
        profiles_dir=PROFILES_DIR,
        deployment_config_path=DEPLOYMENT_CONFIG,
        specialist_packs=("tables-and-forms",),
    )

    assert resolved.specialist_packs == ("tables-and-forms",)
    assert resolved.specialists == ("table_specialist",)
    assert resolved.capabilities["agent_runtime"] is False
    assert resolved.capabilities["direct_invocation"] is True


def test_direct_specialists_staging_omits_workflow_control_plane() -> None:
    profile = resolve_profile(
        "specialists",
        profiles_dir=PROFILES_DIR,
        deployment_config_path=DEPLOYMENT_CONFIG,
        specialist_packs=("tables-and-forms",),
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        destination = Path(temporary_directory) / "staged"
        stage_profile_config(
            profile,
            destination,
            source_root=DEPLOYMENT_DIR / "s3_files",
        )

        assert (destination / "schemas/table_specialist.json").is_file()
        assert not (destination / "workflows").exists()
        assert not (destination / "routing_policies").exists()
        assert not (destination / "workflow_metadata").exists()


def test_headless_profile_rejects_pack_override() -> None:
    with pytest.raises(ProfileResolutionError, match="does not allow"):
        resolve_profile(
            "headless",
            profiles_dir=PROFILES_DIR,
            deployment_config_path=DEPLOYMENT_CONFIG,
            specialist_packs=("tables-and-forms",),
        )


def test_release_artifact_is_deterministic_and_has_a_manifest() -> None:
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        first_release = build_release("headless", Path(first))
        second_release = build_release("headless", Path(second))

        assert first_release == second_release
        manifest = json.loads(
            (Path(first) / "release-manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["release_id"] == first_release
        assert "schemas/table_specialist.json" in manifest["files"]
        assert "workflows/standard-document-analysis.json" in manifest["files"]


def test_release_hash_changes_when_workflow_or_policy_changes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(DEPLOYMENT_DIR / "s3_files", source)
    baseline = build_release("headless", tmp_path / "baseline", source_root=source)

    workflow_path = source / "workflows/standard-document-analysis.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow["version"] = "1.0.1"
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    changed_workflow = build_release(
        "headless", tmp_path / "workflow-change", source_root=source
    )

    policy_path = source / "routing_policies/standard-document-routing.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["policy_version"] = "1.0.1"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    changed_policy = build_release(
        "headless", tmp_path / "policy-change", source_root=source
    )

    assert baseline != changed_workflow
    assert changed_workflow != changed_policy
