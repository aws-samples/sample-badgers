"""Tests for persisted deployment-specific specialist desired state."""

from __future__ import annotations

from pathlib import Path

import pytest

import deployment.specialist_selection as specialist_selection
from deployment.specialist_selection import (
    SelectionError,
    load_or_create_desired,
    record_applied_selection,
    reconciliation,
    save_pack,
)


def test_direct_docx_selection_requires_its_container_path(tmp_path: Path) -> None:
    state = tmp_path / "demo.json"

    desired, changed = load_or_create_desired(
        state,
        "specialists",
        packs=[],
        direct_specialists=["docx_processor"],
    )

    if not (changed is True):
        raise AssertionError
    if not (desired["resolved_specialists"] == ["docx_processor"]):
        raise AssertionError
    if not (desired["container_specialists"] == ["docx_processor"]):
        raise AssertionError
    if not (desired["capabilities"]["container_images"] is True):
        raise AssertionError
    if not (reconciliation(state, desired)["in_sync"] is False):
        raise AssertionError


def test_reconciliation_tracks_additions_removals_and_applied_fingerprint(
    tmp_path: Path,
) -> None:
    state = tmp_path / "demo.json"
    first, _ = load_or_create_desired(
        state,
        "specialists",
        packs=[],
        direct_specialists=["docx_processor"],
    )
    record_applied_selection(state, config_release_id="a" * 64)

    second, changed = load_or_create_desired(
        state,
        "specialists",
        packs=[],
        direct_specialists=["pptx_processor"],
    )
    plan = reconciliation(state, second)

    if not (first["fingerprint"] != second["fingerprint"]):
        raise AssertionError
    if not (changed is True):
        raise AssertionError
    if not (plan["in_sync"] is False):
        raise AssertionError
    if not (plan["add"] == ["pptx_processor"]):
        raise AssertionError
    if not (plan["remove"] == ["docx_processor"]):
        raise AssertionError


def test_pack_name_validation_uses_an_explicit_ascii_pattern(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(specialist_selection, "PACKS_DIR", tmp_path)

    saved = save_pack("review-pack-2", "test pack", ["full_text_specialist"])

    if not (saved.name == "review-pack-2.json"):
        raise AssertionError
    with pytest.raises(SelectionError, match="lowercase letters"):
        save_pack("Review pack", "test pack", ["full_text_specialist"])
