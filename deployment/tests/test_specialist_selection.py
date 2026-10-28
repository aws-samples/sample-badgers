"""Tests for persisted deployment-specific specialist desired state."""

from __future__ import annotations

from pathlib import Path

from deployment.specialist_selection import (
    load_or_create_desired,
    record_applied_selection,
    reconciliation,
)


def test_direct_docx_selection_requires_its_container_path(tmp_path: Path) -> None:
    state = tmp_path / "demo.json"

    desired, changed = load_or_create_desired(
        state,
        "specialists",
        packs=[],
        direct_specialists=["docx_processor"],
    )

    assert changed is True
    assert desired["resolved_specialists"] == ["docx_processor"]
    assert desired["container_specialists"] == ["docx_processor"]
    assert desired["capabilities"]["container_images"] is True
    assert reconciliation(state, desired)["in_sync"] is False


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

    assert first["fingerprint"] != second["fingerprint"]
    assert changed is True
    assert plan["in_sync"] is False
    assert plan["add"] == ["pptx_processor"]
    assert plan["remove"] == ["docx_processor"]
