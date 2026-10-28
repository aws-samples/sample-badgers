"""Tests for deployment-persisted workflow routing configuration."""

from __future__ import annotations

import json
from pathlib import Path

from deployment.release_artifact import build_release
from deployment.routing_configuration import _routing_payload, ensure
from deployment.specialist_selection import load_or_create_desired


def test_default_headless_routing_is_the_valid_pdf_image_template(
    tmp_path: Path,
) -> None:
    state = tmp_path / "demo.json"
    load_or_create_desired(state, "headless")

    routing, changed = ensure(state, "headless")

    assert changed is True
    assert routing["templates"] == ["standard-document-analysis"]
    assert "standard-document-analysis" in routing["workflows"]


def test_native_docx_route_stages_as_part_of_the_immutable_release(
    tmp_path: Path,
) -> None:
    state = tmp_path / "demo.json"
    load_or_create_desired(
        state, "headless", packs=[], direct_specialists=["docx_processor"]
    )
    saved = json.loads(state.read_text(encoding="utf-8"))
    saved["desired_routing_configuration"] = _routing_payload(
        "headless", ["native-docx-analysis"], {"docx_processor"}
    )
    state.write_text(json.dumps(saved), encoding="utf-8")

    release_id = build_release(
        "headless",
        tmp_path / "release",
        direct_specialists=("docx_processor",),
        selection_override=True,
        routing_state=state,
    )

    assert len(release_id) == 64
    workflow = json.loads(
        (tmp_path / "release/workflows/native-docx-analysis.json").read_text(
            encoding="utf-8"
        )
    )
    assert workflow["execution_mode"] == "native_document"
