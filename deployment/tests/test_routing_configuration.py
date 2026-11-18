"""Tests for deployment-persisted workflow routing configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deployment.release_artifact import build_release
from deployment.routing_configuration import (
    RoutingConfigurationError,
    _routing_payload,
    ensure,
)
from deployment.specialist_selection import load_or_create_desired


def test_default_headless_routing_is_the_valid_pdf_image_template(
    tmp_path: Path,
) -> None:
    state = tmp_path / "demo.json"
    load_or_create_desired(state, "headless")

    routing, changed = ensure(state, "headless")

    if not (changed is True):
        raise AssertionError
    if not (routing["templates"] == ["standard-document-analysis"]):
        raise AssertionError
    if not ("standard-document-analysis" in routing["workflows"]):
        raise AssertionError


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

    if not (len(release_id) == 64):
        raise AssertionError
    workflow = json.loads(
        (tmp_path / "release/workflows/native-docx-analysis.json").read_text(
            encoding="utf-8"
        )
    )
    if not (workflow["execution_mode"] == "native_document"):
        raise AssertionError


def test_native_xlsx_route_requires_the_xlsx_processor(tmp_path: Path) -> None:
    state = tmp_path / "demo.json"
    load_or_create_desired(
        state,
        "headless",
        packs=[],
        direct_specialists=["xlsx_processor"],
    )
    saved = json.loads(state.read_text(encoding="utf-8"))
    saved["desired_routing_configuration"] = _routing_payload(
        "headless",
        ["native-xlsx-analysis"],
        {"xlsx_processor"},
    )
    state.write_text(json.dumps(saved), encoding="utf-8")

    build_release(
        "headless",
        tmp_path / "release",
        direct_specialists=("xlsx_processor",),
        selection_override=True,
        routing_state=state,
    )

    workflow = json.loads(
        (tmp_path / "release/workflows/native-xlsx-analysis.json").read_text(
            encoding="utf-8"
        )
    )
    if not (workflow["invariant_capabilities"] == {"native_document": "xlsx_analysis"}):
        raise AssertionError


def test_native_xlsx_route_is_available_in_full_when_processor_is_selected() -> None:
    payload = _routing_payload("full", ["native-xlsx-analysis"], {"xlsx_processor"})
    if not ("native-xlsx-analysis" in payload["workflows"]):
        raise AssertionError


def test_native_xlsx_route_requires_its_processor_in_full() -> None:
    with pytest.raises(RoutingConfigurationError, match="requires selected specialists"):
        _routing_payload("full", ["native-xlsx-analysis"], set())
