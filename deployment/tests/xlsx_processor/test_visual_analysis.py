"""Tests for the opt-in, non-authoritative XLSX Vision component."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from defusedxml import ElementTree as ET


PROCESSOR_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "lambdas" / "containers" / "xlsx_processor"
)
sys.path.insert(0, str(PROCESSOR_DIRECTORY))

from visual_analysis import (  # noqa: E402
    VisualAnalysisError,
    _model_selection,
    grade_for_confidence,
    parse_visual_response,
    prepare_visual_analysis,
)
from workbook_inspector import inspect_xlsx  # noqa: E402
from xlsx_content_tree import build_xlsx_content_tree_xml  # noqa: E402
from test_workbook_inspector import _workbook  # noqa: E402


def test_prepares_embedded_image_with_native_context_only() -> None:
    inspection = inspect_xlsx(_workbook())

    prepared, skipped = prepare_visual_analysis(_workbook(), inspection, max_objects=1)

    assert skipped == []
    assert len(prepared) == 1
    assert prepared[0]["object"]["object_id"] == "drawing-7"
    assert prepared[0]["content_type"] == "image/png"
    assert prepared[0]["image_bytes"].startswith(b"\x89PNG")
    assert '"object_kind": "image"' in prepared[0]["runtime_context"]
    assert '"formula_calculation"' not in prepared[0]["runtime_context"]


def test_keeps_chart_as_a_distinct_pending_visual_type() -> None:
    inspection = inspect_xlsx(_workbook())
    inspection["sheets"][0]["visual_objects"] = [
        {"object_id": "drawing-9", "kind": "chart", "chart_path": "xl/charts/chart1.xml"}
    ]

    prepared, skipped = prepare_visual_analysis(_workbook(), inspection)

    assert prepared == []
    assert skipped == [
        {
            "object_id": "drawing-9",
            "kind": "chart",
            "status": "pending_rendered_chart_input",
            "reason": (
                "A native chart was detected, but no rendered chart image is available "
                "for Vision analysis."
            ),
        }
    ]


def test_validates_model_response_and_school_style_grade() -> None:
    result = parse_visual_response(
        """```json
{
  "object_id": "drawing-7",
  "object_kind": "image",
  "classification": "data_overlay",
  "visual_summary": "An opaque dark green rectangle marked PNG.",
  "visible_text": ["PNG"],
  "relationship_to_covered_cells": "The image visibly overlays the listed cells.",
  "cell_visibility": "obscures",
  "semantic_replacement_suspected": true,
  "confidence_percent": 95,
  "confidence_grade": "A",
  "requires_human_review": true
}
```""",
        visual_object={
            "kind": "image",
            "anchor": {
                "from": {"column": 2, "row": 6},
                "to": {"column": 4, "row": 13},
            },
            "covered_cells": [{"coordinate": "B6"}, {"coordinate": "C6"}],
        },
        expected_object_id="drawing-7",
        expected_kind="image",
        sheet_name="Revenue",
    )

    assert result["analysis_scope"] == "non_authoritative_visual_evidence"
    assert result["confidence_grade"] == "A"
    assert result["spatial_coverage"] == {
        "relation": "drawn_above_native_cells",
        "from": {"column": "B", "row": 6, "cell": "B6"},
        "to": {"column": "D", "row": 13, "cell": "D13"},
        "grid_span": "B6:D13",
        "coordinate_delta": {"columns": 2, "rows": 7},
        "covered_native_cell_count": 2,
        "confidence": "deterministic",
    }
    assert result["visual_effect"]["semantic_replacement_suspected"] is True
    assert grade_for_confidence(89) == "B"


def test_rejects_a_model_response_with_an_inconsistent_grade() -> None:
    with pytest.raises(VisualAnalysisError, match="does not match"):
        parse_visual_response(
            """{
              "object_id": "drawing-7",
              "object_kind": "image",
              "classification": "decorative",
              "visual_summary": "A green box.",
              "visible_text": [],
              "relationship_to_covered_cells": "Unknown.",
              "cell_visibility": "unknown",
              "semantic_replacement_suspected": false,
              "confidence_percent": 70,
              "confidence_grade": "A",
              "requires_human_review": false
            }""",
            visual_object={"kind": "image", "anchor": {}},
            expected_object_id="drawing-7",
            expected_kind="image",
            sheet_name="Revenue",
        )


def test_reads_the_manifest_primary_model_id() -> None:
    assert _model_selection(
        {
            "model_selections": {
                "primary": {"model_id": "us.anthropic.claude-sonnet-4-6"},
                "fallback_list": [],
            },
            "expected_output_tokens": 1200,
        }
    ) == {
        "model_id": "us.anthropic.claude-sonnet-4-6",
        "max_tokens": 1200,
    }


def test_merges_native_and_vision_evidence_into_badgers_xml() -> None:
    inspection = inspect_xlsx(_workbook())
    inspection["processing_guarantees"] = {
        "formula_calculation": "not_performed",
        "workbook_execution": "not_performed",
    }
    inspection["visual_analysis"] = {
        "status": "complete",
        "model_id": "us.anthropic.claude-sonnet-4-6",
        "analyses": [
            {
                "object_id": "drawing-7",
                "sheet_name": "Revenue",
                "classification": "data_overlay",
                "confidence_percent": 95,
                "confidence_grade": "A",
                "requires_human_review": True,
                "analysis_scope": "non_authoritative_visual_evidence",
                "visual_summary": "A green PNG overlay.",
                "relationship_to_covered_cells": "The image obscures displayed cells.",
                "visible_text": ["PNG"],
                "spatial_coverage": {
                    "relation": "drawn_above_native_cells",
                    "grid_span": "A2:B3",
                    "covered_native_cell_count": 4,
                    "confidence": "deterministic",
                    "from": {"cell": "A2", "column": "A", "row": 2},
                    "to": {"cell": "B3", "column": "B", "row": 3},
                    "coordinate_delta": {"columns": 1, "rows": 1},
                },
                "visual_effect": {
                    "cell_visibility": "obscures",
                    "semantic_replacement_suspected": True,
                },
            }
        ],
    }

    root = ET.fromstring(
        build_xlsx_content_tree_xml(
            inspection,
            source_document="sample.xlsx",
            session_id="test-session",
        )
    )

    assert root.tag == "unified_document"
    assert root.get("schema_version") == "2.0"
    vision = root.find(".//vision_analysis")
    assert vision is not None
    assert vision.get("classification") == "data_overlay"
    assert vision.find("./visual_effect").get("semantic_replacement_suspected") == "true"
    coverage = vision.find("./spatial_coverage")
    assert coverage is not None
    assert coverage.get("grid_span") == "A2:B3"
    references = root.findall("./cross_reference_index/cross_reference")
    assert {item.get("target") for item in references} == {
        "xlsx-sheet-1-cell-A2",
        "xlsx-sheet-1-cell-B2",
        "xlsx-sheet-1-cell-A3",
        "xlsx-sheet-1-cell-B3",
    }
