"""Unit tests for the Textract OCR quality gate and canonical page spine."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

HANDLER_PATH = (
    Path(__file__).resolve().parents[2]
    / "lambdas"
    / "code"
    / "textract_ocr"
    / "lambda_handler.py"
)
# The handler needs foundation.job_state only when Lambda invokes it.  The
# quality-gate tests exercise pure functions and must not import the
# Linux-built deployment layer (whose Pillow wheel is intentionally not
# importable on a developer Mac).
foundation_stub = ModuleType("foundation")
foundation_stub.job_state = SimpleNamespace()
sys.modules.setdefault("foundation", foundation_stub)
spec = importlib.util.spec_from_file_location("textract_ocr_handler", HANDLER_PATH)
assert spec and spec.loader
textract_ocr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(textract_ocr)


def _line(text: str, confidence: float, top: float, left: float = 0.1) -> dict:
    return {
        "BlockType": "LINE",
        "Text": text,
        "Confidence": confidence,
        "Geometry": {"BoundingBox": {"Top": top, "Left": left}},
    }


def test_quality_gate_accepts_readable_textract_lines() -> None:
    response = {
        "Blocks": [
            _line("Market outlook remains stable", 98.5, 0.2),
            _line("Revenue increased by twelve percent", 97.0, 0.1),
        ]
    }

    quality = textract_ocr.assess_ocr_quality(response)

    assert quality["status"] == "passed"
    assert quality["line_count"] == 2
    assert quality["mean_line_confidence"] == 97.75


def test_quality_gate_escalates_blank_or_unreliable_ocr() -> None:
    quality = textract_ocr.assess_ocr_quality(
        {"Blocks": [_line("%%", 42.0, 0.1)]}
    )

    assert quality["status"] == "escalate_to_vision"
    assert quality["fallback_route"] == "vision_ai"
    assert "insufficient_readable_text" in quality["reasons"]
    assert "low_mean_line_confidence" in quality["reasons"]


def test_accepted_ocr_builds_a_schema_valid_badgers_spine(
) -> None:
    response = {
        "Blocks": [
            _line("Second line", 99.0, 0.4),
            _line("First line", 99.0, 0.1),
        ]
    }
    quality = textract_ocr.assess_ocr_quality(response)
    xml_payload = textract_ocr.build_content_tree_xml(
        page_number=7,
        session_id="test-session",
        source_image="s3://example-bucket/temp/page_007.b64",
        output_uri="s3://example-bucket/temp/ocr/page_007.xml",
        response=response,
        quality=quality,
    )
    valid, errors = textract_ocr.validate_content_tree(xml_payload)

    assert valid is True
    assert errors == []
    assert xml_payload.index(b"First line") < xml_payload.index(b"Second line")
