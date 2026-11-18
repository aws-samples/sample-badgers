"""Document-wide report index coverage."""

from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


DEPLOYMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOYMENT / "badgers-foundation"))
REPORT_DIR = DEPLOYMENT / "lambdas" / "code" / "html_report_specialist"
sys.path.insert(0, str(REPORT_DIR))
_SPEC = importlib.util.spec_from_file_location(
    "document_index_handler", REPORT_DIR / "lambda_handler.py"
)
if not (_SPEC and _SPEC.loader):
    raise AssertionError
handler = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(handler)


def test_document_index_includes_outline_counts_keywords_and_escaped_text() -> None:
    index = handler.build_document_index(
        [
            {
                "page_number": "1",
                "summary": "A < B",
                "elements": [
                    {"tag": "Title", "text": "A & B"},
                    {"tag": "P", "text": "paragraph"},
                    {"tag": "P", "text": "another"},
                ],
                "keywords": {
                    "primary_keywords": [
                        {"keyword": "A & B", "prominence": "high", "intent": "informational"}
                    ],
                    "topics": [{"topic": "Research"}],
                    "semantic_clusters": [{"cluster_name": "Core concepts"}],
                },
            }
        ]
    )

    root = ET.fromstring(index)
    if not (root.findtext("./page/summary") == "A < B"):
        raise AssertionError
    if not (root.find("./page/outline/heading").attrib["tag"] == "TITLE"):
        raise AssertionError
    if not (root.find("./page/element_counts/element[@tag='P']").attrib["count"] == "2"):
        raise AssertionError
    if not (root.findtext("./page/keywords/keyword") == "A & B"):
        raise AssertionError


def test_reports_ui_uses_persisted_document_index_when_present() -> None:
    ui = (DEPLOYMENT.parent / "ui" / "src" / "components" / "Reports.jsx").read_text()
    renderer = (REPORT_DIR / "renderer.py").read_text()

    if not ("if (manifest.document_index) return manifest.document_index" in ui):
        raise AssertionError
    if not ('report.get("document_index")' in renderer):
        raise AssertionError
