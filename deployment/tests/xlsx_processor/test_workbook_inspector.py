"""Tests for read-only native XLSX package inspection."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pytest


PROCESSOR_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "lambdas" / "containers" / "xlsx_processor"
)
sys.path.insert(0, str(PROCESSOR_DIRECTORY))

from workbook_inspector import XlsxProcessingError, inspect_xlsx  # noqa: E402


def _workbook(*, macro: bool = False) -> bytes:
    files = {
        "[Content_Types].xml": """\
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
</Types>""",
        "xl/workbook.xml": """\
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <definedNames><definedName name="Budget">Revenue!$B$2:$B$3</definedName></definedNames>
 <sheets><sheet name="Revenue" sheetId="1" r:id="rId1"/></sheets>
</workbook>""",
        "xl/_rels/workbook.xml.rels": """\
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Target="worksheets/sheet1.xml"
  Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>
</Relationships>""",
        "xl/sharedStrings.xml": """\
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
 <si><t>Amount</t></si><si><t>Total revenue</t></si>
</sst>""",
        "xl/worksheets/sheet1.xml": """\
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <dimension ref="A1:B4"/>
 <sheetData>
  <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>0</v></c></row>
  <row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2"><v>10</v></c></row>
  <row r="3"><c r="A3" t="inlineStr"><is><t>March</t></is></c><c r="B3"><v>20</v></c></row>
  <row r="4"><c r="A4" t="s"><v>1</v></c><c r="B4"><f>SUM(B2:B3)</f><v>30</v></c></row>
 </sheetData>
 <drawing r:id="rId1"/>
 <tableParts count="1"><tablePart r:id="rId2"/></tableParts>
</worksheet>""",
        "xl/worksheets/_rels/sheet1.xml.rels": """\
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Target="../drawings/drawing1.xml"
  Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing"/>
 <Relationship Id="rId2" Target="../tables/table1.xml"
  Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/table"/>
</Relationships>""",
        "xl/tables/table1.xml": """\
<table xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 id="1" name="RevenueTable" displayName="RevenueTable" ref="A1:B4" headerRowCount="1">
 <tableColumns count="2"><tableColumn id="1" name="Period"/><tableColumn id="2" name="Amount"/></tableColumns>
 <tableStyleInfo name="TableStyleMedium2" showFirstColumn="0" showLastColumn="0"
  showRowStripes="1" showColumnStripes="0"/>
</table>""",
        "xl/drawings/drawing1.xml": """\
<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <xdr:twoCellAnchor>
  <xdr:from><xdr:col>0</xdr:col><xdr:row>1</xdr:row></xdr:from>
  <xdr:to><xdr:col>1</xdr:col><xdr:row>2</xdr:row></xdr:to>
  <xdr:pic><xdr:nvPicPr><xdr:cNvPr id="7" name="Updated totals" descr="New values"/></xdr:nvPicPr>
   <xdr:blipFill><a:blip r:embed="rId1"/></xdr:blipFill></xdr:pic>
 </xdr:twoCellAnchor>
</xdr:wsDr>""",
        "xl/drawings/_rels/drawing1.xml.rels": """\
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Target="../media/image1.png"
  Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"/>
</Relationships>""",
        "xl/media/image1.png": b"\x89PNG\r\n\x1a\nsynthetic-image",
    }
    if macro:
        files["xl/vbaProject.bin"] = b"not-executed"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    return output.getvalue()


def test_inspect_xlsx_preserves_formula_inputs_without_calculating() -> None:
    result = inspect_xlsx(_workbook())

    assert result["schema_version"] == "badgers-xlsx-analysis/v1"
    sheet = result["sheets"][0]
    formula = sheet["formulas"][0]
    assert sheet["name"] == "Revenue"
    assert formula["coordinate"] == "B4"
    assert formula["formula"] == "=SUM(B2:B3)"
    assert formula["cached_value"] == "30"
    assert formula["cached_value_state"] == "present"
    assert formula["functions"] == ["SUM"]
    assert formula["references"] == [
        {
            "reference": "Revenue!B2:B3",
            "kind": "cell_or_range",
            "represented_by": "Amount",
            "resolved_cells": [
                {"coordinate": "B2", "value": "10", "formula": ""},
                {"coordinate": "B3", "value": "20", "formula": ""},
            ],
        }
    ]
    interpretation = formula["interpretation"]
    assert interpretation["how_it_works"].startswith("Adds the values")
    assert interpretation["appears_to_do"] == "Appears to produce the value labelled 'Total revenue'."
    assert "why" not in interpretation
    assert interpretation["confidence_percent"] == 100
    assert interpretation["confidence_grade"] == "A"
    assert result["analysis_metadata"]["formula_calculation"] == "not_performed"
    assert result["analysis_metadata"]["formula_interpretation_confidence_scale"]["ratings"][-1] == {
        "minimum_percent": 100,
        "maximum_percent": 100,
        "grade": "A",
        "meaning": "PERFECT — complete native evidence supports the interpretation",
    }


def test_inspect_xlsx_records_image_overlay_and_native_cells() -> None:
    result = inspect_xlsx(_workbook())

    image = result["sheets"][0]["visual_objects"][0]
    assert image["kind"] == "image"
    assert image["object_id"] == "drawing-7"
    assert image["vision_status"] == "pending_contextual_analysis"
    assert image["covered_cells"] == [
        {"coordinate": "A2", "value": "Total revenue"},
        {"coordinate": "B2", "value": "10"},
        {"coordinate": "A3", "value": "March"},
        {"coordinate": "B3", "value": "20"},
    ]


def test_inspect_xlsx_wraps_table_cells_with_native_context() -> None:
    result = inspect_xlsx(_workbook())

    table = result["sheets"][0]["tables"][0]
    assert table["object_type"] == "table"
    assert table["object_id"] == "table-1"
    assert table["headers"] == ["Period", "Amount"]
    assert table["header_row"] == {
        "worksheet_row": 1,
        "cells": [
            {"column": "A", "coordinate": "A1", "value": "Amount", "data_type": "s", "style_id": ""},
            {"column": "B", "coordinate": "B1", "value": "Amount", "data_type": "s", "style_id": ""},
        ],
    }
    assert [row["worksheet_row"] for row in table["data_rows"]] == [2, 3, 4]
    assert table["data_rows"][0] == {
        "row_index": 1,
        "worksheet_row": 2,
        "cells": [
            {"column": "A", "coordinate": "A2", "value": "Total revenue", "data_type": "s", "style_id": ""},
            {"column": "B", "coordinate": "B2", "value": "10", "data_type": "n", "style_id": ""},
        ],
    }
    assert table["total_row"] is None
    assert table["self_context"]["formula_cells"] == ["B4"]
    assert [cell["coordinate"] for cell in table["data_rows"][-1]["cells"]] == [
        "A4", "B4"
    ]
    assert table["structured_formatting"]["style_name"] == "TableStyleMedium2"
    assert table["structured_formatting"]["show_row_stripes"] is True
    assert table["self_context"]["data_row_count"] == 3
    assert "Period, Amount" in table["self_context"]["summary"]
    assert table["sheet_context"]["sheet_name"] == "Revenue"


def test_inspect_xlsx_rejects_macro_parts_without_executing_them() -> None:
    with pytest.raises(XlsxProcessingError, match="Macro-enabled"):
        inspect_xlsx(_workbook(macro=True))
