"""Build the canonical BADGERS XML content tree for native XLSX evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from xml.etree import ElementTree as ET


def build_xlsx_content_tree_xml(
    inspection: dict[str, Any],
    *,
    source_document: str,
    session_id: str,
) -> bytes:
    """Merge native inspection and optional Vision evidence into BADGERS XML.

    Native cells and formulas remain authoritative. Vision is represented only
    as an enrichment of the matching anchored visual object, with deterministic
    spatial coverage and model-derived visual effect kept distinct.
    """
    root = ET.Element(
        "unified_document",
        {
            "schema_version": "2.0",
            "source_format": "xlsx",
            "processing_mode": "native_xlsx",
            "session_id": session_id,
        },
    )
    metadata = ET.SubElement(root, "metadata")
    ET.SubElement(metadata, "source_document").text = source_document
    ET.SubElement(metadata, "correlation_timestamp").text = datetime.now(UTC).isoformat()
    guarantees = ET.SubElement(metadata, "processing_guarantees")
    for name, value in sorted((inspection.get("processing_guarantees") or {}).items()):
        ET.SubElement(guarantees, name).text = str(value).lower()
    executed = ET.SubElement(metadata, "specialists_executed")
    ET.SubElement(executed, "specialist", {"name": "xlsx_processor", "s3_uri": ""})
    vision = inspection.get("visual_analysis")
    if isinstance(vision, dict) and vision.get("status") == "complete":
        ET.SubElement(
            executed,
            "specialist",
            {
                "name": "xlsx_processor_vision",
                "model_id": str(vision.get("model_id", "")),
                "s3_uri": "",
            },
        )

    analysis_metadata = inspection.get("analysis_metadata") or {}
    ET.SubElement(
        metadata,
        "formula_processing",
        {
            "calculation": str(analysis_metadata.get("formula_calculation", "not_performed")),
            "execution": str(analysis_metadata.get("formula_execution", "not_performed")),
        },
    )

    vision_by_key = _vision_by_sheet_and_object(vision)
    content_tree = ET.SubElement(root, "content_tree")
    cross_reference_index = ET.SubElement(root, "cross_reference_index")
    total_cells = 0
    total_visuals = 0

    for sheet_order, sheet in enumerate(inspection.get("sheets", []), start=1):
        if not isinstance(sheet, dict):
            continue
        sheet_id = f"xlsx-sheet-{sheet_order}"
        section = ET.SubElement(
            content_tree,
            "sect",
            {
                "id": sheet_id,
                "name": str(sheet.get("name", "")),
                "state": str(sheet.get("state", "visible")),
            },
        )
        cells = {
            str(cell.get("coordinate", "")): cell
            for cell in sheet.get("cells", [])
            if isinstance(cell, dict) and cell.get("coordinate")
        }
        formulas = {
            str(formula.get("coordinate", "")): formula
            for formula in sheet.get("formulas", [])
            if isinstance(formula, dict) and formula.get("coordinate")
        }
        table_coordinates: set[str] = set()
        element_order = 0

        for table in sheet.get("tables", []):
            if not isinstance(table, dict):
                continue
            element_order += 1
            table_element = ET.SubElement(
                section,
                "element",
                {
                    "id": f"{sheet_id}-{_safe_id(str(table.get('object_id', 'table')))}",
                    "tag": "Table",
                    "order": str(element_order),
                    "provenance": "native_xlsx",
                    "range": str(table.get("range", "")),
                    "name": str(table.get("name", "")),
                },
            )
            _append_table_context(table_element, table)
            children = ET.SubElement(table_element, "children")
            for row_order, row in enumerate(_table_rows(table), start=1):
                row_element = ET.SubElement(
                    children,
                    "element",
                    {
                        "id": f"{table_element.get('id')}-row-{row_order}",
                        "tag": "TR",
                        "order": str(row_order),
                        "provenance": "native_xlsx",
                        "worksheet_row": str(row.get("worksheet_row", "")),
                    },
                )
                row_children = ET.SubElement(row_element, "children")
                for column_order, cell_record in enumerate(row.get("cells", []), start=1):
                    if not isinstance(cell_record, dict):
                        continue
                    coordinate = str(cell_record.get("coordinate", ""))
                    if coordinate:
                        table_coordinates.add(coordinate)
                    _append_cell_element(
                        row_children,
                        cell_record,
                        formulas.get(coordinate),
                        cell_id=_cell_id(sheet_id, coordinate),
                        tag="TH" if row_order == 1 else "TD",
                        order=column_order,
                        scope="Column" if row_order == 1 else "",
                    )
                    total_cells += 1

        for coordinate, cell in sorted(cells.items(), key=lambda item: _cell_sort_key(item[0])):
            if coordinate in table_coordinates:
                continue
            element_order += 1
            _append_cell_element(
                section,
                cell,
                formulas.get(coordinate),
                cell_id=_cell_id(sheet_id, coordinate),
                tag="P",
                order=element_order,
            )
            total_cells += 1

        for visual in sheet.get("visual_objects", []):
            if not isinstance(visual, dict) or not visual.get("object_id"):
                continue
            element_order += 1
            visual_id = _visual_id(sheet_id, str(visual["object_id"]))
            figure = ET.SubElement(
                section,
                "element",
                {
                    "id": visual_id,
                    "tag": "Figure",
                    "order": str(element_order),
                    "provenance": "native_xlsx",
                    "object_kind": str(visual.get("kind", "")),
                    "drawing_name": str(visual.get("name", "")),
                },
            )
            _append_native_visual(figure, visual)
            analysis = vision_by_key.get((str(sheet.get("name", "")), str(visual["object_id"])))
            if analysis:
                _append_vision_enrichment(
                    figure,
                    analysis,
                    sheet_id=sheet_id,
                    native_coordinates=set(cells),
                    cross_reference_index=cross_reference_index,
                )
            total_visuals += 1

    raw_outputs = ET.SubElement(root, "raw_outputs")
    ET.SubElement(raw_outputs, "output", {"specialist": "xlsx_processor", "s3_uri": ""})
    if isinstance(vision, dict) and vision.get("status") == "complete":
        ET.SubElement(
            raw_outputs,
            "output",
            {
                "specialist": "xlsx_processor_vision",
                "model_id": str(vision.get("model_id", "")),
                "s3_uri": "",
            },
        )
    ET.SubElement(root, "summary").text = (
        f"Read-only native XLSX analysis across {len(inspection.get('sheets', []))} sheet(s), "
        f"{total_cells} cell(s), and {total_visuals} visual object(s)."
    )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)


def _append_table_context(parent: ET.Element, table: dict[str, Any]) -> None:
    context = table.get("self_context") or {}
    ET.SubElement(
        parent,
        "table_context",
        {
            "data_row_count": str(context.get("data_row_count", 0)),
            "column_count": str(context.get("column_count", 0)),
            "summary": str(context.get("summary", "")),
        },
    )
    formatting = table.get("structured_formatting") or {}
    ET.SubElement(
        parent,
        "structured_formatting",
        {key: str(value).lower() for key, value in formatting.items()},
    )


def _append_cell_element(
    parent: ET.Element,
    cell: dict[str, Any],
    formula: dict[str, Any] | None,
    *,
    cell_id: str,
    tag: str,
    order: int,
    scope: str = "",
) -> None:
    attributes = {
        "id": cell_id,
        "tag": tag,
        "order": str(order),
        "provenance": "native_xlsx",
        "coordinate": str(cell.get("coordinate", "")),
        "data_type": str(cell.get("data_type", "")),
        "style_id": str(cell.get("style_id", "")),
    }
    if scope:
        attributes["scope"] = scope
    element = ET.SubElement(parent, "element", attributes)
    ET.SubElement(element, "text").text = str(cell.get("value", ""))
    if formula:
        enrichments = ET.SubElement(element, "enrichments")
        formula_element = ET.SubElement(
            enrichments,
            "formula",
            {
                "expression": str(formula.get("formula", "")),
                "cached_value_state": str(formula.get("cached_value_state", "")),
                "calculation": "not_performed",
                "execution": "not_performed",
            },
        )
        interpretation = formula.get("interpretation") or {}
        if interpretation:
            ET.SubElement(
                formula_element,
                "interpretation",
                {
                    "confidence_percent": str(interpretation.get("confidence_percent", 0)),
                    "confidence_grade": str(interpretation.get("confidence_grade", "F")),
                    "appears_to_do": str(interpretation.get("appears_to_do", "")),
                },
            ).text = str(interpretation.get("how_it_works", ""))


def _append_native_visual(parent: ET.Element, visual: dict[str, Any]) -> None:
    native = ET.SubElement(
        parent,
        "native_visual",
        {
            "object_id": str(visual.get("object_id", "")),
            "kind": str(visual.get("kind", "")),
            "z_order": str(visual.get("z_order", "")),
        },
    )
    if visual.get("description"):
        ET.SubElement(native, "alt_text").text = str(visual["description"])
    anchor = visual.get("anchor")
    if isinstance(anchor, dict):
        ET.SubElement(native, "anchor", _flatten_anchor(anchor))


def _append_vision_enrichment(
    parent: ET.Element,
    analysis: dict[str, Any],
    *,
    sheet_id: str,
    native_coordinates: set[str],
    cross_reference_index: ET.Element,
) -> None:
    enrichments = ET.SubElement(parent, "enrichments")
    visual = analysis.get("visual_effect") or {}
    analysis_element = ET.SubElement(
        enrichments,
        "vision_analysis",
        {
            "classification": str(analysis.get("classification", "unknown")),
            "confidence_percent": str(analysis.get("confidence_percent", 0)),
            "confidence_grade": str(analysis.get("confidence_grade", "F")),
            "requires_human_review": str(
                bool(analysis.get("requires_human_review", False))
            ).lower(),
            "analysis_scope": str(analysis.get("analysis_scope", "")),
        },
    )
    ET.SubElement(analysis_element, "visual_summary").text = str(
        analysis.get("visual_summary", "")
    )
    ET.SubElement(analysis_element, "relationship_to_covered_cells").text = str(
        analysis.get("relationship_to_covered_cells", "")
    )
    ET.SubElement(
        analysis_element,
        "visual_effect",
        {
            "cell_visibility": str(visual.get("cell_visibility", "unknown")),
            "semantic_replacement_suspected": str(
                bool(visual.get("semantic_replacement_suspected", False))
            ).lower(),
        },
    )
    visible_text = analysis.get("visible_text") or []
    if visible_text:
        visible = ET.SubElement(analysis_element, "visible_text")
        for value in visible_text:
            ET.SubElement(visible, "text").text = str(value)

    coverage = analysis.get("spatial_coverage") or {}
    coverage_element = ET.SubElement(
        analysis_element,
        "spatial_coverage",
        {
            key: str(value)
            for key, value in coverage.items()
            if key not in {"from", "to", "coordinate_delta"}
        },
    )
    for key in ("from", "to"):
        endpoint = coverage.get(key)
        if isinstance(endpoint, dict):
            ET.SubElement(
                coverage_element,
                key,
                {name: str(value) for name, value in endpoint.items()},
            )
    delta = coverage.get("coordinate_delta")
    if isinstance(delta, dict):
        ET.SubElement(
            coverage_element,
            "coordinate_delta",
            {name: str(value) for name, value in delta.items()},
        )

    for coordinate in _covered_coordinates(analysis):
        if coordinate not in native_coordinates:
            continue
        ET.SubElement(
            cross_reference_index,
            "cross_reference",
            {
                "source": parent.get("id", ""),
                "target": _cell_id(sheet_id, coordinate),
                "type": "visually_overlays",
                "provenance": "native_anchor_plus_vision",
            },
        )


def _vision_by_sheet_and_object(vision: Any) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(vision, dict):
        return {}
    return {
        (str(item.get("sheet_name", "")), str(item.get("object_id", ""))): item
        for item in vision.get("analyses", [])
        if isinstance(item, dict)
    }


def _table_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    header = table.get("header_row")
    if isinstance(header, dict):
        rows.append(header)
    rows.extend(row for row in table.get("data_rows", []) if isinstance(row, dict))
    total = table.get("total_row")
    if isinstance(total, dict):
        rows.append(total)
    return rows


def _covered_coordinates(analysis: dict[str, Any]) -> list[str]:
    coverage = analysis.get("spatial_coverage")
    if not isinstance(coverage, dict):
        return []
    # Native cell values are deliberately not duplicated in Vision output. The
    # cross-reference range is resolved by the native inspection artifact.
    span = coverage.get("grid_span")
    if not isinstance(span, str) or ":" not in span:
        return []
    start, end = span.split(":", 1)
    return _range_coordinates(start, end)


def _range_coordinates(start: str, end: str) -> list[str]:
    start_parts, end_parts = _coordinate_parts(start), _coordinate_parts(end)
    if start_parts is None or end_parts is None:
        return []
    start_column, start_row = start_parts
    end_column, end_row = end_parts
    minimum_column, maximum_column = sorted((start_column, end_column))
    minimum_row, maximum_row = sorted((start_row, end_row))
    return [
        f"{_column_label(column)}{row}"
        for row in range(minimum_row, maximum_row + 1)
        for column in range(minimum_column, maximum_column + 1)
    ]


def _coordinate_parts(value: str) -> tuple[int, int] | None:
    letters = "".join(character for character in value.upper() if character.isalpha())
    digits = "".join(character for character in value if character.isdigit())
    if not letters or not digits:
        return None
    column = 0
    for character in letters:
        column = column * 26 + ord(character) - 64
    return column, int(digits)


def _column_label(column: int) -> str:
    label = ""
    while column:
        column, remainder = divmod(column - 1, 26)
        label = chr(65 + remainder) + label
    return label


def _cell_id(sheet_id: str, coordinate: str) -> str:
    return f"{sheet_id}-cell-{_safe_id(coordinate)}"


def _visual_id(sheet_id: str, object_id: str) -> str:
    return f"{sheet_id}-visual-{_safe_id(object_id)}"


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "-" for character in value)


def _cell_sort_key(coordinate: str) -> tuple[int, int]:
    parsed = _coordinate_parts(coordinate)
    return (parsed[1], parsed[0]) if parsed else (10**9, 10**9)


def _flatten_anchor(anchor: dict[str, Any]) -> dict[str, str]:
    result = {"type": str(anchor.get("type", ""))}
    for point_name in ("from", "to"):
        point = anchor.get(point_name)
        if isinstance(point, dict):
            result[f"{point_name}_column"] = str(point.get("column", ""))
            result[f"{point_name}_row"] = str(point.get("row", ""))
    return result
