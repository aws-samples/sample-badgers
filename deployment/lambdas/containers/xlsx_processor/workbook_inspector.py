"""Read-only, bounded inspection of an untrusted XLSX/OOXML workbook.

This module deliberately reads package parts rather than opening a spreadsheet
application or using a formula engine.  It never evaluates a formula, executes
workbook content, refreshes a connection, or follows an external reference.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
import zipfile
from io import BytesIO
from typing import Any

from defusedxml.ElementTree import fromstring
from defusedxml.common import DefusedXmlException


class XlsxProcessingError(ValueError):
    """Raised when a workbook is unsafe, malformed, or exceeds a hard limit."""


MAX_SOURCE_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 10_000
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_SHEETS = 200
MAX_CELLS = 100_000
MAX_CELL_TEXT = 16_384
MAX_REFERENCES_PER_FORMULA = 64
MAX_RESOLVED_CELLS_PER_REFERENCE = 32
MAX_VISUAL_OBJECTS = 2_000
MAX_TABLE_GRID_CELLS = 20_000

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
_DRAWING_MAIN_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_CHART_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"
_NS = {
    "main": _MAIN_NS,
    "r": _DOC_REL_NS,
    "pr": _PACKAGE_REL_NS,
    "xdr": _DRAWING_NS,
    "a": _DRAWING_MAIN_NS,
    "c": _CHART_NS,
}

_CELL_REFERENCE = re.compile(
    r"(?:(?P<sheet>'(?:[^']|'')+'|[A-Za-z_][A-Za-z0-9_. ]*)!)?"
    r"(?P<range>\$?[A-Z]{1,3}\$?\d+(?::\$?[A-Z]{1,3}\$?\d+)?)"
)
_STRUCTURED_REFERENCE = re.compile(r"(?P<table>[A-Za-z_][A-Za-z0-9_.]*)\[(?P<column>[^\]]+)\]")
_FUNCTION = re.compile(r"(?<![A-Za-z0-9_.])([A-Z][A-Z0-9._]*)\s*\(", re.IGNORECASE)
_DYNAMIC_REFERENCE_FUNCTIONS = frozenset({"INDIRECT", "OFFSET"})


def inspect_xlsx(payload: bytes) -> dict[str, Any]:
    """Return deterministic workbook facts without executing workbook content."""

    with _open_workbook(payload) as archive:
        names = set(archive.namelist())
        _reject_macro_content(names)
        workbook = _read_xml(archive, "xl/workbook.xml")
        relationships = _relationships(archive, "xl/workbook.xml")
        shared_strings = _shared_strings(archive, names)
        defined_names = _defined_names(workbook)
        sheets = _sheet_descriptors(workbook, relationships)
        if len(sheets) > MAX_SHEETS:
            raise XlsxProcessingError(f"Workbook exceeds the {MAX_SHEETS}-sheet limit.")

        parsed_sheets: list[dict[str, Any]] = []
        warnings = _workbook_warnings(names)
        total_cells = 0
        for descriptor in sheets:
            path = descriptor.pop("_path")
            if path not in names:
                warnings.append(f"missing_worksheet_part:{path}")
                descriptor["warnings"] = ["worksheet_part_missing"]
                descriptor["cells"] = []
                descriptor["formulas"] = []
                descriptor["tables"] = []
                descriptor["visual_objects"] = []
                parsed_sheets.append(descriptor)
                continue
            parsed = _inspect_sheet(archive, path, descriptor, shared_strings)
            total_cells += len(parsed["cells"])
            if total_cells > MAX_CELLS:
                raise XlsxProcessingError(f"Workbook exceeds the {MAX_CELLS}-cell limit.")
            parsed_sheets.append(parsed)

        _enrich_formula_references(parsed_sheets, defined_names)
        _append_unanchored_media(archive, names, parsed_sheets, warnings)

    return {
        "schema_version": "badgers-xlsx-analysis/v1",
        "source_format": "xlsx",
        "analysis_metadata": {
            "formula_calculation": "not_performed",
            "formula_execution": "not_performed",
            "formula_interpretation_confidence_scale": {
                "description": (
                    "A school-style confidence grade for BADGERS' read-only "
                    "interpretation of each formula. It measures how completely "
                    "native formula text, references, and labels support the "
                    "explanation; it is not a measure of calculated-value accuracy."
                ),
                "ratings": [
                    {
                        "minimum_percent": 0,
                        "maximum_percent": 0,
                        "grade": "F",
                        "meaning": "FAIL — no supported formula interpretation",
                    },
                    {"minimum_percent": 1, "maximum_percent": 59, "grade": "F"},
                    {"minimum_percent": 60, "maximum_percent": 69, "grade": "D"},
                    {"minimum_percent": 70, "maximum_percent": 79, "grade": "C"},
                    {"minimum_percent": 80, "maximum_percent": 89, "grade": "B"},
                    {"minimum_percent": 90, "maximum_percent": 99, "grade": "A"},
                    {
                        "minimum_percent": 100,
                        "maximum_percent": 100,
                        "grade": "A",
                        "meaning": "PERFECT — complete native evidence supports the interpretation",
                    },
                ],
            },
        },
        "workbook": {
            "sheet_count": len(parsed_sheets),
            "defined_names": defined_names,
            "external_links_detected": any(name.startswith("xl/externalLinks/") for name in names),
            "connections_detected": "xl/connections.xml" in names,
            "macros_detected": False,
            "interactive_features_detected": sorted(
                feature
                for feature, prefix in {
                    "activeX": "xl/activeX/",
                    "embedded_objects": "xl/embeddings/",
                    "pivot_tables": "xl/pivotTables/",
                    "slicers": "xl/slicers/",
                    "timelines": "xl/timelines/",
                }.items()
                if any(name.startswith(prefix) for name in names)
            ),
        },
        "sheets": parsed_sheets,
        "warnings": sorted(set(warnings)),
    }


def _open_workbook(payload: bytes) -> zipfile.ZipFile:
    if not payload:
        raise XlsxProcessingError("XLSX payload is empty.")
    if len(payload) > MAX_SOURCE_BYTES:
        raise XlsxProcessingError(
            f"XLSX source exceeds the {MAX_SOURCE_BYTES // (1024 * 1024)} MiB limit."
        )
    try:
        archive = zipfile.ZipFile(BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise XlsxProcessingError("XLSX payload is not a valid ZIP/OOXML package.") from error

    entries = archive.infolist()
    if len(entries) > MAX_ARCHIVE_ENTRIES:
        archive.close()
        raise XlsxProcessingError(f"XLSX exceeds the {MAX_ARCHIVE_ENTRIES}-archive-entry limit.")
    expanded = 0
    for entry in entries:
        if entry.flag_bits & 0x1:
            archive.close()
            raise XlsxProcessingError("Encrypted XLSX parts are not supported.")
        if entry.filename.startswith("/") or ".." in entry.filename.split("/"):
            archive.close()
            raise XlsxProcessingError("XLSX contains an unsafe archive entry path.")
        expanded += entry.file_size
        if expanded > MAX_UNCOMPRESSED_BYTES:
            archive.close()
            raise XlsxProcessingError(
                "XLSX uncompressed contents exceed the 256 MiB inspection limit."
            )
    required = {"[Content_Types].xml", "xl/workbook.xml"}
    missing = sorted(required - {entry.filename for entry in entries})
    if missing:
        archive.close()
        raise XlsxProcessingError(
            "XLSX package is missing required OOXML parts: " + ", ".join(missing)
        )
    return archive


def _reject_macro_content(names: set[str]) -> None:
    macro_parts = sorted(
        name
        for name in names
        if name.lower().endswith("vbaproject.bin")
        or name.lower().endswith("vbasignature.bin")
        or "macrosheet" in name.lower()
    )
    if macro_parts:
        raise XlsxProcessingError(
            "Macro-enabled workbook content is not supported; no macro was executed."
        )


def _read_xml(archive: zipfile.ZipFile, path: str):
    try:
        content = archive.read(path)
    except KeyError as error:
        raise XlsxProcessingError(f"XLSX package is missing required part: {path}") from error
    try:
        return fromstring(content)
    except DefusedXmlException as error:
        raise XlsxProcessingError(f"Unsafe XML was rejected in {path}.") from error
    except Exception as error:
        raise XlsxProcessingError(f"Invalid XML in {path}.") from error


def _relationships(archive: zipfile.ZipFile, part_path: str) -> dict[str, str]:
    directory, basename = posixpath.split(part_path)
    relationship_path = posixpath.join(directory, "_rels", f"{basename}.rels")
    if relationship_path not in archive.namelist():
        return {}
    root = _read_xml(archive, relationship_path)
    resolved: dict[str, str] = {}
    for element in root.findall("pr:Relationship", _NS):
        identifier = element.get("Id")
        target = element.get("Target")
        if (
            not identifier
            or not target
            or element.get("TargetMode", "").lower() == "external"
        ):
            continue
        target_path = _resolve_part_path(part_path, target)
        resolved[identifier] = target_path
    return resolved


def _resolve_part_path(part_path: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(posixpath.dirname(part_path), target))


def _shared_strings(archive: zipfile.ZipFile, names: set[str]) -> list[str]:
    if "xl/sharedStrings.xml" not in names:
        return []
    root = _read_xml(archive, "xl/sharedStrings.xml")
    values: list[str] = []
    for item in root.findall("main:si", _NS):
        values.append(_bounded_text("".join(item.itertext())))
    return values


def _defined_names(workbook: Any) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for item in workbook.findall("main:definedNames/main:definedName", _NS):
        name = item.get("name")
        if not name:
            continue
        values.append(
            {
                "name": name,
                "reference": _bounded_text((item.text or "").strip()),
                "scope_sheet_index": item.get("localSheetId", ""),
            }
        )
    return values


def _sheet_descriptors(
    workbook: Any, relationships: dict[str, str]
) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for index, element in enumerate(workbook.findall("main:sheets/main:sheet", _NS), start=1):
        relationship_id = element.get(f"{{{_DOC_REL_NS}}}id", "")
        target = relationships.get(relationship_id)
        if not target:
            target = ""
        descriptors.append(
            {
                "index": index,
                "name": element.get("name", f"Sheet{index}"),
                "state": element.get("state", "visible"),
                "_path": target,
            }
        )
    return descriptors


def _workbook_warnings(names: set[str]) -> list[str]:
    warnings: list[str] = []
    if any(name.startswith("xl/externalLinks/") for name in names):
        warnings.append("external_links_detected_not_dereferenced")
    if "xl/connections.xml" in names:
        warnings.append("connections_detected_not_refreshed")
    if any(name.startswith("xl/activeX/") for name in names):
        warnings.append("activex_detected_not_executed")
    if any(name.startswith("xl/embeddings/") for name in names):
        warnings.append("embedded_objects_detected_not_extracted")
    return warnings


def _inspect_sheet(
    archive: zipfile.ZipFile,
    path: str,
    descriptor: dict[str, Any],
    shared_strings: list[str],
) -> dict[str, Any]:
    root = _read_xml(archive, path)
    cells, formulas, cell_index = _cells(root, shared_strings)
    relationships = _relationships(archive, path)
    descriptor.update(
        {
            "dimensions": _dimension(root),
            "cells": cells,
            "formulas": formulas,
            "tables": _tables(
                archive,
                root,
                relationships,
                cell_index,
                sheet_name=str(descriptor["name"]),
                sheet_dimensions=_dimension(root),
            ),
            "merged_ranges": sorted(
                merge.get("ref", "")
                for merge in root.findall("main:mergeCells/main:mergeCell", _NS)
                if merge.get("ref")
            ),
            "visual_objects": _visual_objects(archive, root, relationships, cell_index),
            "warnings": [],
        }
    )
    return descriptor


def _dimension(root: Any) -> str:
    dimension = root.find("main:dimension", _NS)
    return dimension.get("ref", "") if dimension is not None else ""


def _cells(
    root: Any, shared_strings: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    cells: list[dict[str, Any]] = []
    formulas: list[dict[str, Any]] = []
    cell_index: dict[str, dict[str, Any]] = {}
    shared_formula_masters: dict[str, str] = {}
    for cell in root.findall(".//main:sheetData/main:row/main:c", _NS):
        coordinate = cell.get("r", "")
        if not coordinate:
            continue
        data_type = cell.get("t", "n")
        value = _cell_value(cell, data_type, shared_strings)
        formula_element = cell.find("main:f", _NS)
        cell_record = {
            "coordinate": coordinate,
            "value": value,
            "data_type": data_type,
            "style_id": cell.get("s", ""),
        }
        if formula_element is not None:
            formula_text = _formula_text(formula_element)
            shared_index = formula_element.get("si", "")
            if formula_element.get("t") == "shared":
                if formula_text:
                    shared_formula_masters[shared_index] = formula_text
                elif shared_index in shared_formula_masters:
                    formula_text = shared_formula_masters[shared_index]
            cell_record["formula"] = formula_text
            formulas.append(
                {
                    "coordinate": coordinate,
                    "formula": formula_text,
                    "formula_type": formula_element.get("t", "normal"),
                    "shared_formula_index": shared_index,
                    "cached_value": value,
                    "cached_value_state": "present" if value not in (None, "") else "missing",
                    "references": [],
                    "functions": _formula_functions(formula_text),
                    "dynamic_reference_possible": bool(
                        _DYNAMIC_REFERENCE_FUNCTIONS & set(_formula_functions(formula_text))
                    ),
                    "interpretation": {},
                }
            )
        cells.append(cell_record)
        cell_index[coordinate] = cell_record
    return cells, formulas, cell_index


def _cell_value(cell: Any, data_type: str, shared_strings: list[str]) -> Any:
    if data_type == "inlineStr":
        inline = cell.find("main:is", _NS)
        return _bounded_text("".join(inline.itertext())) if inline is not None else ""
    value_element = cell.find("main:v", _NS)
    raw = value_element.text if value_element is not None else ""
    raw = _bounded_text(raw or "")
    if data_type == "s":
        try:
            return shared_strings[int(raw)]
        except (ValueError, IndexError):
            return {"unresolved_shared_string_index": raw}
    if data_type == "b":
        return raw == "1"
    return raw


def _formula_text(element: Any) -> str:
    text = (element.text or "").strip()
    return f"={text}" if text and not text.startswith("=") else text


def _formula_functions(formula: str) -> list[str]:
    return sorted({match.group(1).upper() for match in _FUNCTION.finditer(formula)})


def _tables(
    archive: zipfile.ZipFile,
    root: Any,
    relationships: dict[str, str],
    cells: dict[str, dict[str, Any]],
    *,
    sheet_name: str,
    sheet_dimensions: str,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for table_part in root.findall("main:tableParts/main:tablePart", _NS):
        relationship_id = table_part.get(f"{{{_DOC_REL_NS}}}id", "")
        path = relationships.get(relationship_id)
        if not path or path not in archive.namelist():
            values.append({"warning": "table_relationship_missing"})
            continue
        table = _read_xml(archive, path)
        table_range = table.get("ref", "")
        table_cells = _cells_in_range(cells, table_range)
        columns = [
            _bounded_text(column.get("name", ""))
            for column in table.findall("main:tableColumns/main:tableColumn", _NS)
            if column.get("name")
        ]
        style = table.find("main:tableStyleInfo", _NS)
        table_row_count, column_count = _range_shape(table_range)
        header_rows = int(table.get("headerRowCount", "1") or "1")
        total_rows = int(table.get("totalsRowCount", "0") or "0")
        formula_cells = [
            cell["coordinate"] for cell in table_cells if cell.get("formula")
        ]
        header_row, data_rows, total_row = _table_grid(
            cells, table_range, header_rows, total_rows
        )
        values.append(
            {
                "object_type": "table",
                "object_id": f"table-{table.get('id', '') or table.get('displayName', '')}",
                "name": table.get("displayName") or table.get("name", ""),
                "range": table_range,
                "headers": columns,
                "header_row": header_row,
                "data_rows": data_rows,
                "total_row": total_row,
                "structured_formatting": {
                    "style_name": style.get("name", "") if style is not None else "",
                    "show_first_column": _as_bool(
                        style.get("showFirstColumn", "0") if style is not None else "0"
                    ),
                    "show_last_column": _as_bool(
                        style.get("showLastColumn", "0") if style is not None else "0"
                    ),
                    "show_row_stripes": _as_bool(
                        style.get("showRowStripes", "0") if style is not None else "0"
                    ),
                    "show_column_stripes": _as_bool(
                        style.get("showColumnStripes", "0") if style is not None else "0"
                    ),
                    "header_row_count": header_rows,
                    "totals_row_count": total_rows,
                },
                "self_context": {
                    "summary": _table_summary(
                        table.get("displayName") or table.get("name", ""),
                        columns,
                        max(table_row_count - header_rows - total_rows, 0),
                        formula_cells,
                    ),
                    "data_row_count": max(table_row_count - header_rows - total_rows, 0),
                    "column_count": column_count,
                    "formula_cells": formula_cells,
                },
                "sheet_context": {
                    "summary": _table_sheet_summary(
                        table_range, sheet_name, sheet_dimensions, len(cells)
                    ),
                    "sheet_name": sheet_name,
                    "sheet_dimensions": sheet_dimensions,
                    "populated_cells_in_sheet": len(cells),
                },
            }
        )
    return values


def _visual_objects(
    archive: zipfile.ZipFile,
    worksheet: Any,
    relationships: dict[str, str],
    cells: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for drawing in worksheet.findall("main:drawing", _NS):
        relationship_id = drawing.get(f"{{{_DOC_REL_NS}}}id", "")
        drawing_path = relationships.get(relationship_id)
        if not drawing_path or drawing_path not in archive.namelist():
            objects.append({"warning": "drawing_relationship_missing"})
            continue
        drawing_relationships = _relationships(archive, drawing_path)
        root = _read_xml(archive, drawing_path)
        for z_order, anchor in enumerate(list(root), start=1):
            if len(objects) >= MAX_VISUAL_OBJECTS:
                raise XlsxProcessingError(
                    f"Workbook exceeds the {MAX_VISUAL_OBJECTS}-visual-object limit."
                )
            object_record = _drawing_object(
                archive, anchor, drawing_relationships, cells, z_order
            )
            if object_record:
                objects.append(object_record)
    return objects


def _drawing_object(
    archive: zipfile.ZipFile,
    anchor: Any,
    relationships: dict[str, str],
    cells: dict[str, dict[str, Any]],
    z_order: int,
) -> dict[str, Any] | None:
    anchor_bounds = _anchor_bounds(anchor)
    properties = anchor.find(".//xdr:cNvPr", _NS)
    name = properties.get("name", "") if properties is not None else ""
    description = properties.get("descr", "") if properties is not None else ""
    identifier = properties.get("id", "") if properties is not None else str(z_order)
    picture = anchor.find(".//xdr:pic", _NS)
    chart = anchor.find(".//c:chart", _NS)
    shape = anchor.find(".//xdr:sp", _NS)
    record: dict[str, Any] = {
        "object_id": f"drawing-{identifier}",
        "z_order": z_order,
        "name": _bounded_text(name),
        "description": _bounded_text(description),
        "anchor": anchor_bounds,
        "covered_cells": _covered_cells(cells, anchor_bounds),
    }
    if picture is not None:
        blip = picture.find(".//a:blip", _NS)
        embed = blip.get(f"{{{_DOC_REL_NS}}}embed", "") if blip is not None else ""
        media_path = relationships.get(embed, "")
        record.update({"kind": "image", "media_path": media_path})
        if media_path and media_path in archive.namelist():
            media = archive.read(media_path)
            record.update(
                {
                    "media_sha256": hashlib.sha256(media).hexdigest(),
                    "media_bytes": len(media),
                    "media_content_type": _media_content_type(media_path),
                    "vision_status": "pending_contextual_analysis",
                }
            )
        else:
            record["warning"] = "image_media_missing"
        return record
    if chart is not None:
        relationship_id = chart.get(f"{{{_DOC_REL_NS}}}id", "")
        chart_path = relationships.get(relationship_id, "")
        record.update({"kind": "chart", "chart_path": chart_path, "vision_status": "pending_contextual_analysis"})
        return record
    if shape is not None:
        text = "".join(shape.itertext()).strip()
        record.update({"kind": "shape", "text": _bounded_text(text)})
        return record
    if anchor.findall(".//xdr:graphicFrame", _NS):
        record.update({"kind": "graphic_frame", "warning": "unsupported_graphic_frame"})
        return record
    return None


def _anchor_bounds(anchor: Any) -> dict[str, Any]:
    start = anchor.find("xdr:from", _NS)
    end = anchor.find("xdr:to", _NS)
    if start is None:
        return {"type": "absolute_or_unknown"}

    def coordinate(node: Any) -> dict[str, int]:
        column = int((node.findtext("xdr:col", default="0", namespaces=_NS) or "0")) + 1
        row = int((node.findtext("xdr:row", default="0", namespaces=_NS) or "0")) + 1
        return {"column": column, "row": row}

    result: dict[str, Any] = {"type": "cell_anchor", "from": coordinate(start)}
    result["to"] = coordinate(end) if end is not None else dict(result["from"])
    return result


def _covered_cells(
    cells: dict[str, dict[str, Any]], anchor: dict[str, Any]
) -> list[dict[str, Any]]:
    start = anchor.get("from")
    end = anchor.get("to")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return []
    values: list[dict[str, Any]] = []
    min_column, max_column = sorted((int(start["column"]), int(end["column"])))
    min_row, max_row = sorted((int(start["row"]), int(end["row"])))
    for coordinate, cell in cells.items():
        parsed = _coordinate_parts(coordinate)
        if parsed is None:
            continue
        column, row = parsed
        if min_column <= column <= max_column and min_row <= row <= max_row:
            values.append({"coordinate": coordinate, "value": cell.get("value", "")})
    return sorted(values, key=lambda item: _coordinate_sort_key(item["coordinate"]))[
        :MAX_RESOLVED_CELLS_PER_REFERENCE
    ]


def _append_unanchored_media(
    archive: zipfile.ZipFile,
    names: set[str],
    sheets: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    linked = {
        item.get("media_path")
        for sheet in sheets
        for item in sheet.get("visual_objects", [])
        if isinstance(item, dict)
    }
    unanchored = sorted(name for name in names if name.startswith("xl/media/") and name not in linked)
    if unanchored:
        warnings.append("unanchored_media_detected")
    if not sheets:
        return
    for index, path in enumerate(unanchored, start=1):
        media = archive.read(path)
        sheets[0]["visual_objects"].append(
            {
                "object_id": f"unanchored-media-{index}",
                "kind": "unanchored_image",
                "media_path": path,
                "media_sha256": hashlib.sha256(media).hexdigest(),
                "media_bytes": len(media),
                "media_content_type": _media_content_type(path),
                "vision_status": "pending_contextual_analysis",
                "warning": "image_has_no_resolved_sheet_anchor",
            }
        )


def _media_content_type(path: str) -> str:
    suffix = posixpath.splitext(path.lower())[1]
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".svg": "image/svg+xml",
    }.get(suffix, "application/octet-stream")


def _enrich_formula_references(
    sheets: list[dict[str, Any]], defined_names: list[dict[str, str]]
) -> None:
    by_name = {str(sheet["name"]): sheet for sheet in sheets}
    names = {item["name"]: item for item in defined_names}
    for sheet in sheets:
        cells = {str(cell["coordinate"]): cell for cell in sheet["cells"]}
        for formula in sheet["formulas"]:
            references = _formula_references(
                str(formula["formula"]), str(sheet["name"]), by_name, names
            )
            formula["references"] = references
            formula["interpretation"] = _formula_interpretation(
                formula, references, cells, str(sheet["name"])
            )


def _formula_references(
    formula: str,
    current_sheet: str,
    sheets: dict[str, dict[str, Any]],
    names: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for match in _CELL_REFERENCE.finditer(formula):
        if len(values) >= MAX_REFERENCES_PER_FORMULA:
            break
        sheet_name = _unquote_sheet_name(match.group("sheet") or "") or current_sheet
        reference = match.group("range")
        target = sheets.get(sheet_name)
        values.append(
            {
                "reference": f"{sheet_name}!{reference}",
                "kind": "cell_or_range",
                "represented_by": _range_label(target, reference),
                "resolved_cells": _resolve_cells(target, reference),
            }
        )
    direct_reference_names = {item["reference"].split("!", 1)[-1] for item in values}
    for match in _STRUCTURED_REFERENCE.finditer(formula):
        token = match.group(0)
        if token in direct_reference_names or len(values) >= MAX_REFERENCES_PER_FORMULA:
            continue
        values.append(
            {
                "reference": token,
                "kind": "structured_table_reference",
                "represented_by": match.group("column"),
                "resolved_cells": [],
            }
        )
    for name, defined in names.items():
        if re.search(rf"(?<![A-Za-z0-9_.]){re.escape(name)}(?![A-Za-z0-9_.])", formula):
            if len(values) >= MAX_REFERENCES_PER_FORMULA:
                break
            values.append(
                {
                    "reference": name,
                    "kind": "defined_name",
                    "represented_by": name,
                    "defined_reference": defined["reference"],
                    "resolved_cells": [],
                }
            )
    return values


def _unquote_sheet_name(value: str) -> str:
    value = value.rstrip("!")
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    return value


def _range_label(sheet: dict[str, Any] | None, reference: str) -> str:
    if not sheet:
        return "unresolved_sheet"
    start = reference.replace("$", "").split(":", 1)[0]
    column, row = _coordinate_parts(start) or (0, 0)
    cells = {str(cell["coordinate"]): cell for cell in sheet.get("cells", [])}
    label_candidates = (
        _coordinate(column, max(row - 1, 1)),
        _coordinate(max(column - 1, 1), row),
    )
    for candidate in label_candidates:
        value = cells.get(candidate, {}).get("value")
        if _is_context_label(value):
            return _bounded_text(value)
    return ""


def _resolve_cells(sheet: dict[str, Any] | None, reference: str) -> list[dict[str, Any]]:
    if not sheet:
        return []
    start_text, _, end_text = reference.replace("$", "").partition(":")
    start = _coordinate_parts(start_text)
    end = _coordinate_parts(end_text or start_text)
    if start is None or end is None:
        return []
    min_column, max_column = sorted((start[0], end[0]))
    min_row, max_row = sorted((start[1], end[1]))
    indexed = {str(cell["coordinate"]): cell for cell in sheet.get("cells", [])}
    values: list[dict[str, Any]] = []
    for row in range(min_row, max_row + 1):
        for column in range(min_column, max_column + 1):
            value = indexed.get(_coordinate(column, row))
            if value is None:
                continue
            values.append(
                {
                    "coordinate": value["coordinate"],
                    "value": value.get("value", ""),
                    "formula": value.get("formula", ""),
                }
            )
            if len(values) >= MAX_RESOLVED_CELLS_PER_REFERENCE:
                return values
    return values


def _formula_interpretation(
    formula: dict[str, Any],
    references: list[dict[str, Any]],
    cells: dict[str, dict[str, Any]],
    sheet_name: str,
) -> dict[str, Any]:
    coordinate = str(formula["coordinate"])
    label = _range_label({"cells": list(cells.values())}, coordinate)
    functions = [str(item) for item in formula["functions"]]
    reference_words = [
        f"{item['reference']} ({item['represented_by'] or 'unlabelled'})"
        for item in references[:4]
    ]
    how_it_works = _formula_mechanics(formula, reference_words)
    appears_to_do = _formula_purpose(label, functions, reference_words)
    confidence_percent = _formula_confidence(formula, references, label)
    return {
        "how_it_works": how_it_works,
        "appears_to_do": appears_to_do,
        "confidence_percent": confidence_percent,
        "confidence_grade": _confidence_grade(confidence_percent),
        "evidence": {
            "formula_cell": f"{sheet_name}!{coordinate}",
            "output_label": label,
            "reference_labels": [
                {
                    "reference": item["reference"],
                    "represented_by": item["represented_by"],
                }
                for item in references
            ],
        },
    }


def _formula_mechanics(formula: dict[str, Any], references: list[str]) -> str:
    functions = set(str(item) for item in formula["functions"])
    expression = str(formula["formula"])
    source = ", ".join(references) or "no statically resolved input"
    if "SUM" in functions:
        return f"Adds the values represented by {source}."
    if "/" in expression:
        return f"Divides one referenced value by another expression using {source}."
    if "-" in expression:
        return f"Subtracts referenced values using {source}."
    if "+" in expression:
        return f"Adds referenced values using {source}."
    if "*" in expression:
        return f"Multiplies referenced values using {source}."
    if functions:
        return f"Applies {', '.join(sorted(functions))} to {source}."
    return f"Uses the expression {expression} with {source}."


def _formula_purpose(label: str, functions: list[str], references: list[str]) -> str:
    if label:
        return f"Appears to produce the value labelled '{label}'."
    if "SUM" in functions:
        return "Appears to calculate a total for the referenced values."
    if references:
        return "Appears to derive a value from the referenced workbook inputs."
    return "Purpose cannot be determined from native formula evidence alone."


def _formula_confidence(
    formula: dict[str, Any], references: list[dict[str, Any]], label: str
) -> int:
    """Score the evidence supporting an interpretation, never a calculated value."""

    if not str(formula.get("formula") or ""):
        return 0
    score = 65
    if references:
        score += 15
    else:
        score -= 20
    if label:
        score += 10
    if references and all(item.get("represented_by") for item in references):
        score += 10
    if any(not item.get("resolved_cells") for item in references if item["kind"] == "cell_or_range"):
        score -= 15
    if formula.get("dynamic_reference_possible"):
        score -= 35
    if formula.get("formula_type") == "shared" and not formula.get("formula"):
        return 0
    return max(0, min(score, 100))


def _confidence_grade(percent: int) -> str:
    if percent < 60:
        return "F"
    if percent < 70:
        return "D"
    if percent < 80:
        return "C"
    if percent < 90:
        return "B"
    return "A"


def _coordinate_parts(value: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"\$?([A-Z]{1,3})\$?(\d+)", value.upper())
    if not match:
        return None
    column = 0
    for character in match.group(1):
        column = column * 26 + ord(character) - ord("A") + 1
    return column, int(match.group(2))


def _coordinate(column: int, row: int) -> str:
    letters = ""
    while column:
        column, remainder = divmod(column - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return f"{letters}{row}"


def _coordinate_sort_key(value: str) -> tuple[int, int]:
    column, row = _coordinate_parts(value) or (0, 0)
    return row, column


def _bounded_text(value: str) -> str:
    return value[:MAX_CELL_TEXT]


def _cells_in_range(
    cells: dict[str, dict[str, Any]], cell_range: str
) -> list[dict[str, Any]]:
    start_text, _, end_text = cell_range.replace("$", "").partition(":")
    start = _coordinate_parts(start_text)
    end = _coordinate_parts(end_text or start_text)
    if start is None or end is None:
        return []
    min_column, max_column = sorted((start[0], end[0]))
    min_row, max_row = sorted((start[1], end[1]))
    return [
        dict(cell)
        for coordinate, cell in sorted(cells.items(), key=lambda item: _coordinate_sort_key(item[0]))
        if (parts := _coordinate_parts(coordinate)) is not None
        and min_column <= parts[0] <= max_column
        and min_row <= parts[1] <= max_row
    ]


def _table_grid(
    cells: dict[str, dict[str, Any]],
    cell_range: str,
    header_rows: int,
    total_rows: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    """Represent a table in reader order, preserving its header/data hierarchy."""

    start_text, _, end_text = cell_range.replace("$", "").partition(":")
    start = _coordinate_parts(start_text)
    end = _coordinate_parts(end_text or start_text)
    if start is None or end is None:
        return None, [], None
    min_column, max_column = sorted((start[0], end[0]))
    min_row, max_row = sorted((start[1], end[1]))
    cell_count = (max_column - min_column + 1) * (max_row - min_row + 1)
    if cell_count > MAX_TABLE_GRID_CELLS:
        raise XlsxProcessingError(
            f"Table range {cell_range} exceeds the {MAX_TABLE_GRID_CELLS}-cell grid limit."
        )

    def row_record(worksheet_row: int, row_index: int | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "worksheet_row": worksheet_row,
            "cells": [
                _table_cell(cells.get(_coordinate(column, worksheet_row)), column, worksheet_row)
                for column in range(min_column, max_column + 1)
            ],
        }
        if row_index is not None:
            record["row_index"] = row_index
        return record

    header_row = row_record(min_row) if header_rows else None
    first_data_row = min_row + header_rows
    final_data_row = max_row - total_rows
    data_rows = [
        row_record(worksheet_row, worksheet_row - first_data_row + 1)
        for worksheet_row in range(first_data_row, final_data_row + 1)
    ]
    total_row = row_record(max_row) if total_rows else None
    return header_row, data_rows, total_row


def _table_cell(
    source: dict[str, Any] | None, column: int, worksheet_row: int
) -> dict[str, Any]:
    """Return one explicit table cell, including blanks within declared bounds."""

    coordinate = _coordinate(column, worksheet_row)
    record = {
        "column": _coordinate(column, 1).rstrip("1"),
        "coordinate": coordinate,
        "value": None,
        "data_type": "blank",
        "style_id": "",
    }
    if source is not None:
        record.update(source)
    return record


def _range_shape(cell_range: str) -> tuple[int, int]:
    start_text, _, end_text = cell_range.replace("$", "").partition(":")
    start = _coordinate_parts(start_text)
    end = _coordinate_parts(end_text or start_text)
    if start is None or end is None:
        return 0, 0
    return abs(end[1] - start[1]) + 1, abs(end[0] - start[0]) + 1


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _table_summary(
    name: str, headers: list[str], row_count: int, formula_cells: list[str]
) -> str:
    label = f"'{name}'" if name else "This table"
    fields = ", ".join(headers) if headers else "unnamed columns"
    summary = f"{label} contains {row_count} rows with columns: {fields}."
    if formula_cells:
        summary += (
            f" It includes formula-driven values in {', '.join(formula_cells[:8])}"
            + ("." if len(formula_cells) <= 8 else ", plus additional cells.")
        )
    return summary


def _table_sheet_summary(
    table_range: str, sheet_name: str, sheet_dimensions: str, populated_cells: int
) -> str:
    return (
        f"The table occupies {table_range or 'an unresolved range'} on sheet "
        f"'{sheet_name}' (sheet extent {sheet_dimensions or 'unknown'}), which has "
        f"{populated_cells} populated cells represented in this analysis."
    )


def _is_context_label(value: Any) -> bool:
    """Return whether a native cell value is useful as a human-facing label."""

    if not isinstance(value, str) or not value.strip():
        return False
    return re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?", value.strip()) is None
