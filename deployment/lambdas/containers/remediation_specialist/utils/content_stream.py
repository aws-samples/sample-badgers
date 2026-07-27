"""Wrap PDF content stream operators in BDC/EMC marked content sequences.

Uses spatial matching (cm matrix → fitz line lookup) for text operators
and cm-based bbox matching for Do (figure/image) operators.
Closes BDC on ET as required by Acrobat.
"""

import logging
from typing import Dict, List, Optional, Tuple

import pikepdf
from pikepdf import Array, Dictionary, Name, Operator

from utils.pdf_accessibility_models import TagRegion

logger = logging.getLogger(__name__)


def _find_line_for_position(fitz_lines, page_height, cm_y, tolerance=5.0):
    """Convert cm y-position to fitz-space and find the matching fitz line.

    cm coordinates: origin at bottom-left, y increases upward.
    fitz coordinates: origin at top-left, y increases downward.
    """
    fitz_y = page_height - cm_y
    best_line = None
    best_dist = float("inf")

    for i, line in enumerate(fitz_lines):
        y0, y1 = line["bbox"][1], line["bbox"][3]
        if y0 - tolerance <= fitz_y <= y1 + tolerance:
            dist = 0 if y0 <= fitz_y <= y1 else min(abs(fitz_y - y0), abs(fitz_y - y1))
            if dist < best_dist:
                best_dist = dist
                best_line = i

    return best_line


def _get_region_for_bbox(regions_on_page: List[TagRegion], bbox):
    """Find the best-matching region for a given bbox.

    Containment-aware: if a smaller region's bbox is fully contained within
    a larger region's bbox, prefer the smaller (more specific) one.
    Otherwise, fall back to largest overlap area (original behavior).
    """
    bx0, by0, bx1, by1 = bbox
    candidates = []

    for region in regions_on_page:
        rx0, ry0, rx1, ry1 = region.bbox
        ix0, iy0 = max(bx0, rx0), max(by0, ry0)
        ix1, iy1 = min(bx1, rx1), min(by1, ry1)
        if ix0 < ix1 and iy0 < iy1:
            overlap = (ix1 - ix0) * (iy1 - iy0)
            region_area = (rx1 - rx0) * (ry1 - ry0)
            candidates.append((region, overlap, region_area))

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0][0]

    # Check for containment: is any candidate's bbox fully inside another's?
    for i, (r_inner, _, area_inner) in enumerate(candidates):
        ix0, iy0, ix1, iy1 = r_inner.bbox
        for j, (r_outer, _, area_outer) in enumerate(candidates):
            if i == j:
                continue
            ox0, oy0, ox1, oy1 = r_outer.bbox
            if ox0 <= ix0 and oy0 <= iy0 and ix1 <= ox1 and iy1 <= oy1:
                # r_inner is fully contained within r_outer — prefer r_inner
                return r_inner

    # No containment — fall back to largest overlap (original behavior)
    best = None
    best_overlap = 0.0
    for region, overlap, _ in candidates:
        if overlap > best_overlap:
            best_overlap = overlap
            best = region
    return best


def wrap_content_stream_fixed(
    pdf,
    fitz_doc,
    regions: Dict[int, List[TagRegion]],
    page_num: int,
    mcid_start: int = 0,
) -> Tuple[bytes, Dict[int, TagRegion], int]:
    """Wrap content stream operators in BDC/EMC using spatial matching.

    Returns:
        (new_content_bytes, mcid_map, next_mcid)
    """
    page = pdf.pages[page_num]
    fitz_page = fitz_doc[page_num]
    page_height = fitz_page.rect.height

    # Build fitz line index
    text_dict = fitz_page.get_text("dict")
    fitz_lines = []
    for block in text_dict.get("blocks", []):
        if block["type"] == 0:
            for line in block.get("lines", []):
                fitz_lines.append({"bbox": line["bbox"]})

    regions_on_page = regions.get(page_num, [])

    try:
        instructions = list(pikepdf.parse_content_stream(page))
    except Exception as e:
        logger.warning("Failed to parse content stream page %d: %s", page_num, e)
        return b"", {}, mcid_start

    new_ops: List[Tuple[list, Operator]] = []
    mcid_map: Dict[int, TagRegion] = {}
    mcid = mcid_start
    in_marked = False
    current_region: Optional[TagRegion] = None
    current_cm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    for instruction in instructions:
        if not hasattr(instruction, "operator"):
            new_ops.append(([Name("/Artifact")], Operator("BMC")))
            if hasattr(instruction, "operands"):
                new_ops.append((list(instruction.operands), instruction.operator))
            new_ops.append(([], Operator("EMC")))
            continue

        operands = (
            list(instruction.operands) if hasattr(instruction, "operands") else []
        )
        op = instruction.operator
        op_name = str(op)

        # Strip existing marked content
        if op_name in ("BMC", "BDC", "EMC"):
            continue

        # Track cm matrix
        if op_name == "cm" and len(operands) >= 6:
            current_cm = [float(str(o)) for o in operands[:6]]
            new_ops.append((operands, op))
            continue

        if op_name == "BT":
            new_ops.append((operands, op))
            continue

        # Close BDC on ET — Acrobat requires this
        if op_name == "ET":
            if in_marked:
                new_ops.append(([], Operator("EMC")))
                in_marked = False
                current_region = None
            new_ops.append((operands, op))
            continue

        # Text operators — spatial match via cm position
        if op_name in ("Tj", "TJ", "'", '"'):
            target_region = None
            line_idx = _find_line_for_position(fitz_lines, page_height, current_cm[5])
            if line_idx is not None:
                bbox = fitz_lines[line_idx]["bbox"]
                target_region = _get_region_for_bbox(regions_on_page, bbox)

            if target_region and target_region != current_region:
                if in_marked:
                    new_ops.append(([], Operator("EMC")))

                if target_region.tag == "Artifact":
                    new_ops.append(([Name("/Artifact")], Operator("BMC")))
                else:
                    mcid_map[mcid] = target_region
                    bdc_dict = Dictionary({"/MCID": mcid})
                    new_ops.append(
                        ([Name(f"/{target_region.tag}"), bdc_dict], Operator("BDC"))
                    )
                    mcid += 1
                in_marked = True
                current_region = target_region

            new_ops.append((operands, op))
            continue

        # Do (image/figure) — spatial cm bbox matching
        if op_name == "Do":
            if in_marked:
                new_ops.append(([], Operator("EMC")))
                in_marked = False
                current_region = None

            img_x0 = current_cm[4]
            img_y_bottom = current_cm[5]
            img_w = current_cm[0]
            img_h = current_cm[3]
            fitz_x0 = img_x0
            fitz_y0 = page_height - img_y_bottom - img_h
            fitz_x1 = img_x0 + img_w
            fitz_y1 = page_height - img_y_bottom
            do_bbox = (fitz_x0, fitz_y0, fitz_x1, fitz_y1)

            fig_region = _get_region_for_bbox(regions_on_page, do_bbox)
            if fig_region and fig_region.tag == "Figure":
                mcid_map[mcid] = fig_region
                bdc_dict = Dictionary({"/MCID": mcid})
                new_ops.append(([Name("/Figure"), bdc_dict], Operator("BDC")))
                new_ops.append((operands, op))
                new_ops.append(([], Operator("EMC")))
                mcid += 1
            else:
                new_ops.append((operands, op))
            continue

        new_ops.append((operands, op))

    if in_marked:
        new_ops.append(([], Operator("EMC")))

    return pikepdf.unparse_content_stream(new_ops), mcid_map, mcid
