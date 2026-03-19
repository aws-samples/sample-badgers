"""
Cell Grid Resolver v3 — Vision-model bbox detection using labeled cell grids.

Instead of asking a vision model for raw normalized coordinates (which are
imprecise), this module:
  1. Overlays a labeled cell grid (A1, B2, ...) on the page image
  2. Sends the gridded image + element descriptions to the vision model
  3. Asks which cells each element occupies (with corner-point sub-positions)
  4. Converts cell references back to normalized bboxes

v3 Enhancements (corner-point + hierarchical refinement):
  - Corner-point system: model returns TL/BR cell + sub-position (top/middle/bottom
    × left/center/right) for ~9× effective resolution over cell-union approach.
  - Resolved anchors: already-located elements are included in the prompt as
    spatial landmarks to help the model orient.
  - Hierarchical refinement: low-confidence or oversized results trigger a second
    pass on a cropped region with a finer grid (~11× vertical resolution gain).
  - Font legibility: uses TrueType font for grid labels when available.
  - Backward compatible: still parses legacy "cells" array responses.

Adapted from the cell-grid detection pattern in unclear_region_detector.py.
"""

import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Font loading (legibility improvement)
# ---------------------------------------------------------------------------

_font_cache: Optional[ImageFont.FreeTypeFont] = None
_font_loaded: bool = False

# Search paths for TrueType fonts — covers Lambda (AL2023), Ubuntu, macOS
_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/google-noto/NotoSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _get_label_font(size: int = 13) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable TrueType font for grid labels, with fallback."""
    global _font_cache, _font_loaded
    if _font_loaded:
        return _font_cache if _font_cache else ImageFont.load_default()

    _font_loaded = True
    for path in _FONT_SEARCH_PATHS:
        if os.path.exists(path):
            try:
                _font_cache = ImageFont.truetype(path, size)
                logger.info("Grid label font: %s @ %dpt", path, size)
                return _font_cache
            except Exception:
                continue

    logger.info("No TrueType font found, using PIL default bitmap font")
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Grid overlay
# ---------------------------------------------------------------------------


def add_cell_grid_overlay(
    img: Image.Image,
    cols: int = 12,
    rows: int = 16,
    color_scheme: str = "cyan",
) -> tuple[Image.Image, dict[str, dict]]:
    """
    Overlay a labeled cell grid on an image.

    Args:
        img: Source image (any mode — will be converted to RGBA internally).
        cols: Number of grid columns (A, B, C, …).
        rows: Number of grid rows (1, 2, 3, …).
        color_scheme: "cyan", "magenta", "green", or "blue".

    Returns:
        (gridded_image, cell_map) where cell_map maps cell names like "A1"
        to {"pixels": (x0,y0,x1,y1), "normalized": (nx0,ny0,nx1,ny1)}.
    """
    img = img.convert("RGBA")
    width, height = img.size

    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    cell_width = width / cols
    cell_height = height / rows

    schemes = {
        "cyan": {
            "line": (0, 255, 255, 100),
            "label_bg": (0, 0, 0, 180),
            "label_text": (0, 255, 255, 255),
        },
        "magenta": {
            "line": (255, 0, 255, 100),
            "label_bg": (0, 0, 0, 180),
            "label_text": (255, 0, 255, 255),
        },
        "green": {
            "line": (0, 255, 0, 100),
            "label_bg": (0, 0, 0, 180),
            "label_text": (0, 255, 0, 255),
        },
        "blue": {
            "line": (0, 102, 255, 100),
            "label_bg": (255, 255, 255, 200),
            "label_text": (0, 102, 255, 255),
        },
    }

    colors = schemes.get(color_scheme, schemes["cyan"])
    line_color = colors["line"]
    label_bg = colors["label_bg"]
    label_color = colors["label_text"]

    col_letters = [chr(ord("A") + i) for i in range(cols)]
    cell_map: dict[str, dict] = {}

    font = _get_label_font(size=13)

    for row in range(rows):
        for col in range(cols):
            cell_name = f"{col_letters[col]}{row + 1}"

            x0 = int(col * cell_width)
            y0 = int(row * cell_height)
            x1 = int((col + 1) * cell_width)
            y1 = int((row + 1) * cell_height)

            cell_map[cell_name] = {
                "pixels": (x0, y0, x1, y1),
                "normalized": (x0 / width, y0 / height, x1 / width, y1 / height),
            }

            # Cell border
            draw.rectangle([x0, y0, x1, y1], outline=line_color, width=1)

            # Cell label (readable font, top-left corner of cell)
            label_x = x0 + 2
            label_y = y0 + 1
            text_bbox = draw.textbbox((label_x, label_y), cell_name, font=font)
            draw.rectangle(
                [
                    text_bbox[0] - 1,
                    text_bbox[1] - 1,
                    text_bbox[2] + 1,
                    text_bbox[3] + 1,
                ],
                fill=label_bg,
            )
            draw.text((label_x, label_y), cell_name, fill=label_color, font=font)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    return result, cell_map


# ---------------------------------------------------------------------------
# Bbox conversion: corner-point + legacy cell-union
# ---------------------------------------------------------------------------

# Sub-position offsets within a cell (0.0 = start edge, 1.0 = end edge)
_SUB_POS = {
    "left": 0.0,
    "center": 0.5,
    "right": 1.0,
    "top": 0.0,
    "middle": 0.5,
    "bottom": 1.0,
}


def cells_to_bbox(
    cells: list[str],
    cell_map: dict[str, dict],
) -> Optional[dict[str, float]]:
    """
    Convert a list of cell names to a normalized bounding box.
    Legacy path — used when model returns flat cell lists.

    Returns:
        {"x0": float, "y0": float, "x1": float, "y1": float} or None.
    """
    min_x0 = float("inf")
    min_y0 = float("inf")
    max_x1 = float("-inf")
    max_y1 = float("-inf")

    for cell in cells:
        cell = cell.upper().strip()
        if cell in cell_map:
            norm = cell_map[cell]["normalized"]
            min_x0 = min(min_x0, norm[0])
            min_y0 = min(min_y0, norm[1])
            max_x1 = max(max_x1, norm[2])
            max_y1 = max(max_y1, norm[3])

    if min_x0 == float("inf"):
        return None

    return {"x0": min_x0, "y0": min_y0, "x1": max_x1, "y1": max_y1}


def corners_to_bbox(
    top_left: dict,
    bottom_right: dict,
    cell_map: dict[str, dict],
) -> Optional[dict[str, float]]:
    """
    Convert corner-point references to a tight normalized bounding box.

    Each corner dict has:
        {"cell": "B3", "v": "top|middle|bottom", "h": "left|center|right"}

    Sub-positions interpolate within the cell:
        "left"/"top" = cell start edge
        "center"/"middle" = cell midpoint
        "right"/"bottom" = cell end edge
    """
    tl_cell = top_left.get("cell", "").upper().strip()
    br_cell = bottom_right.get("cell", "").upper().strip()

    if tl_cell not in cell_map or br_cell not in cell_map:
        # Fall back to legacy cell-union if corner cells are invalid
        fallback_cells = [c for c in [tl_cell, br_cell] if c in cell_map]
        return cells_to_bbox(fallback_cells, cell_map) if fallback_cells else None

    tl_norm = cell_map[tl_cell]["normalized"]  # (x0, y0, x1, y1)
    br_norm = cell_map[br_cell]["normalized"]

    # Interpolate within each cell using sub-position
    h_frac_tl = _SUB_POS.get(top_left.get("h", "left"), 0.0)
    v_frac_tl = _SUB_POS.get(top_left.get("v", "top"), 0.0)
    h_frac_br = _SUB_POS.get(bottom_right.get("h", "right"), 1.0)
    v_frac_br = _SUB_POS.get(bottom_right.get("v", "bottom"), 1.0)

    tl_w = tl_norm[2] - tl_norm[0]
    tl_h = tl_norm[3] - tl_norm[1]
    br_w = br_norm[2] - br_norm[0]
    br_h = br_norm[3] - br_norm[1]

    x0 = tl_norm[0] + h_frac_tl * tl_w
    y0 = tl_norm[1] + v_frac_tl * tl_h
    x1 = br_norm[0] + h_frac_br * br_w
    y1 = br_norm[1] + v_frac_br * br_h

    # Sanity: ensure x0<x1 and y0<y1
    if x0 >= x1 or y0 >= y1:
        return cells_to_bbox([tl_cell, br_cell], cell_map)

    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _image_to_base64(img: Image.Image, max_b64_bytes: int = 4_500_000) -> str:
    """Convert PIL Image to base64 string, ensuring it fits under Bedrock's 5MB limit.

    Resizes and compresses as JPEG if needed. The max_b64_bytes default
    leaves headroom below the 5,242,880 byte API limit.
    """
    MAX_DIM = 2048

    # Convert RGBA to RGB for JPEG compatibility
    if img.mode in ("RGBA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Resize if dimensions exceed max
    if max(img.size) > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)
        logger.info("Resized gridded image to %s", img.size)

    # Encode as JPEG, reduce quality if still too large
    for quality in (85, 70, 55, 40):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        b64_str = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        if len(b64_str) <= max_b64_bytes:
            logger.info(
                "Gridded image encoded: %d bytes b64 at quality=%d",
                len(b64_str),
                quality,
            )
            return b64_str

    # Last resort — already at lowest quality
    logger.warning("Gridded image still %d bytes after max compression", len(b64_str))
    return b64_str


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

# Module-level cache so we don't re-download on every page
_prompt_cache: Optional[str] = None


def _load_prompt_template(
    config_bucket: Optional[str] = None,
    analyzer_name: Optional[str] = None,
) -> str:
    """Load the element-location prompt template.

    Resolution order:
        1. Module cache (already loaded this invocation)
        2. S3: s3://{config_bucket}/{analyzer_name}/prompts/prompt_locate_elements.xml
        3. Local filesystem alongside this .py file
        4. Inline fallback hardcoded in source

    Args:
        config_bucket: S3 bucket for config (from CONFIG_BUCKET env var).
        analyzer_name: Analyzer directory in the bucket (from ANALYZER_NAME env var).
    """
    global _prompt_cache
    if _prompt_cache is not None:
        return _prompt_cache

    # Resolve from env if not passed explicitly
    if config_bucket is None:
        config_bucket = os.environ.get("CONFIG_BUCKET")
    if analyzer_name is None:
        analyzer_name = os.environ.get("ANALYZER_NAME", "remediation_analyzer")

    prompt_content = None

    # Try S3 first when running in Lambda
    if config_bucket and os.environ.get("AWS_EXECUTION_ENV"):
        try:
            import boto3

            s3 = boto3.client("s3")
            s3_key = f"{analyzer_name}/prompts/prompt_locate_elements.xml"
            logger.info("Loading grid prompt from s3://%s/%s", config_bucket, s3_key)
            response = s3.get_object(Bucket=config_bucket, Key=s3_key)
            prompt_content = response["Body"].read().decode("utf-8")
        except Exception as e:
            logger.warning("Failed to load grid prompt from S3: %s", e)

    # Try local filesystem
    if prompt_content is None:
        local_path = Path(__file__).parent / "prompt_locate_elements.xml"
        if local_path.exists():
            logger.info("Loading grid prompt from local: %s", local_path)
            with open(local_path, "r", encoding="utf-8") as f:
                prompt_content = f.read()

    # Extract content between <prompt> tags
    if prompt_content:
        start = prompt_content.find("<prompt>")
        end = prompt_content.find("</prompt>")
        if start >= 0 and end >= 0:
            _prompt_cache = prompt_content[start + len("<prompt>") : end].strip()
            return _prompt_cache

    # Inline fallback
    logger.warning("Using inline fallback prompt template")
    _prompt_cache = _inline_prompt_template()
    return _prompt_cache


# ---------------------------------------------------------------------------
# Grid sizing helpers
# ---------------------------------------------------------------------------


def _auto_grid_size(width: int, height: int) -> tuple[int, int]:
    """
    Choose grid dimensions based on page aspect ratio.

    Targets roughly square cells so that labels are readable and
    cell references map to similarly-sized regions.

    Returns:
        (cols, rows)
    """
    aspect = width / height

    if aspect > 1.3:
        # Landscape
        return 16, 10
    elif aspect < 0.77:
        # Portrait
        return 10, 14
    else:
        # Roughly square
        return 12, 12


# ---------------------------------------------------------------------------
# Anchor context builder
# ---------------------------------------------------------------------------


def _build_anchor_context(
    resolved_anchors: list[dict[str, Any]],
    cols: int,
    rows: int,
) -> str:
    """Build spatial anchor descriptions from already-resolved elements.

    Converts resolved element bboxes to approximate grid cell references
    so the model can use them as spatial landmarks.

    Args:
        resolved_anchors: Elements with known bboxes (from PyMuPDF text search).
        cols: Grid column count.
        rows: Grid row count.

    Returns:
        XML-formatted anchor descriptions, or empty string if no usable anchors.
    """
    if not resolved_anchors:
        return ""

    anchor_lines = []
    for anchor in resolved_anchors:
        if anchor.get("source") == "fallback_stacked":
            continue  # Don't use low-confidence anchors
        bbox = anchor.get("bbox", {})
        if not bbox:
            continue
        text = anchor.get("content", "") or anchor.get("text", "")
        if not text:
            continue

        display = text[:80] + "..." if len(text) > 80 else text

        # Convert normalized PDF coords (bottom-left origin) to image coords (top-left)
        img_y0 = 1.0 - bbox.get("y1", 0)
        img_y1 = 1.0 - bbox.get("y0", 0)

        # Map to approximate grid cells for reference
        anchor_col = chr(ord("A") + min(cols - 1, int(bbox.get("x0", 0) * cols)))
        anchor_row_start = max(1, int(img_y0 * rows) + 1)
        anchor_row_end = max(1, int(img_y1 * rows) + 1)

        anchor_lines.append(
            f'        <anchor type="{anchor.get("type", "P")}" '
            f'region="{anchor_col}{anchor_row_start}-{anchor_col}{anchor_row_end}">'
            f'"{display}"</anchor>'
        )

    if not anchor_lines:
        return ""

    # Cap at 15 anchors to avoid prompt bloat
    anchor_lines = anchor_lines[:15]
    logger.info("Including %d spatial anchors in grid prompt", len(anchor_lines))
    return "\n".join(anchor_lines)


# ---------------------------------------------------------------------------
# Core resolver
# ---------------------------------------------------------------------------


def resolve_elements_via_grid(
    image_data: bytes,
    unresolved_elements: list[dict[str, Any]],
    analyzer: Any,
    _aws_profile: Optional[str] = None,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
    resolved_anchors: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """
    Resolve bounding boxes for elements using a cell-grid vision approach.

    This replaces raw-coordinate vision model calls with a grid-reference
    approach that is significantly more accurate.

    Args:
        image_data: PNG/JPEG bytes of the rendered page.
        unresolved_elements: List of element dicts, each with at least:
            - "id": element identifier
            - "type": element type (H1, P, figure, etc.)
            - "text": text content to locate (for text elements)
            - "alt_text": description (for figure elements)
            - "order": reading order index
        analyzer: Initialized analyzer with .bedrock_client and .config.
        _aws_profile: AWS profile (reserved).
        cols: Grid columns override (auto-sized if None).
        rows: Grid rows override (auto-sized if None).
        resolved_anchors: Already-resolved elements with known bboxes.
            Used as spatial reference points in the prompt to help the
            model locate unresolved elements relative to known positions.

    Returns:
        List of resolved element dicts in the same format as
        _resolve_text_bboxes / _resolve_figure_bboxes output, with
        source="cell_grid_*" or "cell_grid_refined_*".
    """
    if not unresolved_elements:
        return []

    # Load the page image to get dimensions and create grid
    img = Image.open(BytesIO(image_data)).convert("RGB")
    width, height = img.size

    # Auto-size grid if not specified
    if cols is None or rows is None:
        cols, rows = _auto_grid_size(width, height)

    logger.info(
        "Cell grid resolver: %d elements to locate, grid=%dx%d on %dx%d image",
        len(unresolved_elements),
        cols,
        rows,
        width,
        height,
    )

    # ── Pass 1: Coarse grid on full page ──
    resolved = _grid_resolve_pass(
        img,
        unresolved_elements,
        analyzer,
        _aws_profile,
        cols,
        rows,
        resolved_anchors,
    )

    # ── Pass 2: Hierarchical refinement for low-confidence results ──
    refine_candidates = []
    keep = []

    for elem in resolved:
        source = elem.get("source", "")
        if source == "fallback_stacked":
            refine_candidates.append(elem)
        elif _should_refine(elem, cols, rows):
            refine_candidates.append(elem)
        else:
            keep.append(elem)

    if refine_candidates and len(refine_candidates) <= 10:
        logger.info(
            "Hierarchical refinement: %d elements eligible for pass 2",
            len(refine_candidates),
        )
        refined = _hierarchical_refine(
            img,
            image_data,
            refine_candidates,
            analyzer,
            _aws_profile,
            cols,
            rows,
        )
        keep.extend(refined)
    elif refine_candidates:
        # Too many to refine individually — keep pass 1 results
        logger.info(
            "Skipping hierarchical refinement: %d candidates exceeds limit",
            len(refine_candidates),
        )
        keep.extend(refine_candidates)

    return keep


def _should_refine(elem: dict, cols: int, rows: int) -> bool:
    """Determine if an element should go through hierarchical refinement.

    Criteria:
      - Confidence is not "high"
      - Bbox is suspiciously tall (> 1.5× a single cell height)
    """
    source = elem.get("source", "")
    if "high" in source:
        return False

    bbox = elem.get("bbox", {})
    if not bbox:
        return True

    # Check if bbox height exceeds 1.5× cell height in normalized coords
    cell_h_norm = 1.0 / rows
    bbox_h = abs(bbox.get("y1", 0) - bbox.get("y0", 0))
    if bbox_h > cell_h_norm * 1.5:
        return True

    return False


def _grid_resolve_pass(
    img: Image.Image,
    elements: list[dict[str, Any]],
    analyzer: Any,
    aws_profile: Optional[str],
    cols: int,
    rows: int,
    resolved_anchors: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Execute a single grid resolution pass.

    This is the core grid → vision model → bbox pipeline, extracted
    so it can be called for both pass 1 (full page) and pass 2 (crop).
    """
    width, height = img.size

    # Create gridded image
    gridded, cell_map = add_cell_grid_overlay(img, cols=cols, rows=rows)
    gridded_b64 = _image_to_base64(gridded)

    # Build element descriptions for the prompt
    col_letters = [chr(ord("A") + i) for i in range(cols)]
    element_lines = []
    for elem in elements:
        elem_id = elem.get("id", f"order_{elem.get('order', '?')}")
        elem_type = elem.get("type", "unknown")

        # Use text content for text elements, alt_text/caption for figures
        if elem_type in ("figure", "Figure", "image"):
            desc = elem.get("alt_text") or elem.get("caption") or "visual element"
            element_lines.append(
                f'        <element id="{elem_id}" type="{elem_type}">'
                f"Locate the figure/image: {desc}</element>"
            )
        else:
            text = elem.get("text", "") or elem.get("content", "")
            # Truncate long text but keep enough for identification
            display_text = text[:120] + "..." if len(text) > 120 else text
            element_lines.append(
                f'        <element id="{elem_id}" type="{elem_type}">'
                f'Find this text: "{display_text}"</element>'
            )

    element_descriptions = "\n".join(element_lines)

    # Build spatial anchor context from already-resolved elements
    anchor_context = _build_anchor_context(resolved_anchors or [], cols, rows)

    # Build prompt
    try:
        prompt_template = _load_prompt_template()
    except FileNotFoundError:
        prompt_template = _inline_prompt_template()

    prompt = prompt_template.format(
        col_letters=", ".join(col_letters),
        rows=rows,
        element_descriptions=element_descriptions,
        anchor_context=anchor_context,
    )

    # Call vision model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": gridded_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        bedrock_client = analyzer.bedrock_client
        model_id = analyzer.config.get("model_id", "us.anthropic.claude-sonnet-4-6")

        system_prompt = (
            "You are a precise document element locator. Given a page image with a "
            "labeled cell grid overlay and a list of content elements, report the "
            "grid cell and sub-position of each element's top-left and bottom-right "
            "corners. Return ONLY a JSON array."
        )

        resolver_max_tokens = analyzer.global_settings.get("resolver_max_tokens", 4000)

        payload = bedrock_client.create_anthropic_payload(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=resolver_max_tokens,
            temperature=0.1,
        )

        response = bedrock_client.invoke_model(
            model_id=model_id,
            payload=payload,
            profile_name=aws_profile,
        )

        # Extract text
        result_text = ""
        for block in response.get("content", []):
            if block.get("type") == "text":
                result_text = block.get("text", "")
                break

        if not result_text:
            logger.warning("Empty response from cell grid resolver")
            return _fallback_stacked(elements)

        # Parse response
        grid_results = _parse_grid_response(result_text)

    except Exception as e:
        logger.error("Cell grid resolver vision call failed: %s", e)
        return _fallback_stacked(elements)

    # Build lookup: element_id → corner pair or cell list
    cell_lookup: dict[str, tuple[Any, str]] = {}
    for item in grid_results:
        item_id = item.get("id", "")
        confidence = item.get("confidence", "medium")

        # Prefer corner-point format; fall back to legacy cells array
        if item.get("top_left") and item.get("bottom_right"):
            cell_lookup[item_id] = (
                {"top_left": item["top_left"], "bottom_right": item["bottom_right"]},
                confidence,
            )
        elif item.get("cells"):
            cell_lookup[item_id] = ({"cells": item["cells"]}, confidence)

    # Convert cell/corner references to bboxes and build resolved elements
    resolved = []
    resolved_count = 0
    fallback_count = 0

    for elem in elements:
        elem_id = elem.get("id", f"order_{elem.get('order', '?')}")
        elem_type = elem.get("type", "P")
        text = elem.get("text", "") or elem.get("content", "")

        cell_info = cell_lookup.get(elem_id)
        if cell_info:
            geo, confidence = cell_info

            # Corner-point path (preferred)
            if "top_left" in geo:
                bbox = corners_to_bbox(geo["top_left"], geo["bottom_right"], cell_map)
            else:
                # Legacy cell-list path
                bbox = cells_to_bbox(geo["cells"], cell_map)

            if bbox:
                # Convert from top-left origin (image coords) to
                # bottom-left origin (PDF coords) for the tagger
                pdf_bbox = {
                    "x0": bbox["x0"],
                    "y0": 1.0 - bbox["y1"],  # flip Y
                    "x1": bbox["x1"],
                    "y1": 1.0 - bbox["y0"],  # flip Y
                }

                resolved.append(
                    {
                        "type": elem_type,
                        "order": elem.get("order", 0),
                        "alt_text": elem.get("alt_text", ""),
                        "content": text,
                        "id": elem_id,
                        "bbox": pdf_bbox,
                        "source": f"cell_grid_{confidence}",
                    }
                )
                resolved_count += 1
                logger.debug(
                    "Grid-resolved %s → bbox=(%.3f,%.3f)-(%.3f,%.3f) [%s]",
                    elem_id,
                    pdf_bbox["x0"],
                    pdf_bbox["y0"],
                    pdf_bbox["x1"],
                    pdf_bbox["y1"],
                    confidence,
                )
                continue

        # Fallback: model didn't return this element or cells were invalid
        logger.warning("Cell grid resolver: no result for %s, using fallback", elem_id)
        fallback_count += 1
        resolved.append(
            {
                "type": elem_type,
                "order": elem.get("order", 0),
                "alt_text": elem.get("alt_text", ""),
                "content": text,
                "id": elem_id,
                "bbox": _stacked_fallback_bbox(elem.get("order", 0)),
                "source": "fallback_stacked",
            }
        )

    logger.info(
        "Cell grid resolver: %d/%d resolved via grid, %d fell back",
        resolved_count,
        len(elements),
        fallback_count,
    )
    return resolved


# ---------------------------------------------------------------------------
# Hierarchical refinement (pass 2)
# ---------------------------------------------------------------------------

# Fine grid size for refinement pass
_REFINE_COLS = 10
_REFINE_ROWS = 10
_REFINE_PAD_FACTOR = 0.4  # 40% padding around the coarse bbox


def _hierarchical_refine(
    full_img: Image.Image,
    full_image_data: bytes,
    candidates: list[dict[str, Any]],
    analyzer: Any,
    aws_profile: Optional[str],
    coarse_cols: int,
    coarse_rows: int,
) -> list[dict[str, Any]]:
    """Refine element bboxes by re-running grid resolution on cropped regions.

    Groups nearby candidates, crops the full image to their combined bounding
    region (with padding for context), then runs a fine grid pass on the crop.
    Converts results back to full-page coordinates.

    Args:
        full_img: Full page PIL Image.
        full_image_data: Full page image bytes (for fallback).
        candidates: Elements to refine (from pass 1).
        analyzer: Initialized analyzer.
        aws_profile: AWS profile.
        coarse_cols: Pass 1 grid columns (for logging).
        coarse_rows: Pass 1 grid rows (for logging).

    Returns:
        Refined element dicts with updated bboxes.
    """
    width, height = full_img.size

    # Compute the combined bounding region of all candidates (in image coords)
    # Bboxes are in PDF coords (bottom-left origin) — convert to image (top-left)
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for elem in candidates:
        bbox = elem.get("bbox", {})
        if not bbox:
            continue
        # PDF → image coordinate conversion
        img_x0 = bbox.get("x0", 0) * width
        img_y0 = (1.0 - bbox.get("y1", 0)) * height
        img_x1 = bbox.get("x1", 1) * width
        img_y1 = (1.0 - bbox.get("y0", 1)) * height
        min_x = min(min_x, img_x0)
        min_y = min(min_y, img_y0)
        max_x = max(max_x, img_x1)
        max_y = max(max_y, img_y1)

    if min_x >= max_x or min_y >= max_y:
        logger.warning("Could not compute refinement region, returning pass 1 results")
        return candidates

    # Add padding for context
    region_w = max_x - min_x
    region_h = max_y - min_y
    pad_x = int(region_w * _REFINE_PAD_FACTOR)
    pad_y = int(region_h * _REFINE_PAD_FACTOR)

    crop_x0 = max(0, int(min_x) - pad_x)
    crop_y0 = max(0, int(min_y) - pad_y)
    crop_x1 = min(width, int(max_x) + pad_x)
    crop_y1 = min(height, int(max_y) + pad_y)

    crop_w = crop_x1 - crop_x0
    crop_h = crop_y1 - crop_y0

    if crop_w < 50 or crop_h < 50:
        logger.warning("Refinement crop too small (%dx%d), skipping", crop_w, crop_h)
        return candidates

    crop_img = full_img.crop((crop_x0, crop_y0, crop_x1, crop_y1))

    fine_cw = crop_w / _REFINE_COLS
    fine_ch = crop_h / _REFINE_ROWS
    coarse_ch = height / coarse_rows

    logger.info(
        "Hierarchical refinement: crop=(%d,%d)-(%d,%d) %dx%d, "
        "fine grid %dx%d, cell=%dx%dpx (%.0f× finer vertical)",
        crop_x0,
        crop_y0,
        crop_x1,
        crop_y1,
        crop_w,
        crop_h,
        _REFINE_COLS,
        _REFINE_ROWS,
        int(fine_cw),
        int(fine_ch),
        coarse_ch / fine_ch if fine_ch > 0 else 0,
    )

    # Run pass 2 on the crop
    refined_local = _grid_resolve_pass(
        crop_img,
        candidates,
        analyzer,
        aws_profile,
        _REFINE_COLS,
        _REFINE_ROWS,
        resolved_anchors=None,  # No anchors in crop context
    )

    # Convert crop-local bboxes back to full-page coordinates
    refined = []
    for elem in refined_local:
        bbox = elem.get("bbox", {})
        if not bbox or elem.get("source") == "fallback_stacked":
            refined.append(elem)
            continue

        # The bbox is in PDF coords relative to the CROP.
        # Convert: crop-PDF → crop-image → full-image → full-PDF

        # Step 1: crop-PDF → crop-image (normalized, top-left origin)
        crop_img_x0 = bbox["x0"]
        crop_img_y0 = 1.0 - bbox["y1"]
        crop_img_x1 = bbox["x1"]
        crop_img_y1 = 1.0 - bbox["y0"]

        # Step 2: crop-image (normalized) → full-image (pixels)
        full_px_x0 = crop_x0 + crop_img_x0 * crop_w
        full_px_y0 = crop_y0 + crop_img_y0 * crop_h
        full_px_x1 = crop_x0 + crop_img_x1 * crop_w
        full_px_y1 = crop_y0 + crop_img_y1 * crop_h

        # Step 3: full-image (pixels) → full-PDF (normalized, bottom-left origin)
        full_pdf_bbox = {
            "x0": full_px_x0 / width,
            "y0": 1.0 - (full_px_y1 / height),
            "x1": full_px_x1 / width,
            "y1": 1.0 - (full_px_y0 / height),
        }

        elem["bbox"] = full_pdf_bbox
        elem["source"] = elem.get("source", "").replace(
            "cell_grid_", "cell_grid_refined_"
        )
        refined.append(elem)

    return refined


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_grid_response(response_text: str) -> list[dict]:
    """Parse JSON array from model response, handling markdown wrappers.

    Supports both corner-point format and legacy cells format:
        Corner-point: {"id": "...", "top_left": {...}, "bottom_right": {...}, "confidence": "high"}
        Legacy:       {"id": "...", "cells": ["B3", "C3"], "confidence": "high"}
    """
    import re

    # Strip analysis/thinking wrappers
    if "</analysis>" in response_text:
        response_text = response_text.split("</analysis>", 1)[1]

    # Strip markdown code blocks
    response_text = re.sub(r"```json\s*", "", response_text)
    response_text = re.sub(r"```\s*$", "", response_text)
    response_text = response_text.strip()

    try:
        parsed: Any = json.loads(response_text)
        if isinstance(parsed, list):
            result: list[dict] = parsed
            return result
        elif isinstance(parsed, dict) and "elements" in parsed:
            result_: list[dict] = parsed["elements"]
            return result_
        else:
            logger.warning("Unexpected grid response format: %s", type(parsed))
            return []
    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse grid response: %s\nResponse: %s", e, response_text[:500]
        )
        return []


# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------


def _stacked_fallback_bbox(order: int) -> dict[str, float]:
    """Generate a stacked fallback bbox for an unresolved element."""
    return {
        "x0": 0.02,
        "y0": max(0.02, 1.0 - (order * 0.05)),
        "x1": 0.98,
        "y1": min(0.98, 1.0 - (order * 0.05) + 0.04),
    }


def _fallback_stacked(elements: list[dict]) -> list[dict[str, Any]]:
    """Return all elements with stacked fallback bboxes."""
    resolved: list[dict[str, Any]] = []
    for elem in elements:
        text = elem.get("text", "") or elem.get("content", "")
        resolved.append(
            {
                "type": elem.get("type", "P"),
                "order": elem.get("order", 0),
                "alt_text": elem.get("alt_text", ""),
                "content": text,
                "id": elem.get("id", ""),
                "bbox": _stacked_fallback_bbox(elem.get("order", 0)),
                "source": "fallback_stacked",
            }
        )
    return resolved


# ---------------------------------------------------------------------------
# Inline prompt fallback (if XML file not deployed alongside)
# ---------------------------------------------------------------------------


def _inline_prompt_template() -> str:
    """Inline prompt template for environments where the XML file isn't available."""
    return """
    <role>
        <n>Document Element Locator</n>
        <expertise>Precisely mapping content elements to labeled cell grid positions
            using corner-point references for tight bounding boxes</expertise>
    </role>

    <task>Given this document image with a labeled cell grid overlay, locate each listed
        content element by reporting the grid cell and sub-position of its top-left and
        bottom-right corners.</task>

    <grid_structure>
        <columns>{col_letters}</columns>
        <rows>1 through {rows}</rows>
        <cell_format>ColumnRow (e.g., A1, B3, L14)</cell_format>
        <origin>Cell A1 is the top-left corner of the page</origin>
    </grid_structure>

    <known_element_positions>
        The following elements have already been located precisely.
        Use them as spatial reference points to help locate the remaining elements.
{anchor_context}
    </known_element_positions>

    <elements_to_locate>
{element_descriptions}
    </elements_to_locate>

    <instructions>
        For EACH element, report TWO corner positions:
          - top_left: the cell containing the element's top-left corner
          - bottom_right: the cell containing the element's bottom-right corner

        For each corner, also report its approximate position WITHIN the cell:
          - "v": "top", "middle", or "bottom" (vertical third)
          - "h": "left", "center", or "right" (horizontal third)

        Be PRECISE — the sub-positions let you place boundaries between cell edges.
        Use the known element positions above as spatial landmarks.
        Every element MUST get a top_left and bottom_right.
        If you cannot find an element, use your best estimate.
    </instructions>

    <output_format>
        JSON array only, no other text:
        [{{
            "id": "element_id",
            "top_left": {{"cell": "B3", "v": "top", "h": "left"}},
            "bottom_right": {{"cell": "D5", "v": "bottom", "h": "right"}},
            "confidence": "high"
        }}]
    </output_format>
"""
