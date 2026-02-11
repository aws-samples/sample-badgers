"""
Cell Grid Resolver — Vision-model bbox detection using labeled cell grids.

Instead of asking a vision model for raw normalized coordinates (which are
imprecise), this module:
  1. Overlays a labeled cell grid (A1, B2, ...) on the page image
  2. Sends the gridded image + element descriptions to the vision model
  3. Asks which cells each element occupies
  4. Converts cell references back to normalized bboxes

This approach exploits the fact that vision models are excellent at
reading grid labels and associating content with spatial references,
but poor at estimating raw pixel coordinates.

Adapted from the cell-grid detection pattern in unclear_region_detector.py.
"""

import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


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

            # Cell label (small, top-left corner of cell)
            label_x = x0 + 2
            label_y = y0 + 1
            text_bbox = draw.textbbox((label_x, label_y), cell_name)
            draw.rectangle(
                [
                    text_bbox[0] - 1,
                    text_bbox[1] - 1,
                    text_bbox[2] + 1,
                    text_bbox[3] + 1,
                ],
                fill=label_bg,
            )
            draw.text((label_x, label_y), cell_name, fill=label_color)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    return result, cell_map


def cells_to_bbox(
    cells: list[str],
    cell_map: dict[str, dict],
) -> Optional[dict[str, float]]:
    """
    Convert a list of cell names to a normalized bounding box.

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


def _image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


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
# Core resolver
# ---------------------------------------------------------------------------


def resolve_elements_via_grid(
    image_data: bytes,
    unresolved_elements: list[dict[str, Any]],
    analyzer: Any,
    _aws_profile: Optional[str] = None,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
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
        aws_profile: AWS profile (reserved).
        cols: Grid columns override (auto-sized if None).
        rows: Grid rows override (auto-sized if None).

    Returns:
        List of resolved element dicts in the same format as
        _resolve_text_bboxes / _resolve_figure_bboxes output, with
        source="cell_grid_resolver".
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

    # Create gridded image
    gridded, cell_map = add_cell_grid_overlay(img, cols=cols, rows=rows)
    gridded_b64 = _image_to_base64(gridded)

    # Build element descriptions for the prompt
    col_letters = [chr(ord("A") + i) for i in range(cols)]
    element_lines = []
    for elem in unresolved_elements:
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

    # Build prompt
    try:
        prompt_template = _load_prompt_template()
    except FileNotFoundError:
        # Inline fallback if XML file not deployed
        prompt_template = _inline_prompt_template()

    prompt = prompt_template.format(
        col_letters=", ".join(col_letters),
        rows=rows,
        element_descriptions=element_descriptions,
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
                        "media_type": "image/png",
                        "data": gridded_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        bedrock_client = analyzer.bedrock_client
        model_id = analyzer.config.get(
            "model_id", "us.anthropic.claude-sonnet-4-20250514-v1:0"
        )

        system_prompt = (
            "You are a precise document element locator. Given a page image with a "
            "labeled cell grid overlay and a list of content elements, report which "
            "grid cells each element occupies. Return ONLY a JSON array."
        )

        payload = bedrock_client.create_anthropic_payload(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=4000,
            temperature=0.1,
        )

        response = bedrock_client.invoke_model(
            model_id=model_id,
            payload=payload,
        )

        # Extract text
        result_text = ""
        for block in response.get("content", []):
            if block.get("type") == "text":
                result_text = block.get("text", "")
                break

        if not result_text:
            logger.warning("Empty response from cell grid resolver")
            return _fallback_stacked(unresolved_elements)

        # Parse response
        grid_results = _parse_grid_response(result_text)

    except Exception as e:
        logger.error("Cell grid resolver vision call failed: %s", e)
        return _fallback_stacked(unresolved_elements)

    # Build lookup: element_id → cell list
    cell_lookup: dict[str, tuple[list[str], str]] = {}
    for item in grid_results:
        item_id = item.get("id", "")
        item_cells = item.get("cells", [])
        confidence = item.get("confidence", "medium")
        if item_id and item_cells:
            cell_lookup[item_id] = (item_cells, confidence)

    # Convert cell references to bboxes and build resolved elements
    resolved = []
    resolved_count = 0
    fallback_count = 0

    for elem in unresolved_elements:
        elem_id = elem.get("id", f"order_{elem.get('order', '?')}")
        elem_type = elem.get("type", "P")
        text = elem.get("text", "") or elem.get("content", "")

        cell_info = cell_lookup.get(elem_id)
        if cell_info:
            cells, confidence = cell_info
            bbox = cells_to_bbox(cells, cell_map)

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
                    "Grid-resolved %s → cells=%s bbox=(%.3f,%.3f)-(%.3f,%.3f)",
                    elem_id,
                    cells,
                    pdf_bbox["x0"],
                    pdf_bbox["y0"],
                    pdf_bbox["x1"],
                    pdf_bbox["y1"],
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
        len(unresolved_elements),
        fallback_count,
    )
    return resolved


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_grid_response(response_text: str) -> list[dict]:
    """Parse JSON array from model response, handling markdown wrappers."""
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
        <expertise>Precisely mapping content elements to labeled cell grid positions</expertise>
    </role>

    <task>Given this document image with a labeled cell grid overlay, locate each listed
        content element and report which grid cells it occupies.</task>

    <grid_structure>
        <columns>{col_letters}</columns>
        <rows>1 through {rows}</rows>
        <cell_format>ColumnRow (e.g., A1, B3, L14)</cell_format>
        <origin>Cell A1 is the top-left corner</origin>
    </grid_structure>

    <elements_to_locate>
{element_descriptions}
    </elements_to_locate>

    <instructions>
        For EACH element, identify the grid cells where it appears.
        Be TIGHT — only include cells that genuinely overlap the element.
        Every element MUST get a cells array.
        If you cannot find an element, use your best estimate.
    </instructions>

    <output_format>
        JSON array only, no other text:
        [{{"id": "element_id", "cells": ["B3", "C3"], "confidence": "high"}}]
    </output_format>
"""
