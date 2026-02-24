"""
Diagnostic Visualizer — Saves element detection data and bbox overlay images.

Captures the full pipeline output at the page level:
  - Correlation analyzer output (what to find: element types, text, reading order)
  - Grid resolver output (where it is: bboxes, confidence, source tier)
  - Bbox overlay image showing all detected elements color-coded by type

Saves to S3 under:
  {analyzer_name}/diagnostics/{session_id}/{pdf_stem}/
    page_{N}_elements.json
    page_{N}_bboxes.png
    page_{N}_grid.png        (optional: the gridded image the model received)

Can also save locally for development/debugging.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color palette by element type
# ---------------------------------------------------------------------------

_TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    # Headings
    "H1": (255, 60, 60),
    "H2": (255, 150, 50),
    "H3": (255, 200, 80),
    "H4": (220, 180, 100),
    "H5": (200, 160, 100),
    "H6": (180, 140, 100),
    # Text
    "P": (150, 200, 255),
    "BlockQuote": (180, 150, 255),
    "Quote": (180, 150, 255),
    "Code": (170, 220, 170),
    # Tables
    "Table": (0, 200, 120),
    "Caption": (200, 200, 100),
    # Lists
    "L": (100, 200, 200),
    "LI": (80, 180, 180),
    # Figures & media
    "Figure": (100, 220, 100),
    "Formula": (220, 180, 255),
    # Navigation
    "TOC": (200, 200, 200),
    "TOCI": (180, 180, 180),
    "Link": (100, 150, 255),
    # Annotations
    "Note": (255, 100, 255),
    "Artifact": (180, 180, 180),
    # Structural
    "NonStruct": (120, 120, 120),
}

_DEFAULT_COLOR = (200, 200, 200)


def _color_for_type(elem_type: str) -> tuple[int, int, int]:
    """Get display color for an element type."""
    return _TYPE_COLORS.get(elem_type, _DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

_vis_font: Optional[ImageFont.FreeTypeFont] = None
_vis_font_sm: Optional[ImageFont.FreeTypeFont] = None
_vis_font_loaded: bool = False

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/google-noto/NotoSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_vis_fonts():
    """Load TrueType fonts for overlay labels."""
    global _vis_font, _vis_font_sm, _vis_font_loaded
    if _vis_font_loaded:
        return

    _vis_font_loaded = True
    for path in _FONT_PATHS:
        if os.path.exists(path):
            try:
                _vis_font = ImageFont.truetype(path, 12)
                _vis_font_sm = ImageFont.truetype(path, 9)
                return
            except Exception:
                continue

    _vis_font = ImageFont.load_default()
    _vis_font_sm = ImageFont.load_default()


def _get_font(small: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    _load_vis_fonts()
    return _vis_font_sm if small else _vis_font


# ---------------------------------------------------------------------------
# JSON assembly
# ---------------------------------------------------------------------------


def build_page_diagnostic(
    page_number: int,
    correlation_elements: list[dict[str, Any]],
    resolved_elements: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    grid_cols: int,
    grid_rows: int,
    pdf_path: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    """Build the full diagnostic JSON for a single page.

    Merges correlation analyzer output (element identity) with grid resolver
    output (element position) into a single record per element.

    Args:
        page_number: 1-indexed page number.
        correlation_elements: Raw elements from correlation XML for this page.
        resolved_elements: Elements after bbox resolution (all tiers combined).
        image_width: Rendered page image width in pixels.
        image_height: Rendered page image height in pixels.
        grid_cols: Grid columns used by cell grid resolver.
        grid_rows: Grid rows used by cell grid resolver.
        pdf_path: Source PDF path/URI (for reference).
        session_id: Processing session ID.

    Returns:
        Diagnostic dict ready for JSON serialization.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Index resolved elements by ID for fast lookup
    resolved_by_id: dict[str, dict] = {}
    for elem in resolved_elements:
        eid = elem.get("id", "")
        if eid:
            resolved_by_id[eid] = elem

    # Build merged element records
    elements = []
    for corr in correlation_elements:
        eid = corr.get("id", "")
        resolved = resolved_by_id.get(eid, {})

        bbox = resolved.get("bbox", {})
        source = resolved.get("source", "unresolved")

        # Compute pixel bbox from normalized PDF coords for readability
        pixel_bbox = None
        if bbox:
            # PDF coords are bottom-left origin; convert to top-left for pixels
            pixel_bbox = {
                "x0": int(bbox.get("x0", 0) * image_width),
                "y0": int((1.0 - bbox.get("y1", 0)) * image_height),
                "x1": int(bbox.get("x1", 0) * image_width),
                "y1": int((1.0 - bbox.get("y0", 0)) * image_height),
            }

        record = {
            # Identity (from correlation analyzer)
            "id": eid,
            "type": corr.get("type", "unknown"),
            "order": corr.get("order", 0),
            "text": corr.get("text", "") or corr.get("content", ""),
            "alt_text": corr.get("alt_text", ""),

            # Position (from grid resolver / text search / fallback)
            "bbox_normalized_pdf": bbox if bbox else None,
            "bbox_pixels": pixel_bbox,
            "resolution_source": source,

            # Confidence breakdown
            "resolution_tier": _classify_source(source),
        }

        elements.append(record)

    # Also capture any resolved elements not in correlation (shouldn't happen,
    # but defensive)
    corr_ids = {c.get("id", "") for c in correlation_elements}
    for eid, resolved in resolved_by_id.items():
        if eid not in corr_ids:
            bbox = resolved.get("bbox", {})
            pixel_bbox = None
            if bbox:
                pixel_bbox = {
                    "x0": int(bbox.get("x0", 0) * image_width),
                    "y0": int((1.0 - bbox.get("y1", 0)) * image_height),
                    "x1": int(bbox.get("x1", 0) * image_width),
                    "y1": int((1.0 - bbox.get("y0", 0)) * image_height),
                }
            elements.append({
                "id": eid,
                "type": resolved.get("type", "unknown"),
                "order": resolved.get("order", 0),
                "text": resolved.get("content", ""),
                "alt_text": resolved.get("alt_text", ""),
                "bbox_normalized_pdf": bbox if bbox else None,
                "bbox_pixels": pixel_bbox,
                "resolution_source": resolved.get("source", "unknown"),
                "resolution_tier": _classify_source(resolved.get("source", "")),
                "_note": "not in correlation XML",
            })

    # Summary stats
    tier_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for elem in elements:
        tier = elem.get("resolution_tier", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        etype = elem.get("type", "unknown")
        type_counts[etype] = type_counts.get(etype, 0) + 1

    return {
        "metadata": {
            "timestamp": timestamp,
            "session_id": session_id,
            "pdf_path": pdf_path,
            "page_number": page_number,
            "image_dimensions": {"width": image_width, "height": image_height},
            "grid": {"cols": grid_cols, "rows": grid_rows},
            "resolver_version": "v3_corner_point",
        },
        "summary": {
            "total_elements": len(elements),
            "by_type": type_counts,
            "by_resolution_tier": tier_counts,
        },
        "elements": elements,
    }


def _classify_source(source: str) -> str:
    """Classify a resolution source into a human-readable tier label."""
    if not source:
        return "unresolved"
    if source.startswith("text_search"):
        return "tier1_text_search"
    if "refined" in source:
        return "tier2_grid_refined"
    if source.startswith("cell_grid"):
        return "tier2_grid"
    if source == "fallback_stacked":
        return "tier3_fallback"
    if source.startswith("vision_model"):
        return "tier2_vision"
    return source


# ---------------------------------------------------------------------------
# Bbox overlay image rendering
# ---------------------------------------------------------------------------


def render_bbox_overlay(
    page_image: Image.Image,
    resolved_elements: list[dict[str, Any]],
    max_dim: int = 2048,
) -> Image.Image:
    """Render bounding box overlay on a page image.

    Draws color-coded translucent rectangles for each element with
    type labels and corner dots.

    Args:
        page_image: Base page image (any size — will be resized if needed).
        resolved_elements: Elements with bbox data.
        max_dim: Max output dimension (keeps aspect ratio).

    Returns:
        RGB PIL Image with overlay.
    """
    img = page_image.copy().convert("RGBA")

    # Resize for manageable output
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    width, height = img.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = _get_font(small=False)
    font_sm = _get_font(small=True)

    for elem in resolved_elements:
        bbox = elem.get("bbox", {})
        if not bbox:
            continue

        elem_type = elem.get("type", "P")
        source = elem.get("source", "")
        color = _color_for_type(elem_type)

        # Convert PDF coords (bottom-left origin) to image coords (top-left)
        px0 = int(bbox.get("x0", 0) * width)
        py0 = int((1.0 - bbox.get("y1", 0)) * height)
        px1 = int(bbox.get("x1", 0) * width)
        py1 = int((1.0 - bbox.get("y0", 0)) * height)

        # Clamp
        px0, py0 = max(0, px0), max(0, py0)
        px1, py1 = min(width, px1), min(height, py1)

        if px0 >= px1 or py0 >= py1:
            continue

        # Translucent fill + solid border
        draw.rectangle(
            [px0, py0, px1, py1],
            fill=(color[0], color[1], color[2], 30),
            outline=(color[0], color[1], color[2], 200),
            width=2,
        )

        # Corner dots
        r = 4
        draw.ellipse([px0 - r, py0 - r, px0 + r, py0 + r],
                      fill=(color[0], color[1], color[2], 240))
        draw.ellipse([px1 - r, py1 - r, px1 + r, py1 + r],
                      fill=(color[0], color[1], color[2], 240))

        # Label
        text_content = elem.get("content", "") or elem.get("text", "")
        display = text_content[:50] + "..." if len(text_content) > 50 else text_content
        label = f"[{elem_type}] {display}" if display else f"[{elem_type}]"

        # Source tier indicator
        tier = _classify_source(source)
        tier_short = {"tier1_text_search": "T1", "tier2_grid": "T2",
                      "tier2_grid_refined": "T2R", "tier3_fallback": "FB",
                      "tier2_vision": "V"}.get(tier, "?")
        label = f"{tier_short} {label}"

        lx, ly = px0 + 3, py0 + 2
        tb = draw.textbbox((lx, ly), label, font=font_sm)
        draw.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1],
                        fill=(0, 0, 0, 210))
        draw.text((lx, ly), label, fill=(color[0], color[1], color[2], 255),
                  font=font_sm)

    return Image.alpha_composite(img, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# Save to S3 / local
# ---------------------------------------------------------------------------


def save_diagnostics(
    diagnostic_json: dict[str, Any],
    bbox_image: Image.Image,
    gridded_image: Optional[Image.Image] = None,
    output_bucket: Optional[str] = None,
    analyzer_name: str = "remediation_analyzer",
    session_id: str = "local",
    pdf_stem: str = "document",
    page_number: int = 1,
    local_dir: Optional[str] = None,
) -> dict[str, str]:
    """Save diagnostic JSON and overlay images to S3 and/or local disk.

    Args:
        diagnostic_json: Output from build_page_diagnostic().
        bbox_image: Output from render_bbox_overlay().
        gridded_image: Optional gridded image (what the model received).
        output_bucket: S3 bucket for diagnostics. If None, S3 save is skipped.
        analyzer_name: Analyzer name for S3 path prefix.
        session_id: Session ID for S3 path.
        pdf_stem: PDF filename without extension (for path).
        page_number: 1-indexed page number.
        local_dir: If provided, also saves locally to this directory.

    Returns:
        Dict of output URIs/paths: {"json": "...", "bbox_image": "...", ...}
    """
    page_tag = f"page_{page_number:03d}"
    outputs: dict[str, str] = {}

    # Serialize JSON
    json_bytes = json.dumps(diagnostic_json, indent=2, ensure_ascii=False).encode("utf-8")

    # Serialize images
    bbox_buf = BytesIO()
    bbox_image.save(bbox_buf, format="PNG", optimize=True)
    bbox_bytes = bbox_buf.getvalue()

    grid_bytes = None
    if gridded_image is not None:
        grid_buf = BytesIO()
        gridded_image.save(grid_buf, format="PNG", optimize=True)
        grid_bytes = grid_buf.getvalue()

    # ── Local save ──
    if local_dir:
        local_path = Path(local_dir) / pdf_stem
        local_path.mkdir(parents=True, exist_ok=True)

        json_path = local_path / f"{page_tag}_elements.json"
        json_path.write_bytes(json_bytes)
        outputs["json_local"] = str(json_path)

        bbox_path = local_path / f"{page_tag}_bboxes.png"
        bbox_path.write_bytes(bbox_bytes)
        outputs["bbox_image_local"] = str(bbox_path)

        if grid_bytes:
            grid_path = local_path / f"{page_tag}_grid.png"
            grid_path.write_bytes(grid_bytes)
            outputs["grid_image_local"] = str(grid_path)

        logger.info("Diagnostics saved locally: %s", local_path)

    # ── S3 save ──
    if output_bucket:
        try:
            import boto3

            s3 = boto3.client("s3")
            s3_prefix = f"{analyzer_name}/diagnostics/{session_id}/{pdf_stem}"

            # JSON
            json_key = f"{s3_prefix}/{page_tag}_elements.json"
            s3.put_object(
                Bucket=output_bucket,
                Key=json_key,
                Body=json_bytes,
                ContentType="application/json",
            )
            outputs["json_s3"] = f"s3://{output_bucket}/{json_key}"

            # Bbox overlay
            bbox_key = f"{s3_prefix}/{page_tag}_bboxes.png"
            s3.put_object(
                Bucket=output_bucket,
                Key=bbox_key,
                Body=bbox_bytes,
                ContentType="image/png",
            )
            outputs["bbox_image_s3"] = f"s3://{output_bucket}/{bbox_key}"

            # Grid image (optional)
            if grid_bytes:
                grid_key = f"{s3_prefix}/{page_tag}_grid.png"
                s3.put_object(
                    Bucket=output_bucket,
                    Key=grid_key,
                    Body=grid_bytes,
                    ContentType="image/png",
                )
                outputs["grid_image_s3"] = f"s3://{output_bucket}/{grid_key}"

            logger.info(
                "Diagnostics saved to S3: s3://%s/%s/ (%d files)",
                output_bucket, s3_prefix, len([k for k in outputs if k.endswith("_s3")]),
            )

        except Exception as e:
            logger.error("Failed to save diagnostics to S3: %s", e)

    return outputs


# ---------------------------------------------------------------------------
# Convenience: one-call from lambda_handler
# ---------------------------------------------------------------------------


def capture_page_diagnostics(
    page_image_data: bytes,
    page_number: int,
    correlation_elements: list[dict[str, Any]],
    resolved_elements: list[dict[str, Any]],
    grid_cols: int,
    grid_rows: int,
    gridded_image: Optional[Image.Image] = None,
    pdf_path: str = "",
    session_id: str = "",
    output_bucket: Optional[str] = None,
    analyzer_name: str = "remediation_analyzer",
) -> dict[str, str]:
    """One-call convenience for lambda_handler integration.

    Builds the diagnostic JSON, renders the overlay image, and saves both.

    Args:
        page_image_data: Raw image bytes (PNG/JPEG) of the rendered page.
        page_number: 1-indexed page number.
        correlation_elements: Elements from correlation XML for this page.
        resolved_elements: Fully resolved elements (all tiers combined).
        grid_cols: Grid columns used by resolver.
        grid_rows: Grid rows used by resolver.
        gridded_image: Optional gridded PIL Image (what model saw).
        pdf_path: Source PDF path/URI.
        session_id: Processing session ID.
        output_bucket: S3 bucket (from OUTPUT_BUCKET env if None).
        analyzer_name: Analyzer name (from ANALYZER_NAME env if default).

    Returns:
        Dict of output URIs/paths.
    """
    if output_bucket is None:
        output_bucket = os.environ.get("OUTPUT_BUCKET")

    # Bail early if diagnostics are disabled
    if not os.environ.get("ENABLE_DIAGNOSTICS", "").lower() in ("1", "true", "yes"):
        logger.debug("Diagnostics disabled (set ENABLE_DIAGNOSTICS=true to enable)")
        return {}

    try:
        page_img = Image.open(BytesIO(page_image_data)).convert("RGB")
        w, h = page_img.size

        # Build JSON
        diagnostic = build_page_diagnostic(
            page_number=page_number,
            correlation_elements=correlation_elements,
            resolved_elements=resolved_elements,
            image_width=w,
            image_height=h,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            pdf_path=pdf_path,
            session_id=session_id,
        )

        # Render overlay
        bbox_img = render_bbox_overlay(page_img, resolved_elements)

        # Derive pdf_stem
        pdf_stem = Path(pdf_path).stem if pdf_path else "document"
        # Strip s3 prefix artifacts
        if "/" in pdf_stem:
            pdf_stem = pdf_stem.rsplit("/", 1)[-1]

        return save_diagnostics(
            diagnostic_json=diagnostic,
            bbox_image=bbox_img,
            gridded_image=gridded_image,
            output_bucket=output_bucket,
            analyzer_name=analyzer_name,
            session_id=session_id,
            pdf_stem=pdf_stem,
            page_number=page_number,
        )

    except Exception as e:
        logger.error("Failed to capture page diagnostics: %s", e)
        return {}
