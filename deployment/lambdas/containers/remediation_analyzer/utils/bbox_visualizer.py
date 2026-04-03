"""Render resolved bounding boxes on a PDF page image.

Usage:
    from utils.bbox_visualizer import visualize_bboxes
    visualize_bboxes(pdf_path, resolved_elements, page_idx=0, output_path="bbox_debug.png")

Or from CLI:
    python utils/bbox_visualizer.py inputs/source.pdf inputs/correlation.xml
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import fitz
from PIL import Image, ImageDraw, ImageFont

# Color palette — each element_id gets a distinct color
_COLORS = [
    (255, 0, 0, 80),  # red
    (0, 128, 0, 80),  # green
    (0, 0, 255, 80),  # blue
    (255, 165, 0, 80),  # orange
    (128, 0, 128, 80),  # purple
    (0, 128, 128, 80),  # teal
    (255, 0, 255, 80),  # magenta
    (128, 128, 0, 80),  # olive
    (0, 0, 128, 80),  # navy
    (255, 99, 71, 80),  # tomato
    (34, 139, 34, 80),  # forest green
    (70, 130, 180, 80),  # steel blue
]

_BORDER_COLORS = [
    (255, 0, 0, 255),
    (0, 128, 0, 255),
    (0, 0, 255, 255),
    (255, 165, 0, 255),
    (128, 0, 128, 255),
    (0, 128, 128, 255),
    (255, 0, 255, 255),
    (128, 128, 0, 255),
    (0, 0, 128, 255),
    (255, 99, 71, 255),
    (34, 139, 34, 255),
    (70, 130, 180, 255),
]


def visualize_bboxes(
    pdf_path: str,
    resolved_elements: List[Dict],
    page_idx: int = 0,
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> str:
    """Render bounding boxes on a PDF page image.

    Args:
        pdf_path: path to the source PDF
        resolved_elements: list of dicts from resolve_text_bboxes/resolve_figure_bboxes
            Each must have: id, type, bbox (normalized 0-1), source, content/alt_text
        page_idx: 0-indexed page number
        output_path: where to save the PNG (default: outputs/bbox_debug_pageN.png)
        dpi: render resolution

    Returns:
        path to the saved PNG
    """
    doc = fitz.open(pdf_path)
    page = doc[page_idx]

    # Render page to image
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()

    pw, ph = img.width, img.height

    # Create overlay for semi-transparent fills
    overlay = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # Draw on the base image for borders and labels
    img_rgba = img.convert("RGBA")

    # Assign colors by element_id
    eid_color_map: Dict[str, int] = {}
    color_idx = 0

    for elem in resolved_elements:
        eid = elem.get("id", "?")
        if eid not in eid_color_map:
            eid_color_map[eid] = color_idx % len(_COLORS)
            color_idx += 1

    # Draw bboxes
    for elem in resolved_elements:
        bbox = elem.get("bbox", {})
        if not bbox:
            continue

        eid = elem.get("id", "?")
        etype = elem.get("type", "?")
        source = elem.get("source", "?")
        text = elem.get("content", "") or elem.get("alt_text", "")
        ci = eid_color_map.get(eid, 0)

        # Convert normalized bbox to pixel coords
        # bbox is {x0, y0, x1, y1} where y0/y1 are bottom-up normalized
        x0 = bbox["x0"] * pw
        x1 = bbox["x1"] * pw
        # y in bbox is bottom-up (0=bottom, 1=top), convert to top-down pixels
        y0 = (1 - bbox["y1"]) * ph
        y1 = (1 - bbox["y0"]) * ph

        # Ensure correct ordering
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0

        # Semi-transparent fill
        draw_overlay.rectangle([x0, y0, x1, y1], fill=_COLORS[ci])

        # Border
        draw_overlay.rectangle([x0, y0, x1, y1], outline=_BORDER_COLORS[ci], width=2)

        # Label
        label = f"{eid} [{etype}] ({source})"
        if len(label) > 60:
            label = label[:57] + "..."

        # Draw label background
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except (OSError, IOError):
            font = ImageFont.load_default(size=10)

        text_bbox = draw_overlay.textbbox((x0 + 2, y0 - 14), label, font=font)
        label_y = max(0, y0 - 14)
        text_bbox = draw_overlay.textbbox((x0 + 2, label_y), label, font=font)
        draw_overlay.rectangle(
            [text_bbox[0] - 1, text_bbox[1] - 1, text_bbox[2] + 1, text_bbox[3] + 1],
            fill=(255, 255, 255, 200),
        )
        border_rgb = _BORDER_COLORS[ci][:3]
        draw_overlay.text((x0 + 2, label_y), label, fill=border_rgb + (255,), font=font)

    # Composite
    result = Image.alpha_composite(img_rgba, overlay)
    result = result.convert("RGB")

    # Save
    if output_path is None:
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        output_path = str(out_dir / f"bbox_debug_page{page_idx}.png")

    result.save(output_path)
    print(f"Saved bbox visualization: {output_path}")
    print(f"  {len(resolved_elements)} elements, {len(eid_color_map)} unique IDs")

    return output_path


# ── CLI ──
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <pdf_path> <xml_path> [--page N]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    xml_path = sys.argv[2]
    page_idx = 0

    if "--page" in sys.argv:
        try:
            page_idx = int(sys.argv[sys.argv.index("--page") + 1])
        except (IndexError, ValueError):
            sys.exit("--page requires a number")

    # Import and run
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, "deployment/lambdas/containers/remediation_analyzer")

    from utils.spine_parser import parse_correlation_xml
    from utils.bbox_resolver import resolve_text_bboxes, resolve_figure_bboxes

    with open(xml_path, encoding="utf-8") as f:
        xml_content = f.read()

    tree, page_elems = parse_correlation_xml(xml_content)
    elems = page_elems.get(1, [])
    if not elems and page_elems:
        elems = next(iter(page_elems.values()))

    text_elems = [e for e in elems if e["type"] != "figure"]
    fig_elems = [e for e in elems if e["type"] == "figure"]

    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    resolved_text = resolve_text_bboxes(page, text_elems)
    resolved_figs = resolve_figure_bboxes(page, fig_elems)
    doc.close()

    resolved = resolved_text + resolved_figs
    visualize_bboxes(pdf_path, resolved, page_idx=page_idx)
