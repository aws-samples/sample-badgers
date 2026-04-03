"""Resolve bounding boxes for text and figure elements using PyMuPDF.

Strategy priority:
  1. Fitz block match — match element text against fitz text blocks (full paragraph bbox)
  2. PyMuPDF text search — search_for() with start + end text
  3. Sanitized search — strip special chars and retry search_for()
  4. Fallback stacked — off-page placeholder
"""

import re
from typing import Dict, List


def _normalize_text(s: str) -> str:
    return re.sub(r"[\s\xa0\t\r\n]+", " ", s).strip().lower()


def _sanitize_for_search(s: str) -> str:
    """Strip characters that fitz search_for() can't match.

    Removes subscripts, superscripts, special math symbols, curly quotes,
    and other Unicode that PDF text layers encode differently.
    """
    # Replace common Unicode variants with ASCII equivalents
    replacements = {
        "\u201c": '"',
        "\u201d": '"',  # curly double quotes
        "\u2018": "'",
        "\u2019": "'",  # curly single quotes
        "\u2013": "-",
        "\u2014": "-",  # en/em dash
        "\u00b7": " ",  # middle dot (multiplication)
        "\u22c5": " ",  # dot operator
        "\u2212": "-",  # minus sign
        "\u00d7": "x",  # multiplication sign
        "\u2264": "<=",
        "\u2265": ">=",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    # Strip subscript/superscript digits (₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹)
    s = re.sub(r"[\u2080-\u2089\u2070-\u2079]", "", s)

    # Strip any remaining non-ASCII
    s = s.encode("ascii", errors="ignore").decode("ascii")

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def _clean_for_search(s: str) -> str:
    """Light cleanup for end-text search — keep more chars than sanitize."""
    s = (
        s.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00b7", " ")
    )
    return s.rstrip(".,;:!?'\")\u201d\u201c\u2019\u2018 ")


def resolve_text_bboxes(page, text_elements: List[Dict]) -> List[Dict]:
    """Resolve bboxes for text elements via fitz line matching + text search.

    Returns list of dicts with normalized bbox coords and source attribution.
    """
    resolved = []
    pw, ph = page.rect.width, page.rect.height
    text_dict = page.get_text("dict")

    # Build fitz line index — each line has bbox and text
    fitz_lines = []
    for block in text_dict.get("blocks", []):
        if block["type"] == 0:
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "") + " "
                fitz_lines.append(
                    {
                        "bbox": line["bbox"],
                        "text_norm": _normalize_text(line_text),
                        "text_san": _normalize_text(_sanitize_for_search(line_text)),
                    }
                )

    for elem in text_elements:
        text = elem.get("text", "")
        if not text:
            continue

        bbox = None
        source = None

        # Strategy 1: Line-level match (best — finds exact lines for this element)
        bbox, source = _line_match(elem, fitz_lines, pw, ph)

        # Strategy 2: PyMuPDF text search with start + end
        if not bbox:
            bbox, source = _search_text_bbox(page, pw, ph, text)

        # Strategy 3: Sanitized text search (strip special chars)
        if not bbox:
            bbox, source = _sanitized_search(page, pw, ph, text)

        # Strategy 4: Fallback stacked
        if not bbox:
            source = "fallback_stacked"
            order = elem.get("order", 0)
            bbox = {
                "x0": 0.02,
                "y0": max(0.02, 1.0 - order * 0.05),
                "x1": 0.98,
                "y1": min(0.98, 1.0 - order * 0.05 + 0.04),
            }

        resolved.append(
            {
                "type": elem["type"],
                "order": elem["order"],
                "alt_text": elem.get("alt_text", ""),
                "content": text,
                "id": elem.get("id", ""),
                "tag": elem.get("tag", ""),
                "bbox": bbox,
                "source": source,
            }
        )
    return resolved


def _line_match(elem, fitz_lines, pw, ph):
    """Match element text against fitz lines to find the exact vertical extent.

    Finds the first line containing the element's text start, then finds the
    last line containing the element's text end. The bbox spans from the first
    line's top to the last line's bottom, full page width.
    """
    elem_text = elem.get("text", "")
    elem_norm = _normalize_text(elem_text)
    elem_san = _normalize_text(_sanitize_for_search(elem_text))

    # Find the first line that contains the start of this element's text
    start_line_idx = None
    for prefix_len in (40, 30, 20, 15, 10):
        prefix_norm = elem_norm[:prefix_len]
        prefix_san = elem_san[:prefix_len]
        if len(prefix_norm) < 8:
            continue

        for li, fl in enumerate(fitz_lines):
            if prefix_norm in fl["text_norm"] or prefix_san in fl["text_san"]:
                start_line_idx = li
                break
        if start_line_idx is not None:
            break

    if start_line_idx is None:
        return None, None

    # Find the last line that contains the end of this element's text
    end_line_idx = start_line_idx  # default: same line

    # Estimate max lines this element could span (avg ~60 chars per line)
    max_lines = max(4, len(elem_norm) // 40)
    max_end_line = min(start_line_idx + max_lines, len(fitz_lines) - 1)

    if len(elem_norm) > 60:
        # Try progressively shorter suffixes
        for suffix_len in (40, 30, 20, 15, 10):
            suffix_norm = elem_norm[-suffix_len:]
            suffix_san = elem_san[-suffix_len:]
            if len(suffix_norm) < 6:
                continue

            for li in range(start_line_idx, max_end_line + 1):
                if (
                    suffix_norm in fitz_lines[li]["text_norm"]
                    or suffix_san in fitz_lines[li]["text_san"]
                ):
                    end_line_idx = li
                    break
            if end_line_idx > start_line_idx:
                break

        # If suffix matching failed, try distinctive words from the end
        if end_line_idx == start_line_idx:
            words = elem_san.split()
            for word in reversed(words[-8:]):
                if len(word) < 5:
                    continue
                for li in range(start_line_idx, max_end_line + 1):
                    if word in fitz_lines[li]["text_san"]:
                        end_line_idx = li
                if end_line_idx > start_line_idx:
                    break

        # If text matching still failed, use spatial continuity:
        # extend to consecutive lines that are vertically adjacent
        # (no gap > 1.5x line height) and similarly indented
        if end_line_idx == start_line_idx and max_lines > 1:
            start_bb = fitz_lines[start_line_idx]["bbox"]
            line_height = start_bb[3] - start_bb[1]
            max_gap = line_height * 2.0
            start_x0 = start_bb[0]

            for li in range(start_line_idx + 1, max_end_line + 1):
                prev_bb = fitz_lines[li - 1]["bbox"]
                curr_bb = fitz_lines[li]["bbox"]
                gap = curr_bb[1] - prev_bb[3]  # vertical gap between lines
                x_diff = abs(curr_bb[0] - start_x0)

                # Stop if there's a large gap (paragraph break) or big indent change
                # or if the line is empty (paragraph separator)
                if gap > max_gap or x_diff > 100:
                    break
                if not fitz_lines[li]["text_san"].strip():
                    break  # empty line = paragraph boundary
                end_line_idx = li

    # Build bbox from start line top to end line bottom
    start_bb = fitz_lines[start_line_idx]["bbox"]
    end_bb = fitz_lines[end_line_idx]["bbox"]

    # Use the leftmost x0 and rightmost x1 across all lines in range
    x0 = start_bb[0]
    x1 = start_bb[2]
    for li in range(start_line_idx, end_line_idx + 1):
        bb = fitz_lines[li]["bbox"]
        x0 = min(x0, bb[0])
        x1 = max(x1, bb[2])

    y0 = start_bb[1]  # top of first line
    y1 = end_bb[3]  # bottom of last line

    bbox = {
        "x0": x0 / pw,
        "y0": 1 - (y1 / ph),
        "x1": x1 / pw,
        "y1": 1 - (y0 / ph),
    }
    return bbox, "fitz_line_match"


def _search_text_bbox(page, pw, ph, text):
    """PyMuPDF text search for start (and end) of text content."""
    # Try full text first
    rects = page.search_for(text)
    if not rects:
        # Try first 50 chars
        rects = page.search_for(text[:50])
        source = "pymupdf_partial_search" if rects else None
    else:
        source = "pymupdf_text_search"

    if not rects or not source:
        return None, None

    x0 = min(r.x0 for r in rects)
    y0 = min(r.y0 for r in rects)
    x1 = max(r.x1 for r in rects)
    y1 = max(r.y1 for r in rects)

    # For long text, search for the ending to extend bbox
    if len(text) > 80:
        for end_len in (50, 30, 20):
            end_text = _clean_for_search(text[-end_len:])
            end_rects = page.search_for(end_text)
            if end_rects:
                y1 = max(y1, max(r.y1 for r in end_rects))
                x1 = max(x1, max(r.x1 for r in end_rects))
                source = "pymupdf_start_end_search"
                break

    bbox = {
        "x0": x0 / pw,
        "y0": 1 - (y1 / ph),
        "x1": x1 / pw,
        "y1": 1 - (y0 / ph),
    }
    return bbox, source


def _sanitized_search(page, pw, ph, text):
    """Search with sanitized text (special chars stripped)."""
    clean = _sanitize_for_search(text)
    if not clean or len(clean) < 10:
        return None, None

    # Try first 60 chars sanitized
    for prefix_len in (60, 40, 30):
        prefix = clean[:prefix_len]
        if len(prefix) < 10:
            continue
        rects = page.search_for(prefix)
        if rects:
            x0 = min(r.x0 for r in rects)
            y0 = min(r.y0 for r in rects)
            x1 = max(r.x1 for r in rects)
            y1 = max(r.y1 for r in rects)

            # Try to find the end too
            if len(clean) > 80:
                for end_len in (40, 30, 20):
                    end_text = clean[-end_len:]
                    end_rects = page.search_for(end_text)
                    if end_rects:
                        y1 = max(y1, max(r.y1 for r in end_rects))
                        x1 = max(x1, max(r.x1 for r in end_rects))
                        break

            bbox = {
                "x0": x0 / pw,
                "y0": 1 - (y1 / ph),
                "x1": x1 / pw,
                "y1": 1 - (y0 / ph),
            }
            return bbox, "pymupdf_sanitized_search"

    return None, None


def resolve_figure_bboxes(page, figure_elements: List[Dict]) -> List[Dict]:
    """Resolve figure element bboxes from fitz image blocks."""
    resolved = []
    pw, ph = page.rect.width, page.rect.height
    td = page.get_text("dict")
    image_blocks = [b for b in td.get("blocks", []) if b["type"] == 1]

    for i, elem in enumerate(figure_elements):
        if i < len(image_blocks):
            bb = image_blocks[i]["bbox"]
            bbox = {
                "x0": bb[0] / pw,
                "y0": 1 - (bb[3] / ph),
                "x1": bb[2] / pw,
                "y1": 1 - (bb[1] / ph),
            }
            source = "fitz_image_block"
        else:
            bbox = {"x0": 0.1, "y0": 0.2, "x1": 0.9, "y1": 0.6}
            source = "fallback_figure"

        resolved.append(
            {
                "type": "figure",
                "order": elem.get("order", 0),
                "alt_text": elem.get("alt_text", ""),
                "content": "",
                "id": elem.get("id", ""),
                "tag": elem.get("tag", ""),
                "bbox": bbox,
                "source": source,
            }
        )
    return resolved
