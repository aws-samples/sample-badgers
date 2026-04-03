#!/usr/bin/env python3
"""
PDF Content Panel Viewer
Replicates the tree view from Adobe Acrobat's Content panel.
Shows marked content containers (<H1>, <P>, etc.) and the text within them.

Usage:
    python pdf_content_panel.py document.pdf
    python pdf_content_panel.py document.pdf --page 1
    python pdf_content_panel.py document.pdf --json
"""

import sys
import json
from pathlib import Path

try:
    import pikepdf
except ImportError:
    sys.exit("pikepdf not installed. Run:  pip install pikepdf")


def _safe_text(val):
    """Safely extract a text string from a pikepdf operand."""
    if isinstance(val, pikepdf.String):
        try:
            return val.to_str()
        except Exception:
            return bytes(val).decode("latin-1", errors="replace")
    if isinstance(val, bytes):
        return val.decode("latin-1", errors="replace")
    if isinstance(val, str):
        return val
    # For other pikepdf types, check if it wraps bytes
    try:
        raw = bytes(val)
        return raw.decode("latin-1", errors="replace")
    except (TypeError, ValueError):
        pass
    return None


def extract_page_content_tree(page):
    """
    Parse a page's content stream and build the marked-content container tree,
    mirroring what Adobe Acrobat shows in its Content panel.
    """
    tree = []
    stack = [tree]  # stack of child lists; top = current insertion point

    try:
        commands = pikepdf.parse_content_stream(page)
    except Exception as e:
        return [{"type": "error", "message": str(e)}]

    current_text_parts = []

    for operands, operator in commands:
        op = str(operator)

        # ── Marked content begin (container open) ─────────────────────
        if op in ("BMC", "BDC"):
            tag = str(operands[0]) if operands else "unknown"
            props = {}
            if op == "BDC" and len(operands) > 1:
                prop_obj = operands[1]
                if isinstance(prop_obj, pikepdf.Dictionary):
                    for k, v in prop_obj.items():
                        try:
                            props[str(k)] = str(v)
                        except Exception:
                            pass

            node = {
                "type": "container",
                "tag": tag,
                "children": [],
            }
            if props:
                node["properties"] = props

            stack[-1].append(node)
            stack.append(node["children"])

        # ── Marked content end (container close) ──────────────────────
        elif op == "EMC":
            # Flush any pending text
            if current_text_parts:
                text = "".join(current_text_parts).strip()
                if text:
                    stack[-1].append({"type": "text", "content": text})
                current_text_parts = []

            if len(stack) > 1:
                stack.pop()

        # ── Text showing operators ────────────────────────────────────
        elif op in ("Tj", "'", '"'):
            if operands:
                val = operands[-1]
                text = _safe_text(val)
                if text:
                    current_text_parts.append(text)

        elif op == "TJ":
            # TJ takes an array of strings and positioning numbers
            if operands and isinstance(operands[0], pikepdf.Array):
                for item in operands[0]:
                    text = _safe_text(item)
                    if text:
                        current_text_parts.append(text)
                    else:
                        # Large negative kerning values often indicate a space
                        try:
                            num = float(str(item))
                            if num < -200:
                                current_text_parts.append(" ")
                        except (ValueError, TypeError):
                            pass

        # ── Form XObject invocation (nested content) ──────────────────
        elif op == "Do":
            name = str(operands[0]) if operands else "unknown"
            stack[-1].append({"type": "xobject", "name": name})

    # Flush any remaining text
    if current_text_parts:
        text = "".join(current_text_parts).strip()
        if text:
            stack[-1].append({"type": "text", "content": text})

    return tree


def _render_content_tree(nodes, lines, indent=0):
    """Render the content tree into a list of strings (no printing)."""
    prefix = "  " * indent
    for node in nodes:
        ntype = node["type"]

        if ntype == "container":
            tag = node["tag"]
            preview = _get_text_preview(node, max_len=55)
            props_str = ""
            if "properties" in node:
                mcid = node["properties"].get("/MCID", "")
                if mcid:
                    props_str = f"  [MCID {mcid}]"

            lines.append(f"{prefix}📦 Container <{tag}>{props_str} {preview}")

            if node.get("children"):
                _render_content_tree(node["children"], lines, indent + 1)

        elif ntype == "text":
            content = node["content"]
            if len(content) > 70:
                content = content[:67] + "..."
            lines.append(f"{prefix}📝 Text: {content}")

        elif ntype == "xobject":
            lines.append(f"{prefix}🖼  XObject: {node['name']}")

        elif ntype == "error":
            lines.append(f"{prefix}⚠️  Error: {node['message']}")


def print_content_tree(nodes, indent=0):
    """Pretty-print the content tree like Adobe's Content panel."""
    lines = []
    _render_content_tree(nodes, lines, indent)
    print("\n".join(lines))


def _get_text_preview(node, max_len=55):
    """Collect a short text preview from a container's descendants."""
    parts = []

    def _collect(n):
        if n["type"] == "text":
            parts.append(n["content"])
        elif n["type"] == "container" and n.get("children"):
            for child in n["children"]:
                _collect(child)

    for child in node.get("children", []):
        _collect(child)
        if sum(len(p) for p in parts) > max_len:
            break

    preview = " ".join(parts).strip()
    if len(preview) > max_len:
        preview = preview[: max_len - 3] + "..."
    return preview


def extract_all_pages(pdf_path, page_num=None):
    """Extract the content tree for one or all pages."""
    pdf = pikepdf.open(pdf_path)
    results = []

    if page_num is not None:
        idx = page_num - 1
        if idx < 0 or idx >= len(pdf.pages):
            sys.exit(f"Page {page_num} out of range (1-{len(pdf.pages)})")
        pages_to_process = [(idx, pdf.pages[idx])]
    else:
        pages_to_process = list(enumerate(pdf.pages))

    for idx, page in pages_to_process:
        tree = extract_page_content_tree(page)
        results.append(
            {
                "page": idx + 1,
                "content": tree,
            }
        )

    pdf.close()
    return results


def _build_output(results):
    """Build the full text output for content tree results."""
    lines = []
    for page_data in results:
        lines.append("")
        lines.append("═" * 60)
        lines.append(f"  Page {page_data['page']}")
        lines.append("═" * 60)
        if page_data["content"]:
            _render_content_tree(page_data["content"], lines, indent=1)
        else:
            lines.append("  (no marked content on this page)")
    return "\n".join(lines)


def view(pdf_path, page_num=None, as_json=False):
    """Convenience entry point for notebook use.

    Returns the results list. Prints tree or JSON to stdout.
    """
    results = extract_all_pages(pdf_path, page_num)
    if as_json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(_build_output(results))
    return results


def save(pdf_path, page_num=None, as_json=False):
    """Extract the content tree and save to a panel/ directory next to the PDF.

    Saves as <stem>_content_panel.txt (or .json) inside a panel/ folder
    alongside the source PDF.

    Returns the output file path.
    """
    results = extract_all_pages(pdf_path, page_num)
    pdf_p = Path(pdf_path)
    panel_dir = pdf_p.parent / "panel"
    panel_dir.mkdir(exist_ok=True)

    ext = ".json" if as_json else ".txt"
    out_path = panel_dir / f"{pdf_p.stem}_content_panel{ext}"

    if as_json:
        out_path.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
    else:
        out_path.write_text(_build_output(results), encoding="utf-8")

    print(f"Saved content panel → {out_path}")
    return str(out_path)


# ─── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.pdf> [--page N] [--json]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    as_json = "--json" in sys.argv
    page_num = None

    if "--page" in sys.argv:
        try:
            page_num = int(sys.argv[sys.argv.index("--page") + 1])
        except (IndexError, ValueError):
            sys.exit("--page requires a number, e.g. --page 1")

    if not Path(pdf_path).exists():
        sys.exit(f"File not found: {pdf_path}")

    results = extract_all_pages(pdf_path, page_num)

    if as_json:
        print(json.dumps(results, indent=2, default=str))
    else:
        for page_data in results:
            print(f"\n{'═' * 60}")
            print(f"  Page {page_data['page']}")
            print(f"{'═' * 60}")
            if page_data["content"]:
                print_content_tree(page_data["content"], indent=1)
            else:
                print("  (no marked content on this page)")
