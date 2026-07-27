#!/usr/bin/env python3
"""
PDF Tag Panel Viewer
Replicates the tree view from Adobe Acrobat's Tags panel.
Walks the StructTreeRoot and displays the tag hierarchy with attributes.

Usage:
    python pdf_tag_panel.py document.pdf
    python pdf_tag_panel.py document.pdf --page 1
    python pdf_tag_panel.py document.pdf --json
"""

import sys
import json
from pathlib import Path

try:
    import pikepdf
except ImportError:
    sys.exit("pikepdf not installed. Run:  pip install pikepdf")


def _resolve(obj):
    """Identity pass-through. pikepdf 10.x auto-dereferences indirect
    objects on key/attribute access, so no manual resolution needed."""
    return obj


def _str_val(obj):
    """Safely extract a string from a pikepdf object."""
    obj = _resolve(obj)
    if isinstance(obj, pikepdf.String):
        raw = bytes(obj)
        # Detect UTF-16BE BOM and decode properly
        if raw[:2] == b"\xfe\xff":
            try:
                return raw.decode("utf-16-be")
            except UnicodeDecodeError:
                return raw[2:].replace(b"\x00", b"").decode("latin-1", errors="replace")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="replace")
    if isinstance(obj, pikepdf.Name):
        return str(obj).lstrip("/")
    if isinstance(obj, (int, float)):
        return str(obj)
    try:
        return str(obj)
    except Exception:
        return "?"


def _extract_attrs(elem):
    """Pull useful attributes from a StructElem dict."""
    attrs = {}
    for key in ("/Alt", "/ActualText", "/T", "/Lang", "/ID"):
        if key in elem:
            val = _str_val(elem[key])
            if val:
                attrs[key.lstrip("/")] = val

    # /A — attribute dict(s) for table scope, bbox, etc.
    if "/A" in elem:
        a = _resolve(elem["/A"])
        if isinstance(a, pikepdf.Dictionary):
            _collect_attr_dict(a, attrs)
        elif isinstance(a, pikepdf.Array):
            for item in a:
                item = _resolve(item)
                if isinstance(item, pikepdf.Dictionary):
                    _collect_attr_dict(item, attrs)
    return attrs


def _collect_attr_dict(d, attrs):
    """Extract interesting keys from an /A attribute dictionary."""
    for k in (
        "/Scope",
        "/ColSpan",
        "/RowSpan",
        "/Headers",
        "/Summary",
        "/Width",
        "/Height",
        "/BBox",
        "/Placement",
        "/ListNumbering",
    ):
        if k in d:
            attrs[k.lstrip("/")] = _str_val(d[k])


def _mcids_from_k(k_val, page_filter=None):
    """Extract MCID integers from a /K value (int, MCR dict, or array)."""
    mcids = []
    k_val = _resolve(k_val)

    if isinstance(k_val, int):
        mcids.append(int(k_val))
    elif isinstance(k_val, pikepdf.Object) and not isinstance(
        k_val, (pikepdf.Array, pikepdf.Dictionary)
    ):
        try:
            mcids.append(int(k_val))
        except (ValueError, TypeError):
            pass
    elif isinstance(k_val, pikepdf.Dictionary):
        if _str_val(k_val.get("/Type", "")) == "MCR":
            mcid = k_val.get("/MCID")
            if mcid is not None:
                mcids.append(int(mcid))
    elif isinstance(k_val, pikepdf.Array):
        for item in k_val:
            mcids.extend(_mcids_from_k(item, page_filter))
    return mcids


def _walk_struct_elem(elem, page_filter=None):
    """Recursively walk a StructElem and build a dict tree."""
    elem = _resolve(elem)
    if not isinstance(elem, pikepdf.Dictionary):
        return None

    tag_name = _str_val(elem.get("/S", "?"))
    attrs = _extract_attrs(elem)
    mcids = []
    children = []

    k = elem.get("/K")
    if k is not None:
        k = _resolve(k)
        if isinstance(k, pikepdf.Array):
            for item in k:
                item = _resolve(item)
                if isinstance(item, pikepdf.Dictionary) and "/S" in item:
                    child = _walk_struct_elem(item, page_filter)
                    if child:
                        children.append(child)
                else:
                    mcids.extend(_mcids_from_k(item, page_filter))
        elif isinstance(k, pikepdf.Dictionary) and "/S" in k:
            child = _walk_struct_elem(k, page_filter)
            if child:
                children.append(child)
        else:
            mcids.extend(_mcids_from_k(k, page_filter))

    node = {"tag": tag_name}
    if attrs:
        node["attrs"] = attrs
    if mcids:
        node["mcids"] = mcids
    if children:
        node["children"] = children
    return node


def extract_tag_tree(pdf_path, page_num=None):
    """Extract the tag hierarchy from a PDF's StructTreeRoot.

    Returns a dict with 'root' (the tree) and 'role_map' (if present).
    """
    pdf = pikepdf.open(pdf_path)

    if "/StructTreeRoot" not in pdf.Root:
        pdf.close()
        return {
            "root": None,
            "role_map": {},
            "error": "No StructTreeRoot found — PDF is not tagged.",
        }

    struct_root = _resolve(pdf.Root["/StructTreeRoot"])

    # Role map
    role_map = {}
    if "/RoleMap" in struct_root:
        rm = _resolve(struct_root["/RoleMap"])
        if isinstance(rm, pikepdf.Dictionary):
            for k, v in rm.items():
                role_map[str(k).lstrip("/")] = str(v).lstrip("/")

    # Walk from /K (the top-level StructElem, usually Document)
    k = struct_root.get("/K")
    if k is None:
        pdf.close()
        return {
            "root": None,
            "role_map": role_map,
            "error": "StructTreeRoot has no /K entry.",
        }

    k = _resolve(k)
    if isinstance(k, pikepdf.Array):
        # Multiple top-level elements — wrap in a synthetic Document node
        children = []
        for item in k:
            child = _walk_struct_elem(item, page_num)
            if child:
                children.append(child)
        tree = {"tag": "Document", "children": children}
    else:
        tree = _walk_struct_elem(k, page_num)

    pdf.close()
    return {"root": tree, "role_map": role_map}


# ── Pretty printer ──────────────────────────────────────────────────────

_TAG_ICONS = {
    "Document": "📄",
    "Part": "📂",
    "Sect": "📁",
    "Div": "📁",
    "Art": "📰",
    "BlockQuote": "💬",
    "H1": "🔤",
    "H2": "🔤",
    "H3": "🔤",
    "H4": "🔤",
    "H5": "🔤",
    "H6": "🔤",
    "H": "🔤",
    "P": "¶ ",
    "Span": "🏷 ",
    "Link": "🔗",
    "Table": "📊",
    "THead": "📋",
    "TBody": "📋",
    "TFoot": "📋",
    "TR": "➡ ",
    "TH": "🏷 ",
    "TD": "📝",
    "L": "📋",
    "LI": "• ",
    "Lbl": "🔢",
    "LBody": "📝",
    "Figure": "🖼 ",
    "Formula": "🔣",
    "Form": "📝",
    "TOC": "📑",
    "TOCI": "📌",
    "Caption": "💬",
    "Note": "📝",
    "Reference": "📎",
    "NonStruct": "⬜",
    "Private": "🔒",
}


def print_tag_tree(node, indent=0):
    """Pretty-print the tag tree like Adobe's Tags panel."""
    if node is None:
        print("  (no structure tree)")
        return

    prefix = "  " * indent
    tag = node["tag"]
    icon = _TAG_ICONS.get(tag, "🏷 ")

    # Build attribute summary
    parts = []
    attrs = node.get("attrs", {})
    if "Alt" in attrs:
        alt = attrs["Alt"]
        if len(alt) > 50:
            alt = alt[:47] + "..."
        parts.append(f'Alt="{alt}"')
    if "T" in attrs:
        parts.append(f'T="{attrs["T"]}"')
    if "Scope" in attrs:
        parts.append(f"Scope={attrs['Scope']}")
    if "ColSpan" in attrs:
        parts.append(f"ColSpan={attrs['ColSpan']}")
    if "RowSpan" in attrs:
        parts.append(f"RowSpan={attrs['RowSpan']}")
    if "ActualText" in attrs:
        at = attrs["ActualText"]
        if len(at) > 40:
            at = at[:37] + "..."
        parts.append(f'ActualText="{at}"')

    mcids = node.get("mcids", [])
    if mcids:
        if len(mcids) <= 3:
            parts.append(f"MCIDs={mcids}")
        else:
            parts.append(f"MCIDs=[{mcids[0]}..{mcids[-1]}] ({len(mcids)})")

    attr_str = f"  ({', '.join(parts)})" if parts else ""
    print(f"{prefix}{icon}<{tag}>{attr_str}")

    for child in node.get("children", []):
        print_tag_tree(child, indent + 1)


def _render_tag_tree(node, lines, indent=0):
    """Render the tag tree into a list of strings (no printing)."""
    if node is None:
        lines.append("  (no structure tree)")
        return

    prefix = "  " * indent
    tag = node["tag"]
    icon = _TAG_ICONS.get(tag, "🏷 ")

    parts = []
    attrs = node.get("attrs", {})
    if "Alt" in attrs:
        alt = attrs["Alt"]
        if len(alt) > 50:
            alt = alt[:47] + "..."
        parts.append(f'Alt="{alt}"')
    if "T" in attrs:
        parts.append(f'T="{attrs["T"]}"')
    if "Scope" in attrs:
        parts.append(f"Scope={attrs['Scope']}")
    if "ColSpan" in attrs:
        parts.append(f"ColSpan={attrs['ColSpan']}")
    if "RowSpan" in attrs:
        parts.append(f"RowSpan={attrs['RowSpan']}")
    if "ActualText" in attrs:
        at = attrs["ActualText"]
        if len(at) > 40:
            at = at[:37] + "..."
        parts.append(f'ActualText="{at}"')

    mcids = node.get("mcids", [])
    if mcids:
        if len(mcids) <= 3:
            parts.append(f"MCIDs={mcids}")
        else:
            parts.append(f"MCIDs=[{mcids[0]}..{mcids[-1]}] ({len(mcids)})")

    attr_str = f"  ({', '.join(parts)})" if parts else ""
    lines.append(f"{prefix}{icon}<{tag}>{attr_str}")

    for child in node.get("children", []):
        _render_tag_tree(child, lines, indent + 1)


def _build_output(result):
    """Build the full text output for a tag tree result."""
    lines = []
    if result.get("error"):
        lines.append(f"⚠️  {result['error']}")
        return "\n".join(lines)

    if result["role_map"]:
        lines.append(f"Role Map: {result['role_map']}")
        lines.append("")

    lines.append("═" * 60)
    lines.append("  Tag Hierarchy (StructTreeRoot)")
    lines.append("═" * 60)
    _render_tag_tree(result["root"], lines, indent=1)
    return "\n".join(lines)


def view(pdf_path, page_num=None, as_json=False):
    """Convenience entry point for notebook use.

    Returns the result dict. Prints tree or JSON to stdout.
    """
    result = extract_tag_tree(pdf_path, page_num)

    if as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(_build_output(result))

    return result


def save(pdf_path, page_num=None, as_json=False):
    """Extract the tag hierarchy and save to a panel/ directory next to the PDF.

    Saves as <stem>_tag_panel.txt (or .json) inside a panel/ folder
    alongside the source PDF.

    Returns the output file path.
    """
    result = extract_tag_tree(pdf_path, page_num)
    pdf_p = Path(pdf_path)
    panel_dir = pdf_p.parent / "panel"
    panel_dir.mkdir(exist_ok=True)

    ext = ".json" if as_json else ".txt"
    out_path = panel_dir / f"{pdf_p.stem}_tag_panel{ext}"

    if as_json:
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    else:
        out_path.write_text(_build_output(result), encoding="utf-8")

    print(f"Saved tag panel → {out_path}")
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

    view(pdf_path, page_num, as_json)
