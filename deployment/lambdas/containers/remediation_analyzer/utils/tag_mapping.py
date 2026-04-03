"""Tag name mapping from correlation XML types to PDF structure tags.

Covers the full PDF/UA tag set plus lowercase/natural-language variants
that appear in correlation XML and vision model outputs.
"""

TAG_MAP = {
    # ── Headings ──
    "H1": "H1",
    "H2": "H2",
    "H3": "H3",
    "H4": "H4",
    "H5": "H5",
    "H6": "H6",
    "h1": "H1",
    "h2": "H2",
    "h3": "H3",
    "h4": "H4",
    "h5": "H5",
    "h6": "H6",
    "heading": "H1",
    # ── Block-level text ──
    "P": "P",
    "paragraph": "P",
    "Caption": "Caption",
    "caption": "Caption",
    "BlockQuote": "BlockQuote",
    "blockquote": "BlockQuote",
    "Code": "Code",
    "code": "Code",
    "code_block": "Code",
    "Formula": "Formula",
    "formula": "Formula",
    "Note": "Note",
    "note": "Note",
    "footnote": "Note",
    "BibEntry": "BibEntry",
    "bibentry": "BibEntry",
    "bibliography": "BibEntry",
    "Answer": "P",  # handwritten math answer → P
    # ── Figures / images ──
    "Figure": "Figure",
    "figure": "Figure",
    "image": "Figure",
    # ── Inline elements ──
    "Span": "Span",
    "span": "Span",
    "Reference": "Reference",
    "reference": "Reference",
    "Link": "Link",
    "link": "Link",
    "Quote": "Quote",
    "quote": "Quote",
    # ── Table structure ──
    "Table": "Table",
    "table": "Table",
    "THead": "THead",
    "thead": "THead",
    "TBody": "TBody",
    "tbody": "TBody",
    "TFoot": "TFoot",
    "tfoot": "TFoot",
    "TR": "TR",
    "tr": "TR",
    "TH": "TH",
    "th": "TH",
    "TD": "TD",
    "td": "TD",
    # ── List structure ──
    "L": "L",
    "list": "L",
    "LI": "LI",
    "list_item": "LI",
    "Lbl": "Lbl",
    "label": "Lbl",
    "LBody": "LBody",
    # ── TOC ──
    "TOC": "TOC",
    "toc": "TOC",
    "TOCI": "TOCI",
    "toci": "TOCI",
    # ── Grouping ──
    "Document": "Document",
    "Part": "Part",
    "Art": "Art",
    "Sect": "Sect",
    "Div": "Div",
    "NonStruct": "NonStruct",
    # ── Annotations ──
    "Annot": "Annot",
    "Form": "Form",
    # ── Artifacts (excluded from structure tree) ──
    "Artifact": "Artifact",
    "artifact": "Artifact",
    "header": "Artifact",
    "footer": "Artifact",
    "page_number": "Artifact",
    "watermark": "Artifact",
}


def map_tag(tag_type: str) -> str:
    """Map a correlation element type to a valid PDF structure tag."""
    return TAG_MAP.get(tag_type, "P")
