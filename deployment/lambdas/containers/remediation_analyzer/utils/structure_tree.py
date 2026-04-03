"""Build PDF/UA structure tree from MCID map.

Two modes:
  1. Flat (default): Groups MCIDs by element_id into a flat Document → StructElems.
     Used when no StructureElement tree is available.
  2. Hierarchical: Walks a StructureElement tree from spine_parser and produces
     nested pikepdf StructElems that mirror the tree (Sect → Table → TR → TH/TD etc).
     Used when spine_parser provides a v2.0 tree.

The public API is build_structure_tree() which dispatches to the right mode
based on whether a structure_element_tree argument is provided.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Optional

from pikepdf import Array, Dictionary, Name, String

from utils.pdf_accessibility_models import TagRegion

logger = logging.getLogger(__name__)


def _clean_text_for_pdf(text: str) -> str:
    """Clean text before storing as a PDF String value.

    Handles UTF-16BE encoded strings (BOM + null-interleaved bytes)
    that come from some PDF content streams. Also strips null bytes
    and normalizes whitespace.
    """
    if not text:
        return text

    # Detect UTF-16BE BOM as Unicode codepoints (þÿ = \u00fe\u00ff)
    if text.startswith("\u00fe\u00ff") or text.startswith("\xfe\xff"):
        try:
            raw = text.encode("latin-1")
            text = raw.decode("utf-16-be")
        except (UnicodeDecodeError, UnicodeEncodeError):
            # Fallback: just strip the null bytes
            text = text[2:]  # skip BOM
            text = text.replace("\x00", "")

    # Strip null bytes (common in UTF-16 artifacts)
    if "\x00" in text or "\u0000" in text:
        text = text.replace("\x00", "").replace("\u0000", "")

    # Strip BOM if still present after decoding
    text = text.lstrip("\ufeff\u00fe\u00ff")

    return text.strip()


# Tags that are grouping/structural containers (no MCIDs of their own)
_GROUPING_TAGS = {
    "Document",
    "Part",
    "Art",
    "Sect",
    "Div",
    "BlockQuote",
    "TOC",
    "TOCI",
    "NonStruct",
    "Private",
    "Table",
    "THead",
    "TBody",
    "TFoot",
    "TR",
    "L",
    "LI",
    "Lbl",
    "LBody",
}


def build_structure_tree(
    pdf,
    mcid_map: Dict[int, TagRegion],
    page_num: int = 0,
    structure_element_tree=None,
):
    """Inject a complete StructTreeRoot into the PDF.

    Args:
        pdf: pikepdf.Pdf object
        mcid_map: dict mapping MCID int → TagRegion
        page_num: 0-indexed page number
        structure_element_tree: optional StructureElement root from spine_parser.
            If provided, builds nested StructElems mirroring the tree hierarchy.
            If None, falls back to flat grouping by element_id.

    Returns:
        elem_groups dict for caller diagnostics (flat mode) or
        node count int (hierarchy mode).
    """
    if not mcid_map:
        return {} if structure_element_tree is None else 0

    if structure_element_tree is not None:
        return _build_hierarchical(pdf, mcid_map, page_num, structure_element_tree)
    else:
        return _build_flat(pdf, mcid_map, page_num)


# ============================================================================
# Hierarchical builder — walks StructureElement tree
# ============================================================================


def _build_hierarchical(pdf, mcid_map, page_num, se_tree):
    """Build nested StructElems from a StructureElement tree.

    Walks the tree depth-first. For each node:
      - Grouping nodes (Sect, Table, TR, etc.) → StructElem container with /K = children
      - Leaf content nodes (P, H1, TH, TD, Figure, etc.) → StructElem with /K = MCR(s)
      - Nodes with no matching MCIDs are skipped (avoids empty StructElems)
    """

    # Build page ref lookup — each MCID may be on a different page
    def _page_ref_for(mcid_val):
        pg = mcid_map[mcid_val].page
        return pdf.pages[pg].obj

    # Build reverse lookup: element_id → list of MCIDs
    eid_to_mcids: Dict[str, List[int]] = {}
    for mcid_val in sorted(mcid_map.keys()):
        eid = mcid_map[mcid_val].element_id or f"_anon_{mcid_val}"
        eid_to_mcids.setdefault(eid, []).append(mcid_val)

    # Build enrichment lookup: element_id → first TagRegion (for enrichment data)
    eid_to_region: Dict[str, TagRegion] = {}
    for mcid_val in sorted(mcid_map.keys()):
        region = mcid_map[mcid_val]
        eid = region.element_id or f"_anon_{mcid_val}"
        if eid not in eid_to_region:
            eid_to_region[eid] = region

    # ParentTree: MCID → its direct parent StructElem
    max_mcid = max(mcid_map.keys())
    parent_tree_array: List[Optional[object]] = [None] * (max_mcid + 1)

    # ── Root ──
    struct_root = Dictionary(
        {
            "/Type": Name("/StructTreeRoot"),
            "/RoleMap": Dictionary({"/Artifact": Name("/NonStruct")}),
        }
    )
    struct_root = pdf.make_indirect(struct_root)

    # ── Document ──
    doc_elem = Dictionary(
        {
            "/Type": Name("/StructElem"),
            "/S": Name("/Document"),
            "/P": struct_root,
            "/K": Array([]),
        }
    )
    doc_elem = pdf.make_indirect(doc_elem)
    struct_root["/K"] = doc_elem

    # ── Recursive walk ──
    node_count = [0]  # mutable counter

    def _walk_node(se_node, parent_pdf_elem):
        """Recursively convert a StructureElement into pikepdf StructElems."""
        tag = se_node.tag

        # Skip Document node itself (we already created doc_elem)
        if tag == "Document":
            for child in se_node.children:
                _walk_node(child, parent_pdf_elem)
            return

        # Skip Artifact — handled by BMC in content stream, not in structure tree
        if tag == "Artifact":
            return

        is_grouping = tag in _GROUPING_TAGS

        if is_grouping:
            # Check if this subtree has any MCIDs at all
            if not _subtree_has_mcids(se_node, eid_to_mcids):
                return

            elem = Dictionary(
                {
                    "/Type": Name("/StructElem"),
                    "/S": Name(f"/{tag}"),
                    "/P": parent_pdf_elem,
                    "/K": Array([]),
                }
            )

            # TH scope attribute
            if tag == "TH" and se_node.scope:
                elem["/A"] = Dictionary(
                    {
                        "/O": Name("/Table"),
                        "/Scope": Name(f"/{se_node.scope}"),
                    }
                )

            elem = pdf.make_indirect(elem)
            parent_pdf_elem["/K"].append(elem)
            node_count[0] += 1

            # Apply enrichment data to container elements
            _apply_enrichments(elem, se_node, eid_to_region)

            for child in se_node.children:
                _walk_node(child, elem)

        else:
            # Leaf content node — look up MCIDs
            mcids = eid_to_mcids.get(se_node.id, [])
            if not mcids:
                # No MCIDs matched — skip to avoid empty StructElem
                return

            # Build /K value
            if len(mcids) == 1:
                k_val = Dictionary(
                    {
                        "/Type": Name("/MCR"),
                        "/Pg": _page_ref_for(mcids[0]),
                        "/MCID": mcids[0],
                    }
                )
            else:
                k_val = Array(
                    [
                        Dictionary(
                            {
                                "/Type": Name("/MCR"),
                                "/Pg": _page_ref_for(mv),
                                "/MCID": mv,
                            }
                        )
                        for mv in mcids
                    ]
                )

            elem = Dictionary(
                {
                    "/Type": Name("/StructElem"),
                    "/S": Name(f"/{tag}"),
                    "/P": parent_pdf_elem,
                    "/K": k_val,
                }
            )

            # Figure alt text
            if tag == "Figure" and se_node.alt_text:
                elem["/Alt"] = String(_clean_text_for_pdf(se_node.alt_text))

            # TH scope (TH can also be a leaf with MCIDs)
            if tag == "TH" and se_node.scope:
                elem["/A"] = Dictionary(
                    {
                        "/O": Name("/Table"),
                        "/Scope": Name(f"/{se_node.scope}"),
                    }
                )

            region = mcid_map[mcids[0]]
            if region.text_content:
                cleaned = _clean_text_for_pdf(region.text_content)
                if cleaned:
                    elem["/ActualText"] = String(cleaned)

            elem = pdf.make_indirect(elem)
            parent_pdf_elem["/K"].append(elem)
            node_count[0] += 1

            # Register in ParentTree
            for mv in mcids:
                parent_tree_array[mv] = elem

    # Walk from the tree root
    _walk_node(se_tree, doc_elem)

    # ── ParentTree ──
    # Fill any gaps with doc_elem as fallback (shouldn't happen, but safe)
    for i in range(len(parent_tree_array)):
        if parent_tree_array[i] is None and i in mcid_map:
            parent_tree_array[i] = doc_elem

    content_array = pdf.make_indirect(Array(parent_tree_array))
    nums_array = Array([0, content_array])
    parent_tree = Dictionary({"/Nums": nums_array})

    struct_root["/ParentTree"] = pdf.make_indirect(parent_tree)
    struct_root["/ParentTreeNextKey"] = 1

    # ── Wire into catalog ──
    _wire_catalog(pdf, page_num, struct_root)

    logger.info(
        "Built hierarchical structure tree: %d StructElems from %d MCIDs",
        node_count[0],
        len(mcid_map),
    )

    return node_count[0]


def _apply_enrichments(pdf_elem, se_node, eid_to_region):
    """Apply enrichment data from TagRegion to a container StructElem.

    Sets /T (title), /Alt (description for figures), /ActualText (summary).
    Only applies to container-level elements — leaf elements get raw text only.
    """
    region = eid_to_region.get(se_node.id)
    if region is None:
        return

    # /T — Title
    title = getattr(region, "enrichment_title", None) or getattr(
        region, "figure_title", None
    )
    if title:
        pdf_elem["/T"] = String(_clean_text_for_pdf(title))

    # /Alt — for Figure elements, use enrichment description or existing alt_text
    if se_node.tag == "Figure":
        desc = getattr(region, "enrichment_description", None) or region.alt_text
        if desc:
            pdf_elem["/Alt"] = String(_clean_text_for_pdf(desc))

    # /ActualText — structured summary
    actual = getattr(region, "enrichment_actual_text", None) or getattr(
        region, "actual_text", None
    )
    if actual:
        cleaned = _clean_text_for_pdf(actual)
        if cleaned:
            pdf_elem["/ActualText"] = String(cleaned)


def _subtree_has_mcids(se_node, eid_to_mcids):
    """Check if any node in this subtree has matching MCIDs."""
    if se_node.id in eid_to_mcids:
        return True
    for child in se_node.children:
        if _subtree_has_mcids(child, eid_to_mcids):
            return True
    return False


# ============================================================================
# Flat builder — original behavior, groups MCIDs by element_id
# ============================================================================


def _build_flat(pdf, mcid_map, page_num):
    """Build a flat structure tree grouping MCIDs by element_id.

    This is the original behavior — Document → flat list of StructElems.
    Used when no StructureElement tree is available.
    """

    # Build page ref lookup — each MCID may be on a different page
    def _page_ref_for(mcid_val):
        pg = mcid_map[mcid_val].page
        return pdf.pages[pg].obj

    # ── Root + Document ──
    struct_root = Dictionary(
        {
            "/Type": Name("/StructTreeRoot"),
            "/RoleMap": Dictionary({"/Artifact": Name("/NonStruct")}),
        }
    )
    struct_root = pdf.make_indirect(struct_root)

    doc_elem = Dictionary(
        {
            "/Type": Name("/StructElem"),
            "/S": Name("/Document"),
            "/P": struct_root,
            "/K": Array([]),
        }
    )
    doc_elem = pdf.make_indirect(doc_elem)
    struct_root["/K"] = doc_elem

    # ── Group MCIDs by element_id (preserving order) ──
    elem_groups: OrderedDict[str, List[int]] = OrderedDict()
    for mcid_val in sorted(mcid_map.keys()):
        eid = mcid_map[mcid_val].element_id or f"_anon_{mcid_val}"
        elem_groups.setdefault(eid, []).append(mcid_val)

    # ── Build StructElems + ParentTree array ──
    max_mcid = max(mcid_map.keys())
    parent_tree_array = [None] * (max_mcid + 1)

    for eid, mcid_list in elem_groups.items():
        region = mcid_map[mcid_list[0]]

        if len(mcid_list) == 1:
            k_val = Dictionary(
                {
                    "/Type": Name("/MCR"),
                    "/Pg": _page_ref_for(mcid_list[0]),
                    "/MCID": mcid_list[0],
                }
            )
        else:
            k_val = Array(
                [
                    Dictionary(
                        {"/Type": Name("/MCR"), "/Pg": _page_ref_for(mv), "/MCID": mv}
                    )
                    for mv in mcid_list
                ]
            )

        elem = Dictionary(
            {
                "/Type": Name("/StructElem"),
                "/S": Name(f"/{region.tag}"),
                "/P": doc_elem,
                "/K": k_val,
            }
        )
        if region.tag == "Figure" and region.alt_text:
            elem["/Alt"] = String(_clean_text_for_pdf(region.alt_text))
        if region.text_content:
            cleaned = _clean_text_for_pdf(region.text_content)
            if cleaned:
                elem["/ActualText"] = String(cleaned)

        elem = pdf.make_indirect(elem)
        doc_elem["/K"].append(elem)

        for mv in mcid_list:
            parent_tree_array[mv] = elem

    # ── ParentTree ──
    content_array = pdf.make_indirect(Array(parent_tree_array))
    nums_array = Array([0, content_array])
    parent_tree = Dictionary({"/Nums": nums_array})

    struct_root["/ParentTree"] = pdf.make_indirect(parent_tree)
    struct_root["/ParentTreeNextKey"] = 1

    # ── Wire into catalog ──
    _wire_catalog(pdf, page_num, struct_root)

    return elem_groups


# ============================================================================
# Shared helpers
# ============================================================================


def _wire_catalog(pdf, page_num, struct_root):
    """Wire StructTreeRoot, MarkInfo, Lang, ViewerPreferences into the catalog."""
    # Set StructParents on all pages
    for i in range(len(pdf.pages)):
        pdf.pages[i]["/StructParents"] = i
        pdf.pages[i]["/Tabs"] = Name("/S")
    pdf.Root["/StructTreeRoot"] = struct_root
    pdf.Root["/MarkInfo"] = Dictionary(
        {"/Marked": True, "/UserProperties": False, "/Suspects": False}
    )
    pdf.Root["/Lang"] = String("en-US")
    pdf.Root["/ViewerPreferences"] = Dictionary({"/DisplayDocTitle": True})
