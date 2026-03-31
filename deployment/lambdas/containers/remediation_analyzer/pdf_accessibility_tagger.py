"""
CHECKPOINT_02B_TAGGER
=====================
PDF Accessibility Tagger v3 — Nested structure tree support.

Changes from v2:
  - set_structure_tree(): accepts a pre-built StructureElement hierarchy
  - _build_nested_structure_tree(): produces Sect-nested pikepdf trees
  - _build_unified_structure_tree(): dispatches nested vs inferred-flat
  - Backwards compatible: no structure tree → infers sections from headings

Handles PDFs in any state:
  - Image-only (no text layer): inserts invisible text overlays + structure tree
  - Text-layer present: wraps existing content stream operators in BDC/EMC
  - Mixed: handles both on a per-element basis
  - Already tagged: reports status, optionally re-tags

Always produces:
  1. An AccessibilityReport (audit before + after)
  2. A remediated PDF (even on "fail" — let the downstream validator decide)
"""

import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import pikepdf
from pikepdf import Pdf, Dictionary, Array, Name, String, Operator
import fitz  # PyMuPDF

from pdf_accessibility_models import (
    ComplianceLevel,
    ElementRole,
    PageAudit,
    AccessibilityReport,
    TagRegion,
    StructureElement,
    VALID_TAGS,
    TAG_ROLES,
    get_role,
    is_heading,
    heading_level,
    infer_structure_from_flat_regions,
)
from pdf_accessibility_auditor import PDFAccessibilityAuditor

logger = logging.getLogger(__name__)

__version__ = "3.0.0"
__all__ = [
    "PDFAccessibilityTagger",
    "PDFAccessibilityAuditor",
    "AccessibilityReport",
    "TagRegion",
    "StructureElement",
    "ComplianceLevel",
    "tag_pdf",
]


class PDFAccessibilityTagger:
    """Unified PDF accessibility remediation engine.

    Handles any PDF state:
      - Image-only pages → invisible text overlays + structure tree
      - Text-layer pages → wraps existing content in BDC/EMC
      - Mixed → per-element decision
      - Always builds proper MCID ↔ structure tree linkage

    NEW in v3: accepts a StructureElement tree for nested Sect output.
    Without one, infers hierarchy from heading levels (backwards compat).
    """

    def __init__(self, pdf_path: Union[str, Path]):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.pdf = Pdf.open(self.pdf_path)
        self.fitz_doc = fitz.open(str(self.pdf_path))
        self.regions: Dict[int, List[TagRegion]] = {}
        self.report = AccessibilityReport()

        # NEW: optional pre-built structure tree
        self._structure_tree: Optional[StructureElement] = None

        # Run pre-remediation audit
        self.report.pre_checks = PDFAccessibilityAuditor.audit_pdf(
            self.pdf, self.fitz_doc
        )
        self.report.pre_level = PDFAccessibilityAuditor.compute_level(
            self.report.pre_checks
        )

    # -----------------------------------------------------------------------
    # NEW: Structure tree registration
    # -----------------------------------------------------------------------

    def set_structure_tree(self, tree: StructureElement) -> None:
        """Set a pre-built StructureElement tree for nested PDF/UA output.

        When set, _build_unified_structure_tree will produce a properly nested
        Sect hierarchy instead of a flat list. The tree's leaf elements are
        linked to TagRegions via matching element IDs.

        Args:
            tree: Root StructureElement (tag="Document") with full hierarchy.
        """
        if tree.tag != "Document":
            raise ValueError(
                f"Structure tree root must be <Document>, got <{tree.tag}>"
            )
        self._structure_tree = tree
        logger.info(
            "Structure tree set: %d total elements",
            sum(1 for _ in tree.walk()),
        )

    # -----------------------------------------------------------------------
    # Region registration (unchanged from v2)
    # -----------------------------------------------------------------------

    def add_region(
        self,
        page: int,
        bbox: Tuple[float, float, float, float],
        tag: str,
        alt_text: str = "",
        text_content: str = "",
        order: int = 0,
        element_id: str = "",
        source: str = "",
    ) -> None:
        """Add a tagged region using PDF coordinates."""
        if tag not in VALID_TAGS:
            raise ValueError(f"Invalid tag '{tag}'. Valid: {sorted(VALID_TAGS)}")

        region = TagRegion(
            tag=tag,
            bbox=bbox,
            alt_text=alt_text,
            text_content=text_content,
            order=order,
            page=page,
            element_id=element_id,
            source=source,
        )
        self.regions.setdefault(page, []).append(region)

    def add_region_normalized(
        self,
        page: int,
        bbox_normalized: Tuple[float, float, float, float],
        tag: str,
        alt_text: str = "",
        text_content: str = "",
        order: int = 0,
        element_id: str = "",
        source: str = "",
    ) -> None:
        """Add a tagged region using normalized coordinates (0-1 range).

        Coordinate system: (0,0) = bottom-left, (1,1) = top-right.
        Internally converted to PDF user-space coordinates.
        """
        fitz_page = self.fitz_doc[page]
        width = fitz_page.rect.width
        height = fitz_page.rect.height

        nx0, ny0, nx1, ny1 = bbox_normalized
        x0 = nx0 * width
        x1 = nx1 * width
        y0 = (1 - ny1) * height
        y1 = (1 - ny0) * height

        self.add_region(
            page,
            (x0, y0, x1, y1),
            tag,
            alt_text,
            text_content,
            order,
            element_id,
            source,
        )

    # -----------------------------------------------------------------------
    # Page analysis (unchanged from v2)
    # -----------------------------------------------------------------------

    def _analyze_page(self, page_num: int) -> PageAudit:
        """Analyze a page's current state."""
        fitz_page = self.fitz_doc[page_num]
        pike_page = self.pdf.pages[page_num]

        text_dict = fitz_page.get_text("dict")
        text_blocks = [b for b in text_dict.get("blocks", []) if b["type"] == 0]
        has_text = len(text_blocks) > 0

        image_blocks = [b for b in text_dict.get("blocks", []) if b["type"] == 1]
        img_count = len(image_blocks)

        xobj_images = 0
        try:
            resources = pike_page.get("/Resources", Dictionary())
            xobjects = resources.get("/XObject", Dictionary())
            for xname in xobjects.keys():
                xobj = xobjects[xname]
                if hasattr(xobj, "get_object"):
                    xobj = xobj.get_object()
                if xobj.get("/Subtype") == Name("/Image"):
                    xobj_images += 1
        except Exception:
            pass

        total_images = max(img_count, xobj_images)

        annots = pike_page.get("/Annots")
        annot_count = len(list(annots)) if annots else 0

        has_mc = False
        try:
            instructions = list(pikepdf.parse_content_stream(pike_page))
            for inst in instructions:
                if hasattr(inst, "operator"):
                    op_name = str(inst.operator)
                    if op_name in ("BMC", "BDC"):
                        has_mc = True
                        break
        except Exception:
            pass

        return PageAudit(
            page_num=page_num,
            has_text_layer=has_text,
            text_block_count=len(text_blocks),
            has_images=total_images > 0,
            image_count=total_images,
            has_annotations=annot_count > 0,
            annotation_count=annot_count,
            existing_marked_content=has_mc,
        )

    # -----------------------------------------------------------------------
    # Content stream manipulation (unchanged from v2)
    # -----------------------------------------------------------------------

    def _get_region_for_bbox(
        self, page_num: int, bbox: Tuple[float, float, float, float]
    ) -> Optional[TagRegion]:
        """Find the best-matching region for a given bbox (overlap test)."""
        bx0, by0, bx1, by1 = bbox
        best = None
        best_overlap = 0.0

        for region in self.regions.get(page_num, []):
            rx0, ry0, rx1, ry1 = region.bbox
            ix0 = max(bx0, rx0)
            iy0 = max(by0, ry0)
            ix1 = min(bx1, rx1)
            iy1 = min(by1, ry1)

            if ix0 < ix1 and iy0 < iy1:
                overlap = (ix1 - ix0) * (iy1 - iy0)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = region

        return best

    def _wrap_content_stream(
        self, page_num: int, mcid_start: int = 0
    ) -> Tuple[bytes, Dict[int, TagRegion], int]:
        """Wrap content stream operators in BDC/EMC marked content.

        Returns:
            (new_content_bytes, mcid_to_region_map, next_available_mcid)
        """
        page = self.pdf.pages[page_num]
        fitz_page = self.fitz_doc[page_num]
        _page_height = fitz_page.rect.height

        text_dict = fitz_page.get_text("dict")
        text_blocks_by_position = []
        for block in text_dict.get("blocks", []):
            if block["type"] == 0:
                bbox = block["bbox"]
                pdf_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
                text_blocks_by_position.append(pdf_bbox)

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
        text_block_idx = 0

        figure_regions = sorted(
            [r for r in self.regions.get(page_num, []) if r.tag == "Figure"],
            key=lambda r: r.order,
        )
        figure_do_idx = 0

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

            if op_name == "BT":
                new_ops.append((operands, op))
                continue

            if op_name == "ET":
                if in_marked:
                    new_ops.append(([], Operator("EMC")))
                    in_marked = False
                    current_region = None
                new_ops.append((operands, op))
                continue

            if op_name in ("Tj", "TJ", "'", '"'):
                target_region = None

                if text_block_idx < len(text_blocks_by_position):
                    bbox = text_blocks_by_position[text_block_idx]
                    target_region = self._get_region_for_bbox(page_num, bbox)

                if target_region and target_region != current_region:
                    if in_marked:
                        new_ops.append(([], Operator("EMC")))

                    # Determine the content stream tag — use the region's tag
                    # but map Artifact to BMC (no MCID)
                    if target_region.tag == "Artifact":
                        new_ops.append(([Name("/Artifact")], Operator("BMC")))
                        in_marked = True
                        current_region = target_region
                    else:
                        mcid_map[mcid] = target_region
                        bdc_dict = Dictionary({"/MCID": mcid})
                        new_ops.append(
                            ([Name(f"/{target_region.tag}"), bdc_dict], Operator("BDC"))
                        )
                        mcid += 1
                        in_marked = True
                        current_region = target_region
                elif not target_region and not in_marked:
                    fallback = TagRegion(tag="P", bbox=(0, 0, 0, 0), page=page_num)
                    mcid_map[mcid] = fallback
                    bdc_dict = Dictionary({"/MCID": mcid})
                    new_ops.append(([Name("/P"), bdc_dict], Operator("BDC")))
                    mcid += 1
                    in_marked = True
                    current_region = fallback

                new_ops.append((operands, op))
                text_block_idx += 1
                continue

            if op_name == "Do":
                if in_marked:
                    new_ops.append(([], Operator("EMC")))
                    in_marked = False
                    current_region = None

                if figure_do_idx < len(figure_regions):
                    fig_region = figure_regions[figure_do_idx]
                    mcid_map[mcid] = fig_region
                    bdc_dict = Dictionary({"/MCID": mcid})
                    new_ops.append(([Name("/Figure"), bdc_dict], Operator("BDC")))
                    new_ops.append((operands, op))
                    new_ops.append(([], Operator("EMC")))
                    mcid += 1
                    figure_do_idx += 1
                else:
                    new_ops.append(([Name("/Artifact")], Operator("BMC")))
                    new_ops.append((operands, op))
                    new_ops.append(([], Operator("EMC")))
                continue

            if op_name in ("S", "s", "f", "F", "f*", "B", "B*", "b", "b*"):
                if not in_marked:
                    new_ops.append(([Name("/Artifact")], Operator("BMC")))
                    new_ops.append((operands, op))
                    new_ops.append(([], Operator("EMC")))
                else:
                    new_ops.append((operands, op))
                continue

            new_ops.append((operands, op))

        if in_marked:
            new_ops.append(([], Operator("EMC")))

        return pikepdf.unparse_content_stream(new_ops), mcid_map, mcid

    # -----------------------------------------------------------------------
    # Invisible text overlay (unchanged from v2)
    # -----------------------------------------------------------------------

    def _insert_invisible_text_overlays(
        self, _page_num: int, regions: List[TagRegion], mcid_start: int
    ) -> Tuple[bytes, Dict[int, TagRegion], int]:
        """Generate content stream bytes for invisible text overlays."""
        if not regions:
            return b"", {}, mcid_start

        mcid = mcid_start
        mcid_map: Dict[int, TagRegion] = {}
        ops: List[Tuple[list, Operator]] = []
        raw_segments: List[bytes] = []

        for region in regions:
            text = region.text_content or region.alt_text or ""
            if not text and region.tag == "Figure":
                mcid_map[mcid] = region
                mcid += 1
                continue

            if not text:
                text = f"[{region.tag}]"

            x0, y0, _x1, y1 = region.bbox
            font_size = max(1, min(12, y1 - y0))

            mcid_map[mcid] = region
            bdc_dict = Dictionary({"/MCID": mcid})
            ops.append(([Name(f"/{region.tag}"), bdc_dict], Operator("BDC")))

            ops.append(([], Operator("BT")))
            ops.append(([Name("/F1"), font_size], Operator("Tf")))
            ops.append(([3], Operator("Tr")))
            ops.append(([x0, y0], Operator("Td")))

            utf16_bytes = text.encode("utf-16-be")
            hex_str = utf16_bytes.hex().upper()
            raw_segments.append(pikepdf.unparse_content_stream(ops))
            ops.clear()
            raw_segments.append(f"<{hex_str}> Tj\n".encode("ascii"))

            ops.append(([], Operator("ET")))
            ops.append(([], Operator("EMC")))
            mcid += 1

        if not ops and not raw_segments:
            return b"", mcid_map, mcid

        if ops:
            raw_segments.append(pikepdf.unparse_content_stream(ops))

        return b"".join(raw_segments), mcid_map, mcid

    def _ensure_font_resource(self, page_num: int) -> None:
        """Ensure page has a /F1 font resource for invisible text overlays."""
        page = self.pdf.pages[page_num]
        resources = page.get("/Resources")
        if resources is None:
            resources = Dictionary()
            page["/Resources"] = resources

        fonts = resources.get("/Font")
        if fonts is None:
            fonts = Dictionary()
            resources["/Font"] = fonts

        if "/F1" not in fonts:
            to_unicode_cmap = (
                "/CIDInit /ProcSet findresource begin\n"
                "12 dict begin\n"
                "begincmap\n"
                "/CIDSystemInfo\n"
                "<< /Registry (Adobe)\n"
                "/Ordering (UCS)\n"
                "/Supplement 0\n"
                ">> def\n"
                "/CMapName /Adobe-Identity-UCS def\n"
                "/CMapType 2 def\n"
                "1 begincodespacerange\n"
                "<0000> <FFFF>\n"
                "endcodespacerange\n"
                "1 beginbfrange\n"
                "<0000> <FFFF> <0000>\n"
                "endbfrange\n"
                "endcmap\n"
                "CMapName currentdict /CMap defineresource pop\n"
                "end\n"
                "end\n"
            )
            to_unicode_stream = self.pdf.make_stream(to_unicode_cmap.encode("ascii"))

            cid_font = Dictionary(
                {
                    "/Type": Name("/Font"),
                    "/Subtype": Name("/CIDFontType2"),
                    "/BaseFont": Name("/Arial"),
                    "/CIDSystemInfo": Dictionary(
                        {
                            "/Registry": String("Adobe"),
                            "/Ordering": String("Identity"),
                            "/Supplement": 0,
                        }
                    ),
                    "/DW": 1000,
                }
            )

            font_dict = Dictionary(
                {
                    "/Type": Name("/Font"),
                    "/Subtype": Name("/Type0"),
                    "/BaseFont": Name("/Arial"),
                    "/Encoding": Name("/Identity-H"),
                    "/DescendantFonts": Array([self.pdf.make_indirect(cid_font)]),
                    "/ToUnicode": to_unicode_stream,
                }
            )
            fonts["/F1"] = self.pdf.make_indirect(font_dict)

    # -----------------------------------------------------------------------
    # NEW: Nested structure tree builder
    # -----------------------------------------------------------------------

    def _build_element_to_mcid_index(
        self,
        page_mcid_maps: Dict[int, Dict[int, TagRegion]],
    ) -> Dict[str, Tuple[int, int, TagRegion]]:
        """Build reverse index: element_id → (page_num, mcid, TagRegion).

        This links StructureElement nodes to their content stream MCIDs.
        """
        index: Dict[str, Tuple[int, int, TagRegion]] = {}
        for page_num, mcid_map in page_mcid_maps.items():
            for mcid, region in mcid_map.items():
                if region.element_id:
                    index[region.element_id] = (page_num, mcid, region)
        return index

    def _build_nested_structure_tree(
        self,
        tree: StructureElement,
        page_mcid_maps: Dict[int, Dict[int, TagRegion]],
    ) -> None:
        """Build a nested PDF/UA structure tree from a StructureElement hierarchy.

        Walks the StructureElement tree and creates corresponding pikepdf
        StructElem objects with proper parent-child nesting. Grouping elements
        (Sect, BlockQuote, TOC) become containers. Leaf elements with content
        get MCR references linking them to the content stream.

        Artifacts are excluded from the structure tree (they're marked in
        the content stream with BMC /Artifact, not BDC).
        """
        # Build reverse index: element_id → (page, mcid, region)
        elem_mcid_index = self._build_element_to_mcid_index(page_mcid_maps)

        # RoleMap: map non-standard tags to standard equivalents
        role_map = Dictionary({
            "/Artifact": Name("/NonStruct"),
        })

        struct_root = Dictionary(
            {
                "/Type": Name("/StructTreeRoot"),
                "/RoleMap": role_map,
            }
        )
        struct_root = self.pdf.make_indirect(struct_root)

        # Collect all MCR references per page for ParentTree
        # page_num → list of (mcid, pikepdf_struct_elem)
        page_mcr_refs: Dict[int, List[Tuple[int, object]]] = {}

        def _build_node(
            se: StructureElement, pike_parent: object
        ) -> Optional[object]:
            """Recursively build pikepdf StructElem for a StructureElement node.

            Returns the pikepdf StructElem or None for Artifacts.
            """
            # Skip artifacts — they don't go in the structure tree
            if se.role == ElementRole.ARTIFACT or se.tag == "Artifact":
                return None

            # Skip the Document node itself — we handle it as doc_elem
            if se.tag == "Document":
                for child in se.children:
                    _build_node(child, pike_parent)
                return pike_parent

            # Create the StructElem
            elem_dict = Dictionary(
                {
                    "/Type": Name("/StructElem"),
                    "/S": Name(f"/{se.tag}"),
                    "/P": pike_parent,
                }
            )

            # Set PDF/UA attributes
            if se.tag == "Figure" and se.alt_text:
                elem_dict["/Alt"] = String(se.alt_text)

            if se.tag == "TH" and se.scope:
                attr = Dictionary({
                    "/O": Name("/Table"),
                    "/Scope": Name(f"/{se.scope}"),
                })
                elem_dict["/A"] = attr

            if se.lang:
                elem_dict["/Lang"] = String(se.lang)

            if se.placement == "Block":
                layout_attr = Dictionary({
                    "/O": Name("/Layout"),
                    "/Placement": Name("/Block"),
                })
                # Merge with existing /A if present
                existing_a = elem_dict.get("/A")
                if existing_a:
                    # Store as array of attribute dicts
                    elem_dict["/A"] = Array([existing_a, layout_attr])
                else:
                    elem_dict["/A"] = layout_attr

            # Handle BBox if available
            if se.bbox and se.bbox != (0, 0, 0, 0):
                x0, y0, x1, y1 = se.bbox
                bbox_attr = Dictionary({
                    "/O": Name("/Layout"),
                    "/BBox": Array([x0, y0, x1, y1]),
                })
                existing_a = elem_dict.get("/A")
                if existing_a:
                    if isinstance(existing_a, Array):
                        existing_a.append(bbox_attr)
                    else:
                        elem_dict["/A"] = Array([existing_a, bbox_attr])
                else:
                    elem_dict["/A"] = bbox_attr

            # Determine content: MCR leaf vs. grouping container
            role = get_role(se.tag)
            has_block_children = bool(se.children)
            is_grouping = role in (
                ElementRole.GROUPING,
                ElementRole.LIST_WRAPPER,
                ElementRole.LIST_ITEM,
                ElementRole.LIST_BODY,
                ElementRole.LIST_LABEL,
                ElementRole.TABLE_WRAPPER,
                ElementRole.TABLE_SECTION,
                ElementRole.TABLE_ROW,
                ElementRole.TOC_WRAPPER,
                ElementRole.TOC_ITEM,
            )

            if is_grouping or has_block_children:
                # Container node: /K is an Array of child StructElems
                kids = Array([])
                elem_dict["/K"] = kids
                pike_elem = self.pdf.make_indirect(elem_dict)

                # Add to parent's /K
                _append_to_parent_kids(pike_parent, pike_elem)

                # Recurse into block children
                for child in se.children:
                    _build_node(child, pike_elem)

                # Handle inline children — they become nested StructElems
                # with their own MCRs inside this element
                for inline in se.inline_children:
                    _build_inline(inline, pike_elem)

                return pike_elem
            else:
                # Leaf node: /K is an MCR dict linking to content stream
                mcid_info = elem_mcid_index.get(se.id)

                if mcid_info:
                    page_num, mcid, _region = mcid_info
                    page_ref = self.pdf.pages[page_num].obj

                    mcr = Dictionary(
                        {
                            "/Type": Name("/MCR"),
                            "/Pg": page_ref,
                            "/MCID": mcid,
                        }
                    )

                    # If this leaf has inline children, /K becomes an array
                    # containing the MCR + inline StructElems
                    if se.inline_children:
                        kids = Array([mcr])
                        elem_dict["/K"] = kids
                        pike_elem = self.pdf.make_indirect(elem_dict)
                        _append_to_parent_kids(pike_parent, pike_elem)

                        for inline in se.inline_children:
                            _build_inline(inline, pike_elem)

                        # Track for ParentTree
                        page_mcr_refs.setdefault(page_num, []).append(
                            (mcid, pike_elem)
                        )
                    else:
                        elem_dict["/K"] = mcr
                        pike_elem = self.pdf.make_indirect(elem_dict)
                        _append_to_parent_kids(pike_parent, pike_elem)

                        page_mcr_refs.setdefault(page_num, []).append(
                            (mcid, pike_elem)
                        )

                    return pike_elem
                else:
                    # No MCID found — element exists in logical tree but
                    # wasn't matched to content stream. Include as empty StructElem.
                    logger.debug(
                        "No MCID for element %s (%s) — adding as empty StructElem",
                        se.id, se.tag,
                    )
                    pike_elem = self.pdf.make_indirect(elem_dict)
                    _append_to_parent_kids(pike_parent, pike_elem)
                    return pike_elem

        def _build_inline(
            se: StructureElement, pike_parent: object
        ) -> Optional[object]:
            """Build a StructElem for an inline element (Span, Reference, Link, etc.).

            Inline elements may have their own MCID (if the content stream
            wrapper created one) or share their parent's MCR.
            """
            if se.tag == "Artifact":
                return None

            elem_dict = Dictionary(
                {
                    "/Type": Name("/StructElem"),
                    "/S": Name(f"/{se.tag}"),
                    "/P": pike_parent,
                }
            )

            if se.ref_id:
                # For Reference→Note pairing, we'd ideally set /Ref
                # pointing to the Note StructElem. This requires a second
                # pass to resolve. For now, store as /ActualText hint.
                pass

            # Check if this inline has its own MCID
            mcid_info = elem_mcid_index.get(se.id)
            if mcid_info:
                page_num, mcid, _region = mcid_info
                page_ref = self.pdf.pages[page_num].obj
                elem_dict["/K"] = Dictionary(
                    {
                        "/Type": Name("/MCR"),
                        "/Pg": page_ref,
                        "/MCID": mcid,
                    }
                )
                pike_elem = self.pdf.make_indirect(elem_dict)
                _append_to_parent_kids(pike_parent, pike_elem)
                page_mcr_refs.setdefault(page_num, []).append((mcid, pike_elem))
            else:
                # No own MCID — add as empty StructElem child
                # (screen readers still see the hierarchy)
                pike_elem = self.pdf.make_indirect(elem_dict)
                _append_to_parent_kids(pike_parent, pike_elem)

            return pike_elem

        def _append_to_parent_kids(parent: object, child: object):
            """Append a child to parent's /K array, creating it if needed."""
            kids = parent.get("/K")
            if kids is None:
                parent["/K"] = Array([child])
            elif isinstance(kids, Array):
                kids.append(child)
            else:
                # /K was a single item — convert to array
                parent["/K"] = Array([kids, child])

        # --- Build the tree ---

        # Create Document StructElem
        doc_elem = Dictionary(
            {
                "/Type": Name("/StructElem"),
                "/S": Name("/Document"),
                "/P": struct_root,
                "/K": Array([]),
            }
        )
        doc_elem = self.pdf.make_indirect(doc_elem)
        struct_root["/K"] = doc_elem

        # Walk the StructureElement tree, skipping the Document root
        # (we already created doc_elem for it)
        for child in tree.children:
            _build_node(child, doc_elem)

        # --- Build ParentTree ---
        nums_array = Array([])
        global_struct_parent_id = 0

        for page_num in sorted(set(
            list(page_mcid_maps.keys()) +
            list(page_mcr_refs.keys())
        )):
            # Collect all struct elems that have MCRs on this page
            page_elems = page_mcr_refs.get(page_num, [])

            if page_elems:
                # Sort by MCID for deterministic ordering
                page_elems.sort(key=lambda x: x[0])
                content_array = Array([elem for _, elem in page_elems])
                content_array = self.pdf.make_indirect(content_array)
                nums_array.append(global_struct_parent_id)
                nums_array.append(content_array)

            self.pdf.pages[page_num]["/StructParents"] = global_struct_parent_id

            # Handle annotations (links etc.)
            page = self.pdf.pages[page_num]
            annots = page.get("/Annots")
            if annots:
                link_parent_id = global_struct_parent_id + 1
                annots_list = list(annots)
                for annot_ref in annots_list:
                    try:
                        annot = (
                            annot_ref.get_object()
                            if hasattr(annot_ref, "get_object")
                            else annot_ref
                        )
                        annot["/StructParent"] = link_parent_id

                        link_elem = Dictionary(
                            {
                                "/Type": Name("/StructElem"),
                                "/S": Name("/Link"),
                                "/P": doc_elem,
                                "/K": Dictionary(
                                    {
                                        "/Type": Name("/OBJR"),
                                        "/Obj": annot_ref,
                                        "/Pg": self.pdf.pages[page_num].obj,
                                    }
                                ),
                            }
                        )
                        link_elem = self.pdf.make_indirect(link_elem)
                        _append_to_parent_kids(doc_elem, link_elem)

                        nums_array.append(link_parent_id)
                        nums_array.append(link_elem)
                        link_parent_id += 1
                    except Exception:
                        logger.debug(
                            "Skipping malformed annotation on page %d", page_num
                        )
                        continue

                global_struct_parent_id = link_parent_id
            else:
                global_struct_parent_id += 1

        parent_tree = Dictionary({"/Nums": nums_array})
        struct_root["/ParentTree"] = self.pdf.make_indirect(parent_tree)
        struct_root["/ParentTreeNextKey"] = global_struct_parent_id

        self.pdf.Root["/StructTreeRoot"] = struct_root

        # Count tree stats for logging
        def _count_nodes(node):
            count = 1
            kids = node.get("/K")
            if isinstance(kids, Array):
                for kid in kids:
                    try:
                        k = kid.get_object() if hasattr(kid, "get_object") else kid
                        if isinstance(k, Dictionary) and k.get("/S"):
                            count += _count_nodes(k)
                    except Exception:
                        pass
            elif isinstance(kids, Dictionary) and kids.get("/S"):
                count += _count_nodes(kids)
            return count

        total_nodes = _count_nodes(doc_elem)
        logger.info("Built nested structure tree: %d StructElem nodes", total_nodes)

    # -----------------------------------------------------------------------
    # LEGACY: Flat structure tree builder (preserved for backwards compat)
    # -----------------------------------------------------------------------

    def _build_flat_structure_tree(
        self,
        page_mcid_maps: Dict[int, Dict[int, TagRegion]],
    ) -> None:
        """Build a flat structure tree (v2 behavior).

        Every element is a direct child of Document. No Sect nesting.
        Used only when infer_structure_from_flat_regions would produce
        a worse result than flat (e.g., no headings in the document).
        """
        struct_root = Dictionary(
            {
                "/Type": Name("/StructTreeRoot"),
                "/RoleMap": Dictionary({"/Artifact": Name("/NonStruct")}),
            }
        )
        struct_root = self.pdf.make_indirect(struct_root)

        doc_elem = Dictionary(
            {
                "/Type": Name("/StructElem"),
                "/S": Name("/Document"),
                "/P": struct_root,
                "/K": Array([]),
            }
        )
        doc_elem = self.pdf.make_indirect(doc_elem)
        struct_root["/K"] = doc_elem

        nums_array = Array([])
        global_struct_parent_id = 0

        for page_num in sorted(page_mcid_maps.keys()):
            page_ref = self.pdf.pages[page_num].obj
            mcid_map = page_mcid_maps[page_num]

            page_struct_elems = []

            for mcid in sorted(mcid_map.keys()):
                region = mcid_map[mcid]

                elem = Dictionary(
                    {
                        "/Type": Name("/StructElem"),
                        "/S": Name(f"/{region.tag}"),
                        "/P": doc_elem,
                        "/K": Dictionary(
                            {
                                "/Type": Name("/MCR"),
                                "/Pg": page_ref,
                                "/MCID": mcid,
                            }
                        ),
                    }
                )

                if region.tag == "Figure" and region.alt_text:
                    elem["/Alt"] = String(region.alt_text)

                if region.bbox != (0, 0, 0, 0):
                    x0, y0, x1, y1 = region.bbox
                    elem["/A"] = Dictionary(
                        {"/O": Name("/Layout"), "/BBox": Array([x0, y0, x1, y1])}
                    )

                elem = self.pdf.make_indirect(elem)
                doc_elem["/K"].append(elem)
                page_struct_elems.append(elem)

            if page_struct_elems:
                content_array = Array(page_struct_elems)
                content_array = self.pdf.make_indirect(content_array)
                nums_array.append(global_struct_parent_id)
                nums_array.append(content_array)

            self.pdf.pages[page_num]["/StructParents"] = global_struct_parent_id

            page = self.pdf.pages[page_num]
            annots = page.get("/Annots")
            if annots:
                link_parent_id = global_struct_parent_id + 1
                annots_list = list(annots)
                for annot_ref in annots_list:
                    try:
                        annot = (
                            annot_ref.get_object()
                            if hasattr(annot_ref, "get_object")
                            else annot_ref
                        )
                        annot["/StructParent"] = link_parent_id

                        link_elem = Dictionary(
                            {
                                "/Type": Name("/StructElem"),
                                "/S": Name("/Link"),
                                "/P": doc_elem,
                                "/K": Dictionary(
                                    {
                                        "/Type": Name("/OBJR"),
                                        "/Obj": annot_ref,
                                        "/Pg": page_ref,
                                    }
                                ),
                            }
                        )
                        link_elem = self.pdf.make_indirect(link_elem)
                        doc_elem["/K"].append(link_elem)

                        nums_array.append(link_parent_id)
                        nums_array.append(link_elem)
                        link_parent_id += 1
                    except Exception:
                        logger.debug(
                            "Skipping malformed annotation on page %d", page_num
                        )
                        continue

                global_struct_parent_id = link_parent_id
            else:
                global_struct_parent_id += 1

        parent_tree = Dictionary({"/Nums": nums_array})
        struct_root["/ParentTree"] = self.pdf.make_indirect(parent_tree)
        struct_root["/ParentTreeNextKey"] = global_struct_parent_id

        self.pdf.Root["/StructTreeRoot"] = struct_root

    # -----------------------------------------------------------------------
    # DISPATCHER: chooses nested vs inferred vs flat
    # -----------------------------------------------------------------------

    def _build_unified_structure_tree(
        self,
        page_mcid_maps: Dict[int, Dict[int, TagRegion]],
    ) -> None:
        """Build the structure tree, choosing the best strategy.

        Priority:
          1. If a StructureElement tree was set via set_structure_tree() → nested
          2. If flat regions contain headings → infer sections, then nested
          3. No headings at all → flat (v2 behavior)
        """
        if self._structure_tree:
            logger.info("Building nested structure tree from provided hierarchy")
            self._build_nested_structure_tree(self._structure_tree, page_mcid_maps)
            return

        # Check if we have headings to infer from
        has_headings = False
        for mcid_map in page_mcid_maps.values():
            for region in mcid_map.values():
                if is_heading(region.tag):
                    has_headings = True
                    break
            if has_headings:
                break

        if has_headings:
            logger.info("No explicit structure tree — inferring from heading levels")
            inferred = infer_structure_from_flat_regions(self.regions)
            self._build_nested_structure_tree(inferred, page_mcid_maps)
        else:
            logger.info("No headings found — using flat structure tree (v2 compat)")
            self._build_flat_structure_tree(page_mcid_maps)

    # -----------------------------------------------------------------------
    # Metadata (unchanged from v2)
    # -----------------------------------------------------------------------

    def _set_metadata(self, title: str, lang: str, author: str = "") -> None:
        """Set PDF/UA required metadata."""
        self.pdf.Root["/MarkInfo"] = Dictionary(
            {"/Marked": True, "/UserProperties": False, "/Suspects": False}
        )

        self.pdf.Root["/Lang"] = String(lang)
        self.pdf.Root["/ViewerPreferences"] = Dictionary({"/DisplayDocTitle": True})

        for page in self.pdf.pages:
            page["/Tabs"] = Name("/S")

        if title:
            try:
                with self.pdf.open_metadata() as meta:
                    meta["dc:title"] = title
                    meta["dc:language"] = [lang]
                    if author:
                        meta["dc:creator"] = [author]
                    meta["pdfuaid:part"] = "1"
            except Exception as e:
                logger.warning("Could not set XMP metadata: %s", e)
                self.report.warnings.append(f"XMP metadata write failed: {e}")

    # -----------------------------------------------------------------------
    # Main save / remediation entry point
    # -----------------------------------------------------------------------

    def save(
        self,
        output_path: Union[str, Path],
        title: str = "Accessible Document",
        lang: str = "en-US",
        author: str = "",
    ) -> Tuple[Path, AccessibilityReport]:
        """Apply accessibility tags and save the PDF.

        Returns:
            (output_path, AccessibilityReport)
        """
        output_path = Path(output_path)
        all_mcid_maps: Dict[int, Dict[int, TagRegion]] = {}

        for page_num in range(len(self.pdf.pages)):
            page_audit = self._analyze_page(page_num)

            if page_num not in self.regions:
                fitz_page = self.fitz_doc[page_num]
                self.add_region(
                    page=page_num,
                    bbox=(0, 0, fitz_page.rect.width, fitz_page.rect.height),
                    tag="P",
                    text_content="[Page content]",
                )
                self.report.warnings.append(
                    f"Page {page_num}: No regions defined — added full-page fallback"
                )

            regions = sorted(self.regions[page_num], key=lambda r: r.order)
            page_mcid_map: Dict[int, TagRegion] = {}
            next_mcid = 0

            if page_audit.has_text_layer:
                logger.info(
                    "Page %d: text layer found (%d blocks), wrapping content stream",
                    page_num,
                    page_audit.text_block_count,
                )
                new_content, cs_mcid_map, next_mcid = self._wrap_content_stream(
                    page_num, mcid_start=0
                )

                if new_content:
                    self.pdf.pages[page_num]["/Contents"] = self.pdf.make_stream(
                        new_content
                    )
                    page_mcid_map.update(cs_mcid_map)
                    page_audit.content_stream_marked = len(cs_mcid_map)

                unmatched_text_regions = [
                    r
                    for r in regions
                    if r.tag not in ("Figure",)
                    and r not in cs_mcid_map.values()
                    and r.text_content
                ]
                if unmatched_text_regions:
                    self._ensure_font_resource(page_num)
                    overlay_bytes, overlay_map, next_mcid = (
                        self._insert_invisible_text_overlays(
                            page_num, unmatched_text_regions, next_mcid
                        )
                    )
                    if overlay_bytes:
                        existing = self.pdf.pages[page_num].get("/Contents")
                        if existing:
                            existing_bytes = existing.read_bytes()
                            combined = existing_bytes + b"\n" + overlay_bytes
                            self.pdf.pages[page_num]["/Contents"] = (
                                self.pdf.make_stream(combined)
                            )
                        page_mcid_map.update(overlay_map)
                        page_audit.invisible_text_inserted = len(overlay_map)

            else:
                logger.info(
                    "Page %d: no text layer, inserting invisible text overlays",
                    page_num,
                )

                new_content, cs_mcid_map, next_mcid = self._wrap_content_stream(
                    page_num, mcid_start=0
                )
                if new_content:
                    self.pdf.pages[page_num]["/Contents"] = self.pdf.make_stream(
                        new_content
                    )
                    page_mcid_map.update(cs_mcid_map)

                text_regions = [
                    r for r in regions if r.tag != "Figure" and r.text_content
                ]
                if text_regions:
                    self._ensure_font_resource(page_num)
                    overlay_bytes, overlay_map, next_mcid = (
                        self._insert_invisible_text_overlays(
                            page_num, text_regions, next_mcid
                        )
                    )
                    if overlay_bytes:
                        existing = self.pdf.pages[page_num].get("/Contents")
                        if existing:
                            existing_bytes = existing.read_bytes()
                            combined = existing_bytes + b"\n" + overlay_bytes
                        else:
                            combined = overlay_bytes
                        self.pdf.pages[page_num]["/Contents"] = self.pdf.make_stream(
                            combined
                        )
                        page_mcid_map.update(overlay_map)
                        page_audit.invisible_text_inserted = len(overlay_map)

                unmatched_figures = [
                    r
                    for r in regions
                    if r.tag == "Figure" and r not in cs_mcid_map.values()
                ]
                for fig in unmatched_figures:
                    page_mcid_map[next_mcid] = fig
                    next_mcid += 1

            page_audit.elements_resolved = len(page_mcid_map)
            page_audit.elements_failed = len(regions) - len(page_mcid_map)
            self.report.page_audits.append(page_audit)
            all_mcid_maps[page_num] = page_mcid_map

        # Build the structure tree (nested or flat depending on available data)
        self._build_unified_structure_tree(all_mcid_maps)
        self._set_metadata(title, lang, author)
        self.pdf.save(output_path)

        # Update report summary
        self.report.pages_processed = len(self.pdf.pages)
        self.report.total_elements_tagged = sum(len(m) for m in all_mcid_maps.values())
        self.report.total_figures_with_alt = sum(
            1
            for m in all_mcid_maps.values()
            for r in m.values()
            if r.tag == "Figure" and r.alt_text
        )
        self.report.total_figures_without_alt = sum(
            1
            for m in all_mcid_maps.values()
            for r in m.values()
            if r.tag == "Figure" and not r.alt_text
        )
        self.report.invisible_text_overlays_added = sum(
            pa.invisible_text_inserted for pa in self.report.page_audits
        )

        # Post-remediation audit
        try:
            post_pdf = Pdf.open(output_path)
            post_fitz = fitz.open(str(output_path))
            self.report.post_checks = PDFAccessibilityAuditor.audit_pdf(
                post_pdf, post_fitz
            )
            self.report.post_level = PDFAccessibilityAuditor.compute_level(
                self.report.post_checks
            )
            post_pdf.close()
            post_fitz.close()
        except Exception as e:
            logger.warning("Post-remediation audit failed: %s", e)
            self.report.errors.append(f"Post-audit failed: {e}")
            self.report.post_level = ComplianceLevel.NOT_ASSESSED

        return output_path, self.report

    # -----------------------------------------------------------------------
    # Context manager (unchanged from v2)
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Close file handles."""
        self.pdf.close()
        self.fitz_doc.close()

    def __enter__(self) -> "PDFAccessibilityTagger":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Convenience function (updated to accept structure_tree)
# ---------------------------------------------------------------------------


def tag_pdf(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    title: str = "Accessible Document",
    lang: str = "en-US",
    author: str = "",
    regions: Optional[List[Dict]] = None,
    structure_tree: Optional[StructureElement] = None,
) -> Tuple[Path, AccessibilityReport]:
    """Simple function to add accessibility tags to a PDF.

    Args:
        input_path: Source PDF
        output_path: Where to write tagged PDF
        title: Document title for metadata
        lang: Language code
        author: Author name
        regions: List of dicts with keys:
            - page (int), bbox (tuple or dict), tag (str),
            - alt_text (str), text_content (str), order (int),
            - normalized (bool)
        structure_tree: Optional StructureElement tree for nested output.

    Returns:
        (output_path, AccessibilityReport)
    """
    with PDFAccessibilityTagger(input_path) as tagger:
        if structure_tree:
            tagger.set_structure_tree(structure_tree)

        if regions:
            for i, region in enumerate(regions):
                bbox = region.get("bbox", (0, 0, 100, 100))
                tag = region.get("tag", "P")
                alt_text = region.get("alt_text", "")
                text_content = region.get("text_content", "")
                order = region.get("order", i)
                page = region.get("page", 0)
                normalized = region.get("normalized", False)
                element_id = region.get("element_id", "")
                source = region.get("source", "")

                if isinstance(bbox, dict):
                    bbox = (
                        bbox.get("x0", 0),
                        bbox.get("y0", 0),
                        bbox.get("x1", 1),
                        bbox.get("y1", 1),
                    )

                if normalized:
                    tagger.add_region_normalized(
                        page=page,
                        bbox_normalized=bbox,
                        tag=tag,
                        alt_text=alt_text,
                        text_content=text_content,
                        order=order,
                        element_id=element_id,
                        source=source,
                    )
                else:
                    tagger.add_region(
                        page=page,
                        bbox=bbox,
                        tag=tag,
                        alt_text=alt_text,
                        text_content=text_content,
                        order=order,
                        element_id=element_id,
                        source=source,
                    )

        return tagger.save(output_path, title=title, lang=lang, author=author)
