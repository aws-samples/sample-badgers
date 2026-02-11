"""
PDF Accessibility Tagger v2 — Unified audit + remediation engine.

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
    PageAudit,
    AccessibilityReport,
    TagRegion,
    VALID_TAGS,
)
from pdf_accessibility_auditor import PDFAccessibilityAuditor

logger = logging.getLogger(__name__)

__version__ = "2.0.0"
__all__ = [
    "PDFAccessibilityTagger",
    "PDFAccessibilityAuditor",
    "AccessibilityReport",
    "TagRegion",
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
    """

    def __init__(self, pdf_path: Union[str, Path]):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.pdf = Pdf.open(self.pdf_path)
        self.fitz_doc = fitz.open(str(self.pdf_path))
        self.regions: Dict[int, List[TagRegion]] = {}
        self.report = AccessibilityReport()

        # Run pre-remediation audit
        self.report.pre_checks = PDFAccessibilityAuditor.audit_pdf(
            self.pdf, self.fitz_doc
        )
        self.report.pre_level = PDFAccessibilityAuditor.compute_level(
            self.report.pre_checks
        )

    # -----------------------------------------------------------------------
    # Region registration
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
    # Page analysis
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
            xobjects = resources.get("/XObject", Dictionary())  # type: ignore[union-attr]
            for xname in xobjects.keys():  # type: ignore[union-attr]
                xobj = xobjects[xname]  # type: ignore[index]
                if hasattr(xobj, "get_object"):
                    xobj = xobj.get_object()  # type: ignore[operator]
                if xobj.get("/Subtype") == Name("/Image"):  # type: ignore[operator]
                    xobj_images += 1
        except Exception:
            pass

        total_images = max(img_count, xobj_images)

        annots = pike_page.get("/Annots")
        annot_count = len(list(annots)) if annots else 0  # type: ignore[call-overload]

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
    # Content stream manipulation
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
    # Invisible text overlay (for image-only pages / elements)
    # -----------------------------------------------------------------------

    def _insert_invisible_text_overlays(
        self, _page_num: int, regions: List[TagRegion], mcid_start: int
    ) -> Tuple[bytes, Dict[int, TagRegion], int]:
        """Generate content stream bytes for invisible text overlays.

        Returns:
            (overlay_content_bytes, mcid_to_region_map, next_available_mcid)
        """
        if not regions:
            return b"", {}, mcid_start

        mcid = mcid_start
        mcid_map: Dict[int, TagRegion] = {}
        ops: List[Tuple[list, Operator]] = []

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
            ops.append(([String(text)], Operator("Tj")))
            ops.append(([], Operator("ET")))

            ops.append(([], Operator("EMC")))
            mcid += 1

        if not ops:
            return b"", mcid_map, mcid

        return pikepdf.unparse_content_stream(ops), mcid_map, mcid

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
            font_dict = Dictionary(
                {
                    "/Type": Name("/Font"),
                    "/Subtype": Name("/Type1"),
                    "/BaseFont": Name("/Helvetica"),
                    "/Encoding": Name("/WinAnsiEncoding"),
                }
            )
            fonts["/F1"] = self.pdf.make_indirect(font_dict)

    # -----------------------------------------------------------------------
    # Structure tree builder
    # -----------------------------------------------------------------------

    def _build_unified_structure_tree(
        self,
        page_mcid_maps: Dict[int, Dict[int, TagRegion]],
    ) -> None:
        """Build a single document-wide structure tree from all page MCID maps."""
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
                annots_list = list(annots)  # type: ignore[call-overload]
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
    # Metadata
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

        for page_num in range(
            len(self.pdf.pages)
        ):  # pylint: disable=consider-using-enumerate
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
    # Context manager
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
# Convenience function
# ---------------------------------------------------------------------------


def tag_pdf(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    title: str = "Accessible Document",
    lang: str = "en-US",
    author: str = "",
    regions: Optional[List[Dict]] = None,
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

    Returns:
        (output_path, AccessibilityReport)
    """
    with PDFAccessibilityTagger(input_path) as tagger:
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
