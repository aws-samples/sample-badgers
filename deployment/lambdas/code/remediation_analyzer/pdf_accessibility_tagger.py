"""
PDF Accessibility Tagger - Adds PDF/UA compliant accessibility tags to PDF documents.
"""

import logging
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import pikepdf
from pikepdf import Pdf, Dictionary, Array, Name, String, Operator
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__all__ = ["PDFAccessibilityTagger", "tag_pdf", "TagRegion"]


@dataclass
class TagRegion:
    """Defines a tagged region in the PDF."""

    tag: str
    bbox: Tuple[float, float, float, float]
    alt_text: str = ""
    order: int = 0
    page: int = 0


class PDFAccessibilityTagger:
    """Add PDF/UA compliant accessibility tags to PDF documents."""

    VALID_TAGS = {
        "Document",
        "Part",
        "Art",
        "Sect",
        "Div",
        "BlockQuote",
        "Caption",
        "TOC",
        "TOCI",
        "Index",
        "NonStruct",
        "Private",
        "P",
        "H",
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "L",
        "LI",
        "Lbl",
        "LBody",
        "Table",
        "TR",
        "TH",
        "TD",
        "THead",
        "TBody",
        "TFoot",
        "Span",
        "Quote",
        "Note",
        "Reference",
        "BibEntry",
        "Code",
        "Link",
        "Annot",
        "Figure",
        "Formula",
        "Form",
    }

    def __init__(self, pdf_path: Union[str, Path]):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.pdf = Pdf.open(self.pdf_path)
        self.fitz_doc = fitz.open(str(self.pdf_path))
        self.regions: Dict[int, List[TagRegion]] = {}

    def add_region(
        self,
        page: int,
        bbox: Tuple[float, float, float, float],
        tag: str,
        alt_text: str = "",
        order: int = 0,
    ) -> None:
        """Add a tagged region to the document."""
        if tag not in self.VALID_TAGS:
            raise ValueError(f"Invalid tag '{tag}'")

        region = TagRegion(
            tag=tag, bbox=bbox, alt_text=alt_text, order=order, page=page
        )

        if page not in self.regions:
            self.regions[page] = []
        self.regions[page].append(region)

    def add_region_normalized(
        self,
        page: int,
        bbox_normalized: Tuple[float, float, float, float],
        tag: str,
        alt_text: str = "",
        order: int = 0,
    ) -> None:
        """Add a tagged region using normalized coordinates (0-1 range)."""
        fitz_page = self.fitz_doc[page]
        width = fitz_page.rect.width
        height = fitz_page.rect.height

        nx0, ny0, nx1, ny1 = bbox_normalized
        x0 = nx0 * width
        x1 = nx1 * width
        y0 = (1 - ny1) * height
        y1 = (1 - ny0) * height

        self.add_region(page, (x0, y0, x1, y1), tag, alt_text, order)

    def _get_tag_for_position(
        self, page: int, bbox: Tuple[float, float, float, float]
    ) -> str:
        bx0, by0, bx1, by1 = bbox

        for region in self.regions.get(page, []):
            rx0, ry0, rx1, ry1 = region.bbox
            if bx0 < rx1 and bx1 > rx0 and by0 < ry1 and by1 > ry0:
                return region.tag

        return "Span"

    def _wrap_content_stream(self, page_num: int) -> Tuple[bytes, Dict[int, str]]:
        """Wrap content stream operators in BDC/EMC marked content."""
        page = self.pdf.pages[page_num]
        fitz_page = self.fitz_doc[page_num]
        page_height = fitz_page.rect.height

        text_dict = fitz_page.get_text("dict")
        text_blocks = []
        for block in text_dict.get("blocks", []):
            if block["type"] == 0:
                bbox = block["bbox"]
                pdf_bbox = (
                    bbox[0],
                    page_height - bbox[3],
                    bbox[2],
                    page_height - bbox[1],
                )
                text_blocks.append({"bbox": pdf_bbox})

        try:
            instructions = list(pikepdf.parse_content_stream(page))
        except Exception as e:
            raise RuntimeError(f"Failed to parse content stream: {e}") from e

        new_ops: List[Tuple[List, Operator]] = []
        mcid_map: Dict[int, str] = {}
        mcid = 0
        in_marked = False
        current_tag: Optional[str] = None

        for instruction in instructions:
            # Handle both ContentStreamInstruction and ContentStreamInlineImage
            if hasattr(instruction, "operator"):
                operands = (
                    list(instruction.operands)
                    if hasattr(instruction, "operands")
                    else []
                )
                op = instruction.operator
            else:
                # Fallback for inline images or other types
                new_ops.append(([], Operator("q")))
                continue

            op_name = str(op)

            if op_name == "BT":
                new_ops.append((operands, op))
                continue

            if op_name == "ET":
                if in_marked:
                    new_ops.append(([], Operator("EMC")))
                    in_marked = False
                    current_tag = None
                new_ops.append((operands, op))
                continue

            if op_name in ("Tj", "TJ", "'", '"'):
                target_tag = "P"
                if text_blocks:
                    bbox = text_blocks[0]["bbox"]
                    target_tag = self._get_tag_for_position(page_num, bbox)

                if target_tag != current_tag:
                    if in_marked:
                        new_ops.append(([], Operator("EMC")))

                    mcid_map[mcid] = target_tag
                    bdc_dict = Dictionary({"/MCID": mcid})
                    new_ops.append(
                        ([Name(f"/{target_tag}"), bdc_dict], Operator("BDC"))
                    )
                    mcid += 1
                    in_marked = True
                    current_tag = target_tag

                new_ops.append((operands, op))
                continue

            if op_name == "Do":
                if in_marked:
                    new_ops.append(([], Operator("EMC")))
                    in_marked = False
                    current_tag = None

                has_figure = any(
                    r.tag == "Figure" for r in self.regions.get(page_num, [])
                )

                if has_figure:
                    mcid_map[mcid] = "Figure"
                    bdc_dict = Dictionary({"/MCID": mcid})
                    new_ops.append(([Name("/Figure"), bdc_dict], Operator("BDC")))
                    new_ops.append((operands, op))
                    new_ops.append(([], Operator("EMC")))
                    mcid += 1
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

        return pikepdf.unparse_content_stream(new_ops), mcid_map

    def _build_structure_tree(
        self, page_num: int, mcid_map: Dict[int, str]
    ) -> Dict[int, Dictionary]:
        """Build the PDF structure tree and tag annotations."""
        page = self.pdf.pages[page_num]
        page_ref = page.obj

        struct_root = Dictionary({"/Type": Name("/StructTreeRoot")})
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

        content_struct_elems = []

        for mcid, tag in sorted(mcid_map.items()):
            elem = Dictionary(
                {
                    "/Type": Name("/StructElem"),
                    "/S": Name(f"/{tag}"),
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

            if tag == "Figure":
                for region in self.regions.get(page_num, []):
                    if region.tag == "Figure" and region.alt_text:
                        elem["/Alt"] = String(region.alt_text)
                        break

            elem = self.pdf.make_indirect(elem)
            doc_elem["/K"].append(elem)
            content_struct_elems.append((mcid, elem))

        link_struct_elems: Dict[int, Dictionary] = {}

        if "/Annots" in page:
            annots_obj = page["/Annots"]
            struct_parent_id = 1

            # Convert to list to iterate safely
            annots_list = list(annots_obj) if annots_obj else []
            for annot_ref in annots_list:
                try:
                    annot = (
                        annot_ref.get_object()
                        if hasattr(annot_ref, "get_object")
                        else annot_ref
                    )
                    annot["/StructParent"] = struct_parent_id

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

                    link_struct_elems[struct_parent_id] = link_elem
                    struct_parent_id += 1
                except Exception:
                    # Skip malformed annotations - partial tagging is better than failure
                    logger.debug(
                        "Skipping malformed link annotation during structure tagging"
                    )
                    continue

        nums_array = Array([])
        content_array = Array([elem for _, elem in content_struct_elems])
        nums_array.append(0)
        nums_array.append(self.pdf.make_indirect(content_array))

        for struct_parent_id, link_elem in sorted(link_struct_elems.items()):
            nums_array.append(struct_parent_id)
            nums_array.append(link_elem)

        parent_tree = Dictionary({"/Nums": nums_array})
        struct_root["/ParentTree"] = self.pdf.make_indirect(parent_tree)
        struct_root["/ParentTreeNextKey"] = 1 + len(link_struct_elems)

        page["/StructParents"] = 0
        self.pdf.Root["/StructTreeRoot"] = struct_root

        return link_struct_elems

    def _set_metadata(self, title: str, lang: str, author: str = "") -> None:
        self.pdf.Root["/MarkInfo"] = Dictionary(
            {
                "/Marked": True,
                "/UserProperties": False,
                "/Suspects": False,
            }
        )

        self.pdf.Root["/Lang"] = String(lang)

        self.pdf.Root["/ViewerPreferences"] = Dictionary({"/DisplayDocTitle": True})

        for page in self.pdf.pages:
            page["/Tabs"] = Name("/S")

        if title:
            with self.pdf.open_metadata() as meta:
                meta["dc:title"] = title
                meta["dc:language"] = [lang]
                if author:
                    meta["dc:creator"] = [author]

    def save(
        self,
        output_path: Union[str, Path],
        title: str = "Accessible Document",
        lang: str = "en-US",
        author: str = "",
    ) -> Path:
        """Apply accessibility tags and save the PDF."""
        output_path = Path(output_path)

        for page_num, _ in enumerate(self.pdf.pages):
            if page_num not in self.regions:
                fitz_page = self.fitz_doc[page_num]
                self.add_region(
                    page=page_num,
                    bbox=(0, 0, fitz_page.rect.width, fitz_page.rect.height),
                    tag="P",
                )

            new_stream, mcid_map = self._wrap_content_stream(page_num)
            self.pdf.pages[page_num]["/Contents"] = self.pdf.make_stream(new_stream)

            if page_num == 0:
                self._build_structure_tree(page_num, mcid_map)

        self._set_metadata(title, lang, author)
        self.pdf.save(output_path)

        return output_path

    def close(self) -> None:
        """Close file handles."""
        self.pdf.close()
        self.fitz_doc.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def tag_pdf(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    title: str = "Accessible Document",
    lang: str = "en-US",
    author: str = "",
    figure_regions: Optional[List[Dict]] = None,
    figure_alt_text: str = "",
) -> Path:
    """Simple function to add accessibility tags to a PDF."""
    with PDFAccessibilityTagger(input_path) as tagger:
        if figure_regions:
            for i, region in enumerate(figure_regions):
                bbox = region.get("bbox", (0, 0, 100, 100))
                alt = region.get("alt_text", figure_alt_text or f"Figure {i+1}")
                normalized = region.get("normalized", False)

                if normalized:
                    tagger.add_region_normalized(
                        page=region.get("page", 0),
                        bbox_normalized=bbox,
                        tag="Figure",
                        alt_text=alt,
                        order=i,
                    )
                else:
                    tagger.add_region(
                        page=region.get("page", 0),
                        bbox=bbox,
                        tag="Figure",
                        alt_text=alt,
                        order=i,
                    )

        result = tagger.save(output_path, title=title, lang=lang, author=author)
        return Path(result) if not isinstance(result, Path) else result
