"""
CHECKPOINT_02A_MODELS
=====================
PDF Accessibility data models and constants.

Enhanced with:
  - StructureElement: hierarchical node for nested PDF/UA structure trees
  - ElementRole: classifies how a tag participates in the tree
  - TAG_ROLES: maps PDF/UA tag names to their ElementRole
  - TagRegion: extended with hierarchy fields (parent_id, section_depth, etc.)

All existing classes preserved unchanged for backwards compatibility.
"""

import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# EXISTING CLASSES (unchanged)
# ============================================================================


class ComplianceLevel(Enum):
    """Overall compliance verdict."""

    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    FAIL = "fail"
    NOT_ASSESSED = "not_assessed"


@dataclass
class CheckResult:
    """Single audit check."""

    name: str
    passed: bool
    severity: str  # "critical", "major", "minor", "info"
    message: str
    details: Optional[str] = None


@dataclass
class PageAudit:
    """Per-page audit details."""

    page_num: int
    has_text_layer: bool
    text_block_count: int
    has_images: bool
    image_count: int
    has_annotations: bool
    annotation_count: int
    existing_marked_content: bool
    elements_resolved: int = 0
    elements_failed: int = 0
    invisible_text_inserted: int = 0
    content_stream_marked: int = 0


@dataclass
class AccessibilityReport:
    """Full accessibility audit report."""

    # Pre-remediation state
    pre_checks: List[CheckResult] = field(default_factory=list)
    pre_level: ComplianceLevel = ComplianceLevel.NOT_ASSESSED

    # Post-remediation state
    post_checks: List[CheckResult] = field(default_factory=list)
    post_level: ComplianceLevel = ComplianceLevel.NOT_ASSESSED

    # Per-page detail
    page_audits: List[PageAudit] = field(default_factory=list)

    # Summary
    pages_processed: int = 0
    total_elements_tagged: int = 0
    total_figures_with_alt: int = 0
    total_figures_without_alt: int = 0
    invisible_text_overlays_added: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "pre_remediation": {
                "compliance_level": self.pre_level.value,
                "checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "severity": c.severity,
                        "message": c.message,
                        "details": c.details,
                    }
                    for c in self.pre_checks
                ],
            },
            "post_remediation": {
                "compliance_level": self.post_level.value,
                "checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "severity": c.severity,
                        "message": c.message,
                        "details": c.details,
                    }
                    for c in self.post_checks
                ],
            },
            "page_audits": [
                {
                    "page": pa.page_num,
                    "has_text_layer": pa.has_text_layer,
                    "text_blocks": pa.text_block_count,
                    "has_images": pa.has_images,
                    "image_count": pa.image_count,
                    "existing_marked_content": pa.existing_marked_content,
                    "elements_resolved": pa.elements_resolved,
                    "elements_failed": pa.elements_failed,
                    "invisible_text_inserted": pa.invisible_text_inserted,
                    "content_stream_marked": pa.content_stream_marked,
                }
                for pa in self.page_audits
            ],
            "summary": {
                "pages_processed": self.pages_processed,
                "total_elements_tagged": self.total_elements_tagged,
                "total_figures_with_alt": self.total_figures_with_alt,
                "total_figures_without_alt": self.total_figures_without_alt,
                "invisible_text_overlays_added": self.invisible_text_overlays_added,
                "warnings": self.warnings,
                "errors": self.errors,
            },
        }


# ============================================================================
# NEW: ElementRole and TAG_ROLES
# ============================================================================


class ElementRole(Enum):
    """How an element participates in the PDF/UA structure tree."""

    GROUPING = "grouping"  # Sect, Document, Part, Div, BlockQuote, TOC
    BLOCK = "block"  # H1-H6, P, Figure, Table, Formula, L, Note, etc.
    INLINE = "inline"  # Span, Reference, Link, Quote, Code (inline)
    LIST_WRAPPER = "list"  # L
    LIST_ITEM = "list_item"  # LI
    LIST_LABEL = "list_label"  # Lbl
    LIST_BODY = "list_body"  # LBody
    TABLE_WRAPPER = "table"  # Table
    TABLE_SECTION = "table_section"  # THead, TBody, TFoot
    TABLE_ROW = "table_row"  # TR
    TABLE_CELL = "table_cell"  # TH, TD
    TOC_WRAPPER = "toc"  # TOC
    TOC_ITEM = "toc_item"  # TOCI
    ARTIFACT = "artifact"  # Not a struct elem — BMC only


TAG_ROLES: Dict[str, ElementRole] = {
    # Grouping
    "Document": ElementRole.GROUPING,
    "Part": ElementRole.GROUPING,
    "Art": ElementRole.GROUPING,
    "Sect": ElementRole.GROUPING,
    "Div": ElementRole.GROUPING,
    "BlockQuote": ElementRole.GROUPING,
    # Block-level
    "H1": ElementRole.BLOCK,
    "H2": ElementRole.BLOCK,
    "H3": ElementRole.BLOCK,
    "H4": ElementRole.BLOCK,
    "H5": ElementRole.BLOCK,
    "H6": ElementRole.BLOCK,
    "P": ElementRole.BLOCK,
    "Figure": ElementRole.BLOCK,
    "Formula": ElementRole.BLOCK,
    "Caption": ElementRole.BLOCK,
    "Note": ElementRole.BLOCK,
    "BibEntry": ElementRole.BLOCK,
    "Code": ElementRole.BLOCK,
    # Inline
    "Span": ElementRole.INLINE,
    "Reference": ElementRole.INLINE,
    "Link": ElementRole.INLINE,
    "Quote": ElementRole.INLINE,
    # List structure
    "L": ElementRole.LIST_WRAPPER,
    "LI": ElementRole.LIST_ITEM,
    "Lbl": ElementRole.LIST_LABEL,
    "LBody": ElementRole.LIST_BODY,
    # Table structure
    "Table": ElementRole.TABLE_WRAPPER,
    "THead": ElementRole.TABLE_SECTION,
    "TBody": ElementRole.TABLE_SECTION,
    "TFoot": ElementRole.TABLE_SECTION,
    "TR": ElementRole.TABLE_ROW,
    "TH": ElementRole.TABLE_CELL,
    "TD": ElementRole.TABLE_CELL,
    # TOC
    "TOC": ElementRole.TOC_WRAPPER,
    "TOCI": ElementRole.TOC_ITEM,
    # Artifact
    "Artifact": ElementRole.ARTIFACT,
}


def get_role(tag: str) -> ElementRole:
    """Get the ElementRole for a PDF/UA tag name."""
    return TAG_ROLES.get(tag, ElementRole.BLOCK)


def is_heading(tag: str) -> bool:
    """Check if a tag is a heading (H1-H6)."""
    return tag in ("H1", "H2", "H3", "H4", "H5", "H6")


def heading_level(tag: str) -> int:
    """Extract heading level number. Returns 0 if not a heading."""
    if is_heading(tag):
        return int(tag[1:])
    return 0


# ============================================================================
# NEW: StructureElement — hierarchical tree node
# ============================================================================


@dataclass
class StructureElement:
    """A single node in the PDF/UA structure tree.

    This is the LOGICAL model for the document's semantic hierarchy.
    It maps to pikepdf StructElem objects during tree building.
    TagRegion remains the PHYSICAL model for bbox/content stream work.

    The two are linked via matching `id` (StructureElement.id == TagRegion.element_id).
    """

    # Identity
    id: str  # e.g. "elem_001", "sect_intro"
    tag: str  # PDF/UA tag name
    role: ElementRole = ElementRole.BLOCK  # Derived from tag via get_role()

    # Tree position
    parent_id: Optional[str] = None  # ID of parent element
    children: List["StructureElement"] = field(default_factory=list)
    order: int = 0  # Reading order (global)
    depth: int = 0  # Nesting depth (0 = Document)

    # Content
    text: str = ""  # Verbatim text (for text elements)
    alt_text: str = ""  # Alt text (for Figure)

    # Page and position (links to TagRegion for bbox resolution)
    page: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None
    mcid: Optional[int] = None  # Assigned during tree build

    # PDF/UA attributes
    scope: Optional[str] = None  # "Column" or "Row" for TH
    ref_id: Optional[str] = None  # For Reference→Note pairing
    placement: Optional[str] = None  # "Block" or "Inline"
    lang: Optional[str] = None  # Language override

    # Inline children (elements within this element's text run)
    inline_children: List["StructureElement"] = field(default_factory=list)

    # Source tracking
    source_analyzer: str = ""
    confidence: str = "high"

    def add_child(self, child: "StructureElement") -> "StructureElement":
        """Add a block-level child, setting parent reference and depth."""
        child.parent_id = self.id
        child.depth = self.depth + 1
        self.children.append(child)
        return child

    def add_inline(self, child: "StructureElement") -> "StructureElement":
        """Add an inline child within this element's text."""
        child.parent_id = self.id
        child.depth = self.depth + 1
        child.placement = "Inline"
        self.inline_children.append(child)
        return child

    def walk(self):
        """Depth-first traversal yielding every node."""
        yield self
        for child in self.children:
            yield from child.walk()
        for inline in self.inline_children:
            yield from inline.walk()

    def walk_block_only(self):
        """Depth-first traversal yielding only block-level nodes (no inlines)."""
        yield self
        for child in self.children:
            yield from child.walk_block_only()

    def find_by_id(self, target_id: str) -> Optional["StructureElement"]:
        """Find a descendant by ID."""
        for elem in self.walk():
            if elem.id == target_id:
                return elem
        return None

    def leaf_elements(self):
        """Yield only leaf elements (no children or inline_children with content)."""
        for elem in self.walk():
            if not elem.children and elem.tag != "Sect":
                yield elem

    def collect_by_page(self) -> Dict[int, List["StructureElement"]]:
        """Group all leaf elements by page number."""
        by_page: Dict[int, List["StructureElement"]] = {}
        for elem in self.leaf_elements():
            by_page.setdefault(elem.page, []).append(elem)
        return by_page

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict (for debugging/logging)."""
        result = {
            "id": self.id,
            "tag": self.tag,
            "order": self.order,
            "depth": self.depth,
            "page": self.page,
        }
        if self.text:
            result["text"] = self.text[:100]
        if self.alt_text:
            result["alt_text"] = self.alt_text[:100]
        if self.scope:
            result["scope"] = self.scope
        if self.ref_id:
            result["ref_id"] = self.ref_id
        if self.placement:
            result["placement"] = self.placement
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        if self.inline_children:
            result["inline_children"] = [c.to_dict() for c in self.inline_children]
        return result

    def __repr__(self):
        indent = "  " * self.depth
        text_preview = f' "{self.text[:40]}..."' if self.text else ""
        return f"{indent}<{self.tag} id={self.id} order={self.order}{text_preview}>"


# ============================================================================
# NEW: Hierarchy inference — build StructureElement tree from flat regions
# ============================================================================


def infer_structure_from_flat_regions(
    regions_by_page: Dict[int, List["TagRegion"]],
) -> StructureElement:
    """Infer a hierarchical StructureElement tree from flat TagRegions.

    Uses heading levels to create Sect wrappers:
      - H1 opens a depth-1 Sect
      - H2 opens a depth-2 Sect (child of current depth-1 Sect)
      - H3 opens a depth-3 Sect (child of current depth-2 Sect)
      - Non-heading elements go into the innermost open Sect

    This provides a reasonable structure tree even when no explicit hierarchy
    data was provided (backwards compatibility path).
    """
    doc = StructureElement(id="doc", tag="Document", role=ElementRole.GROUPING, depth=0)
    root_sect = doc.add_child(
        StructureElement(id="sect_root", tag="Sect", role=ElementRole.GROUPING)
    )

    # Stack of open sections: [(level, StructureElement)]
    # level 0 = root_sect (always present)
    sect_stack: List[Tuple[int, StructureElement]] = [(0, root_sect)]

    def current_sect() -> StructureElement:
        return sect_stack[-1][1]

    def open_section_for_heading(hlevel: int, heading_elem: StructureElement):
        """Close sections deeper than this heading, open a new one."""
        # Pop sections that are at the same or deeper level
        while len(sect_stack) > 1 and sect_stack[-1][0] >= hlevel:
            sect_stack.pop()

        # Create new Sect and push it
        new_sect = current_sect().add_child(
            StructureElement(
                id=f"sect_{heading_elem.id}",
                tag="Sect",
                role=ElementRole.GROUPING,
            )
        )
        sect_stack.append((hlevel, new_sect))

        # Add the heading as first child of the new section
        new_sect.add_child(heading_elem)

    global_order = 0
    for page_num in sorted(regions_by_page.keys()):
        regions = sorted(regions_by_page[page_num], key=lambda r: r.order)

        for region in regions:
            global_order += 1
            tag = _map_region_tag(region.tag)
            role = get_role(tag)

            elem = StructureElement(
                id=region.element_id or f"auto_{global_order:04d}",
                tag=tag,
                role=role,
                order=global_order,
                page=page_num,
                text=region.text_content,
                alt_text=region.alt_text,
            )

            if role == ElementRole.ARTIFACT:
                # Artifacts go at root level as markers (filtered during tree build)
                root_sect.add_child(elem)
            elif is_heading(tag):
                hlevel = heading_level(tag)
                open_section_for_heading(hlevel, elem)
            else:
                current_sect().add_child(elem)

    return doc


def _map_region_tag(tag: str) -> str:
    """Map TagRegion tag names to canonical PDF/UA tag names."""
    mapping = {
        "h1": "H1",
        "h2": "H2",
        "h3": "H3",
        "h4": "H4",
        "h5": "H5",
        "h6": "H6",
        "heading": "H1",
        "paragraph": "P",
        "image": "Figure",
        "figure": "Figure",
        "table": "Table",
        "list": "L",
        "list_item": "LI",
        "footer": "Artifact",
        "header": "Artifact",
        "caption": "Caption",
        "blockquote": "BlockQuote",
        "code": "Code",
        "formula": "Formula",
    }
    return mapping.get(tag, tag)


# ============================================================================
# EXTENDED: TagRegion with hierarchy fields
# ============================================================================


@dataclass
class TagRegion:
    """Defines a tagged region in the PDF.

    Extended with hierarchy fields for structure tree building.
    All new fields have defaults — fully backwards compatible.
    """

    # Original fields (unchanged)
    tag: str
    bbox: Tuple[float, float, float, float]  # PDF coords (x0, y0, x1, y1)
    alt_text: str = ""
    text_content: str = ""
    order: int = 0
    page: int = 0
    element_id: str = ""
    source: str = ""  # "pymupdf_text_search", "vision_model", "correlation", etc.

    # NEW: hierarchy fields for nested structure tree building
    parent_id: Optional[str] = None  # ID of parent element (for nesting)
    section_depth: Optional[int] = None  # Section nesting depth (1=top, 2=sub, etc.)
    scope: Optional[str] = None  # "Column" or "Row" for TH elements
    ref_id: Optional[str] = None  # For Reference→Note pairing
    placement: Optional[str] = None  # "Block" or "Inline"
    heading_level: Optional[int] = None  # 1-6 for heading elements
    inline_of: Optional[str] = None  # element_id of parent if this is inline

    # NEW: enriched figure fields for richer accessibility tag properties
    figure_title: Optional[str] = None  # /T (Title) — e.g. chart title from enrichments
    actual_text: Optional[str] = None  # /ActualText — rich description from enrichments
    language: Optional[str] = (
        None  # /Lang — per-element language (e.g. "en-US", "zh-CN")
    )

    # NEW: enrichment data from correlation analyzer
    enrichment_title: Optional[str] = None  # /T on container StructElem
    enrichment_description: Optional[str] = (
        None  # extends /Alt or standalone description
    )
    enrichment_actual_text: Optional[str] = None  # /ActualText structured summary
    enrichment_tags: Optional[List[str]] = None  # semantic tags (keyword_topic)
    enrichment_related: Optional[List[Dict]] = (
        None  # [{element_id, relationship, evidence}]
    )


# ============================================================================
# VALID_TAGS — extended with full PDF/UA set
# ============================================================================

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
    # Artifact is NOT a valid structure tag — it's marked content only.
    # We keep it here so validation doesn't reject it at registration time;
    # it gets filtered during tree building.
    "Artifact",
}
