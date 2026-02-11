"""PDF Accessibility data models and constants."""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


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


@dataclass
class TagRegion:
    """Defines a tagged region in the PDF."""

    tag: str
    bbox: Tuple[float, float, float, float]  # PDF coords (x0, y0, x1, y1)
    alt_text: str = ""
    text_content: str = ""  # For invisible text overlay when no text layer
    order: int = 0
    page: int = 0
    element_id: str = ""
    source: str = ""  # "pymupdf_text_search", "vision_model", "correlation", etc.


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
