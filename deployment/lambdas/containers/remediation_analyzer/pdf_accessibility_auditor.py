"""PDF Accessibility Auditor — pre/post flight compliance checks."""

import logging
from typing import List, Tuple

from pikepdf import Pdf, Dictionary, Array, Name
import fitz  # PyMuPDF

from pdf_accessibility_models import (
    CheckResult,
    ComplianceLevel,
)

logger = logging.getLogger(__name__)


class PDFAccessibilityAuditor:
    """Audit a PDF for accessibility compliance indicators."""

    @staticmethod
    def audit_pdf(pdf: Pdf, fitz_doc: fitz.Document) -> List[CheckResult]:
        """Run all checks against an open PDF. Returns list of CheckResults."""
        checks = []

        # 1. MarkInfo
        mark_info = pdf.Root.get("/MarkInfo")
        if mark_info and mark_info.get("/Marked"):
            checks.append(
                CheckResult(
                    "mark_info",
                    True,
                    "critical",
                    "MarkInfo dictionary present with /Marked = true",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "mark_info",
                    False,
                    "critical",
                    "Missing or incomplete MarkInfo dictionary",
                    "PDF/UA requires /MarkInfo with /Marked true",
                )
            )

        # 2. Structure tree root
        struct_root = pdf.Root.get("/StructTreeRoot")
        if struct_root:
            doc_elem = struct_root.get("/K")
            child_count = 0
            if doc_elem:
                kids = doc_elem.get("/K")
                if isinstance(kids, Array):
                    child_count = len(kids)
                elif kids is not None:
                    child_count = 1
            checks.append(
                CheckResult(
                    "structure_tree",
                    True,
                    "critical",
                    f"Structure tree present with {child_count} child element(s)",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "structure_tree",
                    False,
                    "critical",
                    "No structure tree root found",
                    "PDF/UA requires a complete structure tree",
                )
            )

        # 3. Language
        lang = pdf.Root.get("/Lang")
        if lang:
            checks.append(
                CheckResult(
                    "language",
                    True,
                    "critical",
                    f"Document language set: {str(lang)}",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "language",
                    False,
                    "critical",
                    "No document language specified",
                    "PDF/UA requires /Lang on the document catalog",
                )
            )

        # 4. Title
        has_title = False
        vp = pdf.Root.get("/ViewerPreferences")
        if vp and vp.get("/DisplayDocTitle"):
            try:
                with pdf.open_metadata() as meta:
                    title = meta.get("dc:title", "")
                    if title:
                        has_title = True
            except Exception:
                pass

        if has_title:
            checks.append(
                CheckResult(
                    "title",
                    True,
                    "major",
                    "Document title present and DisplayDocTitle set",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "title",
                    False,
                    "major",
                    "Missing document title or DisplayDocTitle not set",
                )
            )

        # 5. Tab order
        all_tabs_set = True
        for page in pdf.pages:
            if page.get("/Tabs") != Name("/S"):
                all_tabs_set = False
                break
        if all_tabs_set:
            checks.append(
                CheckResult(
                    "tab_order",
                    True,
                    "major",
                    "Tab order set to structure on all pages",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "tab_order",
                    False,
                    "major",
                    "Tab order not set to /S (structure) on all pages",
                )
            )

        # 6. Text layer presence (per page)
        pages_with_text = 0
        pages_without_text = 0
        for page_num, _ in enumerate(fitz_doc):
            fitz_page = fitz_doc[page_num]
            text = fitz_page.get_text("text").strip()
            if text:
                pages_with_text += 1
            else:
                pages_without_text += 1

        if pages_without_text > 0:
            checks.append(
                CheckResult(
                    "text_layer",
                    False,
                    "major",
                    f"{pages_without_text}/{len(fitz_doc)} page(s) have no text layer",
                    "Image-only pages need invisible text overlays or OCR for accessibility",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "text_layer", True, "info", "All pages have extractable text"
                )
            )

        # 7. Figure alt text (check structure tree)
        figures_missing_alt = 0
        figures_with_alt = 0
        if struct_root:
            PDFAccessibilityAuditor._count_figure_alt(
                struct_root, figures_with_alt, figures_missing_alt
            )
            figures_with_alt, figures_missing_alt = (
                PDFAccessibilityAuditor._walk_struct_for_figures(struct_root)
            )

        if figures_missing_alt > 0:
            checks.append(
                CheckResult(
                    "figure_alt_text",
                    False,
                    "critical",
                    f"{figures_missing_alt} Figure element(s) missing alt text",
                    "PDF/UA requires alt text on all Figure structure elements",
                )
            )
        elif figures_with_alt > 0:
            checks.append(
                CheckResult(
                    "figure_alt_text",
                    True,
                    "critical",
                    f"All {figures_with_alt} Figure element(s) have alt text",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "figure_alt_text",
                    True,
                    "info",
                    "No Figure elements found in structure tree",
                )
            )

        # 8. PDF/UA identifier
        has_ua_id = False
        try:
            with pdf.open_metadata() as meta:
                raw = str(meta)
                if "pdfuaid" in raw.lower() or "PDF/UA" in raw:
                    has_ua_id = True
        except Exception:
            pass

        if has_ua_id:
            checks.append(
                CheckResult(
                    "pdfua_identifier",
                    True,
                    "minor",
                    "PDF/UA identifier found in metadata",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "pdfua_identifier",
                    False,
                    "minor",
                    "No PDF/UA identifier in XMP metadata",
                    "PDF/UA-1 requires pdfuaid:part in XMP metadata",
                )
            )

        return checks

    @staticmethod
    def _count_figure_alt(node, with_alt, without_alt):  # type: ignore[empty-body]
        """Deprecated — use _walk_struct_for_figures."""

    @staticmethod
    def _walk_struct_for_figures(node) -> Tuple[int, int]:
        """Recursively count figures with/without alt text in structure tree."""
        with_alt = 0
        without_alt = 0

        try:
            s = node.get("/S")
            if s and str(s) == "/Figure":
                if node.get("/Alt"):
                    with_alt += 1
                else:
                    without_alt += 1

            kids = node.get("/K")
            if isinstance(kids, Array):
                for child in kids:  # type: ignore[attr-defined]
                    try:
                        child_obj = child
                        if hasattr(child, "get_object"):
                            child_obj = child.get_object()
                        if isinstance(child_obj, Dictionary) and child_obj.get(
                            "/Type"
                        ) in (Name("/StructElem"), None):
                            w, wo = PDFAccessibilityAuditor._walk_struct_for_figures(
                                child_obj
                            )
                            with_alt += w
                            without_alt += wo
                    except Exception:
                        continue
            elif isinstance(kids, Dictionary):
                if kids.get("/Type") == Name("/StructElem") or kids.get("/S"):
                    w, wo = PDFAccessibilityAuditor._walk_struct_for_figures(kids)
                    with_alt += w
                    without_alt += wo
        except Exception:
            pass

        return with_alt, without_alt

    @staticmethod
    def compute_level(checks: List[CheckResult]) -> ComplianceLevel:
        """Derive overall compliance from check results."""
        critical_fail = any(not c.passed for c in checks if c.severity == "critical")
        major_fail = any(not c.passed for c in checks if c.severity == "major")

        if critical_fail:
            return ComplianceLevel.FAIL
        elif major_fail:
            return ComplianceLevel.PASS_WITH_WARNINGS
        else:
            return ComplianceLevel.PASS
