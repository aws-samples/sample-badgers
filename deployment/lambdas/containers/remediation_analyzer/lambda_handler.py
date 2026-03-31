"""
CHECKPOINT_04E — LAMBDA HANDLER
=================================
PDF Accessibility Remediation Lambda Handler

Full pipeline: PDF -> analyze pages -> apply PDF/UA tags -> output tagged PDF

Changes from previous version:
  - Uses spine_parser.parse_correlation_xml (supports v1.0 flat and v2.0 hierarchical)
  - Stores StructureElement trees per page from correlation data
  - Merges per-page trees into a document-wide tree
  - Passes the tree to tagger.set_structure_tree() for nested Sect output
  - Old inline _parse_correlation_xml removed (replaced by spine_parser)
"""

import base64
import json
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

from pdf_accessibility_tagger import PDFAccessibilityTagger, AccessibilityReport
from pdf_accessibility_models import StructureElement, ElementRole
from spine_parser import parse_correlation_xml
from cell_grid_resolver import resolve_elements_via_grid
from diagnostic_visualizer import capture_page_diagnostics
from pdf_syntax_repair import repair_pdf as syntax_repair_pdf, RepairResult

logger = logging.getLogger()
log_level = os.environ.get("LOGGING_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))


def lambda_handler(event, context):
    """Lambda handler for PDF Accessibility Remediation."""
    try:
        if hasattr(context, "client_context") and context.client_context:
            gateway_id = context.client_context.custom.get(
                "bedrockAgentCoreGatewayId", "unknown"
            )
            tool_name = context.client_context.custom.get(
                "bedrockAgentCoreToolName", "unknown"
            )
            logger.info(
                "Gateway invocation - Gateway: %s, Tool: %s", gateway_id, tool_name
            )

        # Config source detection (matches other analyzer pattern)
        config_bucket = os.environ.get("CONFIG_BUCKET")
        analyzer_name = os.environ.get("ANALYZER_NAME", "remediation_analyzer")

        if os.environ.get("AWS_EXECUTION_ENV") and config_bucket:
            logger.info(
                "Using S3 config: bucket=%s, analyzer=%s", config_bucket, analyzer_name
            )
        else:
            logger.info("Using local config")

        body = json.loads(event["body"]) if "body" in event else event

        session_id = body.get("session_id", "no_session")
        pdf_path = body.get("pdf_path")
        title = body.get("title", "Accessible Document")
        lang = body.get("lang", "en-US")
        render_dpi = body.get("dpi", 150)
        correlation_uri = body.get("correlation_uri")
        page_b64_uris = body.get("page_b64_uris", {})

        logger.info("Processing request for session: %s", session_id)
        logger.info("PDF path: %s", pdf_path)
        if correlation_uri:
            logger.info("Correlation URI provided: %s", correlation_uri)

        if not pdf_path:
            return _error_response("Missing required parameter: pdf_path")

        local_pdf = _download_from_s3(pdf_path)
        logger.info("Downloaded PDF to: %s", local_pdf)

        result = process_pdf(
            pdf_path=local_pdf,
            title=title,
            lang=lang,
            render_dpi=render_dpi,
            aws_profile=body.get("aws_profile"),
            correlation_uri=correlation_uri,
            page_b64_uris=page_b64_uris,
            session_id=session_id,
        )

        output_bucket = os.environ.get("OUTPUT_BUCKET")
        if output_bucket and result.get("output_pdf"):
            s3_uri = _upload_to_s3(
                local_path=result["output_pdf"],
                bucket=output_bucket,
                analyzer_name=analyzer_name,
                session_id=session_id,
                original_key=pdf_path,
            )
            result["s3_output_uri"] = s3_uri
            logger.info("Uploaded tagged PDF to: %s", s3_uri)

            # Upload accessibility report as companion JSON
            if result.get("accessibility_report"):
                report_path = result["output_pdf"].replace(".pdf", "_report.json")
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(result["accessibility_report"], f, indent=2)
                report_s3_uri = _upload_to_s3(
                    local_path=report_path,
                    bucket=output_bucket,
                    analyzer_name=analyzer_name,
                    session_id=session_id,
                    original_key=pdf_path.replace(".pdf", "_report.json"),
                )
                result["s3_report_uri"] = report_s3_uri
                logger.info("Uploaded accessibility report to: %s", report_s3_uri)

        # Include compliance verdict at top level for easy routing
        report_data = result.get("accessibility_report", {})
        compliance = report_data.get("post_remediation", {}).get(
            "compliance_level", "not_assessed"
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "result": result,
                    "success": True,
                    "session_id": session_id,
                    "compliance": compliance,
                }
            ),
        }

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        return _error_response(str(e))


def process_pdf(
    pdf_path: str,
    title: str,
    lang: str,
    render_dpi: int,
    aws_profile: Optional[str] = None,
    correlation_uri: Optional[str] = None,
    page_b64_uris: Optional[dict[str, str]] = None,
    session_id: str = "",
) -> dict[str, Any]:
    """Full PDF remediation pipeline.

    When correlation_uri is provided, uses spine_parser to build a
    StructureElement tree AND flat element list for bbox resolution.
    The tree is passed to tagger.set_structure_tree() for nested output.

    Falls back to full vision model analysis when no correlation data exists.
    """
    import fitz
    from PIL import Image

    work_dir = Path(tempfile.mkdtemp(prefix="pdf_remediation_"))

    # --- Syntax repair pass (fix corrupt xref, streams, etc.) ---
    repair_result: RepairResult | None = None
    if os.environ.get("ENABLE_SYNTAX_REPAIR", "true").lower() == "true":
        try:
            repair_result = syntax_repair_pdf(pdf_path, work_dir=work_dir)
            if repair_result.any_repair_applied:
                logger.info(
                    "Syntax repair applied: %s → %s (%+d bytes)",
                    pdf_path,
                    repair_result.output_path,
                    repair_result.size_delta,
                )
                pdf_path = repair_result.output_path
            else:
                logger.info("Syntax repair: no changes needed")
        except Exception as e:
            logger.error("Syntax repair failed: %s", e, exc_info=True)
            raise RuntimeError(
                f"PDF syntax repair failed and the file could not be made processable. "
                f"Accessibility tagging was not attempted. "
                f"Repair error: {e}"
            ) from e

    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    logger.info("Processing %d pages at %d DPI", num_pages, render_dpi)

    # --- Load correlation data ---
    # NEW: parse_correlation_xml returns (StructureElement tree, flat page elements)
    correlation_trees: dict[int, StructureElement] = {}  # page_num → tree
    correlation_pages: dict[int, list[dict[str, Any]]] = {}  # page_num → flat elements

    if correlation_uri:
        try:
            correlation_xml = _download_from_s3(correlation_uri)
            with open(correlation_xml, "r", encoding="utf-8") as f:
                xml_content = f.read()

            tree, page_elems = parse_correlation_xml(xml_content)

            if tree is not None:
                # Determine which page this correlation covers
                # (parse_correlation_xml returns 1-indexed pages in page_elems)
                for page_num in page_elems:
                    correlation_trees[page_num] = tree
                    correlation_pages[page_num] = page_elems[page_num]

            logger.info(
                "Loaded correlation data: tree=%s, %d page(s) with elements",
                "yes" if tree else "no",
                len(correlation_pages),
            )
        except Exception as e:
            logger.warning("Failed to load correlation data, falling back: %s", e)
            correlation_trees = {}
            correlation_pages = {}

    analyzer = None  # Lazy-init only if needed

    all_results: list[dict[str, Any]] = []
    page_elements: dict[int, list[dict[str, Any]]] = {}

    for page_num in range(num_pages):
        logger.info("Analyzing page %d/%d", page_num + 1, num_pages)

        page = doc[page_num]

        # Check if we have correlation data for this page (1-indexed in XML)
        corr_elements = correlation_pages.get(page_num + 1)

        if corr_elements:
            logger.info(
                "Using correlation-guided path for page %d (%d elements)",
                page_num + 1,
                len(corr_elements),
            )
            # Resolve text element coordinates from PDF text layer
            text_elements = [e for e in corr_elements if e["type"] != "figure"]
            figure_elements = [e for e in corr_elements if e["type"] == "figure"]

            resolved = _resolve_text_bboxes(page, text_elements)

            # Separate successfully resolved from fallback-stacked
            text_resolved = [
                e for e in resolved if e.get("source") != "fallback_stacked"
            ]
            text_unresolved = [
                e for e in resolved if e.get("source") == "fallback_stacked"
            ]

            if text_unresolved:
                logger.info(
                    "%d/%d text elements unresolved via text search, "
                    "routing to cell grid resolver",
                    len(text_unresolved),
                    len(text_elements),
                )

            # Combine unresolved text + ALL figures for a single grid call
            grid_candidates = list(text_unresolved)
            for fig in figure_elements:
                grid_candidates.append(
                    {
                        "id": fig.get("id", ""),
                        "type": "figure",
                        "text": fig.get("caption", ""),
                        "alt_text": fig.get("alt_text", ""),
                        "order": fig.get("order", 0),
                    }
                )

            image_data = None  # Track for diagnostics

            if grid_candidates:
                # Try pre-processed b64 first, fall back to render + optimize
                b64_uri = (page_b64_uris or {}).get(str(page_num))
                if b64_uri:
                    try:
                        local_b64 = _download_from_s3(b64_uri)
                        with open(local_b64, "r", encoding="utf-8") as f:
                            b64_str = f.read().strip()
                        image_data = base64.b64decode(b64_str)
                        logger.info(
                            "Using pre-processed b64 for page %d (%d bytes)",
                            page_num,
                            len(image_data),
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to load b64 for page %d, falling back: %s",
                            page_num,
                            e,
                        )
                        b64_uri = None  # trigger fallback below

                if not b64_uri:
                    # Render from PDF and optimize via ImageProcessor
                    zoom = render_dpi / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    img_path = work_dir / f"page_{page_num}.png"
                    img.save(img_path)

                    with open(img_path, "rb") as f:
                        raw_data = f.read()

                    if analyzer is None:
                        analyzer = _initialize_analyzer(aws_profile)

                    image_data = analyzer.image_processor.optimize_image(raw_data)
                    logger.info(
                        "Rendered + optimized page %d: %d -> %d bytes",
                        page_num,
                        len(raw_data),
                        len(image_data),
                    )

                if analyzer is None:
                    analyzer = _initialize_analyzer(aws_profile)

                logger.info(
                    "Cell grid resolver: locating %d elements (%d text + %d figures)",
                    len(grid_candidates),
                    len(text_unresolved),
                    len(figure_elements),
                )
                grid_resolved = resolve_elements_via_grid(
                    image_data,
                    grid_candidates,
                    analyzer,
                    aws_profile,
                    resolved_anchors=text_resolved,
                )
                text_resolved.extend(grid_resolved)

            # Capture diagnostics (if enabled via ENABLE_DIAGNOSTICS env var)
            if image_data is not None:
                capture_page_diagnostics(
                    page_image_data=image_data,
                    page_number=page_num + 1,  # 1-indexed
                    correlation_elements=corr_elements,
                    resolved_elements=text_resolved,
                    grid_cols=10,
                    grid_rows=14,
                    gridded_image=None,
                    pdf_path=pdf_path,
                    session_id=session_id,
                )

            elements = text_resolved
        else:
            # Fallback: full vision model analysis (original behavior)
            logger.info(
                "No correlation data for page %d, using full analysis", page_num + 1
            )
            zoom = render_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img_path = work_dir / f"page_{page_num}.png"
            img.save(img_path)

            with open(img_path, "rb") as f:
                image_data = f.read()

            if analyzer is None:
                analyzer = _initialize_analyzer(aws_profile)

            analysis_result = analyzer.analyze(image_data, aws_profile)

            try:
                elements = _extract_json_from_response(analysis_result)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Failed to parse analysis for page %d: %s", page_num, e)
                elements = []

        page_elements[page_num] = elements
        all_results.append({"page": page_num, "elements": elements})
        logger.info("Found %d elements on page %d", len(elements), page_num)

    doc.close()

    # --- Build document-wide structure tree from per-page trees ---
    document_tree = _merge_page_trees(correlation_trees, num_pages)

    output_pdf = work_dir / "tagged_output.pdf"

    with PDFAccessibilityTagger(pdf_path) as tagger:
        # NEW: set the hierarchical structure tree if available
        if document_tree is not None:
            tagger.set_structure_tree(document_tree)
            logger.info(
                "Set structure tree: %d nodes",
                sum(1 for _ in document_tree.walk()),
            )

        # Log pre-remediation audit
        logger.info("Pre-remediation compliance: %s", tagger.report.pre_level.value)
        for check in tagger.report.pre_checks:
            level = "PASS" if check.passed else "FAIL"
            logger.info("  [%s] %s: %s", level, check.name, check.message)

        for page_num, elements in page_elements.items():
            for elem in elements:
                if not elem.get("bbox"):
                    continue

                bbox = elem["bbox"]
                tag = _map_element_type_to_pdf_tag(elem.get("type", "P"))
                alt_text = elem.get("alt_text", "")
                text_content = elem.get("content", "")

                if tag == "Figure" and not alt_text:
                    alt_text = text_content[:200] if text_content else "Figure"

                tagger.add_region_normalized(
                    page=page_num,
                    bbox_normalized=(
                        bbox.get("x0", 0),
                        bbox.get("y0", 0),
                        bbox.get("x1", 1),
                        bbox.get("y1", 1),
                    ),
                    tag=tag,
                    alt_text=alt_text,
                    text_content=text_content,
                    order=elem.get("order", 0),
                    element_id=elem.get("id", ""),
                    source=elem.get("source", ""),
                )

        output_path, report = tagger.save(str(output_pdf), title=title, lang=lang)

    # Log post-remediation audit
    logger.info("Post-remediation compliance: %s", report.post_level.value)
    for check in report.post_checks:
        level = "PASS" if check.passed else "FAIL"
        logger.info("  [%s] %s: %s", level, check.name, check.message)

    logger.info(
        "Created tagged PDF: %s (elements: %d, overlays: %d)",
        output_pdf,
        report.total_elements_tagged,
        report.invisible_text_overlays_added,
    )

    return {
        "output_pdf": str(output_pdf),
        "pages_processed": num_pages,
        "analysis": all_results,
        "correlation_used": bool(correlation_pages),
        "structure_tree_used": document_tree is not None,
        "syntax_repair": {
            "applied": repair_result.any_repair_applied if repair_result else False,
            "pass1_ok": repair_result.pass1_ok if repair_result else False,
            "pass2_ok": repair_result.pass2_ok if repair_result else False,
            "size_delta": repair_result.size_delta if repair_result else 0,
            "logs": repair_result.logs if repair_result else [],
        },
        "accessibility_report": report.to_dict(),
    }


# ============================================================================
# NEW: Merge per-page trees into document-wide tree
# ============================================================================


def _merge_page_trees(
    page_trees: dict[int, StructureElement],
    num_pages: int,
) -> Optional[StructureElement]:
    """Merge per-page StructureElement trees into a single document tree.

    For single-page documents, returns the page tree directly.
    For multi-page documents, merges children of each page's root Sect
    into a single Document → Sect hierarchy.

    Returns None if no trees are available.
    """
    if not page_trees:
        return None

    if len(page_trees) == 1:
        # Single page — return as-is
        return next(iter(page_trees.values()))

    # Multi-page: merge all page content under one Document → Sect root
    doc = StructureElement(
        id="doc", tag="Document", role=ElementRole.GROUPING, depth=0
    )
    root_sect = doc.add_child(StructureElement(
        id="sect_root", tag="Sect", role=ElementRole.GROUPING
    ))

    for page_num in sorted(page_trees.keys()):
        page_tree = page_trees[page_num]

        # Each page tree is Document → Sect(s) → content
        # We want to lift the content into our merged root
        for doc_child in page_tree.children:
            if doc_child.tag == "Sect":
                # Lift this sect's children into root_sect
                for sect_child in doc_child.children:
                    root_sect.add_child(sect_child)
            else:
                root_sect.add_child(doc_child)

    total = sum(1 for _ in doc.walk())
    logger.info(
        "Merged %d page tree(s) into document tree: %d total nodes",
        len(page_trees),
        total,
    )

    return doc


# ============================================================================
# Bbox resolution (unchanged from previous version)
# ============================================================================


def _resolve_text_bboxes(
    page: Any, text_elements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Resolve bounding boxes for text elements using PyMuPDF text search.

    Uses page.search_for() to find exact text positions in the PDF text layer.
    Returns elements in the same format expected by the tagger.
    """
    resolved = []

    page_width = page.rect.width
    page_height = page.rect.height

    for elem in text_elements:
        text = elem.get("text", "")
        if not text:
            continue

        elem_id = elem.get("id", "")

        # Search for the text in the PDF page
        rects = page.search_for(text)

        if rects:
            # Use the first match; normalize to 0-1 range
            rect = rects[0]
            resolved.append(
                {
                    "type": elem["type"],
                    "order": elem["order"],
                    "alt_text": elem.get("alt_text", ""),
                    "content": text,
                    "id": elem_id,
                    "bbox": {
                        "x0": rect.x0 / page_width,
                        "y0": 1
                        - (rect.y1 / page_height),  # Flip Y for normalized coords
                        "x1": rect.x1 / page_width,
                        "y1": 1 - (rect.y0 / page_height),
                    },
                    "source": "pymupdf_text_search",
                }
            )
            logger.debug("Text bbox resolved for '%s': %s", text[:40], rect)
        else:
            # Try partial match with first 50 chars
            partial = text[:50]
            rects = page.search_for(partial)
            if rects:
                rect = rects[0]
                resolved.append(
                    {
                        "type": elem["type"],
                        "order": elem["order"],
                        "alt_text": elem.get("alt_text", ""),
                        "content": text,
                        "id": elem_id,
                        "bbox": {
                            "x0": rect.x0 / page_width,
                            "y0": 1 - (rect.y1 / page_height),
                            "x1": rect.x1 / page_width,
                            "y1": 1 - (rect.y0 / page_height),
                        },
                        "source": "pymupdf_partial_search",
                    }
                )
                logger.debug("Partial text bbox resolved for '%s'", partial[:40])
            else:
                # No text layer match — still include so tagger can overlay
                logger.warning(
                    "Could not resolve bbox for text: '%s' — "
                    "will use full-page fallback for invisible overlay",
                    text[:60],
                )
                resolved.append(
                    {
                        "type": elem["type"],
                        "order": elem["order"],
                        "alt_text": elem.get("alt_text", ""),
                        "content": text,
                        "id": elem_id,
                        "bbox": {
                            "x0": 0.02,
                            "y0": max(0.02, 1.0 - (elem["order"] * 0.05)),
                            "x1": 0.98,
                            "y1": min(0.98, 1.0 - (elem["order"] * 0.05) + 0.04),
                        },
                        "source": "fallback_stacked",
                    }
                )

    logger.info(
        "Resolved %d/%d text element bboxes via PyMuPDF",
        len(resolved),
        len(text_elements),
    )
    return resolved


# ============================================================================
# Analyzer initialization (unchanged)
# ============================================================================


def _initialize_analyzer(aws_profile: Optional[str] = None):
    """Initialize the analyzer foundation."""
    from foundation.analyzer_foundation import AnalyzerFoundation
    from foundation.configuration_manager import ConfigurationManager
    from foundation.prompt_loader import PromptLoader
    from foundation.image_processor import ImageProcessor
    from foundation.bedrock_client import BedrockClient
    from foundation.message_chain_builder import MessageChainBuilder
    from foundation.response_processor import ResponseProcessor

    config_bucket = os.environ.get("CONFIG_BUCKET")
    analyzer_name = os.environ.get("ANALYZER_NAME", "remediation_analyzer")

    if os.environ.get("AWS_EXECUTION_ENV") and config_bucket:
        from foundation.s3_config_loader import load_manifest_from_s3

        manifest = load_manifest_from_s3(config_bucket, analyzer_name)
        config = manifest.get("analyzer", manifest)
        config_source = "s3"
    else:
        config = {
            "prompt_files": [
                "remediation_job_role.xml",
                "remediation_context.xml",
                "remediation_coordinate_system.xml",
                "remediation_element_types.xml",
                "remediation_rules.xml",
                "remediation_output_format.xml",
            ],
            "max_examples": 0,
            "analysis_text": "PDF page elements for accessibility tagging",
        }
        config_source = "local"

    analyzer = object.__new__(AnalyzerFoundation)
    analyzer.analyzer_type = analyzer_name
    analyzer.s3_bucket = config_bucket if config_source == "s3" else None
    analyzer.logger = logging.getLogger(f"foundation.{analyzer_name}")
    analyzer.config = config
    analyzer.global_settings = {
        "max_tokens": int(os.environ.get("MAX_TOKENS", "8000")),
        "resolver_max_tokens": int(os.environ.get("RESOLVER_MAX_TOKENS", "4000")),
        "temperature": float(os.environ.get("TEMPERATURE", "0.1")),
        "max_image_size": int(os.environ.get("MAX_IMAGE_SIZE", "20971520")),
        "max_dimension": int(os.environ.get("MAX_DIMENSION", "2048")),
        "jpeg_quality": int(os.environ.get("JPEG_QUALITY", "85")),
        "cache_enabled": os.environ.get("CACHE_ENABLED", "True") == "True",
        "throttle_delay": float(os.environ.get("THROTTLE_DELAY", "1.0")),
        "aws_region": os.environ.get("AWS_REGION", "us-west-2"),
    }

    analyzer.config_manager = ConfigurationManager()

    if config_source == "s3":
        analyzer.prompt_loader = PromptLoader(
            config_source="s3", s3_bucket=config_bucket, analyzer_name=analyzer_name
        )
    else:
        analyzer.prompt_loader = PromptLoader(config_source="local")

    analyzer.image_processor = ImageProcessor()
    analyzer.bedrock_client = BedrockClient(
        aws_region=analyzer.global_settings.get("aws_region", "us-west-2"),
    )
    analyzer.aws_profile = aws_profile
    analyzer.message_builder = MessageChainBuilder()
    analyzer.response_processor = ResponseProcessor()
    analyzer._configure_components()

    return analyzer


# ============================================================================
# Response parsing (unchanged)
# ============================================================================


def _extract_json_from_response(response: str) -> list[dict[str, Any]]:
    """Extract JSON array from model response."""
    import re

    if "</analysis>" in response:
        response = response.split("</analysis>", 1)[1]

    response = re.sub(r"```json\s*", "", response)
    response = re.sub(r"```\s*$", "", response)
    response = response.strip()

    parsed = json.loads(response)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed)}")

    return parsed


# ============================================================================
# Tag mapping (unchanged)
# ============================================================================


def _map_element_type_to_pdf_tag(element_type: str) -> str:
    """Map analyzer element types to valid PDF structure tags."""
    mapping = {
        "H1": "H1",
        "H2": "H2",
        "H3": "H3",
        "H4": "H4",
        "H5": "H5",
        "H6": "H6",
        "P": "P",
        "Figure": "Figure",
        "Table": "Table",
        "Caption": "Caption",
        "L": "L",
        "LI": "LI",
        "Link": "Link",
        "Note": "Note",
        "TOC": "TOC",
        "TOCI": "TOCI",
        "Quote": "Quote",
        "BlockQuote": "BlockQuote",
        "Formula": "Formula",
        "Artifact": "Artifact",
        # Lowercase variants from correlation XML / vision model
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
        "footer": "NonStruct",
        "header": "NonStruct",
        "caption": "Caption",
        "blockquote": "BlockQuote",
        "code": "Code",
        "formula": "Formula",
    }
    return mapping.get(element_type, "P")


# ============================================================================
# S3 helpers (unchanged)
# ============================================================================


def _download_from_s3(s3_uri: str) -> str:
    """Download file from S3 to temp location."""
    import boto3

    if not s3_uri.startswith("s3://"):
        return s3_uri

    s3 = boto3.client("s3")
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]

    ext = Path(key).suffix or ".pdf"
    fd, temp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)

    s3.download_file(bucket, key, temp_path)
    return temp_path


def _upload_to_s3(
    local_path: str,
    bucket: str,
    analyzer_name: str,
    session_id: str,
    original_key: str,
) -> str:
    """Upload file to S3 output bucket."""
    import boto3
    from datetime import datetime

    s3 = boto3.client("s3")

    original_name = Path(original_key).stem
    ext = Path(local_path).suffix or Path(original_key).suffix or ".pdf"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_key = (
        f"{analyzer_name}/results/{session_id}/{original_name}_{timestamp}{ext}"
    )

    s3.upload_file(local_path, bucket, output_key)

    return f"s3://{bucket}/{output_key}"


def _error_response(message: str) -> dict[str, Any]:
    """Return error response."""
    return {
        "statusCode": 500,
        "body": json.dumps({"result": message, "success": False}),
    }
