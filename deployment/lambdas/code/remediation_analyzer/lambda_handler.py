"""
PDF Accessibility Remediation Lambda Handler

Full pipeline: PDF -> analyze pages -> apply PDF/UA tags -> output tagged PDF
"""

import json
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

from pdf_accessibility_tagger import PDFAccessibilityTagger

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

        body = json.loads(event["body"]) if "body" in event else event

        session_id = body.get("session_id", "no_session")
        pdf_path = body.get("pdf_path")
        title = body.get("title", "Accessible Document")
        lang = body.get("lang", "en-US")
        render_dpi = body.get("dpi", 150)
        correlation_uri = body.get("correlation_uri")

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
        )

        output_bucket = os.environ.get("OUTPUT_BUCKET")
        if output_bucket and result.get("output_pdf"):
            s3_uri = _upload_to_s3(
                local_path=result["output_pdf"],
                bucket=output_bucket,
                session_id=session_id,
                original_key=pdf_path,
            )
            result["s3_output_uri"] = s3_uri
            logger.info("Uploaded tagged PDF to: %s", s3_uri)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "result": result,
                    "success": True,
                    "session_id": session_id,
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
) -> dict[str, Any]:
    """Full PDF remediation pipeline.

    When correlation_uri is provided, uses the correlation content spine
    for element metadata and resolves coordinates via PyMuPDF text search
    (text elements) and targeted vision model calls (figures only).
    Falls back to full vision model analysis when no correlation data exists.
    """
    import fitz
    from PIL import Image

    work_dir = Path(tempfile.mkdtemp(prefix="pdf_remediation_"))

    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    logger.info("Processing %d pages at %d DPI", num_pages, render_dpi)

    # Load correlation data if available
    correlation_pages = {}
    if correlation_uri:
        try:
            correlation_xml = _download_from_s3(correlation_uri)
            with open(correlation_xml, "r", encoding="utf-8") as f:
                correlation_pages = _parse_correlation_xml(f.read())
            logger.info(
                "Loaded correlation data for %d page(s)", len(correlation_pages)
            )
        except Exception as e:
            logger.warning("Failed to load correlation data, falling back: %s", e)
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

            # For figures, use targeted vision model call
            if figure_elements:
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

                figure_resolved = _resolve_figure_bboxes(
                    analyzer, image_data, figure_elements, aws_profile
                )
                resolved.extend(figure_resolved)

            elements = resolved
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

    output_pdf = work_dir / "tagged_output.pdf"

    with PDFAccessibilityTagger(pdf_path) as tagger:
        for page_num, elements in page_elements.items():
            for elem in elements:
                if not elem.get("bbox"):
                    continue

                bbox = elem["bbox"]
                tag = _map_element_type_to_pdf_tag(elem.get("type", "P"))
                alt_text = elem.get("alt_text", "")

                if tag == "Figure" and not alt_text:
                    alt_text = elem.get("content", "Figure")[:200]

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
                    order=elem.get("order", 0),
                )

        tagger.save(str(output_pdf), title=title, lang=lang)

    logger.info("Created tagged PDF: %s", output_pdf)

    return {
        "output_pdf": str(output_pdf),
        "pages_processed": num_pages,
        "analysis": all_results,
        "correlation_used": bool(correlation_pages),
    }


def _parse_correlation_xml(xml_content: str) -> dict[int, list[dict[str, Any]]]:
    """Parse correlation XML into a page-indexed dict of elements.

    Returns:
        Dict mapping page number (1-indexed) to list of element dicts with
        keys: id, type, order, text, alt_text, caption.
    """
    pages: dict[int, list[dict[str, Any]]] = {}

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.warning("Failed to parse correlation XML: %s", e)
        return pages

    # Get page number from root attribute
    page_num = int(root.get("page", "1"))

    elements = []
    for elem in root.iter("element"):
        elem_id = elem.get("id", "")
        elem_type = elem.get("type", "paragraph")
        elem_order = int(elem.get("order", "0"))

        # Extract text content
        text_node = elem.find("text")
        text = (
            text_node.text.strip() if text_node is not None and text_node.text else ""
        )

        # Extract alt_text (for figures)
        alt_node = elem.find("alt_text")
        alt_text = (
            alt_node.text.strip() if alt_node is not None and alt_node.text else ""
        )

        # Extract caption (for figures)
        caption_node = elem.find("caption")
        caption = (
            caption_node.text.strip()
            if caption_node is not None and caption_node.text
            else ""
        )

        elements.append(
            {
                "id": elem_id,
                "type": elem_type,
                "order": elem_order,
                "text": text,
                "alt_text": alt_text,
                "caption": caption,
            }
        )

    if elements:
        pages[page_num] = elements
        logger.info("Parsed %d elements for page %d", len(elements), page_num)

    return pages


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
                logger.warning("Could not resolve bbox for text: '%s'", text[:60])

    logger.info(
        "Resolved %d/%d text element bboxes via PyMuPDF",
        len(resolved),
        len(text_elements),
    )
    return resolved


def _resolve_figure_bboxes(
    analyzer: Any,
    image_data: bytes,
    figure_elements: list[dict[str, Any]],
    aws_profile: Optional[str] = None,  # noqa: ARG001
) -> list[dict[str, Any]]:
    """Resolve bounding boxes for figure elements using a targeted vision model call.

    Sends the page image with a focused prompt listing only the figures
    to locate, using descriptions from the correlation data.
    Uses the analyzer's bedrock_client directly with a custom prompt
    rather than the full analysis pipeline.
    """
    import base64

    if not figure_elements:
        return []

    # Build a targeted prompt for figure coordinate extraction
    figure_descriptions = []
    for i, elem in enumerate(figure_elements):
        desc = (
            elem.get("caption") or elem.get("alt_text") or f"Figure {elem.get('id', i)}"
        )
        figure_descriptions.append(f"  {i + 1}. id=\"{elem.get('id', '')}\": {desc}")

    system_prompt = (
        "You are a precise visual element locator. Given a page image and a list of "
        "figure descriptions, return the bounding box coordinates for each figure. "
        "Coordinates must be normalized to 0-1 range where (0,0) is bottom-left and "
        "(1,1) is top-right. Return ONLY a JSON array."
    )

    user_text = (
        "Locate the following figures in this page image and return their "
        "bounding box coordinates as normalized values (0-1 range).\n\n"
        "Figures to locate:\n" + "\n".join(figure_descriptions) + "\n\n"
        "Return a JSON array where each object has:\n"
        '  {"id": "elem_XXX", "bbox": {"x0": float, "y0": float, "x1": float, "y1": float}}\n'
        "Return ONLY the JSON array, no other text."
    )

    image_b64 = base64.b64encode(image_data).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": user_text},
            ],
        }
    ]

    try:
        bedrock_client = analyzer.bedrock_client
        model_id = analyzer.config.get(
            "model_id", "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        )

        payload = bedrock_client.create_anthropic_payload(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=4000,
            temperature=0.1,
        )

        response = bedrock_client.invoke_model(
            model_id=model_id,
            payload=payload,
        )

        # Extract text from response
        result_text = ""
        for block in response.get("content", []):
            if block.get("type") == "text":
                result_text = block.get("text", "")
                break

        if not result_text:
            logger.warning("Empty response from figure bbox model call")
            return []

        bbox_results = _extract_json_from_response(result_text)

        # Build a lookup from the model response
        bbox_lookup = {}
        for item in bbox_results:
            if item.get("id") and item.get("bbox"):
                bbox_lookup[item["id"]] = item["bbox"]

        resolved = []
        for elem in figure_elements:
            elem_id = elem.get("id", "")
            bbox = bbox_lookup.get(elem_id)

            if bbox:
                resolved.append(
                    {
                        "type": "figure",
                        "order": elem["order"],
                        "alt_text": elem.get("alt_text", ""),
                        "content": elem.get("caption", ""),
                        "bbox": bbox,
                        "source": "vision_model_targeted",
                    }
                )
            else:
                logger.warning("No bbox returned for figure %s", elem_id)

        logger.info(
            "Resolved %d/%d figure bboxes via targeted vision model",
            len(resolved),
            len(figure_elements),
        )
        return resolved

    except Exception as e:
        logger.error("Failed to resolve figure bboxes: %s", e)
        return []


def _initialize_analyzer(aws_profile: Optional[str] = None):  # noqa: ARG001
    """Initialize the analyzer foundation.

    Args:
        aws_profile: AWS profile name (reserved for future use with BedrockClient)
    """
    from foundation.analyzer_foundation import AnalyzerFoundation
    from foundation.configuration_manager import ConfigurationManager
    from foundation.prompt_loader import PromptLoader
    from foundation.image_processor import ImageProcessor
    from foundation.bedrock_client import BedrockClient
    from foundation.message_chain_builder import MessageChainBuilder
    from foundation.response_processor import ResponseProcessor

    config_bucket = os.environ.get("CONFIG_BUCKET")
    analyzer_name = "remediation_analyzer"

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
    analyzer.bedrock_client = BedrockClient()
    analyzer.message_builder = MessageChainBuilder()
    analyzer.response_processor = ResponseProcessor()
    analyzer._configure_components()

    return analyzer


def _extract_json_from_response(response: str) -> list[dict[str, Any]]:
    """Extract JSON array from model response.

    Handles formats:
    - <analysis>...</analysis>[{...}]
    - ```json[{...}]```
    - Raw JSON array
    """
    import re

    # Remove analysis wrapper if present
    if "</analysis>" in response:
        response = response.split("</analysis>", 1)[1]

    # Remove markdown code blocks if present
    response = re.sub(r"```json\s*", "", response)
    response = re.sub(r"```\s*$", "", response)

    # Strip whitespace
    response = response.strip()

    # Parse JSON
    parsed = json.loads(response)

    # Ensure it's a list
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed)}")

    return parsed


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
        "heading": "H1",
        "paragraph": "P",
        "image": "Figure",
        "figure": "Figure",
        "table": "Table",
        "list": "L",
        "list_item": "LI",
    }
    return mapping.get(element_type, "P")


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
    os.close(fd)  # Close the file descriptor immediately

    s3.download_file(bucket, key, temp_path)
    return temp_path


def _upload_to_s3(
    local_path: str,
    bucket: str,
    session_id: str,
    original_key: str,
) -> str:
    """Upload file to S3 output bucket."""
    import boto3
    from datetime import datetime

    s3 = boto3.client("s3")

    original_name = Path(original_key).stem
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_key = f"remediated/{session_id}/{original_name}_tagged_{timestamp}.pdf"

    s3.upload_file(local_path, bucket, output_key)

    return f"s3://{bucket}/{output_key}"


def _error_response(message: str) -> dict[str, Any]:
    """Return error response."""
    return {
        "statusCode": 500,
        "body": json.dumps({"result": message, "success": False}),
    }
