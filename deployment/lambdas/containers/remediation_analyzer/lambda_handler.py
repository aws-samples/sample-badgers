"""
PDF Accessibility Remediation Lambda Handler

Full pipeline: PDF -> analyze pages -> apply PDF/UA tags -> output tagged PDF
"""

import json
import logging
import base64
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from pdf_accessibility_tagger import PDFAccessibilityTagger

from foundation.s3_result_saver import save_result_to_s3

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

        logger.info("Processing request for session: %s", session_id)
        logger.info("PDF path: %s", pdf_path)

        if not pdf_path:
            return _error_response("Missing required parameter: pdf_path")

        local_pdf = _download_from_s3(pdf_path)
        logger.info("Downloaded PDF to: %s", local_pdf)

        result = process_pdf(
            pdf_path=local_pdf,
            title=title,
            lang=lang,
            render_dpi=render_dpi,
            session_id=session_id,
            aws_profile=body.get("aws_profile"),
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
    session_id: str,
    aws_profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Full PDF remediation pipeline."""
    import fitz
    from PIL import Image

    work_dir = Path(tempfile.mkdtemp(prefix="pdf_remediation_"))

    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    logger.info("Processing %d pages at %d DPI", num_pages, render_dpi)

    analyzer = _initialize_analyzer(aws_profile)

    all_results = []
    page_elements: Dict[int, List[Dict]] = {}

    for page_num in range(num_pages):
        logger.info("Analyzing page %d/%d", page_num + 1, num_pages)

        page = doc[page_num]
        zoom = render_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_path = work_dir / f"page_{page_num}.png"
        img.save(img_path)

        with open(img_path, "rb") as f:
            image_data = f.read()

        analysis_result = analyzer.analyze(image_data, aws_profile)

        try:
            elements = json.loads(analysis_result)
            page_elements[page_num] = elements
            all_results.append({"page": page_num, "elements": elements})
            logger.info("Found %d elements on page %d", len(elements), page_num)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse analysis for page %d: %s", page_num, e)
            page_elements[page_num] = []

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
    }


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
                "remediation_grid_specification.xml",
                "remediation_element_types.xml",
                "remediation_rules.xml",
                "remediation_coordinate_system.xml",
                "remediation_format.xml",
                "remediation_common_errors.xml",
                "remediation_critical_reminders.xml",
            ],
            "max_examples": 0,
            "analysis_text": "document structure elements",
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
    import os

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


def _error_response(message: str) -> Dict:
    """Return error response."""
    return {
        "statusCode": 500,
        "body": json.dumps({"result": message, "success": False}),
    }
