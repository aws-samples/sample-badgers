"""Native PDF page planning and BADGERS spine extraction.

This tool deliberately reads the PDF's digital layer before any rasterization.
It does not attempt to infer visual semantics from pixels; it produces
deterministic evidence, a page-level route, and a schema-compatible native
spine for pages whose text/order are sufficiently trustworthy.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import boto3
import fitz

from foundation import job_state
from foundation.lambda_error_handler import (
    ResourceNotFoundError,
    ValidationError,
    create_error_response,
)

logger = logging.getLogger()
logger.setLevel(getattr(logging, os.environ.get("LOGGING_LEVEL", "INFO").upper()))

_SEMANTIC_VISUAL = re.compile(
    r"\b(chart|diagram|figure|flow\s*chart|graph|map|matrix|process|"
    r"schema|workflow|architecture|equation|illustration)\b",
    re.IGNORECASE,
)
_TABLE_TOKEN = re.compile(r"\b(table|schedule|exhibit)\b", re.IGNORECASE)


@dataclass
class TextBlock:
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    order: int


@dataclass
class PageAnalysis:
    page_number: int
    text_blocks: list[TextBlock] = field(default_factory=list)
    page_area: float = 1.0
    image_count: int = 0
    image_area_ratio: float = 0.0
    vector_count: int = 0
    column_count: int = 1
    tagged_pdf_structure: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.text_blocks)


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:  # noqa: ARG001
    """Analyze every PDF page, persist native spines, and return a page plan."""
    job_id = ""
    subtask = ""
    try:
        body = json.loads(event["body"]) if "body" in event else event
        document_path = str(body.get("document_path") or "").strip()
        session_id = str(body.get("session_id") or "no_session")
        job_id = str(body.get("job_id") or "")
        doc_id = str(body.get("doc_id") or "")
        specialist_name = os.environ.get("SPECIALIST_NAME", "native_pdf_processor")

        if not document_path:
            raise ValidationError(
                message="Missing required parameter: document_path",
                details={"provided_keys": list(body.keys())},
            )
        if not document_path.lower().endswith(".pdf"):
            raise ValidationError(
                message="native_pdf_processor accepts PDF documents only.",
                details={"document_path": document_path},
            )

        subtask = job_state.subtask_id(specialist_name, document_path)
        job_state.mark_running(
            job_id,
            subtask,
            doc_id=doc_id,
            specialist=specialist_name,
            image_id=job_state.image_identifier(document_path),
            session_id=session_id,
        )

        output_bucket = os.environ.get("OUTPUT_BUCKET", "")
        if not output_bucket:
            raise RuntimeError("OUTPUT_BUCKET is not configured")

        pdf_bytes = _read_document(document_path)
        if not pdf_bytes.startswith(b"%PDF-"):
            raise ValidationError(
                message="The supplied document is not a PDF.",
                details={"document_path": document_path},
            )

        with tempfile.NamedTemporaryFile(suffix=".pdf") as local_pdf:
            local_pdf.write(pdf_bytes)
            local_pdf.flush()
            document = fitz.open(local_pdf.name)
            try:
                if document.needs_pass:
                    raise ValidationError(
                        message=(
                            "Password-protected PDFs are unsupported. Remove password "
                            "protection before uploading this document."
                        ),
                        details={
                            "code": "PASSWORD_PROTECTED_PDF",
                            "document_path": document_path,
                            "retryable": False,
                        },
                    )
                tagged = _has_tagged_structure(document)
                analyses = [
                    _analyze_page(page, page_index + 1, tagged)
                    for page_index, page in enumerate(document)
                ]
            finally:
                document.close()

        s3 = boto3.client("s3")
        plans: list[dict[str, Any]] = []
        for analysis in analyses:
            decision = _decide_route(analysis)
            native_uri = ""
            schema_valid = False
            if decision["native_fidelity"] != "incomplete":
                xml = _build_native_spine(
                    analysis=analysis,
                    document_path=document_path,
                    session_id=session_id,
                    route=decision["route"],
                    decision=decision,
                )
                schema_valid = _validate_native_spine(xml, analysis.page_number)
                if not schema_valid:
                    decision["native_fidelity"] = "incomplete"
                    decision["route"] = "textract_ocr"
                    decision["reasons"].append("native_spine_schema_validation_failed")
                else:
                    native_uri = _put_text(
                        s3,
                        output_bucket,
                        f"{session_id}/native_pdf/page_{analysis.page_number:03d}.xml",
                        xml,
                        "application/xml",
                    )
                    page_subtask = f"{specialist_name}:page_{analysis.page_number:03d}"
                    job_state.mark_running(
                        job_id,
                        page_subtask,
                        doc_id=doc_id,
                        specialist=specialist_name,
                        image_id=job_state.image_identifier(document_path),
                        session_id=session_id,
                    )
                    job_state.mark_complete(job_id, page_subtask, native_uri)

            plans.append(
                {
                    "page_number": analysis.page_number,
                    "route": decision["route"],
                    "native_fidelity": decision["native_fidelity"],
                    "deterministic_review_required": decision[
                        "deterministic_review_required"
                    ],
                    "content_tree_schema_valid": schema_valid,
                    "native_output_uri": native_uri,
                    "visual_semantic_signal": decision["visual_semantic_signal"],
                    "risk_score": decision["risk_score"],
                    "evidence": decision["evidence"],
                    "reasons": decision["reasons"],
                }
            )

        page_plan = {
            "source_format": "pdf",
            "source_document_path": document_path,
            "session_id": session_id,
            "page_count": len(plans),
            "pages": plans,
            "routing_contract": {
                "native_pdf": "native_output_uri is canonical; do not rasterize or call visual specialists",
                "textract_ocr": "rasterize this page only; accept ocr_output_uri only when quality_passed is true",
                "hybrid": "retain native_output_uri as text spine and add targeted visual enrichment",
                "vision_ai": "rasterize this page only and use the visual specialist workflow",
            },
        }
        plan_uri = _put_text(
            s3,
            output_bucket,
            f"{session_id}/native_pdf/page_plan.json",
            json.dumps(page_plan, indent=2),
            "application/json",
        )
        job_state.mark_complete(job_id, subtask, plan_uri)
        return _response(
            200,
            result=json.dumps(page_plan, indent=2),
            page_plan_uri=plan_uri,
            page_count=len(plans),
            success=True,
            session_id=session_id,
        )
    except Exception as error:
        logger.error("Native PDF processing failed: %s", error, exc_info=True)
        failure = str(error)
        if isinstance(error, ValidationError) and error.details.get("code"):
            failure = f"{error.details['code']}: {error.message}"
        job_state.mark_failed(job_id, subtask, failure)
        return create_error_response(error)


def _read_document(document_path: str) -> bytes:
    if document_path.startswith("s3://"):
        bucket, key = _parse_s3_uri(document_path)
        try:
            return bytes(boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read())
        except Exception as error:
            raise ResourceNotFoundError(
                message=f"Failed to download document: {error}",
                details={"document_path": document_path},
            ) from error
    path = Path(document_path)
    if not path.is_file():
        raise ResourceNotFoundError(
            message="Document file not found.",
            details={"document_path": document_path},
        )
    return path.read_bytes()


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parts = uri.removeprefix("s3://").split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise ValidationError(
            message="Invalid S3 document path.",
            details={"document_path": uri},
        )
    return parts[0], parts[1]


def _has_tagged_structure(document: fitz.Document) -> bool:
    """Best-effort tagged-PDF detection; absence is evidence, never a failure."""
    try:
        catalog = document.pdf_catalog()
        return bool(document.xref_get_key(catalog, "StructTreeRoot")[1])
    except Exception:  # malformed PDFs can still have usable text
        return False


def _analyze_page(page: fitz.Page, page_number: int, tagged: bool) -> PageAnalysis:
    rect = page.rect
    analysis = PageAnalysis(
        page_number=page_number,
        page_area=max(rect.width * rect.height, 1.0),
        tagged_pdf_structure=tagged,
    )
    raw = page.get_text("dict", sort=False)
    order = 0
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines: list[str] = []
        font_sizes: list[float] = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(str(span.get("text") or "") for span in spans).strip()
            if line_text:
                lines.append(line_text)
                font_sizes.extend(float(span.get("size") or 0) for span in spans)
        text = "\n".join(lines).strip()
        if not text:
            continue
        bbox_values = block.get("bbox", (0, 0, 0, 0))
        bbox = tuple(float(value) for value in bbox_values)
        analysis.text_blocks.append(
            TextBlock(
                text=text,
                bbox=bbox,
                font_size=median(font_sizes) if font_sizes else 0.0,
                order=order,
            )
        )
        order += 1

    image_rects: list[fitz.Rect] = []
    try:
        for image in page.get_images(full=True):
            image_rects.extend(page.get_image_rects(image[0], transform=False))
    except Exception:
        analysis.warnings.append("image_geometry_unavailable")
    analysis.image_count = len(image_rects)
    analysis.image_area_ratio = min(
        1.0,
        sum(max(image.width * image.height, 0.0) for image in image_rects)
        / analysis.page_area,
    )
    try:
        analysis.vector_count = len(page.get_drawings())
    except Exception:
        analysis.warnings.append("vector_geometry_unavailable")
    analysis.column_count = _detect_columns(analysis.text_blocks, rect.width)
    return analysis


def _detect_columns(blocks: list[TextBlock], page_width: float) -> int:
    """Detect stable left/right text groups; a single wide body block remains one column."""
    if len(blocks) < 4 or page_width <= 0:
        return 1
    narrow = [block for block in blocks if (block.bbox[2] - block.bbox[0]) < page_width * 0.72]
    if len(narrow) < 4:
        return 1
    midpoint = page_width / 2
    left = [block for block in narrow if block.bbox[0] < midpoint]
    right = [block for block in narrow if block.bbox[0] >= midpoint]
    if len(left) < 2 or len(right) < 2:
        return 1
    left_y = {round(block.bbox[1] / 48) for block in left}
    right_y = {round(block.bbox[1] / 48) for block in right}
    return 2 if left_y.intersection(right_y) else 1


def _decide_route(analysis: PageAnalysis) -> dict[str, Any]:
    text = analysis.text
    compact_text = re.sub(r"\s+", "", text)
    printable_ratio = (
        sum(character.isprintable() for character in text) / len(text) if text else 0.0
    )
    text_coverage = min(
        1.0,
        sum(
            max(0.0, block.bbox[2] - block.bbox[0])
            * max(0.0, block.bbox[3] - block.bbox[1])
            for block in analysis.text_blocks
        )
        / analysis.page_area,
    )
    native_text_usable = len(compact_text) >= 24 and printable_ratio >= 0.85
    semantic_visual = bool(_SEMANTIC_VISUAL.search(text))
    table_signal = bool(_TABLE_TOKEN.search(text)) and analysis.vector_count >= 8
    risk = 0
    reasons: list[str] = []

    if analysis.column_count > 1:
        risk += 15
        reasons.append("multiple_text_columns")
    if table_signal:
        risk += 20
        reasons.append("table_caption_with_vector_grid")
    if analysis.image_area_ratio >= 0.60:
        risk += 35
        reasons.append("dominant_embedded_image")
    elif analysis.image_area_ratio >= 0.30:
        risk += 20
        reasons.append("material_embedded_image")
    elif analysis.image_count:
        risk += 8
        reasons.append("embedded_image")
    if analysis.vector_count >= 40:
        risk += 20
        reasons.append("dense_vector_graphics")
    elif analysis.vector_count >= 8 and not table_signal:
        risk += 10
        reasons.append("vector_graphics")
    if semantic_visual:
        risk += 35
        reasons.append("semantic_visual_reference")
    if analysis.warnings:
        risk += 10
        reasons.extend(analysis.warnings)
    risk = min(risk, 100)

    reading_order_confidence = 0.98
    if analysis.column_count > 1:
        reading_order_confidence -= 0.14
    if analysis.vector_count >= 40:
        reading_order_confidence -= 0.10
    if len(analysis.text_blocks) > 35:
        reading_order_confidence -= 0.05
    reading_order_confidence = max(0.0, round(reading_order_confidence, 2))

    if not native_text_usable:
        native_fidelity = "incomplete"
        if semantic_visual or analysis.vector_count >= 40:
            route = "vision_ai"
            reasons.append("native_text_unusable_with_visual_semantic_risk")
        else:
            route = "textract_ocr"
            reasons.append("native_text_unusable")
    elif semantic_visual or risk >= 45:
        native_fidelity = "uncertain"
        route = "hybrid"
    elif reading_order_confidence >= 0.90 and risk < 20:
        native_fidelity = "complete"
        route = "native_pdf"
    else:
        native_fidelity = "uncertain"
        route = "native_pdf"

    return {
        "route": route,
        "native_fidelity": native_fidelity,
        "deterministic_review_required": route == "native_pdf" and (
            risk >= 20 or native_fidelity != "complete"
        ),
        "visual_semantic_signal": semantic_visual,
        "risk_score": risk,
        "reasons": reasons or ["linear_native_text"],
        "evidence": {
            "native_text_characters": len(compact_text),
            "native_text_coverage": round(text_coverage, 3),
            "printable_text_ratio": round(printable_ratio, 3),
            "reading_order_confidence": reading_order_confidence,
            "image_count": analysis.image_count,
            "image_area_ratio": round(analysis.image_area_ratio, 3),
            "vector_count": analysis.vector_count,
            "column_count": analysis.column_count,
            "tagged_pdf_structure": analysis.tagged_pdf_structure,
        },
    }


def _build_native_spine(
    *,
    analysis: PageAnalysis,
    document_path: str,
    session_id: str,
    route: str,
    decision: dict[str, Any],
) -> str:
    root = ET.Element(
        "unified_document",
        {
            "schema_version": "2.0",
            "page": str(analysis.page_number),
            "source_format": "pdf",
            "processing_mode": "native_pdf",
        },
    )
    metadata = ET.SubElement(root, "metadata")
    ET.SubElement(metadata, "source_document").text = document_path
    # Compatibility with existing report/remediation readers. This is intentionally
    # the source PDF rather than a rendered page image.
    ET.SubElement(metadata, "source_image").text = document_path
    ET.SubElement(metadata, "session_id").text = session_id
    ET.SubElement(metadata, "route").text = route
    ET.SubElement(metadata, "native_fidelity").text = decision["native_fidelity"]
    specialists = ET.SubElement(metadata, "specialists_executed")
    ET.SubElement(
        specialists,
        "specialist",
        {"name": "native_pdf_processor", "s3_uri": ""},
    )
    content_tree = ET.SubElement(root, "content_tree")
    section = ET.SubElement(content_tree, "sect", {"id": "sect_root"})
    sizes = [block.font_size for block in analysis.text_blocks if block.font_size > 0]
    body_size = median(sizes) if sizes else 0.0

    for index, block in enumerate(analysis.text_blocks, start=1):
        tag = "P"
        if body_size and block.font_size >= body_size * 1.45:
            tag = "H1" if index == 1 else "H2"
        element = ET.SubElement(
            section,
            "element",
            {
                "id": f"native_p{analysis.page_number:03d}_{index:03d}",
                "tag": tag,
                "order": str(index),
                "page": str(analysis.page_number),
                "bbox": ",".join(f"{value:.1f}" for value in block.bbox),
                "provenance": "native_pdf_text",
            },
        )
        ET.SubElement(element, "text").text = block.text

    if analysis.image_count:
        image_element = ET.SubElement(
            section,
            "element",
            {
                "id": f"native_p{analysis.page_number:03d}_figure",
                "tag": "Figure",
                "order": str(len(analysis.text_blocks) + 1),
                "page": str(analysis.page_number),
                "provenance": "native_pdf_image_object",
            },
        )
        ET.SubElement(image_element, "alt_text").text = (
            f"{analysis.image_count} native PDF image object(s); "
            "visual semantics have not been inferred."
        )

    ET.SubElement(root, "cross_reference_index")
    raw_outputs = ET.SubElement(root, "raw_outputs")
    ET.SubElement(raw_outputs, "output", {"specialist": "native_pdf_processor", "s3_uri": ""})
    ET.SubElement(root, "summary").text = (
        f"Deterministic native PDF extraction for page {analysis.page_number}: "
        f"{len(analysis.text_blocks)} text block(s), {analysis.image_count} image object(s), "
        f"route {route}."
    )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _validate_native_spine(xml: str, page_number: int) -> bool:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return False
    if root.tag != "unified_document" or root.get("schema_version") != "2.0":
        return False
    if root.get("page") != str(page_number):
        return False
    if root.find("./metadata/source_document") is None:
        return False
    tree = root.find("./content_tree/sect[@id='sect_root']")
    if tree is None:
        return False
    for element in tree.findall(".//element"):
        if not element.get("id") or not element.get("tag") or not element.get("page"):
            return False
    return True


def _put_text(s3: Any, bucket: str, key: str, content: str, content_type: str) -> str:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=content.encode("utf-8"),
        ContentType=content_type,
        Tagging="type=native_pdf",
    )
    return f"s3://{bucket}/{key}"


def _response(status_code: int, **body: Any) -> dict[str, Any]:
    return {"statusCode": status_code, "body": json.dumps(body)}
