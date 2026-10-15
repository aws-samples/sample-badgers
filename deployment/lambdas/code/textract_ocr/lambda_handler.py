"""Textract OCR fallback for pages whose PDF native text is unusable."""

from __future__ import annotations

import base64
import json
import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import boto3

from foundation import job_state
from foundation.lambda_error_handler import ValidationError, create_error_response

logger = logging.getLogger()
logger.setLevel(getattr(logging, os.environ.get("LOGGING_LEVEL", "INFO").upper()))


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:  # noqa: ARG001
    """Run synchronous Textract OCR on one previously-rendered PDF page."""
    job_id = ""
    subtask = ""
    try:
        body = json.loads(event["body"]) if "body" in event else event
        image_path = str(body.get("image_path") or "").strip()
        page_number = int(body.get("page_number") or 0)
        source_document_path = str(body.get("source_document_path") or "").strip()
        session_id = str(body.get("session_id") or "no_session")
        job_id = str(body.get("job_id") or "")
        doc_id = str(body.get("doc_id") or "")
        specialist_name = os.environ.get("SPECIALIST_NAME", "textract_ocr")
        if not image_path or page_number < 1 or not source_document_path:
            raise ValidationError(
                message="image_path, source_document_path, and positive page_number are required.",
                details={},
            )

        subtask = job_state.subtask_id(specialist_name, image_path)
        job_state.mark_running(
            job_id,
            subtask,
            doc_id=doc_id,
            specialist=specialist_name,
            image_id=job_state.image_identifier(image_path),
            session_id=session_id,
        )
        image_bytes = _read_image(image_path)
        response = boto3.client("textract").detect_document_text(
            Document={"Bytes": image_bytes}
        )
        lines = [
            block
            for block in response.get("Blocks", [])
            if block.get("BlockType") == "LINE" and str(block.get("Text") or "").strip()
        ]
        quality = _quality(lines)
        xml = _build_spine(
            source_document_path=source_document_path,
            image_path=image_path,
            page_number=page_number,
            session_id=session_id,
            lines=lines,
            quality=quality,
        )
        output_bucket = os.environ.get("OUTPUT_BUCKET", "")
        if not output_bucket:
            raise RuntimeError("OUTPUT_BUCKET is not configured")
        key = f"{session_id}/textract_ocr/page_{page_number:03d}.xml"
        boto3.client("s3").put_object(
            Bucket=output_bucket,
            Key=key,
            Body=xml.encode("utf-8"),
            ContentType="application/xml",
            Tagging=f"session_id={session_id}&type=textract_ocr&page={page_number}",
        )
        output_uri = f"s3://{output_bucket}/{key}"
        job_state.mark_complete(job_id, subtask, output_uri)
        result = {
            "page_number": page_number,
            "quality_passed": quality["quality_passed"],
            "quality_score": quality["quality_score"],
            "quality_reasons": quality["reasons"],
            "ocr_output_uri": output_uri,
            "canonical_output_uri": output_uri if quality["quality_passed"] else "",
            "escalate_to_vision": not quality["quality_passed"],
        }
        return {
            "statusCode": 200,
            "body": json.dumps(
                {"result": json.dumps(result, indent=2), "success": True, "session_id": session_id}
            ),
        }
    except Exception as error:
        logger.error("Textract OCR failed: %s", error, exc_info=True)
        job_state.mark_failed(job_id, subtask, str(error))
        return create_error_response(error)


def _read_image(image_path: str) -> bytes:
    if not image_path.startswith("s3://"):
        raise ValidationError(
            message="Textract OCR requires an S3 page image path.",
            details={"image_path": image_path},
        )
    bucket, key = image_path.removeprefix("s3://").split("/", 1)
    body = bytes(boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read())
    return base64.b64decode(body, validate=True) if key.endswith(".b64") else body


def _quality(lines: list[dict[str, Any]]) -> dict[str, Any]:
    text = "\n".join(str(line.get("Text") or "").strip() for line in lines)
    compact = "".join(text.split())
    printable = sum(char.isprintable() for char in text) / len(text) if text else 0.0
    average_line_length = len(compact) / len(lines) if lines else 0.0
    score = 0
    reasons: list[str] = []
    if len(compact) >= 40:
        score += 45
    else:
        reasons.append("too_little_recovered_text")
    if printable >= 0.90:
        score += 25
    else:
        reasons.append("non_printable_or_corrupt_text")
    if len(lines) >= 2 and average_line_length >= 4:
        score += 20
    else:
        reasons.append("insufficient_coherent_lines")
    confidences = [float(line.get("Confidence") or 0) for line in lines]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    if mean_confidence >= 85:
        score += 10
    else:
        reasons.append("low_textract_confidence")
    return {
        "quality_score": score,
        "quality_passed": score >= 80,
        "reasons": reasons or ["coherent_text_recovered"],
        "text_characters": len(compact),
        "line_count": len(lines),
        "mean_confidence": round(mean_confidence, 1),
    }


def _build_spine(
    *,
    source_document_path: str,
    image_path: str,
    page_number: int,
    session_id: str,
    lines: list[dict[str, Any]],
    quality: dict[str, Any],
) -> str:
    root = ET.Element(
        "unified_document",
        {
            "schema_version": "2.0",
            "page": str(page_number),
            "source_format": "pdf",
            "processing_mode": "textract_ocr",
        },
    )
    metadata = ET.SubElement(root, "metadata")
    ET.SubElement(metadata, "source_document").text = source_document_path
    ET.SubElement(metadata, "source_image").text = image_path
    ET.SubElement(metadata, "session_id").text = session_id
    ET.SubElement(metadata, "ocr_quality_score").text = str(quality["quality_score"])
    executed = ET.SubElement(metadata, "specialists_executed")
    ET.SubElement(executed, "specialist", {"name": "textract_ocr", "s3_uri": ""})
    tree = ET.SubElement(root, "content_tree")
    section = ET.SubElement(tree, "sect", {"id": "sect_root"})
    for index, line in enumerate(lines, start=1):
        geometry = line.get("Geometry", {}).get("BoundingBox", {})
        bbox = ",".join(
            str(round(float(geometry.get(key, 0.0)), 5))
            for key in ("Left", "Top", "Width", "Height")
        )
        element = ET.SubElement(
            section,
            "element",
            {
                "id": f"ocr_p{page_number:03d}_{index:03d}",
                "tag": "P",
                "order": str(index),
                "page": str(page_number),
                "bbox_normalized": bbox,
                "provenance": "textract_ocr",
            },
        )
        ET.SubElement(element, "text").text = str(line.get("Text") or "")
    ET.SubElement(root, "cross_reference_index")
    outputs = ET.SubElement(root, "raw_outputs")
    ET.SubElement(outputs, "output", {"specialist": "textract_ocr", "s3_uri": ""})
    ET.SubElement(root, "summary").text = (
        f"Textract OCR page {page_number}; quality score {quality['quality_score']}/100."
    )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)
