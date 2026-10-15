"""Textract OCR quality-gated fallback for a selected rendered presentation slide."""

import base64
import json
import logging
import os
import xml.etree.ElementTree as ET

import boto3
from foundation import job_state

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):  # noqa: ARG001
    job_id = subtask = ""
    try:
        body = json.loads(event["body"]) if "body" in event else event
        image_path = str(body.get("image_path") or "")
        source_document_path = str(body.get("source_document_path") or "")
        page_number = int(body.get("page_number") or 0)
        session_id = str(body.get("session_id") or "no_session")
        job_id, doc_id = str(body.get("job_id") or ""), str(body.get("doc_id") or "")
        if not image_path.startswith("s3://") or not source_document_path or page_number < 1:
            raise ValueError("image_path (S3), source_document_path, and positive page_number are required")
        subtask = job_state.subtask_id("textract_ocr", image_path)
        job_state.mark_running(job_id, subtask, doc_id=doc_id, specialist="textract_ocr",
                               image_id=job_state.image_identifier(image_path), session_id=session_id)
        bucket, key = image_path.removeprefix("s3://").split("/", 1)
        image = bytes(boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read())
        if key.endswith(".b64"):
            image = base64.b64decode(image, validate=True)
        lines = [block for block in boto3.client("textract").detect_document_text(
            Document={"Bytes": image}).get("Blocks", [])
            if block.get("BlockType") == "LINE" and str(block.get("Text") or "").strip()]
        quality = _quality(lines)
        xml = _spine(source_document_path, image_path, page_number, session_id, lines, quality)
        output_bucket = os.environ["OUTPUT_BUCKET"]
        out_key = f"{session_id}/textract_ocr/page_{page_number:03d}.xml"
        boto3.client("s3").put_object(Bucket=output_bucket, Key=out_key, Body=xml.encode(),
                                      ContentType="application/xml")
        uri = f"s3://{output_bucket}/{out_key}"
        job_state.mark_complete(job_id, subtask, uri)
        result = {"page_number": page_number, **quality, "ocr_output_uri": uri,
                  "canonical_output_uri": uri if quality["quality_passed"] else "",
                  "escalate_to_vision": not quality["quality_passed"]}
        return {"statusCode": 200, "body": json.dumps({"result": json.dumps(result), "success": True, "session_id": session_id})}
    except Exception as error:
        job_state.mark_failed(job_id, subtask, str(error))
        logger.error("Textract OCR failed", exc_info=True)
        return {"statusCode": 400, "body": json.dumps({"error_type": "TEXTRACT_OCR_FAILED", "message": str(error)})}


def _quality(lines):
    text = "\n".join(str(line.get("Text") or "") for line in lines)
    compact = "".join(text.split())
    confidences = [float(line.get("Confidence") or 0) for line in lines]
    confidence = sum(confidences) / len(confidences) if confidences else 0
    score = (45 if len(compact) >= 40 else 0) + (25 if text and sum(c.isprintable() for c in text) / len(text) >= .9 else 0) + (20 if len(lines) >= 2 and len(compact) / len(lines) >= 4 else 0) + (10 if confidence >= 85 else 0)
    return {"quality_score": score, "quality_passed": score >= 80,
            "text_characters": len(compact), "line_count": len(lines),
            "mean_confidence": round(confidence, 1)}


def _spine(source, image, page, session, lines, quality):
    root = ET.Element("unified_document", {"schema_version": "2.0", "page": str(page),
                     "source_format": "pptx", "processing_mode": "textract_ocr"})
    metadata = ET.SubElement(root, "metadata")
    ET.SubElement(metadata, "source_document").text = source
    ET.SubElement(metadata, "source_image").text = image
    ET.SubElement(metadata, "session_id").text = session
    ET.SubElement(metadata, "ocr_quality_score").text = str(quality["quality_score"])
    ET.SubElement(ET.SubElement(metadata, "specialists_executed"), "specialist",
                  {"name": "textract_ocr", "s3_uri": ""})
    section = ET.SubElement(ET.SubElement(root, "content_tree"), "sect", {"id": "sect_root"})
    for index, line in enumerate(lines, 1):
        element = ET.SubElement(section, "element", {"id": f"ocr_s{page:03d}_{index:03d}",
            "tag": "P", "order": str(index), "page": str(page), "provenance": "textract_ocr"})
        ET.SubElement(element, "text").text = str(line.get("Text") or "")
    ET.SubElement(root, "cross_reference_index")
    ET.SubElement(root, "raw_outputs")
    ET.SubElement(root, "summary").text = f"Textract OCR slide {page}; quality {quality['quality_score']}/100."
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)
