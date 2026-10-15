"""Textract OCR first-pass for image-only DOCX pages.

This utility deliberately uses Textract's synchronous DetectDocumentText API on
one already-rendered JPEG page.  A deterministic quality gate decides whether
the result can become a canonical BADGERS page spine or must be escalated to
the existing Vision workflow.  Textract request failures remain failures; an
OCR quality escalation is a successful, auditable routing decision.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any
from xml.etree import ElementTree as ET

import boto3

from foundation import job_state

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGGING_LEVEL", "INFO").upper())

OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "")
CONTENT_TREE_NS = "https://github.com/aws-samples/sample-badgers/schemas/content-tree"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PAGE_NUMBER = 250
_S3_URI = re.compile(r"^s3://([a-z0-9][a-z0-9.-]{1,61}[a-z0-9])/([^?#]+)$")


class TextractOcrError(ValueError):
    """Raised when a Textract OCR request cannot be safely completed."""


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    match = _S3_URI.fullmatch(uri)
    if not match:
        raise TextractOcrError("image_path must be a valid s3://bucket/key URI.")
    return match.group(1), match.group(2)


def _image_bytes(image_path: str) -> bytes:
    """Load a JPEG/PNG page artifact, decoding BADGERS' .b64 convention."""
    bucket, key = _parse_s3_uri(image_path)
    try:
        response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        payload = bytes(response["Body"].read(MAX_IMAGE_BYTES + 1))
    except Exception as error:
        raise TextractOcrError(f"Unable to read OCR page image from S3: {error}") from error
    if len(payload) > MAX_IMAGE_BYTES:
        raise TextractOcrError("OCR page image exceeds Textract's 5 MiB byte limit.")
    if key.endswith(".b64"):
        try:
            payload = base64.b64decode(payload, validate=True)
        except ValueError as error:
            raise TextractOcrError("OCR page image contains invalid base64.") from error
    if not payload:
        raise TextractOcrError("OCR page image is empty.")
    if len(payload) > MAX_IMAGE_BYTES:
        raise TextractOcrError("Decoded OCR page image exceeds Textract's 5 MiB byte limit.")
    return payload


def _line_blocks(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Return non-empty LINE blocks in visual reading order."""
    lines = [
        block
        for block in response.get("Blocks", [])
        if block.get("BlockType") == "LINE" and str(block.get("Text") or "").strip()
    ]
    return sorted(
        lines,
        key=lambda block: (
            float(block.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0)),
            float(block.get("Geometry", {}).get("BoundingBox", {}).get("Left", 0)),
        ),
    )


def assess_ocr_quality(response: dict[str, Any]) -> dict[str, Any]:
    """Make a conservative deterministic decision about OCR usability.

    This cannot establish that prose is factually correct.  It catches the
    operational failures that make OCR non-authoritative: no lines, too little
    readable text, weak Textract confidence, or text dominated by symbols.
    The visual model handles those pages and receives the OCR artifact as a
    non-authoritative hint through the audit trail.
    """
    lines = _line_blocks(response)
    text = "\n".join(str(block["Text"]).strip() for block in lines)
    confidences = [
        float(block["Confidence"])
        for block in lines
        if isinstance(block.get("Confidence"), (int, float))
    ]
    visible = [character for character in text if not character.isspace()]
    readable = [
        character
        for character in visible
        if character.isalnum() or character in ".,;:!?()[]{}'\"-–—/%&+#@"
    ]
    tokens = re.findall(r"[\w]+", text, flags=re.UNICODE)
    meaningful_tokens = [
        token for token in tokens if any(character.isalnum() for character in token)
    ]
    mean_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    readable_ratio = round(len(readable) / len(visible), 3) if visible else 0.0
    reasons: list[str] = []
    if not lines:
        reasons.append("no_detected_text_lines")
    if len(text.strip()) < 12 or len(meaningful_tokens) < 3:
        reasons.append("insufficient_readable_text")
    if mean_confidence < 75.0:
        reasons.append("low_mean_line_confidence")
    if readable_ratio < 0.70:
        reasons.append("symbol_dominated_text")
    status = "passed" if not reasons else "escalate_to_vision"
    return {
        "status": status,
        "fallback_route": "vision_ai" if status != "passed" else None,
        "reasons": reasons or ["readable_textract_output"],
        "line_count": len(lines),
        "character_count": len(text.strip()),
        "token_count": len(meaningful_tokens),
        "mean_line_confidence": mean_confidence,
        "readable_character_ratio": readable_ratio,
        "text": text,
    }


def _quality(lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Compatibility view for the PDF routing tests and legacy callers.

    The public OCR result uses the richer ``status``-based quality contract.
    This adapter retains the former boolean/score fields without maintaining a
    second, potentially divergent OCR decision algorithm.
    """
    quality = assess_ocr_quality(
        {
            "Blocks": [
                {"BlockType": "LINE", **line}
                for line in lines
            ]
        }
    )
    reasons = list(quality["reasons"])
    if "insufficient_readable_text" in reasons:
        reasons.append("too_little_recovered_text")
    if "low_mean_line_confidence" in reasons:
        reasons.append("low_textract_confidence")
    score = 100
    score -= 45 if "insufficient_readable_text" in quality["reasons"] else 0
    score -= 30 if "low_mean_line_confidence" in quality["reasons"] else 0
    score -= 25 if "symbol_dominated_text" in quality["reasons"] else 0
    score -= 20 if "no_detected_text_lines" in quality["reasons"] else 0
    return {
        "quality_passed": quality["status"] == "passed",
        "quality_score": max(score, 0),
        "reasons": reasons,
        "text_characters": quality["character_count"],
        "line_count": quality["line_count"],
        "mean_confidence": quality["mean_line_confidence"],
    }


def build_content_tree_xml(
    *,
    page_number: int,
    session_id: str,
    source_image: str,
    output_uri: str,
    response: dict[str, Any],
    quality: dict[str, Any],
) -> bytes:
    """Create a schema-valid BADGERS spine directly from accepted LINE blocks."""
    ET.register_namespace("", CONTENT_TREE_NS)
    tag = lambda name: f"{{{CONTENT_TREE_NS}}}{name}"  # noqa: E731
    root = ET.Element(
        tag("unified_document"),
        {
            "session_id": session_id,
            "page": str(page_number),
            "schema_version": "2.0",
        },
    )
    metadata = ET.SubElement(root, tag("metadata"))
    ET.SubElement(metadata, tag("source_image")).text = source_image
    ET.SubElement(metadata, tag("correlation_timestamp")).text = datetime.now(UTC).isoformat()
    executed = ET.SubElement(metadata, tag("specialists_executed"))
    ET.SubElement(
        executed,
        tag("specialist"),
        {"name": "textract_ocr", "s3_uri": output_uri},
    )
    content_tree = ET.SubElement(root, tag("content_tree"))
    section = ET.SubElement(content_tree, tag("sect"), {"id": f"textract-page-{page_number}"})
    for order, block in enumerate(_line_blocks(response), start=1):
        element = ET.SubElement(
            section,
            tag("element"),
            {
                "id": f"textract-page-{page_number}-line-{order}",
                "tag": "P",
                "page": str(page_number),
                "order": str(order),
                "placement": "Block",
            },
        )
        ET.SubElement(element, tag("text")).text = str(block["Text"]).strip()
    ET.SubElement(root, tag("summary")).text = (
        f"Textract OCR extracted {quality['line_count']} readable line(s) "
        f"with mean line confidence {quality['mean_line_confidence']:.2f}."
    )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def validate_content_tree(xml_payload: bytes) -> tuple[bool, list[str]]:
    """Validate the constrained Textract output against the BADGERS contract.

    Textract emits only a deliberately narrow subset of content-tree v2:
    metadata, a single section, and block-level P elements.  Checking that
    subset directly avoids adding the 5+ MiB lxml binary to every specialist
    Lambda layer merely to validate a document this function itself creates.
    The richer DOCX native path continues to perform full XSD validation in
    its dedicated container.
    """
    def local_name(element: ET.Element) -> str:
        return element.tag.rsplit("}", 1)[-1]

    errors: list[str] = []
    try:
        root = ET.fromstring(xml_payload)
    except ET.ParseError as error:
        return False, [f"Invalid XML: {error}"]
    if root.tag != f"{{{CONTENT_TREE_NS}}}unified_document":
        errors.append("Root must be the namespaced unified_document element.")
    if not root.get("session_id") or not root.get("schema_version"):
        errors.append("Root requires session_id and schema_version.")
    try:
        if int(root.get("page", "0")) < 1:
            errors.append("Root page must be a positive integer.")
    except ValueError:
        errors.append("Root page must be a positive integer.")
    children = list(root)
    if [local_name(element) for element in children[:2]] != ["metadata", "content_tree"]:
        errors.append("Root requires metadata followed by content_tree.")
        return False, errors
    metadata, content_tree = children[:2]
    metadata_children = list(metadata)
    if [local_name(element) for element in metadata_children] != [
        "source_image",
        "correlation_timestamp",
        "specialists_executed",
    ]:
        errors.append("Metadata does not match the BADGERS content-tree contract.")
    elif not (metadata_children[0].text or "").strip():
        errors.append("Metadata source_image cannot be empty.")
    else:
        specialists = list(metadata_children[2])
        if len(specialists) != 1 or local_name(specialists[0]) != "specialist":
            errors.append("Metadata requires exactly one Textract specialist entry.")
        elif (
            specialists[0].get("name") != "textract_ocr"
            or not specialists[0].get("s3_uri")
        ):
            errors.append("Textract specialist entry is incomplete.")
    sections = list(content_tree)
    if len(sections) != 1 or local_name(sections[0]) != "sect" or not sections[0].get("id"):
        errors.append("Content tree requires exactly one identified section.")
        return False, errors
    for element in list(sections[0]):
        if local_name(element) != "element":
            errors.append("Textract content section may contain only elements.")
            continue
        if element.get("tag") != "P" or not element.get("id"):
            errors.append("Textract element requires id and P tag.")
        if element.get("placement") != "Block":
            errors.append("Textract element placement must be Block.")
        if not any(local_name(child) == "text" for child in list(element)):
            errors.append("Textract element requires text.")
    return not errors, errors


def _put_json(s3: Any, bucket: str, key: str, value: dict[str, Any]) -> str:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )
    return f"s3://{bucket}/{key}"


def _put_xml(s3: Any, bucket: str, key: str, value: bytes) -> str:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=value,
        ContentType="application/xml; charset=utf-8",
        ServerSideEncryption="AES256",
    )
    return f"s3://{bucket}/{key}"


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """Run Textract first-pass OCR and return pass-through or Vision escalation."""
    job_id = ""
    subtask = ""
    try:
        body = json.loads(event["body"]) if "body" in event else event
        image_path = str(body.get("image_path") or "").strip()
        if not image_path:
            raise TextractOcrError("Missing required parameter: image_path.")
        page_number = int(body.get("page_number") or 0)
        if page_number < 1 or page_number > MAX_PAGE_NUMBER:
            raise TextractOcrError(f"page_number must be between 1 and {MAX_PAGE_NUMBER}.")
        session_id = str(body.get("session_id") or "no_session")
        if not OUTPUT_BUCKET:
            raise TextractOcrError("OUTPUT_BUCKET must be configured for Textract OCR.")

        specialist_name = os.environ.get("SPECIALIST_NAME", "textract_ocr")
        job_id = str(body.get("job_id") or "")
        doc_id = str(body.get("doc_id") or "")
        subtask = job_state.subtask_id(specialist_name, image_path)
        job_state.mark_running(
            job_id,
            subtask,
            doc_id=doc_id,
            specialist=specialist_name,
            image_id=job_state.image_identifier(image_path),
            session_id=session_id,
        )

        image = _image_bytes(image_path)
        response = boto3.client("textract").detect_document_text(Document={"Bytes": image})
        quality = assess_ocr_quality(response)
        digest = hashlib.sha256(image).hexdigest()[:16]
        prefix = f"temp/{session_id}/textract-ocr/{digest}"
        s3 = boto3.client("s3")
        content_tree_uri = None
        schema_errors: list[str] = []
        if quality["status"] == "passed":
            xml_key = f"{prefix}/page_{page_number:03d}.xml"
            content_tree_uri = f"s3://{OUTPUT_BUCKET}/{xml_key}"
            content_tree = build_content_tree_xml(
                page_number=page_number,
                session_id=session_id,
                source_image=image_path,
                output_uri=content_tree_uri,
                response=response,
                quality=quality,
            )
            schema_valid, schema_errors = validate_content_tree(content_tree)
            if not schema_valid:
                quality = {
                    **quality,
                    "status": "escalate_to_vision",
                    "fallback_route": "vision_ai",
                    "reasons": [*quality["reasons"], "content_tree_schema_validation_failed"],
                }
                content_tree_uri = None
            else:
                _put_xml(s3, OUTPUT_BUCKET, xml_key, content_tree)
        else:
            schema_valid = False

        # Persist the final gate decision with the raw Textract blocks.  A
        # schema-validation rejection may change a previously passing result
        # into a Vision escalation, so this must be written after validation.
        ocr_result_uri = _put_json(
            s3,
            OUTPUT_BUCKET,
            f"{prefix}/page_{page_number:03d}.json",
            {
                "source_image": image_path,
                "page_number": page_number,
                "quality": quality,
                "content_tree_schema_valid": schema_valid,
                "content_tree_schema_errors": schema_errors,
                "textract_blocks": response.get("Blocks", []),
            },
        )
        result = {
            "source_image": image_path,
            "page_number": page_number,
            "ocr_result_uri": ocr_result_uri,
            "content_tree_uri": content_tree_uri,
            "content_tree_schema_valid": schema_valid,
            "content_tree_schema_errors": schema_errors,
            "quality": quality,
            "next_route": "native_complete" if content_tree_uri else "vision_ai",
        }
        job_state.mark_complete(job_id, subtask, content_tree_uri or ocr_result_uri)
        return {
            "statusCode": 200,
            "body": json.dumps({"success": True, "result": json.dumps(result)}),
        }
    except Exception as error:
        logger.exception("Textract OCR failed")
        job_state.mark_failed(job_id, subtask, str(error))
        return {
            "statusCode": 400 if isinstance(error, TextractOcrError) else 500,
            "body": json.dumps({"success": False, "error": str(error)}),
        }
