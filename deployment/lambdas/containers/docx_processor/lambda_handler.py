"""Lambda entry point for page-aware DOCX inspection and adaptive routing."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import boto3

from docx_processor import (
    DocxProcessingError,
    create_page_plan,
    render_docx_to_pdf,
    serializable_page_plan,
)
from foundation import job_state

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGGING_LEVEL", "INFO").upper())

OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "")
SCHEMA_DIRECTORY = Path(os.environ.get("BADGERS_SCHEMA_DIRECTORY", "/var/task/schemas"))


def _get_document_data(document_path: str) -> bytes:
    if document_path.startswith("s3://"):
        bucket_and_key = document_path[5:].split("/", 1)
        if len(bucket_and_key) != 2:
            raise DocxProcessingError("document_path must use the form s3://bucket/key.")
        bucket, key = bucket_and_key
        try:
            return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as error:
            raise DocxProcessingError(f"Unable to download DOCX from S3: {error}") from error
    path = Path(document_path)
    if not path.is_file():
        raise DocxProcessingError("DOCX file was not found.")
    return path.read_bytes()


def _put_text(s3: Any, bucket: str, key: str, value: str) -> str:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=value.encode("utf-8"),
        ContentType="application/json" if key.endswith(".json") else "application/xml",
        ServerSideEncryption="AES256",
    )
    return f"s3://{bucket}/{key}"


def _put_binary(s3: Any, bucket: str, key: str, value: bytes, content_type: str) -> str:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=value,
        ContentType=content_type,
        ServerSideEncryption="AES256",
    )
    return f"s3://{bucket}/{key}"


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """Build a DOCX page plan and write only visual/OCR candidate images to S3."""
    job_id = ""
    subtask = ""
    try:
        body = json.loads(event["body"]) if "body" in event else event
        document_path = body.get("document_path", "")
        if not document_path:
            raise DocxProcessingError("Missing required parameter: document_path.")
        session_id = body.get("session_id") or hashlib.sha256(
            document_path.encode("utf-8")
        ).hexdigest()[:12]
        max_image_size_mb = float(body.get("max_image_size_mb", 4.0))
        if max_image_size_mb <= 0 or max_image_size_mb > 20:
            raise DocxProcessingError("max_image_size_mb must be between 0 and 20.")

        specialist_name = os.environ.get("SPECIALIST_NAME", "docx_processor")
        job_id = body.get("job_id") or ""
        doc_id = body.get("doc_id") or ""
        subtask = job_state.subtask_id(specialist_name, document_path)
        job_state.mark_running(
            job_id,
            subtask,
            doc_id=doc_id,
            specialist=specialist_name,
            image_id=job_state.image_identifier(document_path),
            session_id=session_id,
        )

        payload = _get_document_data(document_path)
        pdf_payload = render_docx_to_pdf(
            payload, int(os.environ.get("DOCX_RENDER_TIMEOUT_SECONDS", "120"))
        )
        page_plan, images = create_page_plan(
            payload,
            pdf_payload,
            session_id,
            SCHEMA_DIRECTORY,
            max_image_size_mb=max_image_size_mb,
        )

        if not OUTPUT_BUCKET:
            raise DocxProcessingError("OUTPUT_BUCKET must be configured for DOCX processing.")
        s3 = boto3.client("s3")
        document_digest = hashlib.sha256(payload).hexdigest()[:16]
        prefix = f"temp/{session_id}/docx/{document_digest}"
        pdf_uri = _put_binary(
            s3, OUTPUT_BUCKET, f"{prefix}/rendered.pdf", pdf_payload, "application/pdf"
        )
        for page in page_plan["pages"]:
            number = page["page"]
            page["native_output_uri"] = _put_text(
                s3,
                OUTPUT_BUCKET,
                f"{prefix}/native/page_{number:03d}.xml",
                page.pop("native_output_xml"),
            )
            if number in images:
                page["image_uri"] = _put_text(
                    s3,
                    OUTPUT_BUCKET,
                    f"{prefix}/images/page_{number:03d}.b64",
                    images[number],
                )
            else:
                page["image_uri"] = None

        page_plan["rendered_pdf_uri"] = pdf_uri
        page_plan_uri = _put_text(
            s3,
            OUTPUT_BUCKET,
            f"{prefix}/page_plan.json",
            serializable_page_plan(page_plan),
        )
        job_state.mark_complete(job_id, subtask, page_plan_uri)
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "result": json.dumps(
                        {
                            "page_plan_uri": page_plan_uri,
                            "rendered_pdf_uri": pdf_uri,
                            "page_count": page_plan["page_count"],
                            "pages": [
                                {
                                    key: page[key]
                                    for key in (
                                        "page",
                                        "route",
                                        "complexity_score",
                                        "decision_reasons",
                                        "content_tree_schema_valid",
                                        "native_output_uri",
                                        "image_uri",
                                    )
                                }
                                for page in page_plan["pages"]
                            ],
                            "session_id": session_id,
                        }
                    ),
                    "success": True,
                    "session_id": session_id,
                }
            ),
        }
    except Exception as error:
        logger.exception("DOCX processing failed")
        job_state.mark_failed(job_id, subtask, str(error))
        return {
            "statusCode": 400 if isinstance(error, DocxProcessingError) else 500,
            "body": json.dumps({"success": False, "error": str(error)}),
        }
