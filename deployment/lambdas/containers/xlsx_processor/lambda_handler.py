"""Lambda entry point for safe XLSX analysis and conditional Vision enrichment."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import boto3

from foundation import job_state, operation_state
from visual_analysis import (
    VisualAnalysisError,
    analyze_workbook_visuals,
    load_deployed_visual_configuration,
    prepare_visual_analysis,
)
from workbook_inspector import XlsxProcessingError, inspect_xlsx
from xlsx_content_tree import build_xlsx_content_tree_xml


logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGGING_LEVEL", "INFO").upper())

OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "")
CONFIG_BUCKET = os.environ.get("CONFIG_BUCKET", "")


def _get_document_data(document_path: str) -> bytes:
    if document_path.startswith("s3://"):
        bucket_and_key = document_path[5:].split("/", 1)
        if len(bucket_and_key) != 2:
            raise XlsxProcessingError("document_path must use the form s3://bucket/key.")
        bucket, key = bucket_and_key
        try:
            return bytes(boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read())
        except Exception as error:
            raise XlsxProcessingError("Unable to download XLSX from S3.") from error
    path = Path(document_path)
    if not path.is_file():
        raise XlsxProcessingError("XLSX file was not found.")
    return path.read_bytes()


def _put_json(s3: Any, bucket: str, key: str, value: dict[str, Any]) -> str:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(value, indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )
    return f"s3://{bucket}/{key}"


def _put_xml(s3: Any, bucket: str, key: str, value: bytes) -> str:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=value,
        ContentType="application/xml",
        ServerSideEncryption="AES256",
    )
    return f"s3://{bucket}/{key}"


def _artifact_prefix(
    session_id: str, document_digest: str, job_id: str, owner_sub: str
) -> str:
    """Use the workflow-owned private prefix when this is a durable operation."""
    if job_id.startswith("op_") and owner_sub:
        return f"{operation_state.operation_prefix(owner_sub, job_id)}/xlsx/{document_digest}"
    return f"temp/{session_id}/xlsx/{document_digest}"


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """Inspect an XLSX package without calculating or executing its contents."""

    job_id = ""
    subtask = ""
    try:
        body = json.loads(event["body"]) if "body" in event else event
        document_path = str(body.get("document_path") or "").strip()
        if not document_path:
            raise XlsxProcessingError("Missing required parameter: document_path.")
        if not document_path.lower().split("?", 1)[0].split("#", 1)[0].endswith(".xlsx"):
            raise XlsxProcessingError("xlsx_processor accepts .xlsx files only.")
        if not OUTPUT_BUCKET:
            raise XlsxProcessingError("OUTPUT_BUCKET must be configured for XLSX processing.")

        specialist = os.environ.get("SPECIALIST_NAME", "xlsx_processor")
        job_id = str(body.get("job_id") or "")
        doc_id = str(body.get("doc_id") or "")
        session_id = str(body.get("session_id") or "")
        if not session_id:
            session_id = hashlib.sha256(document_path.encode("utf-8")).hexdigest()[:12]
        subtask = job_state.subtask_id(specialist, document_path)
        job_state.mark_running(
            job_id,
            subtask,
            doc_id=doc_id,
            specialist=specialist,
            image_id=job_state.image_identifier(document_path),
            session_id=session_id,
        )

        payload = _get_document_data(document_path)
        inspection = inspect_xlsx(payload)
        prepared_visuals, skipped_visuals = prepare_visual_analysis(payload, inspection)
        if prepared_visuals:
            try:
                system_prompt, model_config = load_deployed_visual_configuration(
                    CONFIG_BUCKET, specialist
                )
                visual_analysis = analyze_workbook_visuals(
                    payload,
                    inspection,
                    system_prompt=system_prompt,
                    model_config=model_config,
                )
            except VisualAnalysisError as error:
                logger.warning("XLSX Vision enrichment was unavailable: %s", error)
                visual_analysis = {
                    "schema_version": "badgers-xlsx-visual-analysis/v1",
                    "status": "failed",
                    "analyses": [],
                    "skipped_objects": skipped_visuals,
                    "failure_reason": str(error),
                    "processing_guarantees": {
                        "workbook_execution": "not_performed",
                        "formula_calculation": "not_performed",
                        "native_cells_authoritative": True,
                        "vision_scope": "embedded_raster_visual_evidence_only",
                    },
                }
                inspection["warnings"].append("vision_analysis_unavailable")
        else:
            visual_analysis = {
                "schema_version": "badgers-xlsx-visual-analysis/v1",
                "status": "not_required",
                "analyses": [],
                "skipped_objects": skipped_visuals,
                "processing_guarantees": {
                    "workbook_execution": "not_performed",
                    "formula_calculation": "not_performed",
                    "native_cells_authoritative": True,
                    "vision_scope": "embedded_raster_visual_evidence_only",
                },
            }
        inspection["visual_analysis"] = visual_analysis
        document_digest = hashlib.sha256(payload).hexdigest()[:16]
        prefix = _artifact_prefix(
            session_id, document_digest, job_id, str(body.get("owner_sub") or "")
        )
        inspection.update(
            {
                "source_document_path": document_path,
                "session_id": session_id,
                "processing_guarantees": {
                    "formula_calculation": "not_performed",
                    "macro_or_script_execution": "not_performed",
                    "external_link_resolution": "not_performed",
                    "connection_refresh": "not_performed",
                    "vision_analysis": (
                        "completed_by_xlsx_processor"
                        if visual_analysis["status"] == "complete"
                        else visual_analysis["status"]
                    ),
                },
            }
        )
        workbook_analysis_uri = _put_json(
            boto3.client("s3"), OUTPUT_BUCKET, f"{prefix}/workbook-analysis.json", inspection
        )
        content_tree_uri = _put_xml(
            boto3.client("s3"),
            OUTPUT_BUCKET,
            f"{prefix}/content-tree.xml",
            build_xlsx_content_tree_xml(
                inspection,
                source_document=document_path,
                session_id=session_id,
            ),
        )
        job_state.mark_complete(job_id, subtask, content_tree_uri)
        result = {
            "artifact_uri": content_tree_uri,
            "content_tree_uri": content_tree_uri,
            "workbook_analysis_uri": workbook_analysis_uri,
            "source_format": "xlsx",
            "sheet_count": inspection["workbook"]["sheet_count"],
            "warnings": inspection["warnings"],
            "vision_analysis_status": visual_analysis["status"],
            "session_id": session_id,
        }
        return {
            "statusCode": 200,
            "body": json.dumps({"result": json.dumps(result), "success": True, "session_id": session_id}),
        }
    except Exception as error:
        logger.error("XLSX processing failed: %s", error, exc_info=True)
        job_state.mark_failed(job_id, subtask, str(error))
        status = 400 if isinstance(error, (XlsxProcessingError, VisualAnalysisError)) else 500
        return {
            "statusCode": status,
            "body": json.dumps(
                {
                    "success": False,
                    "error_type": "XLSX_INTAKE_FAILED",
                    "message": str(error),
                }
            ),
        }
