"""Aggregate Analysis Results Lambda with AgentCore Gateway session tracking."""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime

import boto3

from foundation.lambda_error_handler import (
    create_error_response,
    ValidationError,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """Aggregate analysis results from multiple tasks."""
    try:
        # Log Gateway context information
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

        # Parse input - AgentCore Gateway passes parameters directly in event
        body = json.loads(event["body"]) if "body" in event else event

        # Extract and log session_id from AgentCore Runtime
        session_id = body.get("session_id", "no_session")
        logger.info("Processing request for runtime session_id: %s", session_id)

        # Extract audit_mode flag
        audit_mode = body.get("audit_mode", False)
        if audit_mode:
            logger.info("Audit mode enabled - confidence assessment will be included")

        execution_results_str = body.get("execution_results")
        pdf_name = body.get("pdf_name")

        if not execution_results_str:
            raise ValidationError(
                message="Missing required parameter: execution_results",
                details={"provided_keys": list(body.keys())},
            )

        if not pdf_name:
            raise ValidationError(
                message="Missing required parameter: pdf_name",
                details={"provided_keys": list(body.keys())},
            )

        execution_results = json.loads(execution_results_str)
        if not isinstance(execution_results, list):
            raise ValidationError(
                message="execution_results must be a JSON array",
                details={"type": type(execution_results).__name__},
            )

        # Aggregate results by page
        aggregated = _aggregate_by_page(execution_results, pdf_name)

        # Build session metadata (no LLM cost - pure counting)
        session_metadata = _build_session_metadata(
            execution_results, pdf_name, session_id
        )

        logger.info(
            "Aggregated results for %s: %d pages (session: %s)",
            pdf_name,
            len(aggregated.get("pages", [])),
            session_id,
        )

        # Save aggregated results to S3
        output_bucket = os.environ.get("OUTPUT_BUCKET")
        s3_uris = []
        if output_bucket:
            s3_uris = _save_aggregated_to_s3(
                aggregated, output_bucket, session_id, pdf_name, session_metadata
            )
            logger.info("Saved %d aggregated files to S3", len(s3_uris))

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "result": json.dumps(aggregated),
                    "success": True,
                    "session_id": session_id,
                    "s3_uris": s3_uris,
                }
            ),
        }

    except Exception as e:
        return create_error_response(e)


def _aggregate_by_page(execution_results: list, pdf_name: str) -> dict:
    """Aggregate analysis results organized by page."""
    pages = {}

    for result in execution_results:
        page_num = result.get("page", 0)
        tool_name = result.get("tool", "unknown")
        # Support both "result" (standard analyzers) and "classification" (classify_pdf_content)
        analysis = result.get("result") or result.get("classification", "")
        success = result.get("success", True if analysis else False)

        if page_num not in pages:
            pages[page_num] = {"page": page_num, "analyses": []}

        pages[page_num]["analyses"].append(
            {"tool": tool_name, "result": analysis, "success": success}
        )

    # Convert to sorted list
    page_list = sorted(pages.values(), key=lambda x: x["page"])

    return {"pdf_name": pdf_name, "total_pages": len(page_list), "pages": page_list}


def _build_session_metadata(
    execution_results: list, pdf_name: str, session_id: str
) -> dict:
    """Build session metadata from execution results - no LLM cost, pure counting."""
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Track tool invocations and pages
    tool_pages: dict[str, list[int]] = defaultdict(list)
    tool_success_count: dict[str, int] = defaultdict(int)
    tool_failure_count: dict[str, int] = defaultdict(int)
    pages_with_content: set[int] = set()

    for result in execution_results:
        page_num = result.get("page", 0)
        tool_name = result.get("tool", "unknown")
        success = result.get("success", False)

        tool_pages[tool_name].append(page_num)
        if success:
            tool_success_count[tool_name] += 1
        else:
            tool_failure_count[tool_name] += 1
        pages_with_content.add(page_num)

    # Build content summary with counts per tool
    content_summary = {}
    for tool_name, pages in tool_pages.items():
        content_summary[tool_name] = {
            "count": len(pages),
            "pages": sorted(pages),
            "successful": tool_success_count[tool_name],
            "failed": tool_failure_count[tool_name],
        }

    # Calculate totals
    total_analyses = sum(len(pages) for pages in tool_pages.values())
    total_successful = sum(tool_success_count.values())
    total_failed = sum(tool_failure_count.values())

    return {
        "session_id": session_id,
        "timestamp_completed": timestamp,
        "input_file": {
            "name": pdf_name,
            "type": "pdf",
        },
        "analyzers_invoked": sorted(tool_pages.keys()),
        "content_summary": content_summary,
        "stats": {
            "total_analyses_performed": total_analyses,
            "successful_analyses": total_successful,
            "failed_analyses": total_failed,
            "pages_with_content": len(pages_with_content),
        },
    }


def _save_aggregated_to_s3(
    aggregated: dict,
    output_bucket: str,
    session_id: str,
    pdf_name: str,
    session_metadata: dict,
) -> list:
    """
    Save aggregated results to S3.

    Creates:
    - Per-page combined files: aggregated/page_{n}_combined.xml
    - Master file: aggregated/all_pages_combined.xml

    Returns list of S3 URIs created.
    """
    s3 = boto3.client("s3")
    s3_uris = []
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Save per-page combined results
    for page_data in aggregated.get("pages", []):
        page_num = page_data.get("page", 0)
        page_xml = _format_page_as_xml(page_data, pdf_name)

        s3_key = f"{session_id}/aggregated/page_{page_num}_combined_{timestamp}.xml"
        s3.put_object(
            Bucket=output_bucket,
            Key=s3_key,
            Body=page_xml.encode("utf-8"),
            ContentType="application/xml",
            Tagging=f"session_id={session_id}&type=page_combined&page={page_num}",
        )
        s3_uri = f"s3://{output_bucket}/{s3_key}"
        s3_uris.append(s3_uri)
        logger.info("Saved page %d combined to %s", page_num, s3_uri)

    # Save master file with all pages
    master_xml = _format_all_pages_as_xml(aggregated)
    master_key = f"{session_id}/aggregated/all_pages_combined_{timestamp}.xml"
    s3.put_object(
        Bucket=output_bucket,
        Key=master_key,
        Body=master_xml.encode("utf-8"),
        ContentType="application/xml",
        Tagging=f"session_id={session_id}&type=all_pages_combined",
    )
    master_uri = f"s3://{output_bucket}/{master_key}"
    s3_uris.append(master_uri)
    logger.info("Saved all pages combined to %s", master_uri)

    # Save session metadata JSON
    try:
        metadata_key = f"{session_id}/session_metadata.json"
        s3.put_object(
            Bucket=output_bucket,
            Key=metadata_key,
            Body=json.dumps(session_metadata, indent=2).encode("utf-8"),
            ContentType="application/json",
            Tagging=f"session_id={session_id}&type=session_metadata",
        )
        metadata_uri = f"s3://{output_bucket}/{metadata_key}"
        s3_uris.append(metadata_uri)
        logger.info("Saved session metadata to %s", metadata_uri)
    except Exception as e:
        # Metadata is non-critical - log but don't fail aggregation
        logger.warning("Failed to save session metadata: %s", e)

    return s3_uris


def _format_page_as_xml(page_data: dict, pdf_name: str) -> str:
    """Format a single page's aggregated results as XML."""
    page_num = page_data.get("page", 0)
    analyses = page_data.get("analyses", [])

    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<page_combined_results pdf="{pdf_name}" page="{page_num}">',
    ]

    for analysis in analyses:
        tool = analysis.get("tool", "unknown")
        success = analysis.get("success", False)
        result = analysis.get("result", "")

        xml_parts.append(f'  <analysis tool="{tool}" success="{str(success).lower()}">')
        xml_parts.append(f"    <result><![CDATA[{result}]]></result>")
        xml_parts.append("  </analysis>")

    xml_parts.append("</page_combined_results>")
    return "\n".join(xml_parts)


def _format_all_pages_as_xml(aggregated: dict) -> str:
    """Format all pages' aggregated results as XML."""
    pdf_name = aggregated.get("pdf_name", "unknown")
    total_pages = aggregated.get("total_pages", 0)
    pages = aggregated.get("pages", [])

    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<document_combined_results pdf="{pdf_name}" total_pages="{total_pages}">',
    ]

    for page_data in pages:
        page_num = page_data.get("page", 0)
        analyses = page_data.get("analyses", [])

        xml_parts.append(f'  <page number="{page_num}">')

        for analysis in analyses:
            tool = analysis.get("tool", "unknown")
            success = analysis.get("success", False)
            result = analysis.get("result", "")

            xml_parts.append(
                f'    <analysis tool="{tool}" success="{str(success).lower()}">'
            )
            xml_parts.append(f"      <result><![CDATA[{result}]]></result>")
            xml_parts.append("    </analysis>")

        xml_parts.append("  </page>")

    xml_parts.append("</document_combined_results>")
    return "\n".join(xml_parts)
