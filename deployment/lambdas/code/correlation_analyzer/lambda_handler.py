"""Correlation Analyzer Lambda - Correlates multiple analyzer outputs into unified_document.

This analyzer takes S3 URIs of analyzer XML outputs and correlates them into
a unified_document structure using Opus 4.6 with adaptive thinking.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

import boto3

from foundation.bedrock_client import BedrockClient
from foundation.s3_config_loader import load_manifest_from_s3

logger = logging.getLogger()
log_level = os.environ.get("LOGGING_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))


def lambda_handler(
    event: dict[str, Any], context: Any
) -> dict[str, Any]:  # noqa: ARG001
    """Lambda handler for Correlation Analyzer."""
    try:
        body = json.loads(event["body"]) if "body" in event else event

        session_id = body.get("session_id", "no_session")
        full_text_uri = body.get("full_text_uri")
        analyzer_uris = body.get("analyzer_uris", [])
        page_number = body.get("page_number", "1")
        source_image_path = body.get("source_image_path", "")

        logger.info("Correlating page %s for session %s", page_number, session_id)
        logger.info("Full text URI: %s", full_text_uri)
        logger.info("Other analyzers: %s", [a["analyzer_name"] for a in analyzer_uris])

        if not full_text_uri:
            return _error_response("full_text_uri is required")

        if not analyzer_uris:
            return _error_response(
                "analyzer_uris must contain at least one other analyzer result"
            )

        # Load configuration from manifest
        config_bucket = os.environ.get("CONFIG_BUCKET", "")
        analyzer_name = os.environ.get("ANALYZER_NAME", "correlation_analyzer")
        config = _load_config_from_s3(config_bucket, analyzer_name)

        s3 = boto3.client("s3")

        # Fetch analyzer outputs from S3
        full_text_content = _fetch_s3_content(s3, full_text_uri)

        analyzer_contents = []
        for analyzer in analyzer_uris:
            content = _fetch_s3_content(s3, analyzer["s3_uri"])
            analyzer_contents.append(
                {
                    "name": analyzer["analyzer_name"],
                    "uri": analyzer["s3_uri"],
                    "content": content,
                }
            )

        # Load prompts from S3
        system_prompt = _load_correlation_prompts(s3, config_bucket, config)

        # Build the user message with all analyzer outputs
        user_message = _build_user_message(
            full_text_uri,
            full_text_content,
            analyzer_contents,
            page_number,
            source_image_path,
        )

        # Get model configuration from manifest
        model_config = _get_model_config(config)

        # Use foundation's BedrockClient
        bedrock_client = BedrockClient(
            throttle_delay=float(os.environ.get("THROTTLE_DELAY", "1.0")),
            aws_region=os.environ.get("AWS_REGION", "us-west-2"),
        )

        # Build payload
        payload = bedrock_client.create_anthropic_payload(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=model_config["max_tokens"],
            temperature=0.1,  # Will be overridden to 1 if thinking is enabled
        )

        logger.info(
            "Invoking model %s with max_tokens=%d",
            model_config["model_id"],
            model_config["max_tokens"],
        )

        # Invoke via BedrockClient with full fallback and thinking support
        response = bedrock_client.invoke_model(
            model_id=model_config["model_id"],
            payload=payload,
            fallback_list=model_config.get("fallback_list"),
            max_retries=config.get("max_retries", 2),
            extended_thinking=model_config.get("extended_thinking", False),
            budget_tokens=model_config.get("budget_tokens"),
            adaptive_thinking=model_config.get("adaptive_thinking", False),
            adaptive_effort=model_config.get("effort", "high"),
        )

        # Extract the unified_document from response
        logger.info("Response keys: %s", list(response.keys()))
        logger.info("Content blocks count: %d", len(response.get("content", [])))

        # Log content block types
        for i, block in enumerate(response.get("content", [])):
            logger.info("Block %d type: %s", i, block.get("type"))

        unified_document = None
        for block in response.get("content", []):
            block_type = block.get("type")
            if block_type == "text":
                text_content = block.get("text", "")
                if text_content and text_content.strip():
                    unified_document = text_content
                    logger.info(
                        "Found text block with %d characters", len(unified_document)
                    )
                    break
            elif block_type == "thinking":
                logger.info("Skipping thinking block")
                continue

        logger.info(
            "Extracted unified_document length: %d",
            len(unified_document) if unified_document else 0,
        )

        if not unified_document:
            logger.error(
                "No text content found in response. Full response: %s",
                json.dumps(response, default=str)[:1000],
            )
            return _error_response(
                "No unified_document generated - no text content in response"
            )

        # Clean markdown artifacts from response
        unified_document = _clean_markdown_artifacts(unified_document)

        logger.info(
            "After cleaning, unified_document length: %d", len(unified_document)
        )

        # Extract summary from the unified_document
        summary = _extract_summary(unified_document)

        # Save to S3
        output_bucket = os.environ.get("OUTPUT_BUCKET")
        if not output_bucket:
            return _error_response("OUTPUT_BUCKET not configured")

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        s3_key = f"{session_id}/correlated/unified_document_page_{page_number}_{timestamp}.xml"

        # Verify we have content before writing
        if not unified_document or len(unified_document.strip()) == 0:
            logger.error("unified_document is empty after processing!")
            return _error_response("unified_document is empty after processing")

        body_bytes = unified_document.encode("utf-8")
        logger.info(
            "Writing %d bytes to S3: %s/%s", len(body_bytes), output_bucket, s3_key
        )

        s3.put_object(
            Bucket=output_bucket,
            Key=s3_key,
            Body=body_bytes,
            ContentType="application/xml",
            Tagging=f"session_id={session_id}&type=unified_document&page={page_number}",
        )

        unified_document_uri = f"s3://{output_bucket}/{s3_key}"
        logger.info("Saved unified_document to %s", unified_document_uri)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "unified_document_uri": unified_document_uri,
                    "summary": summary,
                    "success": True,
                    "session_id": session_id,
                }
            ),
        }

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        return _error_response(str(e))


def _load_config_from_s3(bucket: str, analyzer_name: str) -> dict[str, Any]:
    """Load analyzer configuration from S3 manifest."""
    manifest = load_manifest_from_s3(bucket, analyzer_name)
    config: dict[str, Any] = manifest.get("analyzer", manifest)
    return config


def _get_model_config(config: dict) -> dict:
    """Extract model configuration from manifest.

    Returns:
        Dict with model_id, max_tokens, and thinking settings
    """
    model_selections = config.get("model_selections", {})
    primary = model_selections.get("primary", {})

    # Extract fallback list - handle both string and dict formats
    fallback_raw = model_selections.get("fallback_list", [])
    fallback_list = []
    for item in fallback_raw:
        if isinstance(item, dict):
            fallback_list.append(item.get("model_id", item))
        else:
            fallback_list.append(item)

    result = {
        "max_tokens": config.get("expected_output_tokens", 12000),
        "fallback_list": fallback_list,
    }

    # Handle both dict and string formats for primary
    if isinstance(primary, dict):
        result["model_id"] = primary.get("model_id", "us.anthropic.claude-opus-4-6-v1")
        result["adaptive_thinking"] = primary.get("adaptive_thinking", False)
        result["effort"] = primary.get("effort", "high")
        result["extended_thinking"] = primary.get("extended_thinking", False)
        result["budget_tokens"] = primary.get("budget_tokens")
    else:
        result["model_id"] = primary or "us.anthropic.claude-opus-4-6-v1"
        result["adaptive_thinking"] = False
        result["extended_thinking"] = False
        result["effort"] = "high"
        result["budget_tokens"] = None

    return result


def _fetch_s3_content(s3: Any, s3_uri: str) -> str:
    """Fetch content from S3 URI."""
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    response = s3.get_object(Bucket=bucket, Key=key)
    content: str = response["Body"].read().decode("utf-8")
    return content


def _load_correlation_prompts(s3, config_bucket: str, config: dict) -> str:
    """Load and concatenate correlation prompt files from S3."""
    prompt_files = config.get(
        "prompt_files",
        [
            "correlation_job_role.xml",
            "correlation_context.xml",
            "correlation_rules.xml",
            "correlation_tasks.xml",
            "correlation_format.xml",
        ],
    )

    prompts = []
    for filename in prompt_files:
        key = f"prompts/correlation_analyzer/{filename}"
        try:
            response = s3.get_object(Bucket=config_bucket, Key=key)
            prompts.append(response["Body"].read().decode("utf-8"))
        except Exception as e:
            logger.warning("Could not load prompt %s: %s", key, e)

    return "\n\n".join(prompts)


def _build_user_message(
    full_text_uri: str,
    full_text_content: str,
    analyzer_contents: list,
    page_number: str,
    source_image_path: str,
) -> str:
    """Build the user message with all analyzer outputs."""
    parts = [
        f"Correlate the following analyzer outputs for page {page_number}.",
        f"Source image: {source_image_path}",
        "",
        "=== FULL TEXT ANALYZER OUTPUT (CANONICAL SPINE) ===",
        f"S3 URI: {full_text_uri}",
        "",
        full_text_content,
        "",
    ]

    for analyzer in analyzer_contents:
        parts.extend(
            [
                f"=== {analyzer['name'].upper()} ANALYZER OUTPUT ===",
                f"S3 URI: {analyzer['uri']}",
                "",
                analyzer["content"],
                "",
            ]
        )

    parts.append("Produce the unified_document XML following the response_format.")

    return "\n".join(parts)


def _clean_markdown_artifacts(text: str) -> str:
    """Remove markdown code fences and other artifacts from response."""
    import re

    # Strip markdown code fences
    text = re.sub(r"^```(?:xml|json|html|text)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)

    return text.strip()


def _extract_summary(unified_document: str) -> str:
    """Extract summary from unified_document XML."""
    import re

    match = re.search(r"<summary>(.*?)</summary>", unified_document, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "Correlation completed. See unified_document for details."


def _error_response(message: str) -> dict:
    """Return error response."""
    return {
        "statusCode": 500,
        "body": json.dumps(
            {
                "unified_document_uri": None,
                "summary": f"Error: {message}",
                "success": False,
            }
        ),
    }
