"""Text-grounded keyword and topic extraction Lambda."""

import json
import logging
import os
from pathlib import Path

from foundation.lambda_error_handler import (
    create_error_response,
    ValidationError,
    ResourceNotFoundError,
    handle_bedrock_error,
    handle_s3_error,
)
from foundation.s3_result_saver import save_result_to_s3
from foundation import job_state
from foundation.revision_handler import try_revision

# Configure logging from environment variable
logger = logging.getLogger()
log_level = os.environ.get("LOGGING_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))


def lambda_handler(event, context):
    """Lambda handler for Charts Specialist using S3-based configuration."""
    # Job-tracking identifiers are read from the request body inside the try,
    # but must exist out here so the except handler can mark the subtask failed
    # even if the failure happened before they were parsed.
    job_id = ""
    subtask = ""
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

        # Determine config source
        config_bucket = os.environ.get("CONFIG_BUCKET")
        specialist_name = os.environ.get("SPECIALIST_NAME", "charts_specialist")

        # Auto-detect: use S3 if in Lambda and CONFIG_BUCKET is set
        if os.environ.get("AWS_EXECUTION_ENV") and config_bucket:
            config_source = "s3"
            logger.info(
                "Using S3 config: bucket=%s, specialist=%s",
                config_bucket,
                specialist_name,
            )
        else:
            config_source = "local"
            logger.info("Using local config")

        # Parse input - AgentCore Gateway passes parameters directly in event
        body = json.loads(event["body"]) if "body" in event else event

        # ── Revision mode: delegate to foundation revision handler ──
        revision_result = try_revision(event, context)
        if revision_result is not None:
            return revision_result

        # Extract and log session_id from AgentCore Runtime
        session_id = body.get("session_id", "no_session")
        logger.info("Processing request for runtime session_id: %s", session_id)
        # Job tracking (doc_id -> job_id -> subtask_id). All of these no-op when
        # JOBS_TABLE_NAME is unset, so untracked deployments are unaffected.
        job_id = body.get("job_id") or ""
        doc_id = body.get("doc_id") or ""
        source_image_path = str(body.get("source_image_path") or "")
        subtask = job_state.subtask_id(
            specialist_name, source_image_path or body.get("full_text_uri")
        )
        job_state.mark_running(
            job_id,
            subtask,
            doc_id=doc_id,
            specialist=specialist_name,
            image_id=job_state.image_identifier(source_image_path),
            session_id=session_id,
        )

        # Extract audit_mode flag
        audit_mode = body.get("audit_mode", False)
        if audit_mode:
            logger.info("Audit mode enabled - confidence assessment will be included")

        # Set dynamic token estimation from request
        if body.get("dynamic_tokens_enabled", False):
            os.environ["DYNAMIC_TOKENS_ENABLED"] = "true"
            logger.info("Dynamic token estimation enabled")
        else:
            os.environ["DYNAMIC_TOKENS_ENABLED"] = "false"

        if "image_path" in body or "image_data" in body:
            raise ValidationError(
                message=(
                    "Keyword analysis accepts full_text_uri and optional "
                    "correlation_specialist_uri, not image_path or image_data."
                ),
                details={"provided_keys": list(body.keys())},
            )
        full_text = _load_text_input(body, "full_text_uri", required=True)
        correlated_text = _load_text_input(
            body, "correlation_specialist_uri", required=False
        )

        # Load configuration
        if config_source == "s3":
            config = _load_config_from_s3(config_bucket, specialist_name)
        else:
            config = _load_config_from_local()

        # Initialize specialist
        specialist = _initialize_specialist(
            config, config_source, config_bucket, specialist_name
        )

        # Keyword extraction is text-only.  It still uses the same foundation
        # prompt, model selection/fallback, and response cleanup steps as visual
        # specialists, but never attempts to decode a correlated XML file as an
        # image.
        system_prompt = specialist._load_system_prompt(audit_mode=audit_mode)
        user_message = (
            "Extract grounded keywords and topics from the supplied page text.\n"
            f"Page number: {body.get('page_number', '')}\n"
            f"Source image identifier: {source_image_path}\n\n"
            "<full_text>\n"
            f"{full_text}"
            "\n</full_text>\n"
        )
        if correlated_text:
            user_message += (
                "\n<correlated_spine>\n"
                f"{correlated_text}"
                "\n</correlated_spine>\n"
            )
        response = specialist._invoke_bedrock_model(
            system_prompt,
            [{"role": "user", "content": user_message}],
            body.get("aws_profile"),
        )
        result = specialist._finalize_result(specialist._process_response(response))

        # Log result to CloudWatch
        logger.info("Analysis completed successfully for session: %s", session_id)
        logger.info("Result length: %d characters", len(result))

        # Save result to S3
        # Persisting the result is part of succeeding. If the artifact cannot be
        # written the pipeline has not satisfied the request, so fail loudly
        # instead of returning a result the caller cannot retrieve -- a false
        # success also encourages the orchestrator to keep calling a tool that
        # cannot deliver.
        output_bucket = os.environ.get("OUTPUT_BUCKET")
        if not output_bucket:
            raise RuntimeError(
                "OUTPUT_BUCKET is not configured; the analysis result cannot be "
                "persisted."
            )
        s3_uri = save_result_to_s3(
            result=result,
            specialist_name=specialist_name,
            output_bucket=output_bucket,
            session_id=session_id,
            image_path=source_image_path,
            job_id=job_id,
        )
        result = f"{result}\n<!-- S3_RESULT_URI: {s3_uri} -->"
        job_state.mark_complete(job_id, subtask, s3_uri)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {"result": result, "success": True, "session_id": session_id}
            ),
        }

    except Exception as e:
        job_state.mark_failed(job_id, subtask, str(e))
        return create_error_response(e)


def _load_text_input(body: dict, field: str, *, required: bool) -> str:
    """Read a bounded UTF-8 text artifact from S3 for keyword extraction."""
    uri = body.get(field)
    if not uri:
        if required:
            raise ValidationError(
                message=f"Missing required parameter: {field}.",
                details={"provided_keys": list(body.keys())},
            )
        return ""
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise ValidationError(
            message=f"{field} must be an S3 URI.",
            details={field: uri},
        )
    bucket, separator, key = uri[5:].partition("/")
    if not bucket or not separator or not key:
        raise ValidationError(
            message=f"Invalid {field} S3 URI.",
            details={field: uri},
        )
    try:
        import boto3

        value = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as error:
        raise handle_s3_error(error, bucket, key) from error
    return value.decode("utf-8", errors="replace")[:400000]


def _load_config_from_s3(bucket: str, specialist_name: str) -> dict:
    """Load specialist configuration from S3."""
    from foundation.s3_config_loader import load_manifest_from_s3

    from typing import Any

    manifest: dict[str, Any] = load_manifest_from_s3(bucket, specialist_name)
    config: dict[str, Any] = manifest.get("specialist", manifest)

    logger.info("Loaded config from S3 for %s", specialist_name)
    return config


def _load_config_from_local() -> dict:
    """Load specialist configuration from local filesystem."""
    from typing import Any

    manifest_path = Path("/var/task/manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest: dict[str, Any] = json.load(f)

    config: dict[str, Any] = manifest["specialist"]

    # Resolve all paths relative to /var/task
    config["prompt_base_path"] = str(
        Path("/var/task") / config.get("prompt_base_path", "prompts")
    )
    config["examples_path"] = str(Path("/var/task") / config.get("examples_path", ""))

    logger.info("Loaded config from local filesystem")
    return config


def _initialize_specialist(
    config: dict,
    config_source: str,
    s3_bucket: str | None = None,
    specialist_name: str | None = None,
):
    """Initialize the specialist foundation with appropriate config source."""
    from foundation.specialist_foundation import SpecialistFoundation
    from foundation.configuration_manager import ConfigurationManager
    from foundation.prompt_loader import PromptLoader
    from foundation.image_processor import ImageProcessor
    from foundation.bedrock_client import BedrockClient
    from foundation.message_chain_builder import MessageChainBuilder
    from foundation.response_processor import ResponseProcessor

    # Create specialist instance
    specialist = object.__new__(SpecialistFoundation)
    specialist.specialist_type = specialist_name or "charts_specialist"
    specialist.s3_bucket = s3_bucket if config_source == "s3" else None
    specialist.logger = logging.getLogger(f"foundation.{specialist.specialist_type}")
    specialist.config = config
    specialist.global_settings = {
        "max_tokens": int(os.environ.get("MAX_TOKENS", "8000")),
        "temperature": float(os.environ.get("TEMPERATURE", "0.1")),
        "max_image_size": int(os.environ.get("MAX_IMAGE_SIZE", "20971520")),
        "max_dimension": int(os.environ.get("MAX_DIMENSION", "2048")),
        "jpeg_quality": int(os.environ.get("JPEG_QUALITY", "85")),
        "cache_enabled": os.environ.get("CACHE_ENABLED", "True") == "True",
        "throttle_delay": float(os.environ.get("THROTTLE_DELAY", "1.0")),
        "aws_region": os.environ.get("AWS_REGION", "us-west-2"),
    }

    # Initialize components
    specialist.config_manager = ConfigurationManager()

    # Initialize prompt loader with S3 support
    if config_source == "s3":
        specialist.prompt_loader = PromptLoader(
            config_source="s3", s3_bucket=s3_bucket, specialist_name=specialist_name
        )
    else:
        specialist.prompt_loader = PromptLoader(config_source="local")

    specialist.image_processor = ImageProcessor()
    specialist.bedrock_client = BedrockClient()
    specialist.message_builder = MessageChainBuilder()
    specialist.response_processor = ResponseProcessor()
    specialist._configure_components()

    logger.info("Initialized specialist with %s config", config_source)
    return specialist
