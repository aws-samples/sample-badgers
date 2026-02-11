"""Accessibility Analyzer Lambda - S3-based config version with AgentCore Gateway session tracking."""

import json
import logging
import base64
import os
from pathlib import Path

from foundation.s3_result_saver import save_result_to_s3

# Configure logging from environment variable
logger = logging.getLogger()
log_level = os.environ.get("LOGGING_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))


def lambda_handler(event, context):
    """Lambda handler for Accessibility Analyzer using S3-based configuration."""
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
        analyzer_name = os.environ.get("ANALYZER_NAME", "accessibility_analyzer")

        # Auto-detect: use S3 if in Lambda and CONFIG_BUCKET is set
        if os.environ.get("AWS_EXECUTION_ENV") and config_bucket:
            config_source = "s3"
            logger.info(
                "Using S3 config: bucket=%s, analyzer=%s", config_bucket, analyzer_name
            )
        else:
            config_source = "local"
            logger.info("Using local config")

        # Parse input - AgentCore Gateway passes parameters directly in event
        body = json.loads(event["body"]) if "body" in event else event

        # Extract and log session_id from AgentCore Runtime
        session_id = body.get("session_id", "no_session")
        logger.info("Processing request for runtime session_id: %s", session_id)

        # Extract audit_mode flag
        audit_mode = body.get("audit_mode", False)
        if audit_mode:
            logger.info("Audit mode enabled - confidence assessment will be included")

        # Get image data
        image_data = _get_image_data(body)

        # Load configuration
        if config_source == "s3":
            config = _load_config_from_s3(config_bucket, analyzer_name)
        else:
            config = _load_config_from_local()

        # Initialize analyzer
        analyzer = _initialize_analyzer(
            config, config_source, config_bucket, analyzer_name
        )

        # Run analysis
        result = analyzer.analyze(image_data, body.get("aws_profile"), audit_mode)

        # Log result to CloudWatch
        logger.info("Analysis completed successfully for session: %s", session_id)
        logger.info("Result length: %d characters", len(result))

        # Save result to S3
        output_bucket = os.environ.get("OUTPUT_BUCKET")
        if output_bucket:
            try:
                s3_uri = save_result_to_s3(
                    result=result,
                    analyzer_name=analyzer_name,
                    output_bucket=output_bucket,
                    session_id=session_id,
                    image_path=body.get("image_path"),
                )
                result = f"{result}\n<!-- S3_RESULT_URI: {s3_uri} -->"
            except Exception as e:
                logger.error(
                    "Failed to save result to S3: %s", e
                )  # Don't fail the request if S3 save fails

        return {
            "statusCode": 200,
            "body": json.dumps(
                {"result": result, "success": True, "session_id": session_id}
            ),
        }

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"result": str(e), "success": False}),
        }


def _get_image_data(body: dict) -> bytes:
    """Extract image data from request body."""
    if "image_data" in body:
        return base64.b64decode(body["image_data"])

    if "image_path" in body:
        image_path = body["image_path"]

        if image_path.startswith("s3://"):
            # Download from S3
            import boto3

            s3 = boto3.client("s3")
            parts = image_path.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1]
            response = s3.get_object(Bucket=bucket, Key=key)
            data = response["Body"].read()

            # Handle .b64 files (pre-encoded base64 from pdf_to_images)
            if key.endswith(".b64"):
                logger.info("Loading pre-encoded base64 from %s", image_path)
                return base64.b64decode(data.decode("utf-8"))

            return bytes(data)

        # Local file in Lambda environment
        file_path = Path("/var/task") / image_path
        logger.info("Reading local file: %s", file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {image_path}")

        with open(file_path, "rb") as f:
            return f.read()

    raise ValueError("Missing image_data or image_path")


def _load_config_from_s3(bucket: str, analyzer_name: str) -> dict:
    """Load analyzer configuration from S3."""
    from foundation.s3_config_loader import load_manifest_from_s3

    from typing import Any

    manifest: dict[str, Any] = load_manifest_from_s3(bucket, analyzer_name)
    config: dict[str, Any] = manifest.get("analyzer", manifest)

    logger.info("Loaded config from S3 for %s", analyzer_name)
    return config


def _load_config_from_local() -> dict:
    """Load analyzer configuration from local filesystem."""
    from typing import Any

    manifest_path = Path("/var/task/manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest: dict[str, Any] = json.load(f)

    config: dict[str, Any] = manifest["analyzer"]

    # Resolve all paths relative to /var/task
    config["prompt_base_path"] = str(Path("/var/task") / config["prompt_base_path"])
    config["examples_path"] = str(Path("/var/task") / config["examples_path"])

    logger.info("Loaded config from local filesystem")
    return config


def _initialize_analyzer(
    config: dict,
    config_source: str,
    s3_bucket: str | None = None,
    analyzer_name: str | None = None,
):
    """Initialize the analyzer foundation with appropriate config source."""
    from foundation.analyzer_foundation import AnalyzerFoundation
    from foundation.configuration_manager import ConfigurationManager
    from foundation.prompt_loader import PromptLoader
    from foundation.image_processor import ImageProcessor
    from foundation.bedrock_client import BedrockClient
    from foundation.message_chain_builder import MessageChainBuilder
    from foundation.response_processor import ResponseProcessor

    # Create analyzer instance
    analyzer = object.__new__(AnalyzerFoundation)
    analyzer.analyzer_type = analyzer_name or "accessibility_analyzer"
    analyzer.s3_bucket = s3_bucket if config_source == "s3" else None
    analyzer.logger = logging.getLogger(f"foundation.{analyzer.analyzer_type}")
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

    # Initialize components
    analyzer.config_manager = ConfigurationManager()

    # Initialize prompt loader with S3 support
    if config_source == "s3":
        analyzer.prompt_loader = PromptLoader(
            config_source="s3", s3_bucket=s3_bucket, analyzer_name=analyzer_name
        )
    else:
        analyzer.prompt_loader = PromptLoader(config_source="local")

    analyzer.image_processor = ImageProcessor()
    analyzer.bedrock_client = BedrockClient()
    analyzer.message_builder = MessageChainBuilder()
    analyzer.response_processor = ResponseProcessor()
    analyzer._configure_components()

    logger.info("Initialized analyzer with %s config", config_source)
    return analyzer
