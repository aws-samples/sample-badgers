"""Custom Analyzer Lambda - test_analyzer."""
import json
import logging
import base64
import os
from pathlib import Path

from foundation.lambda_error_handler import (
    create_error_response, ValidationError, ResourceNotFoundError, handle_s3_error,
)
from foundation.s3_result_saver import save_result_to_s3

logger = logging.getLogger()
logger.setLevel(getattr(logging, os.environ.get("LOGGING_LEVEL", "INFO").upper(), logging.INFO))

ANALYZER_NAME = "test_analyzer"

def lambda_handler(event, context):
    try:
        config_bucket = os.environ.get("CONFIG_BUCKET")
        analyzer_name = os.environ.get("ANALYZER_NAME", ANALYZER_NAME)
        is_custom = os.environ.get("CUSTOM_ANALYZER", "false").lower() == "true"
        body = json.loads(event["body"]) if "body" in event else event
        session_id = body.get("session_id", "no_session")
        audit_mode = body.get("audit_mode", False)

        image_data = _get_image_data(body)
        config = _load_config_from_s3(config_bucket, analyzer_name, is_custom)
        analyzer = _initialize_analyzer(config, config_bucket, analyzer_name, is_custom)
        result = analyzer.analyze(image_data, body.get("aws_profile"), audit_mode)

        output_bucket = os.environ.get("OUTPUT_BUCKET")
        if output_bucket:
            try:
                s3_uri = save_result_to_s3(result=result, analyzer_name=analyzer_name,
                    output_bucket=output_bucket, session_id=session_id, image_path=body.get("image_path"))
                result = f"{result}\n<!-- S3_RESULT_URI: {s3_uri} -->"
            except Exception as e:
                logger.error("Failed to save result to S3: %s", e)

        return {"statusCode": 200, "body": json.dumps({"result": result, "success": True, "session_id": session_id})}
    except Exception as e:
        return create_error_response(e)

def _get_image_data(body: dict) -> bytes:
    if "image_data" in body:
        return base64.b64decode(body["image_data"])
    if "image_path" in body:
        image_path = body["image_path"]
        if image_path.startswith("s3://"):
            import boto3
            s3 = boto3.client("s3")
            parts = image_path.replace("s3://", "").split("/", 1)
            bucket, key = parts
            response = s3.get_object(Bucket=bucket, Key=key)
            data = response["Body"].read()
            return base64.b64decode(data.decode("utf-8")) if key.endswith(".b64") else bytes(data)
        file_path = Path("/var/task") / image_path
        if file_path.exists():
            with open(file_path, "rb") as f:
                return f.read()
    raise ValidationError(message="Missing image_data or image_path", details={})

def _load_config_from_s3(bucket: str, analyzer_name: str, is_custom: bool) -> dict:
    from foundation.s3_config_loader import load_manifest_from_s3
    manifest = load_manifest_from_s3(bucket, analyzer_name, custom=is_custom)
    return manifest.get("analyzer", manifest)

def _initialize_analyzer(config: dict, s3_bucket: str, analyzer_name: str, is_custom: bool):
    from foundation.analyzer_foundation import AnalyzerFoundation
    from foundation.configuration_manager import ConfigurationManager
    from foundation.prompt_loader import PromptLoader
    from foundation.image_processor import ImageProcessor
    from foundation.bedrock_client import BedrockClient
    from foundation.message_chain_builder import MessageChainBuilder
    from foundation.response_processor import ResponseProcessor

    analyzer = object.__new__(AnalyzerFoundation)
    analyzer.analyzer_type = analyzer_name
    analyzer.s3_bucket = s3_bucket
    analyzer.logger = logging.getLogger(f"foundation.{analyzer_name}")
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
    analyzer.config_manager = ConfigurationManager()
    analyzer.prompt_loader = PromptLoader(config_source="s3", s3_bucket=s3_bucket, analyzer_name=analyzer_name, custom=is_custom)
    analyzer.image_processor = ImageProcessor()
    analyzer.bedrock_client = BedrockClient()
    analyzer.message_builder = MessageChainBuilder()
    analyzer.response_processor = ResponseProcessor()
    analyzer._configure_components()
    return analyzer
